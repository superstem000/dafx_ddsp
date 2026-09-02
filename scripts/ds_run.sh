#!/usr/bin/env bash
# One diffsynth arm on one GPU, with the paths and run directory pinned.
#
#   scripts/ds_run.sh <name> <experiment> <gpu> [hydra overrides...]
#   RESUME=<name> scripts/ds_run.sh <name> <experiment> <gpu> [...]
#
#   scripts/ds_run.sh pre_hybrid pretrain 0
#   RESUME=pre_hybrid scripts/ds_run.sh synth_hybrid resume_synth 1
#
# One GPU per arm, never one arm across GPUs. DDP would divide the work but not
# cleanly: batch_size in the datamodule is PER DEVICE under DDP, so four cards
# at 64 gives an effective 256 and 62 steps/epoch instead of 250 -- which moves
# the ramp off the paper's epochs 50/200 entirely. Correcting it to 16 per
# device then changes the estimator's BatchNorm statistics, which are computed
# per device. The paper's setup is single-GPU and the arms are independent, so
# parallelism belongs across arms, where it costs nothing.
#
# hydra.run.dir is pinned to results/diffsynth/<name> instead of hydra's default
# outputs/<date>/<time>/, so a later stage can find the checkpoint it resumes
# from by name rather than by timestamp.
#
# stdin is closed: nothing here prompts, but a hidden prompt under a detached
# queue hangs a card silently, and that has already happened once in this repo.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: scripts/ds_run.sh <name> <experiment> <gpu> [overrides...]"
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME=$1; EXP=$2; GPU=$3; shift 3

DS="$ROOT/external/diffsynth"

# WHICH DATASET. Pinned to the paper's h2of set by default, overridable for a
# different task -- the chorus chain has its own generated data and its own
# model synth. Passed here rather than as a plain hydra override on the job
# line because this script already sets data.id_dir, and hydra rejects the same
# key twice rather than taking the last one.
ID_DIR=${ID_DIR:-$DS/data/diffsynth_5-6/harmor_2oscfree}
# OOD_DIR=none drops the out-of-domain half entirely. A run that supplies the
# oscillator pitches as conditioning has to: the OOD set is loaded with
# params=False, so those batches carry no targets to take the pitches from, and
# inventing a fundamental for acoustic notes would make val_ood meaningless
# rather than absent. train.py then monitors val_id/lsd instead.
OOD_DIR=${OOD_DIR:-$DS/data/nsynth-train}
if [[ ! -d "$ID_DIR" ]]; then
  echo "ERROR: ID_DIR does not exist: $ID_DIR"
  echo "  (resolved from $PWD)"
  echo "Generate it first -- see scripts/jobs_diffsynth_chorus.txt for the"
  echo "gen_dataset.py command and the smoke test that goes before it."
  exit 1
fi
# RESOLVE TO ABSOLUTE, both of them. This script cds into $DS below and hydra
# changes the working directory again on top of that, so a relative path that
# was valid where the job line ran is looked up somewhere else entirely by the
# time the datamodule opens it -- and the existence check above passes, because
# it runs before the cd. That combination cost a queue: all three chorus jobs
# reached hydra and died on a directory that was right there.
ID_DIR=$(cd "$ID_DIR" && pwd)
if [[ "$OOD_DIR" == "none" ]]; then
  OOD_ARG="data.ood_dir=null"
else
  [[ -d "$OOD_DIR" ]] && OOD_DIR=$(cd "$OOD_DIR" && pwd) || true
  OOD_ARG="data.ood_dir=$OOD_DIR"
fi
RUNDIR="$ROOT/results/diffsynth/$NAME"

EXTRA=()
if [[ -n "${RESUME:-}" ]]; then
  # latest.ckpt, not last.ckpt. last.ckpt is only rewritten when the monitored
  # metric improves, so it silently lags -- on the first attempt it was epoch 37
  # of a 50-epoch base. See train.py for the callback that writes this one.
  # CKPT names a specific file instead. latest.ckpt is the right default -- it
  # is whatever epoch the run reached -- but it is the WRONG choice when several
  # arms branch at once, because they reach different epochs and each branch
  # would start somewhere else. A periodic checkpoint (ep0299.ckpt, written by
  # trainer.checkpoint_every_n_epochs) is the same epoch in every arm, which is
  # what makes the branches comparable.
  CKPT="$ROOT/results/diffsynth/${RESUME}/tb_logs/checkpoints/${CKPT:-latest.ckpt}"
  if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: $NAME resumes from '$RESUME' but no checkpoint at"
    echo "  $CKPT"
    echo "Run the earlier stage first."
    exit 1
  fi
  # Say which epoch is being resumed from. A resume from the wrong point is
  # otherwise invisible until someone reads the epoch axis of a plot.
  python3 - "$CKPT" <<'PYCK'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"    resuming from epoch {ck['epoch']} (global_step {ck['global_step']})")
PYCK
  EXTRA+=("trainer.resume_from_checkpoint=$CKPT")
fi

if [[ -d "$RUNDIR" ]] && [[ "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: $RUNDIR already exists. FORCE=1 to overwrite."
  exit 1
fi

mkdir -p "$RUNDIR"
cd "$DS"

echo "=== $NAME  (experiment=$EXP, gpu=$GPU)${RESUME:+  resuming from $RESUME}"
[[ "$OOD_DIR" == "none" ]] && echo "    no out-of-domain half; val_id only" || true
echo "    -> $RUNDIR"

# Required by torch for deterministic CUBLAS on CUDA >= 10.2; without it
# use_deterministic_algorithms raises as soon as a matmul runs.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CUDA_VISIBLE_DEVICES="$GPU" python train.py \
  experiment="$EXP" \
  data.id_dir="$ID_DIR" \
  "$OOD_ARG" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  hydra.run.dir="$RUNDIR" \
  "${EXTRA[@]}" "$@" < /dev/null
