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
RUNDIR="$ROOT/results/diffsynth/$NAME"

EXTRA=()
if [[ -n "${RESUME:-}" ]]; then
  CKPT="$ROOT/results/diffsynth/${RESUME}/tb_logs/checkpoints/last.ckpt"
  if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: $NAME resumes from '$RESUME' but no checkpoint at"
    echo "  $CKPT"
    echo "Run the pretrain stage first."
    exit 1
  fi
  EXTRA+=("trainer.resume_from_checkpoint=$CKPT")
fi

if [[ -d "$RUNDIR" ]] && [[ "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: $RUNDIR already exists. FORCE=1 to overwrite."
  exit 1
fi

mkdir -p "$RUNDIR"
cd "$DS"

echo "=== $NAME  (experiment=$EXP, gpu=$GPU)${RESUME:+  resuming from $RESUME}"
echo "    -> $RUNDIR"

# Required by torch for deterministic CUBLAS on CUDA >= 10.2; without it
# use_deterministic_algorithms raises as soon as a matmul runs.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CUDA_VISIBLE_DEVICES="$GPU" python train.py \
  experiment="$EXP" \
  data.id_dir="$DS/data/diffsynth_5-6/harmor_2oscfree" \
  data.ood_dir="$DS/data/nsynth-train" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  hydra.run.dir="$RUNDIR" \
  "${EXTRA[@]}" "$@" < /dev/null
