#!/usr/bin/env bash
# The compression ladder as one variable, on the plate encoder.
#
# Six arms differing in exactly one scalar: the eps in log(x + eps), plus the
# linear anchor at the top. Every other setting -- architecture, data, steps,
# learning rate, schedule, normalization, seed -- is identical across arms, so
# a difference between rungs is attributable to eps and to nothing else. That
# is the whole point: the earlier sweep varied loss *family* (linear / c2 / MSS
# / log), which confounds compression with resolution and with the SC term.
#
# eps = 1 is log1p, which is C2; eps = 1e-7 is C1. So the ends of the ladder
# reproduce two arms already run, and diag_eps_ladder.py asserts they do.
#
# Budget: the log arm runs at ~0.5 steps/s against linear's ~1.6, so a full
# 120k-step ladder is weeks of wall clock. Collapse is not a late phenomenon --
# the earlier arms were pinned near the constant-predictor floor long before
# 90k -- so this locates the cliff at a shorter budget first and only extends
# rungs that land near the boundary. STEPS is overridable for that second pass.
#
#   scripts/eps_ladder.sh "0 1 2 3"            # gpu list, default budget
#   STEPS=120000 scripts/eps_ladder.sh "0 1"   # extend a subset
set -euo pipefail

# Datasets and numerics come from docs/DATASETS.md, which is where the recipe
# actually lives. They are defaults here rather than something to remember to
# pass, because the cost of omitting them is measured there and it is not small:
#
#   --fixed-mode-grid 86,282  n_modes otherwise follows the batch maximum, so an
#                             IR renders differently depending on which batch it
#                             lands in. 6.1% of saturation for log, ~0 for
#                             linear -- i.e. a spurious floor in precisely the
#                             arms under test, in precisely the quiet bins log
#                             weights most.
#   --compile-plate           must match what the targets were rendered with.
#   --chunk-elems, --mode-bucket   likewise; a fused kernel is different
#                             arithmetic.
#
# train-p99 keeps 98339 of 100000 and val-p99 996 of 1000, the remainder being
# plates that need a finer grid than the pin. --n-train is set explicitly
# because train_encoder defaults it to 8192, and it doubles as load_dataset's
# `limit`, so leaving it alone would silently train on the first 8192 rows.
GPUS=${1:-"0 1 2 3"}
STEPS=${STEPS:-40000}
LR=${LR:-3e-4}
OUT=${OUT:-results/ddsp/eps_ladder}
TRAIN=${TRAIN:-data/train-p99}
VAL=${VAL:-data/val-p99}
N_TRAIN=${N_TRAIN:-98339}
N_VAL=${N_VAL:-996}
NUMERICS=${NUMERICS:-"--batched-plate --compile-plate --chunk-elems 1000000000 --mode-bucket 1024 --fixed-mode-grid 86,282"}
EXTRA=${EXTRA:-""}
ARMS=${ARMS:-"L1_STFT L1_STFT_eps1 L1_STFT_eps1e1 L1_STFT_eps1e3 L1_STFT_eps1e4 L1_STFT_eps1e5 L1_STFT_eps1e7"}

# Ctrl-C should stop the sweep, not advance it. Without this the per-arm
# failure handling below reads an interrupt as "that arm died, start the next
# one", so interrupting a seven-arm ladder quietly launches three more.
trap 'echo; echo "interrupted -- stopping all arms"; kill 0; exit 130' INT TERM

read -r -a GPU_ARR <<< "$GPUS"
read -r -a ARM_ARR <<< "$ARMS"
NG=${#GPU_ARR[@]}

mkdir -p "$OUT"
echo "arms: ${ARM_ARR[*]}"
echo "gpus: ${GPU_ARR[*]}  steps: $STEPS  lr: $LR  out: $OUT"

for d in "$TRAIN" "$VAL"; do
  if [[ ! -d "$d" ]]; then
    echo "ERROR: $d does not exist. Regenerate it with the make_dataset command"
    echo "in docs/DATASETS.md -- the flags there are load-bearing, not defaults."
    exit 1
  fi
done

echo
echo "docs/DATASETS.md requires diag_gt_floor to read 0.0000e+00 (within ~1e-3"
echo "for log) on the SHUFFLED row before any sweep is attributable. If you have"
echo "not run it against $VAL since these datasets were built, stop and run it."
echo

# The 120k sweep's arguments were never written down, which is the whole reason
# EXTRA has to be guessed at. Record this one's before it starts.
{
  echo "steps=$STEPS lr=$LR arms='$ARMS' gpus='$GPUS'"
  echo "train=$TRAIN n_train=$N_TRAIN  val=$VAL n_val=$N_VAL"
  echo "numerics='$NUMERICS'"
  echo "extra='$EXTRA'"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python=$(command -v python)"
  echo "torch=$(python -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo unknown)"
} > "$OUT/sweep_command.txt"
cat "$OUT/sweep_command.txt"

# Arms are handed to GPUs round-robin and run sequentially within a GPU, so a
# ladder longer than the GPU count still completes without oversubscribing.
for ((g = 0; g < NG; g++)); do
  (
    for ((i = g; i < ${#ARM_ARR[@]}; i += NG)); do
      arm=${ARM_ARR[$i]}
      # One arm crashing should not cancel the arms queued behind it on the same
      # GPU -- that is how a six-hour sweep comes back with three results and
      # exit 0. An interrupt is different and must stop the queue, so the exit
      # code is inspected rather than merely tested: 130 is SIGINT, 143 SIGTERM.
      rc=0
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=${GPU_ARR[$g]} python -m src.ddsp.train_encoder \
        --loss "$arm" \
        --output-dir "$OUT/$arm" \
        --data-dir "$TRAIN" --n-train "$N_TRAIN" \
        --val-data-dir "$VAL" --n-val "$N_VAL" \
        --peak-normalize target \
        --steps "$STEPS" \
        --lr "$LR" \
        --seed 0 \
        $NUMERICS $EXTRA \
        > "$OUT/$arm.log" 2>&1 || rc=$?

      if [[ $rc -eq 130 || $rc -eq 143 ]]; then
        echo "interrupted during $arm (gpu ${GPU_ARR[$g]}) -- not starting the rest"
        break
      elif [[ $rc -ne 0 ]]; then
        echo "FAILED $arm rc=$rc (gpu ${GPU_ARR[$g]}) -- see $OUT/$arm.log" \
          | tee -a "$OUT/failures.txt"
      fi
    done
  ) &
  echo "  gpu ${GPU_ARR[$g]} -> $(for ((i = g; i < ${#ARM_ARR[@]}; i += NG)); do printf '%s ' "${ARM_ARR[$i]}"; done)"
done
wait
echo "ladder complete: $OUT"
