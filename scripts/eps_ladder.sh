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

# EXTRA carries whatever data/schedule flags the 120k sweep was run with. The
# ladder is only single-variable against *that* recipe, not against
# train_encoder's bare defaults, so pass the same flags here -- e.g.
#   EXTRA="--val-data-dir data/val-1000-0.25s --lr-floor 0.05 --lr-hold-frac 0.6"
GPUS=${1:-"0 1 2 3"}
STEPS=${STEPS:-40000}
LR=${LR:-3e-4}
OUT=${OUT:-results/ddsp/eps_ladder}
EXTRA=${EXTRA:-""}
ARMS=${ARMS:-"L1_STFT L1_STFT_eps1 L1_STFT_eps1e1 L1_STFT_eps1e3 L1_STFT_eps1e5 L1_STFT_eps1e7"}

read -r -a GPU_ARR <<< "$GPUS"
read -r -a ARM_ARR <<< "$ARMS"
NG=${#GPU_ARR[@]}

mkdir -p "$OUT"
echo "arms: ${ARM_ARR[*]}"
echo "gpus: ${GPU_ARR[*]}  steps: $STEPS  lr: $LR  out: $OUT"

# Arms are handed to GPUs round-robin and run sequentially within a GPU, so a
# ladder longer than the GPU count still completes without oversubscribing.
for ((g = 0; g < NG; g++)); do
  (
    for ((i = g; i < ${#ARM_ARR[@]}; i += NG)); do
      arm=${ARM_ARR[$i]}
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=${GPU_ARR[$g]} python -m src.ddsp.train_encoder \
        --loss "$arm" \
        --output-dir "$OUT/$arm" \
        --peak-normalize target \
        --steps "$STEPS" \
        --lr "$LR" \
        --seed 0 \
        $EXTRA \
        > "$OUT/$arm.log" 2>&1
    done
  ) &
  echo "  gpu ${GPU_ARR[$g]} -> $(for ((i = g; i < ${#ARM_ARR[@]}; i += NG)); do printf '%s ' "${ARM_ARR[$i]}"; done)"
done
wait
echo "ladder complete: $OUT"
