#!/usr/bin/env bash
# Learning rate as a stated grid, so "you did not tune it" is answerable.
#
# A negative result cannot be defended by more searching -- hyperparameter space
# is unbounded and every extra run buys one more "we also tried X". What it can
# be defended by is a pre-stated grid, the same budget on every cell, best-per-
# arm reported rather than best-overall, and -- the part that actually bites --
# the *control* swept on the same axis. If linear is as insensitive to learning
# rate as log is, then learning rate is measurably not the lever, and that is a
# measurement rather than an assertion.
#
# The existing results/ddsp/lr_* runs cover c2, log and MSS at 1e-4, 3e-5 and
# 1e-5. They do not cover L1_STFT at all, which is the gap: without the control
# there is nothing to compare the insensitivity against. Their recipe is matched
# exactly here (12000 steps, constant rate via lr_floor=1 and lr_hold_frac=1,
# everything else as sweep120k), so those nine runs remain valid cells of this
# grid and only the missing ones are run.
#
# eps1 is log1p, which is c2, and eps1e7 is log -- the same two losses the old
# probes used, under the ladder's names.
#
# The rate is held constant on purpose. Decay would confound "this rate fails"
# with "this schedule fails", and the question here is only about the rate.
#
#   scripts/lr_probe.sh "0 1 2 3"
#   LRS="1e-3 3e-4" LOSSES="L1_STFT" scripts/lr_probe.sh "0 1"
set -euo pipefail

GPUS=${1:-"0 1 2 3"}
STEPS=${STEPS:-12000}
OUT=${OUT:-results/ddsp/lr_probe}
TRAIN=${TRAIN:-data/train-p99}
VAL=${VAL:-data/val-p99}
N_TRAIN=${N_TRAIN:-98304}
N_VAL=${N_VAL:-996}
CLIP=${CLIP:-5000.0}
BATCH=${BATCH:-64}
WARMUP=${WARMUP:-2000}
DEEPSUP=${DEEPSUP:-0.5}
NORM=${NORM:-group}
# Whatever the ladder is finally published under, this grid must use the same
# head -- an LR grid on a different parameterization answers a different
# question. See --head-bound in train_encoder.py for which are live and which
# are recorded failures.
HEAD_BOUND=${HEAD_BOUND:-tanh}
HEAD_GRAD_FLOOR=${HEAD_GRAD_FLOOR:-0.05}
HEAD_CAP=${HEAD_CAP:-3.0}
ADAM_EPS=${ADAM_EPS:-1e-8}
HEAD_HINGE=${HEAD_HINGE:-0}
# The rest of the encoder, also from the sweep120k_* saved args. These are NOT
# train_encoder's defaults, and the gap is not cosmetic: --n-fft 4096 / --hop
# 1024 is the encoder's *input* resolution, and the parameters are fixed by
# where the modes sit, so halving it halves the precision with which a mode can
# be located. It also matches the loss's own 4096, which the default 2048 does
# not. --n-blocks 6 is a deeper trunk than the default 5, and the sweep ran
# with gradient checkpointing off.
N_FFT=${N_FFT:-4096}
HOP=${HOP:-1024}
N_BLOCKS=${N_BLOCKS:-6}
EVAL_EVERY=${EVAL_EVERY:-2000}
GRAD_CKPT=${GRAD_CKPT:---no-grad-checkpoint}
NUMERICS=${NUMERICS:-"--batched-plate --compile-plate --chunk-elems 1000000000 --mode-bucket 1024 --fixed-mode-grid 86,282"}
EXTRA=${EXTRA:-""}

# Both directions from the ladder's 3e-4. Probing only downward invites exactly
# the objection this is meant to close.
LRS=${LRS:-"1e-3 3e-4 1e-4 3e-5 1e-5"}
LOSSES=${LOSSES:-"L1_STFT L1_STFT_eps1 L1_STFT_eps1e7"}

# `trap - INT TERM` first: kill 0 signals this script's own process group,
# which includes this script, so without disarming the handler it re-enters
# itself and bash dies messily (observed as a segfault on core dump) instead
# of exiting 130.
trap 'trap - INT TERM; echo; echo "interrupted -- stopping all cells"; kill 0; exit 130' INT TERM

read -r -a GPU_ARR <<< "$GPUS"
NG=${#GPU_ARR[@]}

CELLS=()
for loss in $LOSSES; do
  for lr in $LRS; do CELLS+=("$loss@$lr"); done
done

mkdir -p "$OUT"
echo "grid: ${#CELLS[@]} cells = $(echo "$LOSSES" | wc -w) losses x $(echo "$LRS" | wc -w) rates"
echo "gpus: ${GPU_ARR[*]}  steps: $STEPS  out: $OUT"
{
  echo "steps=$STEPS losses='$LOSSES' lrs='$LRS' gpus='$GPUS'"
  echo "train=$TRAIN n_train=$N_TRAIN  val=$VAL n_val=$N_VAL"
  echo "n_fft=$N_FFT hop=$HOP n_blocks=$N_BLOCKS grad_ckpt=$GRAD_CKPT"
  echo "head_bound=$HEAD_BOUND head_grad_floor=$HEAD_GRAD_FLOOR head_cap=$HEAD_CAP adam_eps=$ADAM_EPS head_hinge=$HEAD_HINGE norm=$NORM grad_clip=$CLIP batch=$BATCH warmup=$WARMUP deep_sup=$DEEPSUP constant_lr=yes"
  echo "numerics='$NUMERICS'"
  echo "extra='$EXTRA'"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$OUT/sweep_command.txt"
cat "$OUT/sweep_command.txt"

for ((g = 0; g < NG; g++)); do
  (
    for ((i = g; i < ${#CELLS[@]}; i += NG)); do
      cell=${CELLS[$i]}
      loss=${cell%@*}
      lr=${cell#*@}
      name="${loss}_${lr}"

      # A finished cell is not re-run, and the legacy results/ddsp/lr_* runs
      # count as finished cells: eps1 is c2 and eps1e7 is log, same conditions
      # under the pre-ladder names, so six of the fifteen are already in hand.
      legacy=""
      case "$loss" in
        L1_STFT_eps1)   legacy="results/ddsp/lr_L1_STFT_c2_${lr}" ;;
        L1_STFT_eps1e7) legacy="results/ddsp/lr_L1_STFT_log_${lr}" ;;
        *)              legacy="results/ddsp/lr_${loss}_${lr}" ;;
      esac

      done_cell=""
      for cand in "$OUT/$name" "$legacy"; do
        if [[ -f "$cand/history.json" ]] && python - "$cand/history.json" "$STEPS" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
h = d.get("history", [])
sys.exit(0 if h and h[-1]["step"] >= int(sys.argv[2]) else 1)
PY
        then done_cell="$cand"; break; fi
      done
      if [[ -n "$done_cell" ]]; then
        echo "  skip $name (already complete: $done_cell)"
        continue
      fi

      rc=0
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=${GPU_ARR[$g]} python -m src.ddsp.train_encoder \
        --loss "$loss" \
        --output-dir "$OUT/$name" \
        --data-dir "$TRAIN" --n-train "$N_TRAIN" \
        --val-data-dir "$VAL" --n-val "$N_VAL" \
        --peak-normalize target \
        --steps "$STEPS" \
        --lr "$lr" \
        --lr-floor 1.0 --lr-hold-frac 1.0 \
        --warmup-steps "$WARMUP" \
        --batch-size "$BATCH" \
        --deep-supervision "$DEEPSUP" \
        --grad-clip "$CLIP" \
        --norm "$NORM" --head-bound "$HEAD_BOUND" --head-hinge "$HEAD_HINGE" \
        --head-grad-floor "$HEAD_GRAD_FLOOR" --head-cap "$HEAD_CAP" \
        --adam-eps "$ADAM_EPS" \
        --n-fft "$N_FFT" --hop "$HOP" --n-blocks "$N_BLOCKS" \
        --eval-every "$EVAL_EVERY" \
        --seed 0 \
        $NUMERICS $GRAD_CKPT $EXTRA \
        > "$OUT/$name.log" 2>&1 || rc=$?

      if [[ $rc -eq 130 || $rc -eq 143 ]]; then
        echo "interrupted during $name (gpu ${GPU_ARR[$g]}) -- not starting the rest"
        break
      elif [[ $rc -ne 0 ]]; then
        echo "FAILED $name rc=$rc (gpu ${GPU_ARR[$g]}) -- see $OUT/$name.log" \
          | tee -a "$OUT/failures.txt"
      fi
    done
  ) &
  echo "  gpu ${GPU_ARR[$g]} -> $(for ((i = g; i < ${#CELLS[@]}; i += NG)); do printf '%s ' "${CELLS[$i]}"; done)"
done
wait
echo "lr probe complete: $OUT"
echo "table: python -m src.ddsp.report_lr_probe"
