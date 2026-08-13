#!/usr/bin/env bash
# Does the log arm's failure survive a head that cannot saturate?
#
# The encoder ladder's finding is that only the linear loss recovers the
# parameters; every log arm sits above the constant-predictor floor. The
# mechanism looked like tanh saturation: the failing arms' Ly, op_x and op_y
# score 1/3 normalized squared error -- a constant at the EDGE of a uniform
# range -- with prediction spread 0.000 and |z| up to 60, while the linear arm
# has |z|max 3.3 and spread intact. The trunk still discriminates in the failing
# arms (feature sd 0.1969 against the linear arm's 0.0569), so it is the head
# that discards the information.
#
# That makes it a fair reviewer question: did you try the standard fixes? Two
# attempts already failed, both by keeping gradient alive at large |z| --
# stclamp took op_x to 50636 and leakytanh to 577204, on the LINEAR control.
# The lesson is that |z| growing is the failure, so the remedy has to bound |z|
# rather than its gradient. This grid runs the two that do.
#
# Six cells: three head bounds x two losses, at 12000 steps.
#
#   tanh      the current parameterization, as the reference
#   normtanh  LayerNorm on the head's input, removing the trunk drift that
#             supplies most of |z|
#   softcap   out = tanh(c*tanh(z/c)), so the outward pull decays and runaway
#             is impossible
#
#   L1_STFT         the healthy control -- any head that breaks this is out
#   L1_STFT_eps1e1  eps = 0.1. Every log rung fails on the plate, so this is
#                   the MILDEST failing arm rather than the worst: a head that
#                   cannot rescue mild compression is evidence about
#                   compression, while "eps = 1e-7 still collapses" invites
#                   "of course it does"
#
# adam_eps is 1e-8 in all six, against the 1e-16 the earlier ladders used. Once
# a coordinate's surface goes flat, m and v are both ~0 and m/(sqrt(v)+1e-16) is
# noise over noise, so the coordinate takes full-lr steps in a rounding-
# determined direction; at 1e-8 the update decays instead. That is independent
# of the head and may be most of the runaway on its own -- which is exactly why
# the tanh row is in the grid. tanh + adam_eps 1e-8 against the existing tanh +
# 1e-16 ladder isolates the optimizer change from the head change.
#
# Reading it:
#   log recovers under normtanh or softcap  -> the encoder result was a head
#       artifact. Unwelcome, but far better found now than in review.
#   log still fails with |z| bounded        -> the failure survives the obvious
#       fix and is about the loss. That is the clean version of the claim.
#   linear breaks under a head              -> that head is disqualified,
#       whatever it did to the log arm.
#
# 12000 steps rather than 40000 because collapse is not a late phenomenon: the
# log arms sit at the floor from the first eval. This locates the answer, and
# only the surviving configuration is worth a full-length ladder.
#
#   scripts/head_probe.sh "0 1 2 3"
#   HEADS="normtanh" scripts/head_probe.sh "0 1"
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/head_probe.sh <gpu-list>"
  echo "   eg: scripts/head_probe.sh \"0 1 2 3\""
  exit 2
fi

GPUS=$1
HEADS=${HEADS:-"tanh normtanh softcap"}
LOSSES=${LOSSES:-"L1_STFT L1_STFT_eps1e1"}
STEPS=${STEPS:-12000}
ROOT=${ROOT:-results/ddsp/head_probe}

read -r -a GPU_ARR <<< "$GPUS"
NG=${#GPU_ARR[@]}

CELLS=()
for h in $HEADS; do
  for l in $LOSSES; do CELLS+=("$h@$l"); done
done

echo "grid: ${#CELLS[@]} cells = $(echo "$HEADS" | wc -w) heads x $(echo "$LOSSES" | wc -w) losses"
echo "gpus: ${GPU_ARR[*]}  steps: $STEPS  root: $ROOT"
for c in "${CELLS[@]}"; do echo "  $c"; done
echo

trap 'trap - INT TERM; echo; echo "interrupted -- stopping all cells"; kill 0; exit 130' INT TERM

# Each cell is one arm of eps_ladder.sh, so every setting the published ladder
# uses -- datasets, numerics, schedule, batch, clip -- comes from there rather
# than being restated here and drifting.
#
# NORM=batch is not optional: the tanh ladder these compare against ran
# --norm batch while eps_ladder.sh defaults to group.
for ((g = 0; g < NG; g++)); do
  (
    for ((i = g; i < ${#CELLS[@]}; i += NG)); do
      head=${CELLS[$i]%@*}
      loss=${CELLS[$i]#*@}
      rc=0
      NORM=batch \
      HEAD_BOUND="$head" \
      HEAD_CAP=3.0 \
      ADAM_EPS=1e-8 \
      HEAD_HINGE=0 \
      STEPS="$STEPS" \
      OUT="$ROOT/$head" \
      ARMS="$loss" \
        scripts/eps_ladder.sh "${GPU_ARR[$g]}" || rc=$?

      if [[ $rc -eq 130 || $rc -eq 143 ]]; then
        echo "interrupted during $head/$loss -- not starting the rest"
        break
      elif [[ $rc -ne 0 ]]; then
        echo "FAILED $head/$loss rc=$rc" | tee -a "$ROOT/failures.txt"
      fi
    done
  ) &
  echo "  gpu ${GPU_ARR[$g]} -> $(for ((i = g; i < ${#CELLS[@]}; i += NG)); do printf '%s ' "${CELLS[$i]}"; done)"
done
wait

echo
echo "probe complete: $ROOT"
echo "compare with:"
for h in $HEADS; do
  echo "  python -m src.ddsp.monitor_sweep --root $ROOT/$h --metrics nmse ratio zmax_op_x sat"
done
