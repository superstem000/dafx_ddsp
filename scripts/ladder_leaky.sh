#!/usr/bin/env bash
# The leakytanh ladder, with every variable baked in.
#
# This is not a convenience wrapper. Passing the six settings on the command
# line as `VAR=x VAR=y scripts/eps_ladder.sh` is a real hazard: if the pasted
# line acquires a blank line after a trailing backslash, the continuation ends,
# each `VAR=x` becomes an unexported shell variable, and eps_ladder.sh runs on
# its defaults -- the wrong head, the wrong normalization, all seven arms, into
# the published tanh ladder's directory. It did exactly that once. Nothing in
# the output says so except a header that scrolls past.
#
# Baking them in makes that unreachable. There is nothing to omit.
#
#   scripts/ladder_leaky.sh 2                    # the control, alone
#   scripts/ladder_leaky.sh 2 "L1_STFT_eps1e7"   # one named arm
#
# Run the control first and alone. Under tanh it reaches nmse ~ 0.045 by step
# 2000 with sat 0.00; if leakytanh does not track that, the parameterization is
# the problem and the other six arms are wasted GPU-days.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/ladder_leaky.sh <gpu-list> [arms]"
  echo "   eg: scripts/ladder_leaky.sh 2"
  echo "       scripts/ladder_leaky.sh \"2 3\" \"L1_STFT_eps1 L1_STFT_eps1e7\""
  exit 2
fi

GPUS=$1
ARMS=${2:-"L1_STFT"}

# NORM=batch is not optional -- the tanh ladder these compare against ran
# --norm batch while eps_ladder.sh defaults to group, and mixing the two
# produces a run that cannot be attributed to the head bound.
#
# HEAD_HINGE=0 is stated rather than inherited. Adam is per-coordinate
# scale-free, so a penalty on the pre-activation sets direction and never
# magnitude; it did not hold stclamp back and there is no reason to expect it
# to do anything here.
NORM=batch \
HEAD_BOUND=leakytanh \
HEAD_GRAD_FLOOR=0.05 \
HEAD_HINGE=0 \
OUT=${OUT:-results/ddsp/eps_ladder_leaky} \
ARMS="$ARMS" \
  scripts/eps_ladder.sh "$GPUS"
