#!/usr/bin/env bash
# =============================================================================
# Gradient descent over the identifiable 6-parameter plate space.
#
# Unlike the CMA-ES pipeline this is a SINGLE stage: mu is solved analytically
# inside the run (see src/gd/graddescent.py), so there is no stage-2
# ternary_mu step and the target IR must NOT be peak-normalized.
#
# Optional overrides (env vars):
#   PYTHON_BIN, DSET_ROOT, RESULT_ROOT, LOSS, NUM, N_RESTARTS, DURATION
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DSET_ROOT="${DSET_ROOT:-${REPO_ROOT}/random-IR-200-0.2s}"
LOSS="${LOSS:-L1_STFT}"
NUM="${NUM:-10}"
N_RESTARTS="${N_RESTARTS:-16}"
DURATION="${DURATION:-0.25}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/gd/graddescent_$(echo "${LOSS}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' '_')}"

"${PYTHON_BIN}" -m src.gd.graddescent \
  --dataset-dir "${DSET_ROOT}" \
  --output-dir "${RESULT_ROOT}" \
  --loss "${LOSS}" \
  --num "${NUM}" \
  --duration "${DURATION}" \
  --n-restarts "${N_RESTARTS}" \
  --restart-batch 16 \
  --n-epochs 400 \
  --lr 0.02 \
  --lr-schedule cosine \
  --grad-clip-type norm \
  --grad-clip-value 1.0 \
  --mu-mode profile \
  --init-space raw7 \
  --seed 42 \
  "$@"
