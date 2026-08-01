#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.diagnostics.success_rate_pso \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --output_dir "${REPO_ROOT}/results/diagnostics/success_rate_pso" \
  --loss "L1_STFT" \
  --dtype "float32" \
  --n_samples 20 \
  --n_runs 20 \
  --popsizes 10 30 50 100 \
  --success_threshold 0.01 \
  --budget 4000 \
  --w 0.2 \
  --c1 2.0 \
  --c2 1.0 \
  "$@"

# the budget is an attempt to match budget.