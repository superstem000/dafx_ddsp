#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.cmaes.fit_7param_norm_es \
  --dset_root "${REPO_ROOT}/data/2026-DATASET-STRIPPED" \
  --n_samples 16 \
  --output_dir results/final/phase1 \
  --storage results/final/phase1/phase1.db \
  --loss "L1_STFT" \
  --sigma0 0.6 \
  --early_stop_loss 0.01 \
  --tolfun 1e-5 \
  --tolfunhist 1e-5 \
  "$@"
