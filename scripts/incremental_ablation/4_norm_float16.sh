#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.cmaes.fit_7param_norm \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --n_samples 100 \
  --output_dir results/cmaes/incremental_ablation/4_norm_float64 \
  --storage results/cmaes/incremental_ablation/4_norm_float64/cmaes_lhs_norm_float64.db \
  --loss "L1_STFT" \
  --sigma0 0.6 \
  --dtype "float64" \
  "$@"
