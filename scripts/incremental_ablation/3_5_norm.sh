#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.cmaes.fit_5param_norm \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --n_samples 100 \
  --output_dir results/cmaes/incremental_ablation/3_5_norm \
  --storage results/cmaes/incremental_ablation/3_5_norm/cmaes_lhs_5_norm.db \
  --loss "L1_STFT" \
  --sigma0 0.6 \
  "$@"
