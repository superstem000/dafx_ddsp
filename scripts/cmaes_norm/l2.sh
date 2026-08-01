#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.cmaes.fit_7param_norm \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --n_samples 100 \
  --output_dir results/cmaes_norm/l2_4096 \
  --storage results/cmaes_norm/l2_4096/cmaes_lhs_l2_norm.db \
  --duration 0.09288 \
  --loss "L2" \
  --sigma0 0.6 \
  "$@"
