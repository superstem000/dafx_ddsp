#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.cmaes.fit_7param_norm_es \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --n_samples 100 \
  --duration 0.5 \
  --output_dir results/cmaes_norm_es/l1_stft_0p5 \
  --storage results/cmaes_norm_es/l1_stft_0p5/cmaes_lhs_norm_es_l1stft.db \
  --loss "L1_STFT" \
  --sigma0 0.6 \
  --early_stop_loss 0.01 \
  "$@"
