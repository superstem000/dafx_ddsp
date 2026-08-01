#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.cmaes.fit_7param_norm_es \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --n_samples 100 \
  --output_dir results/cmaes_norm_es/l1_stft_tolfun_prunemore \
  --storage results/cmaes_norm_es/l1_stft_tolfun_prunemore/cmaes_lhs_norm_es_l1stft.db \
  --loss "L1_STFT" \
  --sigma0 0.6 \
  --early_stop_loss 0.01 \
  --budget 30000 \
  --tolfun 1e-5 \
  --tolfunhist 1e-5 \
  --pruner_reduction_factor 4 \
  "$@"
