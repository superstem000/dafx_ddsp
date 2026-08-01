#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.ablation.fit_7param_norm_es_amp \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --n_samples 100 \
  --output_dir "${REPO_ROOT}/results/ablation/l1_stft_amp" \
  --storage "${REPO_ROOT}/results/ablation/l1_stft_amp/cmaes_lhs_norm_es_amp.db" \
  --loss "L1_STFT" \
  --sigma0 0.6 \
  --tolfun 0.00001 \
  --tolfunhist 0.00001 \
  --early_stop_loss 0.01 \
  "$@"

