#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.cmaes.fit_7param \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --n_samples 100 \
  --duration 0.09288 \
  --output_dir results/cmaes/l1_stft1024_4096 \
  --storage results/cmaes/l1_stft1024_4096/cmaes_lhs_l1stft.db \
  --loss "L1_STFT_1024" \
  "$@"

# i.e. exactly 4096 samples at 44.1kHz
