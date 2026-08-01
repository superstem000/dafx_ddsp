#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.gd.graddescent \
  --dataset-dir "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --output-dir "${REPO_ROOT}/results/gd/graddescent_l1_stft" \
  --storage "${REPO_ROOT}/results/gd/graddescent_l1_stft/gd_optuna.db" \
  --loss "L1_STFT" \
  --num 10 \
  --n-trials 20 \
  --lr 0.1 \
  --adam-beta1 0.9 \
  --grad-clip-value 1.0 \
  --normalize-target --normalize-pred \
  "$@"
