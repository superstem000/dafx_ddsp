#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.mu_optimization.ternary_mu \
  --cmaes_results "${REPO_ROOT}/results/final/phase1" \
  --dset_root "${REPO_ROOT}/data/2026-DATASET-STRIPPED" \
  --output_dir results/final/phase2 \
  --loss "L1_STFT" \
  "$@" > phase2.log
