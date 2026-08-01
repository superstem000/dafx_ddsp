#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.mu_optimization.ternary_mu \
  --cmaes_results "${REPO_ROOT}/results/cmaes_norm_es/l1_stft_tolfun" \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --output_dir "${REPO_ROOT}/results/mu_refinement/ternary_tolfun" \
  --loss "L1_STFT" \
  --duration 0.25 \
  --device "cuda" \
  --dtype "float32" \
  --n_iters 50 \
  --sanity_n 20 \
  "$@"

