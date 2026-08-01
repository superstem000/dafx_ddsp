#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.baseline.fit_mu6_pso \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --output_dir "${REPO_ROOT}/results/pso/mu_only" \
  --loss "L1_STFT" \
  --dtype "float64" \
  --n_samples 10 \
  --num_particles 5 \
  --max_iter 300 \
  "$@"
