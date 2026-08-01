#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.baseline.baseline_plus \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --n_samples 100 \
  --budget 2000 \
  --duration 0.09288 \
  --num_particles_min 30 \
  --num_particles_max 60 \
  --loss "L1_STFT" \
  --output_dir results/baseline_pso_plus_4096 \
  --storage results/baseline_pso_plus_4096/baseline_plus_optuna.db \
  "$@"

# using 2000 approximately matches the time budget in the baseline's favor.