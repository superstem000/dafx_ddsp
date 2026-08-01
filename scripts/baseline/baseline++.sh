#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.baseline.baseline_plus \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --n_samples 100 \
  --budget 5000 \
  --num_particles_min 30 \
  --num_particles_max 60 \
  --loss "L1_STFT" \
  --output_dir results/baseline_pso_plusplus \
  --storage results/baseline_pso_plusplus/baseline_plusplus_optuna.db \
  "$@"

# using 5000 = far more than the time budget in the baseline's favor, to show that even with a much larger budget, the improvement from the baseline is still significant.