#!/usr/bin/env bash
#
# PSO + Optuna + study-level early stop at loss < 0.01.
# 40 trials × 5000 evals each = up to ~200k evals/IR (matches existing
# `baseline_pso_plusplus` budget but adds the early-stop callback so it
# fires the same as CMA-ES for fair comparison).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_DIR="results/experiment_pruning_es/pso_es_5k"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m experiment_pruning_es.src.baseline_plus_es \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --n_samples 100 \
  --n_trials 40 \
  --budget 5000 \
  --duration 0.25 \
  --num_particles_min 30 \
  --num_particles_max 60 \
  --loss "L1_STFT" \
  --early_stop_loss 0.01 \
  --output_dir "${OUTPUT_DIR}" \
  --storage "${OUTPUT_DIR}/baseline_plus_es.db" \
  "$@"
