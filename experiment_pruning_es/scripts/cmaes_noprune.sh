#!/usr/bin/env bash
#
# CMA-ES with NO pruning (NopPruner) + study-level early-stop at loss < 0.01.
# 40 trials × 5000 evals max per trial = up to ~200k evals/IR (matches PSO++).
#
# Compare against:
#   - results/cmaes_norm_es/l1_stft  (existing: with SHP, 400 trials × 25000)
#   - results/baseline_pso_plusplus  (existing: PSO 40 × 5000, no early stop)
#   - results/experiment_pruning_es/pso_es_5k  (companion run; PSO + ES)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_DIR="results/experiment_pruning_es/cmaes_noprune_5k"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m experiment_pruning_es.src.fit_cmaes_noprune \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --n_samples 100 \
  --n_trials 40 \
  --budget 5000 \
  --duration 0.25 \
  --loss "L1_STFT" \
  --sigma0 0.6 \
  --early_stop_loss 0.01 \
  --lhs_seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  --storage "${OUTPUT_DIR}/cmaes_noprune.db" \
  "$@"
