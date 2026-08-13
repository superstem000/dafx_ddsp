#!/usr/bin/env bash
# =============================================================================
# The compression ladder at ONE CMA-ES restart, both stages, six losses.
#
# RECONSTRUCTED from results/ladder_1restart/*/stage{1,2}.log, which is the only
# record of how those runs were launched -- unlike standard_sweep there was
# never a script, so re-running the result meant reading a log. Every value
# below is either printed in those logs or is the argparse default; nothing is
# guessed. Verified against results/ladder_1restart/l1_stft/stage1.log:
#
#   Trials: 1        Duration: 0.25s      sigma0: 0.6
#   tolfun: 1e-05    tolfunhist: 1e-05
#   pruner_min_resource: 15               pruner_reduction_factor: 3
#   early_threshold=0.007892 (alpha=0.0095)   tolfun frac=0.0052 of terrain range
#
# and stage2.log:
#
#   Targets: random-IR-200-0.2s           Ternary iterations: 50
#   Sanity grid check: on                 mu bounds [2.430, 106.150]
#
# TARGETS: random-IR-200-0.2s carries TORCH-rendered IRs, not the numpy ones
# DatasetGen wrote -- gen_torch_targets_200.py re-rendered them in place, which
# is why gt_loss is exactly 0.0 for every arm here and ~1.4e-05 in
# standard_sweep. That zero is the point (no target/synthesis floor can be
# blamed for the ranking) and also the caveat (it is a matched-model setting).
# See paper/datasets/GENERATORS.md.
#
#   scripts/ladder_1restart.sh
#   LOSSES_OVERRIDE="L1_STFT L1_STFT_log" scripts/ladder_1restart.sh
# =============================================================================
set -uo pipefail   # no -e: one failing loss must not abort the ladder

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DSET_ROOT="${DSET_ROOT:-${REPO_ROOT}/random-IR-200-0.2s}"
N_SAMPLES="${N_SAMPLES:-200}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/ladder_1restart}"
DTYPE="${DTYPE:-float32}"
LOG="${RESULT_ROOT}/ladder_run.log"

declare -A SLUG=(
  ["L1_STFT"]="l1_stft" ["L1_STFT_c2"]="l1_stft_c2" ["L1_STFT_log"]="l1_stft_log"
  ["L1_STFT_pow"]="l1_stft_pow" ["MSS"]="mss" ["SmoothMSS"]="smoothmss"
)
DEFAULT_LOSSES=("L1_STFT" "L1_STFT_c2" "L1_STFT_log" "L1_STFT_pow" "MSS" "SmoothMSS")
if [[ -n "${LOSSES_OVERRIDE:-}" ]]; then
  read -r -a LOSSES <<< "${LOSSES_OVERRIDE}"
else
  LOSSES=("${DEFAULT_LOSSES[@]}")
fi

mkdir -p "${RESULT_ROOT}"
logline() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

logline "=== ladder_1restart START ==="
logline "DSET_ROOT=${DSET_ROOT}  N_SAMPLES=${N_SAMPLES}  DTYPE=${DTYPE}"
logline "LOSSES (${#LOSSES[@]}): ${LOSSES[*]}"

for loss in "${LOSSES[@]}"; do
  slug="${SLUG[$loss]:-$(echo "$loss" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' '_')}"
  s1_out="${RESULT_ROOT}/${slug}/stage1"
  s2_out="${RESULT_ROOT}/${slug}/stage2"
  mkdir -p "${s1_out}" "${s2_out}"
  logline "----- [${loss}] (slug=${slug}) START -----"

  # --n_trials 1 is the whole experiment: it removes the restart budget as a
  # confound, so a difference between arms is the loss and not how many chances
  # each got.
  "${PYTHON_BIN}" -m src.cmaes.fit_7param_norm_es \
    --dset_root "${DSET_ROOT}" \
    --n_samples "${N_SAMPLES}" \
    --output_dir "${s1_out}" \
    --storage "${RESULT_ROOT}/${slug}/${slug}.db" \
    --loss "${loss}" \
    --n_trials 1 \
    --duration 0.25 \
    --sigma0 0.6 \
    --dtype "${DTYPE}" \
    --tolfun 1e-5 \
    --tolfunhist 1e-5 \
    > "${RESULT_ROOT}/${slug}/stage1.log" 2>&1
  logline "[${loss}] stage 1 rc=$?"

  "${PYTHON_BIN}" -m src.mu_optimization.ternary_mu \
    --cmaes_results "${s1_out}" \
    --dset_root "${DSET_ROOT}" \
    --output_dir "${s2_out}" \
    --loss "${loss}" \
    --duration 0.25 \
    --dtype "${DTYPE}" \
    --n_iters 50 \
    > "${RESULT_ROOT}/${slug}/stage2.log" 2>&1
  logline "[${loss}] stage 2 rc=$?"
done

logline "=== ladder_1restart DONE ==="
echo "table + figure: python -m src.analysis.compare_methods --plot docs/figures/nmse_ecdf.png"
