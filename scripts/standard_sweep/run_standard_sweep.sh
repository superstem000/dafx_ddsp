#!/usr/bin/env bash
# =============================================================================
# standard_sweep : run every standard loss through the full 2-stage pipeline.
#
#   Stage 1  ->  src.cmaes.fit_7param_norm_es      (CMA-ES, normalized space)
#   Stage 2  ->  src.mu_optimization.ternary_mu    (ternary mu refinement)
#
# Settings are IDENTICAL to l1stft_tolfun.sh / stage2_tolfun.sh. Only --loss,
# --output_dir, --storage (and stage-2 --cmaes_results) change per loss, so the
# comparison stays apples-to-apples.
#
# Place at:  scripts/standard_sweep/run_standard_sweep.sh
# Run under tmux so it survives disconnects:
#     tmux new -s sweep
#     ./scripts/standard_sweep/run_standard_sweep.sh
#     (detach: Ctrl-b then d   |   reattach: tmux attach -t sweep)
#
# Resumable: completed IRs are skipped by the fitter (no --overwrite), so a
# re-run continues where it stopped.
#
# Fault-isolated: a failure in ONE loss is logged and the loop CONTINUES to the
# next loss (deliberately no `set -e` across the loop). Worst case = one loss
# missing, never a wasted night.
#
# CORRECTED to what results/standard_sweep was actually produced with. The file
# had drifted after the run and would no longer reproduce it: it read
# --n_trials 1 and defaulted N_SAMPLES to 100, where the run used 20 and 50.
#
# Neither the script nor sweep_run.log recorded the difference -- the log carries
# DSET_ROOT, N_SAMPLES and DTYPE only. The results do: n_restarts_run is the
# count of terminal Optuna trials, and it caps at exactly 20 for every one of
# the 14 losses, with LSD, Mel, MSS and SC+LogMag sitting at 20 on all 50 IRs
# without ever early-stopping. A ceiling of exactly 20 on every loss is
# --n_trials 20 and nothing else. N_SAMPLES=50 is logged directly.
#
# Optional overrides (env vars):
#   PYTHON_BIN, DSET_ROOT, N_SAMPLES, RESULT_ROOT, DTYPE
#   LOSSES_OVERRIDE="L1_STFT MSS"   # run only a subset, in this order
#   DTYPE=float64                   # synthesis/loss precision (default float32)
# =============================================================================

set -uo pipefail   # NOTE: intentionally NO -e, so one failing loss won't abort the sweep

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DSET_ROOT="${DSET_ROOT:-${REPO_ROOT}/data/random-IR-100-1.0s}"
N_SAMPLES="${N_SAMPLES:-50}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/standard_sweep}"
DTYPE="${DTYPE:-float32}"
LOG="${RESULT_ROOT}/sweep_run.log"

mkdir -p "${RESULT_ROOT}"

# Fast tier first (STFT / mel / FFT based), then the slow tier (nnAudio / kymatio).
DEFAULT_LOSSES=(
  # ---- fast tier ----
  "L1_STFT" "MSS" "SC+LogMag" "Mel" "LSD" "ESR" "L1" "L2" "L1_STFT_1024"
  # ---- slow tier (nnAudio / kymatio) ----
  "CQT_L1" "VQT" "Gammatone" "MultiResCQT" "Scattering1D"
)
if [[ -n "${LOSSES_OVERRIDE:-}" ]]; then
  read -r -a LOSSES <<< "${LOSSES_OVERRIDE}"
else
  LOSSES=("${DEFAULT_LOSSES[@]}")
fi

slugify() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'; }
ts()      { date '+%Y-%m-%d %H:%M:%S'; }
logline() { echo "[$(ts)] $*" | tee -a "${LOG}"; }

logline "=== standard_sweep START ==="
logline "REPO_ROOT=${REPO_ROOT}"
logline "DSET_ROOT=${DSET_ROOT}  N_SAMPLES=${N_SAMPLES}"
logline "RESULT_ROOT=${RESULT_ROOT}"
logline "DTYPE=${DTYPE}"
logline "LOSSES (${#LOSSES[@]}): ${LOSSES[*]}"

declare -a STATUS
idx=0
for loss in "${LOSSES[@]}"; do
  slug="$(slugify "${loss}")"
  s1_out="${RESULT_ROOT}/${slug}/stage1"
  s2_out="${RESULT_ROOT}/${slug}/stage2"
  db="${s1_out}/cmaes.db"
  mkdir -p "${s1_out}" "${s2_out}"

  logline "----- [${loss}] (slug=${slug}) START -----"
  logline "[${loss}] watch:  tail -f ${RESULT_ROOT}/${slug}/stage1.log"
  loss_start=$(date +%s)

  # ---------------- Stage 1 : CMA-ES (normalized space) ----------------
  logline "[${loss}] stage 1: CMA-ES"
  "${PYTHON_BIN}" -m src.cmaes.fit_7param_norm_es \
    --dset_root "${DSET_ROOT}" \
    --n_samples "${N_SAMPLES}" \
    --output_dir "${s1_out}" \
    --storage "${db}" \
    --loss "${loss}" \
    --sigma0 0.6 \
    --dtype "${DTYPE}" \
    --early_stop_loss 0.01 \
    --budget 30000 \
    --n_trials 20 \
    --lhs_seed 42 \
    --tolfun 1e-5 \
    --tolfunhist 1e-5 \
    >> "${RESULT_ROOT}/${slug}/stage1.log" 2>&1
  s1_rc=$?
  if [[ ${s1_rc} -ne 0 ]]; then
    logline "[${loss}] stage 1 FAILED (rc=${s1_rc}) -> see ${slug}/stage1.log ; skipping stage 2"
    STATUS[$idx]="${loss}: STAGE1_FAIL(rc=${s1_rc})"
    idx=$((idx+1))
    continue
  fi

  # ---------------- Stage 2 : ternary mu refinement --------------------
  logline "[${loss}] stage 2: ternary mu"
  "${PYTHON_BIN}" -m src.mu_optimization.ternary_mu \
    --cmaes_results "${s1_out}" \
    --dset_root "${DSET_ROOT}" \
    --output_dir "${s2_out}" \
    --loss "${loss}" \
    --duration 0.25 \
    --device "cuda" \
    --dtype "${DTYPE}" \
    --n_iters 50 \
    --sanity_n 20 \
    >> "${RESULT_ROOT}/${slug}/stage2.log" 2>&1
  s2_rc=$?

  loss_end=$(date +%s); dt=$((loss_end - loss_start))
  if [[ ${s2_rc} -ne 0 ]]; then
    logline "[${loss}] stage 2 FAILED (rc=${s2_rc}) -> see ${slug}/stage2.log  (${dt}s)"
    STATUS[$idx]="${loss}: STAGE2_FAIL(rc=${s2_rc})"
  else
    logline "[${loss}] DONE in ${dt}s"
    STATUS[$idx]="${loss}: OK (${dt}s)"
  fi
  idx=$((idx+1))
done

logline "=== standard_sweep FINISHED ==="
for s in "${STATUS[@]}"; do logline "  ${s}"; done
logline "Aggregate:  ${PYTHON_BIN} src/standard_sweep/aggregate_sweep.py --root ${RESULT_ROOT} --plot"
