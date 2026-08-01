#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.cmaes.fit_7param \
  --dset_root "${REPO_ROOT}/data/random-IR-100-1.0s" \
  --n_samples 100 \
  --output_dir results/cmaes/instfreq \
  --storage results/cmaes/instfreq/cmaes_lhs_instfreq.db \
  --loss "InstFreq" \
  "$@"
