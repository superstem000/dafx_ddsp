#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

mapfile -t SUMMARY_FILES < <(find "${REPO_ROOT}/results/cmaes" -type f -name "summary.csv" | sort)

if [ "${#SUMMARY_FILES[@]}" -eq 0 ]; then
  echo "No summary.csv files found under ${REPO_ROOT}/results/cmaes" >&2
  exit 1
fi

"${PYTHON_BIN}" -m src.diagnostics.visualize_fails \
  --summaries "${SUMMARY_FILES[@]}" \
  --success_threshold 0.01 \
  --out_png "${REPO_ROOT}/results/diagnostics/visualize_fails/fail_scatter.png" \
  "$@"
