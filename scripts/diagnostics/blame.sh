#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

INPUT_CSV="${1:-${REPO_ROOT}/results/ablation/l1_stft_amp/summary.csv}"
OUT_DIR="${2:-${REPO_ROOT}/results/diagnostics/blame}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.diagnostics.blame "${INPUT_CSV}" --out_dir "${OUT_DIR}"
