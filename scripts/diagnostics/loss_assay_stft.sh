#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.diagnostics.loss_assay_stft \
  --output_dir "${REPO_ROOT}/results/diagnostics/loss_assay_stft" \
  --device "cuda" \
  --dtype "float32" \
  --n_targets 20 \
  --n_samples 100 \
  --batch_size 50 \
  --n_bins 5 \
  --n_walks 10 \
  --walk_length 500 \
  --n_mono_starts 20 \
  --n_mono_steps 100 \
  --fft_configs "4096" "512,1024,2048,4096" "1024,2048,4096" "2048,4096" "500,1372,2916,4096" \
  --reduction "mean" \
  --lengths 11025 \
  --seed 42 \
  "$@"
