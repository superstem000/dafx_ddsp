#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.diagnostics.loss_assay \
  --output_dir "${REPO_ROOT}/results/diagnostics/loss_assay" \
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
  --losses l1_stft l1_cqt waveform_l2 waveform_l1 group_delay gammatone fft \
  --lengths 1024 2048 4096 \
  --n_ffts 128 256 512 1024 \
  --seed 42 \
  "$@"
