# Task A Submission:

Submission for Task A of the 1st DAFx Parameter Estimation Challenge.

## Hardware

1 Quadro RTX 6000 on a system with 48 Intel(R) Xeon(R) Silver 4214 @ 2.20GHz CPU and 256GB RAM.

## Compute time (16 target IRs)

- Stage 1: 1827 s total (mean 114 s per IR; range 11 s – 735 s)
- Stage 2: 13.0 s total (mean 0.66 s per IR)
- **Total: 1840 s**

## Iterations / trials

- **Stage 1** (CMA-ES with restarts): 220 total "runs" across the 16 IRs (range 1 – 168 per IR (cap 400; mean ≈ 14)), **totaling 211933 loss evaluations** (mean ≈ 13k / IR). Each "run" was allocated a budget of 25,000 loss evaluations which can be spread over multiple generations of CMA-ES search. However, in practice all runs terminated much earlier due to other stopping criteria. If any complete run reaches loss < 0.01 on an IR, no further runs were initiated on it. Run counts, CMA-ES generation counts, and loss evaluation counts can be found in `outputs/phase1_summary.csv`. Per-IR details can be found in `outputs/phase1.log`.
- **Stage 2** (ternary search on μ): 50 ternary search iterations per IR (800 total), corresponding to exactly **122 loss evaluations per IR** including sanity checks.

## Reproducing

1. make a new python 3.12 venv, activate it, and do `pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128 && pip install -r requirements.txt`
2. Place the official target dataset at `data/2026-DATASET-STRIPPED/`
3. Stage 1: `bash scripts/phase1.sh` (writes to `results/final/phase1/`)
4. Stage 2: `bash scripts/phase2.sh` (writes to `results/final/phase2/`)
5. Reformat to `submission.csv`, and/or run code in `src/postprocessing` to produce the per-IR csv and the aggregated summary.

## Contents

- `submission.csv` — **final parameter estimates for the 16 target IRs**.
- `src/` — source code (Stage 1, Stage 2, loss functions, plate models).
- `scripts/` — exact shell scripts used for the submitted run.
- `outputs/` — pipeline logs and aggregate CSVs.
- `submission/` — **Contains the per-IR .csv files.**
- `requirements.txt` — Python dependencies.
- `paper.pdf` — Our writeup. 
