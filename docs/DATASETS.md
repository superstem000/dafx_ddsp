# Datasets: what travels, what gets regenerated

The `data/` directory is ~19 GB and is gitignored. None of it is pushed. What
moves between machines is parameters — 6.5 MB and 168 KB — and the audio is
re-rendered on arrival.

That is not only a size decision. A different GPU reduces float32 in a
different order, so targets copied from one machine stop being bit-identical to
synthesis on another, and the loss at true parameters comes back nonzero. On
the L4 this mattered enormously for the compression sweep: before the fixes
below, `log(x + 1e-7)` scored **19.98% of saturation at the true parameters**,
with the quietest magnitude decile carrying 27.4% of the disagreement against
3.7% for linear. A sweep run that way reports our own numerics, concentrated in
exactly the bins log weights most, as if it were what compression does to the
terrain.

## Provenance

| set | parameters | reproducible from |
|---|---|---|
| train, 100000 | `data/train_100000_params.csv.gz` (6.5 MB) | `--number 100000 --seed 1`, verified to 8.7e-14 |
| val, 1000 | `data/val_1000_params.csv` (168 KB) | **no seed found** — predates the current sampler; the CSV is the only copy |

Ten candidate seeds were tried against the val set and all mismatched, so treat
`val_1000_params.csv` as irreplaceable.

A caveat when verifying against a directory: files are named `{i+1:04d}`, so
past 9999 the widths change and `sorted()` gives `"1000" < "10000" < "1001"`.
`--verify-against` compares positionally and will report a spurious mismatch on
any set larger than 9999. Match by the index in the filename instead. This is
harmless for training — `load_dataset` pairs each CSV with its npz by that
index, and with the grid pinned each row renders independently of its batch.

## Regenerating on a new machine

```bash
python -m src.data.make_dataset --params-csv data/val_1000_params.csv \
  --output-dir data/val-p99 --duration 0.25 --render-path training \
  --batched-plate --compile-plate --chunk-elems 1000000000 \
  --mode-bucket 1024 --batch-size 64 --fixed-mode-grid 86,282

python -m src.data.make_dataset --params-csv data/train_100000_params.csv.gz \
  --output-dir data/train-p99 --duration 0.25 --render-path training \
  --batched-plate --compile-plate --chunk-elems 1000000000 \
  --mode-bucket 1024 --batch-size 64 --fixed-mode-grid 86,282
```

Expect `excludes 4 of 1000 (0.40%)` and `excludes 1661 of 100000 (1.66%)` —
plates needing a finer grid than the pin, dropped so no target is rendered
truncated. Leaves 996 val and 98339 train.

Every flag matters and all three of these were found the hard way:

- `--render-path training` renders through the float32 `z` the encoder emits.
  Building plate14 straight from the CSV instead leaves `T0` quantised by its
  *range* rather than its value (a ~6e-5 quantum on a range of `(0.01, 1000)`),
  which is ~1e-4 on the mode frequencies — invisible to a linear loss, 13% of
  saturation to log.
- `--compile-plate`, `--chunk-elems`, `--mode-bucket` must match the training
  run's. A fused kernel is different arithmetic.
- `--fixed-mode-grid 86,282` stops `n_modes` depending on the batch maximum, so
  an IR renders the same whoever it shares a batch with. Without it, random
  training batches disagree with the rendering batch by 6.1% of saturation for
  log. The pin is the p99 grid, not the global max: the max costs 1.79x on
  every step of every run to accommodate the top 1.7% of IRs, while the p99 is
  24,252 modes — what an unpinned batch already paid.

## Verify before training anything

```bash
python -m src.ddsp.diag_gt_floor --data-dir data/val-p99 --n-val 512 \
  --compile-plate --chunk-elems 1000000000 --fixed-mode-grid 86,282
```

`training path (batched, batch=N)` and `training path, SHUFFLED batches` must
both read `0.0000e+00`, or within ~1e-3 for log. The shuffled row is the one
that matters: the sorted row reproduces the batching generation used and agrees
by construction, while shuffled is the condition training actually faces.

Anything at percent-level means a flag above does not match what the data was
rendered with, and the sweep is not attributable until it does.

Run counts for the methods section: 1.66% of training and 0.40% of validation
IRs excluded as needing more than 86x282 modes below Nyquist; loss at true
parameters 0.0000% of saturation for linear and at most 0.074% for log, the
residual being `torch.compile` selecting different kernels in different
processes (two runs of the same command gave 0.074% and exactly 0).

## The quiet3 family — a task whose information is in the quiet region

Everything above describes `raw7`, where the seven searched parameters are
geometry and tension and the damping is pinned. `quiet3` inverts that: geometry
is pinned at val IR 0001 and the three searched parameters are `T0`, `T60_DC`
and `T60_ratio`, with `T60_F1 = T60_ratio * T60_DC`. It exists because the
thesis is about what compression does to *quiet* bins, and in `raw7` a
parameter change is mostly a change to where the loud modes sit. Here it is
mostly a change to how the tail decays.

Which parameters, which bounds and which are pinned all live in
`PARAM_SPACES` in `src/cmaes/fit_7param_norm_es.py`, selected by the
`PLATE_PARAM_SPACE` environment variable, so the generator, the encoder, the
plate packing and the diagnostics cannot disagree about them. **Set it for
generation and for training, to the same value.** The bounds and the evidence
for them are documented at `PLATE_QUIET3` there; in short, measured with
`diag_param_sensitivity --vary`, all three parameters move 10-35% of the
family's own saturation at a 10%-of-range nudge, and all three put the centroid
of that change at or below the neutral 5.5th magnitude decile.

```bash
export PLATE_PARAM_SPACE=quiet3

# 0. Confirm the pinned geometry is the plate the sensitivity sweep measured.
#    --vary pins everything it does not vary at the FIRST parameter set in the
#    directory it reads, so the ranges in PARAM_SPACES describe that one plate;
#    this must say MATCH, against whichever directory that sweep was run on.
python -m src.data.make_dataset --check-pin data/val-p99

# 1. The pin. Geometry is fixed, so every IR in this family needs the same
#    modal grid and this is a formality -- but run it rather than trusting the
#    number below, since it is what the whole dataset is rendered under.
python -m src.data.make_dataset --number 1000 --report-grid

# 2. Generate. NO --compile-plate -- see below. Nothing is excluded: with the
#    grid constant, no parameter set can need a finer one than the pin.
python -m src.data.make_dataset --number 1000 --seed 2 \
  --output-dir data/val-quiet3 --duration 0.25 --render-path training \
  --batched-plate --chunk-elems 1000000000 \
  --mode-bucket 1024 --batch-size 64 --fixed-mode-grid 30,92

python -m src.data.make_dataset --number 100000 --seed 3 \
  --output-dir data/train-quiet3 --duration 0.25 --render-path training \
  --batched-plate --chunk-elems 1000000000 \
  --mode-bucket 1024 --batch-size 64 --fixed-mode-grid 30,92

# 3. Verify, exactly as above. Same requirement: 0.0000e+00 on the SHUFFLED row.
python -m src.ddsp.diag_gt_floor --data-dir data/val-quiet3 --n-val 512 \
  --chunk-elems 1000000000 --fixed-mode-grid 30,92
```

**`--compile-plate` is off for this family, in generation and in training.**
With it, targets rendered by one process and re-synthesized by another disagree
by **7.66% of saturation for log**, 42.6% of it in the quietest decile — the
exact failure shape this diagnostic exists to catch. It is not run-to-run
noise: two runs of the identical command reproduced the figure to the digit, so
each process settles on a stable choice of reduction kernel and the choices
differ between them. Eager renders `0.0000e+00`, with `target and training
synthesis agree bit-for-bit; nothing to decompose`.

`raw7` tolerated compile at ≤0.074% because its quiet bins were never this far
down. `quiet3`'s sit at 3.3e-06 against a peak of 13.4 — at the float32
cancellation floor of the modal sum, which is both the point of the family and
what makes it unforgiving. In the eager table the deliberately mismatched paths
(`batch=1`, unbatched modal sum, no-float32-z) still cost log 3.3–9.1% of
saturation, so every numeric flag has to match what the data was rendered with,
exactly. Eager also generated *faster* here (119 IR/s against 74), so the only
real cost is training step time.

30x92 is 2,760 modes against `raw7`'s 24,252, so both generation and training
are roughly 9x cheaper per step here. Sampling is uniform in `z`, which is
log-uniform in `T0` — that parameter spans two decades and is log-scaled in the
normalized coordinate, so the bottom decade is neither compressed into the last
1% of what the encoder emits nor left unsampled. It is the distribution the
sensitivity sweep measured.

Two differences from `raw7` worth stating before they surprise someone:

- **No composite reduction.** `raw7` reports `val_nmse_6d` because
  `(E, rho, h) -> (c^3 E, c rho, h/c)` leaves the IR identical and scoring
  those three individually would measure drift the loss cannot see. `quiet3`
  has no such symmetry — every parameter is separately identifiable — so
  `val_nmse_6d` is the plain NMSE over the three searched parameters, reported
  under the existing key so the monitor and the plots read unchanged, and
  `val_nmse_7d` beside it is the same number. `--fit-mu` is forced off for the
  same reason: `mu = rho*h` is pinned, so there is no scale left to fit.
- **No batch-composition term.** `n_modes` is constant across the whole
  dataset, not merely pinned per run, so the reduction tree is the same for
  every batch and the one residual `raw7` could only bound is absent here.
  Confirmed: the batched and SHUFFLED rows are identical to the digit.
