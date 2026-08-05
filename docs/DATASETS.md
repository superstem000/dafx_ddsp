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
