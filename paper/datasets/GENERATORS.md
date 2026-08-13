# Which script made which dataset, and which ones have a floor

Three generators produced the data in this paper, and they are not
interchangeable. `docs/DATASETS.md` covers the flags; this covers which script.

## `generators/ModalPlate/DatasetGen.py`

The original numpy plate. Produced the per-IR fitting sets:

| set | IRs | duration | generated |
|---|---|---|---|
| `random-IR-100-1.0s` | 100 | 1.0 s | 2026-04-24 |
| `random-IR-200-0.2s` | 200 | 0.25 s | 2026-07-28 |

These are what the CMA-ES sweeps and the gradient-descent runs fit. Each ships a
`generation_summary.txt` recording the parameter ranges, which are identical
between them: Lx 1.0 and nu 0.25 fixed, Ly [1.1, 4.0], h [0.001, 0.005],
T0 [0.01, 1000], rho [2430, 21230], E [6.7e10, 2.2e11].

## `generators/gen_torch_targets*.py` -- and the one that matters most

Re-render an existing set's IRs through the *torch* plate, so a target and the
model that fits it share a code path exactly. `gen_torch_targets.py` says what
that buys in its own docstring: "targets from the SAME torch synth the fitter
uses as candidate, so gt_loss ~ 0 (matched-model / inverse-crime diagnostic)".

**`gen_torch_targets_200.py` writes back into `random-IR-200-0.2s` in place**
(`out_dir = Path(src_dir)`). So that set's `.npz` files are torch-rendered while
its `generation_summary.txt` still records DatasetGen and 2026-07-28 -- the
summary describes the *parameters*, not the current rendering. Do not read it as
provenance for the audio.

That single fact separates the two CMA-ES results:

| run | IRs | dataset | rendering | `gt_loss` median |
|---|---|---|---|---|
| `standard_sweep/l1_stft` ("CMA-ES full") | 50 | `data/random-IR-100-1.0s` | numpy | **1.37e-05** |
| `ladder_1restart/*` (1 restart) | 200 | `random-IR-200-0.2s` | torch | **exactly 0** |
| `on_separate_50ir/phase1` | 50 | numpy | numpy | 1.33e-05 |

### Why this decides which comparison is usable

On the numpy targets the floor is not small, and it is not the same size for
every loss. Median across the 50 IRs of `standard_sweep`:

| loss | `gt_loss` | `best_loss` | floor as a fraction of what was achieved |
|---|---|---|---|
| L2 | 3.0e-12 | 8.6e-08 | 0.00 |
| ESR | 2.5e-11 | 9.8e-08 | 0.00 |
| L1 | 1.28e-06 | 1.35e-06 | 0.95 |
| L1_STFT | 1.37e-05 | 1.39e-05 | 0.98 |
| Mel | 0.188 | 0.522 | 0.36 |
| MSS | 0.818 | 1.073 | 0.76 |
| SC+LogMag | 0.409 | 0.474 | 0.86 |
| LSD | 13.7 | 10.05 | **1.36** |

LSD's optimizer found a loss *below* the value at the true parameters: on those
targets the ground truth is not the minimum, so "LSD did badly" there is partly
a statement about the targets. MSS and SC+LogMag are three quarters floor. The
compressed and perceptual losses take the mismatch hardest, which is the same
asymmetry `docs/DATASETS.md` measures for the encoder -- so a cross-loss
comparison run on numpy targets is confounded in exactly the direction the paper
is arguing about.

`ladder_1restart` has `gt_loss` **exactly 0.0 for all six arms**, so its
cross-loss comparison carries no floor at all. That makes it the trustworthy one
-- and it also forecloses an obvious objection: the ladder shows log losing to
linear with no target/synthesis disagreement anywhere in the picture, so the
result cannot be attributed to one.

The cost is that a zero floor is a matched-model result, the "inverse crime" the
generator names. It is the right control for comparing losses and the wrong
setting for claiming an absolute accuracy, so quote the ladder for the former
and say plainly which targets it used.

## `generators/make_dataset.py`

The encoder datasets. Its `--render-path` is the flag that separates the two
generations of them:

- `direct` -- the historical path, builds plate14 straight from the CSV. Leaves
  `T0` quantised by its *range* rather than its value, a ~6e-5 quantum on a
  range of (0.01, 1000), which is ~1e-4 on the mode frequencies. Invisible to a
  linear loss and **19.8% of saturation to log(x + 1e-7)**.
- `training` -- renders through the float32 `z` the encoder emits, so targets
  and training synthesis agree bit-for-bit.

`--fixed-mode-grid` is the second axis. Without it `n_modes` follows the batch
maximum, so an IR renders differently depending on which batch it lands in:
6.1% of saturation for log against ~0 for linear.

| set | grid pinned | `gt_loss` observed |
|---|---|---|
| `train-100000-0.25s`, `val-1000-0.25s` | no | **1.2490e-05** |
| `*-v3` | (107, 403) | 0.0 |
| `train-p99`, `val-p99` | (86, 282) | 0.0 |

The 250k linear run (`l1_stft_tgtnorm`) is on the first row; the 120k sweep and
the eps ladder are on the last. That is why their numbers are not on the same
footing, and why `diag_gt_floor` has to read `0.0000e+00` on the SHUFFLED row
before a sweep is attributable at all.

Audio is not shipped. `datasets/` carries the parameter CSVs; the commands in
`docs/DATASETS.md` regenerate the audio, and every flag in them is load-bearing.
