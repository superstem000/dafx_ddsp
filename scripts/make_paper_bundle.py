"""Assemble paper/ from the runs that actually produced the paper's numbers.

Copies, never moves: the working tree stays exactly as it is, and this can be
re-run whenever a sweep finishes to refresh the bundle. Anything already in
paper/ for a given entry is replaced, so a stale partial result cannot survive
a rerun and be mistaken for a final one.

The manifest below is the point of the file. results/ holds many more runs than
the paper uses -- earlier parameterizations, abandoned variants, smoke tests --
and which one is "the" result is a judgement that belongs written down rather
than remembered. Each entry records where it came from and what produced it, and
the generated README carries that through to whoever reads the bundle.

Sources that do not exist are reported and skipped rather than failing the run,
so the bundle can be built while a sweep is still going.

    python scripts/make_paper_bundle.py
    python scripts/make_paper_bundle.py --only plate_ddsp_eps_ladder
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------
# Entries. `sources` are copied into <slug>/results, `scripts` into
# <slug>/scripts. `note` is prose that lands in the README next to the entry --
# state what the run is and what is known to be wrong with it.
# --------------------------------------------------------------------------
MANIFEST = [
    dict(
        section="plate",
        slug="01_cmaes_full",
        title="CMA-ES, 20 restarts, all losses (the 'CMA-ES full' row)",
        sources=["results/standard_sweep"],
        scripts=["scripts/standard_sweep/run_standard_sweep.sh"],
        note=(
            "src/analysis/compare_methods.py line 228 reads "
            "results/standard_sweep/l1_stft for the row it labels 'CMA-ES "
            "full L1_STFT', so this is that run and cmaes_norm_es* is not. "
            "50 IRs from data/random-IR-100-1.0s at float32, 14 losses, two "
            "stages each -- sweep_run.log records DSET_ROOT, N_SAMPLES=50 and "
            "DTYPE per run along with wall clock, so the invocation is "
            "recovered rather than assumed. Everything else comes from "
            "fit_7param_norm_es defaults: n_trials 400, budget 25000, sigma0 "
            "0.6, tolfun 1e-5, lhs_seed 42, popsize 30-60, early stop at "
            "0.01. All 14 losses are kept, not just L1_STFT: the cross-loss "
            "comparison at a full restart budget is the counterpart to the "
            "one-restart ladder."
        ),
    ),
    dict(
        section="plate",
        slug="02_cmaes_ladder_1restart",
        title="CMA-ES compression ladder, one restart per IR",
        sources=["results/ladder_1restart"],
        scripts=["scripts/cmaes_norm_es/l1stft.sh"],
        note=(
            "200 IRs, linear / c2 / log / pow plus mss and smoothmss, each in "
            "two stages, --n_trials 1 against the standard sweep's 20. One "
            "restart per IR is the point: it removes the restart budget as a "
            "confound between losses. compare_methods scores these at stage 2 "
            "using stage2/mu_refined_summary.csv's refined_* columns, so the "
            "figure's medians (l1_stft 1.00e-3) do not match stage1 "
            "summary.csv's nmse (1.59e-2) -- read the right stage."
        ),
    ),
    dict(
        section="plate",
        slug="03_gradient_descent_l1stft",
        title="Per-IR gradient descent, L1-STFT",
        sources=["results/gd/graddescent_l1_stft", "results/gd/raw7_budget"],
        scripts=["scripts/gd.sh"],
        note=(
            "KNOWN INCOMPLETE. gd.sh defaults to NUM=10, so this covers ten IRs "
            "against the CMA-ES runs' hundred, and the smoke/ and torch50/ "
            "directories in results/gd are throwaway. This is the run that "
            "needs redoing at the CMA-ES sample count before it can carry a "
            "claim -- and it is also the run that would separate 'the loss is "
            "bad for gradients' from 'our encoder is bad', since it is "
            "gradient descent without an encoder."
        ),
    ),
    dict(
        section="plate",
        slug="04_ddsp_eps_ladder",
        title="DDSP encoder, log(x + eps) ladder",
        sources=[
            "results/ddsp/eps_ladder",
            "results/ddsp/sweep120k_L1_STFT",
            "results/ddsp/sweep120k_L1_STFT_c2",
            "results/ddsp/sweep120k_L1_STFT_log",
            "results/ddsp/sweep120k_MSS",
            "results/ddsp/l1_stft_tgtnorm",
        ],
        scripts=["scripts/eps_ladder.sh", "scripts/lr_probe.sh"],
        note=(
            "eps_ladder is the single-variable sweep; sweep120k_* is the "
            "earlier four-loss sweep it supersedes, kept because the ladder "
            "runs at 40k steps and sweep120k at 120k. l1_stft_tgtnorm is the "
            "250k linear run and is where the converged linear number comes "
            "from -- the ladder's linear arm reaches ~0.015 at 40k against "
            "that run's 0.0002 at 250k. Note the ladder uses BatchNorm and "
            "sweep120k GroupNorm, so their numbers are not interchangeable."
        ),
    ),
    dict(
        section="plate",
        slug="05_ddsp_lr_probe",
        title="DDSP encoder, learning-rate grid",
        sources=[
            "results/ddsp/lr_probe",
            "results/ddsp/lr_L1_STFT_c2_1e-4",
            "results/ddsp/lr_L1_STFT_c2_3e-5",
            "results/ddsp/lr_L1_STFT_c2_1e-5",
            "results/ddsp/lr_L1_STFT_log_1e-4",
            "results/ddsp/lr_L1_STFT_log_3e-5",
            "results/ddsp/lr_L1_STFT_log_1e-5",
            "results/ddsp/lr_MSS_1e-4",
            "results/ddsp/lr_MSS_3e-5",
            "results/ddsp/lr_MSS_1e-5",
        ],
        scripts=["scripts/lr_probe.sh"],
        note=(
            "The stated grid answering 'you did not tune the learning rate'. "
            "The lr_* directories are the older cells under pre-ladder names "
            "(c2 is eps1, log is eps1e7); lr_probe holds the rest, including "
            "the L1_STFT control the older probes never covered."
        ),
    ),
    dict(
        section="diffmoog",
        slug="06_ddsp_eps_ladder",
        title="DiffMoog encoder, log(x + eps) ladder on the fixed-pitch task",
        sources=["results/diffmoog"],
        scripts=[],
        extra_globs=[
            "external/diffmoog/configs/loss_study/q_*.yaml",
            # The resolved config and the git commit + argv each run actually
            # used. configs/ is the input; config_dump/ is what happened.
            "external/diffmoog/experiments/current/q_*/config_dump/*",
        ],
        note=(
            "The q_* runs are the ladder: ploss, linear at two rates, the eps "
            "ladder, and mss. Scalars only -- experiments/ is 21 GB of "
            "Lightning checkpoints and none of it is needed for a figure. "
            "SAW_FIXED_FILTER is amp plus filter_freq with pitch pinned, which "
            "is why this task can be learned at all."
        ),
    ),
]

# Copied once, shared by every entry, rather than traced per result: an import
# trace would be more precise and would also be one more thing that can be
# subtly wrong.
ANALYSIS = [
    # THE generator for the paper's table and ECDF figure. Reads
    # standard_sweep/l1_stft as "CMA-ES full", every ladder_1restart/* as
    # "CMA-ES 1rst", and the encoder from --ddsp-ckpt. Scores CMA-ES at stage 2
    # via the refined_* columns when mu_refined_summary.csv exists, which is
    # why a stage-1 median read straight off summary.csv does not reproduce the
    # figure's numbers.
    "src/analysis/compare_methods.py",
    # Produces the cross-loss comparison tables; lives under results/ rather
    # than src/, so nothing else in the manifest would pick it up.
    "results/cmaes/compare_all_losses.py",
    # Per-IR NMSE for the CMA-ES full run, every 1-restart ladder arm and the
    # 250k DDSP run -- the data behind the ECDF figure, and the only per-IR
    # record that survives for the full CMA-ES run.
    "docs/figures/nmse_per_ir.csv",
    "docs/figures/nmse_ecdf.pdf",
]

# The original numpy plate and its dataset generator. random-IR-100-1.0s and
# random-IR-200-0.2s -- the CMA-ES and gradient-descent sets -- came from
# DatasetGen.py, not from src/data/make_dataset.py, and nothing else in the
# manifest reaches it.
GENERATORS = [
    ("ModalPlate", "datasets/generators/ModalPlate"),
    # Re-render an existing dataset's IRs through the torch plate, so target
    # and model share a code path. Referenced by train_encoder's docstring as
    # what was done for the fitting datasets.
    ("gen_torch_targets.py", "datasets/generators/gen_torch_targets.py"),
    ("gen_torch_targets_200.py", "datasets/generators/gen_torch_targets_200.py"),
    # The diagnostic that found the float32 target/synthesis disagreement in
    # the first place.
    ("confirm_f32_gt.py", "datasets/generators/confirm_f32_gt.py"),
    ("src/data/make_dataset.py", "datasets/generators/make_dataset.py"),
]

CODE = [
    ("src", "code/plate_src"),
    ("external/diffmoog/src", "code/diffmoog_src"),
    ("external/diffmoog/configs/loss_study", "code/diffmoog_configs"),
    ("external/diffmoog/tools", "code/diffmoog_tools"),
]

DATASETS = [
    "data/train_100000_params.csv.gz",
    "data/val_1000_params.csv",
    "docs/DATASETS.md",
]

# Per-IR parameters for the sets the CMA-ES and gradient-descent runs used.
# Those directories are 80 MB and 75 MB of rendered audio; the parameters are
# what make them regenerable, and they are a few hundred KB.
DATASET_GLOBS = [
    ("data/random-IR-100-1.0s/random_IR_params_*.csv", "random-IR-100-1.0s"),
    ("data/random-IR-100-1.0s/generation_summary.txt", "random-IR-100-1.0s"),
    ("random-IR-200-0.2s/random_IR_params_*.csv", "random-IR-200-0.2s"),
    ("random-IR-200-0.2s/generation_summary.txt", "random-IR-200-0.2s"),
]


SKIP = ["*.pt", "*.ckpt", "*.db", "__pycache__", "events.out.tfevents.*"]


GENERATORS_MD = """# Which script made which dataset, and which ones have a floor

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
"""


def copy(src: Path, dst: Path, skip=None) -> tuple[int, int]:
    """Copy a file or tree. Returns (files, bytes)."""
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        # Checkpoints are the reason experiments/ is 21 GB and results/ddsp is
        # hundreds of MB. Nothing in the paper reads them.
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*(skip or SKIP)))
        files = [p for p in dst.rglob("*") if p.is_file()]
        return len(files), sum(p.stat().st_size for p in files)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return 1, dst.stat().st_size


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{x:.1f}{unit}"
        x /= 1024.0
    return f"{x:.1f}GB"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=Path("paper"))
    p.add_argument("--only", nargs="+", default=None, help="Slugs to rebuild")
    p.add_argument("--with-figures", action="store_true",
                   help="Include per-IR diagnostic PNGs. They are 1212 files "
                        "and ~96 MB in the 1-restart ladder alone, they "
                        "regenerate from the CSVs beside them, and no figure "
                        "in the paper is one of them.")
    args = p.parse_args()
    skip = list(SKIP) + ([] if args.with_figures else ["*_diagnostic.png"])

    root = Path(".").resolve()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or "unknown"

    lines = [
        "# Paper bundle",
        "",
        f"Generated by `scripts/make_paper_bundle.py` at commit `{commit}`.",
        "",
        "Every path under `results/` here is a copy; the working tree is",
        "unchanged. Checkpoints (`*.pt`, `*.ckpt`), Optuna databases and",
        "TensorBoard event files are excluded -- they are large and nothing in",
        "the paper reads them. For DiffMoog the scalars were extracted first",
        "with `external/diffmoog/tools/extract_scalars.py`.",
        "",
        "Datasets are not copied as audio. `datasets/` holds the parameter CSVs",
        "and the regeneration commands; `docs/DATASETS.md` explains why the",
        "flags in them are load-bearing rather than defaults.",
        "",
    ]

    missing = []
    for e in MANIFEST:
        if args.only and e["slug"] not in args.only:
            continue
        base = args.out / e["section"] / e["slug"]
        nf = nb = 0
        got, absent = [], []
        for s in e["sources"]:
            src = root / s
            if not src.exists():
                absent.append(s)
                continue
            f, b = copy(src, base / "results" / Path(s).name, skip)
            nf, nb = nf + f, nb + b
            got.append(s)
        for s in e.get("scripts", []):
            src = root / s
            if src.exists():
                f, b = copy(src, base / "scripts" / Path(s).name)
                nf, nb = nf + f, nb + b
                got.append(s)
            else:
                absent.append(s)
        for pat in e.get("extra_globs", []):
            hits = sorted(root.glob(pat))
            for h in hits:
                f, b = copy(h, base / "scripts" / h.name)
                nf, nb = nf + f, nb + b
            got.append(f"{pat} ({len(hits)} files)")

        missing.extend(absent)
        print(f"{e['slug']:<32} {nf:>5} files  {human(nb):>9}"
              + (f"   MISSING: {', '.join(absent)}" if absent else ""))

        lines += [
            f"## `{e['section']}/{e['slug']}` -- {e['title']}",
            "",
            e["note"],
            "",
            "| copied from |",
            "|---|",
            *[f"| `{g}` |" for g in got],
        ]
        if absent:
            lines += ["", "**Not found at build time:** "
                      + ", ".join(f"`{a}`" for a in absent)]
        lines += [""]

    # Shared code and datasets
    for src, dst in CODE:
        s = root / src
        if s.exists():
            f, b = copy(s, args.out / dst)
            print(f"{dst:<32} {f:>5} files  {human(b):>9}")
        else:
            missing.append(src)
    for d in DATASETS:
        s = root / d
        if s.exists():
            copy(s, args.out / "datasets" / Path(d).name)
        else:
            missing.append(d)
    for pat, sub in DATASET_GLOBS:
        hits = sorted(root.glob(pat))
        for h in hits:
            copy(h, args.out / "datasets" / sub / h.name)
        if not hits:
            missing.append(pat)
        else:
            print(f"{'datasets/' + sub:<32} {len(hits):>5} files")
    for src, dst in GENERATORS:
        sp = root / src
        if sp.exists():
            f, b = copy(sp, args.out / dst)
            print(f"{dst:<32} {f:>5} files  {human(b):>9}")
        else:
            missing.append(src)
    (args.out / "datasets").mkdir(parents=True, exist_ok=True)
    (args.out / "datasets" / "GENERATORS.md").write_text(GENERATORS_MD)
    for a in ANALYSIS:
        s = root / a
        if s.exists():
            copy(s, args.out / "analysis" / Path(a).name)
        else:
            missing.append(a)

    lines += [
        "## `code/`",
        "",
        "Copied wholesale rather than traced per result. An import trace would",
        "be more precise and would also be one more thing that can be quietly",
        "wrong; the trees here are small.",
        "",
        "| tree | from |",
        "|---|---|",
        *[f"| `{dst}` | `{src}` |" for src, dst in CODE],
        "",
    ]

    (args.out / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}/README.md")
    if missing:
        print("\nmissing at build time (entries were skipped, not failed):")
        for m in sorted(set(missing)):
            print(f"  {m}")


if __name__ == "__main__":
    main()
