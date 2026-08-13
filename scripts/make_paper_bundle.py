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
        slug="01_cmaes_full_l1stft",
        title="CMA-ES, L1-STFT, full restart budget",
        sources=[
            "results/cmaes_norm_es_lessdigit/l1_stft",
            "results/cmaes_norm_es_lessdigit/l1_stft_tolfun",
            "results/cmaes_norm_es/l1_stft_tolfun",
        ],
        scripts=[
            "scripts/cmaes_norm_es/l1stft.sh",
            "scripts/cmaes_norm_es/l1stft_tolfun.sh",
        ],
        note=(
            "100 IRs from data/random-IR-100-1.0s, via "
            "src.cmaes.fit_7param_norm_es at sigma0 0.6 with early stop at "
            "loss 0.01. Three directories because three variants exist and "
            "which one the paper quotes is not recorded anywhere: the plain "
            "run, the tolfun-stopped run, and the later _lessdigit "
            "reporting. CONFIRM WHICH, then cut this entry down to one."
        ),
    ),
    dict(
        section="plate",
        slug="02_cmaes_ladder_1restart",
        title="CMA-ES compression ladder, one restart per IR",
        sources=["results/ladder_1restart"],
        scripts=["scripts/cmaes_norm_es/l1stft.sh"],
        note=(
            "linear / c2 / log / pow, plus mss and smoothmss, each in two "
            "stages. One restart per IR is the point: it removes the restart "
            "budget as a confound between losses."
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
    # Produces the cross-loss comparison tables; lives under results/ rather
    # than src/, so nothing else in the manifest would pick it up.
    "results/cmaes/compare_all_losses.py",
    # Per-IR NMSE for the CMA-ES full run, every 1-restart ladder arm and the
    # 250k DDSP run -- the data behind the ECDF figure, and the only per-IR
    # record that survives for the full CMA-ES run.
    "docs/figures/nmse_per_ir.csv",
    "docs/figures/nmse_ecdf.pdf",
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
