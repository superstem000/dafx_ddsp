"""Does the bundle actually stand on its own?

Not a file-count check. The question is whether someone with only paper/ can
recompute the numbers the paper quotes -- which means ground truth has to be
present, loss and chain names have to resolve against the bundled source, the
datasets have to cover the IRs that were fitted, and nothing can have been
silently overwritten on the way in.

Two failure classes are distinguished, because they are not equally serious:

  FUNDAMENTAL  a number cannot be recomputed or a run cannot be reproduced --
               missing ground truth, an unresolvable loss name, a dataset whose
               parameters are absent
  LIMITATION   something cannot be *drawn* but nothing is unrecoverable -- no
               per-IR predictions, no convergence traces

    python scripts/check_paper_bundle.py --bundle paper
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as st
from pathlib import Path

FUND, LIM, OK = [], [], []


def note(bucket, msg):
    bucket.append(msg)


def head(t):
    print(f"\n=== {t}")


def csv_cols(p):
    with open(p) as f:
        return next(csv.reader(f))


def geo(v):
    return math.exp(sum(math.log(max(x, 1e-30)) for x in v) / len(v))


def check_cmaes(b: Path):
    head("CMA-ES: ground truth beside estimates, headline recomputable")
    for tag, rel, col in (
        ("01 full", "plate/01_cmaes_full/results/standard_sweep/l1_stft", "nmse_refined"),
        ("02 ladder", "plate/02_cmaes_ladder_1restart/results/ladder_1restart/l1_stft", "nmse_refined"),
    ):
        s1, s2 = b / rel / "stage1/summary.csv", b / rel / "stage2/mu_refined_summary.csv"
        if not s1.exists() or not s2.exists():
            note(FUND, f"{tag}: stage1 or stage2 summary missing"); print(f"  {tag:<12} MISSING"); continue
        c1, c2 = csv_cols(s1), csv_cols(s2)
        gt1 = [x for x in c1 if x.startswith("gt_")]
        gt2 = [x for x in c2 if x.startswith("gt_")]
        if len(gt1) < 7 or len(gt2) < 7:
            note(FUND, f"{tag}: fewer than 7 ground-truth columns")
        if "gt_loss" not in c1:
            note(FUND, f"{tag}: no gt_loss, the target/synthesis floor is unrecorded")
        rows = list(csv.DictReader(s2.open()))
        v = [float(r[col]) for r in rows if r[col].strip()]
        print(f"  {tag:<12} n={len(v):>3}  gt cols {len(gt1)}/{len(gt2)}  "
              f"geomean={geo(v):.3e}  median={st.median(v):.3e}")
        note(OK, f"{tag}: headline recomputes from the bundle (geomean {geo(v):.3e})")


def check_losses(b: Path):
    head("every loss name used resolves in the bundled registry")
    lp = b / "code/plate_src/loss/losses.py"
    if not lp.exists():
        note(FUND, "losses.py absent: no loss in any result can be resolved"); return
    src = lp.read_text()
    used = set()
    for p in glob.glob(str(b / "plate/*/results/**/stage1/summary.csv"), recursive=True):
        r = list(csv.DictReader(open(p)))
        if r and "loss_name" in r[0]:
            used.add(r[0]["loss_name"])
    bad = [l for l in used if l not in src]
    print(f"  {len(used)} distinct loss names; unresolved: {bad or 'none'}")
    if bad:
        note(FUND, f"loss names not in the bundled registry: {bad}")
    else:
        note(OK, f"all {len(used)} loss names resolve")


def check_diffmoog(b: Path):
    head("diffmoog: configs resolve, and best vs final are both available")
    ns = {}
    pp = b / "code/diffmoog_src/model/loss/spectral_loss_presets.py"
    cp = b / "code/diffmoog_src/synth/synth_chains.py"
    if not pp.exists() or not cp.exists():
        note(FUND, "diffmoog presets or chains absent"); return
    exec(compile(pp.read_text(), "p", "exec"), ns)
    presets, chains = ns["loss_presets"], cp.read_text()
    bad = []
    cfgs = sorted(glob.glob(str(b / "diffmoog/*/scripts/q_*.yaml")))
    cfgs = [c for c in cfgs if "config_dump" not in c]
    for f in cfgs:
        t = open(f).read()
        ch = re.search(r"chain:\s*(\S+)", t)
        lp = re.search(r"loss_preset:\s*(\S+)", t)
        if not (ch and ch.group(1) in chains):
            bad.append(f"{os.path.basename(f)} chain")
        if not (lp and lp.group(1) in presets):
            bad.append(f"{os.path.basename(f)} preset")
    print(f"  {len(cfgs)} q_* configs; unresolved: {bad or 'none'}")
    note(FUND if bad else OK,
         f"diffmoog configs unresolved: {bad}" if bad else f"all {len(cfgs)} q_* configs resolve")

    fm = list(b.glob("diffmoog/*/results/diffmoog/final_metrics.csv"))
    if not fm:
        note(FUND, "diffmoog final_metrics.csv absent"); return
    cols = csv_cols(fm[0])
    has_min = any(c.startswith("min_") for c in cols)
    has_step = any(c.startswith("minstep_") for c in cols)
    print(f"  final_metrics.csv: final_ ={any(c.startswith('final_') for c in cols)}  "
          f"min_ ={has_min}  minstep_ ={has_step}")
    if not (has_min and has_step):
        note(FUND, "final_metrics.csv lacks min_/minstep_: the paper quotes best-epoch "
                   "values and only final-epoch ones are present, which differ ~2x and "
                   "rank the arms differently. Re-run tools/extract_scalars.py")
    else:
        note(OK, "best and final both present, with the step of the best")


def check_datasets(b: Path):
    head("datasets: parameters present for every fitted IR, generators shipped")
    for name, need in (("random-IR-100-1.0s", 50), ("random-IR-200-0.2s", 200)):
        got = len(glob.glob(str(b / f"datasets/{name}/random_IR_params_*.csv")))
        print(f"  {name:<22} need>={need:<4} have={got}")
        note(OK if got >= need else FUND,
             f"{name}: {got} parameter CSVs for {need} fitted IRs")
    for f in ("datasets/train_100000_params.csv.gz", "datasets/val_1000_params.csv",
              "datasets/DATASETS.md", "datasets/GENERATORS.md",
              "datasets/generators/ModalPlate/DatasetGen.py",
              "datasets/generators/gen_torch_targets_200.py",
              "datasets/generators/make_dataset.py",
              "analysis/compare_methods.py", "analysis/nmse_per_ir.csv"):
        ok = (b / f).exists()
        print(f"  {f:<52} {'OK' if ok else 'MISSING'}")
        if not ok:
            note(FUND, f"{f} absent")


def check_collisions(b: Path):
    head("nothing was overwritten on the way in")
    for d in sorted(b.glob("*/*/scripts")):
        files = [p for p in d.iterdir() if p.is_file()]
        cd = [p for p in files if "config_dump" in p.name]
        if cd:
            runs = {p.name.split("_config_dump")[0] for p in cd}
            print(f"  {d.relative_to(b)}: {len(cd)} config_dump files across {len(runs)} runs")
            if len(runs) * 2 != len(cd):
                note(FUND, f"{d}: config_dump files do not pair 2-per-run; possible overwrite")


def check_ddsp(b: Path):
    head("DDSP: reference constants present (limitation: no per-IR predictions)")
    hs = sorted(b.glob("plate/04_ddsp_eps_ladder/results/**/history.json"))
    if not hs:
        note(FUND, "no DDSP history.json in the bundle"); return
    missing = []
    for h in hs:
        d = json.load(h.open())
        for k in ("gt_loss", "saturation", "const_nmse_6d"):
            if k not in d:
                missing.append(f"{h.parent.name}:{k}")
    print(f"  {len(hs)} runs; missing constants: {missing or 'none'}")
    note(FUND if missing else OK,
         f"DDSP missing constants: {missing}" if missing else
         f"all {len(hs)} DDSP runs carry gt_loss, saturation and the constant-predictor floor")
    note(LIM, "DDSP per-IR predictions are not in the bundle (checkpoints excluded); "
              "nmse_per_ir.csv covers l1_stft_tgtnorm only, so the ECDF redraws "
              "but per-arm distributions do not")
    note(LIM, "no convergence traces for CMA-ES or gradient descent -- final values only")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bundle", type=Path, default=Path("paper"))
    a = p.parse_args()
    if not a.bundle.exists():
        print(f"no bundle at {a.bundle}; run scripts/make_paper_bundle.py first")
        raise SystemExit(1)

    check_cmaes(a.bundle)
    check_losses(a.bundle)
    check_diffmoog(a.bundle)
    check_datasets(a.bundle)
    check_collisions(a.bundle)
    check_ddsp(a.bundle)

    print("\n" + "=" * 70)
    print(f"FUNDAMENTAL ({len(FUND)}) -- a number cannot be recomputed or a run reproduced")
    for m in FUND:
        print(f"  ! {m}")
    if not FUND:
        print("  none")
    print(f"\nLIMITATION ({len(LIM)}) -- cannot be drawn, but nothing is unrecoverable")
    for m in LIM:
        print(f"  - {m}")
    print(f"\nVERIFIED ({len(OK)})")
    for m in OK:
        print(f"  + {m}")
    raise SystemExit(1 if FUND else 0)


if __name__ == "__main__":
    main()
