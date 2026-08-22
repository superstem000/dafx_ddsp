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

The diffsynth half is checked the same way, and one of its checks is worth
naming: every arm resumes from the same pretrain, and if Lightning restored RNG
state before the datamodule's setup() the resumed run would draw a different
train/valid split, making pretrain's validation files the resume's training
files. That is leakage across the phase boundary, biased towards the paper's own
result, and invisible in every metric. The split manifests carry a hash of each
split's membership, so comparing them across runs settles it offline -- the
bundle verifies the claim about itself rather than deferring to a script that
has to be re-run on a GPU.

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


# The three the diffsynth tables quote. val_id/param is the paper's own headline
# and exists only in-domain, since NSynth has no true theta.
DS_METRICS = ("val_id/param", "val_id/mfcc", "val_id/lsd")

# log(x + eps) has to keep its knee at the same signal level when the domain
# changes, and log(m) = 0.5*log(p) makes that eps_mag = sqrt(eps_pow). Getting
# it wrong puts the log four decades off -- down where the plate's ladder
# collapses -- while changing nothing that would show up as an error.
DS_EPS = {2: 1e-4, 1: 1e-2}


def _f(pat, text, cast=str):
    m = re.search(pat, text, re.M)
    return cast(m.group(1)) if m else None


def check_diffsynth(b: Path):
    head("diffsynth: numbers present, arms recoverable, one split across every run")
    # 08-11 copy the same tree; 08 is the reproduction section and carries it.
    root = b / "diffsynth/08_reproduction/results/diffsynth"
    if not root.exists():
        note(FUND, "no diffsynth results in the bundle"); print("  MISSING"); return
    runs = sorted(d for d in root.iterdir() if d.is_dir())
    if not runs:
        note(FUND, "diffsynth results directory is empty"); print("  EMPTY"); return

    no_scalars, no_metrics, eps_bad, arms, epochs = [], [], [], {}, {}
    for d in runs:
        sc = d / "scalars.csv"
        if not sc.exists():
            no_scalars.append(d.name)
            continue
        with sc.open() as f:
            rows = list(csv.reader(f))
        cols = rows[0]
        miss = [m for m in DS_METRICS if m not in cols]
        if miss:
            no_metrics.append(f"{d.name}:{','.join(miss)}")
        # Two columns are named "epoch" -- Lightning logs one as a scalar tag on
        # top of the one ds_export_scalars derives -- so index, not DictReader.
        epochs[d.name] = int(rows[-1][1]) if len(rows) > 1 else -1

        hp = d / "tb_logs" / "hparams.yaml"
        if hp.exists():
            t = hp.read_text()
            pw = _f(r"^\s*power:\s*(\d+)", t, int)
            ev = _f(r"^\s*log_eps_v:\s*(\S+)", t, float)
            arms[d.name] = (_f(r"^\s*mag_w:\s*(\S+)", t, float),
                            _f(r"^\s*log_mag_w:\s*(\S+)", t, float), pw, ev)
            if pw in DS_EPS and ev is not None and not math.isclose(ev, DS_EPS[pw], rel_tol=1e-6):
                eps_bad.append(f"{d.name}: power={pw} with log_eps_v={ev:g}, "
                               f"expected {DS_EPS[pw]:g}")

    print(f"  {len(runs)} runs; without scalars.csv: {no_scalars or 'none'}")
    if no_scalars:
        note(FUND, f"{len(no_scalars)} diffsynth runs have no scalars.csv, so none of "
                   f"their numbers can be recomputed -- run scripts/ds_export_scalars.py "
                   f"and rebuild the bundle: {no_scalars}")
    print(f"  runs missing a quoted metric: {no_metrics or 'none'}")
    if no_metrics:
        note(FUND, f"diffsynth runs missing quoted metrics: {no_metrics}")
    elif not no_scalars:
        note(OK, f"all {len(runs)} diffsynth runs carry {', '.join(DS_METRICS)}")

    # power/log_eps_v were declared in the model config partway through, so only
    # the arms run after that carry them in their own hparams. The rest are the
    # published power-domain setting and inherit it from the bundled default --
    # which has to actually say so, or their domain is only inferable from the
    # run name. Check the default rather than assume it.
    dflt = b / "code/diffsynth_configs/model/default.yaml"
    dt = dflt.read_text() if dflt.exists() else ""
    dpw, dev = _f(r"^\s*power:\s*(\d+)", dt, int), _f(r"^\s*log_eps_v:\s*(\S+)", dt, float)
    explicit = sum(1 for v in arms.values() if v[2] is not None)
    print(f"  power/log_eps_v: {explicit}/{len(runs)} arms record it explicitly; "
          f"the rest inherit the config default (power={dpw}, log_eps_v={dev})")
    print(f"  eps/domain mismatches: {eps_bad or 'none'}")
    if eps_bad:
        note(FUND, f"log_eps_v does not match the loss domain: {eps_bad}")
    if dpw is None or dev is None:
        note(FUND, "the bundled model config declares no power/log_eps_v, so the "
                   f"{len(runs) - explicit} arms that do not record them have no "
                   "recoverable loss domain at all")
    elif not math.isclose(dev, DS_EPS.get(dpw, -1), rel_tol=1e-6):
        note(FUND, f"the config default pairs power={dpw} with log_eps_v={dev:g}, "
                   f"expected {DS_EPS.get(dpw)}")
    elif arms:
        note(OK, f"every arm's loss domain resolves -- {explicit} from its own hparams, "
                 f"{len(runs) - explicit} from the bundled default -- and eps tracks the "
                 f"domain as sqrt(eps_pow) in every case")

    # The leakage check, computed rather than asserted. Every arm resumes from
    # pre_base, and Lightning restores RNG state from a checkpoint; if it did so
    # before the datamodule's setup(), a resumed run would draw a different
    # split and pretrain's validation files would become the resume's training
    # files. One hash across every run is what rules that out.
    hashes, unnamed = {}, []
    for d in runs:
        sm = d / "split_manifest.json"
        if not sm.exists():
            continue
        try:
            rec = json.loads(sm.read_text())
        except Exception:
            note(FUND, f"{d.name}: split_manifest.json is unreadable"); continue
        for k in ("id_valid", "ood_valid"):
            if k in rec:
                hashes.setdefault(k, {}).setdefault(rec[k]["sha1"], []).append(d.name)
                if "files" not in rec[k]:
                    unnamed.append(f"{d.name}:{k}")
    if not hashes:
        note(FUND, "no split manifests: whether the resumed arms share the pretrain's "
                   "split cannot be checked from the bundle")
        print("  split manifests: MISSING")
    for k, hs in sorted(hashes.items()):
        n = sum(len(v) for v in hs.values())
        print(f"  {k}: {len(hs)} distinct membership hash(es) across {n} runs"
              + (f"  {next(iter(hs))[:12]}" if len(hs) == 1 else ""))
        if len(hs) == 1:
            note(OK, f"{k}: one split across all {n} runs -- no leakage across the "
                     f"pretrain/resume boundary")
        else:
            note(FUND, f"{k}: {len(hs)} DIFFERENT splits across runs, so some arm "
                       f"validated on another's training files: "
                       + "; ".join(f"{h[:8]}={v}" for h, v in hs.items()))
    if unnamed:
        note(LIM, f"{len(unnamed)} split manifests record only a hash for a valid split, "
                  f"not the membership, so an offline evaluation can detect the wrong "
                  f"split but cannot reproduce the right one: {unnamed[:4]}"
                  + (" ..." if len(unnamed) > 4 else ""))

    # The per-group Param table is normalised against the constant predictor;
    # without this the raw numbers survive but the normalised ones do not.
    pb = root / "param_baseline.json"
    if not pb.exists():
        print("  param_baseline.json: MISSING")
        note(FUND, "param_baseline.json absent: the per-group Param table is normalised "
                   "against the constant predictor and cannot be rebuilt -- run "
                   "scripts/ds_param_baseline.py")
    else:
        g = json.loads(pb.read_text()).get("baseline_l1", {})
        print(f"  param_baseline.json: {len(g)} parameter groups")
        note(OK if len(g) >= 6 else FUND,
             f"constant-predictor baseline covers {len(g)} groups"
             if len(g) >= 6 else f"param_baseline.json has only {len(g)} groups, expected 6")

    rd = b / "analysis/monitor_diffsynth.py"
    print(f"  analysis/monitor_diffsynth.py: {'OK' if rd.exists() else 'MISSING'}")
    if not rd.exists():
        note(FUND, "monitor_diffsynth.py absent: scalars.csv is present but the reader "
                   "that turns it into the paper's tables is not")

    # Not every arm ran to 400, and comparing one that stopped at 318 against one
    # that reached 399 is the easiest mistake to make with this tree. pre_* are
    # excluded: they are the shared pretrain and end at 50 or 200 by design.
    short = {k: v for k, v in sorted(epochs.items())
             if not k.startswith("pre") and v < 399}
    if short:
        print(f"  did not reach epoch 399: "
              + ", ".join(f"{k}@{v}" for k, v in short.items()))
        note(LIM, "arms stopped at different epochs, so any cross-arm number must be "
                  "read at a matched epoch rather than at each run's last: "
                  + ", ".join(f"{k}@{v}" for k, v in short.items()))


# Every floor ds_masking reports, and the two ways each is reported: against the
# full threshold and against the maskers alone. The pair is the point -- a
# single column would let the absolute threshold carry a claim about masking.
MASK_FLOORS = (80, 70, 60, 40, 20)
MASK_COLS = ("frac_energy_masked", "frac_energy_smasked", "frac_bins_masked",
             "frac_total_energy")


def check_masking(b: Path):
    head("masking: the evaluation floor is derived rather than chosen")
    root = b / "diffsynth/12_masking_metric/results"
    csvp = root / "masking/masking.csv"
    if not csvp.exists():
        note(FUND, "no masking/masking.csv in the bundle, so the psychoacoustic "
                   "justification for the evaluation floor cannot be recomputed "
                   "-- run scripts/ds_masking.py and rebuild")
        print("  MISSING"); return

    cols = csv_cols(csvp)
    want = [f"{m}_{F:g}" for F in MASK_FLOORS for m in MASK_COLS]
    miss = [c for c in want if c not in cols]
    with csvp.open() as f:
        rows = list(csv.DictReader(f))
    print(f"  masking.csv: {len(rows)} clips, {len(cols)} columns; "
          f"missing: {miss or 'none'}")
    if miss:
        note(FUND, f"masking.csv is missing columns the text quotes: {miss}")
    elif len(rows) < 500:
        note(LIM, f"masking.csv covers {len(rows)} clips. The 2000-clip run "
                  f"was made under the criterion listening falsified and has "
                  f"NOT been repeated under the corrected one, so section 12 "
                  f"quotes n={len(rows)} and says so. Family rows are n=2-4 at "
                  f"this size and carry nothing -- vocal read 0.748 on two "
                  f"clips and 0.9316 on 67")
    else:
        # The smask column is what separates "masked by the signal" from
        # "below the absolute threshold". If it were ever dropped the paper's
        # sentence would still typeset and would no longer be supported.
        note(OK, f"masking thresholds recomputable over {len(rows)} clips, with "
                 f"masker-only fractions reported beside the full-threshold "
                 f"ones at every floor")

    # The cache must NOT be here: 484 MB, and regenerable. Its absence is the
    # correct state, so what is checked is that the regeneration path exists.
    big = [p for p in root.rglob("*") if p.is_file()
           and p.stat().st_size > 50 * 1024 * 1024]
    if big:
        note(LIM, "large regenerable files copied into the bundle: "
                  + ", ".join(f"{p.name} ({p.stat().st_size >> 20} MB)"
                              for p in big))
    if not (b / "diffsynth/12_masking_metric/scripts/ds_masking.py").exists():
        note(FUND, "ds_masking.py is not in the bundle, so neither masking.csv "
                   "nor the threshold cache behind the `mask` metric column "
                   "can be regenerated")

    # The eval tables. These are the only record of the mask and saturation
    # columns -- no scalars.csv carries them, because they are recomputed from
    # checkpoints rather than logged during training.
    ev = root / "eval"
    logs = sorted(ev.glob("*.log")) if ev.exists() else []
    txt = "\n".join(p.read_text(errors="replace") for p in logs)
    print(f"  eval tables: {[p.name for p in logs] or 'none'}")
    if not logs:
        note(FUND, "no results/eval tables in the bundle; the mask and "
                   "saturation columns are recomputed from checkpoints and "
                   "logged nowhere else, so nothing quoted from them is "
                   "recoverable without a rerun")
    else:
        for tag, msg in (
            ("mask", "the `mask` column"),
            ("SATURATION", "the unrelated-pairs saturation row, without which a "
                           "gap cannot be read as a fraction of the metric's range"),
            ("db70post", "db70post, the pre/post-mel bridge -- without it the "
                         "move to a pre-mel clamp silently restates every dB "
                         "number in sections 08-11"),
        ):
            if tag not in txt:
                note(FUND, f"the eval tables do not carry {msg}")
        if all(t in txt for t in ("mask", "SATURATION", "db70post")):
            note(OK, "mask, saturation and the pre/post-mel bridge column are "
                     "all present in the eval tables")


def check_plate_gamma(b: Path):
    head("plate gamma ladder: shared base, and the arms that resume it")
    root = b / "plate/13_gamma_ladder/results"
    if not root.exists():
        note(LIM, "no plate gamma ladder in the bundle; the cross-system claim "
                  "rests on diffsynth alone")
        print("  MISSING"); return

    got, short_ = {}, []
    for d in sorted(root.rglob("history.json")):
        try:
            h = json.load(d.open())
        except Exception:
            continue
        rows = [r for r in h["history"] if "val_nmse_6d" in r]
        if not rows:
            continue
        name = f"{d.parent.parent.name}/{d.parent.name}"
        got[name] = (rows[-1]["step"], rows[-1]["val_nmse_6d"] / h["const_nmse_6d"])
        # gamma_pre is the 5000-step base by design; everything else is 40000.
        if "gamma_pre/" not in name and rows[-1]["step"] < 40000:
            short_.append(f"{name}@{rows[-1]['step']}")
    # Prefixed by directory: gamma_pre and gamma_ppre both hold an arm called
    # L1_STFT and gamma_ppre/gamma_raw both hold L1_STFT_g1, so the bare arm
    # name prints the same label twice and the pair that IS the experiment --
    # pretrained against from-scratch -- reads as a duplicate.
    print(f"  {len(got)} arms: " + ", ".join(
        f"{k.split('/')[0].replace('gamma_', '')}:{k.split('/')[-1]}={v[1]:.2f}"
        for k, v in sorted(got.items())))
    if not got:
        note(FUND, "plate gamma arms carry no evaluated rows"); return
    if short_:
        note(LIM, "plate gamma arms stopped before 40000, so a cross-arm number "
                  "must be read at a matched step: " + ", ".join(short_))

    # The base is what makes the comparison single-variable. Without it every
    # arm ran its own parameter-only phase divided by its own loss_scale, which
    # differs by ~2000x between L1_STFT and L1_STFT_g1.
    if not any(k.startswith("gamma_pre/") for k in got):
        note(FUND, "the shared parameter-only base is absent, so nothing "
                   "establishes that the arms started the crossfade from one "
                   "point -- and each arm's hold is divided by its own "
                   "loss_scale, which spans ~2000x across this ladder")
    if not any(k.startswith("gamma_raw/") for k in got):
        note(LIM, "no raw arm, so the retraction that the dead zone does NOT "
                  "reproduce here rests on the pretrained arms alone")
    if all(k in got for k in ("gamma_pre/L1_STFT", "gamma_raw/L1_STFT_g1")):
        note(OK, "plate gamma ladder complete: shared base, pretrained arms and "
                 "the from-scratch control that ruled the dead zone out")

    if not (b / "plate/13_gamma_ladder/scripts/jobs_plate_gamma.txt").exists():
        note(FUND, "jobs_plate_gamma.txt is not in the bundle; it carries the "
                   "exponent-domain correction, the shared-base rationale and "
                   "the ladder's own result table, none of which is anywhere else")


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
    check_diffsynth(a.bundle)
    check_masking(a.bundle)
    check_plate_gamma(a.bundle)

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
