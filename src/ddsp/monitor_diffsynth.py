"""Watch the diffsynth arms, and report them in Table 1's shape.

Two views of the same event files.

The default is progress: where each run has got to, how fast, and the current
value of the three quantities the paper reports. The loss weights are shown
alongside, because on the switch schedule a run's behaviour is not interpretable
without knowing which phase it is in -- param_w 10 / sw_w 0 up to epoch 50, then
a 150-epoch ramp, then spectral only.

--table gives the Table 1 shape: Param, LSD and Multi, in-domain and
out-of-domain, at both the best epoch and the final one. Both, because
ModelCheckpoint monitors val_ood/lsd and the paper does not say how its own
table was selected -- and on the plate the two differed by ~2x and ranked
differently, which is exactly the ambiguity worth not repeating.

    python -m src.ddsp.monitor_diffsynth
    python -m src.ddsp.monitor_diffsynth --table
    python -m src.ddsp.monitor_diffsynth --root results/diffsynth --rows 16

Trajectories are shown at milestones across the whole run, not as the last few
epochs. Adjacent epochs differ by noise -- val_ood/lsd moves ~0.1 between
neighbours while a run's total improvement may be ~2 -- so reading recent rows
tells you almost nothing. Each milestone is a local mean, epochs 50 and 200 are
always included as the phase boundaries, and a verdict line says whether each
metric is still moving, judged against its own recent scatter.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

# The metrics the paper reports, under the names diffsynth logs them by.
# 'spec' is the multi-scale spectral loss ("Multi"); validation calls
# train_losses with no weights, so it is unweighted there and comparable across
# arms whatever their sw_w happens to be.
PARAM = "val_id/param"
TAGS = {
    "param": PARAM,
    "id_lsd": "val_id/lsd",
    "id_multi": "val_id/spec",
    "ood_lsd": "val_ood/lsd",
    "ood_multi": "val_ood/spec",
}
SELECT = "val_ood/lsd"   # what ModelCheckpoint monitors


def load(run_dir: str) -> dict | None:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise SystemExit(
            "tensorboard is required to read the event files:\n"
            "  pip install tensorboard\n"
            "(it is in external/diffsynth/requirements-modern.txt)"
        )
    tb = os.path.join(run_dir, "tb_logs")
    files = sorted(glob.glob(os.path.join(tb, "**", "events.out.tfevents.*"),
                             recursive=True))
    if not files:
        return None

    # Parsing is slow because the event files are mostly not scalars: AudioLogger
    # writes 8 audio clips and a 15x30in figure for train, val_id and val_ood
    # every epoch, so a 400-epoch run leaves gigabytes that EventAccumulator has
    # to walk through to reach the numbers. Cache the extracted series against
    # the files' size and mtime -- a finished run is then read once, and only
    # runs still being written to are re-parsed.
    fp = [[f, os.path.getsize(f), int(os.path.getmtime(f))] for f in files]
    cache_path = os.path.join(run_dir, ".monitor_cache.json")
    try:
        cached = json.load(open(cache_path))
        if cached.get("fp") == fp:
            c = cached["data"]
            c["series"] = {k: {int(st): v for st, v in d.items()}
                           for k, d in c["series"].items()}
            c["steps"] = [int(x) for x in c["steps"]]
            return c
    except Exception:
        pass

    ea = EventAccumulator(tb, size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags().get("scalars", []))
    series = {k: {e.step: e.value for e in ea.Scalars(t)}
              for k, t in TAGS.items() if t in have}
    for k, t in (("param_w", "lw/param_w"), ("sw_w", "lw/sw_w"),
                 ("train", "train/total")):
        if t in have:
            series[k] = {e.step: e.value for e in ea.Scalars(t)}
    if not series:
        return None
    # Wall clock from the selection metric if present, else anything.
    ref = SELECT if SELECT in have else sorted(have)[0]
    ev = ea.Scalars(ref)
    steps = sorted({s for d in series.values() for s in d})

    # max_epochs comes from hydra's own dump of the resolved config, so the
    # progress column is right for both the 200-epoch pretrains and the
    # 400-epoch ploss and resumes without being told which is which.
    max_epochs = None
    cfg = os.path.join(run_dir, ".hydra", "config.yaml")
    if os.path.exists(cfg):
        try:
            import yaml
            max_epochs = yaml.safe_load(open(cfg))["trainer"]["max_epochs"]
        except Exception:
            pass
    out = {
        "series": series,
        "steps": steps,
        "elapsed": (ev[-1].wall_time - ev[0].wall_time) if len(ev) > 1 else 0.0,
        "n_ev": len(ev),
        "max_epochs": max_epochs,
    }
    try:
        with open(cache_path, "w") as fh:
            json.dump({"fp": fp, "data": out}, fh)
    except Exception:
        pass    # a read-only or racing run directory is not worth failing over
    return out



def pairs(r: dict, key: str, spe: int):
    """(epoch, value) sorted, for one metric of one run."""
    return sorted((st // spe, v) for st, v in r["series"].get(key, {}).items())


def around(pts, ep, half=2):
    """Mean of the values within +-half epochs of ep, or nan."""
    vals = [v for e, v in pts if abs(e - ep) <= half]
    return sum(vals) / len(vals) if vals else float("nan")


def stdev(vals):
    if len(vals) < 2:
        return float("nan")
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def trend(pts):
    """Compare the last tenth of the run against the tenth before it.

    Epoch-to-epoch values here are dominated by noise -- val_ood/lsd moves by
    ~0.1 between adjacent epochs while the whole run's improvement may be ~2 --
    so a single latest number says almost nothing. Two block means, with the
    verdict gated on the recent scatter, answers "is this still moving" without
    anyone eyeballing rows.
    """
    if len(pts) < 10:
        return None
    n = max(3, len(pts) // 10)
    last = [v for _, v in pts[-n:]]
    prev = [v for _, v in pts[-2 * n:-n]]
    if not prev:
        return None
    lm, pm = sum(last) / len(last), sum(prev) / len(prev)
    noise = stdev(last)
    delta = lm - pm

    # Compare the change to the standard error of the DIFFERENCE OF MEANS, not
    # to the raw scatter. Averaging n points shrinks their error by sqrt(n), so
    # gating on the standard deviation calls block-mean changes significant that
    # are nothing of the kind -- it flagged a +0.60 move in val_id/lsd against a
    # scatter of 0.99 as WORSENING when the standard error of that difference is
    # 0.63. Two standard errors is the usual bar.
    sl, sp = stdev(last), stdev(prev)
    se = ((sl * sl) / len(last) + (sp * sp) / len(prev)) ** 0.5 if sl == sl and sp == sp else float("nan")
    if not (se == se) or abs(delta) < 2 * se:
        verdict = "flat"
    elif delta < 0:
        verdict = "improving"
    else:
        verdict = "WORSENING"
    best_e, best_v = min(pts, key=lambda p: p[1])
    return {"n": n, "last": lm, "prev": pm, "delta": delta, "noise": noise,
            "se": se, "verdict": verdict, "best_v": best_v, "best_e": best_e}


def at(d: dict, step: int, key: str) -> float:
    return d["series"].get(key, {}).get(step, float("nan"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--table", action="store_true",
                   help="Table 1 shape at best and final epoch, rather than progress")
    p.add_argument("--rows", type=int, default=10,
                   help="Roughly how many milestone rows to show across the run")
    p.add_argument("--spread", type=int, default=None, metavar="EPOCH",
                   help="Report every run's value at this epoch and their range. "
                        "Before epoch 50 all pretrains and ploss are the same "
                        "configuration -- deterministic: False means CUDA "
                        "reductions are not order-stable, so they diverge, and "
                        "that divergence IS the run-to-run error bar for every "
                        "later comparison.")
    p.add_argument("--only", default=None, metavar="REGEX",
                   help="Keep only runs whose name matches. The phases are named "
                        "for it: '^pre_|^ploss' is the branch phase (epochs "
                        "50-200/400), '^synth_|^real_' is the resume phase "
                        "(200-400). Mixing them in one table is usually a "
                        "mistake -- they cover different epoch ranges, so a "
                        "milestone row means different things in each column.")
    p.add_argument("--steps-per-epoch", type=int, default=250,
                   help="16000 train / batch 64; only used to label epochs")
    args = p.parse_args()

    runs = {}
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(d):
            continue
        name = Path(d).name
        if args.only and not re.search(args.only, name):
            continue
        r = load(d)
        if r:
            runs[name] = r
    if not runs:
        print(f"no runs with event files under {args.root}")
        return

    if args.spread is not None:
        e0 = args.spread
        print(f"Run-to-run spread at epoch {e0}.")
        if e0 <= 50:
            print("At or below epoch 50 every pretrain and ploss run the same")
            print("loss (param_w 10, sw_w 0), so this is pure seed/CUDA noise")
            print("and is the error bar later differences must clear.\n")
        print(f"{'metric':<12}{'n':>4}{'min':>11}{'max':>11}{'range':>11}"
              f"{'mean':>11}{'sd':>10}")
        for key in ("param", "id_lsd", "id_multi", "ood_lsd", "ood_multi"):
            vals = []
            for n in runs:
                v = around(pairs(runs[n], key, args.steps_per_epoch), e0)
                if v == v:
                    vals.append(v)
            if len(vals) < 2:
                continue
            m = sum(vals) / len(vals)
            print(f"{key:<12}{len(vals):>4}{min(vals):>11.4f}{max(vals):>11.4f}"
                  f"{max(vals) - min(vals):>11.4f}{m:>11.4f}{stdev(vals):>10.4f}")
        return

    if args.table:
        print("Param / LSD / Multi, as in Masuda & Saito Table 1.")
        print("'best' is the epoch minimising val_ood/lsd, which is what")
        print("ModelCheckpoint saves; 'final' is the last epoch. The paper does")
        print("not say which selection it used, so both are reported.\n")
        print(f"{'run':<18}{'sel':>6}{'epoch':>7}{'Param':>9}{'ID LSD':>9}"
              f"{'ID Multi':>10}{'OOD LSD':>9}{'OOD Multi':>11}")
        for name, r in runs.items():
            sel = r["series"].get("ood_lsd", {})
            if not sel:
                print(f"{name:<18}  (no val_ood/lsd logged yet)")
                continue
            best_step = min(sel, key=lambda s: sel[s])
            for tag, step in (("best", best_step), ("final", max(r["steps"]))):
                ep = step // args.steps_per_epoch
                print(f"{name:<18}{tag:>6}{ep:>7}"
                      f"{at(r, step, 'param'):>9.4f}{at(r, step, 'id_lsd'):>9.3f}"
                      f"{at(r, step, 'id_multi'):>10.4f}{at(r, step, 'ood_lsd'):>9.3f}"
                      f"{at(r, step, 'ood_multi'):>11.4f}")
        return

    print(f"{'run':<18}{'epoch':>7}{'of':>6}{'ep/h':>7}{'eta_h':>7}"
          f"{'param_w':>9}{'sw_w':>7}{'train':>10}"
          f"{'ID LSD':>9}{'OOD LSD':>9}{'Param':>9}")
    for name, r in runs.items():
        last = max(r["steps"])
        ep = last // args.steps_per_epoch
        eph = (r["n_ev"] / (r["elapsed"] / 3600)) if r["elapsed"] > 0 else float("nan")
        tot = r["max_epochs"]
        eta = ((tot - ep) / eph) if (tot and eph and eph == eph and eph > 0) else float("nan")
        print(f"{name:<18}{ep:>7}{tot if tot else '?':>6}{eph:>7.1f}{eta:>7.1f}"
              f"{at(r, last, 'param_w'):>9.2f}{at(r, last, 'sw_w'):>7.2f}"
              f"{at(r, last, 'train'):>10.4f}"
              f"{at(r, last, 'id_lsd'):>9.3f}{at(r, last, 'ood_lsd'):>9.3f}"
              f"{at(r, last, 'param'):>9.4f}")

    for key, label in (("ood_lsd", "val_ood/lsd  (what the checkpoint selects on)"),
                       ("id_lsd", "val_id/lsd"),
                       ("param", "val_id/param  (the paper's Param column)"),
                       # train/total is the OPTIMISED objective, so read it only
                       # down a column, never across. Across arms it is a
                       # different quantity per arm -- log(x+1e-4) and |x| live
                       # on unrelated scales, and zeroing one weight halves the
                       # divisor. Even within one arm it is not comparable
                       # across the ramp, since param_w and sw_w are still
                       # moving between epochs 50 and 200; only from 200 on,
                       # where the weights settle at 0 and 1, is it one thing.
                       ("train", "train/total  (per-arm scale -- compare down a "
                                 "column, not across; mixed weights until ep200)")):
        names = [n for n in runs if key in runs[n]["series"]]
        if not names:
            continue
        P = {n: pairs(runs[n], key, args.steps_per_epoch) for n in names}
        w = max(11, max(len(n) for n in names) + 2)

        # Milestones rather than the last N epochs. Adjacent epochs differ by
        # noise; what carries information is the shape across the whole run, and
        # the two phase boundaries -- 50, where the spectral loss starts being
        # introduced, and 200, where the ramp completes and the resume begins.
        top = max(e for n in names for e, _ in P[n])
        marks = {1, 50, 200, top}
        marks |= {round(top * i / args.rows) for i in range(1, args.rows)}
        marks = sorted(e for e in marks if 0 < e <= top)

        print(f"\n=== {label}   (mean over +-2 epochs; * marks a phase boundary)")
        print(f"{'epoch':>7} " + "".join(f"{n:>{w}}" for n in names))
        for e in marks:
            cells = []
            for n in names:
                v = around(P[n], e)
                cells.append(f"{v:>{w}.4f}" if v == v else f"{'-':>{w}}")
            flag = "*" if e in (50, 200) else " "
            print(f"{e:>7}{flag}" + "".join(cells))

        print(f"\n{'':7} {'best (epoch)':>18}{'last':>10}{'prev':>10}"
              f"{'change':>10}{'2*se':>9}  verdict")
        for n in names:
            t = trend(P[n])
            if t is None:
                print(f"{n:<18} too few points yet")
                continue
            print(f"{n:<18}{t['best_v']:>9.4f} ({t['best_e']:>3}){t['last']:>10.4f}"
                  f"{t['prev']:>10.4f}{t['delta']:>+10.4f}{2 * t['se']:>9.4f}"
                  f"  {t['verdict']}")


if __name__ == "__main__":
    main()
