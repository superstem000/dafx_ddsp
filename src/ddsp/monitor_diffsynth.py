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
    python -m src.ddsp.monitor_diffsynth --root results/diffsynth --tail 10
"""

from __future__ import annotations

import argparse
import glob
import os
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
    if not glob.glob(os.path.join(tb, "**", "events.out.tfevents.*"), recursive=True):
        return None
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
    return {
        "series": series,
        "steps": steps,
        "elapsed": (ev[-1].wall_time - ev[0].wall_time) if len(ev) > 1 else 0.0,
        "n_ev": len(ev),
        "max_epochs": max_epochs,
    }


def at(d: dict, step: int, key: str) -> float:
    return d["series"].get(key, {}).get(step, float("nan"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--table", action="store_true",
                   help="Table 1 shape at best and final epoch, rather than progress")
    p.add_argument("--tail", type=int, default=8, help="Trajectory rows to show")
    p.add_argument("--steps-per-epoch", type=int, default=250,
                   help="16000 train / batch 64; only used to label epochs")
    args = p.parse_args()

    runs = {}
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(d):
            continue
        r = load(d)
        if r:
            runs[Path(d).name] = r
    if not runs:
        print(f"no runs with event files under {args.root}")
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
                       ("param", "val_id/param (the paper's Param column)")):
        names = [n for n in runs if key in runs[n]["series"]]
        if not names:
            continue
        print(f"\n=== {label}")
        allsteps = sorted({s for n in names for s in runs[n]["series"][key]})[-args.tail:]
        w = max(10, max(len(n) for n in names) + 2)
        print(f"{'epoch':>7}" + "".join(f"{n:>{w}}" for n in names))
        for s in allsteps:
            cells = []
            for n in names:
                v = runs[n]["series"][key].get(s)
                cells.append(f"{v:>{w}.4f}" if v is not None else f"{'-':>{w}}")
            print(f"{s // args.steps_per_epoch:>7}" + "".join(cells))


if __name__ == "__main__":
    main()
