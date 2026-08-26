"""Rank the class sweep: did magx come first, and if not, is it closing?

    python scripts/ds_sweep_report.py
    python scripts/ds_sweep_report.py --window 10 --dir results/class_sweep

Reads the CSVs ds_class_sweep.sh leaves behind, so it runs mid-sweep on
whatever has finished. Every column is val_ood/mfccdb -- Hann, Slaney, dB at
top_db 80 -- and nothing else. val_id/param, lsd, loud and the torchaudio mfcc
are all in the CSVs and none of them is reported here.

TWO CRITERIA, in the order they matter.

A. RANK. Is magx the lowest of the three at the final epoch. That is the whole
   question; everything else is a consolation prize.

B. TREND, and only over the last --window epochs. The first epochs after the
   branch are a regime change -- param_w reaches 0 and the training set becomes
   real audio at the same step -- and the arms move by whole units there. On
   keyboard, magx-logx went 3.87 at epoch 200 to 1.46 at 239: reading a slope
   through that measures the discontinuity, not the trend. The default window
   is the last 10 epochs of a 30-epoch phase.

   The trend reported is of the GAP, magx minus the better of the other two,
   fitted by least squares over that window. Negative means magx is closing.
   `to_zero` divides the current gap by that slope: how many more epochs at
   this rate before magx is level. It is an extrapolation of a decaying process
   -- ExponentialLR keeps cutting the step size -- so it is a comparison
   between classes, not a prediction. Treat 10x the window as "not close".

WHAT WOULD MAKE A CLASS WORTH A FULL RUN. magx first outright, or a gap under
a few tenths closing steadily. A gap of 1.5 closing at 0.01/epoch is the
keyboard result again and is not worth fourteen hours.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

METRIC = "val_ood/mfccdb"
ARMS = ("magx", "hyb", "log")


def series(path: str, metric: str) -> list[tuple[int, float]]:
    """(epoch, value) for one metric, from a ds_export_scalars CSV.

    The header is ["step", "epoch"] + sorted(tags), and Lightning logs a scalar
    called `epoch` too -- so "epoch" appears twice and the second one is
    Lightning's own record. Prefer it: the first is step // steps_per_epoch,
    which is right only if the caller passed the right divisor, and a class run
    does ~101 steps per epoch where the default is 250.
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return []
    head = rows[0]
    if metric not in head:
        return []
    mi = head.index(metric)
    ei = [i for i, h in enumerate(head) if h == "epoch"]
    ei = ei[-1] if len(ei) > 1 else (ei[0] if ei else 1)
    out = []
    for r in rows[1:]:
        try:
            v = r[mi]
            if v == "":
                continue
            out.append((int(round(float(r[ei]))), float(v)))
        except (ValueError, IndexError):
            continue
    return sorted(out)


def fit(pts: list[tuple[int, float]]) -> float:
    """Least-squares slope per epoch, or nan with fewer than two points."""
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den == 0:
        return float("nan")
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", default="results/class_sweep")
    p.add_argument("--window", type=int, default=10,
                   help="Epochs at the END of the run used for the trend. The "
                        "epochs right after the branch are a regime change, "
                        "not a trend -- see the module docstring.")
    p.add_argument("--metric", default=METRIC,
                   help="Left open for checking, but the sweep exists to "
                        "answer one question and it is about this column.")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*__*.csv")))
    if not files:
        raise SystemExit(f"no *__*.csv under {args.dir}")
    classes: dict[str, dict[str, str]] = {}
    for f in files:
        base = os.path.basename(f)[: -len(".csv")]
        cls, _, arm = base.rpartition("__")
        classes.setdefault(cls, {})[arm] = f

    rows = []
    for cls in sorted(classes):
        got = classes[cls]
        if not all(a in got for a in ARMS):
            print(f"{cls:<24} incomplete: have {sorted(got)}")
            continue
        s = {a: series(got[a], args.metric) for a in ARMS}
        if not all(s[a] for a in ARMS):
            print(f"{cls:<24} no {args.metric} logged "
                  f"(a run predating the metric, or it died before validating)")
            continue
        last = min(max(e for e, _ in s[a]) for a in ARMS)
        fin = {a: dict(s[a])[last] for a in ARMS}
        best_other = min(fin["hyb"], fin["log"])
        gap = fin["magx"] - best_other
        # The gap's own trajectory, so a class where every arm is improving
        # together is not mistaken for one where magx is catching up.
        lo = last - args.window + 1
        common = sorted(set(dict(s["magx"])) & set(dict(s["hyb"]))
                        & set(dict(s["log"])))
        d = {a: dict(s[a]) for a in ARMS}
        gpts = [(e, d["magx"][e] - min(d["hyb"][e], d["log"][e]))
                for e in common if lo <= e <= last]
        slope = fit(gpts)
        tz = (-gap / slope) if (slope == slope and slope < 0 and gap > 0) \
            else float("inf") if gap > 0 else 0.0
        rows.append(dict(cls=cls, last=last, gap=gap, slope=slope, tz=tz,
                         n=len(gpts), **{f"f_{a}": fin[a] for a in ARMS}))

    if not rows:
        raise SystemExit("nothing complete yet")

    print(f"\n=== {args.metric} at the final epoch   (lower is better; "
          f"trend over the last {args.window} epochs)")
    print(f"{'class':<22}{'ep':>5}{'magx':>9}{'hyb':>9}{'log':>9}"
          f"{'gap':>9}{'slope/ep':>11}{'to_zero':>9}  verdict")
    for r in sorted(rows, key=lambda r: r["gap"]):
        if r["gap"] < 0:
            verdict = "MAGX FIRST"
        elif r["tz"] != float("inf") and r["tz"] <= 10 * args.window:
            verdict = f"closing"
        elif r["slope"] == r["slope"] and r["slope"] < 0:
            verdict = "closing slowly"
        else:
            verdict = "no"
        tz = "-" if r["tz"] == float("inf") else f"{r['tz']:.0f}"
        print(f"{r['cls']:<22}{r['last']:>5}{r['f_magx']:>9.4f}{r['f_hyb']:>9.4f}"
              f"{r['f_log']:>9.4f}{r['gap']:>9.4f}{r['slope']:>11.4f}{tz:>9}"
              f"  {verdict}")

    won = [r for r in rows if r["gap"] < 0]
    print(f"\n  {len(won)} of {len(rows)} classes put magx first: "
          f"{', '.join(r['cls'] for r in won) if won else 'none'}")
    print(f"\n  gap is magx minus the BETTER of hybridx and logx, so it is the")
    print(f"  distance to first place, not to a chosen rival. slope is fitted")
    print(f"  over the last {args.window} epochs only: the epochs right after")
    print(f"  the branch are a regime change -- param_w hits 0 and the data")
    print(f"  becomes real audio at the same step -- and on keyboard magx-logx")
    print(f"  fell from 3.87 to 1.46 across epochs 200-239 before flattening at")
    print(f"  1.59. A slope through that measures the discontinuity.")
    print(f"  to_zero extrapolates a DECAYING process: ExponentialLR keeps")
    print(f"  cutting the step size, so the real number is larger, often much.")
    print(f"  Use it to rank classes against each other, never as a forecast.")
    print(f"  30 epochs is 3030 steps against the joint run's 50,000. Nothing")
    print(f"  here has converged and no row is a result -- this says which")
    print(f"  class, if any, is worth a full run, and 'none' is a valid answer.")


if __name__ == "__main__":
    main()
