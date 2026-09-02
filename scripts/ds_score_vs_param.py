"""How a clip's score moves with share2 AND with interval accuracy, jointly.

    python scripts/ds_eval_folder.py ... --csv-clips f5o_clips.csv
    python scripts/ds_osc_usage.py   ... --csv       f5o_osc.csv
    python scripts/ds_score_vs_param.py --scores f5o_clips.csv --params f5o_osc.csv

THE QUESTION. On the Moog subset the arm with the QUIETEST second oscillator
wins the metric while having poor interval accuracy, and the arm that gets the
interval right most often scores worst:

    arm          share2   p_1.5   mfcc/sat
    magx           0.69     54%     1.4489
    hybridx        0.57     30%     1.4292
    logx           0.21     38%     0.9432

Three group medians ordered the same way cannot separate the two explanations.
Every predicted quantity co-varies across arms -- share2, mix1, mix2 all move
together -- so the ordering could be about oscillator level, about interval
accuracy, or about neither.

WHAT THIS DOES. Bins clips by share2 and by |cents| from the true ratio and
reports the median score in each cell, so the two can be read against each
other rather than one at a time:

  MARGINALS, per arm, so the trend is measured where the quantities vary for
  unrelated reasons rather than across arms where they co-vary.

  THE JOINT GRID, pooled over arms, because that is the only way to hold one
  constant while varying the other -- 50 clips per arm cannot fill a 3x3 grid,
  150 can. Pooling reintroduces the cross-arm confound for any single cell,
  but reading DOWN a column holds interval accuracy roughly fixed and varies
  share2, which is exactly the comparison the arm medians cannot make.

THE HEDGING HYPOTHESIS it tests: harmor's osc 2 is a mathematically exact saw
at exactly 1.5*f0, the Moog's is an analog VCO through a driven 4-pole ladder.
Adding that oscillator at full level puts strong partials where the target has
partials of the WRONG shape, so a model that commits to the structure is
exposed to a timbre error a quiet model avoids. If that is right, score should
worsen with share2 even at fixed interval accuracy -- the columns of the grid
should rise downward.

If instead the score improves with interval accuracy at fixed share2 -- the
rows falling to the left -- then the metric is rewarding the structure after
all and the arm ordering is about something else entirely.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys

import numpy as np


def spearman(x, y):
    """Rank correlation, plus a two-sided p from the normal approximation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 4:
        return float("nan"), float("nan")

    def rank(v):
        o = v.argsort()
        r = np.empty(n, float)
        r[o] = np.arange(n, dtype=float)
        # Average ties -- a collapsed arm predicts nearly one value, and
        # arbitrary distinct ranks there would invent a correlation.
        _u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        for i in np.flatnonzero(cnt > 1):
            m = inv == i
            r[m] = r[m].mean()
        return r

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan"), float("nan")
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    return rho, math.erfc(abs(t) / math.sqrt(2))


def terciles(v):
    """Edges splitting v into three roughly equal groups."""
    return [-np.inf, float(np.percentile(v, 100 / 3)),
            float(np.percentile(v, 200 / 3)), np.inf]


def which(v, edges):
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, 2)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scores", required=True, metavar="CSV",
                   help="ds_eval_folder --csv-clips output.")
    p.add_argument("--params", required=True, metavar="CSV",
                   help="ds_osc_usage --csv output, same arms and files.")
    p.add_argument("--metric", default="mfcc")
    p.add_argument("--target-mult", type=float, default=1.5, metavar="R",
                   help="The interval the material actually has, so the "
                        "cents column measures accuracy rather than value.")
    args = p.parse_args()

    sc = {}
    for r in csv.DictReader(open(args.scores)):
        if r["metric"] != args.metric:
            continue
        den = float(r["den"])
        if den > 0:
            sc[(r["arm"], r["file"])] = float(r["num"]) / den
    pr = {(r["arm"], r["file"]): r
          for r in csv.DictReader(open(args.params))}

    keys = sorted(set(sc) & set(pr))
    if not keys:
        raise SystemExit(
            "no (arm, file) pairs in common -- both CSVs must come from the "
            "same --dirs, --match, --n and --seed, and --metric must name a "
            "metric the scores file has.")

    arms = sorted({a for a, _ in keys})
    score = np.array([sc[k] for k in keys])
    share2 = np.array([float(pr[k]["share2"]) for k in keys])
    cents = np.array([abs(1200.0 * math.log2(max(float(pr[k]["mult"]), 1e-9)
                                             / args.target_mult))
                      for k in keys])
    arm_of = np.array([k[0] for k in keys])

    print(f"metric {args.metric}/saturation, target ratio {args.target_mult:g}"
          f", {len(keys)} clips over {len(arms)} arm(s)\n")

    # --- marginals, per arm -------------------------------------------------
    print("=== MEDIAN SCORE BY TERCILE, within each arm")
    print(f"{'arm':<24}{'n':>4}" + "".join(f"{c:>11}" for c in
          ("share2 lo", "mid", "hi", "|cents| lo", "mid", "hi"))
          + f"{'rho s2':>9}{'rho ct':>9}")
    for a in arms + (["POOLED"] if len(arms) > 1 else []):
        m = np.ones(len(keys), bool) if a == "POOLED" else (arm_of == a)
        if m.sum() < 6:
            print(f"{a:<24}{m.sum():>4}  too few clips")
            continue
        cells = []
        for v in (share2[m], cents[m]):
            b = which(v, terciles(v))
            cells += [np.median(score[m][b == i]) if (b == i).any()
                      else float("nan") for i in range(3)]
        r1, p1 = spearman(share2[m], score[m])
        r2, p2 = spearman(cents[m], score[m])
        st = lambda q: "***" if q < .001 else "**" if q < .01 else \
            "*" if q < .05 else ""
        print(f"{a:<24}{m.sum():>4}" + "".join(f"{c:>11.3f}" for c in cells)
              + f"{r1:>6.2f}{st(p1):<3}{r2:>6.2f}{st(p2):<3}")

    # --- the joint grid -----------------------------------------------------
    es, ec = terciles(share2), terciles(cents)
    bs, bc = which(share2, es), which(cents, ec)
    print(f"\n=== MEDIAN SCORE, share2 (rows) x |cents| from "
          f"{args.target_mult:g} (columns), pooled over arms")
    print(f"{'':<22}" + "".join(
        f"{lab:>16}" for lab in
        (f"cents <{ec[1]:.0f}", f"{ec[1]:.0f}-{ec[2]:.0f}", f">{ec[2]:.0f}")))
    for i in range(3):
        lab = (f"share2 <{es[1]:.2f}" if i == 0 else
               f"{es[1]:.2f}-{es[2]:.2f}" if i == 1 else f">{es[2]:.2f}")
        row = []
        for j in range(3):
            m = (bs == i) & (bc == j)
            row.append(f"{np.median(score[m]):>11.3f} n{m.sum():<3}"
                       if m.sum() else f"{'-':>16}")
        print(f"{lab:<22}" + "".join(row))

    print("\n  READ DOWN a column: interval accuracy roughly fixed, osc 2 level\n"
          "  rising. If the score worsens downward, the metric is penalising a\n"
          "  LOUD second oscillator regardless of whether it is in the right\n"
          "  place -- the hedging story.\n"
          "  READ ACROSS a row: level roughly fixed, interval accuracy getting\n"
          "  worse to the right. If the score worsens rightward, the metric IS\n"
          "  rewarding the correct interval and something else explains the arm\n"
          "  ordering.\n"
          "  rho is Spearman against the score, so POSITIVE means larger value\n"
          "  goes with a worse score. Stars are p < .05/.01/.001, normal\n"
          "  approximation, uncorrected for testing two columns.\n"
          "  Pooling arms is what makes the grid readable at all -- 50 clips\n"
          "  per arm cannot fill nine cells -- but any single cell mixes arms,\n"
          "  so the per-arm marginals above are the attributable measurement\n"
          "  and the grid is the picture.")


if __name__ == "__main__":
    main()
