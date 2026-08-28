"""Which parameters do different objectives WANT? Read from a saved probe run.

    python -m src.emt.objectives --dir probe20k
    python -m src.emt.objectives --dir probe20k --k 64

No rendering and no encoder. src/emt/probe.py already saves every draw's score
against every IR plus all twelve parameter columns, so this is a reduction of
that file rather than new work.

THE QUESTION IT ANSWERS, which is not "what are the bounds" but "what should be
SEARCHED". emt10 fixes fp_x, fp_y, op_x, op_y at the medians of the top-32 draws
by mfcc, on the argument that the objective is flat in them. But the probe scores
two things, and the draws that win them are disjoint sets: the mfcc winners are
{1038, 235, 299, ...} and the 62 Hz winners are {1222, 1482, 1402, ...}, with no
overlap at all. If those two sets also disagree about WHERE a parameter should
sit, then that parameter is doing real work for one objective and pinning it at
the other's answer is a choice with a cost -- which is exactly the case that
matters for fp/op, since mode (1,1)'s amplitude is
sin(fp_x*pi)sin(fp_y*pi)sin(op_x*pi)sin(op_y*pi) and nothing else the encoder
searches can turn the fundamental down.

HOW TO READ IT. Per parameter, the median and 90% interval of the top-K draws
under each objective, side by side.

  intervals OVERLAP    the objectives agree; the parameter can be fixed at the
                       shared value and neither objective pays for it.
  intervals DISJOINT   they want different values. Fixing it serves one
                       objective at the other's expense, and the AGREE column
                       says which one the current pin actually serves.
  both intervals wide  neither objective constrains it. Fixing it is free, and
                       the VALUE is a convention rather than a measurement --
                       which is what src/emt/scatter.py showed for fp/op.

The last case is the trap: "flat under mfcc" and "flat under everything" are
different claims, and only the first was ever measured.

COMPATIBILITY, printed below the table. For the top-K under each objective,
their median score under the OTHER one, against what a random draw gets. It is
the same question JOINT asks in probe.py, without the bass threshold being a
free parameter: if the mfcc winners score near-random on bass, one vector cannot
serve both and the model needs a coordinate that decouples them.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.emt.probe import WIDE                                     # noqa: E402

# Every objective is (key in the npz, lower-is-better, label). gap62 is stored
# as |render - target| in dB, so it is already a distance like mfcc is.
OBJECTIVES = [("mfcc", "mfcc"), ("gap62", "bass 62Hz")]


def top_k(score: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k draws with the lowest MEAN score over the IRs.

    Mean rather than per-IR best, matching probe.py's own THE BOUNDS: a draw
    that happens to nail one recording and miss the rest is not evidence about
    where a parameter belongs.
    """
    m = score.mean(axis=1)
    m = np.where(np.isfinite(m), m, np.inf)
    return np.argsort(m)[:k]


def interval(v: np.ndarray, log: bool):
    """median and 90% interval, in the axis the parameter is sampled on."""
    if log:
        lv = np.log10(np.maximum(v, 1e-300))
        return 10 ** np.median(lv), 10 ** np.percentile(lv, 5), 10 ** np.percentile(lv, 95)
    return np.median(v), np.percentile(v, 5), np.percentile(v, 95)


def fmt(x: float) -> str:
    return f"{x:.4g}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", default="probe20k",
                   help="a directory holding probe_scores.npz")
    p.add_argument("--k", type=int, default=32,
                   help="how many top draws define an objective's preference. "
                        "32 matches probe.py's THE BOUNDS.")
    args = p.parse_args()

    f = Path(args.dir) / "probe_scores.npz"
    if not f.exists():
        raise SystemExit(f"{f} not found -- re-run probe.py with --out {args.dir}")
    d = np.load(f, allow_pickle=True)
    n_draw, n_ir = d["mfcc"].shape
    print(f"{f}: {n_draw} draws x {n_ir} IRs, {len(WIDE)} searched dimensions, "
          f"top-{args.k} per objective\n")

    sets = {}
    for key, label in OBJECTIVES:
        sets[label] = top_k(d[key], args.k)

    labels = [l for _, l in OBJECTIVES]
    ov = len(set(sets[labels[0]]) & set(sets[labels[1]]))
    print(f"=== OVERLAP   {ov} of {args.k} draws are in BOTH objectives' top-{args.k}")
    print("  0 means no single draw is good at both, which is the strongest form")
    print("  of 'these objectives want different plates'.\n")

    print("=== WHAT EACH OBJECTIVE WANTS   median [5%, 95%] of its top-K")
    w = 26
    print(f"  {'param':>14}" + "".join(f"{l:>{w}}" for l in labels) + "   verdict")
    disagree, flat = [], []
    for kk, (lo, hi, lg) in WIDE.items():
        v = d[kk]
        cells, ivs = "", []
        for l in labels:
            med, a, b = interval(v[sets[l]], lg)
            ivs.append((a, b))
            cells += f"{fmt(med) + ' [' + fmt(a) + ',' + fmt(b) + ']':>{w}}"
        # General over any number of objectives: they share a value iff the
        # highest lower bound is below the lowest upper bound.
        overlap = max(a for a, _ in ivs) <= min(b for _, b in ivs)
        # "Wide" on the axis the parameter is sampled on: an interval covering
        # over half the box says the objective does not constrain it at all,
        # which is a different finding from the two objectives disagreeing.
        span = (np.log10(hi) - np.log10(lo)) if lg else (hi - lo)
        wide = all(((np.log10(b) - np.log10(a)) if lg else (b - a)) > 0.5 * span
                   for a, b in ivs)
        if wide:
            verdict = "unconstrained"
            flat.append(kk)
        elif overlap:
            verdict = "agree"
        else:
            verdict = "DISAGREE"
            disagree.append(kk)
        print(f"  {kk:>14}" + cells + f"   {verdict}")

    print("\n  agree          fixable at the shared value; neither objective pays.")
    print("  DISAGREE       the two want different values. Fixing it serves one")
    print("                 at the other's expense -- this is the case that says")
    print("                 a parameter belongs in the SEARCHED set.")
    print("  unconstrained  neither objective pins it. Free to fix, but the value")
    print("                 is a convention, not a measurement.")
    if disagree:
        print(f"\n  DISAGREE: {', '.join(disagree)}")
    if flat:
        print(f"  unconstrained: {', '.join(flat)}")

    print("\n=== COMPATIBILITY   each objective's top-K scored under the other")
    print("  'random' is the median over all draws, i.e. what carrying no")
    print("  information about that objective looks like.")
    print(f"  {'top-K by':>14}" + "".join(f"{'-> ' + l:>18}" for l in labels)
          + f"{'':>4}")
    for l_sel in labels:
        row = f"  {l_sel:>14}"
        for (key, l_sc) in OBJECTIVES:
            s = d[key].mean(axis=1)
            s = np.where(np.isfinite(s), s, np.inf)
            row += f"{np.median(s[sets[l_sel]]):>18.4g}"
        print(row)
    row = f"  {'random':>14}"
    for key, _ in OBJECTIVES:
        s = d[key].mean(axis=1)
        s = np.where(np.isfinite(s), s, np.inf)
        row += f"{np.median(s[np.isfinite(s)]):>18.4g}"
    print(row)
    print("\n  A top-K that scores near 'random' under the other objective is one")
    print("  vector serving one goal and abandoning the other. That is not a")
    print("  tuning problem -- it says the model has no setting that does both,")
    print("  and the fix is a coordinate that decouples them rather than a")
    print("  better search over the ones it has.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
