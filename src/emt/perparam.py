"""Per-parameter corr / spread / error, straight out of history.json.

    python -m src.emt.perparam --root results/ddsp/emt8
    python -m src.emt.perparam --root results/ddsp/emt8 --every 250 --metric spread

train_encoder records corr_{k}, spread_{k}, perr_{k} and perr_{k}_p90 for every
searched parameter at EVERY eval, but monitor_sweep only surfaces the aggregate
val_nmse_6d. So the per-coordinate picture is already on disk at 250-step
resolution and nothing needs re-running to see it.

WHAT EACH COLUMN ANSWERS, and they are not redundant -- train_encoder's own note
is that corr and spread are both invariant to a constant offset, so a biased
coordinate reads 1.00/1.00 on both and neither carries a magnitude:

  corr    is this coordinate tracked at all? Near 0 means the encoder's output
          is uncorrelated with the truth. Near 1 means it moves the right way,
          which says nothing about whether it moves the right AMOUNT.
  spread  predicted sd over true sd. THE COLLAPSE DETECTOR. An encoder ignoring
          its input emits a near-constant, so spread near 0 is collapse and
          spread near 1 is a coordinate with the right dynamic range. Above 1 is
          over-dispersion, a different failure.
  perr    the normalised squared error nmse averages, per coordinate, so
          sqrt(perr)*100 is that parameter's error as a percent of its range.
          This is the one with a magnitude.

WHY THIS MATTERS HERE. emt7's encoders emitted near-constants on real audio --
six of seven parameters identical across fifteen IRs -- and the diagnosis was
that rho, E and h carry an exact degeneracy plus bounds that excluded the
answer. emt8 fixes both. spread_{k} is where that shows IN DOMAIN, per step,
without waiting for the campaign to finish or for real-audio evaluation: a
coordinate whose spread sits near 0 through the handover is one the encoder has
stopped reading its input for.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load(root: Path):
    """{arm: (keys, rows)} for every history.json under root."""
    out = {}
    for h in sorted(root.rglob("history.json")):
        try:
            rows = json.loads(h.read_text())["history"]
        except Exception as e:                                   # noqa: BLE001
            print(f"  skip {h}: {e}")
            continue
        if not rows:
            continue
        keys = [m.group(1) for m in
                (re.fullmatch(r"corr_(.+)", k) for k in rows[-1]) if m]
        if keys:
            out[h.parent.name] = (keys, rows)
    return out


SUFFIX = {"spread": ("spread_", ""), "corr": ("corr_", ""),
          "perr": ("perr_", ""), "perr_p90": ("perr_", "_p90")}
NOTE = {
    "spread": "predicted sd / true sd. ~0 = COLLAPSED, ~1 = right range",
    "corr": "predicted vs true correlation. ~0 = not tracked at all",
    "perr": "median normalised sq error; sqrt(x)*100 = % of range. LOWER WINS",
    "perr_p90": "the 90th-percentile tail of the same. LOWER WINS",
}


def compare(arms, args) -> int:
    """Parameters down, arms across, at one step. Which coordinate is each
    arm actually buying?

    THE QUESTION THIS EXISTS FOR. An arm can win the aggregate parameter error
    while losing the audio metric, and the obvious explanation is that not every
    parameter is worth the same number of decibels: a coordinate that shapes only
    the QUIET part of the spectrum moves perr a lot and moves a
    loudness-weighted audio distance barely at all. On emt8 the candidates split
    cleanly -- loss_F1 and T60_ratio set the high-frequency damping, which is
    where these signals are 20-30 dB down, while T60_DC, T0 and Ly live in the
    loud low end. If a compressed arm is winning, it should be winning THERE.
    """
    steps = [{r["step"] for r in rows} for _, rows in arms.values()]
    common = set.intersection(*steps) if steps else set()
    if args.at:
        step = args.at
    elif common:
        step = max(common)
    else:
        step = max(max(s) for s in steps)
    names = sorted(arms)
    print(f"step {step}   arms {names}")

    for metric in args.metric:
        pre, tail = SUFFIX[metric]
        lower_wins = metric.startswith("perr")
        print(f"\n=== {metric}   {NOTE[metric]}")
        keys = arms[names[0]][0]
        print(f"  {'param':>12}" + "".join(f"{n:>18}" for n in names)
              + ("      best" if lower_wins else ""))
        for k in keys:
            vals = {}
            for n in names:
                row = next((r for r in arms[n][1] if r["step"] == step), None)
                v = row.get(f"{pre}{k}{tail}") if row else None
                vals[n] = v if isinstance(v, (int, float)) else None
            cells = "".join(f"{vals[n]:>18.4f}" if vals[n] is not None
                            else f"{'-':>18}" for n in names)
            win = ""
            if lower_wins:
                ok = {n: v for n, v in vals.items() if v is not None}
                if ok:
                    b = min(ok, key=ok.get)
                    win = f"  {b.replace('L1_STFT', '').lstrip('_') or 'linear':>8}"
            print(f"  {k:>12}" + cells + win)
    print("\n  A compressed arm winning ONLY the coordinates that shape the quiet")
    print("  high end -- loss_F1, T60_ratio -- while losing the loud ones is the")
    print("  hypothesis this table tests. It would explain winning the parameter")
    print("  error and losing a loudness-weighted audio distance at the same time.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", nargs="+", default=["results/ddsp/emt8"])
    p.add_argument("--metric", nargs="+", default=["spread", "corr", "perr"],
                   choices=["spread", "corr", "perr", "perr_p90"])
    p.add_argument("--every", type=int, default=0,
                   help="Show only steps that are a multiple of this. 0 = all "
                        "logged evals, which is --eval-every (250 on emt8).")
    p.add_argument("--last", type=int, default=0,
                   help="Show only the last N rows; 0 = all.")
    p.add_argument("--trace", action="store_true",
                   help="Per-step tables instead of the arm comparison. Use it "
                        "to watch one coordinate move through the handover.")
    p.add_argument("--at", type=int, default=0,
                   help="Step to compare arms at; 0 = the last step they share.")
    args = p.parse_args()

    arms = {}
    for r in args.root:
        arms.update(load(Path(r)))
    if not arms:
        raise SystemExit(f"no history.json with per-parameter entries under {args.root}")

    if not args.trace:
        return compare(arms, args)

    for metric in args.metric:
        suffix = {"spread": "spread_", "corr": "corr_",
                  "perr": "perr_", "perr_p90": "perr_"}[metric]
        tail = "_p90" if metric == "perr_p90" else ""
        note = {
            "spread": "predicted sd / true sd. ~0 = COLLAPSED, ~1 = right range",
            "corr": "predicted vs true correlation. ~0 = not tracked at all",
            "perr": "median normalised sq error; sqrt(x)*100 = % of range",
            "perr_p90": "the 90th-percentile tail of the same",
        }[metric]
        print(f"\n=== {metric}   {note}")
        for arm, (keys, rows) in sorted(arms.items()):
            sel = [r for r in rows
                   if not args.every or r["step"] % args.every == 0]
            if args.last:
                sel = sel[-args.last:]
            if not sel:
                continue
            print(f"\n  {arm}")
            print(f"  {'step':>7}" + "".join(f"{k:>12}" for k in keys))
            for r in sel:
                cells = ""
                for k in keys:
                    v = r.get(f"{suffix}{k}{tail}")
                    cells += f"{v:>12.4f}" if isinstance(v, (int, float)) else f"{'-':>12}"
                print(f"  {r['step']:>7}" + cells)
    print("\n  Read spread first. A coordinate flat near 0 while the others move")
    print("  is one the encoder has stopped reading its input for -- that is the")
    print("  collapse emt7 showed on real audio, seen here in domain and per step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
