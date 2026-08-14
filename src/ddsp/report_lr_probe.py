"""The learning-rate grid as the table a methods section needs.

Reports every cell, then best-over-rate per loss. Best-over-rate is the honest
summary for this question: the claim is not "log failed at 3e-4", it is "log
failed at its own best rate, on the same grid where linear succeeded at its
own best rate". Reporting a single shared rate would leave the obvious reply
available, and reporting best-overall would hide which arm each rate favoured.

The 3e-4 column comes from the ladder, not from the grid. lr_probe.sh now
defaults to the ladder's exact recipe, so a cell at the ladder's own rate would
reproduce a ladder arm run for run -- it is read from
results/ddsp/eps_ladder_adam1e8 instead of computed twice.

The older results/ddsp/lr_* runs are NOT folded in by default any more. They
used adam_eps=1e-16, which the head probe showed is a defect rather than a
setting, and mixing two optimizers into one grid would charge their difference
to the learning rate. --legacy restores them for reproducing the earlier table.

    python -m src.ddsp.report_lr_probe
    python -m src.ddsp.report_lr_probe --legacy
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

# The old probes' loss names, mapped onto the ladder's.
ALIASES = {"L1_STFT_c2": "L1_STFT_eps1", "L1_STFT_log": "L1_STFT_eps1e7"}


def cells(roots):
    """(loss, lr, record) for every run directory found under any root."""
    out = []
    for root in roots:
        for hp in sorted(glob.glob(os.path.join(root, "*", "history.json"))):
            name = Path(hp).parent.name
            m = re.match(r"^(?:lr_)?(.+?)_(\d+(?:\.\d+)?e-?\d+)$", name)
            if not m:
                continue
            loss, lr = m.group(1), float(m.group(2))
            try:
                d = json.load(open(hp))
            except Exception:
                continue
            ev = [r for r in d["history"] if "val_nmse_6d" in r]
            if not ev:
                continue
            best = min(ev, key=lambda r: r["val_nmse_6d_geo"] if "val_nmse_6d_geo" in r
                       else r["val_nmse_6d"])
            spreads = [best[k] for k in best if k.startswith("spread_")]
            out.append({
                "loss": ALIASES.get(loss, loss),
                "lr": lr,
                "step": d["history"][-1]["step"],
                "nmse": best["val_nmse_6d"],
                "ratio": best["val_nmse_6d"] / d["const_nmse_6d"],
                "train_over_sat": best["train_loss"] / d["saturation"],
                "spread": sum(spreads) / len(spreads) if spreads else float("nan"),
                "dir": name,
            })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--roots", nargs="+", default=["results/ddsp/lr_probe"])
    p.add_argument("--ladder-root", default="results/ddsp/eps_ladder_adam1e8",
                   help="Where the ladder's arms live; each supplies the cell at "
                        "--ladder-lr, since lr_probe.sh shares its recipe exactly")
    p.add_argument("--ladder-lr", type=float, default=3e-4)
    p.add_argument("--legacy", action="store_true",
                   help="Also read results/ddsp/lr_*, which ran adam_eps=1e-16. "
                        "Off by default: that is a different optimizer, and "
                        "folding it in would charge its effect to the rate.")
    args = p.parse_args()

    roots = list(args.roots)
    if args.legacy:
        roots.append("results/ddsp")
        print("WARNING: --legacy mixes adam_eps=1e-16 runs into the grid.\n")
    rows = cells(roots)

    # The ladder's arms are the grid's own rate, under the same recipe, so they
    # are cells rather than a separate experiment. Their directories are named
    # for the loss alone, with no rate suffix, so cells() cannot match them.
    for hp in sorted(glob.glob(os.path.join(args.ladder_root, "*", "history.json"))):
        name = Path(hp).parent.name
        try:
            d = json.load(open(hp))
        except Exception:
            continue
        ev = [r for r in d["history"] if "val_nmse_6d" in r]
        if not ev:
            continue
        best = min(ev, key=lambda r: r.get("val_nmse_6d_geo", r["val_nmse_6d"]))
        spreads = [best[k] for k in best if k.startswith("spread_")]
        rows.append({
            "loss": ALIASES.get(name, name),
            "lr": args.ladder_lr,
            "step": d["history"][-1]["step"],
            "nmse": best["val_nmse_6d"],
            "ratio": best["val_nmse_6d"] / d["const_nmse_6d"],
            "train_over_sat": best["train_loss"] / d["saturation"],
            "spread": sum(spreads) / len(spreads) if spreads else float("nan"),
            "dir": f"{args.ladder_root}/{name}",
        })
    if not rows:
        print("no lr-probe runs found under " + ", ".join(args.roots))
        return

    # A cell can appear under both roots; keep the one that trained furthest.
    uniq: dict[tuple, dict] = {}
    for r in rows:
        k = (r["loss"], r["lr"])
        if k not in uniq or r["step"] > uniq[k]["step"]:
            uniq[k] = r
    rows = sorted(uniq.values(), key=lambda r: (r["loss"], -r["lr"]))

    print(f"{'loss':<20}{'lr':>9}{'step':>8}{'train/sat':>11}{'nmse_6d':>10}"
          f"{'ratio':>7}{'spread':>9}")
    for r in rows:
        print(f"{r['loss']:<20}{r['lr']:>9.0e}{r['step']:>8}{r['train_over_sat']:>11.3f}"
              f"{r['nmse']:>10.4f}{r['ratio']:>7.2f}{r['spread']:>9.4f}")

    print(f"\nbest over rate, per loss   (ratio < 1 recovers parameters; "
          f"ratio ~ 1 is the constant predictor)")
    print(f"{'loss':<20}{'best lr':>9}{'nmse_6d':>10}{'ratio':>7}{'spread':>9}"
          f"{'rates tried':>13}")
    for loss in sorted({r["loss"] for r in rows}):
        sub = [r for r in rows if r["loss"] == loss]
        b = min(sub, key=lambda r: r["nmse"])
        print(f"{loss:<20}{b['lr']:>9.0e}{b['nmse']:>10.4f}{b['ratio']:>7.2f}"
              f"{b['spread']:>9.4f}{len(sub):>13}")

    # The sentence the methods section wants: how much does the rate move each
    # arm, against how much the loss moves it. If the spread across rates is
    # small for every arm while the spread across arms is large, the rate is
    # measurably not the explanation.
    print()
    for loss in sorted({r["loss"] for r in rows}):
        sub = [r["nmse"] for r in rows if r["loss"] == loss]
        if len(sub) > 1:
            print(f"  {loss:<20} nmse across rates: {min(sub):.4f} - {max(sub):.4f} "
                  f"({max(sub) / max(min(sub), 1e-12):.2f}x)")
    best = {loss: min(r["nmse"] for r in rows if r["loss"] == loss)
            for loss in {r["loss"] for r in rows}}
    if len(best) > 1:
        lo, hi = min(best.values()), max(best.values())
        print(f"  {'across losses (best)':<20} {lo:.4f} - {hi:.4f} "
              f"({hi / max(lo, 1e-12):.2f}x)")


if __name__ == "__main__":
    main()
