"""Watch a running sweep as trajectories, not as a single latest row.

A last-row view cannot tell "has not started yet" from "started and stopped",
which on this problem is the whole question -- the log arms sit near the
constant-predictor floor from step one, so the informative thing is whether
anything is moving, not where it currently is. Each eval is a row here and each
arm a column, so a stalled arm shows as a flat column and a cliff between two
rungs shows as a step across the table.

Reads history.json, which train_encoder rewrites every --log-every steps, so it
works mid-run. Arms are whatever directories exist under the root.

    python -m src.ddsp.monitor_sweep
    python -m src.ddsp.monitor_sweep --root results/ddsp/lr_probe
    python -m src.ddsp.monitor_sweep --metrics ratio spread train_sat
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

METRICS = {
    # ratio > 1 is worse than emitting the training mean; ~1 is the floor
    # itself; below 1 is the only regime in which anything is being recovered.
    "ratio": ("val_nmse_6d / const_nmse_6d", "{:.2f}"),
    # prediction sd over ground-truth sd. Starts near 0 by construction (the
    # output layer is initialized at std 0.01 with zero bias), so this reads as
    # "has it left the initialization", not "has it collapsed".
    "spread": ("mean spread_* over the raw seven", "{:.3f}"),
    "train_sat": ("train_loss / saturation; ~1 = gradient uninformative", "{:.3f}"),
    "nmse": ("val_nmse_6d", "{:.4f}"),
}


def load(root: str):
    arms = {}
    for hp in sorted(glob.glob(os.path.join(root, "*", "history.json"))):
        name = Path(hp).parent.name
        try:
            d = json.load(open(hp))
        except Exception:
            continue
        rows = {}
        for r in d["history"]:
            if "val_nmse_6d" not in r:
                continue
            sp = [r[k] for k in r if k.startswith("spread_")]
            rows[r["step"]] = {
                "ratio": r["val_nmse_6d"] / d["const_nmse_6d"],
                "spread": sum(sp) / len(sp) if sp else float("nan"),
                "train_sat": r["train_loss"] / d["saturation"],
                "nmse": r["val_nmse_6d"],
            }
        h = d["history"]
        arms[name] = {
            "rows": rows,
            "last": h[-1]["step"] if h else 0,
            "rate": h[-1]["step"] / max(h[-1]["elapsed_s"], 1e-9) if h else 0.0,
            "clip": 100.0 * sum(bool(q.get("clipped")) for q in h) / max(len(h), 1),
        }
    return arms


def short(name: str) -> str:
    if name == "L1_STFT":
        return "linear"
    return name.replace("L1_STFT_", "").replace("L1_STFT", "lin")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="results/ddsp/eps_ladder")
    p.add_argument("--metrics", nargs="+", default=["ratio", "spread"],
                   choices=sorted(METRICS))
    p.add_argument("--steps", type=int, default=40000, help="For the ETA column")
    p.add_argument("--tail", type=int, default=0,
                   help="Show only the last N eval rows (0 = all)")
    args = p.parse_args()

    arms = load(args.root)
    if not arms:
        print(f"no runs under {args.root}")
        return

    names = sorted(arms)
    w = max(9, max(len(short(n)) for n in names) + 2)

    print(f"{'arm':<18}{'step':>8}{'st/s':>7}{'clip%':>7}{'eta_h':>7}")
    for n in names:
        a = arms[n]
        eta = (args.steps - a["last"]) / a["rate"] / 3600 if a["rate"] > 0 else float("nan")
        print(f"{short(n):<18}{a['last']:>8}{a['rate']:>7.2f}{a['clip']:>7.1f}{eta:>7.1f}")

    all_steps = sorted({s for a in arms.values() for s in a["rows"]})
    if args.tail:
        all_steps = all_steps[-args.tail:]

    for m in args.metrics:
        desc, fmt = METRICS[m]
        print(f"\n=== {m}   ({desc})")
        print(f"{'step':>8}" + "".join(f"{short(n):>{w}}" for n in names))
        for s in all_steps:
            cells = []
            for n in names:
                r = arms[n]["rows"].get(s)
                cells.append(f"{fmt.format(r[m]):>{w}}" if r else f"{'-':>{w}}")
            print(f"{s:>8}" + "".join(cells))


if __name__ == "__main__":
    main()
