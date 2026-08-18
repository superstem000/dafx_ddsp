"""Write every diffsynth run's scalars to a CSV beside it, for the paper bundle.

    python scripts/ds_export_scalars.py
    python scripts/ds_export_scalars.py --only '^synth_' --out-name scalars.csv

results/diffsynth is ~16 GB, almost all of it TensorBoard event files and
Lightning checkpoints. make_paper_bundle.py already skips *.ckpt and
events.out.tfevents.*, which leaves the hydra configs and split manifests but
no numbers at all -- and the numbers are the only part the paper reads.

This walks each run's event files once and writes scalars.csv into the run
directory: one row per (epoch, tag), covering every scalar the run logged, not
just the handful monitor_diffsynth displays. A 400-epoch run is ~200 KB, so the
whole set is a few MB and can live in the bundle and in git.

Deliberately NOT reusing monitor_diffsynth's cache. .monitor_cache.json exists
in most run directories and holds the same numbers, but it is keyed on file
size and mtime and carries only the tags that monitor happens to plot -- it is
an accelerator for one tool, not a record. This writes a documented format that
does not change when the monitor's tag list does.

Epoch is derived from the optimiser step, since Lightning logs against
global_step: 16000 training examples at batch 64 is 250 steps per epoch.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from pathlib import Path


def export(run_dir: str, out_name: str, steps_per_epoch: int) -> tuple[int, int] | None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb = os.path.join(run_dir, "tb_logs")
    if not glob.glob(os.path.join(tb, "**", "events.out.tfevents.*"), recursive=True):
        return None
    ea = EventAccumulator(tb, size_guidance={"scalars": 0})
    ea.Reload()
    tags = sorted(ea.Tags().get("scalars", []))
    if not tags:
        return None

    rows: dict[int, dict[str, float]] = {}
    for t in tags:
        for e in ea.Scalars(t):
            rows.setdefault(e.step, {})[t] = e.value

    out = os.path.join(run_dir, out_name)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "epoch"] + tags)
        for step in sorted(rows):
            r = rows[step]
            w.writerow([step, step // steps_per_epoch]
                       + ["" if t not in r else f"{r[t]:.6g}" for t in tags])
    return len(rows), len(tags)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--only", default=None, metavar="REGEX")
    p.add_argument("--out-name", default="scalars.csv")
    p.add_argument("--steps-per-epoch", type=int, default=250,
                   help="16000 train examples / batch 64")
    args = p.parse_args()

    runs = [d for d in sorted(glob.glob(os.path.join(args.root, "*")))
            if os.path.isdir(d) and (not args.only or re.search(args.only, Path(d).name))]
    if not runs:
        print(f"no runs under {args.root}")
        return

    total = 0
    for d in runs:
        name = Path(d).name
        res = export(d, args.out_name, args.steps_per_epoch)
        if res is None:
            print(f"{name:<24} no event files")
            continue
        n_rows, n_tags = res
        size = os.path.getsize(os.path.join(d, args.out_name))
        total += size
        print(f"{name:<24} {n_rows:>5} rows x {n_tags:>3} tags   {size/1024:>7.0f} KB")
    print(f"\n{total/1024/1024:.1f} MB total")


if __name__ == "__main__":
    main()
