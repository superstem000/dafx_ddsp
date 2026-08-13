"""TensorBoard scalars out of the experiment tree, into CSVs the paper can use.

experiments/ is 21 GB, almost all of it Lightning checkpoints, and none of what
a figure needs is in them. The scalars are, and they compress to a few hundred
KB -- so this pulls them out and the tree stays local.

Two outputs per run: a full per-step history, and one row of final values. The
final-value row is what a table wants; the history is what a curve wants, and
having it means a reviewer question about "did it plateau or was it still
moving" does not require the 21 GB back.

size_guidance is set explicitly on every key. It is a *cap*, and 0 means retain
everything -- so leaving images and audio unset is how a summary read ends up
loading every logged spectrogram and taking ten minutes.

    python external/diffmoog/tools/extract_scalars.py \
        --root external/diffmoog/experiments/current \
        --out results/diffmoog
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

GUIDANCE = {
    "scalars": 100000,
    "histograms": 1,
    "images": 1,
    "audio": 1,
    "tensors": 1,
    "compressedHistograms": 1,
}


def event_dirs(run_dir: str):
    """Deepest directories under a run that actually contain event files."""
    seen = set()
    for f in glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"),
                       recursive=True):
        seen.add(os.path.dirname(f))
    return sorted(seen)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="external/diffmoog/experiments/current")
    p.add_argument("--out", default="results/diffmoog")
    p.add_argument("--runs", nargs="+", default=None,
                   help="Experiment names to extract; default is all of them")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = args.runs or sorted(
        d.name for d in Path(args.root).iterdir() if d.is_dir()
    )

    finals = []
    for name in names:
        dirs = event_dirs(os.path.join(args.root, name))
        if not dirs:
            print(f"  {name}: no event files")
            continue
        # A run can be resumed into several version_ dirs; take them in order
        # and let later steps win, which is what the training actually did.
        merged: dict[int, dict] = {}
        tags: set[str] = set()
        for d in dirs:
            ea = EventAccumulator(d, size_guidance=GUIDANCE)
            ea.Reload()
            for t in ea.Tags()["scalars"]:
                tags.add(t)
                for e in ea.Scalars(t):
                    merged.setdefault(e.step, {})[t] = e.value

        if not merged:
            print(f"  {name}: no scalars")
            continue

        cols = ["step"] + sorted(tags)
        hp = os.path.join(args.out, f"{name}_history.csv")
        with open(hp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for s in sorted(merged):
                w.writerow([s] + [merged[s].get(t, "") for t in cols[1:]])

        # Three numbers per tag, not one. Every arm here degrades substantially
        # between its best epoch and its last -- q_ploss runs 0.047 at step 2279
        # and 0.113 at 7999 -- so "final" and "best" disagree by about 2x and
        # they do not even rank the arms the same way. Reporting one of them
        # unlabelled is how a table ends up meaning something other than what
        # its caption says.
        #
        # min_step is the important one. The best params_loss and the best
        # mfcc_mae occur at different steps, so a row of per-metric minima is
        # not one model -- quoting it as if a single checkpoint achieved all of
        # them overstates the result, and min_step is what makes that visible.
        ordered = sorted(merged)
        rows_out = {}
        for t in tags:
            vals = [(st, merged[st][t]) for st in ordered if t in merged[st]]
            if not vals:
                continue
            lo = min(vals, key=lambda x: x[1])
            rows_out[f"final_{t}"] = vals[-1][1]
            rows_out[f"min_{t}"] = lo[1]
            rows_out[f"minstep_{t}"] = lo[0]
        finals.append({"run": name, "last_step": max(merged), **rows_out})
        print(f"  {name}: {len(merged)} steps, {len(tags)} tags -> {hp}")

    if finals:
        cols = ["run", "last_step"] + sorted(
            {k for r in finals for k in r} - {"run", "last_step"}
        )
        fp = os.path.join(args.out, "final_metrics.csv")
        with open(fp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(finals)
        print(f"\n{len(finals)} runs -> {fp}")


if __name__ == "__main__":
    main()
