"""Which datasets did a finished run train on? Read it back off its metrics.

The 120k sweep's command is not recorded anywhere -- history.json stores
results, not arguments -- and data/ holds several plausible train and val sets
(train-100000-0.25s, -v3, train-50000-s1, train-20000; val-1000-0.25s, -v2,
-v3). Guessing wrong would make the eps ladder non-comparable to the arms it is
meant to interpolate, while looking entirely healthy.

const_nmse_6d pins it down. It is the median NMSE of always predicting the
training mean, so it is a deterministic function of the (train, val) pair and
nothing else -- not of the loss, the seed, the schedule or the architecture.
Every run records it. Enumerating the pairs on disk and matching the recorded
value therefore identifies which pair a run used, exactly.

Only parameters are needed, never audio: z comes from the per-IR CSV through
Raw7Space.gt_z, and rows whose npz is absent are skipped the same way
load_dataset skips them. On-the-fly training sets are covered too, since
synth_dataset's z is just uniform noise from a seeded generator.

    python -m src.ddsp.diag_recover_recipe --reference results/ddsp/log_tgtnorm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.ddsp.train_encoder import constant_predictor_nmse
from src.gd.graddescent import PARAM_KEYS, Raw7Space, _read_params_csv

# Raw7Space.gt_z touches only the bounds, so a CPU space with no configured
# plate is enough and costs nothing.
_SPACE = Raw7Space(torch.device("cpu"), torch.float32, normalize=False)


def z_from_dir(d: Path) -> torch.Tensor | None:
    """Parameters of a dataset directory, in the order load_dataset would see."""
    csvs = sorted(d.glob("random_IR_params_*.csv"))
    if not csvs:
        return None
    zs = []
    for c in csvs:
        rid = c.stem.split("_")[-1]
        if not (d / f"random_IR_{rid}.npz").exists():
            continue
        zs.append(_SPACE.gt_z(_read_params_csv(c)))
    if not zs:
        return None
    return torch.as_tensor(np.asarray(zs), dtype=torch.float32)


def z_synthetic(n: int, seed: int) -> torch.Tensor:
    """synth_dataset's z, without rendering any audio."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand((n, len(PARAM_KEYS)), generator=g) * 2.0 - 1.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reference", type=Path, default=Path("results/ddsp/log_tgtnorm"),
                   help="Run directory, or the history.json inside one")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--n-train", nargs="+", type=int, default=[8192],
                   help="On-the-fly training sizes to consider")
    p.add_argument("--seeds", nargs="+", type=int, default=[0],
                   help="Seeds to consider for on-the-fly sets")
    p.add_argument("--tol", type=float, default=1e-9)
    args = p.parse_args()

    ref = args.reference
    if ref.is_dir():
        ref = ref / "history.json"
    target = float(json.load(ref.open())["const_nmse_6d"])
    print(f"reference {ref}\n  const_nmse_6d = {target!r}\n")

    dirs = sorted(d for d in args.data_root.iterdir() if d.is_dir())
    trains, vals = {}, {}
    for d in dirs:
        z = z_from_dir(d)
        if z is None:
            continue
        (vals if "val" in d.name else trains)[d.name] = z
        print(f"  {d.name:<28} {z.shape[0]:>7} IRs")
    for n in args.n_train:
        for s in args.seeds:
            trains[f"<generated n={n} seed={s}>"] = z_synthetic(n, s)
    # A validation set is generated with seed + 1, so any val dir could also be
    # a generated set; those are covered by the same enumeration.
    for n in (512,):
        for s in args.seeds:
            vals[f"<generated n={n} seed={s + 1}>"] = z_synthetic(n, s + 1)

    print(f"\n{len(trains)} train x {len(vals)} val candidates\n")
    print(f"{'train':<30}{'val':<30}{'const_nmse_6d':>16}{'':>4}")
    hits = []
    for tn, tz in trains.items():
        for vn, vz in vals.items():
            c6, _ = constant_predictor_nmse(tz, vz)
            hit = abs(c6 - target) <= args.tol
            if hit:
                hits.append((tn, vn))
            print(f"{tn:<30}{vn:<30}{c6:>16.12f}{'  <== MATCH' if hit else '':>4}")

    print()
    if len(hits) == 1:
        t, v = hits[0]
        print(f"recipe recovered: --data-dir data/{t}  --val-data-dir data/{v}")
    elif hits:
        print(f"{len(hits)} pairs match to {args.tol:g}; distinguish them by another "
              f"recorded constant, or tighten --tol")
    else:
        print("no pair matches. The run used a dataset not in --data-root, or an "
              "n-train/seed outside those enumerated -- widen --n-train/--seeds.")


if __name__ == "__main__":
    main()
