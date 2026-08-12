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

from src.ddsp.train_encoder import z_to_dicts
from src.gd.graddescent import (
    PARAM_KEYS,
    Raw7Space,
    _read_params_csv,
    nmse_6d,
    seven_to_six,
)

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


def const6(train_mean: np.ndarray, gt_val6: list) -> float:
    """constant_predictor_nmse's 6d half, from a precomputed training mean.

    Only the mean of z_train enters, so scanning --n-train is a scan over
    prefix means rather than a rerun of anything expensive. The constant
    predictor emits the same vector for every validation example, so its
    six-composite form is computed once here rather than once per example --
    constant_predictor_nmse materialises n_val identical rows because it is
    written for clarity, not for being called in a loop.
    """
    est6 = seven_to_six(z_to_dicts(train_mean[None, :])[0])
    return float(np.median([nmse_6d(est6, g6) for g6 in gt_val6]))


def scan_prefixes(zt: np.ndarray, zv: np.ndarray, n_vals, target: float):
    """Best (n_train, n_val) prefix pair for one directory pair.

    --n-train and --n-val reach load_dataset as its `limit`, which takes
    csvs[:limit], so a run does not necessarily use a whole directory. Comparing
    only whole directories is what made the first pass miss by ~1e-6.
    """
    cum = np.cumsum(zt.astype(np.float64), axis=0)
    counts = np.arange(1, len(zt) + 1, dtype=np.float64)[:, None]
    means = cum / counts  # prefix means, all at once

    best = None
    for nv in n_vals:
        if nv > len(zv):
            continue
        gt_val = [seven_to_six(g) for g in z_to_dicts(zv[:nv])]
        # Coarse pass over the whole range, then a fine pass around the winner:
        # the metric is smooth in n_train, so this finds the exact prefix
        # without evaluating a hundred thousand of them.
        step = max(1, len(zt) // 200)
        cand = list(range(step, len(zt) + 1, step))
        if len(zt) not in cand:
            cand.append(len(zt))
        # 500 -> 25 -> 1 for a 100k directory, so the exact prefix is reached.
        for _ in range(3):
            scored = [(abs(const6(means[n - 1], gt_val) - target), n) for n in cand]
            err, n_best = min(scored)
            if best is None or err < best[0]:
                best = (err, n_best, nv)
            lo, hi = max(1, n_best - step), min(len(zt), n_best + step)
            step = max(1, step // 20)
            cand = list(range(lo, hi + 1, step))
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reference", type=Path, default=Path("results/ddsp/log_tgtnorm"),
                   help="Run directory, or the history.json inside one")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--n-train", nargs="+", type=int, default=[8192],
                   help="On-the-fly training sizes to consider")
    p.add_argument("--seeds", nargs="+", type=int, default=[0],
                   help="Seeds to consider for on-the-fly sets")
    p.add_argument("--n-val", nargs="+", type=int, default=[512, 1000],
                   help="Validation prefix lengths to try; --n-val defaults to 512")
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

    # val-1000-0.25s, -v2 and -v3 scored identically to twelve decimals on the
    # first pass, which says their parameters are the same and only the
    # rendered audio differs. This metric cannot separate them, so say so once
    # instead of printing the same number three times.
    groups: dict[bytes, list[str]] = {}
    for vn, vz in vals.items():
        groups.setdefault(vz.numpy().tobytes(), []).append(vn)
    for names in groups.values():
        if len(names) > 1:
            print(f"\nidentical parameters (indistinguishable here): {', '.join(names)}")
    vals = {names[0]: vals[names[0]] for names in groups.values()}

    n_vals = sorted({*args.n_val, *(len(v) for v in vals.values())})
    print(f"\n{len(trains)} train x {len(vals)} val candidates; "
          f"n_val tried {n_vals}; n_train scanned over every prefix\n")
    print(f"{'train':<30}{'val':<24}{'n_train':>9}{'n_val':>7}{'const_nmse_6d':>16}{'':>4}")

    hits = []
    for tn, tz in trains.items():
        for vn, vz in vals.items():
            got = scan_prefixes(tz.numpy(), vz.numpy(), n_vals, target)
            if got is None:
                continue
            err, nt, nv = got
            hit = err <= args.tol
            if hit:
                hits.append((tn, vn, nt, nv))
            print(f"{tn:<30}{vn:<24}{nt:>9}{nv:>7}{target + err:>16.12f}"
                  f"{'  <== MATCH' if hit else '':>4}")

    print()
    if len(hits) == 1:
        t, v, nt, nv = hits[0]
        gen = t.startswith("<")
        src = "(generated; omit --data-dir)" if gen else f"--data-dir data/{t}"
        print(f"recipe recovered: {src} --val-data-dir data/{v} "
              f"--n-train {nt} --n-val {nv}")
    elif hits:
        print(f"{len(hits)} candidates match to {args.tol:g}. They agree on the "
              f"training mean, so pick by which datasets plausibly existed at the "
              f"time, or tighten --tol.")
    else:
        print("no pair matches. Widen --n-train/--seeds for generated sets, or the "
              "run used a dataset no longer in --data-root.")


if __name__ == "__main__":
    main()
