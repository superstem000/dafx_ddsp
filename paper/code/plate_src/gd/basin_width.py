"""Measure where a loss puts its optimum, and how close a local method must
already be to descend to it -- in a choice of parameter space.

Two properties, measured separately because they can trade off:

    argmin offset   -- distance from the true parameter to the loss minimum.
                       An accuracy property: does the loss identify the right
                       answer at all.
    basin halfwidth -- radius within which the loss rises monotonically away
                       from ground truth.  A reachability property: outside it
                       a descent step is as likely to move away as toward.

Both are reported as a fraction of each coordinate's full search range, so the
numbers are comparable between parameter spaces whose normalized intervals
differ ([0,1] here, [-1,1] for CMA-ES).

Why the space matters
---------------------
Basin width is not a property of the loss alone; it is a property of the loss
*composed with a parametrization*.  Log-scaling a coordinate that spans decades
compresses it into the unit interval, so a fixed physical basin reports as a
different fraction than it would under linear scaling.  Comparing a gradient
method against the CMA-ES sweep therefore requires measuring on the terrain
CMA-ES actually searches, not a reparametrized one.

  --space raw7       the CMA-ES terrain: seven raw parameters mapped linearly
                     from [-1,1], via the very helpers fit_7param_norm_es uses,
                     so the two cannot silently diverge.
  --space composite6 the identifiable space graddescent.py searches: five shape
                     coordinates, log-scaled where they span decades, with mu
                     profiled out exactly.

  --normalize        peak-normalize target and candidate, as run_cmaes does
                     (BatchedModalPlateTorch.forward defaults normalize=True).
                     This discards the amplitude that identifies mu, so mu
                     profiling is disabled when it is set.

Usage:
    # the terrain CMA-ES solves in one restart
    python -m src.gd.basin_width --space raw7 --normalize --losses L1_STFT

    # the terrain graddescent.py descends
    python -m src.gd.basin_width --space composite6 --losses L1_STFT
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.cmaes.fit_7param_norm_es import (
    BOUNDS_HI_NP,
    BOUNDS_LO_NP,
    PARAM_KEYS,
    norm_to_physical,
    physical_to_plate14_tensor,
)
from src.cmaes.fit_7param_norm_es import load_target_ir_from_npz as load_target_normalized
from src.gd.graddescent import SAMPLE_RATE, SHAPE_KEYS, NormBox, _read_params_csv, solve_mu_by_scale
from src.loss.loss_selector import select_loss_function
from src.mu_optimization.ternary_mu import MU_MAX, MU_MIN
from src.mu_optimization.ternary_mu import load_target_ir_from_npz as load_target_raw
from src.mu_optimization.ternary_mu import seven_to_six
from src.plate.SevenParamPlate import BatchedModalPlateTorch as SevenParamPlate
from src.plate.SixParamPlate import BatchedModalPlateTorch as SixParamPlate


def build_offsets(n_per_decade: int, min_off: float, max_off: float) -> np.ndarray:
    """Signed offsets about 0, log-spaced, so narrow basins are resolved."""
    n = max(2, int(round(n_per_decade * np.log10(max_off / min_off))))
    mag = np.logspace(np.log10(min_off), np.log10(max_off), n)
    return np.concatenate([-mag[::-1], [0.0], mag])


def descent_halfwidth(offsets: np.ndarray, losses: np.ndarray, rel_tol: float = 1e-9) -> float:
    """Largest |offset| out to which the loss rises monotonically from centre.

    Walks outward in each direction and stops at the first non-increasing step.
    Returns the smaller of the two half-widths: the radius within which descent
    points home from either side.
    """
    centre = int(np.argmin(np.abs(offsets)))
    reaches = []
    for direction in (1, -1):
        idx, reach = centre, 0.0
        while True:
            nxt = idx + direction
            if nxt < 0 or nxt >= len(offsets):
                break
            if losses[nxt] <= losses[idx] * (1.0 + rel_tol):
                break
            idx = nxt
            reach = abs(offsets[idx])
        reaches.append(reach)
    return float(min(reaches))


class Raw7Space:
    """The terrain CMA-ES searches: seven raw parameters, linear from [-1, 1]."""

    name = "raw7"
    keys = list(PARAM_KEYS)
    lo, hi, width = -1.0, 1.0, 2.0

    def __init__(self, device, dtype, normalize: bool):
        self.device, self.dtype, self.normalize = device, dtype, normalize
        self.plate = SevenParamPlate(
            sample_rate=SAMPLE_RATE, device=device, dtype=dtype, drop_sub_20hz_modes=False
        )

    def gt_z(self, gt7: Dict[str, float]) -> np.ndarray:
        phys = np.array([gt7[k] for k in PARAM_KEYS], dtype=np.float64)
        return -1.0 + 2.0 * (phys - BOUNDS_LO_NP) / (BOUNDS_HI_NP - BOUNDS_LO_NP)

    def synth(self, z: np.ndarray, duration: float) -> torch.Tensor:
        plate14 = physical_to_plate14_tensor(norm_to_physical(z), self.device).to(dtype=self.dtype)
        with torch.no_grad():
            # normalize=True is run_cmaes's effective default; keep it identical.
            return self.plate(plate14, duration=duration, normalize=self.normalize)


class Composite6Space:
    """The terrain graddescent.py searches: five shape coords, mu profiled out."""

    name = "composite6"
    keys = list(SHAPE_KEYS)
    lo, hi, width = 0.0, 1.0, 1.0

    def __init__(self, device, dtype, normalize: bool):
        self.device, self.dtype, self.normalize = device, dtype, normalize
        self.plate = SixParamPlate(
            sample_rate=SAMPLE_RATE, device=device, dtype=dtype, drop_sub_20hz_modes=False
        )
        self.box = NormBox(SHAPE_KEYS, device=device, dtype=dtype)
        self.mu_ref = float(np.sqrt(MU_MIN * MU_MAX))

    def gt_z(self, gt7: Dict[str, float]) -> np.ndarray:
        six = seven_to_six(gt7)
        return self.box.to_unit_np(np.array([six[k] for k in SHAPE_KEYS]))

    def synth(self, z: np.ndarray, duration: float) -> torch.Tensor:
        zt = torch.as_tensor(z, device=self.device, dtype=self.dtype)
        mu = torch.full((len(z),), self.mu_ref, device=self.device, dtype=self.dtype)
        six = torch.cat([mu.unsqueeze(1), self.box.to_physical(zt)], dim=1)
        with torch.no_grad():
            pred = self.plate(six, duration=duration, normalize=self.normalize)
        return pred


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, default=Path("random-IR-200-0.2s"))
    p.add_argument("--ids", type=str, nargs="+", default=["0001"])
    p.add_argument("--losses", type=str, nargs="+", default=["L1_STFT"])
    p.add_argument("--space", type=str, default="composite6", choices=["composite6", "raw7"])
    p.add_argument(
        "--normalize", action="store_true",
        help="Peak-normalize target and candidate, as run_cmaes does",
    )
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--n-per-decade", type=int, default=12)
    p.add_argument("--min-offset", type=float, default=1e-5, help="Smallest offset, range fraction")
    p.add_argument("--max-offset", type=float, default=0.5, help="Largest offset, range fraction")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    out_dir = args.output_dir or Path("results/gd/basin_width") / (
        f"{args.space}{'_norm' if args.normalize else ''}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    space = (Raw7Space if args.space == "raw7" else Composite6Space)(device, dtype, args.normalize)
    loss_fns = {n: select_loss_function(n, sample_rate=SAMPLE_RATE, device=device) for n in args.losses}

    # mu is a pure output scale, so peak normalization erases it; profiling it
    # would then be solving for a quantity the loss cannot see.
    profile_mu = (args.space == "composite6") and not args.normalize

    offsets = build_offsets(args.n_per_decade, args.min_offset, args.max_offset)
    print(f"device={device}  space={space.name}  normalize={args.normalize}  profile_mu={profile_mu}")
    print(f"losses={args.losses}  {len(offsets)} offsets/coord  coords={space.keys}\n")

    profiles: List[Dict] = []
    summary: List[Dict] = []

    for rid in args.ids:
        gt7 = _read_params_csv(args.dataset_dir / f"random_IR_params_{rid}.csv")
        npz = args.dataset_dir / f"random_IR_{rid}.npz"
        loader = load_target_normalized if args.normalize else load_target_raw
        target_np = loader(npz, args.duration, SAMPLE_RATE)
        target = torch.as_tensor(target_np, device=device, dtype=dtype).unsqueeze(0)
        duration = target_np.shape[0] / float(SAMPLE_RATE)
        z_gt = space.gt_z(gt7)

        for j, key in enumerate(space.keys):
            # Offsets leaving the box are dropped rather than clipped, so the
            # monotone walk never reads a boundary plateau as a wall.
            z_vals = z_gt[j] + offsets * space.width
            keep = (z_vals >= space.lo) & (z_vals <= space.hi)
            z_vals, off = z_vals[keep], offsets[keep]

            z = np.tile(z_gt, (len(z_vals), 1))
            z[:, j] = z_vals
            pred = space.synth(z, duration)

            for name, fn in loss_fns.items():
                if profile_mu:
                    mu_b = torch.full((len(z_vals),), space.mu_ref, device=device, dtype=dtype)
                    _, lv = solve_mu_by_scale(pred, mu_b, target, fn, 40, dtype)
                else:
                    with torch.no_grad():
                        lv = fn(target.expand(len(z_vals), -1), pred)
                lv = np.nan_to_num(lv.cpu().numpy().astype(np.float64), nan=np.inf)

                i_best = int(np.argmin(lv))
                summary.append(
                    {
                        "id": rid,
                        "space": space.name,
                        "normalize": args.normalize,
                        "loss": name,
                        "coord": key,
                        "argmin_offset": float(off[i_best]),
                        "loss_at_gt": float(lv[int(np.argmin(np.abs(off)))]),
                        "loss_at_argmin": float(lv[i_best]),
                        "basin_halfwidth": descent_halfwidth(off, lv),
                    }
                )
                for o, v in zip(off, lv):
                    profiles.append(
                        {"id": rid, "loss": name, "coord": key, "offset": float(o), "value": float(v)}
                    )

    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "basin_summary.csv", index=False)
    pd.DataFrame(profiles).to_csv(out_dir / "basin_profiles.csv", index=False)

    print(f"{'loss':<20} {'median basin':>13} {'median |argmin off|':>21}   (range fractions)")
    for name in args.losses:
        sub = df[df["loss"] == name]
        if not sub.empty:
            print(
                f"{name:<20} {sub['basin_halfwidth'].median():>13.2e} "
                f"{sub['argmin_offset'].abs().median():>21.2e}"
            )
    print("\nper-coordinate basin half-width:")
    print(df.pivot_table(index="coord", columns="loss", values="basin_halfwidth", aggfunc="median"))

    n = len(space.keys)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4), squeeze=False)
    pdf = pd.DataFrame(profiles)
    for ax, key in zip(axes[0], space.keys):
        for name in args.losses:
            sub = pdf[(pdf["coord"] == key) & (pdf["loss"] == name) & (pdf["id"] == args.ids[0])]
            if sub.empty:
                continue
            v = sub["value"].to_numpy()
            rng = v.max() - v.min()
            ax.plot(sub["offset"], (v - v.min()) / rng if rng > 0 else v * 0, linewidth=1.0, label=name)
        ax.set_xscale("symlog", linthresh=args.min_offset)
        ax.axvline(0.0, color="k", linewidth=0.8, linestyle="--")
        ax.set_title(key)
        ax.set_xlabel("offset from GT (range fraction)")
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("min-max scaled loss")
    axes[0][-1].legend(fontsize=7)
    plt.suptitle(
        f"Loss about ground truth | space={space.name} normalize={args.normalize} | IR {args.ids[0]}",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_dir / "basin_profiles.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote basin_summary.csv, basin_profiles.csv, basin_profiles.png to {out_dir}")


if __name__ == "__main__":
    main()
