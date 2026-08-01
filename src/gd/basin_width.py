"""Measure the gradient-reachability of a loss: where its optimum sits, and how
close you must already be for a local method to descend to it.

This is a gradient-specific landscape feature, complementary to the ELA features
in monotonicity_decomp.py.  Monotonicity there measures the *global* fraction of
downhill steps toward ground truth; what a gradient method actually needs is the
*local* radius inside which descent is uninterrupted:

    argmin offset  -- distance from the true parameter to the loss minimum, in
                      normalized coordinates.  This is an accuracy property: it
                      says whether the loss identifies the right answer at all.

    basin halfwidth -- how far from ground truth you can start along a
                      coordinate and still have the loss increase monotonically
                      with distance.  This is a reachability property: outside
                      it, a descent step is as likely to move away as toward.

The two can trade off.  A compression that flattens spectral peaks may widen the
basin while displacing the optimum, which is exactly the tension between making
a loss optimizable and making it an accurate estimator.  Reporting both per loss
makes that tradeoff measurable instead of assumed.

Offsets are sampled logarithmically about ground truth, so a basin narrower than
a uniform grid could resolve is still measured rather than reported as zero.

Because the synthesized IRs depend only on the parameters, not the loss, each
coordinate is synthesized once and every loss is evaluated against the cached
batch.

Usage:
    python -m src.gd.basin_width --dataset-dir random-IR-200-0.2s \
        --ids 0001 0002 0003 \
        --losses L1_STFT L1_STFT_log L1_STFT_c2 L1_STFT_pow MSS
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

from src.gd.graddescent import SAMPLE_RATE, SHAPE_KEYS, NormBox, _read_params_csv, solve_mu_by_scale
from src.loss.loss_selector import select_loss_function
from src.mu_optimization.ternary_mu import MU_MAX, MU_MIN, load_target_ir_from_npz, seven_to_six
from src.plate.SixParamPlate import BatchedModalPlateTorch as SixParamPlate


def build_offsets(n_per_decade: int, min_off: float, max_off: float) -> np.ndarray:
    """Signed offsets about 0, log-spaced, so narrow basins are resolved."""
    n = max(2, int(round(n_per_decade * np.log10(max_off / min_off))))
    mag = np.logspace(np.log10(min_off), np.log10(max_off), n)
    return np.concatenate([-mag[::-1], [0.0], mag])


def descent_halfwidth(offsets: np.ndarray, losses: np.ndarray, rel_tol: float = 1e-9) -> float:
    """Largest |offset| out to which the loss rises monotonically from the centre.

    Walks outward in each direction from offset 0 and stops at the first step
    that does not increase.  Returns the smaller of the two half-widths, i.e.
    the radius within which descent points home from either side.
    """
    centre = int(np.argmin(np.abs(offsets)))
    out = []
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
        out.append(reach)
    return float(min(out))


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, default=Path("random-IR-200-0.2s"))
    p.add_argument("--ids", type=str, nargs="+", default=["0001"])
    p.add_argument("--losses", type=str, nargs="+", default=["L1_STFT"])
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--output-dir", type=Path, default=Path("results/gd/basin_width"))
    p.add_argument("--n-per-decade", type=int, default=12, help="Offset samples per decade")
    p.add_argument("--min-offset", type=float, default=1e-5, help="Smallest offset, normalized units")
    p.add_argument("--max-offset", type=float, default=0.5, help="Largest offset, normalized units")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    plate = SixParamPlate(sample_rate=SAMPLE_RATE, device=device, dtype=dtype, drop_sub_20hz_modes=False)
    box = NormBox(SHAPE_KEYS, device=device, dtype=dtype)
    loss_fns = {name: select_loss_function(name, sample_rate=SAMPLE_RATE, device=device) for name in args.losses}

    offsets = build_offsets(args.n_per_decade, args.min_offset, args.max_offset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  losses={args.losses}  {len(offsets)} offsets per coordinate\n")

    profiles: List[Dict] = []
    summary: List[Dict] = []

    for rid in args.ids:
        gt6 = seven_to_six(_read_params_csv(args.dataset_dir / f"random_IR_params_{rid}.csv"))
        target_np = load_target_ir_from_npz(
            args.dataset_dir / f"random_IR_{rid}.npz", args.duration, SAMPLE_RATE
        )
        target = torch.as_tensor(target_np, device=device, dtype=dtype).unsqueeze(0)
        duration = target_np.shape[0] / float(SAMPLE_RATE)
        z_gt = box.to_unit_np(np.array([gt6[k] for k in SHAPE_KEYS]))
        mu_ref_scalar = float(np.sqrt(MU_MIN * MU_MAX))

        for j, key in enumerate(SHAPE_KEYS):
            # Offsets that would leave the box are dropped, not clipped, so the
            # monotonic walk never sees an artificial plateau at the boundary.
            z_vals = z_gt[j] + offsets
            keep = (z_vals >= 0.0) & (z_vals <= 1.0)
            z_vals, off = z_vals[keep], offsets[keep]

            z = torch.as_tensor(np.tile(z_gt, (len(z_vals), 1)), device=device, dtype=dtype)
            z[:, j] = torch.as_tensor(z_vals, device=device, dtype=dtype)
            mu_b = torch.full((len(z_vals),), mu_ref_scalar, device=device, dtype=dtype)
            six = torch.cat([mu_b.unsqueeze(1), box.to_physical(z)], dim=1)
            with torch.no_grad():
                pred = plate(six, duration=duration, normalize=False)

            for name, fn in loss_fns.items():
                # mu is profiled out per loss, so this is the objective a fitter
                # with an exact mu solve actually descends.
                _, lv = solve_mu_by_scale(pred, mu_b, target, fn, 40, dtype)
                lv = lv.cpu().numpy().astype(np.float64)

                hw = descent_halfwidth(off, lv)
                i_best = int(np.argmin(lv))
                summary.append(
                    {
                        "id": rid,
                        "loss": name,
                        "coord": key,
                        "z_gt": float(z_gt[j]),
                        "argmin_offset": float(off[i_best]),
                        "loss_at_gt": float(lv[int(np.argmin(np.abs(off)))]),
                        "loss_at_argmin": float(lv[i_best]),
                        "basin_halfwidth": hw,
                    }
                )
                for o, v in zip(off, lv):
                    profiles.append({"id": rid, "loss": name, "coord": key, "offset": float(o), "value": float(v)})

    df = pd.DataFrame(summary)
    df.to_csv(args.output_dir / "basin_summary.csv", index=False)
    pd.DataFrame(profiles).to_csv(args.output_dir / "basin_profiles.csv", index=False)

    print(f"{'loss':<18} {'median basin':>13} {'median |argmin off|':>21}")
    for name in args.losses:
        sub = df[df["loss"] == name]
        if sub.empty:
            continue
        print(f"{name:<18} {sub['basin_halfwidth'].median():>13.2e} {sub['argmin_offset'].abs().median():>21.2e}")

    # One panel per coordinate; each loss is min-max scaled so shapes are
    # comparable even though their absolute magnitudes differ by decades.
    fig, axes = plt.subplots(1, len(SHAPE_KEYS), figsize=(4 * len(SHAPE_KEYS), 4), squeeze=False)
    pdf = pd.DataFrame(profiles)
    for ax, key in zip(axes[0], SHAPE_KEYS):
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
        ax.set_xlabel("offset from GT (normalized)")
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("min-max scaled loss")
    axes[0][-1].legend(fontsize=7)
    plt.suptitle(f"Loss profiles about ground truth (IR {args.ids[0]})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.output_dir / "basin_profiles.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote basin_summary.csv, basin_profiles.csv, basin_profiles.png to {args.output_dir}")


if __name__ == "__main__":
    main()
