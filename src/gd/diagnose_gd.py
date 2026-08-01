"""Decide whether gradient descent can work on this objective at all.

Three checks, cheapest and most decisive first:

  1. Is the mu-rescaling identity y(mu) = y(mu_ref)*(mu_ref/mu) actually exact?
     The whole "profile mu out" design in graddescent.py rests on it.

  2. Given the *ground-truth* shape, does solve_mu_by_scale recover the true mu?
     This isolates the mu solve from shape convergence, which the end-to-end
     mu_rel_error reported by graddescent.py does not.

  3. Along each shape coordinate, with the others held at ground truth, is the
     loss actually minimized at the true value, and how wide is the basin?
     If the minimum is elsewhere, or the basin is only a few percent of the
     range wide, no amount of Adam budget will find it from a random start and
     the problem is the objective, not the optimizer.

Usage:
    python -m src.gd.diagnose_gd --dataset-dir random-IR-200-0.2s --id 0001
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.gd.graddescent import (
    LOG_KEYS,
    SAMPLE_RATE,
    SHAPE_KEYS,
    NormBox,
    _read_params_csv,
    solve_mu_by_scale,
)
from src.loss.loss_selector import select_loss_function
from src.mu_optimization.ternary_mu import (
    COMPOSITE_BOUNDS,
    MU_MAX,
    MU_MIN,
    load_target_ir_from_npz,
    seven_to_six,
)
from src.plate.SixParamPlate import BatchedModalPlateTorch as SixParamPlate


def synth(plate, mu, shape, duration):
    """mu: [K] tensor, shape: [K,5] tensor -> [K,T]."""
    six = torch.cat([mu.unsqueeze(1), shape], dim=1)
    with torch.no_grad():
        return plate(six, duration=duration, normalize=False)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, default=Path("random-IR-200-0.2s"))
    p.add_argument("--id", type=str, default="0001")
    p.add_argument("--loss", type=str, default="L1_STFT")
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--n-sweep", type=int, default=81, help="Points per 1-D sweep")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    loss_fn = select_loss_function(args.loss, sample_rate=SAMPLE_RATE, device=device)
    plate = SixParamPlate(sample_rate=SAMPLE_RATE, device=device, dtype=dtype, drop_sub_20hz_modes=False)
    box = NormBox(SHAPE_KEYS, device=device, dtype=dtype)

    d = args.dataset_dir
    gt6 = seven_to_six(_read_params_csv(d / f"random_IR_params_{args.id}.csv"))
    target_np = load_target_ir_from_npz(d / f"random_IR_{args.id}.npz", args.duration, SAMPLE_RATE)
    target = torch.as_tensor(target_np, device=device, dtype=dtype).unsqueeze(0)
    duration = target_np.shape[0] / float(SAMPLE_RATE)

    z_gt = torch.as_tensor(
        box.to_unit_np(np.array([gt6[k] for k in SHAPE_KEYS])), device=device, dtype=dtype
    ).unsqueeze(0)
    shape_gt = box.to_physical(z_gt)
    mu_gt = float(gt6["mu"])

    print(f"IR {args.id}   loss={args.loss}   device={device}")
    print(f"GT: mu={mu_gt:.6g}  " + "  ".join(f"{k}={gt6[k]:.6g}" for k in SHAPE_KEYS))
    print(f"target peak |amp| = {np.abs(target_np).max():.3e}\n")

    # --- 1. mu rescaling identity ------------------------------------------
    m1, m2 = mu_gt, mu_gt * 3.7
    y1 = synth(plate, torch.tensor([m1], device=device, dtype=dtype), shape_gt, duration)
    y2 = synth(plate, torch.tensor([m2], device=device, dtype=dtype), shape_gt, duration)
    a, b = y1 * m1, y2 * m2
    rel = float((a - b).abs().max() / a.abs().max())
    print(f"[1] mu identity: max rel deviation of y*mu across mu={m1:.4g} vs {m2:.4g}: {rel:.3e}")
    print(f"    {'PASS' if rel < 1e-4 else 'FAIL -- profiling mu out is invalid'}\n")

    # --- 2. mu recovery at the true shape ----------------------------------
    mu_ref = torch.tensor([float(np.sqrt(MU_MIN * MU_MAX))], device=device, dtype=dtype)
    ref = synth(plate, mu_ref, shape_gt, duration)
    mu_est, _ = solve_mu_by_scale(ref, mu_ref, target, loss_fn, 40, dtype)
    err = abs(float(mu_est[0]) - mu_gt) / mu_gt
    print(f"[2] mu solve at GT shape: est={float(mu_est[0]):.6g}  true={mu_gt:.6g}  rel err={err:.3e}")
    print(f"    {'PASS' if err < 1e-3 else 'FAIL -- the 1-D mu search is broken'}\n")

    # --- 3. per-coordinate loss profile through the true optimum -----------
    print("[3] 1-D loss profiles (all other coordinates held at ground truth,")
    print("    mu profiled out at every point, so this is the real objective):\n")
    print(f"    {'coord':<12} {'loss@GT':>11} {'best loss':>11} {'argmin z':>9} {'z_GT':>7} {'basin':>7}")

    n = args.n_sweep
    for j, key in enumerate(SHAPE_KEYS):
        z = z_gt.repeat(n, 1).clone()
        z[:, j] = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
        shape = box.to_physical(z)
        mu_b = mu_ref.repeat(n)
        pred = synth(plate, mu_b, shape, duration)
        _, losses = solve_mu_by_scale(pred, mu_b, target, loss_fn, 40, dtype)
        losses = losses.cpu().numpy()

        zg = float(z_gt[0, j])
        i_gt = int(round(zg * (n - 1)))
        i_best = int(np.argmin(losses))
        l_gt, l_best = float(losses[i_gt]), float(losses[i_best])

        # Width of the contiguous region around the true optimum that is within
        # 10% of the way from the global min to the median loss -- i.e. how much
        # of the range a random start could land in and still descend to GT.
        thresh = l_best + 0.1 * (float(np.median(losses)) - l_best)
        lo = hi = i_best
        while lo > 0 and losses[lo - 1] <= thresh:
            lo -= 1
        while hi < n - 1 and losses[hi + 1] <= thresh:
            hi += 1
        basin = (hi - lo) / (n - 1)

        flag = "" if abs(i_best - i_gt) <= 1 else "   <-- min NOT at GT"
        print(
            f"    {key:<12} {l_gt:>11.4e} {l_best:>11.4e} "
            f"{i_best/(n-1):>9.3f} {zg:>7.3f} {basin:>7.1%}{flag}"
        )

    print(
        "\n    A basin of a few percent means a uniformly random start has that "
        "\n    probability of being in the right valley; with 5 coordinates the "
        "\n    joint probability is roughly the product."
    )


if __name__ == "__main__":
    main()
