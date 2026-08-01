"""Stage-2 μ-refinement via ternary search.

Run from repo root:
    python ternary_mu_refine.py
or with overrides:
    python ternary_mu_refine.py --cmaes_results results/cmaes_norm_es/l1_stft --device cpu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.loss.loss_selector import select_loss_function
from src.plate.SixParamPlate import BatchedModalPlateTorch as SixParamPlate

SAMPLE_RATE = 44100
NU = 0.25

RHO_MIN, RHO_MAX = 2430.0, 21230.0
H_MIN, H_MAX = 0.001, 0.005
MU_MIN = RHO_MIN * H_MIN  # 2.43
MU_MAX = RHO_MAX * H_MAX  # 106.15

COMPOSITE_KEYS = ["mu", "D_div_mu", "T0_div_mu", "Ly", "op_x", "op_y"]
COMPOSITE_BOUNDS = {
    "mu": (2.43, 106.15),
    "D_div_mu": (0.28052546, 201.188843),
    "T0_div_mu": (0.000094206, 411.522634),
    "Ly": (1.1, 4.0),
    "op_x": (0.51, 1.0),
    "op_y": (0.51, 1.0),
}

FIXED_PLATE_PARAMS = dict(
    Lx=1.0, T60_DC=6.0, T60_F1=2.0, loss_F1=500.0, fp_x=0.335, fp_y=0.467,
)


def load_target_ir_from_npz(npz_path: Path, duration_s: float, expected_sr: int) -> np.ndarray:
    """No peak normalization — μ would be lost otherwise."""
    with np.load(npz_path) as data:
        if "ir" not in data.files:
            raise KeyError(f"Expected key 'ir' in {npz_path}, found keys: {data.files}")
        ir = np.asarray(data["ir"])
        sr = int(np.asarray(data["sample_rate"]).item()) if "sample_rate" in data else expected_sr

    if sr != expected_sr:
        raise ValueError(f"Sample rate mismatch in {npz_path.name}: {sr} (expected {expected_sr})")

    if ir.ndim > 1:
        ir = np.mean(ir, axis=-1)
    ir = ir.astype(np.float64, copy=False)
    ir = ir[: int(duration_s * expected_sr)]
    return ir


def seven_to_six(p7: Dict[str, float]) -> Dict[str, float]:
    rho, h, E, T0 = float(p7["rho"]), float(p7["h"]), float(p7["E"]), float(p7["T0"])
    mu = rho * h
    D = E * (h ** 3) / (12.0 * (1.0 - NU ** 2))
    return {
        "mu": mu,
        "D_div_mu": D / mu,
        "T0_div_mu": T0 / mu,
        "Ly": float(p7["Ly"]),
        "op_x": float(p7["op_x"]),
        "op_y": float(p7["op_y"]),
    }


def nmse_6d(est_6: Dict[str, float], gt_6: Dict[str, float]) -> float:
    errs = []
    for key in COMPOSITE_KEYS:
        lo, hi = COMPOSITE_BOUNDS[key]
        e = (est_6[key] - lo) / (hi - lo)
        g = (gt_6[key] - lo) / (hi - lo)
        errs.append((e - g) ** 2)
    return float(np.mean(errs))


def evaluate_at_mu(
    mu_values: np.ndarray,
    fixed_5: Dict[str, float],
    target_t: torch.Tensor,
    duration: float,
    synth: SixParamPlate,
    loss_fn,
    device: str,
    dtype: torch.dtype,
) -> np.ndarray:
    mu_arr = np.clip(np.asarray(mu_values, dtype=np.float64), MU_MIN, MU_MAX)
    six = np.tile(
        np.array(
            [fixed_5["mu"], fixed_5["D_div_mu"], fixed_5["T0_div_mu"],
             fixed_5["Ly"], fixed_5["op_x"], fixed_5["op_y"]],
            dtype=np.float64,
        ),
        (len(mu_arr), 1),
    )
    six[:, 0] = mu_arr
    six_t = torch.tensor(six, dtype=dtype, device=device)
    with torch.no_grad():
        pred = synth(six_t, duration, normalize=False)
        losses = loss_fn(target_t.expand(len(mu_arr), -1), pred).detach().cpu().numpy()
    return np.nan_to_num(losses, nan=1e6, posinf=1e6, neginf=1e6)


def ternary_search(
    fixed_5: Dict[str, float],
    target_t: torch.Tensor,
    duration: float,
    synth: SixParamPlate,
    loss_fn,
    device: str,
    dtype: torch.dtype,
    evaluate_fn=evaluate_at_mu,
    lo: float = MU_MIN,
    hi: float = MU_MAX,
    n_iters: int = 50,
) -> Tuple[float, float]:
    for _ in range(n_iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        losses = evaluate_fn(
            np.array([m1, m2]), fixed_5, target_t, duration, synth, loss_fn, device, dtype
        )
        if losses[0] < losses[1]:
            hi = m2
        else:
            lo = m1

    best_mu = 0.5 * (lo + hi)
    best_loss = float(
        evaluate_fn(np.array([best_mu]), fixed_5, target_t, duration, synth, loss_fn, device, dtype)[0]
    )
    return best_mu, best_loss


def grid_sanity_check(
    fixed_5: Dict[str, float],
    target_t: torch.Tensor,
    duration: float,
    synth: SixParamPlate,
    loss_fn,
    device: str,
    dtype: torch.dtype,
    evaluate_fn=evaluate_at_mu,
    n_grid: int = 20,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    mus = np.linspace(MU_MIN, MU_MAX, n_grid)
    losses = evaluate_fn(mus, fixed_5, target_t, duration, synth, loss_fn, device, dtype)
    n_local_min = 0
    for i in range(1, len(losses) - 1):
        if losses[i] < losses[i - 1] and losses[i] < losses[i + 1]:
            n_local_min += 1
    return (n_local_min <= 1), mus, losses


def plot_mu_error_histogram(df: pd.DataFrame, out_path: Path):
    """Histogram of |mu - mu_GT| before vs after refinement, log-scale x-axis.
    Both distributions overlaid so the rightward-then-leftward shift is visible.
    """
    valid = df.dropna(subset=["mu_orig_err", "mu_refined_err"])
    if len(valid) == 0:
        return

    err_orig = np.abs(valid["mu_orig_err"].values)
    err_refn = np.abs(valid["mu_refined_err"].values)

    # Floor very small values so log binning works
    floor = 1e-6
    err_orig = np.clip(err_orig, floor, None)
    err_refn = np.clip(err_refn, floor, None)

    lo = min(err_orig.min(), err_refn.min())
    hi = max(err_orig.max(), err_refn.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 30)

    median_orig = float(np.median(err_orig))
    median_refn = float(np.median(err_refn))
    mean_orig = float(np.mean(err_orig))
    mean_refn = float(np.mean(err_refn))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(err_orig, bins=bins, alpha=0.55, color="#d62728",
            edgecolor="white", linewidth=0.5,
            label=f"before  (median {median_orig:.3g}, mean {mean_orig:.3g})")
    ax.hist(err_refn, bins=bins, alpha=0.55, color="#2ca02c",
            edgecolor="white", linewidth=0.5,
            label=f"after   (median {median_refn:.3g}, mean {mean_refn:.3g})")

    ax.axvline(median_orig, color="#d62728", lw=1.0, ls="--", alpha=0.8)
    ax.axvline(median_refn, color="#2ca02c", lw=1.0, ls="--", alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("|mu - mu_GT|  (log scale)")
    ax.set_ylabel("# of IRs")
    ax.set_title(
        f"Absolute mu error before vs after refinement (n={len(valid)})\n"
        f"median improved by {median_orig / max(median_refn, 1e-12):.0f}x"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def run(args):
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print("=" * 70)
    print(f"  TERNARY-SEARCH μ-REFINEMENT on {device.upper()} ({args.dtype})")
    print("=" * 70)
    print(f"  CMA-ES results: {args.cmaes_results}")
    print(f"  Targets: {args.dset_root}")
    print(f"  Loss: {args.loss}, Duration: {args.duration}s")
    print(f"  μ search bounds: [{MU_MIN:.3f}, {MU_MAX:.3f}]")
    print(f"  Ternary iterations: {args.n_iters}  (~{args.n_iters * 2 + 1} evals/IR)")
    print(f"  Sanity grid check: {'on' if args.sanity_grid else 'off'}")
    print()

    cmaes_dir = Path(args.cmaes_results)
    target_dir = Path(args.dset_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = cmaes_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.csv at {summary_path}")
    cmaes = pd.read_csv(summary_path)
    print(f"  Loaded {len(cmaes)} CMA-ES results")

    synth = SixParamPlate(
        sample_rate=SAMPLE_RATE,
        device=device,
        dtype=dtype,
        **FIXED_PLATE_PARAMS,
    )
    loss_fn = select_loss_function(args.loss, sample_rate=SAMPLE_RATE, device=device)

    rows = []
    sanity_failures = []
    t_start = time.time()

    for idx, row in cmaes.iterrows():
        fname = row["filename"]
        npz_path = target_dir / fname
        if not npz_path.exists():
            print(f"  [{idx + 1}/{len(cmaes)}] SKIP {fname} (no npz at {npz_path})")
            continue

        p7_est = {k: float(row[k]) for k in ["E", "rho", "h", "Ly", "T0", "op_x", "op_y"]}
        p6_est = seven_to_six(p7_est)

        p7_gt = None
        gt_csv_path = target_dir / fname.replace("random_IR_", "random_IR_params_").replace(".npz", ".csv")
        if gt_csv_path.exists():
            try:
                gt_df = pd.read_csv(gt_csv_path)
                p7_gt = {k: float(gt_df.iloc[0][k]) for k in ["E", "rho", "h", "Ly", "T0", "op_x", "op_y"]}
                p6_gt = seven_to_six(p7_gt)
            except Exception:
                p7_gt = None
                p6_gt = None
        else:
            p6_gt = None

        target_ir = load_target_ir_from_npz(npz_path, args.duration, SAMPLE_RATE)
        target_t = torch.tensor(target_ir, dtype=dtype, device=device).unsqueeze(0)
        plate_eval_count = 0

        def evaluate_counted(mu_values, fixed_5, target_t, duration, synth, loss_fn, device, dtype):
            nonlocal plate_eval_count
            mu_arr = np.asarray(mu_values)
            plate_eval_count += int(mu_arr.size)
            return evaluate_at_mu(mu_values, fixed_5, target_t, duration, synth, loss_fn, device, dtype)

        sanity_ok = True
        sanity_grid_min_mu = None
        if args.sanity_grid:
            sanity_ok, mus, grid_losses = grid_sanity_check(
                p6_est, target_t, args.duration, synth, loss_fn, device, dtype,
                evaluate_fn=evaluate_counted,
                n_grid=args.sanity_n,
            )
            sanity_grid_min_mu = float(mus[int(np.argmin(grid_losses))])
            if not sanity_ok:
                sanity_failures.append(fname)

        t0 = time.time()
        best_mu, best_loss = ternary_search(
            p6_est, target_t, args.duration, synth, loss_fn, device, dtype,
            evaluate_fn=evaluate_counted,
            n_iters=args.n_iters,
        )
        elapsed = time.time() - t0

        cmaes_loss = float(evaluate_counted(
            np.array([p6_est["mu"]]), p6_est, target_t, args.duration,
            synth, loss_fn, device, dtype,
        )[0])

        p6_refined = {**p6_est, "mu": best_mu}
        nmse_orig = nmse_6d(p6_est, p6_gt) if p6_gt else None
        nmse_refined = nmse_6d(p6_refined, p6_gt) if p6_gt else None
        nmse_summary_orig = float(row["nmse"]) if "nmse" in row and row["nmse"] is not None else None

        rec = {
            "filename": fname,
            # Raw 7-parameter estimates from stage 1 (CMA-ES)
            **{f"est_{k}": p7_est[k] for k in ["E", "rho", "h", "Ly", "T0", "op_x", "op_y"]},
            # Raw 7-parameter ground-truth from dataset CSV (if available)
            **({f"gt_{k}": p7_gt[k] for k in ["E", "rho", "h", "Ly", "T0", "op_x", "op_y"]} if p7_gt else {}),
            # Composite 6-parameter estimates before/after mu refinement
            "est_mu": p6_est["mu"],
            "est_D_div_mu": p6_est["D_div_mu"],
            "est_T0_div_mu": p6_est["T0_div_mu"],
            "est_Ly": p6_est["Ly"],
            "est_op_x": p6_est["op_x"],
            "est_op_y": p6_est["op_y"],
            "refined_mu": p6_refined["mu"],
            "refined_D_div_mu": p6_refined["D_div_mu"],
            "refined_T0_div_mu": p6_refined["T0_div_mu"],
            "refined_Ly": p6_refined["Ly"],
            "refined_op_x": p6_refined["op_x"],
            "refined_op_y": p6_refined["op_y"],
            # Composite 6-parameter ground-truth (if available)
            "gt_mu": p6_gt["mu"] if p6_gt else None,
            "gt_D_div_mu": p6_gt["D_div_mu"] if p6_gt else None,
            "gt_T0_div_mu": p6_gt["T0_div_mu"] if p6_gt else None,
            "gt_Ly": p6_gt["Ly"] if p6_gt else None,
            "gt_op_x": p6_gt["op_x"] if p6_gt else None,
            "gt_op_y": p6_gt["op_y"] if p6_gt else None,
            "mu_orig": p6_est["mu"],
            "mu_refined": best_mu,
            "mu_gt": p6_gt["mu"] if p6_gt else None,
            "mu_orig_err": (p6_est["mu"] - p6_gt["mu"]) if p6_gt else None,
            "mu_refined_err": (best_mu - p6_gt["mu"]) if p6_gt else None,
            "loss_orig": cmaes_loss,
            "loss_refined": best_loss,
            "loss_delta": best_loss - cmaes_loss,
            "nmse_orig": nmse_orig,
            "nmse_refined": nmse_refined,
            "nmse_delta": (nmse_refined - nmse_orig) if (nmse_orig is not None) else None,
            "nmse_summary_orig": nmse_summary_orig,
            "sanity_ok": sanity_ok,
            "sanity_grid_min_mu": sanity_grid_min_mu,
            "total_evaluations": int(plate_eval_count),
            "elapsed_s": round(elapsed, 3),
        }
        rows.append(rec)

        if (idx + 1) % 10 == 0 or idx < 5:
            if p6_gt:
                print(
                    f"  [{idx + 1:3d}/{len(cmaes)}] {fname:30s} "
                    f"mu {p6_est['mu']:7.3f} -> {best_mu:7.3f} (GT {p6_gt['mu']:7.3f})"
                )
            else:
                print(f"  [{idx + 1:3d}/{len(cmaes)}] {fname:30s} mu {p6_est['mu']:7.3f} -> {best_mu:7.3f}")
            if nmse_orig is not None:
                pct = 100.0 * (nmse_refined - nmse_orig) / max(nmse_orig, 1e-15)
                print(
                    f"          NMSE {nmse_orig:.6f} -> {nmse_refined:.6f} ({pct:+.1f}%)  "
                    f"loss {cmaes_loss:.6f} -> {best_loss:.6f}  ({elapsed:.1f}s)"
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "mu_refined_summary.csv", index=False)
    plot_mu_error_histogram(df, out_dir / "mu_error_histogram.png")

    print()
    print("=" * 70)
    print("  AGGREGATE")
    print("=" * 70)
    if "nmse_orig" in df and df["nmse_orig"].notna().any():
        print(f"  Mean NMSE   orig: {df['nmse_orig'].mean():.6f}")
        print(f"              refn: {df['nmse_refined'].mean():.6f}")
        print(f"  Median NMSE orig: {df['nmse_orig'].median():.6f}")
        print(f"              refn: {df['nmse_refined'].median():.6f}")
        improved = (df["nmse_refined"] < df["nmse_orig"]).sum()
        worsened = (df["nmse_refined"] > df["nmse_orig"]).sum()
        unchanged = len(df) - improved - worsened
        print(f"  IRs improved: {improved} / {len(df)}  worsened: {worsened}  unchanged: {unchanged}")
    print(f"  Mean elapsed/IR:    {df['elapsed_s'].mean():.2f} s")
    if sanity_failures:
        print(f"  WARN: Sanity check (multi-modal loss-vs-mu) flagged on: {len(sanity_failures)} IRs")
        for f in sanity_failures[:5]:
            print(f"      {f}")
    print(f"  Total runtime:      {time.time() - t_start:.1f} s")
    print(f"  Wrote: {out_dir / 'mu_refined_summary.csv'}")


def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cmaes_results", type=str, default="results/cmaes_norm_es/l1_stft")
    p.add_argument("--dset_root", type=str, default="data/random-IR-100-1.0s")
    p.add_argument("--output_dir", type=str, default="results/mu_refinement/ternary")
    p.add_argument("--loss", type=str, default="L1_STFT")
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--n_iters", type=int, default=50)
    p.add_argument("--sanity_grid", action="store_true", default=True)
    p.add_argument("--sanity_n", type=int, default=20)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
