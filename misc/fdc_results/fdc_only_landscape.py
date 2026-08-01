"""
FDC-Only Landscape Analysis — Slim Companion to ela_landscape.py
=================================================================

Computes classical (non-binned) Fitness-Distance Correlation for the same
losses, target IRs, and random samples as the full ELA script, in both
7D physical-parameter space and 6D composite-parameter space, using both
Pearson and Spearman correlations.

Why: the full ELA script computes quantile-binned Spearman FDC, which is
hard to compare directly against monotonicity as a single per-loss number.
This script gives one classical FDC value per (loss, distance-space, method)
for clean head-to-head plotting.

Reuses (verbatim) from ela_landscape.py:
  - Constants, bounds, FIXED_PLATE_PARAMS
  - physical_to_plate14, normalize_7d, normalize_6d, to_composite_np
  - generate_sobol_targets (Sobol seed=42, same target IRs)
  - synthesize_batch, compute_losses_batch, sample_and_evaluate
  - np.random.seed(t_idx * 1000) — identical sample positions per IR

Output:
  fdc_results/fdc_summary.csv
    columns: loss, fdc_pearson_7d, fdc_pearson_6d, fdc_spearman_7d, fdc_spearman_6d

Usage:
  python fdc_only_landscape.py --n_targets 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats.qmc import Sobol

import torch

from src.loss.loss_selector import select_loss_function
from src.plate.SevenParamPlate import BatchedModalPlateTorch

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS (verbatim from ela_landscape.py)
# ═══════════════════════════════════════════════════════════════════

SAMPLE_RATE = 44100
NU = 0.25

PARAM_KEYS = ["E", "rho", "h", "Ly", "T0", "op_x", "op_y"]
PARAM_BOUNDS = {
    "E": (6.7e10, 2.2e11), "rho": (2430.0, 21230.0), "h": (0.001, 0.005),
    "Ly": (1.1, 4.0), "T0": (0.01, 1000.0), "op_x": (0.51, 1.0), "op_y": (0.51, 1.0),
}
BOUNDS_LO = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS])
BOUNDS_HI = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS])
BOUNDS_RANGE = BOUNDS_HI - BOUNDS_LO

COMPOSITE_KEYS = ["mu", "D_mu", "T0_mu", "Ly", "op_x", "op_y"]
COMPOSITE_BOUNDS = {
    "mu": (2.43, 106.15), "D_mu": (0.28052546, 201.188843),
    "T0_mu": (0.000094206, 411.522634), "Ly": (1.1, 4.0),
    "op_x": (0.51, 1.0), "op_y": (0.51, 1.0),
}

FIXED_PLATE_PARAMS = {
    "Lx": 1.0, "nu": 0.25, "T60_DC": 6.0, "T60_F1": 2.0,
    "loss_F1": 500.0, "fp_x": 0.335, "fp_y": 0.467,
}


# ═══════════════════════════════════════════════════════════════════
# HELPERS (verbatim from ela_landscape.py)
# ═══════════════════════════════════════════════════════════════════

def physical_to_plate14(phys_batch_np, device):
    E, rho, h, Ly, T0, op_x, op_y = [phys_batch_np[:, i] for i in range(7)]
    cols = np.stack([
        np.full_like(E, FIXED_PLATE_PARAMS["Lx"]), Ly, h, T0, rho, E,
        np.full_like(E, FIXED_PLATE_PARAMS["nu"]),
        np.full_like(E, FIXED_PLATE_PARAMS["T60_DC"]),
        np.full_like(E, FIXED_PLATE_PARAMS["T60_F1"]),
        np.full_like(E, FIXED_PLATE_PARAMS["loss_F1"]),
        np.full_like(E, FIXED_PLATE_PARAMS["fp_x"]),
        np.full_like(E, FIXED_PLATE_PARAMS["fp_y"]),
        op_x, op_y,
    ], axis=1)
    return torch.tensor(cols, dtype=torch.float32, device=device)


def normalize_7d(params, gt):
    p_norm = (params - BOUNDS_LO) / BOUNDS_RANGE
    g_norm = (gt - BOUNDS_LO) / BOUNDS_RANGE
    return np.sqrt(np.sum((p_norm - g_norm) ** 2, axis=-1))


def to_composite_np(params_7d):
    if params_7d.ndim == 1:
        params_7d = params_7d.reshape(1, -1)
    E, rho, h, Ly, T0, op_x, op_y = [params_7d[:, i] for i in range(7)]
    mu = rho * h
    D = (E * h ** 3) / (12 * (1 - NU ** 2))
    return np.stack([mu, D / mu, T0 / mu, Ly, op_x, op_y], axis=1)


def normalize_6d(params_7d, gt_7d):
    comp = to_composite_np(params_7d)
    comp_gt = to_composite_np(gt_7d)
    blo = np.array([COMPOSITE_BOUNDS[k][0] for k in COMPOSITE_KEYS])
    bhi = np.array([COMPOSITE_BOUNDS[k][1] for k in COMPOSITE_KEYS])
    p_norm = (comp - blo) / (bhi - blo)
    g_norm = (comp_gt - blo) / (bhi - blo)
    return np.sqrt(np.sum((p_norm - g_norm) ** 2, axis=-1))


def generate_sobol_targets(n_targets, seed=42):
    sampler = Sobol(d=7, scramble=True, seed=seed)
    n_pow2 = 1
    while n_pow2 < n_targets + 1:
        n_pow2 *= 2
    raw = sampler.random(n_pow2)
    raw = np.clip(raw[1: n_targets + 1], 0.05, 0.95)
    return BOUNDS_LO + raw * BOUNDS_RANGE


def synthesize_batch(synth, params_7d_np, duration, device):
    plate14 = physical_to_plate14(params_7d_np, device)
    with torch.no_grad():
        return synth(plate14, duration)


def compute_losses_batch(loss_fns, target_t, candidate_batch):
    B = candidate_batch.shape[0]
    target_exp = target_t.expand(B, -1)
    results = {}
    for name, fn in loss_fns.items():
        try:
            val = fn(target_exp, candidate_batch).detach().cpu().numpy()
            if not np.any(np.isnan(val)):
                results[name] = val
        except Exception:
            pass
    return results


def sample_and_evaluate(synth, loss_fns, gt_params, target_ir_t, duration, device,
                        n_samples, batch_size):
    """Shared sampling: returns dist_7d, dist_6d, all_losses dict."""
    gt = gt_params.reshape(1, -1)
    samples = BOUNDS_LO + np.random.rand(n_samples, 7) * BOUNDS_RANGE
    dist_7d = normalize_7d(samples, gt)
    dist_6d = normalize_6d(samples, gt)

    all_losses = {name: [] for name in loss_fns}
    for i in range(0, n_samples, batch_size):
        batch = samples[i: i + batch_size]
        audios = synthesize_batch(synth, batch, duration, device)
        bl = compute_losses_batch(loss_fns, target_ir_t, audios)
        for name, vals in bl.items():
            all_losses[name].append(vals)

    all_losses = {name: np.concatenate(v) for name, v in all_losses.items() if v}
    return dist_7d, dist_6d, all_losses


# ═══════════════════════════════════════════════════════════════════
# CLASSICAL (NON-BINNED) FDC
# ═══════════════════════════════════════════════════════════════════

def measure_classical_fdc(dist_7d, dist_6d, all_losses):
    """For each loss, compute Pearson and Spearman correlation of (distance, loss)
    in both 7D and 6D distance spaces, across all samples (no binning).
    Returns: {name: {pearson_7d, pearson_6d, spearman_7d, spearman_6d}}."""
    results = {}
    for name, loss_vals in all_losses.items():
        # Filter NaN/inf defensively
        ok = np.isfinite(loss_vals) & np.isfinite(dist_7d) & np.isfinite(dist_6d)
        if ok.sum() < 20:
            results[name] = {k: np.nan for k in
                             ("pearson_7d", "pearson_6d", "spearman_7d", "spearman_6d")}
            continue
        lv = loss_vals[ok]
        d7 = dist_7d[ok]
        d6 = dist_6d[ok]
        p7, _ = scipy_stats.pearsonr(d7, lv)
        p6, _ = scipy_stats.pearsonr(d6, lv)
        s7, _ = scipy_stats.spearmanr(d7, lv)
        s6, _ = scipy_stats.spearmanr(d6, lv)
        results[name] = {
            "pearson_7d": p7, "pearson_6d": p6,
            "spearman_7d": s7, "spearman_6d": s6,
        }
    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_targets", type=int, default=100)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="fdc_results")
    p.add_argument("--losses", nargs="+", default=[
        "L1_STFT", "ESR", "GroupDelay", "ComplexSTFT", "Gammatone",
        "CQT+LogDec", "SC+LogMag", "VQT", "Envelope", "InstFreq",
        "MSS", "Dispersion", "CQT_L1", "Mel",
    ])
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FDC-ONLY LANDSCAPE ANALYSIS")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Targets: {args.n_targets}")
    print(f"  Samples/target: {args.n_samples}")
    print(f"  Duration: {args.duration}s")
    print(f"  Losses: {args.losses}")

    synth = BatchedModalPlateTorch(sample_rate=SAMPLE_RATE, device=device)

    loss_fns = {}
    for name in args.losses:
        try:
            fn = select_loss_function(name, sample_rate=SAMPLE_RATE, device=device)
            loss_fns[name] = fn
            print(f"    Loaded: {name}")
        except Exception as e:
            print(f"    FAILED: {name} ({e})")

    if not loss_fns:
        print("No losses loaded.")
        return

    print(f"\n  Generating {args.n_targets} target IRs (Sobol seed=42)...")
    gt_params_all = generate_sobol_targets(args.n_targets, seed=42)
    target_irs = []
    for gt in gt_params_all:
        ir = synthesize_batch(synth, gt.reshape(1, -1), args.duration, device)
        target_irs.append(ir.squeeze(0))
    print(f"  Done.")

    # Accumulators: list of per-IR values to be aggregated (median) per loss
    all_pearson_7d = {name: [] for name in loss_fns}
    all_pearson_6d = {name: [] for name in loss_fns}
    all_spearman_7d = {name: [] for name in loss_fns}
    all_spearman_6d = {name: [] for name in loss_fns}

    t_start = time.time()

    for t_idx in range(args.n_targets):
        gt = gt_params_all[t_idx]
        target_t = target_irs[t_idx].unsqueeze(0)

        # IDENTICAL seed to ela_landscape.py — same 2000 samples per IR.
        np.random.seed(t_idx * 1000)
        dist_7d, dist_6d, all_losses = sample_and_evaluate(
            synth, loss_fns, gt, target_t, args.duration, device,
            args.n_samples, args.batch_size
        )

        per_ir = measure_classical_fdc(dist_7d, dist_6d, all_losses)
        for name, vals in per_ir.items():
            all_pearson_7d[name].append(vals["pearson_7d"])
            all_pearson_6d[name].append(vals["pearson_6d"])
            all_spearman_7d[name].append(vals["spearman_7d"])
            all_spearman_6d[name].append(vals["spearman_6d"])

        elapsed = time.time() - t_start
        per_target = elapsed / (t_idx + 1)
        remaining = per_target * (args.n_targets - t_idx - 1)
        print(f"  Target {t_idx + 1}/{args.n_targets} done. "
              f"Elapsed {elapsed:.0f}s, ~{remaining:.0f}s remaining.")

    # ═══════════════════════════════════════════════════════════════
    # AGGREGATE
    # ═══════════════════════════════════════════════════════════════

    rows = []
    for name in loss_fns:
        rows.append({
            "loss": name,
            "fdc_pearson_7d":  np.nanmedian(all_pearson_7d[name])  if all_pearson_7d[name]  else np.nan,
            "fdc_pearson_6d":  np.nanmedian(all_pearson_6d[name])  if all_pearson_6d[name]  else np.nan,
            "fdc_spearman_7d": np.nanmedian(all_spearman_7d[name]) if all_spearman_7d[name] else np.nan,
            "fdc_spearman_6d": np.nanmedian(all_spearman_6d[name]) if all_spearman_6d[name] else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("fdc_pearson_7d", ascending=False)
    out_csv = out_dir / "fdc_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print("  RESULTS (sorted by Pearson FDC in 7D)")
    print("=" * 60)
    print(f"  {'Loss':>15s} | {'P_7D':>7s} {'P_6D':>7s} | {'S_7D':>7s} {'S_6D':>7s}")
    print("  " + "-" * 56)
    for _, r in df.iterrows():
        print(f"  {r['loss']:>15s} | "
              f"{r['fdc_pearson_7d']:>7.3f} {r['fdc_pearson_6d']:>7.3f} | "
              f"{r['fdc_spearman_7d']:>7.3f} {r['fdc_spearman_6d']:>7.3f}")

    print(f"\n  Wrote {out_csv}")
    print(f"  Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()