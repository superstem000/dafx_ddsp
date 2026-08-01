"""
Exploratory Landscape Analysis (ELA) for Audio Loss Functions — v3
===================================================================

Metrics:
  1. FDC (Spearman, quantile-binned) — Jones & Forrest 1995
  2. Autocorrelation Length — Weinberger 1990
  3. Entropy (Information Content) — Vassilev et al. 2000
  4. Meta-model Linear Fit (R² and slope) — Mersmann et al. 2011
  5. y-Distribution (skewness, kurtosis) — Mersmann et al. 2011
  6. Monotonicity — our adaptation, inspired by FDC

Usage:
  python -m src.cmaes.ela_landscape
  python -m src.cmaes.ela_landscape --n_targets 20 --losses L1_STFT ESR MSS CQT+LogDec
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.stats.qmc import Sobol

import torch

from src.loss.loss_selector import available_loss_names, select_loss_function
from src.plate.SevenParamPlate import BatchedModalPlateTorch

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
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
# HELPERS
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
    """Shared sampling: returns samples [n,7], dist_7d, dist_6d, all_losses dict."""
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
    return samples, dist_7d, dist_6d, all_losses


# ═══════════════════════════════════════════════════════════════════
# METRIC 1: FDC (Spearman, quantile-binned)
# ═══════════════════════════════════════════════════════════════════

def measure_fdc(dist_7d, dist_6d, all_losses, n_bins=5):
    results_7d, results_6d = {}, {}

    for name, loss_vals in all_losses.items():
        # 7D quantile bins
        pct_7d = np.percentile(dist_7d, np.linspace(0, 100, n_bins + 1))
        corrs_7d = []
        for b in range(n_bins):
            lo, hi = pct_7d[b], pct_7d[b + 1]
            mask = (dist_7d >= lo) & (dist_7d <= hi if b == n_bins - 1 else dist_7d < hi)
            if mask.sum() < 20:
                corrs_7d.append(np.nan)
            else:
                r, _ = scipy_stats.spearmanr(dist_7d[mask], loss_vals[mask])
                corrs_7d.append(r)
        results_7d[name] = corrs_7d

        # 6D quantile bins
        pct_6d = np.percentile(dist_6d, np.linspace(0, 100, n_bins + 1))
        corrs_6d = []
        for b in range(n_bins):
            lo, hi = pct_6d[b], pct_6d[b + 1]
            mask = (dist_6d >= lo) & (dist_6d <= hi if b == n_bins - 1 else dist_6d < hi)
            if mask.sum() < 20:
                corrs_6d.append(np.nan)
            else:
                r, _ = scipy_stats.spearmanr(dist_6d[mask], loss_vals[mask])
                corrs_6d.append(r)
        results_6d[name] = corrs_6d

    return results_7d, results_6d


# ═══════════════════════════════════════════════════════════════════
# METRIC 2: Autocorrelation Length
# ═══════════════════════════════════════════════════════════════════

def measure_autocorrelation(synth, loss_fns, gt_params, target_ir_t, duration, device,
                            n_walks=10, walk_length=500, step_frac=0.05):
    step_size = BOUNDS_RANGE * step_frac
    all_ac_lengths = {name: [] for name in loss_fns}
    all_sequences = {name: [] for name in loss_fns}

    for w in range(n_walks):
        pos = BOUNDS_LO + np.random.rand(7) * BOUNDS_RANGE
        walk_points = [pos.copy()]
        for _ in range(walk_length - 1):
            d = np.random.randn(7)
            d /= np.linalg.norm(d)
            pos = np.clip(pos + d * step_size, BOUNDS_LO, BOUNDS_HI)
            walk_points.append(pos.copy())

        walk_points = np.array(walk_points)
        walk_losses = {name: [] for name in loss_fns}
        for i in range(0, walk_length, 50):
            batch = walk_points[i: i + 50]
            audios = synthesize_batch(synth, batch, duration, device)
            bl = compute_losses_batch(loss_fns, target_ir_t, audios)
            for name, vals in bl.items():
                walk_losses[name].append(vals)

        for name in list(walk_losses.keys()):
            if not walk_losses[name]:
                continue
            seq = np.concatenate(walk_losses[name])
            all_sequences[name].append(seq)

            centered = seq - np.mean(seq)
            var = np.var(seq)
            if var < 1e-15:
                all_ac_lengths[name].append(walk_length)
                continue

            max_lag = min(50, len(seq) // 2)
            ac = np.array([np.mean(centered[:len(seq) - lag] * centered[lag:]) / var
                           for lag in range(1, max_lag + 1)])
            below = np.where(ac < 1.0 / np.e)[0]
            all_ac_lengths[name].append(below[0] + 1 if len(below) > 0 else max_lag)

    return {name: np.median(vals) if vals else np.nan
            for name, vals in all_ac_lengths.items()}, all_sequences


# ═══════════════════════════════════════════════════════════════════
# METRIC 3: Entropy (Information Content)
# ═══════════════════════════════════════════════════════════════════

def measure_entropy(all_sequences):
    results = {}
    for name, seqs in all_sequences.items():
        entropies = []
        for seq in seqs:
            if len(seq) < 3:
                continue
            diffs = np.diff(seq)
            eps = np.std(seq) * 0.01 if np.std(seq) > 0 else 1e-10
            labels = np.where(diffs > eps, 1, np.where(diffs < -eps, -1, 0))
            counts = np.array([np.sum(labels == v) for v in [-1, 0, 1]], dtype=float) + 1e-10
            probs = counts / counts.sum()
            entropies.append(-np.sum(probs * np.log2(probs)))
        results[name] = np.median(entropies) if entropies else np.nan
    return results


# ═══════════════════════════════════════════════════════════════════
# METRIC 4: Meta-model Linear Fit (R² and slope)
# ═══════════════════════════════════════════════════════════════════

def measure_metamodel(dist_7d, all_losses):
    results = {}
    for name, loss_vals in all_losses.items():
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(dist_7d, loss_vals)
        results[name] = {"r_squared": r_value ** 2, "slope": slope}
    return results


# ═══════════════════════════════════════════════════════════════════
# METRIC 5: y-Distribution (skewness, kurtosis)
# ═══════════════════════════════════════════════════════════════════

def measure_y_distribution(all_losses):
    results = {}
    for name, loss_vals in all_losses.items():
        results[name] = {
            "skewness": float(scipy_stats.skew(loss_vals)),
            "kurtosis": float(scipy_stats.kurtosis(loss_vals)),
        }
    return results


# ═══════════════════════════════════════════════════════════════════
# METRIC 6: Monotonicity
# ═══════════════════════════════════════════════════════════════════

def measure_monotonicity(synth, loss_fns, gt_params, target_ir_t, duration, device,
                         n_starts=20, n_steps=100):
    gt = gt_params.reshape(1, -1)
    results = {name: [] for name in loss_fns}

    for s in range(n_starts):
        start = BOUNDS_LO + np.random.rand(7) * BOUNDS_RANGE
        alphas = np.linspace(0, 1, n_steps)
        path = np.array([start + a * (gt.flatten() - start) for a in alphas])

        path_losses = {name: [] for name in loss_fns}
        for i in range(0, n_steps, 50):
            batch = path[i: i + 50]
            audios = synthesize_batch(synth, batch, duration, device)
            bl = compute_losses_batch(loss_fns, target_ir_t, audios)
            for name, vals in bl.items():
                path_losses[name].append(vals)

        for name in list(path_losses.keys()):
            if not path_losses[name]:
                continue
            seq = np.concatenate(path_losses[name])
            if len(seq) < 3:
                continue
            results[name].append(np.mean(np.diff(seq) < 0))

    return {name: np.median(vals) if vals else np.nan for name, vals in results.items()}


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_targets", type=int, default=20)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--n_walks", type=int, default=10)
    p.add_argument("--walk_length", type=int, default=500)
    p.add_argument("--n_mono_starts", type=int, default=20)
    p.add_argument("--n_mono_steps", type=int, default=100)
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="ela_results")
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
    print("  EXPLORATORY LANDSCAPE ANALYSIS v3")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Targets: {args.n_targets}")
    print(f"  Duration: {args.duration}s")
    print(f"  Losses: {args.losses}")
    print(f"  Metrics: FDC(quantile), Autocorrelation, Entropy,")
    print(f"           Meta-model, y-Distribution, Monotonicity")

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

    print(f"\n  Generating {args.n_targets} random target IRs...")
    gt_params_all = generate_sobol_targets(args.n_targets, seed=42)
    target_irs = []
    for gt in gt_params_all:
        ir = synthesize_batch(synth, gt.reshape(1, -1), args.duration, device)
        target_irs.append(ir.squeeze(0))
    print(f"  Done.")

    # Accumulators
    n_bins = 5
    all_fdc_7d = {name: [] for name in loss_fns}
    all_fdc_6d = {name: [] for name in loss_fns}
    all_autocorr = {name: [] for name in loss_fns}
    all_entropy = {name: [] for name in loss_fns}
    all_mm_r2 = {name: [] for name in loss_fns}
    all_mm_slope = {name: [] for name in loss_fns}
    all_skewness = {name: [] for name in loss_fns}
    all_kurtosis = {name: [] for name in loss_fns}
    all_mono = {name: [] for name in loss_fns}

    t_start = time.time()

    for t_idx in range(args.n_targets):
        gt = gt_params_all[t_idx]
        target_t = target_irs[t_idx].unsqueeze(0)

        print(f"\n  ── Target {t_idx + 1}/{args.n_targets} ──")

        # Shared sampling for FDC + Meta-model + y-Distribution
        print(f"    Sampling {args.n_samples} points...")
        np.random.seed(t_idx * 1000)
        samples, dist_7d, dist_6d, all_losses = sample_and_evaluate(
            synth, loss_fns, gt, target_t, args.duration, device,
            args.n_samples, args.batch_size
        )

        # 1. FDC
        fdc_7d, fdc_6d = measure_fdc(dist_7d, dist_6d, all_losses, n_bins)
        for name in fdc_7d:
            all_fdc_7d[name].append(fdc_7d[name])
            all_fdc_6d[name].append(fdc_6d[name])

        # 4. Meta-model
        mm = measure_metamodel(dist_7d, all_losses)
        for name, vals in mm.items():
            all_mm_r2[name].append(vals["r_squared"])
            all_mm_slope[name].append(vals["slope"])

        # 5. y-Distribution
        yd = measure_y_distribution(all_losses)
        for name, vals in yd.items():
            all_skewness[name].append(vals["skewness"])
            all_kurtosis[name].append(vals["kurtosis"])

        # 2. Autocorrelation + 3. Entropy (shared walks)
        print(f"    Random walks ({args.n_walks} × {args.walk_length})...")
        np.random.seed(t_idx * 1000 + 1)
        ac_lengths, ac_sequences = measure_autocorrelation(
            synth, loss_fns, gt, target_t, args.duration, device,
            args.n_walks, args.walk_length
        )
        for name, val in ac_lengths.items():
            all_autocorr[name].append(val)

        ent = measure_entropy(ac_sequences)
        for name, val in ent.items():
            all_entropy[name].append(val)

        # 6. Monotonicity
        print(f"    Monotonicity ({args.n_mono_starts} × {args.n_mono_steps})...")
        np.random.seed(t_idx * 1000 + 2)
        mono = measure_monotonicity(
            synth, loss_fns, gt, target_t, args.duration, device,
            args.n_mono_starts, args.n_mono_steps
        )
        for name, val in mono.items():
            all_mono[name].append(val)

        elapsed = time.time() - t_start
        per_target = elapsed / (t_idx + 1)
        remaining = per_target * (args.n_targets - t_idx - 1)
        print(f"    Elapsed: {elapsed:.0f}s, ~{remaining:.0f}s remaining")

    # ═══════════════════════════════════════════════════════════════
    # AGGREGATE
    # ═══════════════════════════════════════════════════════════════

    loss_names = list(loss_fns.keys())

    summary_rows = []
    for name in loss_names:
        row = {"loss": name}
        row["monotonicity"] = np.nanmedian(all_mono[name])
        row["autocorr_length"] = np.nanmedian(all_autocorr[name])
        row["entropy"] = np.nanmedian(all_entropy[name])
        row["metamodel_r2"] = np.nanmedian(all_mm_r2[name])
        row["metamodel_slope"] = np.nanmedian(all_mm_slope[name])
        row["skewness"] = np.nanmedian(all_skewness[name])
        row["kurtosis"] = np.nanmedian(all_kurtosis[name])

        fdc_7d_med = np.nanmedian(all_fdc_7d[name], axis=0) if all_fdc_7d[name] else [np.nan] * n_bins
        fdc_6d_med = np.nanmedian(all_fdc_6d[name], axis=0) if all_fdc_6d[name] else [np.nan] * n_bins
        for b in range(n_bins):
            row[f"fdc_7d_q{b}"] = fdc_7d_med[b] if b < len(fdc_7d_med) else np.nan
            row[f"fdc_6d_q{b}"] = fdc_6d_med[b] if b < len(fdc_6d_med) else np.nan
        row["fdc_7d_global"] = np.nanmean(fdc_7d_med)
        row["fdc_6d_global"] = np.nanmean(fdc_6d_med)
        summary_rows.append(row)

    sdf = pd.DataFrame(summary_rows).sort_values("monotonicity", ascending=False)
    sdf.to_csv(out_dir / "ela_summary.csv", index=False)

    # Print table
    print(f"\n{'=' * 95}")
    print(f"  RESULTS (sorted by monotonicity)")
    print(f"{'=' * 95}")
    print(f"  {'Loss':>15s} | {'Mono':>6s} {'AC':>5s} {'Entr':>5s} | {'R²':>5s} {'Slope':>6s} | {'Skew':>6s} {'Kurt':>6s} | {'FDC7':>5s} {'FDC6':>5s}")
    print("  " + "-" * 85)
    for _, r in sdf.iterrows():
        print(f"  {r['loss']:>15s} | {r['monotonicity']:>6.3f} {r['autocorr_length']:>5.1f} {r['entropy']:>5.3f} | "
              f"{r['metamodel_r2']:>5.3f} {r['metamodel_slope']:>6.3f} | "
              f"{r['skewness']:>6.2f} {r['kurtosis']:>6.2f} | "
              f"{r['fdc_7d_global']:>5.3f} {r['fdc_6d_global']:>5.3f}")

    # ═══════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════

    loss_names_sorted = sdf["loss"].tolist()
    n_losses = len(loss_names_sorted)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_losses))

    # ── Figure 1: FDC heatmaps (quantile-binned) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, n_losses * 0.4)))
    q_labels = ["Q1\n(nearest)", "Q2", "Q3", "Q4", "Q5\n(farthest)"]

    for ax, dim_label, prefix in [(ax1, "7D", "fdc_7d_q"), (ax2, "6D", "fdc_6d_q")]:
        matrix = np.array([[sdf[sdf["loss"] == ln][f"{prefix}{b}"].values[0]
                            for b in range(n_bins)] for ln in loss_names_sorted])
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.3, vmax=0.6)
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(q_labels, fontsize=8)
        ax.set_yticks(range(n_losses))
        ax.set_yticklabels(loss_names_sorted, fontsize=9)
        for i in range(n_losses):
            for j in range(n_bins):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if val < 0 else "black")
        ax.set_title(f"FDC by Distance Quantile ({dim_label})\n~{args.n_samples // n_bins} pts/bin",
                     fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Fitness-Distance Correlation — Quantile-Binned (Spearman)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "fdc_binned.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fdc_binned.png")

    # ── Figure 2: All scalar metrics ──
    metric_configs = [
        ("monotonicity", "Monotonicity\n(higher = navigable)"),
        ("autocorr_length", "Autocorrelation Length\n(higher = smoother)"),
        ("entropy", "Entropy\n(higher = responsive)"),
        ("metamodel_r2", "Meta-model R²\n(higher = linear slope)"),
        ("metamodel_slope", "Meta-model Slope\n(steeper descent)"),
        ("skewness", "Skewness\n(loss distribution shape)"),
        ("kurtosis", "Kurtosis\n(loss distribution tails)"),
    ]

    n_metrics = len(metric_configs)
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten()

    for idx, (key, title) in enumerate(metric_configs):
        ax = axes[idx]
        vals = [sdf[sdf["loss"] == ln][key].values[0] for ln in loss_names_sorted]
        ax.barh(range(n_losses), vals, color=colors)
        ax.set_yticks(range(n_losses))
        ax.set_yticklabels(loss_names_sorted if idx % 4 == 0 else [], fontsize=8)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(v, i, f" {v:.3f}", va="center", fontsize=6)

    axes[-1].set_visible(False)  # 7 metrics, 8 slots

    plt.suptitle("ELA Metrics — Sorted by Monotonicity (best at top)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "ela_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: ela_metrics.png")

    # ── Figure 3: Metric correlation matrix ──
    metric_keys = ["monotonicity", "autocorr_length", "entropy", "metamodel_r2",
                    "metamodel_slope", "skewness", "kurtosis"]
    metric_labels = ["Mono", "AC_len", "Entropy", "R²", "Slope", "Skew", "Kurt"]
    metric_vals = np.array([[sdf[sdf["loss"] == ln][k].values[0] for k in metric_keys]
                            for ln in loss_names_sorted])

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = np.corrcoef(metric_vals.T)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=10)
    for i in range(len(metric_labels)):
        for j in range(len(metric_labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(corr[i, j]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Between ELA Metrics", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "metric_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: metric_correlations.png")

    # ── Figure 4: Each metric vs convergence rate ──
    conv_path = Path("results/loss_comparison/summary.csv")
    if conv_path.exists():
        conv_df = pd.read_csv(conv_path)

        def match_conv(loss_name):
            ln = loss_name.lower().replace("_", "").replace("+", "")
            for _, cr in conv_df.iterrows():
                cn = str(cr["loss"]).lower().replace("_", "").replace("+", "")
                if cn == ln:
                    return cr["conv_rate"]
            return None

        plot_metrics = [
            ("monotonicity", "Monotonicity"),
            ("autocorr_length", "Autocorrelation Length"),
            ("entropy", "Entropy"),
            ("metamodel_r2", "Meta-model R²"),
            ("metamodel_slope", "Meta-model Slope"),
            ("skewness", "Skewness"),
            ("kurtosis", "Kurtosis"),
        ]

        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        axes = axes.flatten()

        for idx, (key, label) in enumerate(plot_metrics):
            ax = axes[idx]
            xs, ys, names_plot = [], [], []
            for _, r in sdf.iterrows():
                cr = match_conv(r["loss"])
                if cr is not None:
                    xs.append(r[key])
                    ys.append(cr)
                    names_plot.append(r["loss"])

            ax.scatter(xs, ys, s=60, zorder=5, color="steelblue")
            for x, y, n in zip(xs, ys, names_plot):
                ax.annotate(n, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6)

            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel("Convergence Rate" if idx % 4 == 0 else "", fontsize=9)
            ax.grid(True, alpha=0.3)

            if len(xs) >= 4:
                r_val, p_val = scipy_stats.pearsonr(xs, ys)
                ax.set_title(f"{label}\nr={r_val:.2f}, p={p_val:.3f}", fontweight="bold", fontsize=9)
            else:
                ax.set_title(label, fontweight="bold", fontsize=9)

        axes[-1].set_visible(False)

        plt.suptitle("ELA Metrics vs CMA-ES Convergence Rate", fontweight="bold", fontsize=13)
        plt.tight_layout()
        plt.savefig(out_dir / "metrics_vs_convergence.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: metrics_vs_convergence.png")

    print(f"\n  Total time: {time.time() - t_start:.0f}s")
    print(f"  All outputs in: {out_dir}/")


if __name__ == "__main__":
    main()