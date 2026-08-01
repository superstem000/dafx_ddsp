#!/usr/bin/env python3
"""
monotonicity_decomp.py
======================

FULL Exploratory Landscape Analysis (ELA) for the decomposition-sweep losses,
correlated against BOTH convergence rate and geomean-NMSE (read from the
sweep's stage-2 summaries).

This is the decomposition-sweep counterpart of ela_landscape.py: it computes
the SAME landscape feature set, but

  * on the losses that actually have a decomp_sweep/<slug>/stage2 result
    (auto-discovered), and
  * on the SAME target IRs the sweep fit (gt params read from stage-2), so the
    features and the outcomes they predict sit on identical targets, and
  * correlated against the sweep's real outcomes (conv_rate, geomean_nmse)
    rather than a separate loss_comparison table.

Feature set (identical definitions to ela_landscape.py):
  1. FDC (Spearman, quantile-binned, 7D + 6D)          — Jones & Forrest 1995
     + FDC total (Spearman, whole-range unbinned, 7D + 6D)   <-- NEW
  2. Autocorrelation Length                            — Weinberger 1990
  3. Entropy (Information Content)                      — Vassilev et al. 2000
  4. Meta-model Linear Fit (R^2, slope)                — Mersmann et al. 2011
  5. y-Distribution (skewness, kurtosis)               — Mersmann et al. 2011
  6. Monotonicity (fraction downhill toward gt)        — our adaptation

Every feature is then correlated (Pearson + Spearman) against conv_rate and
against log10(geomean_nmse), so you can see which landscape feature best
predicts recovery on the decomposition cells.

Losses are auto-discovered: any sub-dir of --results that has
stage2/mu_refined_summary.csv is included (folder slug -> loss name via
select_loss_function's normalization, e.g. l1_stft_log -> L1_STFT_log).

Usage (from repo root ~/dafxchal):
  python monotonicity_decomp.py \
      --results results/decomp_sweep \
      --n_targets 20 \
      --n_samples 2000 \
      --device cuda \
      --out results/monotonicity_decomp

Set --device cpu to avoid the GPU while a sweep is running (slower).
NOTE: the full set (sampling + walks) is much heavier than monotonicity-only.
Drop --n_samples / --n_walks / --walk_length for a quick pass.
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

import torch

from src.loss.loss_selector import select_loss_function
from src.plate.SevenParamPlate import BatchedModalPlateTorch

# ---- constants copied from ela_landscape.py so behaviour matches exactly ----
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


def synthesize_batch(synth, params_7d_np, duration, device):
    plate14 = physical_to_plate14(params_7d_np, device)
    with torch.no_grad():
        return synth(plate14, duration)


def compute_losses_batch(loss_fns, target_t, candidate_batch):
    B = candidate_batch.shape[0]
    target_exp = target_t.expand(B, -1)
    out = {}
    for name, fn in loss_fns.items():
        try:
            val = fn(target_exp, candidate_batch).detach().cpu().numpy()
            if not np.any(np.isnan(val)):
                out[name] = val
        except Exception:
            pass
    return out


# ---- distance metrics (7D physical, 6D composite) -------------------------
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


def sample_and_evaluate(synth, loss_fns, gt_params, target_ir_t, duration, device,
                        n_samples, batch_size):
    """Shared uniform sampling: returns samples[n,7], dist_7d, dist_6d, all_losses."""
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
# METRIC 1a: FDC (Spearman, quantile-binned) — as in ela_landscape.py
# ═══════════════════════════════════════════════════════════════════
def measure_fdc(dist_7d, dist_6d, all_losses, n_bins=5):
    results_7d, results_6d = {}, {}
    for name, loss_vals in all_losses.items():
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
# METRIC 1b: FDC total (Spearman over the FULL distance range, unbinned)
#   Classic Jones & Forrest 1995 FDC. Distinct from fdc_*_global, which is
#   the mean of the within-shell binned correlations.
# ═══════════════════════════════════════════════════════════════════
def measure_fdc_total(dist_7d, dist_6d, all_losses):
    out_7d, out_6d = {}, {}
    for name, loss_vals in all_losses.items():
        if len(loss_vals) >= 20 and np.std(loss_vals) > 0:
            r7, _ = scipy_stats.spearmanr(dist_7d, loss_vals)
            r6, _ = scipy_stats.spearmanr(dist_6d, loss_vals)
        else:
            r7 = r6 = np.nan
        out_7d[name] = r7
        out_6d[name] = r6
    return out_7d, out_6d


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
# METRIC 4: Meta-model Linear Fit (R^2 and slope)
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
# METRIC 6: Monotonicity (verbatim from ela_landscape.py)
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
    return {name: (np.median(v) if v else np.nan) for name, v in results.items()}


# ---- decomp-sweep readers -------------------------------------------------
def discover_cells(results_root: Path):
    """{slug: {'stage2_summary': path}} for cells that have a stage-2 summary."""
    cells = {}
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        s2 = d / "stage2" / "mu_refined_summary.csv"
        if s2.exists():
            cells[d.name] = {"stage2_summary": s2}
    return cells


def load_targets(summary_csv: Path, n_targets: int):
    """(filename, gt_params[7]) for up to n_targets IRs. All cells share targets."""
    df = pd.read_csv(summary_csv).head(n_targets)
    targets = []
    for _, row in df.iterrows():
        gt = np.array([float(row[f"gt_{k}"]) for k in PARAM_KEYS], dtype=np.float64)
        targets.append((str(row["filename"]), gt))
    return targets


def convergence_and_geomean(summary_csv: Path, thresh=0.02):
    df = pd.read_csv(summary_csv)
    v = df["nmse_refined"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    conv = float(np.mean(v < thresh))
    vv = np.clip(v, 1e-20, None)
    geomean = float(np.exp(np.mean(np.log(vv))))
    return conv, geomean, len(v)


# ---- feature list (col, human label, higher_is_better) --------------------
FEATURES = [
    ("monotonicity",     "Monotonicity",        True),
    ("fdc_7d_q0",        "FDC near-shell Q1 (7D)", True),
    ("fdc_6d_q0",        "FDC near-shell Q1 (6D)", True),
    ("fdc_7d_q4",        "FDC far-shell Q5 (7D)",  True),
    ("fdc_6d_q4",        "FDC far-shell Q5 (6D)",  True),
    ("fdc_7d_total",     "FDC total (7D)",      True),
    ("fdc_6d_total",     "FDC total (6D)",      True),
    ("fdc_7d_global",    "FDC mean-of-bins (7D)", True),
    ("fdc_6d_global",    "FDC mean-of-bins (6D)", True),
    ("metamodel_r2",     "Meta-model R^2",      True),
    ("metamodel_slope",  "Meta-model slope",    True),
    ("autocorr_length",  "Autocorr length",     True),
    ("entropy",          "Entropy",             True),
    ("skewness",         "Skewness",            None),
    ("kurtosis",         "Kurtosis",            None),
]


# ---- main -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/decomp_sweep",
                    help="decomp_sweep root (auto-discovers cells with stage2)")
    ap.add_argument("--losses", nargs="+", default=None,
                    help="explicit slugs; default = all discovered cells")
    ap.add_argument("--n_targets", type=int, default=20,
                    help="how many target IRs to average features over")
    ap.add_argument("--n_samples", type=int, default=2000,
                    help="uniform samples per target (FDC / meta-model / y-dist)")
    ap.add_argument("--n_walks", type=int, default=10)
    ap.add_argument("--walk_length", type=int, default=500)
    ap.add_argument("--n_mono_starts", type=int, default=20)
    ap.add_argument("--n_mono_steps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--duration", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/monotonicity_decomp")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results)
    n_bins = 5

    cells = discover_cells(results_root)
    if args.losses:
        cells = {k: v for k, v in cells.items() if k in set(args.losses)}
    if not cells:
        raise SystemExit(f"No cells with stage2 found under {results_root}")
    print(f"device={device}  cells={list(cells)}")

    synth = BatchedModalPlateTorch(sample_rate=SAMPLE_RATE, device=device)
    loss_fns = {}
    for slug in cells:
        try:
            loss_fns[slug] = select_loss_function(slug, sample_rate=SAMPLE_RATE, device=device)
        except Exception as e:
            print(f"  [skip loss] {slug}: {e}")
    cells = {k: v for k, v in cells.items() if k in loss_fns}

    any_summary = next(iter(cells.values()))["stage2_summary"]
    targets = load_targets(any_summary, args.n_targets)
    print(f"targets: {len(targets)} IRs")
    print(f"features: FDC(binned+total, 7D+6D), meta-model, autocorr, entropy, "
          f"y-dist, monotonicity")

    # accumulators
    acc = {k: {slug: [] for slug in loss_fns} for k in
           ["mono", "autocorr", "entropy", "mm_r2", "mm_slope",
            "skew", "kurt", "fdc_total_7d", "fdc_total_6d"]}
    all_fdc_7d = {slug: [] for slug in loss_fns}
    all_fdc_6d = {slug: [] for slug in loss_fns}

    t0 = time.time()
    for ti, (npz_name, gt) in enumerate(targets):
        target_ir = synthesize_batch(synth, gt.reshape(1, -1), args.duration, device).squeeze(0)
        target_t = target_ir.unsqueeze(0)
        print(f"\n  -- target {ti+1}/{len(targets)} --")

        # shared uniform sampling: FDC (binned + total), meta-model, y-dist
        np.random.seed(ti * 1000)
        _, dist_7d, dist_6d, all_losses = sample_and_evaluate(
            synth, loss_fns, gt, target_t, args.duration, device,
            args.n_samples, args.batch_size)

        fdc_7d, fdc_6d = measure_fdc(dist_7d, dist_6d, all_losses, n_bins)
        for name in fdc_7d:
            all_fdc_7d[name].append(fdc_7d[name])
            all_fdc_6d[name].append(fdc_6d[name])

        tot_7d, tot_6d = measure_fdc_total(dist_7d, dist_6d, all_losses)
        for name in tot_7d:
            acc["fdc_total_7d"][name].append(tot_7d[name])
            acc["fdc_total_6d"][name].append(tot_6d[name])

        for name, v in measure_metamodel(dist_7d, all_losses).items():
            acc["mm_r2"][name].append(v["r_squared"])
            acc["mm_slope"][name].append(v["slope"])
        for name, v in measure_y_distribution(all_losses).items():
            acc["skew"][name].append(v["skewness"])
            acc["kurt"][name].append(v["kurtosis"])

        # random walks: autocorr + entropy
        np.random.seed(ti * 1000 + 1)
        ac_lengths, ac_sequences = measure_autocorrelation(
            synth, loss_fns, gt, target_t, args.duration, device,
            args.n_walks, args.walk_length)
        for name, v in ac_lengths.items():
            acc["autocorr"][name].append(v)
        for name, v in measure_entropy(ac_sequences).items():
            acc["entropy"][name].append(v)

        # straight-line walks: monotonicity
        np.random.seed(ti * 1000 + 2)
        for name, v in measure_monotonicity(
                synth, loss_fns, gt, target_t, args.duration, device,
                args.n_mono_starts, args.n_mono_steps).items():
            acc["mono"][name].append(v)

        el = time.time() - t0
        print(f"    {el:.0f}s, ~{el/(ti+1)*(len(targets)-ti-1):.0f}s left")

    # ---- aggregate: features + outcomes per loss ----
    rows = []
    for slug in loss_fns:
        conv, geomean, n = convergence_and_geomean(cells[slug]["stage2_summary"])
        fdc_7d_med = np.nanmedian(all_fdc_7d[slug], axis=0) if all_fdc_7d[slug] else [np.nan] * n_bins
        fdc_6d_med = np.nanmedian(all_fdc_6d[slug], axis=0) if all_fdc_6d[slug] else [np.nan] * n_bins
        row = {
            "loss": slug,
            "monotonicity":     float(np.nanmedian(acc["mono"][slug])),
            "fdc_7d_total":     float(np.nanmedian(acc["fdc_total_7d"][slug])),
            "fdc_6d_total":     float(np.nanmedian(acc["fdc_total_6d"][slug])),
            "fdc_7d_global":    float(np.nanmean(fdc_7d_med)),
            "fdc_6d_global":    float(np.nanmean(fdc_6d_med)),
            "metamodel_r2":     float(np.nanmedian(acc["mm_r2"][slug])),
            "metamodel_slope":  float(np.nanmedian(acc["mm_slope"][slug])),
            "autocorr_length":  float(np.nanmedian(acc["autocorr"][slug])),
            "entropy":          float(np.nanmedian(acc["entropy"][slug])),
            "skewness":         float(np.nanmedian(acc["skew"][slug])),
            "kurtosis":         float(np.nanmedian(acc["kurt"][slug])),
            "conv_rate":        conv,
            "geomean_nmse":     geomean,
            "n_stage2":         n,
        }
        for b in range(n_bins):
            row[f"fdc_7d_q{b}"] = fdc_7d_med[b] if b < len(fdc_7d_med) else np.nan
            row[f"fdc_6d_q{b}"] = fdc_6d_med[b] if b < len(fdc_6d_med) else np.nan
        rows.append(row)

    sdf = pd.DataFrame(rows).sort_values("monotonicity", ascending=False)
    sdf.to_csv(out_dir / "ela_decomp_summary.csv", index=False)

    # ---- print feature table ----
    print("\n" + "=" * 108)
    print(f"  {'loss':>20s} | {'mono':>5s} {'fdcT7':>6s} {'fdcT6':>6s} "
          f"{'mmR2':>5s} {'slope':>7s} {'AC':>5s} {'entr':>5s} | "
          f"{'conv':>5s} {'geo_nmse':>9s}")
    print("  " + "-" * 104)
    for _, r in sdf.iterrows():
        print(f"  {r['loss']:>20s} | {r['monotonicity']:>5.3f} "
              f"{r['fdc_7d_total']:>6.3f} {r['fdc_6d_total']:>6.3f} "
              f"{r['metamodel_r2']:>5.3f} {r['metamodel_slope']:>7.3g} "
              f"{r['autocorr_length']:>5.1f} {r['entropy']:>5.3f} | "
              f"{r['conv_rate']:>5.2f} {r['geomean_nmse']:>9.2e}")

    # ---- correlate every feature vs both outcomes ----
    def corr_pair(xcol, ycol, logy):
        x = sdf[xcol].to_numpy(dtype=float)
        yr = sdf[ycol].to_numpy(dtype=float)
        y = np.log10(np.clip(yr, 1e-20, None)) if logy else yr
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 4:
            return (np.nan,) * 4
        r, p = scipy_stats.pearsonr(x[m], y[m])
        rs, ps = scipy_stats.spearmanr(x[m], y[m])
        return r, p, rs, ps

    outcomes = [("conv_rate", "conv", False), ("geomean_nmse", "geomean(log)", True)]
    corr_rows = []
    print("\n" + "=" * 88)
    print(f"  {'feature':>18s} | {'vs conv: r':>11s} {'p':>6s} {'rho':>6s} | "
          f"{'vs geo: r':>10s} {'p':>6s} {'rho':>6s}")
    print("  " + "-" * 84)
    for fcol, flabel, _hib in FEATURES:
        cr = {"feature": fcol}
        printed = [f"  {fcol:>18s} |"]
        for ycol, ylab, logy in outcomes:
            r, p, rs, ps = corr_pair(fcol, ycol, logy)
            cr[f"{ycol}_pearson_r"] = r
            cr[f"{ycol}_pearson_p"] = p
            cr[f"{ycol}_spearman_rho"] = rs
            cr[f"{ycol}_spearman_p"] = ps
            printed.append(f" {r:>6.3f} {p:>5.3f} {rs:>6.3f} |")
        corr_rows.append(cr)
        print(" ".join(printed).rstrip("|").rstrip())
    cdf = pd.DataFrame(corr_rows)
    cdf.to_csv(out_dir / "feature_outcome_correlations.csv", index=False)

    # ---- figure: feature vs outcome scatter grid ----
    n_feat = len(FEATURES)
    fig, axes = plt.subplots(n_feat, 2, figsize=(9, 2.6 * n_feat))
    if n_feat == 1:
        axes = axes.reshape(1, 2)
    for fi, (fcol, flabel, _hib) in enumerate(FEATURES):
        for ci, (ycol, ylab, logy) in enumerate(outcomes):
            ax = axes[fi, ci]
            xs = sdf[fcol].to_numpy(dtype=float)
            ys = sdf[ycol].to_numpy(dtype=float)
            ax.scatter(xs, ys, s=42, color="steelblue", zorder=5)
            for _, r in sdf.iterrows():
                ax.annotate(r["loss"], (r[fcol], r[ycol]),
                            textcoords="offset points", xytext=(3, 3), fontsize=5)
            yy = np.log10(np.clip(ys, 1e-20, None)) if logy else ys
            m = np.isfinite(xs) & np.isfinite(yy)
            if m.sum() >= 4:
                rr, pp = scipy_stats.pearsonr(xs[m], yy[m])
                _, _, rho, _ = corr_pair(fcol, ycol, logy)
                b, a = np.polyfit(xs[m], yy[m], 1)
                xr = np.linspace(xs[m].min(), xs[m].max(), 50)
                fit = 10 ** (b * xr + a) if logy else (b * xr + a)
                ax.plot(xr, fit, "k--", lw=1, alpha=0.6)
                ax.set_title(f"{flabel} vs {ylab}\nr={rr:.2f} rho={rho:.2f} p={pp:.3f}",
                             fontsize=8)
            else:
                ax.set_title(f"{flabel} vs {ylab}", fontsize=8)
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel(flabel, fontsize=8)
            ax.set_ylabel(ylab, fontsize=8)
            ax.grid(alpha=0.3)
    plt.tight_layout()
    fp = out_dir / "feature_vs_outcome.png"
    plt.savefig(fp, dpi=150)
    plt.close()

    # ---- figure: FDC quantile heatmap (binned), 7D + 6D ----
    loss_order = sdf["loss"].tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(4, len(loss_order) * 0.4)))
    q_labels = ["Q1\n(near)", "Q2", "Q3", "Q4", "Q5\n(far)"]
    for ax, dim, prefix in [(ax1, "7D", "fdc_7d_q"), (ax2, "6D", "fdc_6d_q")]:
        mat = np.array([[sdf[sdf["loss"] == ln][f"{prefix}{b}"].values[0]
                         for b in range(n_bins)] for ln in loss_order])
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-0.3, vmax=0.6)
        ax.set_xticks(range(n_bins)); ax.set_xticklabels(q_labels, fontsize=8)
        ax.set_yticks(range(len(loss_order))); ax.set_yticklabels(loss_order, fontsize=8)
        for i in range(len(loss_order)):
            for j in range(n_bins):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if v < 0 else "black")
        ax.set_title(f"FDC by distance quantile ({dim})", fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "fdc_binned.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  saved: {out_dir / 'ela_decomp_summary.csv'}")
    print(f"  saved: {out_dir / 'feature_outcome_correlations.csv'}")
    print(f"  saved: {fp}")
    print(f"  saved: {out_dir / 'fdc_binned.png'}")
    print(f"  total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()