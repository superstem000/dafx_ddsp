#!/usr/bin/env python3
"""
analyze_log_weighting.py

Proves, empirically, WHY log-magnitude compression relocates the minimum for
modal-IR parameter recovery, via two mechanisms:

  (A) Per-bin loss weighting.  For target-vs-(gt-synth), decompose each STFT
      bin's contribution to the loss under three magnitude transforms:
        linear : |St - Sc|                 (== L1_STFT)
        C1     : |log(St+eps) - log(Sc+eps)|   (eps=1e-7, the paper's floored log)
        C2     : |log1p(St) - log1p(Sc)|        (the paper's Smooth-MSS fix)
      Sorting bins by TRUE (target) energy shows where each metric puts its
      loss mass.  Linear concentrates in the loud modes; C1 spreads it into the
      near-zero floor; C2 sits close to linear at low energy (log1p ~ linear
      there) but still compresses mid/high energy.

  (B) Relocation.  Along the straight line in physical-parameter space from
      ground truth -> log's own failed fit, evaluate each metric.  Linear is
      minimized at truth (increasing away from it); C1 is *lower* at the wrong
      end (min relocated); C2 is mostly restored.  This is best<gt made visible.

The mechanism figures use self-contained STFT transforms (not the library
loss_fn) so per-bin contributions are exactly interpretable; the qualitative
weighting property is independent of the exact window.

Reads gt + fitted params directly from the stage-1 result CSVs (columns
E,rho,h,Ly,T0,op_x,op_y and gt_*), so no re-optimization is done -- this is a
handful of forward syntheses, ~1000x cheaper than one CMA-ES fit.

Usage (run from repo root, e.g. ~/dafxchal):
  python analyze_log_weighting.py \
      --targets /path/to/ir_npz_dir \
      --results ~/dafxchal/results/decomp_sweep \
      --log_cell l1_stft_log \
      --duration 0.25 \
      --device cpu \
      --out ~/dafxchal/results/log_weighting

Set --duration to whatever the sweep used (must match, or gt-synth won't line
up with the recorded best<gt). --device cpu is the zero-GPU-interference option;
--device cuda is fine too (see notes at bottom of the chat message).
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")  # headless: never needs a display, safe over SSH
import matplotlib.pyplot as plt

# --- interface, imported from the fitter so the param mapping is IDENTICAL ----
# The fitter lives at src/cmaes/fit_7param_norm_es.py (not repo root), so load it
# by file path. It internally does `from src.loss...`/`from src.plate...`, which
# resolve because we run from the repo root (~/dafxchal). Importing only defines
# functions/constants; its main() is guarded by if __name__ == "__main__".
import importlib.util as _ilu


def _load_fitter():
    import glob as _g
    candidates = [
        Path.home() / "dafxchal" / "src" / "cmaes" / "fit_7param_norm_es.py",
    ]
    for c in candidates:
        if c.exists():
            path = str(c)
            break
    else:
        hits = _g.glob(
            str(Path.home() / "dafxchal" / "**" / "fit_7param_norm_es.py"),
            recursive=True,
        )
        if not hits:
            raise SystemExit("Could not find fit_7param_norm_es.py under ~/dafxchal")
        path = hits[0]
    spec = _ilu.spec_from_file_location("fit_7param_norm_es", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_fit = _load_fitter()
PARAM_KEYS = _fit.PARAM_KEYS
SAMPLE_RATE = _fit.SAMPLE_RATE
physical_to_plate14_tensor = _fit.physical_to_plate14_tensor
load_target_ir_from_npz = _fit.load_target_ir_from_npz

from src.plate.SevenParamPlate import BatchedModalPlateTorch

EPS_C1 = 1e-7  # matches the paper's C1 = log(x + eps)


# ----------------------------------------------------------------------------
# STFT + the three magnitude transforms
# ----------------------------------------------------------------------------
def stft_mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """x: [N] -> magnitude [F, T] (flattened by caller)."""
    w = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    S = torch.stft(x, n_fft, hop, win_length=n_fft, window=w, return_complex=True)
    return S.abs()


def transforms(S: torch.Tensor):
    """Return the three transformed magnitude spectrograms."""
    return {
        "linear": S,
        "C1": torch.log(S + EPS_C1),
        "C2": torch.log1p(S),
    }


def per_bin_contrib(St: torch.Tensor, Sc: torch.Tensor):
    """|transform(St) - transform(Sc)| per bin, for each metric. Flattened."""
    Tt, Tc = transforms(St), transforms(Sc)
    return {k: (Tt[k] - Tc[k]).abs().flatten() for k in Tt}


# ----------------------------------------------------------------------------
# data helpers
# ----------------------------------------------------------------------------
def load_cell_rows(results_root: Path, cell: str) -> pd.DataFrame:
    d = results_root / cell / "stage1"
    files = sorted(glob.glob(str(d / "result_*.csv")))
    if not files:
        raise SystemExit(f"No result_*.csv in {d}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df


def params_from_row(row, prefix=""):
    """Pull the 7 physical params (fitted if prefix='', gt if prefix='gt_')."""
    return np.array([[float(row[f"{prefix}{k}"]) for k in PARAM_KEYS]], dtype=np.float64)


def synth_ir(synth, phys_1x7: np.ndarray, duration: float, device, dtype) -> torch.Tensor:
    plate = physical_to_plate14_tensor(phys_1x7, device).to(dtype=dtype)
    with torch.no_grad():
        aud = synth(plate, duration)          # [1, N]
    return aud.squeeze(0)


def match_len(a: torch.Tensor, b: torch.Tensor):
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


# ----------------------------------------------------------------------------
# (A) per-bin weighting analysis
# ----------------------------------------------------------------------------
def weighting_analysis(df, targets_dir, synth, args, device, dtype):
    """Cumulative loss-mass vs target-energy percentile, per IR (for SD bands)."""
    n_q = 20
    # per-IR quantile fractions, per metric: list of [n_q] arrays
    per_ir = {m: [] for m in ("linear", "C1", "C2")}

    rows = df.head(args.n_metric)
    for _, row in rows.iterrows():
        npz = Path(targets_dir) / str(row["target_npz"])
        if not npz.exists():
            print(f"  [skip] missing target {npz}")
            continue
        target = load_target_ir_from_npz(npz, args.duration, SAMPLE_RATE)
        tgt = torch.tensor(target, dtype=dtype, device=device)
        gt = synth_ir(synth, params_from_row(row, "gt_"), args.duration, device, dtype)
        tgt, gt = match_len(tgt, gt)

        St = stft_mag(tgt, args.n_fft, args.hop)
        Sg = stft_mag(gt, args.n_fft, args.hop)
        contribs = per_bin_contrib(St, Sg)

        energy = St.flatten()
        order = torch.argsort(energy)          # low -> high target energy
        idx_groups = np.array_split(order.cpu().numpy(), n_q)
        for m in per_ir:
            c = contribs[m].detach().cpu().numpy()
            tot = c.sum() + 1e-30
            per_ir[m].append(np.array([c[g].sum() / tot for g in idx_groups]))

    used = len(per_ir["linear"])
    if used == 0:
        raise SystemExit("No IRs usable for weighting analysis (targets missing?)")

    # stack -> [n_ir, n_q]; cumulative per IR, then mean/SD across IRs
    pct = (np.arange(1, n_q + 1) / n_q) * 100.0
    half = n_q // 2
    cum_mean, cum_sd, frac_low, frac_low_sd = {}, {}, {}, {}
    for m in per_ir:
        M = np.vstack(per_ir[m])                 # [n_ir, n_q]
        C = np.cumsum(M, axis=1)                 # cumulative per IR
        cum_mean[m] = C.mean(axis=0)
        cum_sd[m] = C.std(axis=0)
        low = M[:, :half].sum(axis=1)            # per-IR floor fraction
        frac_low[m] = low.mean()
        frac_low_sd[m] = low.std()

    styles = {"linear": ("C0", "-"), "C2": ("C2", "--"), "C1": ("C3", ":")}
    labels = {"linear": "linear (L1-STFT)", "C1": "C1: log(x+ε)", "C2": "C2: log(1+x)"}

    # ---- Figure 1: cumulative loss mass vs energy percentile, +/- 1 SD band ----
    plt.figure(figsize=(5.2, 4.0))
    for m in ("linear", "C2", "C1"):
        col, ls = styles[m]
        plt.plot(pct, cum_mean[m], ls, color=col, lw=2, label=labels[m])
        plt.fill_between(pct, cum_mean[m] - cum_sd[m], cum_mean[m] + cum_sd[m],
                         color=col, alpha=0.15, linewidth=0)
    plt.xlabel("STFT bins ordered by target energy (low → high, %)")
    plt.ylabel("cumulative fraction of loss")
    plt.title(f"Where each metric puts its loss mass  (n={used}, ±1 SD)")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    f1 = Path(args.out) / "fig1_loss_mass_by_energy.png"
    plt.savefig(f1, dpi=160)
    plt.close()

    # ---- Figure 2: bar of floor fraction, with SD error bars ----
    plt.figure(figsize=(4.2, 3.6))
    ms = ["linear", "C2", "C1"]
    vals = [frac_low[m] for m in ms]
    errs = [frac_low_sd[m] for m in ms]
    plt.bar(range(len(ms)), vals, yerr=errs, capsize=4,
            color=[styles[m][0] for m in ms])
    plt.xticks(range(len(ms)), [labels[m] for m in ms], rotation=15, fontsize=8)
    plt.ylabel("loss mass in lowest-energy 50% of bins")
    for i, v in enumerate(vals):
        plt.text(i, v + errs[i] + 0.02, f"{v*100:.0f}%", ha="center", fontsize=9)
    plt.ylim(0, min(1.05, max(vals) * 1.3 + 0.05))
    plt.title("Floor up-weighting")
    plt.tight_layout()
    f2 = Path(args.out) / "fig2_floor_fraction.png"
    plt.savefig(f2, dpi=160)
    plt.close()

    print(f"\n[A] Per-bin weighting  (averaged over {used} IRs, target vs gt-synth)")
    print(f"    loss mass in lowest-energy 50% of bins  (mean ± SD):")
    for m in ("linear", "C2", "C1"):
        print(f"      {labels[m]:<18} {frac_low[m]*100:5.1f}% ± {frac_low_sd[m]*100:.1f}")
    print(f"    -> {f1.name}, {f2.name}")
    return frac_low


# ----------------------------------------------------------------------------
# (B) relocation along gt -> wrong-fit line
# ----------------------------------------------------------------------------
def scalar_loss(St, Sc):
    """mean |transform diff| per metric -> scalar."""
    Tt, Tc = transforms(St), transforms(Sc)
    return {k: float((Tt[k] - Tc[k]).abs().mean()) for k in Tt}


def relocation_analysis(df, targets_dir, synth, args, device, dtype):
    fails = df[(df["best_loss"] < df["gt_loss"]) & (df["nmse"] > 0.02)]
    if len(fails) == 0:
        print("\n[B] No best<gt & nmse>0.02 IRs in this cell -- skipping relocation.")
        return
    fails = fails.head(args.max_alpha_irs)
    alphas = np.linspace(0.0, 1.0, args.alpha_steps)

    curves = {m: [] for m in ("linear", "C1", "C2")}  # normalized loss/loss(gt)
    end_ratio = {m: [] for m in curves}               # loss(wrong)/loss(gt) per IR
    reloc_count = {m: 0 for m in curves}

    for _, row in fails.iterrows():
        npz = Path(targets_dir) / str(row["target_npz"])
        if not npz.exists():
            continue
        target = load_target_ir_from_npz(npz, args.duration, SAMPLE_RATE)
        tgt = torch.tensor(target, dtype=dtype, device=device)
        p_gt = params_from_row(row, "gt_")[0]
        p_wrong = params_from_row(row, "")[0]

        per_metric = {m: [] for m in curves}
        for a in alphas:
            p = (1 - a) * p_gt + a * p_wrong
            cand = synth_ir(synth, p[None, :], args.duration, device, dtype)
            t2, c2 = match_len(tgt, cand)
            St, Sc = stft_mag(t2, args.n_fft, args.hop), stft_mag(c2, args.n_fft, args.hop)
            L = scalar_loss(St, Sc)
            for m in curves:
                per_metric[m].append(L[m])
        for m in curves:
            arr = np.array(per_metric[m])
            arr = arr / (arr[0] + 1e-30)      # normalize to loss at gt (=1.0)
            curves[m].append(arr)
            end_ratio[m].append(arr[-1])
            if arr[-1] < 1.0:                 # wrong end scores below gt
                reloc_count[m] += 1

    n_used = len(curves["linear"])
    labels = {"linear": "linear (L1-STFT)", "C1": "C1: log(x+ε)", "C2": "C2: log(1+x)"}
    cols = {"linear": "C0", "C1": "C3", "C2": "C2"}

    # ---- Figure 3: normalized loss along gt->wrong, per-panel linear y-axis ----
    # (per-panel autoscale, NOT shared: linear/C2 show their steep wall away from
    #  truth; C1 autoscales to ~[0.8,1.5] so its dip below 1.0 and its flatness
    #  are both visible -- the point a shared/log axis hides.)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    for ax, m in zip(axes, ("linear", "C2", "C1")):
        for arr in curves[m]:
            ax.plot(alphas, arr, color=cols[m], alpha=0.35, lw=1)
        mean_c = np.mean(np.vstack(curves[m]), axis=0)
        ax.plot(alphas, mean_c, color=cols[m], lw=2.5, label="mean")
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_title(f"{labels[m]}\nwrong<gt in {reloc_count[m]}/{n_used}")
        ax.set_xlabel("α   (0 = ground truth → 1 = log's wrong fit)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("loss / loss(ground truth)")
    plt.tight_layout()
    f3 = Path(args.out) / "fig3_relocation_path.png"
    plt.savefig(f3, dpi=160)
    plt.close()

    # preference ratio = median loss(wrong)/loss(gt): >1 truth preferred, <1 relocated
    med_end = {m: float(np.median(end_ratio[m])) for m in curves}

    print(f"\n[B] Relocation along gt→wrong  ({n_used} failed IRs)")
    print(f"    NOTE: C1's wrong<gt is partly definitional (IRs were selected as")
    print(f"    best_loss<gt_loss under the log loss). The non-circular result is")
    print(f"    that linear & C2 rank those SAME wrong points ABOVE truth.")
    print(f"    wrong end scores BELOW ground truth (min relocated):")
    for m in ("linear", "C2", "C1"):
        print(f"      {labels[m]:<18} {reloc_count[m]}/{n_used}")
    print(f"\n    preference ratio  median[ loss(wrong)/loss(gt) ]  (>1 = truth wins):")
    for m in ("linear", "C2", "C1"):
        print(f"      {labels[m]:<18} {med_end[m]:.3g}")
    print(f"    -> as compression increases (linear→C2→C1) truth's advantage over")
    print(f"       the wrong point shrinks by orders of magnitude and inverts at C1.")
    print(f"    -> {f3.name}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="dir with random_IR_*.npz")
    ap.add_argument("--results", required=True, help="decomp_sweep root")
    ap.add_argument("--log_cell", default="l1_stft_log",
                    help="cell whose failed fits define the relocation set")
    ap.add_argument("--duration", type=float, default=0.25,
                    help="MUST match the duration the sweep used")
    ap.add_argument("--n_fft", type=int, default=4096)
    ap.add_argument("--hop", type=int, default=1024)
    ap.add_argument("--n_metric", type=int, default=50,
                    help="how many IRs to average the weighting analysis over")
    ap.add_argument("--max_alpha_irs", type=int, default=60,
                    help="how many failed IRs to draw relocation paths for (covers all)")
    ap.add_argument("--alpha_steps", type=int, default=25)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu = zero GPU interference (recommended while sweep runs)")
    ap.add_argument("--out", default="results/log_weighting")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float32
    torch.set_grad_enabled(False)

    print(f"device={device}  duration={args.duration}s  n_fft={args.n_fft} hop={args.hop}")
    synth = BatchedModalPlateTorch(device=device, dtype=dtype).to(device)
    df = load_cell_rows(Path(args.results), args.log_cell)
    print(f"loaded {len(df)} rows from cell '{args.log_cell}'")

    weighting_analysis(df, args.targets, synth, args, device, dtype)
    relocation_analysis(df, args.targets, synth, args, device, dtype)
    print(f"\nfigures + output written to: {args.out}")


if __name__ == "__main__":
    main()
