#!/usr/bin/env python3
"""Aggregate the standard-loss sweep into one comparison table.

For each <loss>/ directory under --root, reads:
    stage1/summary.csv            (CMA-ES: best_loss, gt_loss, nmse [pre-mu])
    stage2/mu_refined_summary.csv (ternary: nmse_orig, nmse_refined [final])

and reports, per loss:
    conv_rate                post-ternary convergence  = frac(nmse_refined < thr)
    s2_nmse_mean / _median   final 6D NMSE (the accuracy number)
    s1_nmse_mean             stage-1 NMSE (pre-mu; mu-dominated, for reference)
    s1_frac_reached_gt_loss  frac of IRs whose converged loss reached GT loss
    s1_loss_gap_mean         mean(best_loss - gt_loss)

Standalone (no repo imports). Run either way:
    python src/standard_sweep/aggregate_sweep.py --root results/standard_sweep --plot
    python -m src.standard_sweep.aggregate_sweep --root results/standard_sweep --plot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_read(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def aggregate(root: Path, conv_threshold: float) -> pd.DataFrame:
    rows = []
    for slug_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        s1 = _safe_read(slug_dir / "stage1" / "summary.csv")
        s2 = _safe_read(slug_dir / "stage2" / "mu_refined_summary.csv")
        if s1 is None and s2 is None:
            continue

        # canonical loss name from stage-1 CSV if present, else fall back to slug
        loss_name = slug_dir.name
        if s1 is not None and "loss_name" in s1.columns and len(s1):
            loss_name = str(s1["loss_name"].iloc[0])

        rec = {"loss": loss_name, "slug": slug_dir.name,
               "n_stage1": 0, "n_stage2": 0}

        # ---------------- Stage 1 ----------------
        if s1 is not None and len(s1):
            rec["n_stage1"] = int(len(s1))
            if "nmse" in s1.columns:
                nm = pd.to_numeric(s1["nmse"], errors="coerce").dropna()
                if len(nm):
                    rec["s1_nmse_mean"] = float(nm.mean())
                    rec["s1_nmse_median"] = float(nm.median())
            if "best_loss" in s1.columns and "gt_loss" in s1.columns:
                bl = pd.to_numeric(s1["best_loss"], errors="coerce")
                gl = pd.to_numeric(s1["gt_loss"], errors="coerce")
                mask = bl.notna() & gl.notna()
                if mask.any():
                    rec["s1_best_loss_mean"] = float(bl[mask].mean())
                    rec["s1_gt_loss_mean"] = float(gl[mask].mean())
                    rec["s1_loss_gap_mean"] = float((bl[mask] - gl[mask]).mean())
                    # reached GT loss if converged within 5% of the GT loss value
                    rec["s1_frac_reached_gt_loss"] = float((bl[mask] <= gl[mask] * 1.05).mean())

        # ---------------- Stage 2 ----------------
        if s2 is not None and len(s2):
            rec["n_stage2"] = int(len(s2))
            if "nmse_refined" in s2.columns:
                nr = pd.to_numeric(s2["nmse_refined"], errors="coerce").dropna()
                if len(nr):
                    rec["s2_nmse_mean"] = float(nr.mean())
                    rec["s2_nmse_median"] = float(nr.median())
                    rec["conv_rate"] = float((nr < conv_threshold).mean())

        rows.append(rec)

    df = pd.DataFrame(rows)
    if "conv_rate" in df.columns:
        df = df.sort_values("conv_rate", ascending=False, na_position="last").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/standard_sweep")
    ap.add_argument("--conv_threshold", type=float, default=0.02,
                    help="NMSE threshold for 'converged' (post-ternary)")
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--plot", action="store_true",
                    help="also save a convergence-rate bar chart PNG")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root not found: {root}")
        return

    df = aggregate(root, args.conv_threshold)
    if df.empty:
        print(f"No results found under {root}")
        return

    out_csv = Path(args.out_csv) if args.out_csv else root / "standard_sweep_comparison.csv"
    df.to_csv(out_csv, index=False, float_format="%.10g")

    cols = ["loss", "n_stage2", "conv_rate", "s2_nmse_mean", "s2_nmse_median",
            "s1_nmse_mean", "s1_frac_reached_gt_loss", "s1_loss_gap_mean"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.4g}"):
        print("\n=== standard_sweep comparison (sorted by post-ternary convergence) ===\n")
        print(df[cols].to_string(index=False))
    print(f"\nWrote: {out_csv}")

    if args.plot and "conv_rate" in df.columns:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            d = df.dropna(subset=["conv_rate"])
            plt.figure(figsize=(11, 4))
            plt.bar(d["loss"].astype(str), d["conv_rate"])
            plt.ylabel(f"Convergence rate (NMSE < {args.conv_threshold})")
            plt.ylim(0, 1)
            plt.xticks(rotation=45, ha="right")
            for i, v in enumerate(d["conv_rate"]):
                plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            plt.tight_layout()
            p = root / "standard_sweep_convergence.png"
            plt.savefig(p, dpi=130)
            print(f"Wrote: {p}")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
