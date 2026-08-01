"""Compare optimizer runs by NMSE, compute, and convergence rate.

Default comparison (all on L1_STFT loss, 0.25 s signals):

    1. cmaes_norm_es/l1_stft        — CMA-ES + SHP + study-ES (existing headline)
    2. cmaes_noprune_5k              — CMA-ES + NopPruner + study-ES (NEW)
    3. baseline_pso_plusplus         — PSO + Optuna, no early stop (existing)
    4. pso_es_5k                     — PSO + Optuna + study-ES (NEW)

Outputs (written into --analysis_dir):

    summary_table.csv          — one row per method (means / medians / convergence rate)
    per_ir_long.csv            — long-format per-IR table for downstream plotting
    compute_vs_nmse.png        — scatter, per-IR points, color=method
    convergence_rate.png       — bar chart, fraction of IRs reaching threshold
    paired_nmse_<A>_vs_<B>.png — paired scatter for each pair of methods
    failure_overlap.csv        — for each method: list of failure-IR stems

Run as:
    python -m experiment_pruning_es.analysis.compare_runs \\
        --runs cmaes_norm_es=results/cmaes_norm_es/l1_stft \\
               cmaes_noprune=results/experiment_pruning_es/cmaes_noprune_5k \\
               pso_plusplus=results/baseline_pso_plusplus \\
               pso_es=results/experiment_pruning_es/pso_es_5k \\
        --analysis_dir results/experiment_pruning_es/analysis
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Default failure threshold (NMSE > THRESH = "wrong basin").
DEFAULT_FAILURE_THRESHOLD = 0.05
# Default convergence threshold (NMSE < THRESH = "converged").
DEFAULT_CONVERGENCE_THRESHOLDS = (0.001, 0.01, 0.05)

METHOD_COLORS = {
    "cmaes_norm_es": "#1f77b4",
    "cmaes_noprune": "#2ca02c",
    "pso_plusplus": "#d62728",
    "pso_es": "#ff7f0e",
}


def load_run(name: str, path: Path) -> pd.DataFrame:
    summary_path = path / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.csv found in {path}")
    df = pd.read_csv(summary_path)
    df["method"] = name
    if "nmse" not in df.columns:
        raise ValueError(f"summary.csv at {path} has no `nmse` column")
    if "runtime" not in df.columns:
        raise ValueError(f"summary.csv at {path} has no `runtime` column")
    # `total_evals_used` may not exist on older runs (pre-fork).
    if "total_evals_used" not in df.columns:
        df["total_evals_used"] = np.nan
    return df


def aggregate_summary(per_ir: pd.DataFrame, thresholds: tuple) -> pd.DataFrame:
    rows = []
    for method, g in per_ir.groupby("method"):
        n = len(g)
        nmse = g["nmse"].dropna()
        runtime = g["runtime"]
        evals = g["total_evals_used"].dropna()

        row = {
            "method": method,
            "n_irs": n,
            "mean_nmse": float(nmse.mean()) if len(nmse) else np.nan,
            "median_nmse": float(nmse.median()) if len(nmse) else np.nan,
            "std_nmse": float(nmse.std()) if len(nmse) else np.nan,
            "mean_runtime_s": float(runtime.mean()),
            "median_runtime_s": float(runtime.median()),
            "mean_evals_used": float(evals.mean()) if len(evals) else np.nan,
            "median_evals_used": float(evals.median()) if len(evals) else np.nan,
        }
        for t in thresholds:
            row[f"convrate_lt_{t}"] = float((nmse < t).mean()) if len(nmse) else np.nan
        row[f"failrate_gt_{DEFAULT_FAILURE_THRESHOLD}"] = (
            float((nmse > DEFAULT_FAILURE_THRESHOLD).mean()) if len(nmse) else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_compute_vs_nmse(per_ir: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, g in per_ir.groupby("method"):
        x = g["runtime"].values
        y = g["nmse"].values
        c = METHOD_COLORS.get(method, None)
        ax.scatter(x, y, alpha=0.6, s=24, label=method, color=c, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Runtime per IR (s, log)")
    ax.set_ylabel("NMSE (log)")
    ax.set_title("Per-IR runtime vs NMSE — each point is one IR")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path / "compute_vs_nmse.png", dpi=130)
    plt.close(fig)


def plot_convergence_rate(per_ir: pd.DataFrame, out_path: Path, thresholds=DEFAULT_CONVERGENCE_THRESHOLDS):
    methods = sorted(per_ir["method"].unique())
    n_thresh = len(thresholds)

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * n_thresh * len(methods) / 2), 5))
    width = 0.8 / len(methods)
    x = np.arange(n_thresh)
    for i, method in enumerate(methods):
        sub = per_ir[per_ir["method"] == method]["nmse"].dropna()
        rates = [float((sub < t).mean()) for t in thresholds]
        c = METHOD_COLORS.get(method, None)
        ax.bar(
            x + i * width - 0.4 + width / 2,
            rates,
            width,
            label=method,
            color=c,
            edgecolor="black",
            linewidth=0.5,
        )
        for j, r in enumerate(rates):
            ax.text(x[j] + i * width - 0.4 + width / 2, r + 0.01, f"{int(r * 100)}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"NMSE < {t}" for t in thresholds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of IRs")
    ax.set_title("Convergence rate by threshold")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path / "convergence_rate.png", dpi=130)
    plt.close(fig)


def plot_paired_nmse(per_ir: pd.DataFrame, method_a: str, method_b: str, out_path: Path):
    a = per_ir[per_ir["method"] == method_a].set_index("filename")["nmse"]
    b = per_ir[per_ir["method"] == method_b].set_index("filename")["nmse"]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return
    a, b = a.loc[common], b.loc[common]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(a, b, alpha=0.55, s=22, color="#444", edgecolors="none")
    lim_lo = min(a.min(), b.min(), 1e-7)
    lim_hi = max(a.max(), b.max(), 1.0)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(f"{method_a} NMSE")
    ax.set_ylabel(f"{method_b} NMSE")
    ax.set_title(f"Paired per-IR NMSE: {method_a} vs {method_b}\n(below diagonal: {method_b} is better)")
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    safe_a = method_a.replace("/", "_")
    safe_b = method_b.replace("/", "_")
    plt.savefig(out_path / f"paired_nmse_{safe_a}_vs_{safe_b}.png", dpi=130)
    plt.close(fig)


def write_failure_overlap(per_ir: pd.DataFrame, out_path: Path, threshold: float = DEFAULT_FAILURE_THRESHOLD):
    rows = []
    for method, g in per_ir.groupby("method"):
        fails = g[g["nmse"] > threshold]["filename"].tolist()
        rows.append(
            {
                "method": method,
                f"n_failures_gt_{threshold}": len(fails),
                "failure_files": ";".join(sorted(fails)),
            }
        )
    pd.DataFrame(rows).to_csv(out_path / "failure_overlap.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="space-separated NAME=PATH pairs (e.g. cmaes_noprune=results/.../cmaes_noprune_5k)",
    )
    ap.add_argument(
        "--analysis_dir",
        type=str,
        default="results/experiment_pruning_es/analysis",
        help="output directory for plots and CSVs",
    )
    ap.add_argument(
        "--failure_threshold",
        type=float,
        default=DEFAULT_FAILURE_THRESHOLD,
        help="NMSE above this is counted as a failure (basin missed).",
    )
    args = ap.parse_args()

    out_path = Path(args.analysis_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    parsed = []
    for r in args.runs:
        if "=" not in r:
            raise ValueError(f"Bad --runs entry {r!r}; expected NAME=PATH")
        name, path = r.split("=", 1)
        parsed.append((name, Path(path)))

    print("Loading runs:")
    frames = []
    for name, path in parsed:
        print(f"  {name:<20} <- {path}")
        try:
            frames.append(load_run(name, path))
        except FileNotFoundError as e:
            print(f"    [skipped] {e}")

    if not frames:
        print("Nothing loaded; aborting.")
        return

    per_ir = pd.concat(frames, ignore_index=True)
    per_ir.to_csv(out_path / "per_ir_long.csv", index=False)

    summary = aggregate_summary(per_ir, DEFAULT_CONVERGENCE_THRESHOLDS)
    summary.to_csv(out_path / "summary_table.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))

    plot_compute_vs_nmse(per_ir, out_path)
    plot_convergence_rate(per_ir, out_path, DEFAULT_CONVERGENCE_THRESHOLDS)
    write_failure_overlap(per_ir, out_path, args.failure_threshold)

    methods = sorted(per_ir["method"].unique())
    for a, b in itertools.combinations(methods, 2):
        plot_paired_nmse(per_ir, a, b, out_path)

    print(f"\nWrote outputs to {out_path}")


if __name__ == "__main__":
    main()
