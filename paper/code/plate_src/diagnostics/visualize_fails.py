"""Visualize where failed IRs lie in parameter space.

Loads summary.csv from multiple result folders, counts failures per IR,
and plots:
- D_mu vs T0_mu
- op_x vs op_y
- Ly vs op_y

Color by failure count across runs using an automatic colormap.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NU = 0.25


def _read_summary_rows(summary_csv: Path) -> list[dict]:
    with summary_csv.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(row: dict, key: str, default: float | None = None) -> float | None:
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _compute_gt_composites(row: dict) -> dict:
    gt_E = _as_float(row, "gt_E")
    gt_rho = _as_float(row, "gt_rho")
    gt_h = _as_float(row, "gt_h")
    gt_T0 = _as_float(row, "gt_T0")
    gt_Ly = _as_float(row, "gt_Ly")
    gt_op_x = _as_float(row, "gt_op_x")
    gt_op_y = _as_float(row, "gt_op_y")

    if None not in (gt_E, gt_rho, gt_h, gt_T0, gt_Ly, gt_op_x, gt_op_y):
        mu = gt_rho * gt_h
        D = gt_E * (gt_h**3) / (12.0 * (1.0 - NU**2))
        return {
            "D_mu": D / mu,
            "T0_mu": gt_T0 / mu,
            "Ly": gt_Ly,
            "op_x": gt_op_x,
            "op_y": gt_op_y,
        }

    # Fallback to summary-point values if gt_* are missing.
    return {
        "D_mu": float(_as_float(row, "D_mu", 0.0)),
        "T0_mu": float(_as_float(row, "T0_mu", 0.0)),
        "Ly": float(_as_float(row, "Ly", 0.0)),
        "op_x": float(_as_float(row, "op_x", 0.0)),
        "op_y": float(_as_float(row, "op_y", 0.0)),
    }


def run(args):
    summary_paths = [Path(p) for p in args.summaries]
    for p in summary_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing summary.csv: {p}")

    runs = [_read_summary_rows(p) for p in summary_paths]
    if not runs or not runs[0]:
        raise ValueError("No rows found in first summary")

    # Build filename->row maps for robust alignment.
    maps = []
    for rows in runs:
        m = {}
        for r in rows:
            fn = r.get("filename") or r.get("target_npz")
            if fn:
                m[fn] = r
        maps.append(m)

    ref_filenames = [r.get("filename") or r.get("target_npz") for r in runs[0]]
    ref_filenames = [f for f in ref_filenames if f is not None]

    fail_counts = []
    comps = []

    for fn in ref_filenames:
        failures = 0
        picked_row = None
        for m in maps:
            r = m.get(fn)
            if r is None:
                # Missing row in a run is treated as failure for that run.
                failures += 1
                continue
            if picked_row is None:
                picked_row = r
            bl = _as_float(r, "best_loss", np.inf)
            # Failure criterion: loss is NOT below threshold.
            if bl is None or bl >= args.success_threshold:
                failures += 1

        if picked_row is None:
            continue

        fail_counts.append(int(failures))
        comps.append(_compute_gt_composites(picked_row))

    if not comps:
        raise ValueError("No aligned IR rows found across summaries")

    fail_counts_np = np.asarray(fail_counts, dtype=np.int64)
    unique_counts = sorted(int(v) for v in np.unique(fail_counts_np))
    cmap = plt.get_cmap("Greys", max(1, len(unique_counts)))
    count_to_color = {c: cmap(i) for i, c in enumerate(unique_counts)}
    colors = [count_to_color[int(c)] for c in fail_counts_np]

    D_mu = np.array([c["D_mu"] for c in comps], dtype=np.float64)
    T0_mu = np.array([c["T0_mu"] for c in comps], dtype=np.float64)
    op_x = np.array([c["op_x"] for c in comps], dtype=np.float64)
    op_y = np.array([c["op_y"] for c in comps], dtype=np.float64)
    Ly = np.array([c["Ly"] for c in comps], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].scatter(D_mu, T0_mu, c=colors, s=28, alpha=0.9, edgecolors="none")
    axes[0].set_xlabel("D_mu")
    axes[0].set_ylabel("T0_mu")
    axes[0].set_title("D_mu vs T0_mu")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(op_x, op_y, c=colors, s=28, alpha=0.9, edgecolors="none")
    axes[1].set_xlabel("op_x")
    axes[1].set_ylabel("op_y")
    axes[1].set_title("op_x vs op_y")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(Ly, op_y, c=colors, s=28, alpha=0.9, edgecolors="none")
    axes[2].set_xlabel("Ly")
    axes[2].set_ylabel("op_y")
    axes[2].set_title("Ly vs op_y")
    axes[2].grid(True, alpha=0.3)

    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=count_to_color[c], label=f"fails={c}", markersize=7)
        for c in unique_counts
    ]
    axes[2].legend(handles=legend_handles, loc="best")

    fig.suptitle(f"IR Failure Count Visualization (threshold={args.success_threshold})")
    fig.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"Saved: {out_png}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--summaries",
        nargs="+",
        default=[
            "results/cmaes/l1_stft/summary.csv",
            "results/cmaes/l1_stft_v2/summary.csv",
            "results/cmaes/incremental_ablation/1_norm/summary.csv",
        ],
        help="List of summary.csv paths to compare",
    )
    p.add_argument("--success_threshold", type=float, default=0.01)
    p.add_argument("--out_png", type=str, default="results/diagnostics/visualize_fails/fail_scatter.png")
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
