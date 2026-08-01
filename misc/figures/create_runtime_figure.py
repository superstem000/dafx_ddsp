"""
Plot per-IR Stage-1 runtime: concentration curve + broken-axis histogram.
=========================================================================

Two publication-style figures over the 100-IR validation set, both showing
that runtime is dominated by a small tail of hard IRs:

  (1) runtime_cumulative_tolfun.*
      A concentration (Lorenz-style) curve: cumulative share of total
      runtime vs. fraction of IRs, sorted fastest-first. The gap below the
      diagonal shows how few IRs account for most of the compute.

  (2) runtime_hist_linear_tolfun.*
      A LINEAR-axis histogram with a broken x-axis: the dense low-runtime
      bulk is shown at true linear spacing, and the far tail is shown in a
      separate right panel so its distance (not just its existence) is
      visible without the fast mode collapsing into one bar.

Reads:
  results/cmaes_norm_es/l1_stft_tolfun/summary.csv

Writes:
  figures/tolfun/runtime_cumulative_tolfun.{pdf,png}
  figures/tolfun/runtime_hist_linear_tolfun.{pdf,png}

Usage:
  python plot_runtime.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────
SUMMARY_CSV = Path("results/cmaes_norm_es/l1_stft_tolfun/summary.csv")
OUT_DIR     = Path("figures/tolfun")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not SUMMARY_CSV.exists():
    raise FileNotFoundError(
        f"Could not find {SUMMARY_CSV} (run from the repo root)."
    )

# ── Load ───────────────────────────────────────────────────────────────
df = pd.read_csv(SUMMARY_CSV).dropna(subset=["runtime"])
rt = df["runtime"].values.astype(np.float64)
n  = len(rt)
print(f"Loaded {n} IRs from {SUMMARY_CSV}")

mean_rt, med_rt = rt.mean(), np.median(rt)
print(f"  runtime:  median={med_rt:.1f}s   mean={mean_rt:.1f}s"
      f"   min={rt.min():.1f}s   max={rt.max():.1f}s")

# ── Shared style (matches mu-refinement figure) ────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.serif"       : ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size"        : 9,
    "axes.titlesize"   : 9,
    "axes.labelsize"   : 9,
    "xtick.labelsize"  : 8,
    "ytick.labelsize"  : 8,
    "legend.fontsize"  : 8,
    "axes.linewidth"   : 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size" : 3,
    "ytick.major.size" : 3,
    "pdf.fonttype"     : 42,
    "ps.fonttype"      : 42,
})

FIG_W = 3.4
LINE_COLOR = "#222222"
REF_COLOR  = "#888888"
BAR_FACE   = "#cccccc"
BAR_EDGE   = "#222222"
GRID_COLOR = "#dddddd"


# ────────────────────────────────────────────────────────────────────────
# (1) Concentration / Lorenz curve
# ────────────────────────────────────────────────────────────────────────
def plot_cumulative():
    rt_sorted = np.sort(rt)                       # ascending (fastest first)
    cum_frac  = np.concatenate([[0.0], np.cumsum(rt_sorted) / rt_sorted.sum()])
    ir_frac   = np.concatenate([[0.0], np.arange(1, n + 1) / n])

    # Concentration stat: share of total runtime from the slowest decile.
    k = max(1, int(round(0.10 * n)))
    top_share = rt_sorted[-k:].sum() / rt_sorted.sum()

    fig, ax = plt.subplots(figsize=(FIG_W, 2.6))

    ax.plot([0, 1], [0, 1], color=REF_COLOR, lw=0.8, linestyle="--",
            zorder=2, label="uniform")
    ax.plot(ir_frac, cum_frac, color=LINE_COLOR, lw=1.3, zorder=3,
            label="observed")
    ax.fill_between(ir_frac, cum_frac, ir_frac, color=LINE_COLOR,
                    alpha=0.07, zorder=1)

    # Mark the slowest-decile point.
    x_mark = 1.0 - k / n
    y_mark = 1.0 - top_share
    ax.plot([x_mark], [y_mark], "o", ms=3.5, color=LINE_COLOR, zorder=4)
    ax.annotate(
        f"slowest {int(round(100*k/n))}% of IRs\n"
        f"= {top_share*100:.0f}% of runtime",
        xy=(x_mark, y_mark), xytext=(0.30, 0.72),
        textcoords="axes fraction", fontsize=7.5, color=LINE_COLOR,
        arrowprops=dict(arrowstyle="-", lw=0.6, color=LINE_COLOR),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Fraction of IRs (fastest first)")
    ax.set_ylabel("Cumulative share of runtime")
    ax.grid(True, which="major", color=GRID_COLOR, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    plt.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"runtime_cumulative_tolfun.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close()


# ────────────────────────────────────────────────────────────────────────
# (2) Broken-axis linear histogram
# ────────────────────────────────────────────────────────────────────────
def plot_linear_broken():
    # Linear bins across the full range.
    BIN_W = 20.0
    hi    = np.ceil(rt.max() / BIN_W) * BIN_W
    bins  = np.arange(0, hi + BIN_W, BIN_W)

    # Break the x-axis between the dense bulk and the sparse tail. The left
    # panel holds the bulk at fine linear spacing, the right panel the far
    # outliers. From the data the bulk sits below ~260 s; the tail is a
    # handful of IRs above it.
    rt_sorted = np.sort(rt)
    left_max  = 260.0
    tail = rt_sorted[rt_sorted > left_max]
    right_min = (np.floor(tail.min() / BIN_W) * BIN_W) if tail.size else left_max
    right_max = hi + BIN_W          # pad so the last bar isn't flush to the edge

    fig, (axL, axR) = plt.subplots(
        1, 2, sharey=True, figsize=(FIG_W, 2.6),
        gridspec_kw={"width_ratios": [3.0, 1.15], "wspace": 0.07},
    )

    for ax in (axL, axR):
        ax.hist(rt, bins=bins, color=BAR_FACE, edgecolor=BAR_EDGE,
                linewidth=0.6, zorder=3)
        ax.grid(True, which="major", color=GRID_COLOR, lw=0.4, zorder=0)
        ax.set_axisbelow(True)

    axL.axvline(med_rt, color=LINE_COLOR, lw=1.0, linestyle="-", zorder=4,
                label=f"median ({med_rt:.0f} s)")
    axL.axvline(mean_rt, color=LINE_COLOR, lw=1.0, linestyle="--", zorder=4,
                label=f"mean ({mean_rt:.0f} s)")

    axL.set_xlim(0, left_max)
    axR.set_xlim(right_min, right_max)

    # Anchor the right panel's ticks so one lands at the tail max's hundred
    # (e.g. 900), making the far outlier bar readable rather than unlabeled.
    if tail.size:
        tick_hi = np.floor(rt.max() / 100.0) * 100.0
        ticks_R = np.arange(tick_hi, right_min, -200.0)[::-1]
        axR.set_xticks(ticks_R)

    # Hide the facing spines and the right panel's y ticks.
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.tick_params(axis="y", which="both", left=False)

    # Diagonal break marks (scaled so both look the same size).
    d = 0.015
    wl, wr = 3.0, 1.15
    kw = dict(color="k", clip_on=False, lw=0.7)
    axL.plot((1 - d, 1 + d), (-d, +d), transform=axL.transAxes, **kw)
    axL.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=axL.transAxes, **kw)
    dr = d * wl / wr
    axR.plot((-dr, +dr), (-d, +d), transform=axR.transAxes, **kw)
    axR.plot((-dr, +dr), (1 - d, 1 + d), transform=axR.transAxes, **kw)

    axL.set_ylabel("Number of IRs")
    axL.legend(loc="upper right", frameon=False, fontsize=8)
    for ax in (axL, axR):
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    # One shared x-label centred under both panels, placed manually so it
    # clears the tick labels (a broken axis is not tight_layout-compatible).
    fig.subplots_adjust(bottom=0.20, left=0.13, right=0.97, top=0.95)
    fig.text(0.55, 0.03, "Stage 1 runtime per IR (s)",
             ha="center", fontsize=9)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"runtime_hist_linear_tolfun.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close()


if __name__ == "__main__":
    plot_cumulative()
    plot_linear_broken()