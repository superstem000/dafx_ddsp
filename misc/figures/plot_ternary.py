"""
Plot per-IR NMSE before vs after ternary mu refinement.
========================================================

Produces a publication-style log-log scatter showing every IR's NMSE
before and after the stage-2 ternary refinement, with a y=x diagonal
for reference. Points below the diagonal = improvement.

Reads:
  results/mu_refinement/ternary/mu_refined_summary.csv

Writes:
  figures/mu_refinement_scatter.pdf
  figures/mu_refinement_scatter.png

Usage:
  python plot_mu_refinement.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────
SUMMARY_CSV = Path("results/mu_refinement/ternary_tolfun/mu_refined_summary.csv")
OUT_DIR     = Path("figures/tolfun")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────
df = pd.read_csv(SUMMARY_CSV).dropna(subset=["nmse_orig", "nmse_refined"])
print(f"Loaded {len(df)} IRs from {SUMMARY_CSV}")

orig = df["nmse_orig"].values.astype(np.float64)
refn = df["nmse_refined"].values.astype(np.float64)

# Floor any zeros so they show on a log plot
EPS = 0
orig = np.clip(orig, EPS, None)
refn = np.clip(refn, EPS, None)

# Aggregate stats for caption / debug
mean_o, mean_r = orig.mean(), refn.mean()
med_o,  med_r  = np.median(orig), np.median(refn)
improved      = int((refn < orig).sum())
print(f"  NMSE_orig:    mean={mean_o:.3e}   median={med_o:.3e}")
print(f"  NMSE_refined: mean={mean_r:.3e}   median={med_r:.3e}")
print(f"  IRs improved: {improved}/{len(df)}")


# ── Plot style (matches landscape figure) ──────────────────────────────
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

# Single-column width. ~3.4 in matches typical conference column width.
FIG_W = 3.4
FIG_H = 3.0

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

POINT_COLOR = "#222222"
DIAG_COLOR  = "#888888"
GRID_COLOR  = "#dddddd"

# Common axis range with breathing room (in log space)
lo = min(orig.min(), refn.min()) * 0.3
hi = max(orig.max(), refn.max()) * 3.0

# Diagonal y = x
diag = np.array([lo, hi])
ax.plot(diag, diag, color=DIAG_COLOR, lw=0.8, linestyle="--",
        zorder=2, label=r"$y = x$")

# Scatter
ax.scatter(orig, refn, s=14, color=POINT_COLOR, alpha=0.6,
           edgecolors="none", zorder=3)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel(r"NMSE before $\mu$ refinement")
ax.set_ylabel(r"NMSE after $\mu$ refinement")

ax.grid(True, which="major", color=GRID_COLOR, lw=0.4, zorder=0)
ax.grid(True, which="minor", color=GRID_COLOR, lw=0.25, zorder=0, alpha=0.6)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(0.6)

ax.legend(loc="upper left", frameon=False, fontsize=8)

plt.tight_layout(pad=0.4)

out_pdf = OUT_DIR / "mu_refinement_scatter_tolfun_v2.pdf"
out_png = OUT_DIR / "mu_refinement_scatter_tolfun_v2.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"\nWrote {out_pdf}")
print(f"Wrote {out_png}")
plt.close()