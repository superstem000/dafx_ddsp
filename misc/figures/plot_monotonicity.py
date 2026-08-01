"""
Plot landscape features vs CMA-ES convergence rate.
====================================================

Produces a 3-panel publication-style figure:
  (a) Monotonicity        vs convergence rate
  (b) Pearson  FDC (7D)   vs convergence rate
  (c) Spearman FDC (7D)   vs convergence rate
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────
ELA_CSV  = Path("ela_results/ela_summary.csv")
FDC_CSV  = Path("fdc_results/fdc_summary.csv")
CONV_CSV = Path("results/cmaes/loss_comparison/summary.csv")
OUT_DIR  = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOSSES = [
    "L1_STFT", "ESR", "GroupDelay", "ComplexSTFT", "Gammatone",
    "CQT+LogDec", "SC+LogMag", "VQT", "Envelope", "InstFreq",
    "MSS", "Dispersion", "CQT_L1", "Mel",
]

PRETTY = {
    "L1_STFT": "L1 STFT", "ESR": "ESR", "GroupDelay": "GroupDelay",
    "ComplexSTFT": "ComplexSTFT", "Gammatone": "Gammatone",
    "CQT+LogDec": "CQT+LogDec", "SC+LogMag": "SC+LogMag", "VQT": "VQT",
    "Envelope": "Envelope", "InstFreq": "InstFreq", "MSS": "MSS",
    "Dispersion": "Dispersion", "CQT_L1": "CQT L1", "Mel": "Mel",
}

ANNOTATE_OFFSETS = {
    "L1_STFT":    (-6, 4),
    "ESR":        (6, -2),
    "GroupDelay": (6, 2),
    "Dispersion": (6, -2),
}
ANNOTATE_HA = {
    "L1_STFT":    "right",
}
ANNOTATE = set(ANNOTATE_OFFSETS.keys())


# CSVs disagree on capitalization and separators. Canonicalize.
def canon(name):
    return str(name).lower().replace("+", "_")


# ── Load CSVs ──────────────────────────────────────────────────────────
ela  = pd.read_csv(ELA_CSV)
fdc  = pd.read_csv(FDC_CSV)
conv = pd.read_csv(CONV_CSV)

for df_ in (ela, fdc, conv):
    df_["_key"] = df_["loss"].apply(canon)

print("Loss keys present in each CSV:")
print(f"  ELA:  {sorted(ela['_key'].tolist())}")
print(f"  FDC:  {sorted(fdc['_key'].tolist())}")
print(f"  Conv: {sorted(conv['_key'].tolist())}")
print()

ela_l  = ela.set_index("_key")
fdc_l  = fdc.set_index("_key")
conv_l = conv.set_index("_key")

rows = []
for L in LOSSES:
    key = canon(L)
    missing = []
    if key not in ela_l.index:  missing.append("ela")
    if key not in fdc_l.index:  missing.append("fdc")
    if key not in conv_l.index: missing.append("conv")
    if missing:
        print(f"[WARN] {L} (key={key}): missing in {missing}")
        continue
    rows.append({
        "loss"      : L,
        "label"     : PRETTY[L],
        "mono"      : ela_l.loc[key, "monotonicity"],
        "fdc_p7"    : fdc_l.loc[key, "fdc_pearson_7d"],
        "fdc_s7"    : fdc_l.loc[key, "fdc_spearman_7d"],
        "conv_rate" : conv_l.loc[key, "conv_rate"],
    })
df = pd.DataFrame(rows)
print(f"\nLoaded {len(df)} losses.")


# ── Stats ──────────────────────────────────────────────────────────────
def pearson(x, y):
    r, p = stats.pearsonr(x, y)
    return r, p

stats_mono = pearson(df["mono"],   df["conv_rate"])
stats_p7   = pearson(df["fdc_p7"], df["conv_rate"])
stats_s7   = pearson(df["fdc_s7"], df["conv_rate"])

print(f"\nPearson correlations vs conv_rate:")
print(f"  Monotonicity:       r={stats_mono[0]:+.3f}, p={stats_mono[1]:.4f}")
print(f"  Pearson FDC (7D):   r={stats_p7[0]:+.3f}, p={stats_p7[1]:.4f}")
print(f"  Spearman FDC (7D):  r={stats_s7[0]:+.3f}, p={stats_s7[1]:.4f}")


# ── Plotting style ─────────────────────────────────────────────────────
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

FIG_W = 7.0
FIG_H = 2.7

fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), sharey=True)

panels = [
    ("mono",   "Monotonicity",      stats_mono, axes[0], "a"),
    ("fdc_p7", "Pearson FDC (7D)",  stats_p7,   axes[1], "b"),
    ("fdc_s7", "Spearman FDC (7D)", stats_s7,   axes[2], "c"),
]

POINT_COLOR     = "#222222"
ANNOTATE_COLOR  = "#222222"
FIT_COLOR       = "#888888"
GRID_COLOR      = "#dddddd"

for col, xlabel, (r, p), ax, tag in panels:
    x = df[col].values
    y = df["conv_rate"].values

    ax.scatter(x, y, s=22, color=POINT_COLOR, zorder=3, edgecolors="none")

    xs = np.linspace(x.min(), x.max(), 50)
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(xs, slope * xs + intercept,
            color=FIT_COLOR, lw=0.8, linestyle="--", zorder=2)

    for _, row in df.iterrows():
        if row["loss"] in ANNOTATE:
            dx, dy = ANNOTATE_OFFSETS.get(row["loss"], (4, 3))
            ha = ANNOTATE_HA.get(row["loss"], "left")
            ax.annotate(
                row["label"],
                xy=(row[col], row["conv_rate"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.5,
                color=ANNOTATE_COLOR,
                ha=ha,
            )

    # Title: panel tag + r,p stats. Build math string carefully to avoid
    # adjacent '$' tokens that break mathtext.
    sig = "^{*}" if p < 0.05 else ""
    title_text = rf"({tag})  $r = {r:+.2f}{sig},\; p = {p:.3f}$"
    ax.set_title(title_text, fontsize=9, pad=6, loc="left")

    ax.set_xlabel(xlabel)
    ax.grid(True, color=GRID_COLOR, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

axes[0].set_ylabel("CMA-ES convergence rate")

y_min = df["conv_rate"].min() - 0.06
y_max = df["conv_rate"].max() + 0.06
for ax in axes:
    ax.set_ylim(y_min, y_max)

plt.tight_layout(pad=0.4, w_pad=0.8)

out_pdf = OUT_DIR / "landscape_vs_convergence.pdf"
out_png = OUT_DIR / "landscape_vs_convergence.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"\nWrote {out_pdf}")
print(f"Wrote {out_png}")
plt.close()