"""
Loss Comparison Dashboard
==========================

Reads results/cmaes/{loss_name}/*.csv and creates a unified comparison
of all losses across all IRs.

Usage:
  python compare_all_losses.py
  python compare_all_losses.py --results_dir cmaes --output loss_comparison
"""

import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NU = 0.25
PARAM_KEYS = ['E', 'rho', 'h', 'Ly', 'T0', 'op_x', 'op_y']
COMPOSITE_KEYS = ['mu', 'D_mu', 'T0_mu', 'Ly', 'op_x', 'op_y']
COMPOSITE_BOUNDS = {
    'mu': (2.43, 106.15), 'D_mu': (0.28, 201.19), 'T0_mu': (0.000094, 411.52),
    'Ly': (1.1, 4.0), 'op_x': (0.51, 1.0), 'op_y': (0.51, 1.0),
}


def to_composite(row):
    mu = row['rho'] * row['h']
    D = (row['E'] * row['h']**3) / (12 * (1 - NU**2))
    return {'mu': mu, 'D_mu': D/mu, 'T0_mu': row['T0']/mu,
            'Ly': row['Ly'], 'op_x': row['op_x'], 'op_y': row['op_y']}


def to_composite_gt(row):
    mu = row['gt_rho'] * row['gt_h']
    D = (row['gt_E'] * row['gt_h']**3) / (12 * (1 - NU**2))
    return {'mu': mu, 'D_mu': D/mu, 'T0_mu': row['gt_T0']/mu,
            'Ly': row['gt_Ly'], 'op_x': row['gt_op_x'], 'op_y': row['gt_op_y']}


def norm_error(est, gt, key):
    lo, hi = COMPOSITE_BOUNDS[key]
    return abs((est - lo) / (hi - lo) - (gt - lo) / (hi - lo))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/cmaes")
    p.add_argument("--output", type=str, default="loss_comparison")
    p.add_argument("--converge_threshold", type=float, default=0.02,
                   help="NMSE threshold to consider 'converged'")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover all losses
    loss_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(loss_dirs)} losses: {[d.name for d in loss_dirs]}")

    # Load all data
    all_data = []
    for loss_dir in loss_dirs:
        loss_name = loss_dir.name
        csv_files = sorted(loss_dir.glob("*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    if not row.get('gt_available', False):
                        continue
                    est_6 = to_composite(row)
                    gt_6 = to_composite_gt(row)
                    per_param = {k: norm_error(est_6[k], gt_6[k], k) for k in COMPOSITE_KEYS}
                    nmse = row.get('nmse', np.mean([per_param[k]**2 for k in COMPOSITE_KEYS]))

                    all_data.append({
                        'loss': loss_name,
                        'filename': row['filename'],
                        'nmse': nmse,
                        'best_loss': row['best_loss'],
                        'runtime': row.get('runtime', 0),
                        'converged': nmse < args.converge_threshold,
                        **{f'err_{k}': per_param[k] for k in COMPOSITE_KEYS},
                    })
            except Exception as e:
                print(f"  Error reading {csv_file}: {e}")

    df = pd.DataFrame(all_data)
    print(f"Total results: {len(df)}")
    print(f"Losses: {df['loss'].nunique()}, IRs: {df['filename'].nunique()}")

    # ── Summary stats per loss ──
    summary = []
    for loss_name in sorted(df['loss'].unique()):
        sub = df[df['loss'] == loss_name]
        n = len(sub)
        n_conv = sub['converged'].sum()
        summary.append({
            'loss': loss_name,
            'n_irs': n,
            'converged': n_conv,
            'conv_rate': n_conv / n if n > 0 else 0,
            'mean_nmse': sub['nmse'].mean(),
            'median_nmse': sub['nmse'].median(),
            'mean_runtime': sub['runtime'].mean(),
            **{f'mean_err_{k}': sub[f'err_{k}'].mean() for k in COMPOSITE_KEYS},
            **{f'median_err_{k}': sub[f'err_{k}'].median() for k in COMPOSITE_KEYS},
        })

    sdf = pd.DataFrame(summary).sort_values('conv_rate', ascending=False)
    sdf.to_csv(out_dir / "summary.csv", index=False)

    print(f"\n{'Loss':>20s} | {'IRs':>4s} {'Conv':>4s} {'Rate':>6s} | {'Mean NMSE':>10s} {'Med NMSE':>10s}")
    print("  " + "-" * 65)
    for _, r in sdf.iterrows():
        print(f"  {r['loss']:>18s} | {r['n_irs']:>4.0f} {r['converged']:>4.0f} {r['conv_rate']:>5.0%} | "
              f"{r['mean_nmse']:>10.4f} {r['median_nmse']:>10.4f}")

    loss_names = sdf['loss'].tolist()
    n_losses = len(loss_names)

    # ═══════════════════════════════════════════════════════════════
    # FIGURE 1: Convergence rate + mean NMSE bar chart
    # ═══════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, n_losses * 0.8), 8))

    x = np.arange(n_losses)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_losses))

    # Convergence rate
    rates = sdf['conv_rate'].values
    bars = ax1.bar(x, rates, color=colors, edgecolor='black', linewidth=0.5)
    for bar, rate, n_conv, n_ir in zip(bars, rates, sdf['converged'].values, sdf['n_irs'].values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{rate:.0%}\n({n_conv:.0f}/{n_ir:.0f})', ha='center', fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(loss_names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Convergence Rate')
    ax1.set_title(f'Convergence Rate (NMSE < {args.converge_threshold})', fontweight='bold')
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, alpha=0.3, axis='y')

    # Median NMSE
    med_nmse = sdf['median_nmse'].values
    bars = ax2.bar(x, med_nmse, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, med_nmse):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', fontsize=7, rotation=90)
    ax2.set_xticks(x)
    ax2.set_xticklabels(loss_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Median NMSE')
    ax2.set_title('Median NMSE (lower = better)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / "convergence_and_nmse.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: convergence_and_nmse.png")

    # ═══════════════════════════════════════════════════════════════
    # FIGURE 2: Per-parameter error by loss (heatmap)
    # ═══════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(5, n_losses * 0.4)))

    # Mean error heatmap
    matrix_mean = np.array([[sdf[sdf['loss'] == ln][f'mean_err_{k}'].values[0]
                             for k in COMPOSITE_KEYS] for ln in loss_names])
    im1 = ax1.imshow(matrix_mean, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(COMPOSITE_KEYS)))
    ax1.set_xticklabels(COMPOSITE_KEYS, fontsize=10, fontweight='bold')
    ax1.set_yticks(range(n_losses))
    ax1.set_yticklabels(loss_names, fontsize=9)
    for i in range(n_losses):
        for j in range(len(COMPOSITE_KEYS)):
            ax1.text(j, i, f'{matrix_mean[i,j]:.3f}', ha='center', va='center', fontsize=8)
    ax1.set_title('Mean Per-Param Error (lower = better)', fontweight='bold')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Median error heatmap
    matrix_med = np.array([[sdf[sdf['loss'] == ln][f'median_err_{k}'].values[0]
                            for k in COMPOSITE_KEYS] for ln in loss_names])
    im2 = ax2.imshow(matrix_med, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(COMPOSITE_KEYS)))
    ax2.set_xticklabels(COMPOSITE_KEYS, fontsize=10, fontweight='bold')
    ax2.set_yticks(range(n_losses))
    ax2.set_yticklabels(loss_names, fontsize=9)
    for i in range(n_losses):
        for j in range(len(COMPOSITE_KEYS)):
            ax2.text(j, i, f'{matrix_med[i,j]:.3f}', ha='center', va='center', fontsize=8)
    ax2.set_title('Median Per-Param Error (lower = better)', fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    plt.suptitle('Per-Parameter Error by Loss', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "per_param_error.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_param_error.png")

    # ═══════════════════════════════════════════════════════════════
    # FIGURE 3: Per-IR NMSE across losses (which IRs are hard?)
    # ═══════════════════════════════════════════════════════════════
    ir_names = sorted(df['filename'].unique())
    n_irs = len(ir_names)

    # Build matrix: losses × IRs
    nmse_matrix = np.full((n_losses, n_irs), np.nan)
    for i, ln in enumerate(loss_names):
        for j, ir in enumerate(ir_names):
            sub = df[(df['loss'] == ln) & (df['filename'] == ir)]
            if len(sub) > 0:
                nmse_matrix[i, j] = sub['nmse'].values[0]

    fig, ax = plt.subplots(figsize=(max(14, n_irs * 0.3), max(5, n_losses * 0.4)))
    im = ax.imshow(np.log10(nmse_matrix + 1e-8), cmap='RdYlGn_r', aspect='auto',
                   vmin=-4, vmax=-0.5)
    ax.set_xticks(range(n_irs))
    ax.set_xticklabels([ir.replace('random_IR_', '').replace('.npz', '') for ir in ir_names],
                       rotation=90, fontsize=6)
    ax.set_yticks(range(n_losses))
    ax.set_yticklabels(loss_names, fontsize=9)
    ax.set_xlabel('IR')
    plt.colorbar(im, ax=ax, label='log10(NMSE)', shrink=0.8)
    ax.set_title('Per-IR NMSE by Loss (green = good, red = bad)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "per_ir_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_ir_heatmap.png")

    # ═══════════════════════════════════════════════════════════════
    # FIGURE 4: NMSE distribution per loss (box plot)
    # ═══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(max(12, n_losses * 0.8), 5))
    box_data = [df[df['loss'] == ln]['nmse'].values for ln in loss_names]
    bp = ax.boxplot(box_data, labels=loss_names, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_yscale('log')
    ax.set_ylabel('NMSE (log scale)')
    ax.set_title('NMSE Distribution by Loss', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / "nmse_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: nmse_boxplot.png")

    # ═══════════════════════════════════════════════════════════════
    # FIGURE 5: Which IRs does each loss fail on?
    # ═══════════════════════════════════════════════════════════════
    # Count how many losses converge per IR
    ir_conv_counts = {}
    for ir in ir_names:
        sub = df[df['filename'] == ir]
        ir_conv_counts[ir] = sub['converged'].sum()

    # Sort IRs by difficulty (fewer losses converge = harder)
    ir_sorted = sorted(ir_names, key=lambda x: ir_conv_counts[x])

    fig, ax = plt.subplots(figsize=(max(12, n_losses * 0.8), max(5, min(n_irs, 30) * 0.25)))
    conv_matrix = np.zeros((len(ir_sorted[:30]), n_losses))  # Cap at 30 hardest IRs
    for j, ln in enumerate(loss_names):
        for i, ir in enumerate(ir_sorted[:30]):
            sub = df[(df['loss'] == ln) & (df['filename'] == ir)]
            if len(sub) > 0:
                conv_matrix[i, j] = 1 if sub['converged'].values[0] else 0

    im = ax.imshow(conv_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(n_losses))
    ax.set_xticklabels(loss_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(ir_sorted[:30])))
    ax.set_yticklabels([ir.replace('random_IR_', '').replace('.npz', '') for ir in ir_sorted[:30]],
                       fontsize=7)
    ax.set_ylabel('IR (sorted by difficulty)')
    ax.set_title('Convergence per IR per Loss (green=converged, red=failed)\nSorted: hardest IRs at top',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "failure_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: failure_matrix.png")

    # ═══════════════════════════════════════════════════════════════
    # FIGURE 6: Complementarity — do different losses fail on different IRs?
    # ═══════════════════════════════════════════════════════════════
    # For each pair of losses, count IRs where at least one converges
    if n_losses >= 2:
        union_matrix = np.zeros((n_losses, n_losses))
        for i, ln1 in enumerate(loss_names):
            for j, ln2 in enumerate(loss_names):
                for ir in ir_names:
                    s1 = df[(df['loss'] == ln1) & (df['filename'] == ir)]
                    s2 = df[(df['loss'] == ln2) & (df['filename'] == ir)]
                    c1 = s1['converged'].values[0] if len(s1) > 0 else False
                    c2 = s2['converged'].values[0] if len(s2) > 0 else False
                    if c1 or c2:
                        union_matrix[i, j] += 1

        fig, ax = plt.subplots(figsize=(max(8, n_losses * 0.7), max(6, n_losses * 0.6)))
        im = ax.imshow(union_matrix / len(ir_names), cmap='YlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(n_losses))
        ax.set_xticklabels(loss_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(n_losses))
        ax.set_yticklabels(loss_names, fontsize=9)
        for i in range(n_losses):
            for j in range(n_losses):
                val = union_matrix[i, j] / len(ir_names)
                ax.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=8)
        ax.set_title('Union Convergence Rate (if we pick best of each pair)\nDiagonal = single loss rate',
                     fontweight='bold')
        plt.colorbar(im, ax=ax, label='Rate', shrink=0.8)
        plt.tight_layout()
        plt.savefig(out_dir / "complementarity.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: complementarity.png")

    print(f"\n  All outputs in: {out_dir}/")


if __name__ == "__main__":
    main()