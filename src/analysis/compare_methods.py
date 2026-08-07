"""One table: per-composite error for CMA-ES, the loss ladder, and the encoder.

Aggregates across IRs with the geometric mean by default, because the tail
dominates the arithmetic one and the two disagree by a lot: the 250k encoder
run has mean/median 19.8x, so its RMS is set by a minority of IRs rather than
by typical behaviour. Geomean is also what the CMA-ES pipeline is quoted in, so
the headline number here is directly comparable to its ~5e-6.

Every row is put at the same pipeline stage. For CMA-ES that means stage 2 when
mu_refined_summary.csv exists (the ternary scale stage) and stage 1 otherwise,
labelled either way rather than silently mixed. For the encoder it means
running the scale stage here, since history.json's per-coordinate nmse_{k} is
computed from the encoder's own mu and is therefore pre-ternary while its other
five composites are not.

No single scalar summarises these runs, so three views are printed. Per
composite, geomean across IRs and RMS across IRs -- the first describes the
typical IR, the second is set by the failures, and for one CMA-ES restart they
disagree about who wins by a factor of 130. Then the NMSE_6d quantiles, which
show why: with one restart the distribution has no middle, so any measure of
the middle is reporting a value no IR actually has. Its geomean sits 100x below
its own median.

    python -m src.analysis.compare_methods
    python -m src.analysis.compare_methods --ddsp-ckpt results/ddsp/sweep_L1_STFT/encoder_best.pt
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from src.mu_optimization.ternary_mu import COMPOSITE_BOUNDS

KEYS = ("mu", "D_div_mu", "T0_div_mu", "Ly", "op_x", "op_y")
NU = 0.25
FLOOR = 1e-30


def composites_from_raw(row, prefix=""):
    E, rho, h, Ly, T0, ox, oy = (
        float(row[prefix + k]) for k in ("E", "rho", "h", "Ly", "T0", "op_x", "op_y")
    )
    return {"mu": rho * h, "D_div_mu": E * h * h / (12.0 * (1.0 - NU ** 2) * rho),
            "T0_div_mu": T0 / (rho * h), "Ly": Ly, "op_x": ox, "op_y": oy}


def load_cmaes(run_dir: Path):
    """Post-ternary composites if stage 2 ran, else stage 1. Returns (est, gt, stage)."""
    s2, s1 = run_dir / "stage2/mu_refined_summary.csv", run_dir / "stage1/summary.csv"
    if s2.exists():
        rows = list(csv.DictReader(s2.open()))
        est = [{k: float(r["refined_" + k]) for k in KEYS} for r in rows]
        return est, [composites_from_raw(r, "gt_") for r in rows], "stage2"
    if s1.exists():
        rows = list(csv.DictReader(s1.open()))
        est = [{"mu": float(r["mu"]), "D_div_mu": float(r["D_mu"]),
                "T0_div_mu": float(r["T0_mu"]), "Ly": float(r["Ly"]),
                "op_x": float(r["op_x"]), "op_y": float(r["op_y"])} for r in rows]
        return est, [composites_from_raw(r, "gt_") for r in rows], "stage1"
    return None, None, None


def load_ddsp(ckpt_path: Path, batch_size: int, chunk_elems: int, n_val, fit_mu: bool):
    """Run the encoder and, unless told not to, the scale stage after it."""
    import torch

    from src.cmaes.fit_7param_norm_es import BOUNDS_HI_NP, BOUNDS_LO_NP, PARAM_KEYS
    from src.ddsp.train_encoder import Encoder, build_parser, fit_mu_scale, load_dataset
    from src.gd.graddescent import Raw7Space
    from src.loss.loss_selector import select_loss_function
    from src.mu_optimization.ternary_mu import seven_to_six

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = build_parser().parse_args([])
    for k, v in ck["args"].items():
        if hasattr(a, k):
            setattr(a, k, v)
    a.batch_size, a.chunk_elems = batch_size, chunk_elems
    if n_val is not None:
        a.n_val = n_val

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    space = Raw7Space(dev, torch.float32, normalize=False)
    space.configure_plate(a.chunk_elems, False, a.batched_plate, False, a.mode_bucket,
                          getattr(a, "fixed_mode_grid", None))
    z_va, x_va = load_dataset(space, Path(a.val_data_dir), a.duration, dev, a.n_val)

    model = Encoder(n_out=len(PARAM_KEYS), width=a.width, n_fft=a.n_fft, hop=a.hop,
                    n_blocks=a.n_blocks, max_ch=a.max_ch, input_mode=a.input_mode).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    loss_fn = select_loss_function(a.loss, sample_rate=44100, device=dev)
    R, H = PARAM_KEYS.index("rho"), PARAM_KEYS.index("h")

    phys, mus = [], []
    with torch.no_grad():
        for i in range(0, x_va.shape[0], a.batch_size):
            xb = x_va[i : i + a.batch_size]
            z = model(xb, float(ck["scale"]))
            ph = BOUNDS_LO_NP + ((z.cpu().numpy() + 1.0) / 2.0) * (BOUNDS_HI_NP - BOUNDS_LO_NP)
            phys.append(ph)
            if fit_mu:
                pred = space.forward(z, None, a.duration)
                mu_p = torch.as_tensor(ph[:, R] * ph[:, H], device=dev, dtype=pred.dtype)
                mus.append(fit_mu_scale(loss_fn, xb, pred, mu_p).cpu().numpy())
            else:
                mus.append(ph[:, R] * ph[:, H])

    ph, mu_hat = np.concatenate(phys), np.concatenate(mus)
    gt = BOUNDS_LO_NP + ((z_va.cpu().numpy() + 1.0) / 2.0) * (BOUNDS_HI_NP - BOUNDS_LO_NP)
    to6 = lambda row: seven_to_six({k: float(v) for k, v in zip(PARAM_KEYS, row)})
    est = [dict(to6(r), mu=float(m)) for r, m in zip(ph, mu_hat)]
    return est, [to6(r) for r in gt], ("post-mu" if fit_mu else "encoder mu")


def score(est, gt):
    """Per-composite error as % of bound range, both ways, plus the NMSE_6d spread.

    Geomean and RMS are both reported because the gap between them is a
    property of the method, not of the metric. 1-restart CMA-ES is bimodal --
    it either lands in the basin and recovers ground truth, or misses and fails
    outright -- so its mean sits ~3600x above its geomean. The encoder has no
    basin to miss and degrades gracefully, at ~12x. Reporting either alone
    tells half the story: on the typical IR CMA-ES wins by 19x, on the average
    IR the encoder wins by 4x.
    """
    geo, rms, sq = {}, {}, []
    for k in KEYS:
        lo, hi = COMPOSITE_BOUNDS[k]
        err = np.array([abs(e[k] - g[k]) / (hi - lo) for e, g in zip(est, gt)])
        sq.append(err ** 2)
        geo[k] = 100.0 * float(np.exp(np.mean(np.log(np.maximum(err, FLOOR)))))
        rms[k] = 100.0 * math.sqrt(float(np.mean(err ** 2)))
    n6 = np.mean(np.stack(sq), axis=0)
    return geo, rms, n6


# Categorical slots 1-6 of the reference palette, in fixed order. Validated for
# the adjacent pairlist a line chart uses: worst adjacent CVD dE 9.1, worst
# adjacent normal-vision dE 19.6. Three slots sit under 3:1 on the light
# surface, so the relief rule applies -- hence the legend, the direct labels on
# the right-hand panel, and the printed tables above the figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
LADDER = ["l1_stft", "mss", "smoothmss", "l1_stft_c2", "l1_stft_pow", "l1_stft_log"]


def plot(data, out_path: Path):
    """Log-x ECDF: the quantile function drawn, so no threshold has to be picked.

    Two panels because there are two claims. Left, the compression ladder at one
    CMA-ES restart -- the curves separate cleanly and monotonically. Right, the
    methods, where the shapes differ in kind rather than degree: one restart is
    a near-vertical rise at 1e-12 and a second at 1e-1 with nothing between,
    while the encoder is a single narrow band. That shape is exactly what makes
    a scalar summary of either misleading.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def ecdf(ax, v, color, label):
        v = np.sort(np.maximum(v, 1e-16))
        ax.step(v, np.arange(1, v.size + 1) / v.size, where="post",
                color=color, lw=2, label=label, solid_joinstyle="round")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)

    for i, name in enumerate(LADDER):
        key = next((k for k in data if k.endswith(name) and "1rst" in k), None)
        if key:
            ecdf(ax1, data[key], PALETTE[i], name)
    ax1.set_title("CMA-ES, one restart: the compression ladder", loc="left", fontsize=11)
    ax1.legend(fontsize=8.5, loc="lower right", frameon=False)

    panel2 = [(k, v) for k, v in data.items()
              if k.startswith("DDSP") or k.startswith("CMA-ES full")
              or k == "CMA-ES 1rst   l1_stft"]
    for i, (k, v) in enumerate(panel2):
        ecdf(ax2, v, PALETTE[i], k)
        vs = np.sort(np.maximum(v, 1e-16))
        ax2.annotate(k.replace("CMA-ES", "CMA-ES").strip(), (vs[-1], 1.0),
                     xytext=(4, -2 - 11 * i), textcoords="offset points",
                     fontsize=8.5, color="#52514e", va="top")
    ax2.set_title("Methods: shape, not just level", loc="left", fontsize=11)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.grid(True, which="both", lw=0.5, alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_xlabel("NMSE$_{6d}$  (log scale)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax1.set_ylabel("fraction of IRs at or below")
    ax1.set_ylim(0, 1.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"\nwrote {out_path} and {out_path.with_suffix('.pdf')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--ddsp-ckpt", type=Path, action="append", default=None,
                   help="Encoder checkpoint(s) to include; repeatable")
    p.add_argument("--no-fit-mu", action="store_true",
                   help="Report the encoder's own mu instead of running the scale stage")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk-elems", type=int, default=20_000_000)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--plot", type=Path, default=None,
                   help="Write a log-x ECDF of NMSE_6d here (PNG, plus a PDF alongside)")
    p.add_argument("--dump-csv", type=Path, default=None,
                   help="Write the raw per-IR NMSE_6d values, one column per method")
    args = p.parse_args()

    rows = []
    full = args.results / "standard_sweep/l1_stft"
    est, gt, stage = load_cmaes(full)
    if est:
        rows.append(("CMA-ES full   L1_STFT", stage, *score(est, gt)))
    ladder = args.results / "ladder_1restart"
    for d in sorted(x for x in ladder.glob("*") if x.is_dir()) if ladder.exists() else []:
        est, gt, stage = load_cmaes(d)
        if est:
            rows.append((f"CMA-ES 1rst   {d.name}", stage, *score(est, gt)))
    for ck in args.ddsp_ckpt or []:
        est, gt, stage = load_ddsp(ck, args.batch_size, args.chunk_elems,
                                   args.n_val, not args.no_fit_mu)
        rows.append((f"DDSP  {ck.parent.name}", stage, *score(est, gt)))

    rows.sort(key=lambda r: float(np.exp(np.mean(np.log(np.maximum(r[4], FLOOR))))))
    hdr = (f"{'method':26s} {'stage':>8s} {'n':>5s} " + " ".join(f"{k:>10s}" for k in KEYS))
    rule = "-" * 26 + " " + "-" * 8 + " " + "-" * 5 + " " + " ".join("-" * 10 for _ in KEYS)

    for title, idx in (("geometric mean across IRs -- the typical IR", 2),
                       ("RMS across IRs -- tail-weighted", 3)):
        print(f"\n|error| per composite, {title}, as % of that parameter's bound range")
        print(hdr); print(rule)
        for name, stage, geo, rms, _ in rows:
            per = geo if idx == 2 else rms
            print(f"{name:26s} {stage:>8s} {len(_):5d} " +
                  " ".join(f"{per[k]:10.3f}" for k in KEYS))

    # Quantiles rather than a pass/fail count: a threshold both picks an
    # arbitrary cut and says nothing about quality within a mode -- two losses
    # that never fail, one landing at 1e-14 and one at 1e-3, would score
    # identically. The quantile function carries the same information with no
    # cut to choose, and "spread" (decades from p10 to p90) is the
    # threshold-free bimodality measure: an optimizer that either recovers
    # ground truth or misses the basin entirely spans ten decades, one that is
    # uniformly mediocre spans two.
    print(f"\nNMSE_6d quantiles across IRs -- what the best 10%, 25%, ... achieve")
    print(f"{'method':26s} {'n':>5s} {'geo':>10s} " +
          " ".join(f"{q:>10s}" for q in ("p10", "p25", "p50", "p75", "p90")) +
          f" {'spread':>8s}")
    print("-" * 26 + " " + "-" * 5 + " " + " ".join("-" * 10 for _ in range(6)) + " " + "-" * 8)
    for name, stage, geo, rms, n6 in rows:
        g = float(np.exp(np.mean(np.log(np.maximum(n6, FLOOR)))))
        qs = np.percentile(n6, [10, 25, 50, 75, 90])
        dec = math.log10(max(qs[4], FLOOR) / max(qs[0], FLOOR))
        print(f"{name:26s} {len(n6):5d} {g:10.3e} " +
              " ".join(f"{q:10.3e}" for q in qs) + f" {dec:7.1f}d")

    if args.plot:
        plot({name: n6 for name, _, _, _, n6 in rows}, args.plot)
    if args.dump_csv:
        args.dump_csv.parent.mkdir(parents=True, exist_ok=True)
        names = [r[0] for r in rows]
        cols = [r[4] for r in rows]
        with args.dump_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(names)
            for i in range(max(len(c) for c in cols)):
                w.writerow(["" if i >= len(c) else f"{c[i]:.9e}" for c in cols])
        print(f"wrote {args.dump_csv}")

    print("\nspread is log10(p90/p10), in decades. It separates 'never fails, never")
    print("exact' from 'exact or nothing' without picking a threshold, and it is why")
    print("the two tables above disagree about who wins: geomean and median describe")
    print("the middle of a distribution that, for one restart, has no middle.")
    print("Rows are not on the same IRs: CMA-ES arms use their own evaluation sets and")
    print("the encoder uses its validation split. Compare within a column with care.\n")


if __name__ == "__main__":
    main()
