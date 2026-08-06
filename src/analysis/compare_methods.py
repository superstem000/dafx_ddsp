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

Worth knowing when reading the output: --stat geo reports the geometric mean of
|error| per coordinate, which is a typical-case figure, while NMSE_6d is a mean
of squares across coordinates first and then a geomean across IRs. They answer
different questions and will not agree; both are printed.

    python -m src.analysis.compare_methods
    python -m src.analysis.compare_methods --stat rms
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


def score(est, gt, stat):
    """Per-composite error as % of bound range, plus geomean NMSE_6d."""
    per, sq = {}, []
    for k in KEYS:
        lo, hi = COMPOSITE_BOUNDS[k]
        err = np.array([abs(e[k] - g[k]) / (hi - lo) for e, g in zip(est, gt)])
        sq.append(err ** 2)
        per[k] = 100.0 * (np.exp(np.mean(np.log(np.maximum(err, FLOOR)))) if stat == "geo"
                          else math.sqrt(float(np.mean(err ** 2))))
    n6 = np.mean(np.stack(sq), axis=0)
    per["NMSE_6d"] = float(np.exp(np.mean(np.log(np.maximum(n6, FLOOR)))))
    return per, len(est)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--stat", choices=["geo", "rms"], default="geo",
                   help="How to aggregate |error| across IRs per composite")
    p.add_argument("--ddsp-ckpt", type=Path, action="append", default=None,
                   help="Encoder checkpoint(s) to include; repeatable")
    p.add_argument("--no-fit-mu", action="store_true",
                   help="Report the encoder's own mu instead of running the scale stage")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk-elems", type=int, default=20_000_000)
    p.add_argument("--n-val", type=int, default=None)
    args = p.parse_args()

    rows = []
    full = args.results / "standard_sweep/l1_stft"
    est, gt, stage = load_cmaes(full)
    if est:
        rows.append(("CMA-ES full   L1_STFT", stage, *score(est, gt, args.stat)))
    ladder = args.results / "ladder_1restart"
    for d in sorted(x for x in ladder.glob("*") if x.is_dir()) if ladder.exists() else []:
        est, gt, stage = load_cmaes(d)
        if est:
            rows.append((f"CMA-ES 1rst   {d.name}", stage, *score(est, gt, args.stat)))
    for ck in args.ddsp_ckpt or []:
        est, gt, stage = load_ddsp(ck, args.batch_size, args.chunk_elems,
                                   args.n_val, not args.no_fit_mu)
        rows.append((f"DDSP  {ck.parent.name}", stage, *score(est, gt, args.stat)))

    label = "geometric mean" if args.stat == "geo" else "RMS"
    print(f"\n|error| per composite, {label} across IRs, as % of that parameter's bound range")
    print(f"{'method':26s} {'stage':>8s} {'n':>5s} " +
          " ".join(f"{k:>10s}" for k in KEYS) + f" {'NMSE_6d':>10s}")
    print("-" * 26 + " " + "-" * 8 + " " + "-" * 5 + " " +
          " ".join("-" * 10 for _ in KEYS) + " " + "-" * 10)
    for name, stage, per, n in sorted(rows, key=lambda r: r[2]["NMSE_6d"]):
        print(f"{name:26s} {stage:>8s} {n:>5d} " +
              " ".join(f"{per[k]:10.3f}" for k in KEYS) + f" {per['NMSE_6d']:10.3e}")
    print("\nNMSE_6d is the geometric mean across IRs of the per-IR mean-of-squares over")
    print("the six composites -- the statistic CMA-ES is quoted in. The per-composite")
    print("columns aggregate |error| directly, so they answer a different question and")
    print("will not reproduce it.")
    print("Rows are not on the same IRs: CMA-ES arms use their own evaluation sets and")
    print("the encoder uses its validation split. Compare within a column with care.\n")


if __name__ == "__main__":
    main()
