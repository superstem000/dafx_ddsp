"""Per-coordinate accuracy for a trained encoder, post scale stage.

NMSE_6d normalizes by each parameter's bound range, not by its value, so the
headline number answers "how far across the search box" and not "within what
percent". Those diverge hard here: mu, D_div_mu and T0_div_mu span decades and
their truths sit near the bottom of that span -- 92% and 96% of val truths in
the lowest fifth for D_div_mu and T0_div_mu -- so an error that is unremarkable
as a fraction of range can be several hundred percent of the value. Both
columns are printed, because a paper needs the second and the metric reports
the first.

Why this cannot come out of history.json: the per-coordinate nmse_{k} rows
there are computed from the encoder's own mu, not the fitted one, so the mu
entry is pre-ternary and every share is normalized by a total that includes it.
The five other composites are unaffected -- the scale stage moves only mu and
leaves every ratio and position untouched -- but mu and the shares need the fit
rerun, and val_nmse_6d_postmu is a median over IRs of a mean over coordinates,
so the mu term is not separable from it.

Runs alongside a training job: batch and chunk default well below the training
run's, since the saved args carry --chunk-elems 1e9.

    python -m src.ddsp.report_params
    python -m src.ddsp.report_params --ckpt results/ddsp/other/encoder_best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.cmaes.fit_7param_norm_es import BOUNDS_HI_NP, BOUNDS_LO_NP, PARAM_KEYS
from src.ddsp.train_encoder import Encoder, build_parser, fit_mu_scale, load_dataset
from src.gd.graddescent import Raw7Space
from src.loss.loss_selector import select_loss_function
from src.mu_optimization.ternary_mu import COMPOSITE_BOUNDS, nmse_6d, seven_to_six

KEYS = ("mu", "D_div_mu", "T0_div_mu", "Ly", "op_x", "op_y")
R, H = PARAM_KEYS.index("rho"), PARAM_KEYS.index("h")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", type=Path,
                   default=Path("results/ddsp/l1_stft_tgtnorm/encoder_last.pt"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk-elems", type=int, default=20_000_000)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--no-fit-mu", action="store_true",
                   help="Report the encoder's own mu instead of the fitted one")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = build_parser().parse_args([])
    for k, v in ck["args"].items():
        if hasattr(a, k):
            setattr(a, k, v)
    a.batch_size, a.chunk_elems = args.batch_size, args.chunk_elems
    if args.n_val is not None:
        a.n_val = args.n_val

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    space = Raw7Space(dev, torch.float32, normalize=False)
    space.configure_plate(a.chunk_elems, False, a.batched_plate, False, a.mode_bucket)
    z_va, x_va = load_dataset(space, Path(a.val_data_dir), a.duration, dev, a.n_val)

    model = Encoder(n_out=len(PARAM_KEYS), width=a.width, n_fft=a.n_fft, hop=a.hop,
                    n_blocks=a.n_blocks, max_ch=a.max_ch, input_mode=a.input_mode).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    loss_fn = select_loss_function(a.loss, sample_rate=44100, device=dev)

    phys, mu_hat = [], []
    with torch.no_grad():
        for i in range(0, x_va.shape[0], a.batch_size):
            xb = x_va[i : i + a.batch_size]
            z = model(xb, float(ck["scale"]))
            ph = BOUNDS_LO_NP + ((z.cpu().numpy() + 1.0) / 2.0) * (BOUNDS_HI_NP - BOUNDS_LO_NP)
            phys.append(ph)
            if args.no_fit_mu:
                mu_hat.append(ph[:, R] * ph[:, H])
            else:
                pred = space.forward(z, None, a.duration)
                mu_p = torch.as_tensor(ph[:, R] * ph[:, H], device=dev, dtype=pred.dtype)
                mu_hat.append(fit_mu_scale(loss_fn, xb, pred, mu_p).cpu().numpy())

    ph, muf = np.concatenate(phys), np.concatenate(mu_hat)
    gt = BOUNDS_LO_NP + ((z_va.cpu().numpy() + 1.0) / 2.0) * (BOUNDS_HI_NP - BOUNDS_LO_NP)
    est6 = [dict(seven_to_six({k: float(v) for k, v in zip(PARAM_KEYS, row)}), mu=float(m))
            for row, m in zip(ph, muf)]
    gt6 = [seven_to_six({k: float(v) for k, v in zip(PARAM_KEYS, row)}) for row in gt]
    n6 = np.array([nmse_6d(e, g) for e, g in zip(est6, gt6)])

    tag = "encoder mu (no scale stage)" if args.no_fit_mu else "post scale stage"
    print(f"\n{args.ckpt}   step {ck['step']}   n_val {len(n6)}   [{tag}]")
    print(f"NMSE_6d   geo {np.exp(np.mean(np.log(np.maximum(n6,1e-30)))):.3e}   "
          f"med {np.median(n6):.3e}   mean {np.mean(n6):.3e}   "
          f"p90 {np.percentile(n6,90):.3e}   mean/med {np.mean(n6)/np.median(n6):.2f}x")
    print(f"  overall RMS error {100*np.sqrt(np.mean(n6)):.3f}% of range\n")

    rows = {}
    for k in KEYS:
        e = np.array([d[k] for d in est6], float)
        t = np.array([d[k] for d in gt6], float)
        lo, hi = COMPOSITE_BOUNDS[k]
        rows[k] = {
            "nmse": float(np.mean(((e - t) / (hi - lo)) ** 2)),
            "rel": float(np.median(np.abs(e / t - 1.0))),
            "dlog": float(np.median(np.abs(np.log(np.maximum(e, 1e-300))
                                           - np.log(np.maximum(t, 1e-300))))),
            "span": hi - lo,
            "botfrac": float(np.mean((t - lo) / (hi - lo) < 0.2)),
        }
    tot = sum(r["nmse"] for r in rows.values())
    print(f"{'composite':12s} {'rms %range':>11s} {'+- units':>11s} {'share':>7s} "
          f"{'med |rel|':>10s} {'med |dlog|':>11s} {'GT low 20%':>11s}")
    for k in sorted(KEYS, key=lambda k: -rows[k]["nmse"]):
        r = rows[k]
        rms = np.sqrt(r["nmse"])
        print(f"{k:12s} {100*rms:10.3f}% {rms*r['span']:11.4g} {100*r['nmse']/tot:6.1f}% "
              f"{100*r['rel']:9.2f}% {r['dlog']:11.4f} {100*r['botfrac']:10.0f}%")
    print("\n  rms %range is what NMSE_6d measures; med |rel| is the error a reader")
    print("  expects from '+-x%'. They diverge most where GT clusters at the bottom")
    print("  of a decade-spanning range, since a fixed fraction of range is then a")
    print("  large fraction of the value.")


if __name__ == "__main__":
    main()
