"""Per-parameter error for checkpoints that were trained before it was logged.

train_encoder now writes perr_<param> at every eval, but a run already in flight
does not have it, and the question the column exists to answer is about runs
already in flight. Checkpoints carry the weights and the args that built them,
so the numbers are recoverable without retraining anything.

    python -m src.ddsp.diag_param_error --root results/ddsp/quiet3_ppre \
      --val-data-dir data/val-quiet3 --n-val 1000 \
      --chunk-elems 200000000 --fixed-mode-grid 30,92

WHY NOT corr AND spread, which the runs DO log. Both are invariant to a constant
offset, so a coordinate that is uniformly wrong by a third of its range prints
c=1.00 s=1.00, and neither carries a magnitude. On quiet3 the bad evals came in
three different shapes -- shrinkage (spread 0.458), over-dispersion (1.165) and
decorrelation (corr 0.663) -- and no combination of those two says which
parameter is wrong or by how much. err% below is the direct statement: the
median |estimate - truth| as a percentage of that parameter's search range.

No synthesis is involved. The encoder maps audio to z, and z maps to physical
parameters by a formula; the plate is never called, so this is fast and cannot
disagree with training over numerics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.cmaes.fit_7param_norm_es import PARAM_KEYS, PARAM_SPACE
from src.ddsp.train_encoder import Encoder, load_dataset, parse_mode_grid, z_to_dicts
from src.gd.graddescent import Raw7Space, param_sq_errs


def load_model(ckpt: Path, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck.get("args", {})
    if not a:
        return None, None
    model = Encoder(
        n_out=len(PARAM_KEYS), width=a.get("width", 32), n_fft=a.get("n_fft", 2048),
        hop=a.get("hop", 512), n_blocks=a.get("n_blocks", 5),
        max_ch=a.get("max_ch", 256), input_mode=a.get("input_mode", "norm_amp"),
        norm=a.get("norm", "group"), head_bound=a.get("head_bound", "tanh"),
        head_grad_floor=a.get("head_grad_floor", 0.05),
        head_cap=a.get("head_cap", 3.0),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, nargs="+", required=True,
                   help="Sweep directories; every arm subdirectory is read")
    p.add_argument("--ckpt", default="encoder_last.pt",
                   help="Which checkpoint per arm. encoder_best.pt is selected on "
                        "the best single eval, which under this much eval-to-eval "
                        "jitter favours whichever arm wandered most -- last is the "
                        "less flattering and more comparable choice.")
    p.add_argument("--val-data-dir", type=Path, required=True)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk-elems", type=int, default=200_000_000)
    p.add_argument("--mode-bucket", type=int, default=1024)
    p.add_argument("--fixed-mode-grid", type=parse_mode_grid, default=None)
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Only gt_z and the bounds are used -- load_dataset reads targets from disk
    # and the encoder is never asked to synthesize -- so the plate settings here
    # affect nothing but are accepted to match the training invocation.
    space = Raw7Space(dev, torch.float32, normalize=False)
    space.configure_plate(args.chunk_elems, False, True, False,
                          args.mode_bucket, args.fixed_mode_grid)
    z_val, x_val = load_dataset(space, args.val_data_dir, args.duration, dev,
                                limit=args.n_val)
    print(f"param space {PARAM_SPACE}   {x_val.shape[0]} val IRs   "
          f"checkpoint {args.ckpt}\n")

    arms = sorted(c for r in args.root for c in r.glob(f"*/{args.ckpt}"))
    if not arms:
        raise SystemExit(f"no {args.ckpt} under " + " ".join(str(r) for r in args.root))

    w = max(14, max(len(c.parent.name) for c in arms) + 2)
    head = f"{'arm':<{w}}{'step':>7}"
    for k in PARAM_KEYS:
        head += f"{k + ' err%':>16}"
    head += f"{'nmse':>10}"
    print(head)

    for c in arms:
        model, ck = load_model(c, dev)
        if model is None:
            print(f"{c.parent.name:<{w}}  no args in checkpoint")
            continue
        scale = ck.get("scale", 1.0)
        with torch.no_grad():
            zp = torch.cat([model(x_val[i:i + args.batch_size].to(dev), scale)
                            for i in range(0, x_val.shape[0], args.batch_size)])
        est = z_to_dicts(zp.cpu().numpy())
        gt = z_to_dicts(z_val.cpu().numpy())
        pe = np.array([param_sq_errs(e, g) for e, g in zip(est, gt)], dtype=np.float64)
        # sqrt of the median squared error, as a percentage of the range. Median
        # rather than mean to match what val_nmse_6d reports, with p90 beside it
        # because a tail is a different failure from a uniformly worse fit.
        row = f"{c.parent.name:<{w}}{ck.get('step', 0):>7}"
        for i in range(len(PARAM_KEYS)):
            med = 100.0 * np.sqrt(np.median(pe[:, i]))
            p90 = 100.0 * np.sqrt(np.percentile(pe[:, i], 90))
            row += f"{f'{med:.2f}/{p90:.2f}':>16}"
        row += f"{np.median(pe.mean(axis=1)):>10.5f}"
        print(row)

    print("\n  err% is median/p90 of |estimate - truth| as a percent of that "
          "parameter's\n  search range. For a log-scaled coordinate the range is "
          "measured in log units,\n  the same normalization z and nmse use. nmse is "
          "the median over IRs of the\n  mean squared error over parameters -- the "
          "headline number, for cross-check\n  against the training logs.")


if __name__ == "__main__":
    main()
