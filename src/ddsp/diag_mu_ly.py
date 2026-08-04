"""Is the scale stage absorbing Ly's error into mu?

Under --peak-normalize the training loss sees no absolute level at all, and
level enters the synthesis only as 1/(0.25*mu*Lx*Ly) -- so only the *product*
mu*Ly was ever visible in amplitude. Two consequences worth telling apart:

  * Ly loses the amplitude cue and must be recovered from mode frequencies
    alone, which is a harder route;
  * the scale stage, free to choose mu after the fact, can cancel whatever Ly
    error remains. An encoder with Ly 30% high still matches the level exactly
    by putting mu 30% low, at zero cost to the loss.

If the second is happening, mu's recovered value is not measuring mu -- it is
measuring mu*Ly divided by a wrong Ly, and mu|dlog| will sit on a floor set by
Ly's error no matter how good the shape fit becomes. The signature is a
correlation near -1 between the two log errors, a regression slope near -1, and
a product error much smaller than either factor's.

Runs safely alongside a training job: chunking and batch size default well
below the training run's, since the saved args carry --chunk-elems 1e9 which
asks for ~4 GB chunks and OOMs against a job already holding the card.

    python -m src.ddsp.diag_mu_ly
    python -m src.ddsp.diag_mu_ly --ckpt results/ddsp/other/encoder_best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.cmaes.fit_7param_norm_es import BOUNDS_HI_NP, BOUNDS_LO_NP, PARAM_KEYS
from src.ddsp.train_encoder import Encoder, build_parser, fit_mu_scale, load_dataset
from src.gd.graddescent import Raw7Space
from src.loss.loss_selector import select_loss_function

R, H, L = PARAM_KEYS.index("rho"), PARAM_KEYS.index("h"), PARAM_KEYS.index("Ly")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", type=Path,
                   default=Path("results/ddsp/l1_stft_peaknorm/encoder_last.pt"))
    p.add_argument("--batch-size", type=int, default=16,
                   help="Smaller than training's, to share the GPU")
    p.add_argument("--chunk-elems", type=int, default=20_000_000,
                   help="Modal-sum chunk; the training default of 1e9 will OOM here")
    p.add_argument("--n-val", type=int, default=None, help="Override for a quicker read")
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

    phys, mu_fit = [], []
    with torch.no_grad():
        for i in range(0, x_va.shape[0], a.batch_size):
            xb = x_va[i : i + a.batch_size]
            z = model(xb, float(ck["scale"]))
            pred = space.forward(z, None, a.duration)
            ph = BOUNDS_LO_NP + ((z.cpu().numpy() + 1.0) / 2.0) * (BOUNDS_HI_NP - BOUNDS_LO_NP)
            mu_p = torch.as_tensor(ph[:, R] * ph[:, H], device=dev, dtype=pred.dtype)
            mu_fit.append(fit_mu_scale(loss_fn, xb, pred, mu_p).cpu().numpy())
            phys.append(ph)

    ph, muf = np.concatenate(phys), np.concatenate(mu_fit)
    gt = BOUNDS_LO_NP + ((z_va.cpu().numpy() + 1.0) / 2.0) * (BOUNDS_HI_NP - BOUNDS_LO_NP)
    d_mu = np.log(muf) - np.log(gt[:, R] * gt[:, H])
    d_ly = np.log(ph[:, L]) - np.log(gt[:, L])
    d_mu_enc = np.log(ph[:, R] * ph[:, H]) - np.log(gt[:, R] * gt[:, H])

    print(f"{args.ckpt}   step {ck['step']}   n_val {len(d_mu)}")
    print(f"  corr(dlog mu_fit, dlog Ly_pred)   {np.corrcoef(d_mu, d_ly)[0, 1]:+.3f}"
          f"   slope {np.polyfit(d_ly, d_mu, 1)[0]:+.3f}")
    print(f"  median |dlog mu_fit|              {np.median(np.abs(d_mu)):.4f}")
    print(f"  median |dlog Ly_pred|             {np.median(np.abs(d_ly)):.4f}")
    print(f"  median |dlog (mu_fit * Ly_pred)|  {np.median(np.abs(d_mu + d_ly)):.4f}")
    print(f"  median |dlog mu_encoder|          {np.median(np.abs(d_mu_enc)):.4f}"
          f"   (untrained under --peak-normalize; for reference only)")
    print()
    if np.corrcoef(d_mu, d_ly)[0, 1] < -0.7 and \
            np.median(np.abs(d_mu + d_ly)) < 0.5 * np.median(np.abs(d_mu)):
        print("  CONFIRMED: mu is absorbing Ly's error. mu|dlog| is floored by Ly and")
        print("  will not improve with training. The product mu*Ly is what the pipeline")
        print("  actually determines; splitting it needs either Ly fitted jointly in the")
        print("  scale stage or a level term retained in the loss.")
    else:
        print("  Not the confound: mu's error is its own, so mu|dlog| should fall as the")
        print("  shape fit improves. If it stays flat anyway, look elsewhere -- the")
        print("  ternary bracket, or the loss's sensitivity to scale.")


if __name__ == "__main__":
    main()
