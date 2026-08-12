"""Is the eps ladder a single-variable axis, and where does each knee sit?

Two things have to be true before a sweep over eps means anything.

First, the ladder must agree with the arms already run. log(x + eps) at eps = 1
is log1p, which is C2; at eps = 1e-7 it is C1. So L1_STFT_eps1 must reproduce
L1_STFT_c2 and L1_STFT_eps1e7 must reproduce L1_STFT_log, to floating point. If
they do not, the rungs are not the same family as the arms they are meant to
interpolate and nothing about the sweep is comparable to the earlier results.

Second, eps is only interpretable relative to the magnitudes it meets. eps is a
knee: bins far above it are compressed logarithmically, bins far below it are
in the loss's flat region where the compressed value is ~log(eps) regardless of
the bin. Which regime a rung is in is therefore a property of the *data*, not
of eps alone -- train_encoder.peak_normalized() records that eps = 1e-7 sits at
percentile 96.2 of val-set bins unnormalized and 0.0 normalized, i.e. the same
number is "above nearly everything" or "below everything" depending only on the
scaling. This prints that percentile for every rung, so the sweep's x-axis can
be labelled by where the knee actually falls rather than by a bare exponent.

    python -m src.ddsp.diag_eps_ladder --n-val 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.ddsp.train_encoder import load_dataset, peak_normalized
from src.gd.graddescent import Raw7Space
from src.loss.losses import _EPS_LADDER
from src.loss.loss_selector import select_loss_function

SAMPLE_RATE = 44100
N_FFT = 4096

# the two rungs that must coincide with arms already run
EQUIVALENCES = (("L1_STFT_eps1", "L1_STFT_c2"), ("L1_STFT_eps1e7", "L1_STFT_log"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/val-1000-0.25s"))
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device)
    space = Raw7Space()
    _z, x = load_dataset(space, args.data_dir, args.duration, dev, args.n_val)

    # A perturbed second signal, so the losses are evaluated somewhere with a
    # gradient rather than at a shared zero.
    g = torch.Generator(device="cpu").manual_seed(0)
    y = x * (1.0 + 0.05 * torch.randn(x.shape, generator=g).to(dev))

    rungs = [f"L1_STFT_eps{t}" for t in _EPS_LADDER]

    print("=== rungs resolve and produce finite, differentiable values ===")
    for name in rungs:
        fn = peak_normalized(select_loss_function(name, sample_rate=SAMPLE_RATE, device=dev), "target")
        pred = y.clone().requires_grad_(True)
        val = fn(x, pred).mean()
        val.backward()
        gn = pred.grad.norm().item()
        ok = np.isfinite(val.item()) and np.isfinite(gn) and gn > 0
        print(f"  {name:<18} loss {val.item():.6e}   grad {gn:.6e}   {'OK' if ok else 'BAD'}")

    print("\n=== rungs coincide with the arms they interpolate ===")
    for rung, arm in EQUIVALENCES:
        a = peak_normalized(select_loss_function(rung, sample_rate=SAMPLE_RATE, device=dev), "target")(x, y)
        b = peak_normalized(select_loss_function(arm, sample_rate=SAMPLE_RATE, device=dev), "target")(x, y)
        rel = ((a - b).abs() / b.abs().clamp(min=1e-30)).max().item()
        print(f"  {rung:<18} vs {arm:<16} max rel {rel:.3e}   {'OK' if rel < 1e-6 else 'MISMATCH'}")

    # Where each knee falls, on exactly the magnitudes the loss sees: STFT of
    # the target after the same target-peak normalization the loss applies.
    print("\n=== knee position in the val-set bin distribution ===")
    tp = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    win = torch.hann_window(N_FFT, device=dev)
    mag = torch.stft(x / tp, N_FFT, N_FFT // 4, window=win, return_complex=True).abs()
    bins = mag.flatten().detach().cpu().numpy()
    bins = bins[np.isfinite(bins)]
    print(f"  {len(bins)} bins; median {np.median(bins):.3e}, "
          f"p1 {np.percentile(bins, 1):.3e}, p99 {np.percentile(bins, 99):.3e}")
    print(f"  {'rung':<18} {'eps':>10}   {'pctile of bins below eps':>26}   regime")
    for tag, eps in _EPS_LADDER.items():
        pct = 100.0 * float((bins < eps).mean())
        if pct < 5:
            regime = "knee below data -- pure log"
        elif pct > 95:
            regime = "knee above data -- ~linear"
        else:
            regime = "knee inside data"
        print(f"  {'L1_STFT_eps' + tag:<18} {eps:>10.0e}   {pct:>25.1f}%   {regime}")


if __name__ == "__main__":
    main()
