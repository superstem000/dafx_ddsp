"""How much does each FFT size actually contribute to each half of the loss?

    python scripts/ds_scale_balance.py
    python scripts/ds_scale_balance.py --n 128 --mode other

The multi-scale spectral loss computes both halves at all six FFT sizes
(loss.py:14), sums them, and divides by len(fft_sizes)*(mag_w+log_mag_w). No
per-scale weighting is applied: spectrogram_loss takes norm['spec'][n_fft] and
norm['logspec'][n_fft], upstream left the hook in, and every config in this repo
leaves norm=None, so both divisors are 1.0.

That is fine for the log half and not for the linear half. multiscale_fft
returns POWER (amp = re^2 + im^2, spectral.py:10) from an unnormalised
torch.stft, so a partial's power grows as N^2 with the window; the mean absolute
difference of two power spectrograms grows with it. The log half takes a
difference of logs, which is a relative error -- dimensionless, and independent
of N. So the prediction is that the linear half is dominated by n_fft=2048 while
the log half is roughly balanced across the six.

If that holds, "mag vs log" is not only linear versus log: it is also
effectively-single-scale versus genuinely-multi-scale, which is a confound in
the arm comparison and a candidate mechanism for the linear arms losing Param
accuracy once param_w reaches 0 and the spectral loss is on its own.

The last column is the constant that would remove it -- norm['spec'][n_fft] set
to the measured per-scale term, so every scale contributes equally.

WHAT COUNTS AS AN ERROR. The split depends a little on what the prediction is,
so two bracketing proxies are reported rather than one:

  perturb   x * (1 + 0.05 * randn), a small relative error everywhere. This is
            the same proxy diag_eps_ladder.py uses on the plate, so the two
            diagnostics are measuring at comparable operating points.
  other     a different clip from the same dataset -- a large, structured error
            of the kind an undertrained model makes.

A real mid-training residual sits between them. If the two agree on which scale
dominates, the conclusion does not depend on the choice.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "external", "diffsynth")
sys.path.insert(0, DS)

import torch                                        # noqa: E402
from diffsynth.data import WaveParamDataset         # noqa: E402
from diffsynth.spectral import multiscale_fft       # noqa: E402
from diffsynth.util import log_eps                  # noqa: E402

FFT_SIZES = [64, 128, 256, 512, 1024, 2048]


def terms(target: torch.Tensor, pred: torch.Tensor, sizes: list[int]):
    """Per-scale (mag term, log term), exactly as spectrogram_loss computes them."""
    t_specs = multiscale_fft(target, sizes)
    p_specs = multiscale_fft(pred, sizes)
    mag, lg, scale = [], [], []
    for t, p in zip(t_specs, p_specs):
        mag.append(torch.mean(torch.abs(p - t)).item())
        lg.append(torch.mean(torch.abs(log_eps(p) - log_eps(t))).item())
        scale.append(torch.mean(t).item())
    return mag, lg, scale


def neff(shares):
    """Effective number of scales, 1/sum(share^2). 6.0 is equal weighting."""
    return 1.0 / sum(s * s for s in shares)


def report(label: str, mag, lg, scale, sizes):
    ms, ls = sum(mag), sum(lg)
    mshare = [m / ms for m in mag]
    lshare = [l / ls for l in lg]
    # Normalising by mean|target| rather than by the measured error: the error
    # depends on which proxy is used (the two below differ by ~75x), so
    # constants derived from it would bake one arbitrary error model into the
    # loss. mean|target| is a property of the data alone. It is also a single
    # constant per scale, not per bin, so it changes only the window balance --
    # no quiet bin is up-weighted and the compression axis is untouched.
    nrm = [m / s for m, s in zip(mag, scale)]
    tn = sum(nrm)
    print(f"\n=== {label}")
    print(f"{'n_fft':>7}{'mean|target|':>14}{'mag term':>13}{'share':>8}"
          f"{'log term':>12}{'share':>8}{'mag share if norm':>19}")
    for n, m, l, s, q in zip(sizes, mag, lg, scale, nrm):
        print(f"{n:>7}{s:>14.4g}{m:>13.4g}{100 * m / ms:>7.1f}%"
              f"{l:>12.4g}{100 * l / ls:>7.1f}%{100 * q / tn:>18.1f}%")
    print(f"{'total':>7}{'':>14}{ms:>13.4g}{100.0:>7.1f}%{ls:>12.4g}{100.0:>7.1f}%"
          f"{100.0:>18.1f}%")
    print(f"  effective scales (1/sum share^2, 6.0 = equal):  "
          f"mag {neff(mshare):.1f}   log {neff(lshare):.1f}   "
          f"mag normalised {neff([q / tn for q in nrm]):.1f}")
    print(f"  the two shortest windows carry:  mag {100 * sum(mshare[:2]):.1f}%   "
          f"log {100 * sum(lshare[:2]):.1f}%")
    print("  norm['spec'] = {"
          + ", ".join(f"{n}: {s:.4g}" for n, s in zip(sizes, scale)) + "}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir",
                   default=os.path.join(DS, "data", "diffsynth_5-6", "harmor_2oscfree"))
    p.add_argument("--n", type=int, default=64, help="clips to average over")
    p.add_argument("--perturb", type=float, default=0.05,
                   help="relative noise for the 'perturb' proxy")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ds = WaveParamDataset(args.data_dir, params=False)
    n = min(args.n, len(ds))
    if n < 2:
        raise SystemExit(f"need at least 2 clips, found {len(ds)}")
    x = torch.stack([torch.as_tensor(ds[i]["audio"]) for i in range(n)]).float()
    print(f"{n} clips, {x.shape[-1]} samples each, "
          f"peak {x.abs().amax().item():.4g}, rms {x.pow(2).mean().sqrt().item():.4g}")

    y = x * (1.0 + args.perturb * torch.randn_like(x))
    report(f"perturb  (x * (1 + {args.perturb} * randn))",
           *terms(x, y, FFT_SIZES), FFT_SIZES)

    # Roll by one so every clip is paired with a different one.
    report("other  (a different clip as the prediction)",
           *terms(x, x.roll(1, dims=0), FFT_SIZES), FFT_SIZES)


if __name__ == "__main__":
    main()
