"""Where each diffsynth parameter's signature lives -- the plate sweep, ported.

    python scripts/ds_param_sensitivity.py \
        --conf external/diffsynth/configs/synth/<name>.yaml --n 24
    python scripts/ds_param_sensitivity.py --conf ... --only harmor_cutoff --detail

Run this BEFORE building a task around any parameter. The plate campaign is the
argument for it: every design decision there was made by reasoning from mechanism
-- "damping acts on the decay, decay is late, late is quiet" -- and the numbers
overturned it three separate times. Decay turned out near-neutral at learnable
perturbation sizes; T0 looked like the most quiet-biased parameter in the
synthesizer and 98% of that was float32 noise below -120 dB; a longer window
looked obviously right and made things worse. None of that was visible without
measuring, and all of it was cheap to measure.

WHAT IT REPORTS, per parameter, identical to the plate's version because the
analysis is literally the same code (src/analysis/band_sensitivity):

  dnorm%   what a LINEAR loss has to work with -- L1 on the magnitude
           spectrogram, as a percentage of the distance between two unrelated
           patches. Small means the loud bands are starved.
  gnorm%   the same in the log domain: what a COMPRESSED loss has to work with.
  gain     gnorm%/dnorm%. Above 1, compression sees more of this parameter.
  logdec   mean decile of the log-domain difference, 1 quietest to 10 loudest,
           read against the NEUTRAL printed beneath rather than against 5.5.

AND THE WARNING THE PLATE EARNED: gain does NOT predict which loss wins. It
measures how much of a perturbation a loss can SEE; whether that loss's minimum
sits at the true parameters is a separate property, and only the second decides
the outcome. On quiet7, E had gain 2.04 and the compressed arms landed at 53% of
its range -- they saw it perfectly well and went somewhere else. So read gain as
a necessary condition, never a sufficient one.

WHY DIFFSYNTH MIGHT DO WHAT THE PLATE COULD NOT. The plate has exactly two
parameters whose information is not in the loud partials -- its damping model is
alpha + beta*omega^2, and loss_F1 is degenerate with T60_F1. Its other quiet
region, the valleys between modes, is 120 dB down in float32 rounding rather
than physics. diffsynth has quiet regions that are designed in at real levels: a
lowpass stopband sits 40-80 dB down by filter order, and FM sideband amplitudes
follow J_n(beta), tiny and hypersensitive at high order while the loud low-order
ones barely move.

PARAMETERS ARE ALREADY NORMALIZED. fill_params takes [batch, frames, n_params]
in [0,1] and that IS the search space, so a step is a fraction of range for every
parameter with no per-parameter bounds table and no convention to choose. The
plate needed --step range/value/mul precisely because it lacked this.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "diffsynth"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf                                    # noqa: E402
from diffsynth.modelutils import construct_synth_from_conf          # noqa: E402

from src.analysis.band_sensitivity import DB_BANDS, EPS, decompose, stft_mag  # noqa: E402


def render(synth, p: torch.Tensor, n_samples: int) -> torch.Tensor:
    """[B, 1, P] normalized parameters -> [B, n_samples] audio."""
    with torch.no_grad():
        audio, _ = synth(synth.fill_params(p), n_samples)
    return audio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--conf", required=True, help="Synth config yaml")
    ap.add_argument("--n", type=int, default=24, help="Reference patches")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--audio-len", type=float, default=4.0)
    ap.add_argument("--rel", type=float, nargs="+", default=[0.02, 0.05, 0.10],
                    help="Step as a fraction of each parameter's [0,1] range")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--pin", nargs="+", default=None, metavar="NAME=V",
                    help="Hold a parameter at V (normalized 0-1) in the reference "
                         "family. Where a parameter's signature lives is measured "
                         "at whatever the others are, so the pinned half is a "
                         "design variable too.")
    ap.add_argument("--detail", action="store_true",
                    help="Per-parameter dB-band table, with rel% per band")
    ap.add_argument("--floor-db", type=float, default=None,
                    help="Floor the log measure this far below the reference "
                         "peak, instead of at eps 1e-7. Decides whether a low "
                         "logdec means the signature is quiet or merely below "
                         "the arithmetic -- which is what it meant on the plate.")
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # The synth configs interpolate ${data.sample_rate} from the composed hydra
    # tree, which does not exist when one yaml is loaded on its own. Supplying
    # the node keeps this a plain script rather than requiring a hydra app just
    # to instantiate a synth.
    conf = OmegaConf.merge(OmegaConf.create({"data": {"sample_rate": args.sr}}),
                           OmegaConf.load(args.conf))
    synth = construct_synth_from_conf(conf).to(dev)
    names = list(synth.ext_param_sizes.keys())
    sizes = [synth.ext_param_sizes[k] for k in names]
    # A parameter of size > 1 occupies several columns; each is swept separately
    # and labelled, since they are independent coordinates of the search space.
    cols = [(n, j) for n, s in zip(names, sizes) for j in range(s)]
    label = [n if s == 1 else f"{n}[{j}]"
             for n, s in zip(names, sizes) for j in range(s)]
    P = len(cols)
    n_samples = int(args.audio_len * args.sr)
    print(f"{Path(args.conf).name}   {P} parameter columns   {args.n} patches   "
          f"{args.audio_len}s @ {args.sr} Hz")

    g = torch.Generator().manual_seed(args.seed)
    base = torch.rand((args.n, 1, P), generator=g).to(dev)
    if args.pin:
        for item in args.pin:
            k, v = item.split("=")
            if k not in label:
                raise SystemExit(f"unknown parameter {k!r}; have: {', '.join(label)}")
            base[:, :, label.index(k)] = float(v)
        print("operating point: " + ", ".join(args.pin))

    x_ref = render(synth, base, n_samples)
    A_n = stft_mag(x_ref, args.n_fft, args.hop, True)
    perm = torch.randperm(x_ref.shape[0],
                          generator=torch.Generator().manual_seed(args.seed)).to(dev)
    sat_n = float((A_n - A_n[perm]).abs().sum())
    if args.floor_db is None:
        eps_n = EPS
    else:
        eps_n = float(A_n.max()) * 10.0 ** (-args.floor_db / 20.0)
        print(f"log floor: {args.floor_db:g} dB below the reference peak "
              f"(eps {eps_n:.3g})")
    sat_g = float((torch.log(A_n + eps_n) - torch.log(A_n[perm] + eps_n)).abs().sum())
    print(f"saturation: {sat_n:.5g} linear, {sat_g:.5g} log -- the denominators\n")

    want = [i for i, l in enumerate(label) if not args.only or l in args.only]
    if args.only:
        missing = set(args.only) - set(label)
        if missing:
            raise SystemExit(f"unknown: {', '.join(missing)}; have: {', '.join(label)}")

    res, bands, neutral = {}, {}, []
    for i in want:
        for rel in args.rel:
            runs = []
            for sign in (+1.0, -1.0):
                p = base.clone()
                # Clamped to [0,1]: outside it is not a patch the synth defines,
                # and a step that walks off the range measures the clamp.
                p[:, :, i] = (p[:, :, i] + sign * rel).clamp(0.0, 1.0)
                x_p = render(synth, p, n_samples)
                ok = torch.isfinite(x_p).all(dim=-1)
                if not bool(ok.any()):
                    continue
                runs.append(decompose(A_n[ok],
                                      stft_mag(x_p[ok], args.n_fft, args.hop, True),
                                      eps_n))
            if not runs:
                res[(i, rel)] = (float("nan"),) * 4
                continue

            def m(k):
                v = [r[k] for r in runs]
                return ([sum(c) / len(c) for c in zip(*v)] if isinstance(v[0], list)
                        else sum(v) / len(v))

            dn = 100.0 * m(4) / max(sat_n, 1e-30)
            gn = 100.0 * m(5) / max(sat_g, 1e-30)
            res[(i, rel)] = (dn, gn, gn / dn if dn > 0 else float("nan"), m(1))
            bands[(i, rel)] = (m(2), m(3), m(6), m(7))
            neutral.append(m(8))

    w = max(10, max(len(label[i]) for i in want) + 2)
    for rel in args.rel:
        print(f"\n=== {100*rel:g}% of range")
        print(f"{'param':<{w}}{'dnorm%':>10}{'gnorm%':>10}{'gain':>8}{'logdec':>8}")
        for i in want:
            d, gg, r, c = res[(i, rel)]
            print(f"{label[i]:<{w}}{d:>10.3f}{gg:>10.3f}{r:>8.2f}{c:>8.2f}")
    if neutral:
        print(f"\n  NEUTRAL for this floor: {sum(neutral)/len(neutral):.2f} "
              f"-- compare logdec against THIS, not against 5.5.")
    print("  gain > 1 means compression sees more of the parameter. It does NOT")
    print("  mean compression will win on it: whether a loss's minimum sits at the")
    print("  true parameters is a different property, and it is the one that decides.")

    if args.detail:
        for i in want:
            for rel in args.rel:
                lb, gb, cb, rb = bands[(i, rel)]
                print(f"\n=== {label[i]}   by dB below peak (at {100*rel:g}%)")
                print(f"{'band':>10}{'bins':>8}{'linear':>9}{'log':>9}{'rel%':>9}")
                sl, sg = max(sum(lb), 1e-30), max(sum(gb), 1e-30)
                sc = max(sum(cb), 1)
                for (lo, hi), a, b, c, rr in zip(DB_BANDS, lb, gb, cb, rb):
                    print(f"{f'{lo}-{hi}':>10}{100*c/sc:>7.1f}%{100*a/sl:>8.1f}%"
                          f"{100*b/sg:>8.1f}%{100*rr:>8.2f}%")
        print("\n  rel% is the mean |a-b|/(a+eps) over the band -- how much the signal")
        print("  in that band actually MOVED. Rising with depth means a compressed loss")
        print("  has something there. Above 100% means those bins decorrelated, which")
        print("  is a statement that they are chaotic rather than a magnitude.")


if __name__ == "__main__":
    main()
