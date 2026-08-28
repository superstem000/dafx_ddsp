"""Is a real EMT-140 reachable by this model AT ALL, at any parameter setting?

    python -m src.emt.probe --wav-dir data/EMT-140 --n 2048

Every plate number against real audio so far is a test of an ENCODER: one
forward pass, and if it emits a constant -- which on emt7 it did, six of seven
parameters identical across fifteen very different IRs -- then nothing measured
downstream is about the model, the space or the loss. This removes the encoder.
It draws n points at random from a deliberately WIDE box, renders each one, and
scores all of them against every target.

It answers four questions the encoder cannot:

  REACHABLE   the best score any point achieves, against the saturation
              reference (two unrelated real IRs). If the best random draw out of
              thousands is still at saturation, no encoder can do better and the
              model is the wall, not the training.
  THE BASS    emt7's renders sat +28 dB over target at 62 Hz, arm-independent
              and unchanged by the drive point. Does ANY point get within 10 dB?
              If not, the excess is structural -- a simply-supported plate has
              an m=n=1 mode with one antinode across the whole surface, and no
              parameter suppresses a mode the boundary condition guarantees.
  THE TARGET  the targets' own spectra. An EMT-140 recording passes through a
              moving-coil pickup, a transformer and tube electronics, all of
              which roll off low. If the RECORDINGS have no bass, "+28 dB at
              62 Hz" is the model behaving normally against a high-passed
              reference and is not a model defect at all. Read this first.
  THE BOUNDS  where the best points actually sit in each dimension. That is the
              evidence for the next box, rather than another guess -- and a
              top-K median hard against an edge of THIS box means widen further.

WHY A WIDE BOX, AND WHY UNPHYSICAL IS FINE. emt7 narrowed rho and E to steel on
the theory that a real plate is made of steel. The railing says that reasoning
was wrong: this is sound matching, not metrology, and if a 3 mm plate of
something with the density of aluminium matches an EMT-140 better than 0.5 mm
of steel, that is the answer to the question actually being asked.

WHAT IT IS NOT. Random search in twelve dimensions is not optimisation, and the
best of n draws is a LOWER bound on what the model can do -- a real fit would
do better. That is the right direction for the load-bearing conclusion: if even
a lower bound clears saturation the model is fine, and if the bass is reachable
at all, one draw finding it is proof.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.emt.band import brickwall_lowpass                  # noqa: E402
from src.gd.graddescent import SAMPLE_RATE                  # noqa: E402
from src.plate.SevenParamPlate import (                     # noqa: E402
    BatchedModalPlateTorch as SevenParamPlate,
)
from src.loss.losses import (                               # noqa: E402
    _get_dct, _get_mel_fb, _stft_mag, configure_loss_runtime,
)

P14 = list(SevenParamPlate.PARAM_ORDER)

# The wide box. Chosen to CONTAIN the answer rather than to describe a plate:
# every emt7 bound that railed is opened well past where it railed, and the two
# that were fixed on physical grounds (rho, E) are searched again over a range
# no real steel occupies. (lo, hi, log?)
WIDE = {
    # THE FINAL BOX. Six searched dimensions, and every bound below is the union
    # of what the FIVE objectives asked for rather than what mfcc alone did --
    # narrowing on one objective's evidence is what excluded decay's T60_DC last
    # time. 2048 draws over six axes is 3.56 points per axis, against 2.14 at ten
    # and 1.89 at the twelve this started with.
    #
    # medians per objective (mfcc / bass / tilt / decay / onset) in brackets.
    "Ly":      (1.7, 3.1, False),          # 2.21 2.14 2.19 2.35 2.65
    "h":       (5e-4, 2.5e-3, True),       # 1.18e-3 8.1e-4 8.2e-4 1.70e-3 1.31e-3
    "T0":      (1e4, 4e5, True),           # 8.5e4 1.6e5 1.3e5 4.7e4 2.1e4
    "rho":     (4500.0, 11000.0, False),   # 6934 6460 6460 8080 8980
    # THE TWO THAT DISAGREE, and the reason they stay wide. decay wants
    # T60_DC 3.88 [3.67, 3.99] against mfcc's 1.95 [1.64, 2.25], and
    # T60_F1_ratio 0.0169 [0.008, 0.034] against mfcc's 0.583 [0.385, 0.903] --
    # a factor of 34. The previous box's (1.0, 4.0) had decay pressed against
    # the ceiling, because that ceiling was set from mfcc's CI alone. Both
    # answers have to fit or the probe cannot report the disagreement it found.
    "T60_DC":  (1.0, 5.0, True),           # 1.95 2.03 1.97 3.88 2.39
    "T60_F1_ratio": (0.005, 0.98, True),   # 0.583 0.147 0.071 0.017 0.351
}
# WHY E AND loss_F1 ARE NO LONGER SEARCHED. Both are EXACT degeneracies, so
# dropping them costs no reachable render and buys resolution everywhere else.
#
#   E.   The modal sum sees only mu = rho*h, D/mu = E*h^2/(12(1-nu^2)rho) and
#        T0/mu, so (E, rho, h) -> (c^3 E, c rho, h/c) is invariant. Searching all
#        three spends a dimension on a flat direction -- which is exactly why
#        objectives.py reported rho AND E as "unconstrained" with intervals
#        covering nearly the whole box under BOTH objectives. They are not
#        uninformative, they are degenerate with each other. With E fixed,
#        (h, rho, T0) -> (mu, D/mu, T0/mu) is a bijection.
#
#   loss_F1.  alpha = 3ln10/dOmSq * (OmDamp2^2/T60_DC - OmDamp1^2/T60_F1), and
#        OmDamp1 = 0, so dOmSq = (2*pi*loss_F1)^2 cancels and alpha = 3ln10/T60_DC
#        with no loss_F1 in it at all. It survives only inside
#        beta = [3ln10/T60_DC] * (1-r) / (r * (2*pi*loss_F1)^2), so T60_F1_ratio
#        and loss_F1 enter through ONE combination. Verified: (T60_DC 2.0,
#        r 0.50, loss_F1 1e4) and (2.0, 0.20, 2e4) both give alpha 3.453878 and
#        beta 8.748774e-10 -- bit-identical renders. 10000 rather than the
#        probe's own 2.4e4 winner because a round reference frequency makes
#        T60_F1_ratio read directly as "T60 at 10 kHz over T60 at DC".
#
# 12 searched dimensions to 10. At 2048 draws the mean nearest-neighbour spacing
# goes 0.530 -> 0.474 of the box per axis, and T0's own axis improves 2.3x on top
# of that. More draws is the weak lever by comparison: 64x the samples at 12
# dimensions only reaches 0.375.
# fp_x, fp_y, op_x, op_y JOIN THE FIXED SET, and unlike emt8's pins this is
# measured rather than assumed. emt8 fixed them on their flatness under mfcc
# alone -- "flat under mfcc" and "flat under everything" are different claims,
# and only the first had ever been tested. All five objectives now report them
# unconstrained, ONSET INCLUDED, which is the one that could have pinned them:
# the drive point sets which modes the strike excites and nothing else the
# encoder searches touches the first 20 ms. It does not need them either.
#
#   fp_x   mfcc 0.134  bass 0.215  tilt 0.103  decay 0.339  onset 0.283
#   fp_y        0.299       0.313       0.209        0.256        0.249
#   op_x        0.564       0.684       0.406        0.513        0.468
#   op_y        0.604       0.567       0.421        0.431        0.704
#
# every 5-95% interval covering most of its range. The values below are the
# MEDIAN OF THE FIVE MEDIANS, so no objective is served preferentially -- and
# because the objectives are flat in them the exact value costs little either
# way. They differ slightly from emt8's (0.22, 0.18) / (0.48, 0.59), which came
# from the 22 kHz mfcc top-32 only.
#
# Fixing both fp AND op also disposes of the exchange symmetry for free: P is a
# product of sin(fp_x pi m)sin(fp_y pi n) and sin(op_x pi m)sin(op_y pi n), so
# swapping the pairs renders bit-identically and searching all four carries a
# hard two-fold ambiguity no data resolves.
FIXED = {"Lx": 1.0, "nu": 0.30, "E": 1.85e11, "loss_F1": 10000.0,
         "fp_x": 0.215, "fp_y": 0.256, "op_x": 0.513, "op_y": 0.567}

# rho and E stay wide INDIVIDUALLY; what is bounded is their RATIO. Cost goes as
# sqrt(rho/E)/h, so the expensive corner of an independent box is max-rho with
# min-E -- 12000 over 6e10 = 2.0e-7, a sound speed of 2200 m/s. No metal is near
# that: steel, aluminium and glass all sit at 3.6-3.9e-8, because sqrt(E/rho) is
# ~5000 m/s across common solids, and this probe's own winners landed at 4.2e-8.
# So the corner costing 58.7 h/arm at fmax 20000 is a material that does not
# exist. Bounding the ratio removes it and leaves every real one, plus margin.
RHO_OVER_E = (1.5e-8, 7.0e-8)

# T0 is tension per unit LENGTH (N/m), so T0/h is a STRESS in Pa -- and that is
# a physical bound the box has never used. Without it T0 has climbed every time
# it was given room: median 4,857 in the (0.1, 1e6) box, 44,300 at (1e3, 1e6),
# 104,800 at (1e4, 5e5), with the top-32 VALUES then pressing 4.87e5 against the
# ceiling. In stress that last one is 415 MPa at the winners' h of 1.17 mm --
# mild steel yields near 250 and spring steel near 500, so the search was
# already asking for a plate that tears.
#
# The winners themselves are not the problem: 104,800 / 1.17e-3 = 89 MPa, well
# inside the elastic range, and the CI [6.17e4, 1.55e5] is 53-132 MPa. Bounding
# the ratio deletes the draws that cannot exist and stops T0's ceiling being a
# number to chase. 4e8 rather than 2.5e8 so a spring-steel plate is admitted;
# 1e6 because below ~1 MPa the tension term is irrelevant next to stiffness and
# the draws are indistinguishable.
#
# Applied exactly as RHO_OVER_E is: T0 is sampled inside the window h allows,
# rather than rejected, so h keeps its uniform marginal and no draw is wasted.
T0_OVER_H = (1e6, 4e8)


def dd_grid(Lx, Ly, h, T0, rho, E, nu, fmax):
    """SevenParamPlate's own DDx/DDy, for pinning the grid from the box corner."""
    w = 2.0 * math.pi * fmax
    D = E * h ** 3 / (12.0 * (1.0 - nu * nu))
    disc = max((-T0 + math.sqrt(max(T0 * T0 + 4.0 * w * w * rho * h * D, 0.0)))
               / (2.0 * D), 0.0)
    s = math.sqrt(disc)
    return math.floor(Lx / math.pi * s), math.floor(Ly / math.pi * s)


def corner(box, fmax):
    """Densest draw: max Ly, min h, min T0, and the largest ADMITTED rho/E."""
    e = FIXED["E"]
    rho = min(box["rho"][1], RHO_OVER_E[1] * e)
    # Lower T0 means more modes, so the corner takes the smallest T0 the stress
    # bound ADMITS at the corner's h -- not the box floor, which may be excluded.
    t0 = max(box["T0"][0], box["h"][0] * T0_OVER_H[0])
    return dd_grid(FIXED["Lx"], box["Ly"][1], box["h"][0], t0,
                   rho, e, FIXED["nu"], fmax)


def sample(box, n, seed, dev):
    """[n, 14] in PARAM_ORDER, uniform in the box (log-uniform where flagged)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cols, named = {}, {}
    for k, (lo, hi, lg) in box.items():
        u = torch.rand(n, generator=g)
        named[k] = (torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
                    if lg else lo + u * (hi - lo))
    # E used to be resampled inside the rho/E window. It is FIXED now (see the
    # note on FIXED), and rho's floor was raised to 1.5e-8 * 1.85e11 = 2775 so
    # every draw already satisfies the ratio bound without rejection.

    # T0 inside the window h allows, same trick RHO_OVER_E used: an exact
    # interval keeps every draw usable and leaves h's marginal uniform. Log
    # because T0 is log-sampled, so the conditional has to be too.
    s_lo, s_hi = T0_OVER_H
    t_lo = torch.clamp(named["h"] * s_lo, min=box["T0"][0])
    t_hi = torch.clamp(named["h"] * s_hi, max=box["T0"][1])
    t_hi = torch.maximum(t_hi, t_lo * (1.0 + 1e-6))
    u = torch.rand(n, generator=g)
    named["T0"] = torch.exp(torch.log(t_lo) + u * (torch.log(t_hi) - torch.log(t_lo)))

    # The one derived column. named keeps the ratio, since that is the quantity
    # with a meaningful bound to report against in THE BOUNDS.
    cols["T60_F1"] = named["T60_F1_ratio"] * named["T60_DC"]
    for k in P14:
        if k in cols:
            continue
        cols[k] = (named[k] if k in named
                   else torch.full((n,), float(FIXED[k])))
    return torch.stack([cols[k] for k in P14], dim=1).to(dev), named


def third_octave(lo, hi, n_fft, dev):
    """Centres and their bin masks, DROPPING any band no FFT bin lands in.

    At n_fft 2048 the bin spacing is 21.5 Hz while a third octave at 50 Hz is
    11.6 Hz wide, so that band contains no bin at all. Summing it gives a hard
    zero, which then survives the log as -291 dB and reads as a brick-wall high
    pass in the recording rather than as an absent measurement. Hence
    --band-fft 8192 (5.4 Hz bins, >=2 per band from 40 Hz up) and dropping
    anything still empty instead of reporting it.
    """
    k = np.arange(-20, 14)
    c = 1000.0 * 2.0 ** (k / 3.0)
    c = c[(c >= lo) & (c <= hi)]
    f = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)
    rows = [((f >= v * 2 ** (-1 / 6)) & (f < v * 2 ** (1 / 6))).astype(np.float32)
            for v in c]
    keep = [i for i, r in enumerate(rows) if r.sum() > 0]
    if len(keep) != len(c):
        print(f"  (dropped {len(c) - len(keep)} third-octave band(s) with no FFT "
              f"bin at n_fft {n_fft})")
    return c[keep], torch.from_numpy(np.stack([rows[i] for i in keep])).to(dev)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--wav-dir", default="data/EMT-140")
    p.add_argument("--n", type=int, default=2048, help="random draws")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=4.0,
                   help="render and score length. No encoder here, so this is "
                        "not tied to any checkpoint's training window.")
    p.add_argument("--fmax", type=float, default=12000.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--chunk-elems", type=int, default=800_000_000,
        help="NOT 4e8, which was tuned for emt7's 22,000-mode grid and is wrong "
             "for any grid this size. Per-mode tensors are re-read on EVERY "
             "launch and launch count is Ts*B*n_pad/chunk_elems, so total "
             "re-read traffic goes as n_pad^2 at fixed chunk_elems -- the rule "
             "jobs_emt8.txt states as 'WHENEVER THE MODE COUNT CHANGES, "
             "chunk_elems MUST BE RESCALED WITH IT.' At this box's 20 kHz corner "
             "of (188, 660) = 124,080 modes, --batch 16 and 4 s:\n"
             "  4e8   876 launches x 31.8 MB = 27.8 GB traffic, peak 4.8 GB\n"
             "  8e8   438 launches x 31.8 MB = 13.9 GB traffic, peak 9.6 GB\n"
             "against an emt10 training step's 6.7 GB. The raw arithmetic puts a "
             "2048-draw run at ~6 min -- 368 forward-equivalents of a training "
             "step, which is ~123 steps at 0.34 st/s -- and the traffic is what "
             "took it to 20-30. Per-clip traffic goes as batch/chunk_elems, so "
             "--batch 8 with this halves it again to 3.5 GB; src/emt/bench_modal.py "
             "is what settles whether the occupancy loss is worth it.")
    p.add_argument("--mode-bucket", type=int, default=1024)
    p.add_argument("--no-compile", action="store_true",
                   help="eager modal sum. 8.5x slower here; compile is safe "
                        "because nothing in this script is compared against a "
                        "dataset rendered elsewhere, so the numerics contract "
                        "35e4529 describes does not apply.")
    p.add_argument("--top", type=int, default=32, help="K for the bounds report")
    p.add_argument(
        "--bootstrap", type=int, default=200, metavar="N",
        help="Resamples for the interval on each top-K median. A median over 32 "
             "of 2048 draws in twelve dimensions is not obviously precise, and "
             "without an interval there is no way to tell a real bound from "
             "noise -- the un-lowpassed run put T0's median at 4.4e4 and the "
             "lowpassed one at 2.0e3, and only an interval says whether those "
             "two numbers actually disagree. 0 disables.")
    p.add_argument("--bass-tol", type=float, nargs="+", default=[10.0, 6.0, 3.0, 1.0],
                   metavar="DB",
                   help="thresholds for the JOINT table: best mfcc among draws "
                        "that ALSO match 62 Hz within this many dB.")
    p.add_argument("--n-fft", type=int, default=2048, help="metric FFT, matching loss_mfcc")
    p.add_argument("--band-fft", type=int, default=8192,
                   help="FFT for the third-octave tables only. 2048 gives 21.5 Hz "
                        "bins and leaves the 50 Hz band empty; 8192 gives 5.4 Hz.")
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-mfcc", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--lowpass", type=float, default=None, metavar="HZ",
        help="Band-limit the TARGETS to this before scoring, normally --fmax. "
             "The renders stop at fmax, so ~19 of the 128 mel bands are floored "
             "in every draw and live in every target: without this, every score "
             "carries a constant penalty for a band nothing can reach. There is "
             "no encoder here, so this is the only place a lowpass applies.")
    p.add_argument("--out", default=None, help="write the best render per IR here")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    configure_loss_runtime(SAMPLE_RATE, dev)

    gx, gy = corner(WIDE, args.fmax)
    print(f"wide box, {len(WIDE)} searched dimensions, {args.n} draws at "
          f"{args.duration}s, fmax {args.fmax:.0f}")
    print(f"mode grid pinned at this box's corner: {gx},{gy} = {gx*gy:,} cells "
          f"({gx*gy/22000:.1f}x emt7's)\n")

    plate = SevenParamPlate(sample_rate=SAMPLE_RATE, device=dev,
                            dtype=torch.float32, drop_sub_20hz_modes=False,
                            fmax=args.fmax)
    plate.chunk_elems = args.chunk_elems
    plate.grad_checkpoint = False
    plate.batched_modal_sum = True
    plate.compile_modal_sum = not args.no_compile
    plate.mode_bucket = args.mode_bucket
    plate.fixed_mode_grid = (gx, gy)

    fb = _get_mel_fb(args.n_fft, args.n_mels)
    dct = _get_dct(args.n_mfcc, args.n_mels)

    def mfcc(x):
        mel = torch.matmul(fb.unsqueeze(0), _stft_mag(x, args.n_fft, args.hop) ** 2)
        return torch.matmul(dct.unsqueeze(0), 10.0 * torch.log10(mel + 1e-10))

    def peak(x):
        return x / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)

    cen, M = third_octave(40.0, 20000.0, args.band_fft, dev)
    i62 = int(np.argmin(np.abs(cen - 62.5)))

    def bands(x):
        """[B, n_bands] band energy shares, so this is shape and not level."""
        P = _stft_mag(x, args.band_fft, args.hop) ** 2
        e = torch.einsum("bf,cf->bc", P.sum(dim=-1), M)
        return e / e.sum(dim=1, keepdim=True).clamp(min=1e-30)

    # ONE time-resolved tensor, three objectives off it. TILT, DECAY and ONSET
    # are all reductions of [B, n_bands, n_frames], so adding them costs no
    # extra rendering -- which is the whole reason they belong in this file
    # rather than in a second pass over the wavs.
    n_on = max(1, int(round(0.020 * SAMPLE_RATE / args.hop)))

    def bands_t(x):
        """[B, n_bands, n_frames] band energy over time."""
        P = _stft_mag(x, args.band_fft, args.hop) ** 2
        return torch.einsum("bft,cf->bct", P, M)

    def shape_decay_onset(bt):
        """(tilt_dB, T60_s, onset_share), each [B, n_bands], from bands_t.

        TILT is the time-summed share in dB, i.e. exactly what bands() gives,
        so it is level-free and comparable across draws.

        DECAY is a least-squares slope of 10*log10(energy) against time over the
        WHOLE render, converted to a T60. A full-length fit is biased by the
        noise floor at the tail -- a proper T20 would window between -5 and
        -25 dB -- but the bias applies identically to the render and to the
        target, and only their RATIO is scored. Vectorising a per-band variable
        window over 2048 draws x 15 IRs x 27 bands is not worth the accuracy.

        ONSET is the share of each band's energy inside the first 20 ms, which
        is why_dark's ONSET column and the one place a pinned drive point can
        show up: fp/op set which modes the strike excites, and nothing else the
        encoder searches does.
        """
        e = bt.sum(dim=-1)
        sh = e / e.sum(dim=1, keepdim=True).clamp(min=1e-30)
        tilt = 10.0 * torch.log10(sh.clamp(min=1e-30))

        L = 10.0 * torch.log10(bt.clamp(min=1e-30))          # [B, C, T]
        n_t = L.shape[-1]
        t = torch.arange(n_t, device=L.device, dtype=L.dtype) * (args.hop / SAMPLE_RATE)
        tc = t - t.mean()
        slope = (tc * (L - L.mean(dim=-1, keepdim=True))).sum(-1) / (tc * tc).sum()
        t60 = -60.0 / slope.clamp(max=-1e-3)                 # dB/s -> s

        onset = bt[..., :n_on].sum(-1) / e.clamp(min=1e-30)
        return tilt, t60, onset

    # --- targets ----------------------------------------------------------
    from pathlib import Path
    wavs = sorted(str(q) for q in Path(args.wav_dir).rglob("*.wav"))
    want = int(round(args.duration * SAMPLE_RATE))
    tg, names = [], []
    for w in wavs:
        x, sr = sf.read(w, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE,
                                 res_type="soxr_hq").astype(np.float32)
        x = np.pad(x, (0, max(0, want - len(x))))[:want]
        tg.append(x)
        names.append(Path(w).stem)
    T = torch.from_numpy(np.stack(tg)).to(dev)
    if args.lowpass:
        T = brickwall_lowpass(T, args.lowpass, SAMPLE_RATE)
    if args.lowpass and abs(args.lowpass - args.fmax) > 1.0:
        print(f"  NOTE --lowpass {args.lowpass:.0f} != --fmax {args.fmax:.0f}. "
              f"They should match: the point is to score the target over the "
              f"band the renders occupy.")
    # Peak AFTER band-limiting, so the level the metric sees is the level of the
    # band being compared rather than of a transient that was partly removed.
    T = peak(T)
    with torch.no_grad():
        Tm, Tb = mfcc(T), bands(T)
        Ttilt, Tt60, Ton = shape_decay_onset(bands_t(T))
    print(f"{len(names)} target(s) from {args.wav_dir}"
          + (f", band-limited to {args.lowpass:.0f} Hz" if args.lowpass else "")
          + "\n")

    # --- THE TARGET: is the reference itself high-passed? -----------------
    print("=== THE TARGET   each recording's own third-octave share, dB below its peak band")
    print("  A recording rolled off at the bottom makes a model with ordinary bass")
    print("  look +28 dB hot at 62 Hz. Read this before blaming the model.")
    tdb = 10.0 * torch.log10(Tb.clamp(min=1e-30))
    tdb = tdb - tdb.amax(dim=1, keepdim=True)
    show = [i for i, c in enumerate(cen) if c <= 400.0]
    print(f"  {'ir':<22}" + "".join(f"{c:>8.0f}" for c in cen[show]))
    for i, nm in enumerate(names):
        print(f"  {nm:<22}" + "".join(f"{tdb[i, j].item():>8.1f}" for j in show))
    print(f"  {'MEAN':<22}" + "".join(f"{tdb[:, j].mean().item():>8.1f}" for j in show))

    # --- saturation, the only thing that makes a score readable -----------
    with torch.no_grad():
        sat = float(torch.stack([
            (Tm[i] - Tm[(i + 1) % len(names)]).abs().mean() for i in range(len(names))
        ]).mean())

    # --- draw, render, score ----------------------------------------------
    Z, named = sample(WIDE, args.n, args.seed, dev)
    # [n_draws, n_ir] for both criteria. n is thousands and n_ir is fifteen, so
    # keeping every score costs nothing, and every table below is a reduction of
    # this rather than a separate incremental tracker to get wrong.
    D = torch.empty((args.n, len(names)), device=dev)
    G62 = torch.empty((args.n, len(names)), device=dev)
    TL = torch.empty((args.n, len(names)), device=dev)   # tilt, mean |dB| error
    DC = torch.empty((args.n, len(names)), device=dev)   # decay, mean |log10 T60 ratio|
    ON = torch.empty((args.n, len(names)), device=dev)   # onset, mean |share| error
    tb62 = (10.0 * torch.log10(Tb.clamp(min=1e-30)))[:, i62]

    print(f"\nrendering {args.n} draws...")
    with torch.no_grad():
        for s0 in range(0, args.n, args.batch):
            z = Z[s0 : s0 + args.batch]
            raw = plate(z, duration=args.duration, normalize=False)
            # A draw whose modal sum diverged is not a bad fit, it is not a
            # rendering at all. Excluded and counted; min() over a nan is nan
            # and would have taken the whole table down with it.
            ok = torch.isfinite(raw).all(dim=-1)
            y = peak(torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0))
            d = (mfcc(y).unsqueeze(1) - Tm.unsqueeze(0)).abs().mean(dim=(-2, -1))
            yb62 = (10.0 * torch.log10(bands(y).clamp(min=1e-30)))[:, i62]
            g62 = (yb62.unsqueeze(1) - tb62.unsqueeze(0)).abs()
            ytilt, yt60, yon = shape_decay_onset(bands_t(y))
            # [draws, 1, bands] against [1, n_ir, bands] -> [draws, n_ir]
            tl = (ytilt.unsqueeze(1) - Ttilt.unsqueeze(0)).abs().mean(dim=-1)
            # T60 in log ratio: scale-free, and a render twice as long as the
            # target scores the same as one half as long.
            dc = (torch.log10(yt60.clamp(min=1e-3)).unsqueeze(1)
                  - torch.log10(Tt60.clamp(min=1e-3)).unsqueeze(0)).abs().mean(dim=-1)
            on = (yon.unsqueeze(1) - Ton.unsqueeze(0)).abs().mean(dim=-1)
            bad = ~ok.unsqueeze(1) | ~torch.isfinite(d)
            D[s0 : s0 + z.shape[0]] = torch.where(bad, torch.inf, d)
            G62[s0 : s0 + z.shape[0]] = torch.where(bad, torch.inf, g62)
            for _t, _v in ((TL, tl), (DC, dc), (ON, on)):
                _t[s0 : s0 + z.shape[0]] = torch.where(
                    bad | ~torch.isfinite(_v), torch.inf, _v)
            if (s0 // args.batch) % 20 == 0:
                print(f"  {min(s0 + args.batch, args.n)}/{args.n}", flush=True)

    best, best_i = D.min(dim=0)
    b62, b62_i = G62.min(dim=0)
    n_bad = int((~torch.isfinite(D[:, 0])).sum())
    if n_bad:
        print(f"\n  {n_bad}/{args.n} draws discarded as non-finite renders "
              f"({100*n_bad/args.n:.1f}%).")
    if n_bad == args.n:
        raise SystemExit("every draw diverged -- the box admits unstable damping")

    # --- REACHABLE ---------------------------------------------------------
    print(f"\n=== REACHABLE   best of {args.n} random draws, plate mfcc")
    print(f"  saturation (two unrelated real IRs) = {sat:.4f}")
    print(f"  emt7's encoders scored 2.45-2.80 x saturation on these IRs\n")
    print(f"  {'ir':<22}{'best':>10}{'/sat':>8}{'draw':>7}")
    for i, nm in enumerate(names):
        print(f"  {nm:<22}{best[i].item():>10.4f}{best[i].item()/sat:>8.3f}"
              f"{best_i[i].item():>7d}")
    print(f"  {'MEAN':<22}{best.mean().item():>10.4f}"
          f"{best.mean().item()/sat:>8.3f}")

    # --- THE BASS ----------------------------------------------------------
    print(f"\n=== THE BASS   closest any draw gets at {cen[i62]:.0f} Hz, |render - target| dB")
    print("  emt7's encoders were +28 dB here, every arm, drive point irrelevant.")
    print(f"  {'ir':<22}{'best |dB|':>11}{'draw':>7}")
    for i, nm in enumerate(names):
        print(f"  {nm:<22}{b62[i].item():>11.1f}{b62_i[i].item():>7d}")
    print(f"  {'MEAN':<22}{b62.mean().item():>11.1f}")
    print("  Under 10 dB anywhere: the bass is a BOUNDS problem, widen and retrain.")
    print("  Never under 10 dB: it is the boundary condition, and no box fixes it.")

    # --- JOINT -------------------------------------------------------------
    # THE BASS on its own is nearly free: with thousands of draws and a
    # continuous parameter, SOME draw lands on any single band. The question
    # that matters is whether a draw can match 62 Hz AND be a good fit, and the
    # winners above say it is not the same draw (1046 nails bright_1's bass, 494
    # wins its mfcc). So: best mfcc subject to the bass constraint.
    #
    # Against a SAME-SIZE CONTROL, because a constrained best is worse partly
    # from having fewer candidates. The median best-of-m from the unconstrained
    # column is the q-quantile at q = 1 - 0.5**(1/m); if the constrained best
    # matches that, the constraint costs nothing and the two are compatible. If
    # it is much worse, the bass and the rest are genuinely in tension -- which
    # is what a structural mismatch looks like.
    print(f"\n=== JOINT   best mfcc among draws ALSO within N dB at {cen[i62]:.0f} Hz")
    print("  ctrl = median best-of-m from all draws, m = how many qualified.")
    print("  best ~ ctrl: no tension, the constraint is free.")
    print("  best >> ctrl: the bass and the rest cannot be had together.\n")
    tols = sorted(args.bass_tol, reverse=True)
    print(f"  {'ir':<20}{'free':>8}" +
          "".join(f"{f'<={t:g}dB':>10}{'ctrl':>8}{'m':>6}" for t in tols))
    for i, nm in enumerate(names):
        row = f"  {nm:<20}{best[i].item():>8.2f}"
        for t in tols:
            m = G62[:, i] <= t
            cnt = int(m.sum())
            if cnt == 0:
                row += f"{'-':>10}{'-':>8}{0:>6}"
                continue
            col = D[:, i][torch.isfinite(D[:, i])]
            q = 1.0 - 0.5 ** (1.0 / cnt)
            ctrl = float(torch.quantile(col, min(max(q, 0.0), 1.0)))
            row += f"{float(D[:, i][m].min()):>10.2f}{ctrl:>8.2f}{cnt:>6d}"
        print(row)

    # --- SHAPE, DECAY, ONSET -----------------------------------------------
    print("\n=== THE OTHER THREE   best of the same draws, per objective")
    print("  Free off the same renders, so these cost no GPU time. Each is a")
    print("  mean over third-octave bands of |render - target|:")
    print("    tilt    dB of the time-summed band share; level-free")
    print("    decay   |log10(T60_render / T60_target)|, so 0.30 is a factor of 2")
    print("    onset   share of each band's energy in the first 20 ms")
    print("  ONSET IS THE ONE THAT MATTERS FOR THE SEARCHED SET. fp/op set which")
    print("  modes the strike excites and nothing else the encoder searches does,")
    print("  so if they are unconstrained here too they can be fixed -- and if")
    print("  they are not, they belong in the search.\n")
    print(f"  {'ir':<22}" + "".join(f"{h:>10}" for h in ("tilt dB", "decay", "onset")))
    for i, nm in enumerate(names):
        print(f"  {nm:<22}{TL[:, i].min().item():>10.3f}"
              f"{DC[:, i].min().item():>10.3f}{ON[:, i].min().item():>10.4f}")
    print(f"  {'MEAN of the bests':<22}{TL.min(dim=0).values.mean().item():>10.3f}"
          f"{DC.min(dim=0).values.mean().item():>10.3f}"
          f"{ON.min(dim=0).values.mean().item():>10.4f}")
    print(f"  {'MEDIAN draw (random)':<22}"
          + "".join(f"{float(np.median(t[torch.isfinite(t)].cpu().numpy())):>10.3f}"
                    for t in (TL, DC, ON)))
    print("\n  read best against the random median: an objective whose best is")
    print("  near the median is one no draw in this box does well, which is a")
    print("  statement about the MODEL rather than about the search.")

    # --- THE BOUNDS --------------------------------------------------------
    k = min(args.top, args.n)
    # Top K by MEAN score over the fifteen IRs, not per-IR winners: a parameter
    # that matters concentrates for all of them, and one that does not stays
    # uniform over its range no matter which IR you rank by.
    pool = torch.argsort(D.mean(dim=1))[:k].cpu().tolist()
    print(f"\n=== THE BOUNDS   the top {k} draws by mean score, per parameter")
    print("  A median hard against an edge of the WIDE box means widen further;")
    print("  one sitting mid-range with a tight spread is where emt8's bound goes.")
    sc = D.mean(dim=1).cpu().numpy()
    boot = {}
    if args.bootstrap:
        rng = np.random.default_rng(0)
        acc = {kk: [] for kk in WIDE}
        for _ in range(args.bootstrap):
            idx = rng.integers(0, args.n, args.n)
            top = idx[np.argsort(sc[idx])[:k]]
            for kk in WIDE:
                acc[kk].append(float(np.median(named[kk].numpy()[top])))
        boot = {kk: (float(np.percentile(v, 5)), float(np.percentile(v, 95)))
                for kk, v in acc.items()}
    print(f"  {'param':>13}{'box':>22}{'median':>12}"
          + (f"{'90% interval':>26}" if boot else "") + f"{'spread of the K':>26}")
    for kk, (lo, hi, lg) in WIDE.items():
        v = named[kk][torch.as_tensor(pool)].numpy()
        med = float(np.median(v))
        edge = "  <- AT AN EDGE" if (med <= lo * 1.05 or med >= hi * 0.95) else ""
        ci = ""
        if boot:
            b = boot[kk]
            # An interval spanning most of the box means the median is noise.
            span = ((math.log(max(b[1], 1e-30)) - math.log(max(b[0], 1e-30)))
                    / max(math.log(hi) - math.log(lo), 1e-9)) if lo > 0 else 1.0
            ci = f"[{b[0]:.3g}, {b[1]:.3g}]" + ("!" if span > 0.5 else "")
        print(f"  {kk:>13}{f'[{lo:.3g},{hi:.3g}]':>22}{med:>12.4g}"
              + (f"{ci:>26}" if boot else "")
              + f"{f'[{v.min():.3g}, {v.max():.3g}]':>26}{edge}")
    if boot:
        print("  ! = the 90% interval covers over half the box on a log axis:")
        print("      that median is noise and should not set a bound.")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with torch.no_grad():
            for i, nm in enumerate(names):
                bi = int(best_i[i])
                y = peak(plate(Z[bi : bi + 1],
                               duration=args.duration, normalize=False))
                sf.write(os.path.join(args.out, f"{nm}__target.wav"),
                         T[i].cpu().numpy(), SAMPLE_RATE)
                sf.write(os.path.join(args.out, f"{nm}__probe.wav"),
                         y[0].cpu().numpy(), SAMPLE_RATE)
        # Every score, so the tables above can be recomputed without re-rendering.
        np.savez(os.path.join(args.out, "probe_scores.npz"),
                 mfcc=D.cpu().numpy(), gap62=G62.cpu().numpy(),
                 tilt=TL.cpu().numpy(), decay=DC.cpu().numpy(),
                 onset=ON.cpu().numpy(),
                 names=np.array(names), bands=cen,
                 **{k: named[k].numpy() for k in WIDE})
        csvp = os.path.join(args.out, "probe_best.csv")
        with open(csvp, "w") as fh:
            fh.write("ir,draw,mfcc,gap62_db," + ",".join(WIDE) + "\n")
            for i, nm in enumerate(names):
                bi = int(best_i[i])
                fh.write(f"{nm},{bi},{best[i].item():.6g},{G62[bi, i].item():.4g},"
                         + ",".join(f"{float(named[k][bi]):.8g}" for k in WIDE) + "\n")
        print(f"\nwrote {args.out}: best draw per IR beside its target, and "
              f"{csvp}.\n  src.emt.why_dark --dir {args.out} reads it directly, "
              f"which is how to see\n  WHERE the best reachable point still differs "
              f"-- the same four-way split,\n  but with the encoder taken out of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
