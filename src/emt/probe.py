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
    "Ly":      (1.0, 3.5, False),
    "h":       (0.0003, 0.003, True),      # 0.3 mm to 3 mm; emt7 railed at 0.56
    "T0":      (0.1, 1e6, True),           # emt7 railed at its 500 ceiling
    "rho":     (2500.0, 12000.0, False),   # emt7 railed at its 7000 floor
    "E":       (6e10, 2.6e11, False),      # emt7 railed at its 1.7e11 floor
    "T60_DC":  (0.3, 12.0, True),
    # NOT an absolute T60_F1: a RATIO of T60_DC, which is how quiet7 carries it
    # and the reason emt7 could pin it at 1.2 safely. beta is
    # 3ln10/dOmSq * (1/T60_F1 - 1/T60_DC), so T60_F1 > T60_DC makes beta
    # negative, sig negative at high omega, and r = exp(-sig*k) > 1 -- the mode
    # GROWS over 176,400 samples until it overflows to inf and poisons every
    # metric to nan. Damping has to increase with frequency; the ratio is what
    # enforces it. 0.95 rather than 1.0 keeps beta strictly positive.
    "T60_F1_ratio": (0.02, 0.95, True),
    "loss_F1": (200.0, 50000.0, True),     # emt7 railed at both 2000 and 8000
    "fp_x":    (0.02, 0.5, False),         # emt7 pinned the drive point
    "fp_y":    (0.02, 0.5, False),
    "op_x":    (0.05, 0.95, False),
    "op_y":    (0.05, 0.95, False),
}
FIXED = {"Lx": 1.0, "nu": 0.30}


def dd_grid(Lx, Ly, h, T0, rho, E, nu, fmax):
    """SevenParamPlate's own DDx/DDy, for pinning the grid from the box corner."""
    w = 2.0 * math.pi * fmax
    D = E * h ** 3 / (12.0 * (1.0 - nu * nu))
    disc = max((-T0 + math.sqrt(max(T0 * T0 + 4.0 * w * w * rho * h * D, 0.0)))
               / (2.0 * D), 0.0)
    s = math.sqrt(disc)
    return math.floor(Lx / math.pi * s), math.floor(Ly / math.pi * s)


def corner(box, fmax):
    """Densest draw in the box: max Ly, min h, min T0, max rho, min E."""
    return dd_grid(FIXED["Lx"], box["Ly"][1], box["h"][0], box["T0"][0],
                   box["rho"][1], box["E"][0], FIXED["nu"], fmax)


def sample(box, n, seed, dev):
    """[n, 14] in PARAM_ORDER, uniform in the box (log-uniform where flagged)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cols, named = {}, {}
    for k, (lo, hi, lg) in box.items():
        u = torch.rand(n, generator=g)
        named[k] = (torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
                    if lg else lo + u * (hi - lo))
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
    p.add_argument("--chunk-elems", type=int, default=400_000_000)
    p.add_argument("--mode-bucket", type=int, default=1024)
    p.add_argument("--no-compile", action="store_true",
                   help="eager modal sum. 8.5x slower here; compile is safe "
                        "because nothing in this script is compared against a "
                        "dataset rendered elsewhere, so the numerics contract "
                        "35e4529 describes does not apply.")
    p.add_argument("--top", type=int, default=32, help="K for the bounds report")
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
    T = peak(torch.from_numpy(np.stack(tg)).to(dev))
    with torch.no_grad():
        Tm, Tb = mfcc(T), bands(T)
    print(f"{len(names)} target(s) from {args.wav_dir}\n")

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
            bad = ~ok.unsqueeze(1) | ~torch.isfinite(d)
            D[s0 : s0 + z.shape[0]] = torch.where(bad, torch.inf, d)
            G62[s0 : s0 + z.shape[0]] = torch.where(bad, torch.inf, g62)
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

    # --- THE BOUNDS --------------------------------------------------------
    k = min(args.top, args.n)
    # Top K by MEAN score over the fifteen IRs, not per-IR winners: a parameter
    # that matters concentrates for all of them, and one that does not stays
    # uniform over its range no matter which IR you rank by.
    pool = torch.argsort(D.mean(dim=1))[:k].cpu().tolist()
    print(f"\n=== THE BOUNDS   the top {k} draws by mean score, per parameter")
    print("  A median hard against an edge of the WIDE box means widen further;")
    print("  one sitting mid-range with a tight spread is where emt8's bound goes.")
    print(f"  {'param':>10}{'wide box':>24}{'winners: med [min,max]':>34}")
    for kk, (lo, hi, lg) in WIDE.items():
        v = named[kk][torch.as_tensor(pool)].numpy()
        edge = ""
        if np.median(v) <= lo * 1.05 or np.median(v) >= hi * 0.95:
            edge = "  <- AT AN EDGE"
        print(f"  {kk:>10}{f'[{lo:.3g}, {hi:.3g}]':>24}"
              f"{f'{np.median(v):.4g} [{v.min():.3g}, {v.max():.3g}]':>34}{edge}")

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
