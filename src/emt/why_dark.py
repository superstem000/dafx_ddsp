"""Why the renders sound darker than the real EMT-140, in numbers.

    python -m src.emt.why_dark --dir emt7_listen --fmax 12000

Reads the wavs eval_real_ir already wrote (`<stem>__target.wav` against
`<stem>__<arm>.wav`) and decomposes "too dark" into four separable causes, in
the order they can be acted on:

  UNREACHABLE   energy in the target above the renderer's ceiling. No parameter
                setting can produce it; it is a property of --fmax alone. This
                is the part where changing the encoder, the loss or the search
                bounds does nothing at all.
  TILT          the third-octave spectrum of the render minus the target, in dB,
                BELOW the ceiling. This is what the parameters could fix and did
                not, and its shape says which: a monotone downward tilt is the
                damping corner (loss_F1) or the modal density (h); a notch is
                geometry.
  DECAY         per-octave T60. "Darker" and "longer" are the same measurement
                seen twice -- a plate whose high bands ring too long sounds dull
                because the highs smear rather than because they are quiet.
  ONSET         band energy in the first 20 ms over the band's total. The crash.
                fp_x/fp_y are pinned, so this is identical across every render
                in the dataset by construction and cannot track the target.

Nothing here loads a checkpoint or a GPU: it compares audio to audio, so it
says what differs, never why the encoder chose it.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
HOP = 512
# Bands are analysed at a LONGER window than the rest. At n_fft 2048 the bin
# spacing is 21.5 Hz and a third octave at 50 Hz is 11.6 Hz wide, so that band
# contains NO bin and reads as digital silence -- which is how an earlier run of
# this printed -291 dB at 50 Hz and looked like a brick-wall high pass in the
# recordings. At 8192 the spacing is 5.4 Hz and every band from 40 Hz up has at
# least two. HOP is unchanged, so decay time resolution is not affected.
BAND_FFT = 8192


def third_octave_edges(lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Centres on the ISO R40 grid, and the edges at 2^(+-1/6) around them."""
    k = np.arange(-20, 14)                       # 1000 * 2^(k/3): 25 Hz .. 20 kHz
    c = 1000.0 * 2.0 ** (k / 3.0)
    c = c[(c >= lo) & (c <= hi)]
    return c, np.stack([c * 2.0 ** (-1 / 6), c * 2.0 ** (1 / 6)])


def stft_power(x: np.ndarray, n_fft: int = BAND_FFT) -> np.ndarray:
    """[frames, bins] power. Plain numpy so this runs in either venv."""
    w = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    n = 1 + max(0, (len(x) - n_fft) // HOP)
    if n < 1:
        x = np.pad(x, (0, n_fft - len(x)))
        n = 1
    idx = np.arange(n_fft)[None, :] + HOP * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(x[idx] * w, axis=-1)) ** 2


def band_energy(P: np.ndarray, edges: np.ndarray,
                n_fft: int = BAND_FFT) -> np.ndarray:
    """[bands, frames] energy, summing the bins that fall in each band.

    A band with no bin in it is nan, never zero: zero survives every later log
    as a very negative number that reads as real silence in the signal rather
    than as an absent measurement.
    """
    f = np.fft.rfftfreq(n_fft, 1.0 / SR)
    out = np.empty((edges.shape[1], P.shape[0]), dtype=np.float64)
    for b in range(edges.shape[1]):
        m = (f >= edges[0, b]) & (f < edges[1, b])
        out[b] = P[:, m].sum(axis=1) if m.any() else np.nan
    return out


def t60(e: np.ndarray) -> float:
    """T20 x 3 by Schroeder backward integration; nan if it never reaches -25 dB.

    T20 rather than T30 because a 4 s render of a plate whose damper is open
    does not always fall 35 dB inside the file, and a fit over a range the
    signal never entered is a straight line through noise.
    """
    if e.sum() <= 0:
        return float("nan")
    edc = np.cumsum(e[::-1])[::-1]
    db = 10.0 * np.log10(np.maximum(edc / edc[0], 1e-30))
    i5 = int(np.argmax(db <= -5.0)) if (db <= -5.0).any() else -1
    i25 = int(np.argmax(db <= -25.0)) if (db <= -25.0).any() else -1
    if i5 < 0 or i25 <= i5:
        return float("nan")
    t = np.arange(i5, i25 + 1) * (HOP / SR)
    a = np.polyfit(t, db[i5 : i25 + 1], 1)[0]
    return float("nan") if a >= 0 else 3.0 * (-20.0 / a)


def read(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"{p} is {sr} Hz; eval_real_ir writes {SR}")
    return x / max(float(np.abs(x).max()), 1e-12)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", default="emt7_listen",
                   help="an eval_real_ir output directory")
    p.add_argument("--fmax", type=float, default=12000.0,
                   help="the renderer's ceiling, for the UNREACHABLE section. "
                        "Must be what the checkpoints trained at; eval_real_ir "
                        "prints it in its banner.")
    args = p.parse_args()

    d = Path(args.dir)
    targets = sorted(d.glob("*__target.wav"))
    if not targets:
        raise SystemExit(f"no *__target.wav in {d}")
    stems = [t.name[: -len("__target.wav")] for t in targets]
    arms = sorted({q.name.split("__", 1)[1][:-4] for q in d.glob("*__*.wav")
                   if not q.name.endswith("__target.wav")})
    print(f"{d}: {len(stems)} IRs, arms {arms}, ceiling {args.fmax:.0f} Hz\n")

    cen, edges = third_octave_edges(40.0, 20000.0)
    oc_c, oc_e = third_octave_edges(62.5, 16000.0)
    oc = np.arange(0, len(oc_c), 3)              # every third band = octaves

    # --- UNREACHABLE ------------------------------------------------------
    f = np.fft.rfftfreq(BAND_FFT, 1.0 / SR)
    above = f >= args.fmax
    print(f"=== UNREACHABLE   target energy above {args.fmax:.0f} Hz")
    print("  no parameter setting can produce this; it is --fmax and nothing else")
    print(f"  {'ir':<28}{'share':>9}{'dB below total':>16}")
    shares = []
    for t, stem in zip(targets, stems):
        P = stft_power(read(t))
        s = float(P[:, above].sum() / max(P.sum(), 1e-30))
        shares.append(s)
        print(f"  {stem:<28}{100*s:>8.2f}%{10*np.log10(max(s,1e-30)):>16.1f}")
    print(f"  {'MEAN':<28}{100*float(np.mean(shares)):>8.2f}%"
          f"{10*np.log10(max(float(np.mean(shares)),1e-30)):>16.1f}\n")

    # --- TILT -------------------------------------------------------------
    # Averaged over IRs, and only below the ceiling: above it the answer is
    # "-inf dB" for every arm and would swamp the part that is actionable.
    acc = {a: defaultdict(list) for a in arms}
    dec = {a: defaultdict(list) for a in arms}
    dec_t = defaultdict(list)
    ons = {a: defaultdict(list) for a in arms}
    ons_t = defaultdict(list)
    n20 = max(1, int(0.020 * SR / HOP))

    for t, stem in zip(targets, stems):
        xt = read(t)
        Bt = band_energy(stft_power(xt), edges)
        Ot = band_energy(stft_power(xt), oc_e)
        for j in oc:
            dec_t[("t60", j)].append(t60(Ot[j]))
            ons_t[j].append(Ot[j][:n20].sum() / max(Ot[j].sum(), 1e-30))
        for a in arms:
            q = d / f"{stem}__{a}.wav"
            if not q.exists():
                continue
            xr = read(q)
            Br = band_energy(stft_power(xr), edges)
            Or = band_energy(stft_power(xr), oc_e)
            tt, tr = max(Bt.sum(), 1e-30), max(Br.sum(), 1e-30)
            for b in range(len(cen)):
                # Both sides normalised by TOTAL energy, so this is spectral
                # shape and not level -- the wavs are already peak-normalised
                # and peak is not energy.
                acc[a][b].append(10 * np.log10(max(Br[b].sum() / tr, 1e-30))
                                 - 10 * np.log10(max(Bt[b].sum() / tt, 1e-30)))
            for j in oc:
                dec[a][("t60", j)].append(t60(Or[j]))
                ons[a][j].append(Or[j][:n20].sum() / max(Or[j].sum(), 1e-30))

    print("=== TILT   render minus target, dB, per third octave, shape not level")
    print("  negative = the render has less there. A monotone slide toward the")
    print("  right is the damping corner or the modal density; a notch is geometry.")
    print(f"  {'band':>8}" + "".join(f"{a:>18}" for a in arms))
    for b, c in enumerate(cen):
        if c >= args.fmax:
            continue
        vals = [float(np.mean(acc[a][b])) for a in arms]
        if not np.all(np.isfinite(vals)):
            print(f"  {c:>7.0f}" + "".join(f"{'n/a':>18}" for _ in arms))
            continue
        print(f"  {c:>7.0f}" + "".join(f"{v:>18.1f}" for v in vals))
    print()

    print("=== DECAY   T60 per octave, seconds (nan = never fell 25 dB in file)")
    print(f"  {'band':>8}{'target':>10}" + "".join(f"{a:>18}" for a in arms))
    for j in oc:
        v = np.array(dec_t[("t60", j)], dtype=float)
        row = "".join(
            f"{float(np.nanmean(np.array(dec[a][('t60', j)], dtype=float))):>18.2f}"
            for a in arms)
        print(f"  {oc_c[j]:>7.0f}{float(np.nanmean(v)):>10.2f}" + row)
    print()

    print("=== ONSET   share of each octave's energy in the first 20 ms")
    print("  fp_x/fp_y are pinned, so every render's column is the same plate")
    print("  struck in the same place, whatever the target did.")
    print(f"  {'band':>8}{'target':>10}" + "".join(f"{a:>18}" for a in arms))
    for j in oc:
        row = "".join(f"{100*float(np.mean(ons[a][j])):>17.2f}%" for a in arms)
        print(f"  {oc_c[j]:>7.0f}{100*float(np.mean(ons_t[j])):>9.2f}%" + row)

    print("\n  Read UNREACHABLE first. Whatever share it reports is a ceiling on")
    print("  how close any of these arms can get, and it is spent before the")
    print("  encoder makes a single decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
