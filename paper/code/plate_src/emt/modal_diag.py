"""Why does a rendered IR ring as discrete tones where the real one does not?

    python -m src.emt.modal_diag --dir emt11_listen_last
    python -m src.emt.modal_diag --dir emt11_listen_last --ir emt_140_dark_4
    python -m src.emt.modal_diag --selftest

Reads the wavs eval_real_ir wrote -- <ir>__target.wav and <ir>__<arm>.wav -- and
measures, per octave band, the things that decide whether a band is heard as a
pitched hum or as reverb. No parameters, no model: this is what the audio
actually contains.

THE QUESTION. Listening finds a low buzz on the arms, absent from the target,
and its severity orders eps < linear < hybrid -- which is the order of their
modal spacing derived from (c, area), so spacing is clearly involved in the
DIFFERENCES between arms. But the same derivation says the real plate should
buzz at least as much: its modes are as narrow (5.43 s at 62 Hz is a 0.41 Hz
bandwidth) and its area is SMALLER (2.0 m^2 against linear's 2.816), so its
spacing is wider. Modal overlap predicts the target is the most discrete of the
four. It is the only one that sounds clean. Something the parameters do not
capture is smoothing it, and this measures the audio to find out what.

WHAT IT MEASURES, per octave band, for the target and every arm.

  FLATNESS   geometric mean / arithmetic mean of the power spectrum. 1.0 is
             white and toneless; near 0 is a few sharp peaks over silence. This
             is the direct form of "tonal or noise-like" and it needs no peak
             picking to be believed.
  PEAKS/Hz   prominent maxima per Hz, after subtracting a running median so a
             peak means "stands above its own neighbourhood" rather than "is
             loud". This is MODAL DENSITY MEASURED, and comparing it to the
             c^2/(2 pi f A) prediction is a check on that formula as much as on
             the plate.
  SPACING    the median gap between those peaks, in Hz. If it lands in the
             15-75 Hz roughness band, adjacent modes beat at a rate the ear
             hears as buzz rather than as tremolo.
  PROMINENCE median height of a peak over the local median, in dB. A mode 3 dB
             up is part of the texture; one 20 dB up is a tone.
  FLOOR      the band's late level relative to its peak, in dB -- the last 10%
             of the file, which for a real recording is its noise floor and for
             a synthetic render is numerical zero.

THE FLOOR COLUMN IS A HYPOTHESIS, not decoration. A real IR carries tape and
preamp noise; these renders carry none. A mode ringing 50 dB down is plainly
audible in a noiseless render and completely masked under a floor at -45. If
the target's floor is high and the arms' is not, the difference in what you
hear may be masking rather than modal structure -- and the fix is to add
matched noise to the renders before a listening test, which is presentation
matching rather than hiding anything.

READ IT AS A DISCRIMINATION. If the arms are much peakier (low flatness, high
prominence) than the target, the modal structure is the cause and no loss
function fixes it -- the renderer makes a lattice whatever the parameters. If
flatness and prominence are similar and only FLOOR differs, it is masking, and
that is fixable in the stimulus preparation.

--selftest checks the measurements against signals whose answer is known: a
synthetic lattice of decaying sinusoids at a chosen spacing, and band-limited
noise with the same decay. The first must come back peaky with that spacing,
the second flat. If it does not, nothing in the table means anything.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, find_peaks, sosfiltfilt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

SAMPLE_RATE = 44100
BANDS = (31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0)


def discover(d: Path):
    clips: dict = defaultdict(dict)
    for p in sorted(d.glob("*__*.wav")):
        m = re.match(r"(.+?)__(.+)\.wav$", p.name)
        if m:
            clips[m.group(1)][m.group(2)] = p
    return {k: v for k, v in clips.items() if "target" in v}


def band_edges(fc: float):
    return fc / np.sqrt(2.0), fc * np.sqrt(2.0)


def spectrum(x: np.ndarray, sr: int):
    """Magnitude spectrum of the whole IR, so resolution is 1/duration.

    No framing: a mode is a property of the entire decay, and a 2048-point
    frame at 21 Hz resolution cannot separate modes 14 Hz apart -- which is
    exactly the spacing in question.
    """
    n = 1 << int(np.ceil(np.log2(len(x))))
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), n))
    return X, np.fft.rfftfreq(n, 1.0 / sr)


def flatness(p: np.ndarray) -> float:
    """Geometric over arithmetic mean of power. 1 = white, ->0 = tonal."""
    p = np.maximum(p, 1e-30)
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def running_median(y: np.ndarray, w: int) -> np.ndarray:
    """Local baseline, so 'peak' means locally prominent rather than loud."""
    w = max(3, w | 1)
    pad = np.pad(y, w // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, w), axis=-1)


def band_stats(x: np.ndarray, sr: int, fc: float, prom_db: float):
    lo, hi = band_edges(fc)
    if hi >= sr / 2 * 0.999:
        return None
    X, f = spectrum(x, sr)
    sel = (f >= lo) & (f <= hi)
    if sel.sum() < 32:
        return None
    p = X[sel] ** 2
    db = 10.0 * np.log10(np.maximum(p, 1e-30))
    # Baseline over ~1/8 of the band, so the median tracks the envelope and not
    # the modes themselves.
    base = running_median(db, max(3, sel.sum() // 8))
    rel = db - base
    idx, props = find_peaks(rel, prominence=prom_db)
    df = float(f[sel][1] - f[sel][0])
    fr = f[sel][idx] if len(idx) else np.array([])
    gaps = np.diff(fr) if len(fr) > 1 else np.array([])

    # Late level relative to peak, in this band, from the time signal.
    sos = butter(4, [lo / (sr / 2), min(hi, sr / 2 * 0.999) / (sr / 2)],
                 btype="band", output="sos")
    xb = sosfiltfilt(sos, x)
    n = len(xb)
    e_peak = float(np.max(np.abs(xb[: max(1, n // 20)])))
    e_late = float(np.sqrt(np.mean(xb[int(0.9 * n):] ** 2)))
    floor = 20.0 * np.log10(max(e_late, 1e-30) / max(e_peak, 1e-30))

    return dict(
        flatness=flatness(p),
        peaks_per_hz=len(idx) / max(hi - lo, 1e-9),
        spacing=float(np.median(gaps)) if len(gaps) else float("nan"),
        prominence=float(np.median(props["prominences"])) if len(idx) else float("nan"),
        floor=floor,
        res=df,
    )


def mono(path: Path):
    x, sr = sf.read(path, dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def selftest(prom_db: float) -> int:
    """Known answers: a lattice at 15 Hz, and noise with the same decay."""
    sr, dur, T60 = SAMPLE_RATE, 4.0, 2.4
    t = np.arange(int(dur * sr)) / sr
    env = 10.0 ** (-3.0 * t / T60)
    rng = np.random.default_rng(0)

    lattice = np.zeros_like(t)
    fs = np.arange(48.0, 400.0, 15.0)
    for fq in fs:
        lattice += np.sin(2 * np.pi * fq * t + rng.uniform(0, 2 * np.pi))
    lattice *= env

    sos = butter(4, [40 / (sr / 2), 400 / (sr / 2)], btype="band", output="sos")
    noise = sosfiltfilt(sos, rng.standard_normal(t.size)) * env

    ok = True
    print(f"synthetic lattice, {len(fs)} modes spaced 15.0 Hz, T60 {T60}s")
    for fc in (125.0, 250.0):
        s = band_stats(lattice, sr, fc, prom_db)
        bad = not (10 <= s["spacing"] <= 20) or s["flatness"] > 0.2
        ok &= not bad
        print(f"  {fc:>6.0f} Hz  flatness {s['flatness']:.4f}  spacing "
              f"{s['spacing']:>5.1f} Hz  prominence {s['prominence']:>5.1f} dB"
              f"{'   <-- FAIL' if bad else ''}")
    print(f"\nband-limited noise, same decay (must be FLAT, no lattice)")
    for fc in (125.0, 250.0):
        s = band_stats(noise, sr, fc, prom_db)
        bad = s["flatness"] < 0.2
        ok &= not bad
        print(f"  {fc:>6.0f} Hz  flatness {s['flatness']:.4f}  spacing "
              f"{s['spacing']:>5.1f} Hz  prominence {s['prominence']:>5.1f} dB"
              f"{'   <-- FAIL' if bad else ''}")
    print("\n" + ("OK -- tonal and noise-like are separated"
                  if ok else "FAILED -- do not trust the table"))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--dir", type=Path, help="an eval_real_ir output directory")
    p.add_argument("--ir", default=None,
                   help="one IR stem; default averages over all of them")
    p.add_argument("--prom-db", type=float, default=6.0,
                   help="how far above its local baseline a maximum must stand "
                        "to count as a mode")
    args = p.parse_args()

    if args.selftest:
        return selftest(args.prom_db)
    if not args.dir:
        p.error("--dir required (or --selftest alone)")

    clips = discover(args.dir)
    if not clips:
        raise SystemExit(f"no <ir>__<arm>.wav under {args.dir}")
    stems = [args.ir] if args.ir else sorted(clips)
    missing = [s for s in stems if s not in clips]
    if missing:
        raise SystemExit(f"not found: {', '.join(missing)}")
    arms = ["target"] + sorted({a for s in stems for a in clips[s] if a != "target"})

    acc: dict = {a: defaultdict(lambda: defaultdict(list)) for a in arms}
    res = None
    for stem in stems:
        for a in arms:
            if a not in clips[stem]:
                continue
            x, sr = mono(clips[stem][a])
            for fc in BANDS:
                s = band_stats(x, sr, fc, args.prom_db)
                if s is None:
                    continue
                res = s["res"]
                for k, v in s.items():
                    if k != "res" and np.isfinite(v):
                        acc[a][fc][k].append(v)

    print(f"{args.dir}: {len(stems)} IR(s), {len(arms) - 1} arm(s) + target")
    print(f"spectral resolution {res:.2f} Hz, peak threshold {args.prom_db:.0f} dB "
          f"over the local baseline\n")

    labels = {"flatness": "FLATNESS   1 = noise-like, ->0 = a few sharp tones",
              "peaks_per_hz": "PEAKS per Hz   modal density, measured",
              "spacing": "SPACING (Hz)   median gap; 15-75 Hz beats as roughness",
              "prominence": "PROMINENCE (dB)   height over the local baseline",
              "floor": "FLOOR (dB)   late level vs peak; a real IR's noise floor"}
    for key, title in labels.items():
        print(f"=== {title}")
        print(f"    {'':>12}" + "".join(f"{f:>10.0f}" for f in BANDS))
        for a in arms:
            cells = ""
            for fc in BANDS:
                v = acc[a][fc][key]
                if not v:
                    cells += f"{'-':>10}"
                elif key == "flatness":
                    cells += f"{np.mean(v):>10.4f}"
                elif key == "peaks_per_hz":
                    cells += f"{np.mean(v):>10.3f}"
                else:
                    cells += f"{np.mean(v):>10.1f}"
            print(f"    {a[:12]:>12}" + cells)
        print()

    print("HOW TO READ IT")
    print("  arms much peakier than the target (lower flatness, higher")
    print("  prominence)      -> modal structure. The renderer makes a lattice")
    print("                      whatever the parameters, so no loss fixes it.")
    print("  similar peakiness, target's FLOOR far higher")
    print("                   -> masking. The target's own ringing is buried in")
    print("                      recording noise that the renders do not have;")
    print("                      add matched noise before any listening test.")
    print("  both             -> both, and the floor is the cheaper half to fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
