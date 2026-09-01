"""Is this pack one oscillator, or two stacked? Read off the partials at the KNOWN f0.

    python scripts/ds_harmonic_probe.py --dirs data/juno/*saw-bass-split data/juno/*pulse-bass

WHAT DECIDES IT. A single saw at f0 has partials at f0, 2f0, 3f0 and nowhere
else. A sub-oscillator an octave down adds a series at f0/2, which lands on
f0/2, 1.5f0, 2.5f0 -- the HALF-multiples, where a single oscillator has
nothing. So the level at f0/2 relative to f0, and the level of the half-
multiples generally, is the whole measurement. Two oscillators in unison or
detuned would show neither: they would look like one oscillator here, and the
give-away for those is beating, which ds_source_diag's AM columns cover.

WHY IT CAN BE READ AT ALL. The pitch is in the filename, so nothing has to be
estimated -- which matters because yin returned 1345 cents of apparent
instability on this pack, i.e. it lost the fundamental entirely, and every
harmonic measurement that goes through an f0 tracker inherits that.

ONE FFT OVER THE WHOLE NOTE, not a spectrogram average. A 1.3 s note at 16 kHz
gives sub-Hz resolution, which is what separates f0/2 from f0 at the bottom of
a bass pack; a 1024-point frame at 15.6 Hz spacing cannot. The decay smears the
partials slightly and that is fine -- the question is which partials EXIST, not
their exact envelope.

The harmonic profile is reported beside a saw's own 1/k, so a pack that is a
plain sawtooth is visible as such: -6 dB per doubling, every harmonic present.
A square would show the odd harmonics only.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                       # noqa: E402

from ds_eval_folder import audio_files, load_clip        # noqa: E402
from ds_pitch_error import MIDI_RE, midi_of              # noqa: E402


def spectrum(y: np.ndarray, sr: int):
    """(freqs, magnitude) of the active span, one windowed FFT, sub-Hz bins."""
    pk = np.abs(y).max()
    if pk <= 0:
        return None, None
    nz = np.nonzero(np.abs(y) >= pk * 10.0 ** (-60.0 / 20.0))[0]
    if nz.size < 1024:
        return None, None
    seg = y[nz[0]:nz[-1] + 1]
    n = int(2 ** np.ceil(np.log2(max(seg.size, 4096))))
    mag = np.abs(np.fft.rfft(seg * np.hanning(seg.size), n))
    return np.fft.rfftfreq(n, 1.0 / sr), mag


def at(freqs, mag, f, half_width_hz=None):
    """Peak magnitude near f, over a window that scales with the bin spacing.

    A partial drifts and the FFT of a decaying note is not a line, so taking a
    single bin would read the shoulder rather than the peak. The window is
    +-3 bins by default, which at these lengths is a couple of Hz.
    """
    if f <= 0 or f >= freqs[-1]:
        return 0.0
    df = freqs[1] - freqs[0]
    w = max(1, int(round((half_width_hz or 3 * df) / df)))
    i = int(round(f / df))
    lo, hi = max(0, i - w), min(len(mag), i + w + 1)
    return float(mag[lo:hi].max()) if hi > lo else 0.0


def db(x, ref):
    return 20.0 * np.log10(max(x, 1e-12) / max(ref, 1e-12))


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dirs", nargs="+", required=True, metavar="DIR")
    p.add_argument("--midi-re", default=MIDI_RE)
    p.add_argument("--match", default=None, metavar="REGEX")
    p.add_argument("--n", type=int, default=20, help="Clips per folder.")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--length", type=float, default=4.0)
    p.add_argument("--harmonics", type=int, default=6)
    args = p.parse_args()

    K = args.harmonics
    print(f"{'folder':<26}{'n':>4}{'f0/2':>8}{'1.5f0':>8}{'2.5f0':>8}"
          + "".join(f"{'h' + str(k):>8}" for k in range(2, K + 1)))
    for d in args.dirs:
        files = audio_files(d)
        if args.match:
            files = [f for f in files if re.search(args.match, os.path.basename(f))]
        files = [f for f in files if midi_of(f, args.midi_re) is not None]
        files = files[:args.n]
        if not files:
            print(f"{os.path.basename(os.path.normpath(d))[:26]:<26} "
                  f"no files with a MIDI number")
            continue
        rows = []
        for f in files:
            midi = midi_of(f, args.midi_re)
            f0 = 440.0 * 2.0 ** ((midi - 69) / 12.0)
            y, _a, _pk, _raw = load_clip(f, args.sr, args.length)
            fr, mg = spectrum(y, args.sr)
            if fr is None:
                continue
            h1 = at(fr, mg, f0)
            if h1 <= 0:
                continue
            rows.append([db(at(fr, mg, f0 / 2), h1),
                         db(at(fr, mg, 1.5 * f0), h1),
                         db(at(fr, mg, 2.5 * f0), h1)]
                        + [db(at(fr, mg, k * f0), h1) for k in range(2, K + 1)])
        if not rows:
            print(f"{os.path.basename(os.path.normpath(d))[:26]:<26} nothing measurable")
            continue
        m = np.median(np.array(rows), axis=0)
        print(f"{os.path.basename(os.path.normpath(d))[:26]:<26}{len(rows):>4}"
              + "".join(f"{v:>8.1f}" for v in m))

    print(f"{'saw, ideal 1/k':<26}{'':>4}{'-inf':>8}{'-inf':>8}{'-inf':>8}"
          + "".join(f"{20 * np.log10(1.0 / k):>8.1f}" for k in range(2, K + 1)))
    print(f"{'square, ideal':<26}{'':>4}{'-inf':>8}{'-inf':>8}{'-inf':>8}"
          + "".join(f"{(20 * np.log10(1.0 / k) if k % 2 else -60.0):>8.1f}"
                    for k in range(2, K + 1)))
    print("\n  All levels in dB relative to the partial at the TRUE f0.\n"
          "  f0/2, 1.5f0, 2.5f0  the half-multiples. A single oscillator at f0\n"
          "                      has NOTHING there. Energy here means a second\n"
          "                      source an octave down -- the Juno-6's sub-\n"
          "                      oscillator -- and how much says how loud it is.\n"
          "                      Below about -40 dB is one oscillator.\n"
          "  h2..hK              the harmonic profile. A saw falls 6 dB per\n"
          "                      doubling with every harmonic present; a square\n"
          "                      has the even ones missing. Both ideals are\n"
          "                      printed underneath.\n"
          "  Two oscillators in UNISON or detuned look identical to one here;\n"
          "  ds_source_diag's AM_hz/AM_dep columns are what catch those.")


if __name__ == "__main__":
    main()
