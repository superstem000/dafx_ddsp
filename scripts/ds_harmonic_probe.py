"""What is this pack actually made of: fundamental, oscillator count, ratios.

    python scripts/ds_harmonic_probe.py --dirs data/juno/korg-mono-poly \
                                               data/juno/*saw-bass-split

WHAT THE PREVIOUS VERSION GOT WRONG. It tested one hypothesis -- "is there a
second source an octave DOWN" -- by looking at f0/2 and the half-multiples, and
took the filename's pitch as the reference. That cannot see an oscillator an
octave UP, a fifth, a detuned unison, or a third oscillator, and it reports a
large number whenever the label and the sounding pitch disagree without being
able to say which. This assumes none of it.

STEP 1, THE FUNDAMENTAL, WITHOUT ASSUMING THE LABEL. Over a grid of candidates
spanning +-2 octaves around the labelled pitch, score each by

    score(F) = mean dB at k*F  -  mean dB at (k-1/2)*F

and take the best. That difference is what makes it octave-robust, which a
plain harmonic sum is not:

  the true F           every k*F is a real partial, every (k-1/2)*F is empty
                       -> large positive score
  F/2, an octave down  its harmonics land on the true partials AND on the gaps
                       between them, so hit and miss are both real: score ~ 0
  2F, an octave up     its (k-1/2) points are the odd true partials, also real:
                       score ~ 0

So subharmonics and overtones both collapse toward zero and only the true
fundamental peaks. SCORE is reported as the confidence: a clean harmonic sound
gives tens of dB, and a few dB means the detection should not be trusted.

STEP 2, THE OCTAVE OFFSET. SEMIS is 12*log2(F_detected / F_labelled). The Juno
and Korg bass packs both come back at -12: they sound an octave below their
written note, a DCO set to 16'. That is a fact about the pack, and it has to be
established before any pitch error is attributed to a model.

STEP 3, HOW MANY OSCILLATORS AND AT WHAT RATIO. Take the partial amplitudes at
k*F, fit the smooth rolloff every single oscillator has -- dB against log k,
whose slope is -20*ALPHA, with ALPHA 1.0 for an ideal saw -- and look at what
is left. A second oscillator at n*F adds energy to every n-th partial and
nothing else, so BUMPn is the mean residual on multiples of n minus the mean
residual off them:

  BUMP2 large   a second oscillator an octave up (or a sub an octave down,
                which is the same comb seen from the other end -- STEP 1
                decides which, since it locates the lowest one)
  BUMP3 large   a twelfth
  ODD_ONLY      the even partials sitting far below the fit: a square or a
                narrow pulse rather than a saw

WHAT IT STILL CANNOT SEE: two oscillators in UNISON or slightly detuned share
every partial, so no comb analysis separates them. Detuning shows up as beating
instead -- ds_source_diag's AM_hz and AM_dep columns -- and exact unison is
simply not identifiable from one recording.

One windowed FFT over the active span rather than a spectrogram average, for
sub-Hz bins: separating F from F/2 at the bottom of a bass pack needs it, and a
1024-point frame at 15.6 Hz spacing cannot.
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
    pk = np.abs(y).max()
    if pk <= 0:
        return None, None
    nz = np.nonzero(np.abs(y) >= pk * 10.0 ** (-60.0 / 20.0))[0]
    if nz.size < 2048:
        return None, None
    seg = y[nz[0]:nz[-1] + 1]
    n = int(2 ** np.ceil(np.log2(max(seg.size, 8192))))
    mag = np.abs(np.fft.rfft(seg * np.hanning(seg.size), n))
    return np.fft.rfftfreq(n, 1.0 / sr), mag


def peak_db(freqs, mag, f):
    """dB at the strongest bin within +-3 bins of f; -120 outside the band."""
    if f <= 0 or f >= freqs[-1]:
        return -120.0
    df = freqs[1] - freqs[0]
    i = int(round(f / df))
    lo, hi = max(0, i - 3), min(len(mag), i + 4)
    return 20.0 * np.log10(max(float(mag[lo:hi].max()), 1e-12)) if hi > lo else -120.0


def find_f0(freqs, mag, f_guess, f_max=7600.0, span_oct=2.0, step_cents=10.0):
    """(F, score) maximising harmonics-minus-half-multiples. See module docstring."""
    best, best_s = f_guess, -1e9
    n = int(2 * span_oct * 1200 / step_cents) + 1
    for c in np.linspace(-span_oct * 1200, span_oct * 1200, n):
        F = f_guess * 2.0 ** (c / 1200.0)
        if F < 16.0:
            continue
        K = int(min(20, f_max // F))
        if K < 4:
            continue
        hit = np.mean([peak_db(freqs, mag, k * F) for k in range(1, K + 1)])
        miss = np.mean([peak_db(freqs, mag, (k - 0.5) * F) for k in range(1, K + 1)])
        if hit - miss > best_s:
            best, best_s = F, hit - miss
    return best, best_s


def structure(freqs, mag, F, f_max=7600.0, K=16):
    """(alpha, bump2, bump3, bump4, odd_only) from the partial amplitudes at k*F."""
    K = int(min(K, f_max // F))
    if K < 6:
        return (np.nan,) * 5
    k = np.arange(1, K + 1)
    a = np.array([peak_db(freqs, mag, kk * F) for kk in k])
    a = a - a[0]
    # Every oscillator rolls off smoothly; the fit removes that so what is left
    # is comb structure. Slope is -20*alpha, so an ideal saw fits alpha = 1.
    slope, icpt = np.polyfit(np.log10(k), a, 1)
    r = a - (slope * np.log10(k) + icpt)
    def bump(n):
        on, off = r[(k % n) == 0], r[(k % n) != 0]
        return float(on.mean() - off.mean()) if on.size and off.size else np.nan
    odd, even = r[k % 2 == 1], r[k % 2 == 0]
    return (float(-slope / 20.0), bump(2), bump(3), bump(4),
            float(odd.mean() - even.mean()) if odd.size and even.size else np.nan)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dirs", nargs="+", required=True, metavar="DIR")
    p.add_argument("--midi-re", default=MIDI_RE)
    p.add_argument("--match", default=None, metavar="REGEX")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--length", type=float, default=4.0)
    p.add_argument("--csv", default=None, metavar="PATH")
    args = p.parse_args()

    print(f"{'folder':<26}{'n':>4}{'F_hz':>9}{'semis':>8}{'score':>8}"
          f"{'alpha':>8}{'BUMP2':>8}{'BUMP3':>8}{'BUMP4':>8}{'ODD_ONLY':>10}")
    rows = []
    for d in args.dirs:
        files = audio_files(d)
        if args.match:
            files = [f for f in files if re.search(args.match, os.path.basename(f))]
        files = [f for f in files if midi_of(f, args.midi_re) is not None][:args.n]
        g = os.path.basename(os.path.normpath(d))[:26]
        if not files:
            print(f"{g:<26} no files with a MIDI number")
            continue
        acc = []
        for f in files:
            lab = 440.0 * 2.0 ** ((midi_of(f, args.midi_re) - 69) / 12.0)
            y, _a, _pk, _raw = load_clip(f, args.sr, args.length)
            fr, mg = spectrum(y, args.sr)
            if fr is None:
                continue
            F, sc = find_f0(fr, mg, lab)
            st = structure(fr, mg, F)
            acc.append([F, 12.0 * np.log2(F / lab), sc, *st])
            rows.append([g, os.path.basename(f), f"{lab:.2f}", f"{F:.2f}"]
                        + [f"{v:.2f}" for v in acc[-1][1:]])
        if not acc:
            print(f"{g:<26} nothing measurable")
            continue
        m = np.nanmedian(np.array(acc), axis=0)
        print(f"{g:<26}{len(acc):>4}{m[0]:>9.1f}{m[1]:>8.1f}{m[2]:>8.1f}"
              f"{m[3]:>8.2f}{m[4]:>8.1f}{m[5]:>8.1f}{m[6]:>8.1f}{m[7]:>10.1f}")

    print("\n  F_hz      fundamental found WITHOUT using the label, by maximising\n"
          "            (dB at k*F) - (dB at (k-1/2)*F), which peaks only at the\n"
          "            true F: an octave down or up scores ~0 because its hits\n"
          "            and its misses are both real partials\n"
          "  semis     12*log2(F / labelled pitch). -12 means the pack sounds an\n"
          "            octave below its written note\n"
          "  score     that difference in dB, i.e. confidence. Tens of dB is a\n"
          "            clean harmonic sound; a few dB means do not trust F\n"
          "  alpha     rolloff exponent of the partials. An ideal saw is 1.00;\n"
          "            larger means a darker source or a closed filter\n"
          "  BUMPn     dB by which every n-th partial sits ABOVE the smooth fit.\n"
          "            A second oscillator at n*F does exactly that. Above ~3 dB\n"
          "            is a real second source; near 0 is one oscillator\n"
          "  ODD_ONLY  odd partials above even ones: a square or narrow pulse\n"
          "            rather than a saw\n"
          "  Two oscillators in unison or slightly detuned share every partial\n"
          "  and cannot be separated here at all -- detuning shows as beating,\n"
          "  in ds_source_diag's AM_hz / AM_dep.")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["folder", "file", "label_hz", "F_hz", "semis", "score",
                        "alpha", "bump2", "bump3", "bump4", "odd_only"])
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
