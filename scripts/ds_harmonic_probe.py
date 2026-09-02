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

ODD MULTIPLES ARE THE EXCEPTION, and the score does not catch them. A candidate
at 3F has every hit on a real partial (3F, 6F, 9F) and every miss on an empty
bin (1.5F, 4.5F), so it scores as well as F; 5F and 7F likewise. Inside the
default +-2 octave window around a correct label this rarely bites, but widen
the window and the detector walks up to 3F and reports it at full confidence --
which is exactly what happened when ds_crepe_pitch searched 32-2024 Hz and got
a bass pack's median "fundamental" at 262 Hz with a 73-1544 Hz spread. find_f0
therefore tests the argmax's integer divisors afterwards and keeps the lowest
that still explains the spectrum. Anything measured before that fix, including
this pack's F = 77.8 and its -12 semis, is worth re-running.

STEP 2, THE OFFSET, AND A TRAP IN IT. SEMIS is 12*log2(F_detected/F_labelled).
A -12 does NOT by itself mean the pack sounds an octave below its written note.
It means the LOWEST COMMON SERIES is an octave below the label -- and for a
fifth pair that series is a phantom neither oscillator plays. The Moog pack
reads -12 with P1 = -39.7 dB: its oscillators are at the written note and a
fifth above it, and the -12 is an artifact of where their common subharmonic
falls. Read SEMIS together with P1 or it will mislead, as it did here.

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


def _score(freqs, mag, F, f_max=7600.0):
    """Harmonics minus half-multiples at F, or None if F cannot be scored."""
    if F < 16.0:
        return None
    K = int(min(20, f_max // F))
    if K < 4:
        return None
    hit = np.mean([peak_db(freqs, mag, k * F) for k in range(1, K + 1)])
    miss = np.mean([peak_db(freqs, mag, (k - 0.5) * F) for k in range(1, K + 1)])
    return hit - miss


def find_f0(freqs, mag, f_guess, f_max=7600.0, span_oct=2.0, step_cents=10.0,
            divisor_tol=3.0, max_divisor=5):
    """(F, score) maximising harmonics-minus-half-multiples. See module docstring.

    THE SCORE ALONE IS NOT ENOUGH, and a wide search window exposes it. The
    half-multiple term rules out F/2 and 2F, which is what it was tested on,
    but NOT odd multiples. A candidate at 3F has every hit on a real partial
    (3F, 6F, 9F) and every miss on an empty bin (1.5F, 4.5F), so it scores as
    well as F does; 5F and 7F likewise. With the original +-2 octave window
    anchored on a filename's pitch that never mattered, because 3F is 1.58
    octaves up and rarely reachable from a correct label. Widen the window and
    the detector walks up to 3F and reports it with full confidence.

    So after taking the argmax, test its integer divisors and keep the LOWEST
    one that still explains the spectrum within divisor_tol dB. If 3F really
    does explain it, F explains it at least as well, and F is the answer.

    Testing named divisors rather than "the lowest candidate within tol"
    deliberately: the 10-cent grid puts many near-equal candidates just below
    any peak, and taking the lowest of those would drift the estimate down a
    few cents at a time for no reason.

    divisor_tol=0 restores the pre-fix behaviour.
    """
    best, best_s = f_guess, -1e9
    n = int(2 * span_oct * 1200 / step_cents) + 1
    for c in np.linspace(-span_oct * 1200, span_oct * 1200, n):
        F = f_guess * 2.0 ** (c / 1200.0)
        s = _score(freqs, mag, F, f_max)
        if s is not None and s > best_s:
            best, best_s = F, s
    if divisor_tol > 0:
        for d in range(max_divisor, 1, -1):       # lowest F first
            s = _score(freqs, mag, best / d, f_max)
            if s is not None and s >= best_s - divisor_tol:
                return best / d, s
    return best, best_s


def structure(freqs, mag, F, f_max=7600.0, K=16):
    """(alpha, P1, bump2, bump3, bump4, odd_only) from the partials at k*F.

    P1 is the residual at k=1: how far the FUNDAMENTAL itself sits above or
    below the rolloff the rest of the partials follow. It is what separates an
    octave pair from a fifth. Two oscillators at F and 2F have their lowest
    common series at F, so partial 1 is real and P1 ~ 0. Two at F and 1.5F have
    theirs at F/2 -- 1.5F is 3*(F/2) -- so the detected fundamental is a
    PHANTOM that neither oscillator plays, partial 1 is empty, and P1 goes
    strongly negative while BUMP2 and BUMP3 are both large. That combination
    is a fifth, and nothing else produces it.
    """
    K = int(min(K, f_max // F))
    if K < 6:
        return (np.nan,) * 6
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
    return (float(-slope / 20.0), float(r[0]), bump(2), bump(3), bump(4),
            float(odd.mean() - even.mean()) if odd.size and even.size else np.nan)


def top_peaks(freqs, mag, n=12, f_max=2000.0, floor_db=-45.0):
    """The n strongest spectral peaks below f_max, as [(hz, dB below max)].

    The check that needs no theory at all. Every conclusion about a pack's
    oscillators rests on find_f0 having located the right comb, and find_f0 can
    be wrong -- its score compares a candidate's harmonics against its own
    half-multiples, and for a candidate an octave LOW the half-multiples are
    quarter-multiples of the truth, all empty, so the difference stays large.
    The score is therefore biased toward low candidates and cannot be used to
    audit itself.

    Reading the peak list settles it directly: if the partials are at 45, 68,
    91, 113 Hz then 22.7 is real and every other harmonic is present; if they
    are at 45, 91, 136 then the fundamental is 45 and 22.7 was a slip.
    """
    lo = 20.0 * np.log10(max(float(mag.max()), 1e-12)) + floor_db
    m = mag[:len(mag)]
    idx = [i for i in range(2, len(m) - 1)
           if m[i] > m[i - 1] and m[i] >= m[i + 1]
           and freqs[i] <= f_max
           and 20.0 * np.log10(max(float(m[i]), 1e-12)) >= lo]
    idx.sort(key=lambda i: -m[i])
    peaks = [(float(freqs[i]),
              20.0 * np.log10(float(m[i]) / max(float(mag.max()), 1e-12)))
             for i in idx[:n]]
    return sorted(peaks)


def classify(p1, b2, b3):
    """One of 'fifth', 'octave', 'single', '?' from the residual shape.

    A FIFTH is the only structure that empties the detected fundamental: two
    oscillators at 1.5:1 have their lowest common series an octave below the
    lower one, so partial 1 is a phantom (P1 far negative) while multiples of
    2 AND of 3 both carry real energy. An OCTAVE pair keeps its fundamental
    (P1 ~ 0) and lifts only the even partials.
    """
    if not np.isfinite(p1):
        return "?"
    if p1 < -15.0 and b3 > 3.0:
        return "fifth"
    if p1 > -10.0 and b2 > 3.0 and b3 < 3.0:
        return "octave"
    if p1 > -10.0 and b2 < 3.0 and b3 < 3.0:
        return "single"
    return "?"


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
    p.add_argument("--peaks", type=int, default=0, metavar="N",
                   help="Also print the N strongest partials, in Hz, for the "
                        "first --peak-files clips of each folder. This is the "
                        "check that does not depend on find_f0 being right, "
                        "and nothing else here is independent of it.")
    p.add_argument("--peak-files", type=int, default=3, metavar="N")
    args = p.parse_args()

    print(f"{'folder':<26}{'n':>4}{'F_hz':>9}{'semis':>8}{'score':>8}"
          f"{'alpha':>8}{'P1':>8}{'BUMP2':>8}{'BUMP3':>8}{'BUMP4':>8}"
          f"{'ODD_ONLY':>10}")
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
        acc, shown = [], []
        for f in files:
            lab = 440.0 * 2.0 ** ((midi_of(f, args.midi_re) - 69) / 12.0)
            y, _a, _pk, _raw = load_clip(f, args.sr, args.length)
            fr, mg = spectrum(y, args.sr)
            if fr is None:
                continue
            F, sc = find_f0(fr, mg, lab)
            st = structure(fr, mg, F)
            if len(shown) < args.peak_files:
                shown.append((f, (fr, mg, F)))
            acc.append([F, 12.0 * np.log2(F / lab), sc, *st])
            rows.append([g, os.path.basename(f), f"{lab:.2f}", f"{F:.2f}"]
                        + [f"{v:.2f}" for v in acc[-1][1:]]
                        + [classify(acc[-1][4], acc[-1][5], acc[-1][6])])
        if not acc:
            print(f"{g:<26} nothing measurable")
            continue
        if args.peaks:
            print()
            for f, (fr2, mg2, F2) in shown[:args.peak_files]:
                pk = top_peaks(fr2, mg2, args.peaks)
                print(f"  {os.path.basename(f)[:52]:<52} F={F2:.1f} Hz")
                print("    " + "  ".join(f"{hz:.1f}({db:+.0f})" for hz, db in pk))
                if pk:
                    base = pk[0][0]
                    print("    as multiples of the lowest peak "
                          f"{base:.1f} Hz:  "
                          + " ".join(f"{hz / base:.2f}" for hz, _d in pk))
            print()
        kinds = [classify(r[4], r[5], r[6]) for r in acc]
        m = np.nanmedian(np.array(acc), axis=0)
        print(f"{g:<26}{len(acc):>4}{m[0]:>9.1f}{m[1]:>8.1f}{m[2]:>8.1f}"
              f"{m[3]:>8.2f}{m[4]:>8.1f}{m[5]:>8.1f}{m[6]:>8.1f}{m[7]:>8.1f}"
              f"{m[8]:>10.1f}   " + "  ".join(
                  f"{k} {100.0 * kinds.count(k) / len(kinds):.0f}%"
                  for k in ("octave", "fifth", "single", "?")
                  if kinds.count(k)))

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
          "  P1        how far the FUNDAMENTAL sits above/below the rolloff the\n"
          "            other partials follow. ~0 means F is really played. Very\n"
          "            negative means F is a PHANTOM nothing plays -- which is\n"
          "            what a FIFTH looks like: oscillators at 2F and 3F, so\n"
          "            P1 far below zero with BUMP2 and BUMP3 both large.\n"
          "            An OCTAVE pair instead has P1 ~ 0 and only BUMP2 large\n"
          "  BUMPn     dB by which every n-th partial sits ABOVE the smooth fit.\n"
          "            A second oscillator at n*F does exactly that. Above ~3 dB\n"
          "            is a real second source; near 0 is one oscillator\n"
          "  ODD_ONLY  odd partials above even ones: a square or narrow pulse\n"
          "            rather than a saw\n"
          "  the tail   per-file classification counts. 'fifth' is P1 < -15 with\n"
          "            BUMP3 > 3; 'octave' is P1 > -10 with BUMP2 > 3 and BUMP3\n"
          "            small; 'single' is neither bump. A mixed pack shows here\n"
          "            and nowhere else, since every other column is a median\n"
          "  A fifth is P1 very negative + BUMP2 and BUMP3 both large; an octave is\n"
          "  P1 ~ 0 + BUMP2 large alone. Read them per FILE, not as a median: a\n"
          "  pack can be part one and part the other, and the median hides it.\n"
          "  Two oscillators in unison or slightly detuned share every partial\n"
          "  and cannot be separated here at all -- detuning shows as beating,\n"
          "  in ds_source_diag's AM_hz / AM_dep.")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["folder", "file", "label_hz", "F_hz", "semis", "score",
                        "alpha", "p1", "bump2", "bump3", "bump4", "odd_only",
                        "kind"])
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
