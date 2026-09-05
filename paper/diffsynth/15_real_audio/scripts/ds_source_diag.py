"""What is in a folder of audio that h2of cannot make, measured rather than guessed.

    python scripts/ds_source_diag.py --dirs external/diffsynth/data/diffsynth_5-6/harmor_2oscfree \
        data/juno/*open-pad data/juno/*pulse-bass data/juno/*saw-bass-split --n 30

WHY. ds_eval_folder says the saw-bass pack reproduces the in-domain arm
ordering under every silence treatment and the pulse-bass pack reverses it
under all of them. That is a fact about the material, and the list of
candidate causes -- chorus, PWM, the sub-oscillator, noise, analog drift --
was reasoning from the Juno-6's front panel and from h2of.yaml's
static_params, not from the recordings. This measures each candidate on the
audio so the explanation is evidence.

The reference row is the synthetic training set itself. Every column below is
only interesting as a DIFFERENCE from it: harmor with an ADSR on amplitude and
cutoff is what these models were trained to inverse, so whatever that row says
is by definition inside the hypothesis space, and a folder that departs from it
in some column is departing in a way no parameter setting can absorb.

WHAT EACH COLUMN IS FOR, and which knob it maps to:

  AM_hz / AM_dep   Chorus. The Juno-6's ensemble is two delayed copies whose
                   delay is swept by a slow LFO, which beats against the dry
                   signal at the LFO rate -- an amplitude modulation at 0.2-12
                   Hz that survives after the decay trend is divided out. h2of
                   has no delay line at all (h1ofcf does, via ChorusFlanger), so
                   depth here is unmodellable residual. Tremolo would look the
                   same and the panel does not separate them; either way it is
                   modulation the synth cannot produce.

  EO / EO_sd       Pulse width, and EO_sd is the PWM detector. A square wave is
                   odd harmonics only; as the duty cycle leaves 50% the even
                   harmonics rise, so the even/odd energy ratio IS the duty
                   cycle, read off the spectrum. A saw has every harmonic at a
                   fixed ratio, so EO is constant and EO_sd ~ 0. Sweeping the
                   width makes EO_sd large. harmor's M_OSC and MULT are both in
                   h2of.yaml's static_params, so the harmonic PROFILE cannot
                   move within a note under any parameters -- only `cutoff` and
                   `amplitudes` can -- and a large EO_sd is therefore the one
                   measurement that would explain pulse-bass specifically.

  cent_10/50/90    Spectral centroid at a tenth, half and nine tenths of the
                   active span, in Hz. This is the filter envelope, and it is
                   the one time-varying thing h2of CAN do: `cutoff` comes from
                   its own ADSR. A big fall is a normal filter sweep and costs
                   the model nothing.

  cents_sd         Pitch instability in cents -- vibrato, VCO drift, or the
                   detune beating of two oscillators. BFRQ is static in h2of, so
                   any of it is unrepresentable.

  harm             Fraction of energy at integer multiples of f0. Low means
                   noise, inharmonicity, or a second voice at an unrelated
                   pitch. harmor is a harmonic oscillator bank plus a filter and
                   has no noise generator (NOISE_A/NOISE_C are envelope jitter,
                   not audio), so the shortfall from the reference row is
                   material the synth has no term for.

  atk_ms / t20_ms  The amplitude envelope: time to peak, and time from peak down
                   20 dB. Both are inside the hypothesis space, since
                   `amplitudes` is a free per-frame curve at inference. Reported
                   because a large mismatch changes which part of the clip
                   carries the error, not because the model cannot fit it.

EVERYTHING IS MEASURED ON THE ACTIVE SPAN, frames within 60 dB of the clip's
loudest, so a pack that ships each note with a long silent tail is not compared
against one that does not on statistics dominated by the tail. Per-clip values
are reduced by MEDIAN over the folder, not mean: sample packs contain the
occasional dud and one clip whose f0 tracker lost the fundamental should not
move a column.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                       # noqa: E402

from ds_eval_folder import audio_files, load_clip        # noqa: E402

N_FFT = 2048
HOP = 256


def frames(y: np.ndarray) -> np.ndarray:
    """Magnitude spectrogram, (bins, frames)."""
    import librosa
    return np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=True))


def active_mask(S: np.ndarray, db: float = 60.0) -> np.ndarray:
    e = (S ** 2).sum(axis=0)
    return e >= e.max() * 10.0 ** (-db / 10.0)


def harmonic_amps(S: np.ndarray, f0: np.ndarray, sr: int, f_max: float = 7600.0):
    """Per-frame amplitude at each integer multiple of f0.

    Takes the max over the bin and its two neighbours, because f0 drifts and a
    partial an eighth of a bin off centre would otherwise read as a dip in the
    harmonic profile rather than as the tracking error it is.
    """
    n_bins = S.shape[0]
    df = sr / N_FFT
    k_max = 40
    out = np.zeros((k_max, S.shape[1]))
    for t in range(S.shape[1]):
        if not np.isfinite(f0[t]) or f0[t] <= 0:
            continue
        for k in range(1, k_max + 1):
            f = k * f0[t]
            if f > f_max:
                break
            b = int(round(f / df))
            if 1 <= b < n_bins - 1:
                out[k - 1, t] = S[b - 1:b + 2, t].max()
    return out


def am_spectrum(e: np.ndarray, sr: int):
    """Peak modulation frequency in 0.2-12 Hz and its depth, after detrending.

    The decay of a note is itself a huge low-frequency component, so the
    envelope is divided by a slow moving average rather than analysed raw --
    what is left is the modulation ON the decay, which is what a chorus or a
    tremolo contributes and what a plain ADSR does not.
    """
    fps = sr / HOP
    if e.size < 16:
        return float("nan"), float("nan")
    w = max(3, int(round(fps / 3.0)) | 1)          # ~0.33 s, odd
    pad = w // 2
    smooth = np.convolve(np.pad(e, pad, mode="edge"), np.ones(w) / w, "valid")
    r = e / np.maximum(smooth, 1e-12) - 1.0
    r = r - r.mean()
    n = int(2 ** np.ceil(np.log2(r.size)))
    mag = np.abs(np.fft.rfft(r * np.hanning(r.size), n))
    freq = np.fft.rfftfreq(n, 1.0 / fps)
    band = (freq >= 0.2) & (freq <= 12.0)
    if not band.any() or mag[band].max() <= 0:
        return float("nan"), float("nan")
    i = np.argmax(np.where(band, mag, 0.0))
    # Depth as the peak's height over the median of the band: 1.0 is no
    # structure at all, large means one modulation rate dominates.
    med = np.median(mag[band])
    return float(freq[i]), float(mag[i] / med) if med > 0 else float("nan")


def clip_stats(y: np.ndarray, sr: int) -> dict | None:
    import librosa
    S = frames(y)
    m = active_mask(S)
    if m.sum() < 8:
        return None
    idx = np.nonzero(m)[0]
    lo, hi = idx[0], idx[-1] + 1
    Sa = S[:, lo:hi]
    e = (Sa ** 2).sum(axis=0)

    f0 = librosa.yin(y, fmin=32.0, fmax=2000.0, sr=sr,
                     frame_length=N_FFT, hop_length=HOP)
    f0 = f0[lo:min(hi, f0.size)]
    if f0.size < Sa.shape[1]:
        f0 = np.pad(f0, (0, Sa.shape[1] - f0.size), mode="edge")
    f0med = float(np.median(f0))
    cents = 1200.0 * np.log2(np.maximum(f0, 1e-6) / max(f0med, 1e-6))
    cents_sd = float(np.std(cents))

    ha = harmonic_amps(Sa, f0, sr)
    hp = ha ** 2
    tot = (Sa ** 2).sum(axis=0)
    harm = float(np.median(hp.sum(axis=0) / np.maximum(tot, 1e-20)))
    odd = hp[0::2].sum(axis=0)
    even = hp[1::2].sum(axis=0)
    eo = even / np.maximum(odd, 1e-20)
    # In log2, so a ratio that doubles and one that halves count the same. Only
    # frames carrying real energy, or the quiet tail dominates the spread.
    keep = tot >= tot.max() * 10.0 ** (-30.0 / 10.0)
    eol = np.log2(np.maximum(eo[keep], 1e-6)) if keep.any() else np.array([0.0])

    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    cen = (Sa * freqs[:, None]).sum(axis=0) / np.maximum(Sa.sum(axis=0), 1e-20)
    n = cen.size
    c10, c50, c90 = (float(cen[min(n - 1, int(n * f))]) for f in (0.1, 0.5, 0.9))

    fps = sr / HOP
    pk = int(np.argmax(e))
    atk = pk / fps * 1000.0
    after = e[pk:]
    below = np.nonzero(after <= e[pk] * 10.0 ** (-20.0 / 10.0))[0]
    t20 = float(below[0]) / fps * 1000.0 if below.size else float("nan")

    am_hz, am_dep = am_spectrum(e, sr)
    return dict(f0=f0med, cents_sd=cents_sd, am_hz=am_hz, am_dep=am_dep,
                eo=float(np.median(2.0 ** eol)), eo_sd=float(np.std(eol)),
                harm=harm, c10=c10, c50=c50, c90=c90, atk=atk, t20=t20)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dirs", nargs="+", required=True, metavar="DIR")
    p.add_argument("--n", type=int, default=30,
                   help="Clips per folder. yin is the slow part; 30 is plenty "
                        "for a median.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--length", type=float, default=4.0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = {}
    for d in args.dirs:
        files = audio_files(d)
        if not files:
            raise SystemExit(f"no audio files under {d}")
        if args.n and len(files) > args.n:
            files = sorted(rng.sample(files, args.n))
        g = os.path.basename(os.path.normpath(d))[:26]
        st = []
        for f in files:
            y, _a, _pk, _raw = load_clip(f, args.sr, args.length)
            s = clip_stats(y, args.sr)
            if s:
                st.append(s)
        if not st:
            print(f"{g}: nothing measurable")
            continue
        rows[g] = {k: float(np.nanmedian([s[k] for s in st])) for k in st[0]}
        print(f"  {g}: {len(st)} clip(s)")

    print(f"\n=== SOURCE   median over clips, on the active span only")
    print(f"{'folder':<28}{'f0_hz':>8}{'cents_sd':>10}{'AM_hz':>8}"
          f"{'AM_dep':>8}{'harm':>8}")
    for g, r in rows.items():
        print(f"{g:<28}{r['f0']:>8.1f}{r['cents_sd']:>10.1f}{r['am_hz']:>8.2f}"
              f"{r['am_dep']:>8.2f}{r['harm']:>8.3f}")
    print("  cents_sd  pitch instability; BFRQ is static in h2of, so any is\n"
          "            unrepresentable\n"
          "  AM_hz/dep amplitude modulation left after the decay is divided\n"
          "            out. A chorus beats at its LFO rate; h2of has no delay\n"
          "            line, so depth here is unmodellable residual\n"
          "  harm      energy at integer multiples of f0. The shortfall from\n"
          "            the reference row is noise or a second pitch, and\n"
          "            harmor has no term for either")

    print(f"\n=== TIMBRE AND ENVELOPE")
    print(f"{'folder':<28}{'EO':>8}{'EO_sd':>8}{'cent_10':>9}{'cent_50':>9}"
          f"{'cent_90':>9}{'atk_ms':>9}{'t20_ms':>9}")
    for g, r in rows.items():
        print(f"{g:<28}{r['eo']:>8.3f}{r['eo_sd']:>8.3f}{r['c10']:>9.0f}"
              f"{r['c50']:>9.0f}{r['c90']:>9.0f}{r['atk']:>9.0f}"
              f"{r['t20']:>9.0f}")
    print("  EO        even/odd harmonic energy ratio = the duty cycle read\n"
          "            off the spectrum. A saw sits near 1, a square near 0\n"
          "  EO_sd     ITS SPREAD OVER TIME, in log2 -- the PWM detector.\n"
          "            M_OSC and MULT are static_params, so h2of cannot move\n"
          "            the harmonic profile within a note at any setting\n"
          "  cent_*    spectral centroid at 10/50/90% of the active span; a\n"
          "            fall is a filter sweep, which `cutoff` CAN do\n"
          "  atk/t20   time to peak, and peak to -20 dB. Both fittable, since\n"
          "            `amplitudes` is a free per-frame curve at inference")


if __name__ == "__main__":
    main()
