"""Score the arms through a dry singing voice, and diagnose the IRs per band.

    python -m src.ddsp.eval_convolved --dir emt12_listen_last \
        --dry data/dry/m5_dona_straight.wav --out emt12_conv

Reads the wavs src/ddsp/eval_real_ir.py already wrote -- <ir>__target.wav and
<ir>__<arm>.wav -- so it needs no checkpoint, no GPU and no plate renderer, and
it cannot disagree with those renders because it IS those renders.

WHY CONVOLVE AT ALL, since it adds no information. Y = S*H, so every difference
between a render and the target is still entirely H's. What convolution does is
REWEIGHT the comparison by where the source has energy, and for a singing voice
that is a large reweighting: roughly 100 Hz - 3 kHz, which is
  - squarely where the modes are sparse enough to be heard individually, so
    modal colouration gets the most weight it can get, and
  - nowhere near 8-16 kHz, where the arms' worst decay errors live. emt12's eps
    arm rings 2.18 s at 8 kHz against a target of 0.98 and a voice barely
    excites it.
So the convolved numbers and the bare-IR numbers answer different questions and
may well rank the arms differently. Both are reported; neither replaces the
other. The bare-IR metric is source-independent and comparable to every number
in this project, and stays the headline.

WHAT IS NOT TRUE, and it is tempting: convolved MFCC is not bare-IR MFCC plus a
constant. log|Y| = log|S| + log|H| holds per bin for a full-length DFT, but the
mel pooling here happens BEFORE the log, and a 46 ms analysis frame against a
4 s IR smears energy across frames. Treat this as its own measurement.

TWO NORMALISATIONS, deliberately different.

  THE METRIC uses peak normalisation, exactly as eval_real_ir does, so a number
  here sits on the same scale as every other number in the project.
  THE WRITTEN WAVS are loudness-matched by BS.1770, because they are for ears.
  A darker render carries less energy, plays quieter, and gets rated worse for
  its level rather than its timbre -- and level is the one thing a listening
  test must not leak. --no-wavs skips them if you only want the tables.

THE PER-BAND DIAGNOSIS, and it is not decoration. EDT, T20, T30 and C50 per
octave band, computed on the IRs by ISO 3382's definitions.

  EDT   slope of the Schroeder curve over 0 to -10 dB, x6. Early decay, and the
        one that correlates with perceived reverberance.
  T20   slope over -5 to -25 dB, x3.   T30   over -5 to -35 dB, x2.
  C50   10log10(energy before 50 ms / energy after), the clarity ratio.

EDT vs T20 vs T30 ON THE TARGET ANSWERS AN OPEN QUESTION. If a real plate's
decay were a single exponential the three would agree. If EDT is much shorter
than T30, the early decay is faster than the late one, the decay is NOT
exponential, and then sig = alpha + beta*omega^2 cannot represent it however
long the training clip is -- which would mean the ~50 dB gap the arms show at
4 s is a model limitation and not an estimation failure. The arms are single
exponentials by construction, so their three numbers agree by definition; the
target's are the measurement.

A REAL RECORDING HAS A NOISE FLOOR and a Schroeder integral run into it reads
the noise, not the plate. The integration is truncated where the band envelope
falls within --noise-margin dB of the level estimated from the last 10% of the
file. That is a crude Lundeby; a band whose usable range does not reach -35 dB
reports its T30 as nan rather than a number that means the noise floor.
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
import torch
from scipy.signal import butter, fftconvolve, sosfiltfilt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.loss.losses import (                                # noqa: E402
    _get_dct, _get_mel_fb, _stft_mag, configure_loss_runtime)

SAMPLE_RATE = 44100
BANDS = (62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)


# --------------------------------------------------------------------- io ---
def discover(d: Path):
    """{ir: {arm: path}} from eval_real_ir's '<ir>__<arm>.wav' naming."""
    clips: dict = defaultdict(dict)
    for p in sorted(d.glob("*__*.wav")):
        m = re.match(r"(.+?)__(.+)\.wav$", p.name)
        if m:
            clips[m.group(1)][m.group(2)] = p
    return {k: v for k, v in clips.items() if "target" in v}


def mono(path: Path):
    x, sr = sf.read(path, dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def loudness_db(x: np.ndarray, sr: int, mode: str) -> float:
    if mode == "rms":
        return 20.0 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-12))
    import pyloudnorm                                        # noqa: PLC0415
    return float(pyloudnorm.Meter(sr).integrated_loudness(x))


# ------------------------------------------------------- room acoustics ---
def band_filter(x: np.ndarray, sr: int, fc: float, order: int = 4):
    """One octave band, zero-phase so the decay is not time-shifted."""
    lo, hi = fc / np.sqrt(2.0), fc * np.sqrt(2.0)
    nyq = sr / 2.0
    if hi >= nyq * 0.999:
        hi = nyq * 0.999
    if lo >= hi:
        return None
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def decay_curve(x: np.ndarray, sr: int, noise_margin: float):
    """Schroeder backward integral in dB, truncated above the noise floor.

    Returns (edc_db, n_used). Integrating the whole file would integrate the
    recording's noise, which flattens the tail and shortens every fitted slope.
    """
    e = x ** 2
    n = len(e)
    noise = float(np.mean(e[int(0.9 * n):])) if n > 20 else 0.0
    # Envelope on a 10 ms window, then the last point still clear of the floor.
    w = max(1, int(0.010 * sr))
    env = np.convolve(e, np.ones(w) / w, mode="same")
    thr = noise * 10.0 ** (noise_margin / 10.0)
    above = np.nonzero(env > thr)[0]
    cut = int(above[-1]) + 1 if len(above) else n
    cut = max(cut, int(0.05 * sr))
    edc = np.cumsum(e[:cut][::-1])[::-1]
    edc = np.maximum(edc, 1e-300)
    return 10.0 * np.log10(edc / edc[0]), cut


def slope_time(edc_db: np.ndarray, sr: int, top: float, bot: float) -> float:
    """Seconds for 60 dB, from a least-squares fit between two dB levels."""
    i0 = np.nonzero(edc_db <= top)[0]
    i1 = np.nonzero(edc_db <= bot)[0]
    if not len(i0) or not len(i1):
        return float("nan")
    a, b = int(i0[0]), int(i1[0])
    if b - a < int(0.01 * sr):
        return float("nan")
    t = np.arange(a, b) / sr
    p = np.polyfit(t, edc_db[a:b], 1)
    return float("nan") if p[0] >= 0 else float(-60.0 / p[0])


def acoustics(h: np.ndarray, sr: int, noise_margin: float):
    """EDT, T20, T30, C50 per octave band."""
    out = {}
    for fc in BANDS:
        xb = band_filter(h, sr, fc)
        if xb is None:
            continue
        edc, _ = decay_curve(np.ascontiguousarray(xb), sr, noise_margin)
        n50 = int(0.050 * sr)
        e = xb ** 2
        early, late = float(e[:n50].sum()), float(e[n50:].sum())
        out[fc] = dict(
            EDT=slope_time(edc, sr, 0.0, -10.0),
            T20=slope_time(edc, sr, -5.0, -25.0),
            T30=slope_time(edc, sr, -5.0, -35.0),
            C50=10.0 * np.log10(max(early, 1e-30) / max(late, 1e-30)),
        )
    return out


# ------------------------------------------------------------------ main ---
def selftest() -> int:
    """Check the decay fits against signals whose answer is known.

    The acoustics above are the part of this file most able to be quietly
    wrong -- a sign error, an off-by-one in the fit range, or a Schroeder
    integral run into the noise all produce plausible numbers. Two synthetic
    cases pin it:

      a SINGLE exponential at a known T60, where EDT, T20 and T30 must all
      return that T60 and the EDT/T30 ratio must be 1.00;
      a BI-exponential, fast early and slow late, where EDT must come back
      near the early rate and T30 near the late one, so the ratio is well
      below 1. That is the shape the target is being tested for.

    If the first case does not reproduce its own T60 to within a few percent,
    nothing in the per-band table means anything.
    """
    sr = SAMPLE_RATE
    t = np.arange(int(3.0 * sr)) / sr
    rng = np.random.default_rng(0)
    ok = True

    print("single exponential, true T60 = 1.50 s  (EDT, T20, T30 must all "
          "read 1.50, ratio 1.00)")
    h = rng.standard_normal(t.size) * 10.0 ** (-3.0 * t / 1.5)
    for fc, m in acoustics(h, sr, 5.0).items():
        if fc < 125 or fc > 8000:
            continue
        r = m["EDT"] / m["T30"] if m["T30"] else float("nan")
        bad = not (0.9 <= r <= 1.1) or not (1.35 <= m["T30"] <= 1.65)
        ok &= not bad
        print(f"  {fc:>7.0f} Hz  EDT {m['EDT']:.2f}  T20 {m['T20']:.2f}  "
              f"T30 {m['T30']:.2f}  ratio {r:.2f}{'   <-- FAIL' if bad else ''}")

    print("\nbi-exponential, early 0.40 s / late 2.50 s  (EDT must land near "
          "the early rate, T30 near the late, ratio well under 1)")
    h2 = rng.standard_normal(t.size) * (10.0 ** (-3.0 * t / 0.4)
                                        + 0.02 * 10.0 ** (-3.0 * t / 2.5))
    for fc, m in acoustics(h2, sr, 5.0).items():
        if fc < 125 or fc > 8000:
            continue
        r = m["EDT"] / m["T30"] if m["T30"] else float("nan")
        print(f"  {fc:>7.0f} Hz  EDT {m['EDT']:.2f}  T20 {m['T20']:.2f}  "
              f"T30 {m['T30']:.2f}  ratio {r:.2f}")

    print("\n" + ("OK -- the fits recover a known decay"
                  if ok else "FAILED -- do not trust the per-band table"))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--selftest", action="store_true",
                   help="check the decay fits against known signals and exit")
    p.add_argument("--dir", type=Path,
                   help="an eval_real_ir output directory")
    p.add_argument("--dry", type=Path,
                   help="dry mono source, e.g. a VocalSet excerpt")
    p.add_argument("--dry-start", type=float, default=0.0)
    p.add_argument("--dry-dur", type=float, default=6.0,
                   help="seconds of the dry source to use")
    p.add_argument("--out", type=Path, default=Path("convolved_eval"))
    p.add_argument("--match", choices=("lufs", "rms"), default="lufs")
    p.add_argument("--target-db", type=float, default=-23.0)
    p.add_argument("--noise-margin", type=float, default=5.0,
                   help="dB above the estimated noise floor at which the "
                        "Schroeder integration is truncated")
    p.add_argument("--no-wavs", action="store_true")
    p.add_argument("--device", default="cpu",
                   help="cpu by default: these are 60 short clips and a GPU "
                        "here only contends with training.")
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-mfcc", type=int, default=20)
    args = p.parse_args()

    if args.selftest:
        return selftest()
    missing = [f for f, v in (("--dir", args.dir), ("--dry", args.dry)) if not v]
    if missing:
        p.error(f"{' and '.join(missing)} required (or use --selftest alone)")

    clips = discover(args.dir)
    if not clips:
        raise SystemExit(f"no <ir>__<arm>.wav under {args.dir}")
    names = sorted(clips)
    arms = sorted({a for s in names for a in clips[s] if a != "target"})

    dry, sr = mono(args.dry)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"{args.dry} is {sr} Hz, expected {SAMPLE_RATE}. "
                         f"Resample it first -- silently resampling here would "
                         f"put a different anti-alias filter on the source than "
                         f"the IRs ever saw.")
    a0 = int(args.dry_start * sr)
    dry = dry[a0:a0 + int(args.dry_dur * sr)]
    if len(dry) < int(0.5 * sr):
        raise SystemExit(f"only {len(dry) / sr:.2f} s of dry audio selected")
    print(f"{args.dir}: {len(names)} IRs x {len(arms)} arms")
    print(f"dry: {args.dry.name}  {args.dry_start:.1f}-"
          f"{args.dry_start + len(dry) / sr:.1f} s  ({len(dry) / sr:.2f} s used)")

    dev = torch.device(args.device)
    configure_loss_runtime(SAMPLE_RATE, dev)
    win = torch.hann_window(args.n_fft, device=dev)

    # Bit-identical to eval_real_ir's definitions -- same helpers, same order.
    def mfcc(x, top_db=None):
        fb = _get_mel_fb(args.n_fft, args.n_mels)
        dct = _get_dct(args.n_mfcc, args.n_mels)
        mel = torch.matmul(fb.unsqueeze(0), _stft_mag(x, args.n_fft, args.hop) ** 2)
        db = 10.0 * torch.log10(mel + 1e-10)
        if top_db is not None:
            db = torch.maximum(db, db.amax(dim=(-2, -1), keepdim=True) - top_db)
        return torch.matmul(dct.unsqueeze(0), db)

    def linmag(x):
        return torch.stft(x, args.n_fft, hop_length=args.hop, window=win,
                          center=True, return_complex=True).abs()

    def peak_norm(x):
        return x / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)

    # --- convolve -------------------------------------------------------
    conv, irs = {}, {}
    for stem in names:
        for tag in ["target"] + arms:
            if tag not in clips[stem]:
                continue
            h, hsr = mono(clips[stem][tag])
            if hsr != SAMPLE_RATE:
                raise SystemExit(f"{clips[stem][tag]} is {hsr} Hz")
            irs[(stem, tag)] = h
            conv[(stem, tag)] = fftconvolve(dry, h)[: len(dry) + len(h)]
    print(f"convolved {len(conv)} stimuli, "
          f"{len(next(iter(conv.values()))) / sr:.2f} s each")

    t = {k: peak_norm(torch.from_numpy(v).float().to(dev)[None, :])
         for k, v in conv.items()}
    l1 = torch.nn.functional.l1_loss

    rows: dict = defaultdict(dict)
    for stem in names:
        for arm in arms:
            if (stem, arm) not in t:
                continue
            a, b = t[(stem, "target")], t[(stem, arm)]
            rows[arm][stem] = {
                "mfcc": l1(mfcc(a), mfcc(b)).item(),
                "mfcc_db80": l1(mfcc(a, 80.0), mfcc(b, 80.0)).item(),
                "linmag": l1(linmag(a), linmag(b)).item(),
            }

    # Saturation, rolled by one exactly as eval_real_ir does: consecutive names
    # are usually the same brightness, which is the harder reference.
    sat = {k: [] for k in ("mfcc", "mfcc_db80", "linmag")}
    for i, stem in enumerate(names):
        a = t[(stem, "target")]
        b = t[(names[(i + 1) % len(names)], "target")]
        sat["mfcc"].append(l1(mfcc(a), mfcc(b)).item())
        sat["mfcc_db80"].append(l1(mfcc(a, 80.0), mfcc(b, 80.0)).item())
        sat["linmag"].append(l1(linmag(a), linmag(b)).item())

    for key in ("mfcc", "mfcc_db80", "linmag"):
        print(f"\n=== {key} ON THE CONVOLVED AUDIO   "
              f"(peak-normalised both sides; lower is better)")
        print(f"{'ir':<34}" + "".join(f"{a[:22]:>24}" for a in arms))
        for stem in names:
            print(f"{stem[:34]:<34}"
                  + "".join(f"{rows[a][stem][key]:>24.4f}" for a in arms))
        print(f"{'MEAN':<34}"
              + "".join(f"{np.mean([rows[a][s][key] for s in names]):>24.4f}"
                        for a in arms))
        s = float(np.mean(sat[key]))
        print(f"{'SATURATION (another real IR)':<34}{s:>24.4f}")
        print(f"{'  arm / saturation':<34}"
              + "".join(f"{np.mean([rows[a][s2][key] for s2 in names]) / s:>24.3f}"
                        for a in arms))

    # --- per-band diagnosis ---------------------------------------------
    print(f"\n=== PER-BAND ACOUSTICS ON THE IRs   mean over {len(names)} IRs")
    print("  EDT/T20/T30 in seconds, C50 in dB.  nan = that band never reaches")
    print("  the fit range above its noise floor.")
    agg: dict = {tag: defaultdict(lambda: defaultdict(list))
                 for tag in ["target"] + arms}
    for (stem, tag), h in irs.items():
        for fc, m in acoustics(h, sr, args.noise_margin).items():
            for k, v in m.items():
                agg[tag][fc][k].append(v)

    for measure in ("EDT", "T20", "T30", "C50"):
        print(f"\n  {measure}")
        print(f"    {'':>10}" + "".join(f"{f:>9.0f}" for f in BANDS))
        for tag in ["target"] + arms:
            cells = ""
            for fc in BANDS:
                v = [x for x in agg[tag][fc][measure] if np.isfinite(x)]
                cells += f"{np.mean(v):>9.2f}" if v else f"{'nan':>9}"
            print(f"    {tag[:10]:>10}" + cells)

    # THE QUESTION THE TABLE EXISTS FOR.
    print("\n=== IS THE DECAY A SINGLE EXPONENTIAL?   EDT / T30, per band")
    print("  1.00 means the early and late decay have the same slope. The arms")
    print("  are single exponentials by construction so theirs is 1.00 by")
    print("  definition; only the TARGET's row is a measurement. Well below")
    print("  1.00 means the early decay is faster than the late one, which")
    print("  sig = alpha + beta*omega^2 cannot represent at any clip length.")
    print(f"    {'':>10}" + "".join(f"{f:>9.0f}" for f in BANDS))
    for tag in ["target"] + arms:
        cells = ""
        for fc in BANDS:
            e = [x for x in agg[tag][fc]["EDT"] if np.isfinite(x)]
            t3 = [x for x in agg[tag][fc]["T30"] if np.isfinite(x)]
            cells += (f"{np.mean(e) / np.mean(t3):>9.2f}"
                      if e and t3 and np.mean(t3) > 0 else f"{'nan':>9}")
        print(f"    {tag[:10]:>10}" + cells)

    # --- wavs -----------------------------------------------------------
    if not args.no_wavs:
        args.out.mkdir(parents=True, exist_ok=True)
        gains, peak = {}, 0.0
        for k, y in conv.items():
            g = 10.0 ** ((args.target_db - loudness_db(y, sr, args.match)) / 20.0)
            gains[k] = g
            peak = max(peak, float(np.abs(y * g).max()))
        head = min(1.0, 0.98 / peak) if peak > 0.98 else 1.0
        if head < 1.0:
            print(f"\n  peak after matching {peak:.3f} -- one common "
                  f"{20 * np.log10(head):+.2f} dB applied to every stimulus, so "
                  f"the match is preserved")
        for (stem, tag), y in conv.items():
            sf.write(args.out / f"{stem}__{tag}.wav",
                     (y * gains[(stem, tag)] * head).astype(np.float32),
                     sr, subtype="PCM_24")
        print(f"  wrote {len(conv)} loudness-matched wavs to {args.out}")
        print(f"\nFor the listening test:\n"
              f"  python scripts/make_webmushra.py --dir {args.out} "
              f"--id {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
