"""Does the resynthesis SOUND at the right pitch: two detectors, both sides.

    python scripts/ds_crepe_pitch.py --dirs data/juno/moog-minitaur \
        --arms oct_pre_magx_halfw oct_pre_hybridx oct_pre_logx_halfw \
        --n 50 --device cuda:3 --folder-peak 0.5

WHY NOT ds_pitch_error. That script reads the model's OWN parameters -- BFRQ
for the fundamental, MULT for the second oscillator -- and compares them to the
pitch in the filename. Two things break it. The label is not the sounding pitch
(these packs come back at -12 semis), and harmor has TWO oscillators which the
model is free to place independently, so when it splits the note between them
neither BFRQ nor BFRQ*MULT is "the pitch" and reading either one is wrong.

This measures the thing actually in question: estimate the pitch of the TARGET
audio and of the RESYNTHESIS, and compare. No filename, no parameters, no
assumption about how many oscillators either side used. If the model conveys
the pitch by any means -- one oscillator, two, or a beat between them -- an
audio-domain detector hears it.

TWO DETECTORS, because on this material they disagree and the disagreement
matters. On the Moog, CREPE reports the target at 160.7 Hz and the probe at
77.8 -- an octave apart.

  crepe   torchcrepe, a neural pitch model, on the same hop / fmin / fmax as
          diffsynth.f0.compute_f0. Trained largely on speech and acoustic
          instruments; a synth bass near the bottom of its range is exactly
          where its octave slips live, and it reported only 0.67 periodicity
          with 64% of frames voiced on this pack
  probe   ds_harmonic_probe's find_f0, maximising (mean dB at k*F) - (mean dB
          at (k-1/2)*F). That peaks ONLY at the true fundamental, because a
          subharmonic's hits and its misses are both real partials and score
          ~0 -- a structural guarantee about octaves rather than a learned
          tendency. Validated in this repo: a saw at F scores 87.4 against
          23.6 for the alternatives, and saw F + saw 2F still picks F

Where they disagree, believe probe. Both are applied identically to target and
output, so either gives a valid COMPARISON even when its absolute pitch is
off by an octave; what the disagreement costs is the interpretation of tgt_hz.

THE PROBE SEARCH IS ANCHORED, at --anchor +- --span octaves, the same window
for the target and for every resynthesis. It must not be centred on each
clip's own target: find_f0 always returns its argmax, so for a resynthesis
with no real harmonic structure the answer comes from wherever the window is,
and a window that follows the target makes the two correlate for reasons that
have nothing to do with the model -- which shows up as a slope near 1.0
sitting next to a median error of more than an octave. The default window is
31.6-2024 Hz, i.e. FREQ_RANGE, so nothing the model can emit falls outside it.

out_conf is the median score of the detections ON THE RESYNTHESIS, and it is
the column that says whether the rest of that row means anything: the probe
has no voicing gate and will report a confident-looking pitch for noise. Tens
of dB is a real harmonic sound; single digits is the detector shrugging.

WHAT IS REPORTED, per arm, per folder, per detector:

  tgt_hz      median estimated f0 of the TARGET clips
  med_cents   median absolute error in cents, per clip then across clips.
              100 cents is a semitone, 1200 an octave
  within50    per cent of clips inside 50 cents -- audibly in tune
  oct_err     per cent of clips whose error is within 50 cents of a whole
              number of octaves and at least half an octave out: the harmonic
              structure was found and the register was not. A LOW oct_err
              beside a large med_cents is worse news than a high one -- it
              means the misses are arbitrary rather than register slips
  slope       regression of predicted log-f0 on target log-f0, across clips.
              1.0 means the estimate MOVES with the target one for one; 0.0
              means it predicts the same pitch whatever it is handed. This is
              the number that showed the published models were pitch-blind
              (0.2-0.6), and the one to watch

CREPE frames are gated by periodicity on BOTH sides -- an unvoiced frame has
no pitch to be wrong about, and CREPE returns an arbitrary one there. A clip
with no surviving frames is dropped, which is why n can be well under --n; a
low n is itself a finding, since it means the resynthesis is not pitched
enough to track. The probe has no such gate: it is one windowed FFT over the
clip's active span and always returns something, with a score saying how much
to believe it.

The per-clip CREPE error is a median over frames, not a mean, so one bad frame
at a note transition cannot move it.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "external", "diffsynth"))

import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402

from ds_eval_folder import audio_files, load_clip, load_model   # noqa: E402
from ds_harmonic_probe import spectrum, find_f0           # noqa: E402

try:
    import torchcrepe
except ImportError:                                      # pragma: no cover
    torchcrepe = None

HOP = 128


def probe_f0(y: np.ndarray, sr: int, guess: float, span: float):
    """(F_hz, score_dB) from ds_harmonic_probe's octave-robust detector.

    The alternative to CREPE, and better suited to this material. CREPE is a
    neural model trained largely on speech and acoustic instruments; a synth
    bass near the bottom of its range is exactly where its octave slips live,
    and on the Moog it reports 160.7 Hz where the probe reports 77.8.

    find_f0 maximises (mean dB at k*F) - (mean dB at (k-1/2)*F), which peaks
    ONLY at the true fundamental: a subharmonic's hits and its misses are both
    real partials, so it scores ~0. That is a structural guarantee about
    octaves rather than a learned tendency, which is why it is worth having as
    a second opinion instead of trusting one estimator.

    The score is the confidence in dB. Tens of dB is a clean harmonic sound; a
    few dB means the detection should not be believed.
    """
    fr, mg = spectrum(y, sr)
    if fr is None:
        return None, None
    return find_f0(fr, mg, guess, span_oct=span)


def crepe_f0(audio: torch.Tensor, sr: int, device: str, batch_size: int):
    """(f0_hz, periodicity) as [clips, frames] numpy, same settings as
    diffsynth.f0.compute_f0 so this and the training-time estimator agree.

    ONE CLIP PER CALL. torchcrepe.predict takes (1, samples) and frames it
    internally; handing it (50, 64000) does not mean a batch of 50, it means
    25000 frames pushed through conv1 in one go, which asks for 24 GiB.
    """
    f0s, pers = [], []
    for i in range(audio.shape[0]):
        f, p = torchcrepe.predict(
            audio[i:i + 1], sr, hop_length=HOP, pad=False, device=device,
            batch_size=batch_size, model="full", fmin=32.0, fmax=2000.0,
            return_periodicity=True)
        f0s.append(f[0].cpu().numpy())
        pers.append(p[0].cpu().numpy())
    return np.stack(f0s), np.stack(pers)


def clip_cents(f_tgt, p_tgt, f_out, p_out, thresh):
    """Median |cents| error over frames voiced on BOTH sides, or None."""
    m = (p_tgt >= thresh) & (p_out >= thresh) & (f_tgt > 0) & (f_out > 0)
    if not m.any():
        return None, None, None
    c = 1200.0 * np.log2(f_out[m] / f_tgt[m])
    return (float(np.median(np.abs(c))), float(np.median(c)),
            float(np.median(f_tgt[m])))


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dirs", nargs="+", required=True, metavar="DIR")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--ckpt", default="latest.ckpt")
    p.add_argument("--n", type=int, default=50, help="Clips per folder; 0 = all.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--length", type=float, default=4.0)
    p.add_argument("--match", default=None, metavar="REGEX")
    p.add_argument("--folder-peak", type=float, default=None, metavar="P",
                   help="One gain per folder putting its MEDIAN peak at P. The "
                        "model is not level-invariant, so pass whatever "
                        "ds_eval_folder was given or the two disagree about "
                        "what the model was shown.")
    p.add_argument("--min-periodicity", type=float, default=0.2, metavar="F",
                   help="crepe only: frames below this on either side are "
                        "unvoiced and carry no pitch to compare.")
    p.add_argument("--estimator", default="both",
                   choices=("crepe", "probe", "both"),
                   help="Which pitch detector. 'probe' is ds_harmonic_probe's "
                        "octave-robust one, which is structurally immune to "
                        "the octave slips CREPE is prone to on bass material; "
                        "'both' runs each so their agreement is visible.")
    p.add_argument("--span", type=float, default=3.0, metavar="OCT",
                   help="probe only: half-width of the search, in octaves, "
                        "around --anchor. The default pair covers 31.6-2024 "
                        "Hz, which is FREQ_RANGE, so nothing the model can "
                        "emit falls outside the window.")
    p.add_argument("--anchor", type=float, default=253.0, metavar="HZ",
                   help="probe only: centre of the search, THE SAME for the "
                        "target and for every resynthesis. It must not be "
                        "derived per clip from that clip's target -- a window "
                        "that moves with the target makes an unpitched "
                        "resynthesis correlate with it and inflates slope "
                        "toward 1 for no reason involving the model.")
    p.add_argument("--min-score", type=float, default=0.0, metavar="DB",
                   help="probe only: drop clips whose detection scores below "
                        "this on either side. 0 keeps everything and lets the "
                        "reported scores speak; ~10 dB is where a detection "
                        "starts being worth believing.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--csv", default=None, metavar="PATH")
    args = p.parse_args()

    dev = args.device
    rng = random.Random(args.seed)

    # Clips first, once, so every arm sees the same ones.
    folders = {}
    for d in args.dirs:
        files = audio_files(d)
        if args.match:
            files = [f for f in files if re.search(args.match, os.path.basename(f))]
        if args.n and len(files) > args.n:
            files = rng.sample(files, args.n)
        files.sort()
        g = os.path.basename(os.path.normpath(d))
        raw = [load_clip(f, args.sr, args.length) for f in files]
        peaks = [r[2] for r in raw]
        gain = 1.0
        if args.folder_peak is not None:
            nz = [q for q in peaks if q > 0]
            med = float(np.median(nz)) if nz else 0.0
            gain = args.folder_peak / med if med > 0 else 1.0
        folders[g] = (files, np.stack([r[0] for r in raw]) * gain, gain)
        print(f"{g:<26}{len(files):>4} clips   gain {gain:.3f}")

    want_crepe = args.estimator in ("crepe", "both")
    want_probe = args.estimator in ("probe", "both")
    if want_crepe and torchcrepe is None:
        print("torchcrepe is not installed.  pip install torchcrepe   "
              "(or pass --estimator probe)")
        raise SystemExit(2)

    # Target pitch, once per folder -- it does not depend on the arm.
    tgt = {}
    for g, (files, x, _gain) in folders.items():
        d = {}
        if want_crepe:
            f0, per = crepe_f0(torch.from_numpy(x).float(), args.sr, dev,
                               args.batch_size)
            d["crepe"] = (f0, per)
            voiced = f0[per >= args.min_periodicity]
            med = float(np.median(voiced)) if voiced.size else float("nan")
            print(f"{g:<26}target crepe  {med:8.1f} Hz   periodicity "
                  f"{float(np.median(per)):.2f}   voiced "
                  f"{100.0 * voiced.size / max(per.size, 1):.0f}%")
        if want_probe:
            F, S = [], []
            for i, f in enumerate(files):
                Fi, si = probe_f0(x[i], args.sr, args.anchor, args.span)
                F.append(Fi if Fi is not None else np.nan)
                S.append(si if si is not None else np.nan)
            d["probe"] = (np.array(F), np.array(S))
            lo, hi = args.anchor * 2 ** -args.span, args.anchor * 2 ** args.span
            print(f"{g:<26}target probe  {np.nanmedian(F):8.1f} Hz   score "
                  f"{np.nanmedian(S):.1f} dB   p10-p90 "
                  f"{np.nanpercentile(F, 10):.0f}-{np.nanpercentile(F, 90):.0f}"
                  f" Hz   window {lo:.0f}-{hi:.0f} Hz")
        tgt[g] = d

    print(f"\n{'folder':<20}{'arm':<24}{'est':<7}{'n':>4}{'tgt_hz':>9}"
          f"{'med_cents':>11}{'within50':>10}{'oct_err':>9}{'slope':>8}"
          f"{'out_conf':>10}")
    rows = []

    def emit(folder, arm, est, per_clip, conf=None):
        """per_clip: list of [abs_cents, signed_cents, target_hz]."""
        if not per_clip:
            print(f"{folder[:20]:<20}{arm:<24}{est:<7}  nothing measurable")
            return
        a = np.array(per_clip)
        off = a[:, 0]
        # An octave error is a NEAR-MISS on a whole number of octaves, and far
        # enough out that it is not just an in-tune estimate.
        oct_e = 100.0 * float(((np.abs(off - 1200 * np.round(off / 1200)) <= 50)
                               & (off >= 600)).mean())
        lg_t = np.log2(a[:, 2])
        lg_o = lg_t + a[:, 1] / 1200.0
        slope = (float(np.polyfit(lg_t, lg_o, 1)[0])
                 if len(lg_t) > 2 and np.ptp(lg_t) > 1e-6 else float("nan"))
        c = "" if conf is None else f"{np.nanmedian(conf):>9.1f}dB"
        print(f"{folder[:20]:<20}{arm:<24}{est:<7}{len(a):>4}"
              f"{np.median(a[:, 2]):>9.1f}{np.median(off):>11.1f}"
              f"{100.0 * float((off <= 50).mean()):>9.0f}%{oct_e:>8.0f}%"
              f"{slope:>8.2f}{c}")

    for arm in args.arms:
        model, _cfg, note = load_model(os.path.join(args.root, arm),
                                       args.ckpt, dev)
        if model is None:
            print(f"{'':<20}{arm:<24}  skipped -- {note}")
            continue
        for g, (files, x, _gain) in folders.items():
            with torch.no_grad():
                out, _ = model({"audio": torch.from_numpy(x).float().to(dev)})
            out = out.detach().cpu()

            if want_crepe:
                ft, pt = tgt[g]["crepe"]
                fo, po = crepe_f0(out, args.sr, dev, args.batch_size)
                acc = []
                for i, f in enumerate(files):
                    a, signed, hz = clip_cents(ft[i], pt[i], fo[i], po[i],
                                               args.min_periodicity)
                    if a is None:
                        continue
                    acc.append([a, signed, hz])
                    rows.append([arm, g, "crepe", os.path.basename(f),
                                 f"{hz:.2f}", f"{signed:.1f}", f"{a:.1f}", ""])
                emit(g, arm, "crepe", acc)

            if want_probe:
                Ft, St = tgt[g]["probe"]
                on = out.numpy()
                acc, conf = [], []
                for i, f in enumerate(files):
                    if not np.isfinite(Ft[i]):
                        continue
                    # ANCHORED, not centred on this clip's target. A window
                    # that follows the target makes an unpitched resynthesis
                    # correlate with it and fakes a slope near 1.
                    Fo, so = probe_f0(on[i], args.sr, args.anchor, args.span)
                    if Fo is None:
                        continue
                    if args.min_score > 0 and (so < args.min_score
                                               or St[i] < args.min_score):
                        continue
                    signed = 1200.0 * float(np.log2(Fo / Ft[i]))
                    acc.append([abs(signed), signed, float(Ft[i])])
                    conf.append(so)
                    rows.append([arm, g, "probe", os.path.basename(f),
                                 f"{Ft[i]:.2f}", f"{signed:.1f}",
                                 f"{abs(signed):.1f}", f"{so:.1f}"])
                emit(g, arm, "probe", acc, conf)

    print("\n  med_cents  median |error| in cents; 100 = a semitone, 1200 = an octave\n"
          "  within50   per cent of clips audibly in tune\n"
          "  oct_err    per cent whose error is a whole number of octaves --\n"
          "             the harmonic structure was found, the register was not\n"
          "  slope      predicted log-f0 regressed on target log-f0 ACROSS clips.\n"
          "             1.0 = the estimate moves with the target; 0.0 = it\n"
          "             predicts one pitch regardless of what it is given\n"
          "  est        which detector. crepe is a neural pitch model, prone to\n"
          "             octave slips on bass; probe is ds_harmonic_probe's\n"
          "             hits-minus-half-multiples search, structurally immune\n"
          "             to them. Where the two disagree, believe probe -- but\n"
          "             a large disagreement is itself worth looking at, since\n"
          "             both are applied identically to target and output.")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["arm", "folder", "est", "file", "tgt_hz",
                        "cents", "abs_cents", "score_db"])
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
