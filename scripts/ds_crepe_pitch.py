"""Does the resynthesis SOUND at the right pitch: CREPE on audio, both sides.

    python scripts/ds_crepe_pitch.py --dirs data/juno/moog-minitaur \
        --arms oct_pre_magx_halfw oct_pre_hybridx oct_pre_logx_halfw \
        --n 50 --device cuda:3 --folder-peak 0.5

WHY NOT ds_pitch_error. That script reads the model's OWN parameters -- BFRQ
for the fundamental, MULT for the second oscillator -- and compares them to the
pitch in the filename. Two things break it. The label is not the sounding pitch
(these packs come back at -12 semis), and harmor has TWO oscillators which the
model is free to place independently, so when it splits the note between them
neither BFRQ nor BFRQ*MULT is "the pitch" and reading either one is wrong.

This measures the thing that is actually in question: run CREPE on the TARGET
audio and on the RESYNTHESIS, and compare. No filename, no parameters, no
assumption about how many oscillators either side used. If the model conveys
the pitch by any means -- one oscillator, two, or a beat between them -- CREPE
hears it, because CREPE hears what a listener hears.

WHAT IS REPORTED, per arm and per folder:

  tgt_hz      median CREPE f0 of the TARGET clips. Sanity check first: this
              should agree with ds_harmonic_probe's F_hz. If it does not,
              CREPE is not tracking this material and nothing below means
              anything
  tgt_per     median periodicity (confidence) on the target. Below ~0.3 and
              the same warning applies
  med_cents   median absolute error in cents, per clip then across clips.
              100 cents is a semitone, 1200 an octave
  within50    per cent of clips inside 50 cents -- audibly in tune
  oct_err     per cent of clips whose error is within 50 cents of a whole
              number of octaves and at least half an octave out. This is the
              failure to separate from a random miss: an octave error means
              the harmonic structure was found and the register was not
  slope       regression of predicted log-f0 on target log-f0, across clips.
              1.0 means the estimate MOVES with the target one for one; 0.0
              means it predicts the same pitch whatever it is handed. This is
              the number that showed the published models were pitch-blind
              (0.2-0.6), and the one to watch

FRAMES ARE GATED BY PERIODICITY on both sides -- an unvoiced frame has no
pitch to be wrong about, and CREPE returns an arbitrary one there. A clip with
no surviving frames is dropped and counted in the header rather than silently
contributing a garbage number.

The per-clip error is a median over frames, not a mean, so one bad frame at a
note transition cannot move it.
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

try:
    import torchcrepe
except ImportError:                                      # pragma: no cover
    torchcrepe = None

HOP = 128


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
                   help="Frames below this on either side are unvoiced and "
                        "carry no pitch to compare.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--csv", default=None, metavar="PATH")
    args = p.parse_args()

    if torchcrepe is None:
        print("torchcrepe is not installed.  pip install torchcrepe")
        raise SystemExit(2)

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

    # CREPE on the targets once -- it does not depend on the arm.
    tgt_f0 = {}
    for g, (_files, x, _gain) in folders.items():
        f0, per = crepe_f0(torch.from_numpy(x).float(), args.sr, dev,
                           args.batch_size)
        tgt_f0[g] = (f0, per)
        voiced = f0[per >= args.min_periodicity]
        med_hz = float(np.median(voiced)) if voiced.size else float("nan")
        print(f"{g:<26}target  f0 {med_hz:8.1f} Hz   periodicity "
              f"{float(np.median(per)):.2f}   voiced frames "
              f"{100.0 * voiced.size / max(per.size, 1):.0f}%")

    print(f"\n{'folder':<22}{'arm':<24}{'n':>4}{'tgt_hz':>9}{'med_cents':>11}"
          f"{'within50':>10}{'oct_err':>9}{'slope':>8}")
    rows = []
    for arm in args.arms:
        model, _cfg, note = load_model(os.path.join(args.root, arm),
                                       args.ckpt, dev)
        if model is None:
            print(f"{'':<22}{arm:<24}  skipped -- {note}")
            continue
        for g, (files, x, _gain) in folders.items():
            ft, pt = tgt_f0[g]
            with torch.no_grad():
                out, _ = model({"audio": torch.from_numpy(x).float().to(dev)})
            fo, po = crepe_f0(out.detach().cpu(), args.sr, dev,
                              args.batch_size)
            acc, lg_t, lg_o = [], [], []
            for i, f in enumerate(files):
                a, signed, hz = clip_cents(ft[i], pt[i], fo[i], po[i],
                                           args.min_periodicity)
                if a is None:
                    continue
                acc.append([a, hz])
                lg_t.append(np.log2(hz))
                lg_o.append(np.log2(hz) + signed / 1200.0)
                rows.append([arm, g, os.path.basename(f), f"{hz:.2f}",
                             f"{signed:.1f}", f"{a:.1f}"])
            if not acc:
                print(f"{g[:22]:<22}{arm:<24}  no voiced frames on either side")
                continue
            a = np.array(acc)
            med = float(np.median(a[:, 0]))
            w50 = 100.0 * float((a[:, 0] <= 50).mean())
            # An octave error is a NEAR-MISS on a whole number of octaves, and
            # far enough out that it is not just an in-tune estimate.
            off = a[:, 0]
            oct_e = 100.0 * float(((np.abs(off - 1200 * np.round(off / 1200))
                                    <= 50) & (off >= 600)).mean())
            slope = (float(np.polyfit(lg_t, lg_o, 1)[0])
                     if len(lg_t) > 2 and np.ptp(lg_t) > 1e-6 else float("nan"))
            print(f"{g[:22]:<22}{arm:<24}{len(a):>4}{np.median(a[:, 1]):>9.1f}"
                  f"{med:>11.1f}{w50:>9.0f}%{oct_e:>8.0f}%{slope:>8.2f}")

    print("\n  med_cents  median |error| in cents; 100 = a semitone, 1200 = an octave\n"
          "  within50   per cent of clips audibly in tune\n"
          "  oct_err    per cent whose error is a whole number of octaves --\n"
          "             the harmonic structure was found, the register was not\n"
          "  slope      predicted log-f0 regressed on target log-f0 ACROSS clips.\n"
          "             1.0 = the estimate moves with the target; 0.0 = it\n"
          "             predicts one pitch regardless of what it is given\n"
          "  Check tgt_hz against ds_harmonic_probe's F_hz before reading any of\n"
          "  it: if CREPE and the harmonic probe disagree about the target's own\n"
          "  pitch, the measurement is not about the model.")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["arm", "folder", "file", "tgt_hz", "cents", "abs_cents"])
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
