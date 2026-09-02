"""What each arm DOES with its two oscillators: ratio, balance, waveform.

    python scripts/ds_osc_usage.py --dirs data/juno/moog-minitaur \
        --arms oct_synth_magx_halfw oct_synth_hybridx oct_synth_logx_halfw \
        --n 50 --device cuda:3 --folder-peak 0.5

This asks a different question from every other script here. Not "how good is
the output" but "what did the model build". Listening to the oct arms it
sounded like only the linear one was using BOTH oscillators at an octave while
the others collapsed onto one -- and that is directly readable from the
predicted parameters, so it does not have to stay an impression.

harmor pins osc 1 to f0 and puts osc 2 at f0*MULT, with a per-oscillator
amplitude (sep_amp) and a per-oscillator saw<->square blend. So three things
say how the pair is being used:

  MULT      where osc 2 sits. The oct dataset draws it from {1, 2, 4}, so
            p_1 / p_2 / p_4 are the fractions of clips the arm places within
            50 cents of unison, an octave, and two octaves. A model that
            learned the quantized structure should put nearly all its mass on
            those three; one that did not will show up in `other`
  share2    osc 2's share of the mean control amplitude. Near 0 means the
            second oscillator is switched off and the arm is really a
            one-oscillator model whatever MULT says -- which is the thing to
            check before believing any MULT number. Near 0.5 is a balanced
            pair
  both      per cent of clips where BOTH oscillators carry real level
            (share2 between --both-lo and 1-(--both-lo)) AND MULT is not
            unison. That is the "actually using two oscillators an octave
            apart" count, one number
  mix1/2    each oscillator's saw<->square blend, 0 = saw, 1 = square

Amplitudes are the CONTROL level, not rendered power: harmor's saw and square
profiles do not sum to the same total, and the filter is applied afterwards, so
share2 indicates the split rather than measuring it.

MULT and osc_mix are static_params, so fill_params slices them to the last
frame and they are one value per clip. amplitudes is per-frame and is averaged
over frames, the same convention ds_pitch_error uses for its share column.
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

from diffsynth.processor import SCALE_FNS                # noqa: E402
from ds_eval_folder import audio_files, load_clip, load_model   # noqa: E402
from ds_pitch_error import harmor_of                     # noqa: E402


def predict(model, x: torch.Tensor):
    """(mult, share2, mix1, mix2) per clip, scaled to physical units.

    Scaled through harmor's own param_desc and the same SCALE_FNS table
    Processor.process uses, so a change to the MULT range cannot desync this
    from what the renderer actually does.
    """
    est = model.estimate_param({"audio": x})
    dag_in = model.synth.fill_params(est)
    proc, conn = harmor_of(model.synth)

    d = proc.param_desc["f0_mult"]
    mult = SCALE_FNS[d["type"]](dag_in[conn["f0_mult"]],
                                d["range"][0], d["range"][1])
    mult = mult[:, -1, 0].detach().cpu().numpy()

    d = proc.param_desc["osc_mix"]
    mix = SCALE_FNS[d["type"]](dag_in[conn["osc_mix"]],
                               d["range"][0], d["range"][1])
    mix = mix[:, -1, :].detach().cpu().numpy()

    a = dag_in[conn["amplitudes"]].mean(dim=1).detach().cpu().numpy()
    share2 = a[:, 1] / np.maximum(a.sum(axis=1), 1e-12)
    return mult, share2, mix[:, 0], mix[:, 1]


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
    p.add_argument("--folder-peak", type=float, default=None, metavar="P")
    p.add_argument("--tol-cents", type=float, default=50.0, metavar="C",
                   help="How close to 1 / 2 / 4 a predicted MULT has to be to "
                        "count as that ratio.")
    p.add_argument("--both-lo", type=float, default=0.2, metavar="F",
                   help="Minimum share for an oscillator to count as in use, "
                        "for the `both` column.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--csv", default=None, metavar="PATH")
    args = p.parse_args()

    rng = random.Random(args.seed)
    folders = {}
    for d in args.dirs:
        files = audio_files(d)
        if args.match:
            files = [f for f in files if re.search(args.match, os.path.basename(f))]
        if args.n and len(files) > args.n:
            files = rng.sample(files, args.n)
        files.sort()
        raw = [load_clip(f, args.sr, args.length) for f in files]
        gain = 1.0
        if args.folder_peak is not None:
            nz = [r[2] for r in raw if r[2] > 0]
            med = float(np.median(nz)) if nz else 0.0
            gain = args.folder_peak / med if med > 0 else 1.0
        g = os.path.basename(os.path.normpath(d))
        folders[g] = (files, np.stack([r[0] for r in raw]) * gain)
        print(f"{g:<26}{len(files):>4} clips   gain {gain:.3f}")

    print(f"\n{'folder':<20}{'arm':<24}{'n':>4}{'MULT_med':>10}{'p_1':>7}"
          f"{'p_2':>7}{'p_4':>7}{'other':>7}{'share2':>9}{'both':>7}"
          f"{'mix1':>7}{'mix2':>7}")
    rows = []
    for arm in args.arms:
        model, _cfg, note = load_model(os.path.join(args.root, arm),
                                       args.ckpt, args.device)
        if model is None:
            print(f"{'':<20}{arm:<24}  skipped -- {note}")
            continue
        for g, (files, x) in folders.items():
            with torch.no_grad():
                mult, share2, mix1, mix2 = predict(
                    model, torch.from_numpy(x).float().to(args.device))
            near = {}
            claimed = np.zeros(len(mult), dtype=bool)
            for r in (1.0, 2.0, 4.0):
                m = np.abs(1200.0 * np.log2(mult / r)) <= args.tol_cents
                near[r] = 100.0 * float(m.mean())
                claimed |= m
            other = 100.0 * float((~claimed).mean())
            both = 100.0 * float(((share2 >= args.both_lo)
                                  & (share2 <= 1.0 - args.both_lo)
                                  & (mult >= 1.5)).mean())
            print(f"{g[:20]:<20}{arm:<24}{len(mult):>4}{np.median(mult):>10.2f}"
                  f"{near[1.0]:>6.0f}%{near[2.0]:>6.0f}%{near[4.0]:>6.0f}%"
                  f"{other:>6.0f}%{np.median(share2):>9.2f}{both:>6.0f}%"
                  f"{np.median(mix1):>7.2f}{np.median(mix2):>7.2f}")
            for i, f in enumerate(files):
                rows.append([arm, g, os.path.basename(f), f"{mult[i]:.3f}",
                             f"{share2[i]:.3f}", f"{mix1[i]:.3f}",
                             f"{mix2[i]:.3f}"])

    print("\n  MULT_med  median predicted ratio of osc 2 to osc 1\n"
          "  p_1/2/4   per cent of clips within --tol-cents of unison, an\n"
          "            octave, two octaves. The oct dataset only ever contains\n"
          "            those three, so a trained arm should have little `other`\n"
          "  share2    osc 2's share of the mean control amplitude. Near 0 means\n"
          "            the second oscillator is OFF and the arm is really a\n"
          "            one-oscillator model whatever MULT says -- check this\n"
          "            before reading any MULT column\n"
          "  both      per cent of clips using both oscillators (each above\n"
          "            --both-lo) at a non-unison ratio. The one number for\n"
          "            'is this arm actually building an octave pair'\n"
          "  mix1/2    saw<->square blend per oscillator, 0 = saw, 1 = square\n"
          "  Control level, not rendered power: the saw and square profiles do\n"
          "  not sum alike and the filter comes after, so share2 indicates the\n"
          "  split rather than measuring it.")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["arm", "folder", "file", "mult", "share2", "mix1", "mix2"])
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
