"""How far off is each arm's pitch, against the MIDI number in the filename.

    python scripts/ds_pitch_error.py --dirs data/juno/*saw-bass-split \
        --arms synth_magx_halfw synth_hybridx synth_logx_halfw --device cuda:3

WHY. Spectral distances are poor at pitch -- Turian & Henry make it their whole
subject, and Torres and Masuda & Saito report the same -- so a listening test on
resynthesised notes risks measuring f0 error and nothing else, because pitch
dominates perceptual judgement in a way timbre does not. Before designing around
that with an f0-conditioned model (h2of_f0, Masuda & Saito's Raw-f0 setup, which
costs a dataset and three retrains) it is worth knowing whether pitch error is
actually the dominant term here, and whether it is one arm or all of them.

NO ESTIMATOR IN THE LOOP. Sample libraries put the pitch in the filename -- the
Juno packs end '<midi>-<velocity>.aiff' -- so the reference is exact. CREPE is
what you use when you only have audio; here it would only add its own error to
the measurement.

IN DOMAIN f0 IS THE EASY PARAMETER: the per-group table puts val_id/param's
f0_hz group at 0.031 of the constant-predictor error, against cutoff at 0.445.
So this is a question about transfer, not about the task.

TWO ERRORS, REPORTED SEPARATELY, because they mean different things:

  cents      the signed distance from the true pitch. An arm that is
             consistently sharp or flat is doing something different from one
             that is scattered.
  oct_fold   the same after folding into +-600 cents, i.e. ignoring octave
             errors. A pitch tracker landing an octave out is a different
             failure from one landing on an unrelated note, and the gap between
             the `within50` and `oct50` columns is exactly how much of the
             error is octaves.

AND THE SECOND OSCILLATOR, which f0 conditioning would NOT fix. harmor puts
oscillator 2 at f0 * MULT with MULT drawn on (1, 8) continuous, so a
non-integer multiple stacks an inharmonic partial series on the fundamental.
med_MULT and osc2_semis say where each arm is putting it; a real synth patch
uses unison (~1), an octave (2) or a fifth (1.5), which is 0, 12 or 7
semitones. Anything else is an inharmonic pair, and the generator draws MULT
uniformly, so most of the training set is one.
"""

from __future__ import annotations

import argparse
import csv as _csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "external", "diffsynth"))
sys.path.insert(0, HERE)

import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402

from diffsynth.processor import SCALE_FNS                # noqa: E402
from ds_eval_folder import audio_files, load_clip, load_model   # noqa: E402

MIDI_RE = r"-(\d{1,3})-\d{1,3}\.[A-Za-z0-9]+$"


def midi_of(path: str, pattern: str) -> int | None:
    m = re.search(pattern, os.path.basename(path))
    if not m:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 127 else None


def harmor_of(synth):
    """The Harmor node and its connection map, found by capability not by name."""
    for processor, connections in synth.dag:
        if "f0_hz" in processor.param_desc and "f0_mult" in processor.param_desc:
            return processor, connections
    raise SystemExit("no processor with f0_hz and f0_mult in this synth")


def predict(model, x: torch.Tensor):
    """(f0_hz, f0_mult) per clip, scaled to physical units.

    get_params cannot be used: calculate_params SKIPS Gen processors, and in
    the model synth harmor IS the only node, so nothing would be scaled. The
    scaling therefore comes from harmor's own param_desc through the same
    SCALE_FNS table Processor.process uses -- one source of truth, so a change
    to FREQ_RANGE or to the MULT range cannot desync this from the renderer.
    """
    est = model.estimate_param({"audio": x})
    dag_in = model.synth.fill_params(est)
    proc, conn = harmor_of(model.synth)
    out = []
    for name in ("f0_hz", "f0_mult"):
        d = proc.param_desc[name]
        v = dag_in[conn[name]]
        out.append(SCALE_FNS[d["type"]](v, d["range"][0], d["range"][1]))
    # static, so one value per clip: take the last frame.
    return (out[0][:, -1, 0].detach().cpu().numpy(),
            out[1][:, -1, 0].detach().cpu().numpy())


def fold(c: np.ndarray) -> np.ndarray:
    """Cents folded into +-600, i.e. the error ignoring octaves."""
    return (c + 600.0) % 1200.0 - 600.0


def summarise(cents: np.ndarray, mult: np.ndarray) -> dict:
    f = fold(cents)
    return {
        "n": len(cents),
        "med": float(np.median(cents)),
        "abs": float(np.median(np.abs(cents))),
        "w50": float(np.mean(np.abs(cents) <= 50.0)),
        "o50": float(np.mean(np.abs(f) <= 50.0)),
        "mult": float(np.median(mult)),
        "semis": float(np.median(12.0 * np.log2(np.maximum(mult, 1e-6)))),
    }


def slope_of(true_hz: np.ndarray, pred_hz: np.ndarray) -> float:
    """Octaves of predicted pitch per octave of true pitch.

    1.0 tracks; 0.0 is a constant regardless of input. The single number that
    says whether an arm is hearing pitch at all, as opposed to being offset.
    """
    a = np.log2(np.maximum(true_hz, 1e-6))
    b = np.log2(np.maximum(pred_hz, 1e-6))
    return float(np.polyfit(a, b, 1)[0]) if len(a) > 1 else float("nan")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dirs", nargs="+", required=True, metavar="DIR")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--ckpt", default="latest.ckpt")
    p.add_argument("--midi-re", default=MIDI_RE, metavar="REGEX",
                   help="Group 1 is the MIDI note. Default matches "
                        "'...-<midi>-<velocity>.<ext>'.")
    p.add_argument("--match", default=None, metavar="REGEX",
                   help="Keep only basenames matching, as in ds_eval_folder.")
    p.add_argument("--trim-pad", action="store_true",
                   help="Cut the zero padding this script added, so the LAST "
                        "frame lands inside the note. fill_params reads every "
                        "static parameter -- f0_hz, M_OSC, MULT, Q_FILT -- from "
                        "[:, -1:, :], and on a 1.3 s sample in a 4 s window "
                        "that frame sits 2.7 s into silence. Training never "
                        "does this: every synthetic clip sounds through the "
                        "whole window, so the last frame is always in-signal.")
    p.add_argument("--folder-peak", type=float, default=None, metavar="P",
                   help="One gain per run, putting the median peak at P. The "
                        "synthetic set's median is 0.496 and the Juno saw-bass "
                        "pack's is 0.145, and the estimator's input "
                        "normalisation is BatchNorm with affine=False, which "
                        "in eval subtracts the TRAINING mean rather than each "
                        "clip's own.")
    p.add_argument("--midi-offset", type=float, default=0.0, metavar="SEMIS",
                   help="Added to the filename's MIDI number before converting "
                        "to Hz. A patch does not necessarily sound at its "
                        "written pitch: ds_harmonic_probe measures the Juno "
                        "saw-bass pack with its partial at f0/2 LOUDER than the "
                        "one at the labelled f0 (+4.4 dB), and a harmonic "
                        "profile matching an ideal saw whose fundamental is "
                        "f0/2, so that pack sounds an OCTAVE DOWN -- a DCO set "
                        "to 16'. Pass -12 for it. Verify with the probe rather "
                        "than assuming; the slope is unaffected either way, but "
                        "the error size and which end is worst both are.")
    p.add_argument("--length", type=float, default=4.0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--csv", default=None, metavar="PATH")
    args = p.parse_args()

    files, midis = [], []
    for d in args.dirs:
        for f in audio_files(d):
            if args.match and not re.search(args.match, os.path.basename(f)):
                continue
            m = midi_of(f, args.midi_re)
            if m is None:
                continue
            files.append(f)
            midis.append(m)
    if not files:
        raise SystemExit(f"no files with a MIDI number matching {args.midi_re!r}")
    midis = np.array(midis, dtype=float) + args.midi_offset
    true_hz = 440.0 * 2.0 ** ((midis - 69.0) / 12.0)
    print(f"{len(files)} file(s), MIDI {midis.min():g}-{midis.max():g} "
          f"(offset {args.midi_offset:+g}), "
          f"{true_hz.min():.1f}-{true_hz.max():.1f} Hz")
    # FREQ_RANGE is (32, 2000): a target outside it cannot be predicted at all,
    # and would read as a large error that is really a bound.
    out = ((true_hz < 32.0) | (true_hz > 2000.0)).sum()
    if out:
        print(f"  WARNING: {out} file(s) lie outside harmor's FREQ_RANGE "
              f"(32-2000 Hz) and cannot be reached by any arm")

    loaded = [load_clip(f, args.sr, args.length) for f in files]
    x = np.stack([c[0] for c in loaded])
    peaks = np.array([c[2] for c in loaded])
    raws = np.array([c[3] for c in loaded])
    gain = 1.0
    if args.folder_peak is not None and np.median(peaks) > 0:
        gain = args.folder_peak / float(np.median(peaks))
        x = x * gain
    if args.trim_pad:
        keep = min(x.shape[1], int(np.ceil(raws.max() * args.sr / 256.0)) * 256)
        x = x[:, :keep]
    print(f"  window {x.shape[1] / args.sr:.2f}s  median peak "
          f"{np.median(peaks) * gain:.3f}  gain {gain:.2f}")
    x = torch.tensor(x, dtype=torch.float32)

    # Tertiles of the actual pitch set, so "low/mid/high" means something for
    # this pack rather than being fixed MIDI numbers.
    q1, q2 = np.quantile(midis, [1 / 3, 2 / 3])
    bands = [("low", midis <= q1), ("mid", (midis > q1) & (midis <= q2)),
             ("high", midis > q2)]

    rows, per_arm = [], {}
    for arm in args.arms:
        model, _cfg, note = load_model(os.path.join(args.root, arm),
                                       args.ckpt, args.device)
        if model is None:
            print(f"{arm:<24} skipped: {note}")
            continue
        f0s, mus = [], []
        with torch.no_grad():
            for i in range(0, x.shape[0], args.batch_size):
                a, b = predict(model, x[i:i + args.batch_size].to(args.device))
                f0s.append(a)
                mus.append(b)
        f0 = np.concatenate(f0s)
        mu = np.concatenate(mus)
        cents = 1200.0 * np.log2(np.maximum(f0, 1e-6) / true_hz)
        per_arm[arm] = (cents, mu, note, f0)
        for j, f in enumerate(files):
            rows.append((arm, os.path.basename(f), int(midis[j]),
                         f"{true_hz[j]:.3f}", f"{f0[j]:.3f}",
                         f"{cents[j]:.2f}", f"{mu[j]:.4f}"))

    if not per_arm:
        raise SystemExit("no arm produced results")

    print(f"\n=== PITCH, against the filename's MIDI number")
    print(f"{'arm':<24}{'ckpt':>12}{'slope':>8}{'med_cents':>11}{'|cents|':>9}"
          f"{'within50':>10}{'oct50':>8}{'med_MULT':>10}{'osc2_semis':>12}")
    for arm, (cents, mu, note, f0) in per_arm.items():
        s = summarise(cents, mu)
        print(f"{arm:<24}{note:>12}{slope_of(true_hz, f0):>8.2f}"
              f"{s['med']:>11.1f}{s['abs']:>9.1f}"
              f"{s['w50']:>9.0%}{s['o50']:>8.0%}{s['mult']:>10.2f}"
              f"{s['semis']:>12.1f}")

    print(f"\n=== BY PITCH BAND   median |cents|, tertiles of this set")
    print(f"{'arm':<24}" + "".join(f"{n + ' (n=' + str(int(m.sum())) + ')':>18}"
                                   for n, m in bands))
    for arm, (cents, _mu, _n, _f) in per_arm.items():
        print(f"{arm:<24}" + "".join(
            f"{np.median(np.abs(cents[m])):>18.1f}" if m.any() else f"{'-':>18}"
            for _n2, m in bands))

    print("\n  slope      octaves of PREDICTED pitch per octave of true pitch.\n"
          "             1.0 tracks, 0.0 is a constant regardless of input\n"
          "  med_cents  signed; a consistent sign is a bias, scatter is not\n"
          "  within50   inside a quarter tone of the true pitch\n"
          "  oct50      the same after folding out octave errors; the gap\n"
          "             between the two columns IS the octave-error rate\n"
          "  med_MULT   oscillator 2 sits at f0 * MULT, drawn on (1,8). A real\n"
          "             patch uses unison (1), a fifth (1.5) or an octave (2) --\n"
          "             0, 7 or 12 semitones. Anything else is an inharmonic\n"
          "             pair, and f0 conditioning would NOT fix it.")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["arm", "file", "midi", "true_hz", "pred_hz", "cents",
                        "mult"])
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
