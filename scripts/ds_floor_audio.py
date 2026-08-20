"""Render what a dB floor throws away, so the ladder can be listened to.

    python scripts/ds_floor_audio.py --n 5 --out ~/floor_demo
    python scripts/ds_floor_audio.py --family synth_lead --top-db 60 40

WHY. ds_mfcc_check's db80/db60/db40/db20 columns clamp everything more than N
dB below a clip's peak, and on the real branch the arm ordering flips somewhere
in that ladder: hybridx wins while content in the 70-80 dB shell still counts,
magx wins as soon as it does not. That is an argument about audibility, and an
argument about audibility should be settled by listening rather than by
asserting that Stevens' law or librosa's default makes 60 dB reasonable.

ds_masking.py now answers the same question from psychoacoustics -- a per-bin
masking threshold with no floor to choose. This stays because a threshold model
is still a model, and being able to HEAR what a floor discards is the check
that does not depend on believing MPEG-1.

WHAT IS WRITTEN, per clip and per floor:

    <stem>_orig.wav         the target, untouched
    <stem>_keep<N>.wav      only the bins WITHIN N dB of the peak
    <stem>_drop<N>.wav      only the bins BELOW that -- what the floor discards
    <stem>.png              every one of the above as a spectrogram, on ONE
                            fixed dB colour scale so the panels are comparable

and the whole directory as floor_demo.tar.gz, since this runs over ssh and the
point is to get it onto a machine with speakers.

The figure's colour scale is pinned to [peak-100 dB, peak] for every panel. A
per-panel autoscale would renormalise drop<N> until the discarded content
looked as loud as the original, which is the opposite of what the figure is
for.

keep+drop sums back to orig sample-for-sample, since the mask is a partition of
the same complex STFT and the phase is the original throughout. So drop<N> is
literally the part of the signal the metric stops scoring, at its natural level
rather than normalised -- play it at the same gain as orig.

The question each pair answers:

    keep60 vs orig    does removing everything 60 dB down change what you hear
    drop60 alone      is that removed part audible on its own

If keep60 is indistinguishable and drop60 is silence-with-a-hiss, then the
metric's disagreement between magx and hybridx lives entirely in content below
audibility, and db60/db40 are the honest rungs to quote.

--mask adds keepmask/dropmask, gated on the MPEG-1 threshold instead of an
offset from the peak, and it is read DIFFERENTLY from the dB pairs. Masking
claims the removed content is inaudible IN THE PRESENCE OF the rest of the
signal, which is the condition keepmask reproduces and dropmask destroys. So
keepmask vs orig is the test; dropmask WILL be audible on its own, and that is
the definition of masking rather than a failure of it. A dB floor makes the
weaker claim that what it discards is quiet in absolute terms, which is why
drop60 being near-silent means something and dropmask being audible does not.

GATING, not clamping. The metric clamps to a floor value; doing that to audio
would BOOST the quiet bins up to the floor, which is not a thing to listen to.
Zeroing them asks the question the clamp is a proxy for: does this content
matter at all.

The peak is over the whole clip, matching the metric (and librosa's
power_to_db), not per frame. On a note with a decay that means later frames lose
proportionally more, which is exactly the effect worth hearing.

center=True here where the metric uses center=False -- the metric never inverts,
and centring avoids edge artefacts in the resynthesis that would be mistaken for
the gate's doing.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

import ds_masking as dm                                  # noqa: E402


def masking_floor(mag, win):
    """Per-bin, per-frame masking threshold for one clip, in the same dB units
    as 20*log10(mag).

    Computed here rather than read from ds_masking's cache. The cache is keyed
    by a content hash of DATASET-loaded audio, and this script reads wav files
    off disk -- so a lookup would miss. Ten clips is 2470 frames of numpy,
    which costs nothing, and it keeps the demo independent of whether a cache
    has been dumped.

    The frames are the ones being gated, which here are center=True (the metric
    uses center=False). Centring is right for a resynthesis -- it avoids edge
    artefacts that would be mistaken for the gate's doing -- and the masking
    model does not care how the frames were placed.
    """
    freqs = np.fft.rfftfreq(dm.N_FFT, 1.0 / dm.SR)
    bin_hz = dm.SR / dm.N_FFT
    zb, ath_b = dm.bark(freqs), dm.ath(freqs)
    bands = dm.build_bands(freqs)
    P = dm.PN + 20.0 * np.log10(
        np.maximum(2.0 * mag.numpy() / float(win.sum()), 1e-30))
    T = np.empty_like(P)
    for t in range(P.shape[1]):
        lev, zm, ton, _raw = dm.frame_maskers(
            P[:, t], freqs, zb, ath_b, bands, "hz", "all", bin_hz)
        T[:, t], _tm = dm.global_threshold(lev, zm, ton, zb, ath_b)
    # Back from SPL to the 20*log10(mag) scale the caller compares against.
    return torch.from_numpy(P - T).float()


def main() -> None:
    # Line-buffer stdout. Lightning's "Seed set to 0" goes to stderr and shows
    # up at once, while every progress line here is a print() -- so piped into
    # tee the two separate and the log looks stalled for minutes at a time with
    # checkpoints already finished. Same fix as gpu_queue.py needed.
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--audio-dir",
                   default="external/diffsynth/data/nsynth-train/audio",
                   help="Directory of .wav files to sample from")
    p.add_argument("--family", default=None, metavar="PREFIX",
                   help="Restrict to one NSynth family, e.g. synth_lead")
    p.add_argument("--n", type=int, default=5, help="How many clips")
    p.add_argument("--top-db", type=float, nargs="*",
                   default=[80.0, 60.0, 40.0, 20.0],
                   help="Pass with no values for none, e.g. --top-db --mask "
                        "to hear only the psychoacoustic gate.")
    p.add_argument(
        "--mask", action="store_true",
        help="Also gate on the MPEG-1 masking threshold instead of a constant "
             "offset from the peak, writing _keepmask/_dropmask. THIS is the "
             "check that does not require believing the model: keepmask should "
             "be indistinguishable from orig. dropmask is a different matter "
             "-- see the note printed at the end.")
    p.add_argument("--out", default="floor_demo")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-png", action="store_true", help="Skip the figures")
    p.add_argument("--no-tar", action="store_true",
                   help="Skip bundling the output directory")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if args.family:
        files = [f for f in files
                 if os.path.basename(f).startswith(args.family)]
    if not files:
        raise SystemExit(f"no wavs under {args.audio_dir}"
                         + (f" matching {args.family}*" if args.family else ""))
    random.Random(args.seed).shuffle(files)
    files = files[: args.n]

    os.makedirs(args.out, exist_ok=True)
    win = torch.hann_window(args.n_fft)
    print(f"{len(files)} clips -> {args.out}\n")
    print(f"{'clip':<34}{'gate':>7}{'bins kept':>11}{'energy kept':>13}")

    for path in files:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        x = torch.from_numpy(audio)
        stem = os.path.splitext(os.path.basename(path))[0]
        sf.write(os.path.join(args.out, f"{stem}_orig.wav"), audio, sr)

        st = torch.stft(x, args.n_fft, hop_length=args.hop, window=win,
                        center=True, return_complex=True)
        mag = st.abs()
        panels = [("orig", mag)]
        # N dB is on POWER, as in the metric's 10*log10, so the amplitude
        # threshold is 10^(-N/20) of the peak amplitude.
        peak = mag.max()
        total = (mag ** 2).sum()
        gates = [(f"{db:g}", mag >= peak * (10.0 ** (-db / 20.0)))
                 for db in args.top_db]
        if args.mask:
            # Above the threshold by any margin is kept. Per bin AND per frame,
            # so unlike the dB floors this one moves with the signal -- a
            # sustain that decays does not lose proportionally more, it loses
            # whatever its own maskers stop covering.
            gates.append(("mask", masking_floor(mag, win) >= 0.0))
        for tag, keep in gates:
            kept_st = torch.where(keep, st, torch.zeros_like(st))
            drop_st = st - kept_st
            for which, spec in (("keep", kept_st), ("drop", drop_st)):
                y = torch.istft(spec, args.n_fft, hop_length=args.hop,
                                window=win, center=True, length=x.shape[-1])
                sf.write(os.path.join(args.out, f"{stem}_{which}{tag}.wav"),
                         y.numpy(), sr)
            panels.append((f"keep{tag}", kept_st.abs()))
            panels.append((f"drop{tag}", drop_st.abs()))
            frac_bins = keep.float().mean().item()
            frac_energy = ((mag * keep) ** 2).sum().item() / max(total.item(), 1e-30)
            print(f"{stem:<34}{tag:>7}{100 * frac_bins:>10.1f}%"
                  f"{100 * frac_energy:>12.4f}%")

        if not args.no_png:
            # One fixed colour scale for every panel, spanning 100 dB down from
            # the clip's peak. Autoscaling each panel would renormalise drop<N>
            # until the discarded content looked as loud as the original.
            ref = 20.0 * torch.log10(mag.max().clamp(min=1e-12))
            ncol = 3
            nrow = (len(panels) + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow),
                                     squeeze=False)
            for ax, (tag, m) in zip(axes.ravel(), panels):
                d = (20.0 * torch.log10(m.clamp(min=1e-12))).numpy()
                ax.imshow(d, origin="lower", aspect="auto", cmap="magma",
                          vmin=float(ref) - 100.0, vmax=float(ref),
                          extent=[0, x.shape[-1] / sr, 0, sr / 2000.0])
                ax.set_title(tag, fontsize=9)
                ax.set_ylabel("kHz", fontsize=7)
                ax.tick_params(labelsize=6)
            for ax in axes.ravel()[len(panels):]:
                ax.axis("off")
            fig.suptitle(f"{stem}   (colour scale: peak-100 dB .. peak)",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"{stem}.png"), dpi=110)
            plt.close(fig)

    print(f"\nkeep<N> + drop<N> reconstructs orig exactly -- same complex STFT, "
          f"same phase.\nPlay drop<N> at the SAME gain as orig; it is not "
          f"normalised, and that is the point.")
    if args.mask:
        print(
            "\nHOW TO READ keepmask/dropmask, because one of them is a trap.\n"
            "\n"
            "  keepmask vs orig  IS the test. Masking says the removed content\n"
            "                    is inaudible IN THE PRESENCE OF the rest of\n"
            "                    the signal, and that is exactly the condition\n"
            "                    keepmask reproduces. If these are\n"
            "                    indistinguishable, the threshold is doing what\n"
            "                    it claims and the evaluation floor is earned.\n"
            "\n"
            "  dropmask alone    IS NOT a test, and it WILL be audible. Take\n"
            "                    the masker away and what it was masking\n"
            "                    becomes audible again -- that is the\n"
            "                    definition of masking, not a failure of it.\n"
            "                    Hearing dropmask proves nothing either way.\n"
            "\n"
            "This is where the psychoacoustic gate differs from the dB floors.\n"
            "drop60 being near-silent is evidence a 60 dB floor discards little;\n"
            "dropmask can be perfectly audible on its own while keepmask is\n"
            "still identical to orig, and both statements are consistent.")

    if not args.no_tar:
        base = os.path.basename(os.path.normpath(args.out))
        tar = os.path.normpath(args.out) + ".tar.gz"
        subprocess.run(["tar", "czf", tar, "-C",
                        os.path.dirname(os.path.abspath(args.out)) or ".",
                        base], check=True)
        mb = os.path.getsize(tar) / 1024 / 1024
        print(f"\nbundled: {tar}  ({mb:.1f} MB)")
        print(f"  scp <user>@<host>:{os.path.abspath(tar)} .")


if __name__ == "__main__":
    main()
