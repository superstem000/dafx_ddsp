"""Resynthesise the same clips with several arms, to listen to them side by side.

    python scripts/ds_compare_audio.py --arms real_hybridx real_magx_halfw --n 5
    python scripts/ds_compare_audio.py --arms synth_magx synth_hybridx --domain id

Every audio number in this project is a mean over 512 clips of a distance whose
compression is itself the thing under dispute. At some point that has to be
checked against what the models actually sound like, and on the real branch
there is a specific reason to: hybridx wins above the crossover and magx_halfw
below it, and the crossover itself moves with a convention -- 70 dB with the
clamp after the mel filterbank, between 70 and 80 with it before. Under the
psychoacoustic mask, which has no floor to choose, magx_halfw leads by 3.9 se,
but that is 4.1% of the metric's range against unrelated audio. Statistically
clear and perceptually modest is exactly the regime where the number should be
checked by ear before it goes in a paper as a perceptual claim.

Per clip it writes the target, one resynthesis per arm, and a spectrogram panel
for each on ONE colour scale pinned to the target's peak, then tars the lot --
this runs over ssh and the point is to get it to a machine with speakers.

THE SAME CLIPS FOR EVERY ARM. load_arm reproduces train.py's RNG order exactly
(seed, construct EstimatorSynth, then set up the datamodule), so the validation
split comes out identical for each arm and the first batch is the same audio.
Without that the comparison would be between different clips, which is the
mistake this whole script exists to avoid making by ear.

Levels are NOT normalised. An arm that gets the overall gain wrong should sound
like it does.

The per-clip table gives g0.3 and db70 for each arm -- the two rungs the ladder
disagrees across -- so the clips where the arms differ most can be found rather
than guessed at. Read down an arm's column; the two metrics have unrelated
scales.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "external", "diffsynth")
sys.path.insert(0, DS)
sys.path.insert(0, HERE)

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import soundfile as sf                                   # noqa: E402
import torch                                             # noqa: E402
import torch.nn.functional as F                          # noqa: E402
from torch.utils.data import DataLoader                  # noqa: E402

import ds_param_breakdown as pb                          # noqa: E402
import ds_mfcc_check as mc                               # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--arms", nargs="+", required=True,
                   help="Run directory names, e.g. real_hybridx real_magx_halfw")
    p.add_argument("--ckpt", default="latest.ckpt")
    p.add_argument("--domain", default="ood", choices=("id", "ood"))
    p.add_argument("--n", type=int, default=5, help="How many clips")
    p.add_argument(
        "--skip", type=int, default=0, metavar="N",
        help="Skip the first N clips of the split and take the next --n. A "
             "fixed offset rather than a random draw, so every arm still gets "
             "the SAME clips -- which is the one property this comparison "
             "cannot lose. --skip 5, 10, 15 walk through the split five at a "
             "time.")
    p.add_argument("--out", default="compare_audio")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--no-tar", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    metrics = {"g0.3": mc.make_mfcc(args.device, window="hann", log="pow",
                                    gamma=0.3, mel_norm="slaney",
                                    mel_scale="slaney"),
               "db70": mc.make_mfcc(args.device, window="hann", log="db",
                                    top_db=70.0, mel_norm="slaney",
                                    mel_scale="slaney")}

    target = None
    resyn = {}
    scores = {}
    for arm in args.arms:
        d = os.path.join(args.root, arm)
        model, cfg, dm, note = pb.load_arm(d, args.ckpt, args.device, args.n)
        if model is None:
            print(f"{arm:<20} skipped: {note}")
            continue
        if args.domain == "id":
            vset, how = pb.val_split(d, dm)
        else:
            vset, how = dm.ood_datasets["valid"], "ood split reproduced"
        if vset is None:
            print(f"{arm:<20} skipped: {how}")
            continue
        if args.skip:
            from torch.utils.data import Subset
            hi = min(args.skip + args.n, len(vset))
            if args.skip >= len(vset):
                raise SystemExit(f"--skip {args.skip} is past the end of the "
                                 f"{len(vset)}-clip split")
            vset = Subset(vset, range(args.skip, hi))
        loader = DataLoader(vset, batch_size=args.n, num_workers=0)
        batch = next(iter(loader))
        batch = {k: (v.to(args.device) if torch.is_tensor(v) else
                     {kk: vv.to(args.device) for kk, vv in v.items()})
                 for k, v in batch.items()}
        with torch.no_grad():
            out, _o = model(batch)
        if target is None:
            target = batch["audio"].cpu()
        resyn[arm] = out.cpu()
        scores[arm] = {
            k: [F.l1_loss(fn(target[i:i + 1]), fn(resyn[arm][i:i + 1])).item()
                for i in range(target.shape[0])]
            for k, fn in metrics.items()}
        print(f"{arm:<20} {note:<12} {args.n} clips -- {how}")

    if target is None:
        raise SystemExit("no arm produced audio")
    arms = list(resyn)
    sr = int(cfg.data.sample_rate)

    print(f"\nper-clip distances (read DOWN an arm's column; the two metrics "
          f"have unrelated scales)")
    head = "".join(f"{a + '/' + k:>26}" for a in arms for k in metrics)
    print(f"{'clip':>6}{head}")
    for i in range(target.shape[0]):
        row = "".join(f"{scores[a][k][i]:>26.4f}" for a in arms for k in metrics)
        print(f"{args.skip + i:>6}{row}")

    win = torch.hann_window(1024)
    for i in range(target.shape[0]):
        # Numbered by position in the split, not within this batch, so two
        # runs at different --skip cannot produce colliding filenames.
        c = args.skip + i
        sf.write(os.path.join(args.out, f"clip{c}_target.wav"),
                 target[i].numpy(), sr)
        for a in arms:
            sf.write(os.path.join(args.out, f"clip{c}_{a}.wav"),
                     resyn[a][i].numpy(), sr)
        if args.no_png:
            continue
        panels = [("target", target[i])] + [(a, resyn[a][i]) for a in arms]
        # Fixed colour scale from the TARGET's peak, 100 dB down, for every
        # panel. Autoscaling each would hide exactly the level and decay
        # differences worth seeing.
        tm = torch.stft(target[i], 1024, hop_length=256, window=win,
                        center=True, return_complex=True).abs()
        ref = float(20.0 * torch.log10(tm.max().clamp(min=1e-12)))
        fig, axes = plt.subplots(1, len(panels),
                                 figsize=(4.2 * len(panels), 3.0), squeeze=False)
        for ax, (tag, y) in zip(axes.ravel(), panels):
            m = torch.stft(y, 1024, hop_length=256, window=win, center=True,
                           return_complex=True).abs()
            ax.imshow((20.0 * torch.log10(m.clamp(min=1e-12))).numpy(),
                      origin="lower", aspect="auto", cmap="magma",
                      vmin=ref - 100.0, vmax=ref,
                      extent=[0, y.shape[-1] / sr, 0, sr / 2000.0])
            ax.set_title(tag, fontsize=9)
            ax.set_ylabel("kHz", fontsize=7)
            ax.tick_params(labelsize=6)
        fig.suptitle(f"clip {c}   (colour scale: target peak-100 dB .. peak)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"clip{c}.png"), dpi=110)
        plt.close(fig)

    print(f"\nwrote {args.out}: target + {len(arms)} arm(s) per clip, "
          f"levels unnormalised")
    if not args.no_tar:
        base = os.path.basename(os.path.normpath(args.out))
        tar = os.path.normpath(args.out) + ".tar.gz"
        subprocess.run(["tar", "czf", tar, "-C",
                        os.path.dirname(os.path.abspath(args.out)) or ".",
                        base], check=True)
        print(f"bundled: {tar}  ({os.path.getsize(tar) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
