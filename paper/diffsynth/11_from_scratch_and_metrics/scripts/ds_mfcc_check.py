"""Recompute the MFCC distance with a window, and with library conventions.

    python scripts/ds_mfcc_check.py --only '^(synth|real)_'
    python scripts/ds_mfcc_check.py --only '^synth_magx' --ckpt 'ep*.ckpt'

WHY. model.py's Mfcc reaches torch.stft through MelSpec.forward, which calls
spectrogram(audio, n_fft, hop_length, power, center) -- five positional
arguments, so `window` keeps its None default and the transform runs with a
RECTANGULAR window. compute_lsd, two functions away, passes a Hann window
explicitly. So the two audio metrics do not share an analysis window, and the
MFCC one has full spectral leakage smearing every strong partial into its
neighbouring bands. That is a defect rather than a convention, and it weakens
the argument MFCC is being used for -- leakage is not perceptual.

Three variants are reported so the effect is visible rather than asserted:

  as_logged   exactly what val_id/mfcc contains: rectangular window, HTK mel
              scale, unnormalised filterbank, natural log with eps 1e-6, no dB
              conversion and no top_db floor, 40 mels -> 20 DCT coefficients
  hann        the same with a Hann window, changing one thing
  standard    Hann, Slaney mel scale and Slaney filterbank normalisation, dB
              (10*log10) with an 80 dB floor -- torchaudio/librosa conventions.
              Expect roughly 4.343x the natural-log values from the log base
              alone, so read it against itself, not against the other columns.

The ORDERING across arms is what matters. If it survives all three the result
does not depend on the metric's implementation details; if it moves, that is
worth knowing before the number goes in a table.

Only checkpointed epochs can be recomputed -- MFCC was never logged any other
way. --ckpt takes a glob, so 'ep*.ckpt' gives the 50-epoch trajectory for runs
that had trainer.checkpoint_every_n_epochs set, and the default latest.ckpt
gives the final epoch for everything.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "external", "diffsynth")
sys.path.insert(0, DS)
sys.path.insert(0, HERE)

import torch                                            # noqa: E402
import torch.nn.functional as F                         # noqa: E402
from torch.utils.data import DataLoader                 # noqa: E402
from torchaudio.transforms import MelScale              # noqa: E402
from torchaudio.functional import create_dct            # noqa: E402

import ds_param_breakdown as pb                         # noqa: E402


def make_mfcc(device, window="rect", log="nat", top_db=None,
              mel_norm=None, mel_scale="htk", sr=16000, n_fft=1024, hop=256,
              n_mels=40, n_mfcc=20, f_min=40.0, f_max=7600.0):
    """A callable audio -> MFCC, mirroring spectral.py but with the knobs open."""
    win = None if window == "rect" else torch.hann_window(n_fft, device=device)
    ms = MelScale(n_mels, sr, f_min, f_max, n_fft // 2 + 1, mel_norm,
                  mel_scale).to(device)
    dct = create_dct(n_mfcc, n_mels, "ortho").to(device)

    def f(audio):
        # MelSpec.pad_end: zero-pad up to a whole number of hops.
        rem = (audio.shape[-1] - n_fft) % hop
        if rem:
            audio = F.pad(audio, (0, hop - rem), "constant")
        st = torch.stft(audio, n_fft, hop_length=hop, window=win,
                        center=False, return_complex=True)
        power = st.real ** 2 + st.imag ** 2
        mel = ms(power)
        if log == "nat":
            lm = torch.log(mel + 1e-6)
        else:
            lm = 10.0 * torch.log10(mel + 1e-10)
            if top_db is not None:
                lm = torch.maximum(lm, lm.amax(dim=(-2, -1), keepdim=True) - top_db)
        return torch.matmul(lm.transpose(1, 2), dct).transpose(1, 2)

    return f


VARIANTS = (
    ("as_logged", dict(window="rect", log="nat")),
    ("hann", dict(window="hann", log="nat")),
    ("standard", dict(window="hann", log="db", top_db=80.0,
                      mel_norm="slaney", mel_scale="slaney")),
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--only", default=None, metavar="REGEX")
    p.add_argument("--ckpt", default="latest.ckpt",
                   help="Checkpoint file name or glob within the run's "
                        "checkpoints/ directory, e.g. 'ep*.ckpt'")
    p.add_argument("--domain", default="id", choices=("id", "ood"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    fns = {n: make_mfcc(args.device, **kw) for n, kw in VARIANTS}

    runs = [d for d in sorted(glob.glob(os.path.join(args.root, "*")))
            if os.path.isdir(d) and (not args.only or re.search(args.only, Path(d).name))]
    rows = []
    for d in runs:
        name = Path(d).name
        cdir = os.path.join(d, "tb_logs", "checkpoints")
        for ck in sorted(glob.glob(os.path.join(cdir, args.ckpt))):
            model, cfg, dm, note = pb.load_arm(d, os.path.basename(ck),
                                               args.device, args.batch_size)
            if model is None:
                print(f"{name:<20} {os.path.basename(ck):<14} skipped: {note}")
                continue
            if args.domain == "id":
                vset, how = pb.val_split(d, dm)
            else:
                vset, how = dm.ood_datasets["valid"], "ood split reproduced"
            if vset is None:
                print(f"{name:<20} skipped: {how}")
                continue
            loader = DataLoader(vset, batch_size=args.batch_size, num_workers=0)
            acc, n = {k: 0.0 for k in fns}, 0
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= args.batches:
                        break
                    batch = {k: (v.to(args.device) if torch.is_tensor(v) else
                                 {kk: vv.to(args.device) for kk, vv in v.items()})
                             for k, v in batch.items()}
                    resyn, _out = model(batch)
                    tgt = batch["audio"]
                    for k, fn in fns.items():
                        acc[k] += F.l1_loss(fn(tgt), fn(resyn)).item()
                    n += 1
            if not n:
                continue
            ep = int(note.split()[-1]) if note.split()[-1].isdigit() else -1
            rows.append((name, ep, {k: v / n for k, v in acc.items()}))
            print(f"{name:<20} {note:<12} {n} batches -- {how}")

    if not rows:
        return
    w = max(20, max(len(r[0]) for r in rows) + 2)
    print(f"\n=== MFCC on the {args.domain} validation split, "
          f"{args.batches * args.batch_size} clips")
    print("'standard' uses dB rather than natural log, so it runs ~4.34x the "
          "others by\nconvention alone -- compare each column down, not "
          "across.\n")
    print(f"{'run':<{w}}{'epoch':>7}" + "".join(f"{n:>12}" for n, _ in VARIANTS))
    for name, ep, v in rows:
        print(f"{name:<{w}}{ep:>7}" + "".join(f"{v[n]:>12.4f}" for n, _ in VARIANTS))


if __name__ == "__main__":
    main()
