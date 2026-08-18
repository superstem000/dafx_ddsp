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

Then a POWER-CEPSTRUM ladder, g1 / g0.6 / g0.3 / g0.15: `standard` with its dB
step replaced by mel^gamma and nothing else changed, so moving along it changes
the metric's compression and only that.

  Every audio metric here measures in the log domain -- MFCC logs mel-band
  totals, LSD logs per bin -- and a log-trained model is optimised for exactly
  that domain, so ranking losses by them is close to circular. g1 is a linear
  metric and the linear-trained arms win it by construction; as_logged is the
  log one and they lose it. So a crossover exists, and where it sits is the
  question. Stevens' law puts loudness at I^0.3, which makes g0.3 the
  perceptually calibrated rung -- pure linear and pure log are both wrong, on
  opposite sides, and MFCC and LSD sit far past perception. If the crossover is
  below 0.3, the linear loss wins where hearing actually is.

  Quoting a stated ladder with a reference fixed by psychoacoustics is what
  separates this from picking whichever metric happens to be flattering.

The ORDERING across arms is what matters. If it survives all three log variants
the result does not depend on the metric's implementation details; if it moves,
that is worth knowing before the number goes in a table. Read every column
DOWN -- dB is ~4.34x natural log by base alone, and each gamma is a different
power of the same numbers, so nothing compares across columns.

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
from diffsynth.spectral import compute_lsd              # noqa: E402

import ds_param_breakdown as pb                         # noqa: E402


def make_mfcc(device, window="rect", log="nat", top_db=None, gamma=0.3,
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
        elif log == "pow":
            # Stevens' law: loudness ~ I^0.3 with I the intensity. `mel` is
            # already a POWER mel spectrogram here, so the exponent applies
            # directly with no conversion. gamma=1 is a linear metric, gamma->0
            # approaches the log one, and 0.3 is where perception actually
            # sits -- between them, which is the whole point: MFCC and LSD both
            # measure at gamma->0, far past perception, and that is the domain
            # a log-trained model is optimised for.
            lm = (mel + 1e-12) ** gamma
        else:
            lm = 10.0 * torch.log10(mel + 1e-10)
            if top_db is not None:
                lm = torch.maximum(lm, lm.amax(dim=(-2, -1), keepdim=True) - top_db)
        return torch.matmul(lm.transpose(1, 2), dct).transpose(1, 2)

    return f


_FIXED = (
    ("as_logged", dict(window="rect", log="nat")),
    ("hann", dict(window="hann", log="nat")),
    ("standard", dict(window="hann", log="db", top_db=80.0,
                      mel_norm="slaney", mel_scale="slaney")),
)


def build_variants(gammas):
    """The three log variants, plus one power-cepstrum column per gamma.

    The gamma columns are `standard` with its dB step replaced by mel^gamma and
    nothing else changed -- same Hann window, same Slaney mel scale and
    filterbank normalisation -- so moving down the gamma list changes the
    metric's COMPRESSION and only that.

    Why this is worth having. Every audio metric in this project measures in
    the log domain: MFCC logs mel-band totals, LSD logs per bin. A log-trained
    model is optimised for that domain, so ranking losses by them is close to
    circular. gamma=1 is a linear metric and the linear-trained arms must win
    it by construction; gamma->0 is the log metric and they must lose. There is
    therefore a crossover, and where it sits is the actual question.

    Stevens' law puts loudness at I^0.3, so gamma=0.3 is the perceptually
    calibrated rung -- both pure linear and pure log are wrong, on opposite
    sides, and MFCC/LSD sit far past perception at gamma->0. If the crossover
    is below 0.3 the linear loss wins where hearing actually is; if above, it
    does not. Reading the answer off a stated ladder with a reference point
    fixed by psychoacoustics is what keeps this from being metric-shopping.
    """
    return _FIXED + tuple(
        (f"g{g:g}", dict(window="hann", log="pow", gamma=g,
                         mel_norm="slaney", mel_scale="slaney"))
        for g in gammas
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--only", default=None, metavar="REGEX")
    p.add_argument("--ckpt", default="latest.ckpt",
                   help="Checkpoint file name or glob within the run's "
                        "checkpoints/ directory, e.g. 'ep*.ckpt'")
    p.add_argument("--domain", default="id", choices=("id", "ood"))
    p.add_argument(
        "--gamma", type=float, nargs="*", default=[1.0, 0.6, 0.3, 0.15],
        metavar="G",
        help="Power-cepstrum exponents to add as extra columns: mel^gamma in "
             "place of the log, everything else held at the `standard` "
             "conventions. 1.0 is a linear metric, ->0 approaches the log one, "
             "and 0.3 is Stevens' law (loudness ~ I^0.3) -- the perceptually "
             "calibrated rung, which MFCC and LSD both sit far past. The "
             "default ladder brackets it on both sides so the crossover is "
             "visible rather than assumed. Pass no values to skip them.",
    )
    p.add_argument(
        "--family", default=None, metavar="PREFIX",
        help="Score against one NSynth instrument family instead of the ood "
             "valid split -- e.g. --family synth_lead. NSynth filenames lead "
             "with the family (bass_synthetic_..., synth_lead_synthetic_...), "
             "so this is a basename prefix match over the WHOLE ood pool.\n\n"
             "Why the whole pool and not the valid split: a family is a few "
             "percent of NSynth, so intersecting it with a 2000-file valid "
             "split leaves ~47 clips. That is only sound for arms which never "
             "trained on ood at all -- the synth_* branch, which trains on "
             "harmor throughout. For real_* the whole pool IS its training "
             "set, so this refuses to run on them unless --allow-seen.\n\n"
             "The point is a model-mismatch ladder. harmor is 2 oscillators "
             "morphing saw<->square through a 2nd-order lowpass, which is what "
             "a synth lead is, and is nothing like a mallet or a plucked "
             "string. Scoring one arm on harmor (no mismatch), on synth_lead "
             "(mild) and on all of NSynth (severe) separates 'this loss fits "
             "better' from 'this loss distributes irreducible bias better'.",
    )
    p.add_argument(
        "--allow-seen", action="store_true",
        help="Score --family on real_* arms anyway. They trained on 16000 of "
             "the 20000-file ood pool, so most of the family is in their "
             "training set and the number is not held out.",
    )
    p.add_argument(
        "--crepe", action="store_true",
        help="Add an L1 distance between CREPE embeddings. Needs torchcrepe "
             "(pip install torchcrepe; it bundles its own weights).\n\n"
             "The point is independence. Every other column here -- MFCC in any "
             "variant, LSD, the gamma ladder -- is a spectral distance under "
             "some compression, so a reviewer can answer the gamma result with "
             "'you changed the compression to suit yourself'. A learned "
             "embedding makes no compression choice at all, so if it agrees "
             "with the gamma ladder that is a second, structurally different "
             "axis rather than the same one re-scaled.\n\n"
             "Read it for what it is, though: CREPE is a PITCH model, so this "
             "mostly measures whether the resynthesis has the right pitch, not "
             "whether it sounds the same. It is a check on the gamma result, "
             "not a general perceptual metric. OpenL3 would be that, and needs "
             "downloaded weights.\n\n"
             "Slow on CPU -- ~124 frames per 4 s clip through the model, per "
             "arm, per checkpoint. Use --device cuda, or cut --batches.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    variants = build_variants(args.gamma)
    fns = {n: make_mfcc(args.device, **kw) for n, kw in variants}

    # Imported only when asked for, so torchcrepe stays an optional dependency
    # and every existing invocation keeps working without it.
    crepe = None
    if args.crepe:
        from diffsynth.perceptual.crepe import CREPELoss
        crepe = CREPELoss().to(args.device).eval()
    # Column order after the cepstral ones: LSD first since it is the metric
    # the paper already quotes, then the embedding distance.
    extra = ["lsd"] + (["crepe"] if crepe is not None else [])

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
            if args.family:
                if name.startswith("real_") and not args.allow_seen:
                    print(f"{name:<20} skipped: trains on ood, the family is "
                          f"not held out for it (--allow-seen to override)")
                    continue
                from diffsynth.data import WaveParamDataset
                # params=False: the ood side has no ground-truth parameters,
                # which is how IdOodDataModule builds it too.
                ds = WaveParamDataset(cfg.data.ood_dir, cfg.data.sample_rate,
                                      cfg.data.length, False, False)
                ds.raw_files = [f for f in ds.raw_files
                                if os.path.basename(f).startswith(args.family)]
                if not ds.raw_files:
                    print(f"{name:<20} skipped: no files matching "
                          f"'{args.family}*' under {cfg.data.ood_dir}/audio")
                    continue
                vset = ds
                how = f"{len(ds.raw_files)} {args.family} files, whole ood pool"
            elif args.domain == "id":
                vset, how = pb.val_split(d, dm)
            else:
                vset, how = dm.ood_datasets["valid"], "ood split reproduced"
            if vset is None:
                print(f"{name:<20} skipped: {how}")
                continue
            loader = DataLoader(vset, batch_size=args.batch_size, num_workers=0)
            # LSD alongside, because it is the other audio number the paper
            # quotes and it disagrees with MFCC often enough that reading one
            # without the other has already been misleading once.
            acc, n = {k: 0.0 for k in list(fns) + extra}, 0
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
                    acc["lsd"] += float(compute_lsd(tgt, resyn))
                    if crepe is not None:
                        acc["crepe"] += float(
                            crepe.perceptual_loss(tgt, resyn))
                    n += 1
            if not n:
                continue
            ep = int(note.split()[-1]) if note.split()[-1].isdigit() else -1
            rows.append((name, ep, {k: v / n for k, v in acc.items()}))
            print(f"{name:<20} {note:<12} {n} batches -- {how}")

    if not rows:
        return
    w = max(20, max(len(r[0]) for r in rows) + 2)
    where = (f"the {args.family} family" if args.family
             else f"the {args.domain} validation split")
    print(f"\n=== MFCC on {where}, up to "
          f"{args.batches * args.batch_size} clips")
    print("Every column has its own scale -- dB is ~4.34x natural log by base "
          "alone, and each\ngamma is a different power of the same numbers. "
          "Read each column DOWN, never across.\nWhat carries information is "
          "the ORDER of the arms within a column, and how that order\nchanges "
          "as the metric's compression moves from g1 (linear) toward "
          "as_logged (log).\ng0.3 is Stevens' law and is the rung to quote.\n")
    print(f"{'run':<{w}}{'epoch':>7}"
          + "".join(f"{n:>12}" for n, _ in variants)
          + "".join(f"{k:>12}" for k in extra))
    for name, ep, v in rows:
        print(f"{name:<{w}}{ep:>7}"
              + "".join(f"{v[n]:>12.4f}" for n, _ in variants)
              + "".join(f"{v[k]:>12.4f}" for k in extra))


if __name__ == "__main__":
    main()
