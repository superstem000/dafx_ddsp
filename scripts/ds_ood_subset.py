"""Which arm wins on OOD depends on WHICH out-of-domain audio you score against.

    python scripts/ds_ood_subset.py --arms synth_magx_halfw synth_hybridx synth_logx_halfw
    python scripts/ds_ood_subset.py --arms ... --group-by source
    python scripts/ds_ood_subset.py --arms ... --only synth_lead

val_ood/mfcc is a mean over a random NSynth subsample containing acoustic
guitars, bowed strings, brass and organs alongside synthesizer leads. An
estimator for a harmonic-oscillator-plus-filter synth cannot represent most of
that at all, so a large part of that mean is a distance between the target and
whatever the model degrades to -- and an arm can win it by degrading more
gracefully rather than by fitting anything.

The hypothesis this exists to test is that on the SYNTHETIC portion alone --
where the target is something the synth could in principle produce -- the
ranking changes, and specifically that the linear arm beats the compressed ones
there even where it loses on the full set. That is a claim about the evaluation
set rather than about the losses, and it is cheap to check because it needs no
retraining: the arms already exist and NSynth encodes the category in the
filename.

NSynth files are <family>_<source>_<pitch>-<velocity>-<id>.wav, so
  family: bass, brass, flute, guitar, keyboard, mallet, organ, reed, string,
          synth_lead, vocal
  source: acoustic, electronic, synthetic
"synth" is ambiguous between the two -- synth_lead is an instrument family of
~thousands of files, synthetic is a source spanning several families -- so both
groupings are reported and neither is chosen here.

THE SAME CLIPS FOR EVERY ARM. load_arm reproduces train.py's RNG order exactly,
so the ood subsample and its splits come out identical per arm. Without that
the comparison would be between different clips, and the whole point is that
WHICH clips is the variable.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "external", "diffsynth"))
sys.path.insert(0, HERE)

import torch                                             # noqa: E402
import torch.nn.functional as F                          # noqa: E402
from torch.utils.data import DataLoader, Subset          # noqa: E402

import ds_param_breakdown as pb                          # noqa: E402
import ds_mfcc_check as mc                               # noqa: E402


def source_files(ds):
    """Filenames behind a possibly nested Subset, in the dataset's own order.

    setup() wraps the ood set twice -- Subset(ood_dat, random_choice) and then
    random_split on top -- so an index into the valid split is two hops from a
    path. Resolved by walking the wrappers rather than by assuming a depth,
    since the id side has one fewer.
    """
    idx = None
    while isinstance(ds, Subset):
        idx = ds.indices if idx is None else [ds.indices[i] for i in idx]
        ds = ds.dataset
    files = ds.raw_files
    return [files[i] for i in (idx if idx is not None else range(len(files)))]


_NAME = re.compile(r"^(?P<family>[a-z]+(?:_[a-z]+)?)_"
                   r"(?P<source>acoustic|electronic|synthetic)_")


def categorize(path: str) -> tuple[str, str]:
    m = _NAME.match(os.path.basename(path))
    return (m.group("family"), m.group("source")) if m else ("?", "?")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--ckpt", default="latest.ckpt")
    p.add_argument("--split", default="valid", choices=("valid", "test", "train"))
    p.add_argument("--group-by", default="both",
                   choices=("family", "source", "both"))
    p.add_argument("--only", nargs="+", default=None, metavar="GROUP",
                   help="Score only these groups, e.g. --only synth_lead")
    p.add_argument("--min-n", type=int, default=8,
                   help="Skip groups with fewer clips than this. A mean over "
                        "three files is not a measurement and printing it as "
                        "one invites reading it as one.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    metrics = {
        "mfcc": mc.make_mfcc(dev, window="hann", log="db", top_db=80.0,
                             mel_norm="slaney", mel_scale="slaney"),
        "mfcc03": mc.make_mfcc(dev, window="hann", log="pow", gamma=0.3,
                               mel_norm="slaney", mel_scale="slaney"),
    }

    per_arm = {}
    groups = None
    for arm in args.arms:
        d = os.path.join(args.root, arm)
        model, cfg, dm, note = pb.load_arm(d, args.ckpt, dev, args.batch_size)
        if model is None:
            print(f"{arm:<24} skipped: {note}")
            continue
        vset = dm.ood_datasets[args.split]
        files = source_files(vset)

        if groups is None:
            groups = defaultdict(list)
            for i, f in enumerate(files):
                fam, src = categorize(f)
                if args.group_by in ("family", "both"):
                    groups[fam].append(i)
                if args.group_by in ("source", "both"):
                    groups[src].append(i)
                groups["ALL"].append(i)
            groups = {k: v for k, v in groups.items()
                      if len(v) >= args.min_n and (not args.only or k in args.only
                                                   or k == "ALL")}
            print(f"ood {args.split}: {len(files)} clips")
            for k in sorted(groups, key=lambda k: -len(groups[k])):
                print(f"  {k:<14}{len(groups[k]):>6}")
            print()

        scores = defaultdict(lambda: defaultdict(float))
        counts = defaultdict(int)
        for gname, idxs in groups.items():
            loader = DataLoader(Subset(vset, idxs), batch_size=args.batch_size,
                                num_workers=0)
            for batch in loader:
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                with torch.no_grad():
                    out, _ = model(batch)
                tgt = batch["audio"]
                n = tgt.shape[0]
                for mname, fn in metrics.items():
                    # sum, then divide by n -- a mean of per-batch means would
                    # weight a short final batch equally with a full one.
                    # No .cpu(): make_mfcc builds its window on `dev`, and
                    # torch.stft requires signal and window on the same device.
                    scores[gname][mname] += float(
                        F.l1_loss(fn(tgt), fn(out))) * n
                counts[gname] += n
        per_arm[arm] = {g: {m: s / counts[g] for m, s in ms.items()}
                        for g, ms in scores.items()}
        print(f"{arm:<24} {note}")

    if not per_arm:
        raise SystemExit("no arm produced results")

    arms = list(per_arm)
    for mname in metrics:
        print(f"\n=== ood {args.split} / {mname}   (lower is better; "
              f"BEST per row in the last column)")
        w = max(14, max(len(g) for g in groups) + 2)
        print(f"{'group':<{w}}{'n':>6}" + "".join(f"{a:>22}" for a in arms)
              + f"{'winner':>22}")
        for g in sorted(groups, key=lambda k: -len(groups[k])):
            vals = {a: per_arm[a][g][mname] for a in arms if g in per_arm[a]}
            best = min(vals, key=vals.get)
            print(f"{g:<{w}}{len(groups[g]):>6}"
                  + "".join(f"{vals[a]:>22.4f}" for a in arms)
                  + f"{best:>22}")

    print("\n  A ranking that flips between ALL and a subgroup is a fact about "
          "the\n  evaluation set, not about the losses. Read the group sizes: a "
          "flip on\n  a group of 30 against a set of 2000 is noise until shown "
          "otherwise.")


if __name__ == "__main__":
    main()
