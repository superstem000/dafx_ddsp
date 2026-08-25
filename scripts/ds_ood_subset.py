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
    p.add_argument("--delta", action="store_true",
                   help="Treat --arms as BASELINES followed by FINALS, in the "
                        "same order, and add a table of baseline minus final: "
                        "how much each arm IMPROVED on that group. Requires an "
                        "even count.\n"
                        "This is the measure that controls for the starting "
                        "point, and the starting points are not equal. At epoch "
                        "199, before any real audio, pre_magx_halfw is worse "
                        "than both compressed bases on 33 of 34 classes -- so an "
                        "endpoint table mixes inherited pretraining differences "
                        "with what the resume actually did. On keyboard_acoustic "
                        "magx starts 0.33 behind and finishes 0.30 ahead, a "
                        "swing only the delta shows; on string_acoustic magx "
                        "improves 1.34 against hybrid's 3.38.\n"
                        "All arms must share an ood_dir, since the grouping is "
                        "built once from the first arm's split.")
    p.add_argument("--ood-dir", default=None, metavar="DIR",
                   help="Score the arms on THIS out-of-domain set instead of "
                        "the one they were configured with -- e.g. the "
                        "keyboard-only split, for arms trained on all of "
                        "NSynth. Pass --id-dir from the same run too: the ood "
                        "subsample is len(id_dat) indices drawn from the ood "
                        "directory (data.py:92), so id_dir decides WHICH clips "
                        "and how many, and an id set larger than the ood set "
                        "raises there outright.\n"
                        "LEAKAGE IS CHECKED, NOT ASSUMED. An arm trained on "
                        "all of NSynth may have trained on the very clips a "
                        "class-specific run holds out, and scoring it on those "
                        "would be train error dressed as validation. Whenever "
                        "this is set, each arm's own training split is "
                        "reproduced and the overlap is reported before any "
                        "table is printed.")
    p.add_argument("--id-dir", default=None, metavar="DIR",
                   help="In-domain directory to pair with --ood-dir. Only its "
                        "SIZE matters for an ood table, but the size is what "
                        "selects the ood subsample, so it has to match the run "
                        "whose split is being borrowed.")
    p.add_argument("--split", default="valid", choices=("valid", "test", "train"))
    p.add_argument("--domain", default="ood", choices=("ood", "id"),
                   help="id scores the in-domain split, where the grouping is "
                        "meaningless but the METRICS are the point: linear's "
                        "established win is on val_id/param, a ground-truth "
                        "parameter metric, and whether it also wins on in-domain "
                        "AUDIO metrics has never been checked. If it does not, "
                        "the OOD ranking needs no domain explanation at all.")
    p.add_argument("--group-by", default="both",
                   choices=("family", "source", "both", "cross"),
                   help="cross adds family_source intersections -- "
                        "organ_electronic, bass_synthetic -- alongside the "
                        "marginals. The marginals can mislead: a family's "
                        "number is a mix of however its sources happen to be "
                        "distributed, and 'electronic organ' is a different "
                        "instrument from 'acoustic organ' in every way that "
                        "matters to a harmonic-oscillator model.")
    p.add_argument("--only", nargs="+", default=None, metavar="GROUP",
                   help="Score only these groups, e.g. --only synth_lead")
    p.add_argument("--min-n", type=int, default=8,
                   help="Skip groups with fewer clips than this. A mean over "
                        "three files is not a measurement and printing it as "
                        "one invites reading it as one.")
    p.add_argument("--active-db", type=float, default=40.0, metavar="DB",
                   help="Score only frames whose energy is within this many dB "
                        "of the target's loudest frame. A silent frame is "
                        "silent in both target and resynthesis and clamps to "
                        "the same floor, so it contributes nothing to the sum "
                        "while still counting in the mean -- which divides a "
                        "mostly-finished clip's score down. mallet_acoustic is "
                        "active in 32% of frames and vocal_acoustic in 85%, so "
                        "the raw column ranked partly by note length. 0 scores "
                        "whole clips.")
    p.add_argument("--norm", default="sat", choices=("sat", "none"),
                   help="sat divides by the distance between two unrelated "
                        "clips of the same group, making rows comparable "
                        "across groups. none reports the plain L1 -- the "
                        "number val_ood/mfccdb logs, on the scale it is read "
                        "at. Use none when the question is about an arm's own "
                        "trajectory rather than about ranking groups. NOT "
                        "val_ood/mfcc, which is torchaudio's rectangular-"
                        "window HTK cepstrum and reads 1.6-2.6 where this "
                        "column reads 7-14; runs started before mfccdb existed "
                        "have no logged curve comparable to this table.")
    p.add_argument("--top-db", default="80", metavar="DB",
                   help="The mfcc column's dynamic-range bound, in dB below "
                        "each clip's own loudest mel band. 'none' removes it.\n"
                        "IT DECIDES WHAT THE COLUMN CAN SEE. At the default 80 "
                        "anything quieter is clamped flat in BOTH target and "
                        "resynthesis, so it contributes exactly zero -- which "
                        "means a class whose quiet region is empty and deep is "
                        "invisible to the metric no matter what a log loss did "
                        "with it. bass_electronic has 64% of its bins 80-120 dB "
                        "down; none of that reaches the default column. Removing "
                        "the bound is how to ask whether a linear arm's "
                        "advantage lives down there.\n"
                        "'none' is not unbounded, and the difference matters: "
                        "--mfcc-eps still sits inside the log, at 1e-10, which "
                        "is an ABSOLUTE floor near -100 dB rather than a "
                        "peak-relative one. So a quiet recording is bounded "
                        "sooner than a loud one, and the column stops being "
                        "comparable across clips of different level in a way "
                        "the top_db version is not.")
    p.add_argument("--mfcc-eps", type=float, default=None, metavar="E",
                   help="Additive eps inside the log, default 1e-10. With "
                        "--top-db none this is the only thing bounding the "
                        "column, so it is the knob that sets how deep it looks.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    top_db = None if str(args.top_db).lower() in ("none", "off") else float(args.top_db)

    dev = args.device if torch.cuda.is_available() else "cpu"
    # A LINEAR audio metric alongside the compressed ones, and it is the
    # control the first version lacked. mfcc is log-mel at top_db 80 and mfcc03
    # is gamma 0.3, so both REWARD a log-domain loss for optimizing something
    # structurally like what they measure -- metric-loss alignment, which has
    # nothing to do with whether the target is a synth lead or a bowed string.
    # Without a metric on plain magnitude there is no way to tell that apart
    # from a real difference in fit.
    win = torch.hann_window(1024, device=dev)

    def _linmag(x):
        return torch.stft(x, 1024, hop_length=256, window=win, center=True,
                          return_complex=True).abs()

    # The column NAMES the bound it was computed under, so a table pasted into
    # a document cannot be read as the default one when it is not. "mfcc" means
    # top_db 80 and nothing else.
    mfcc_key = ("mfcc" if top_db == 80.0 else
                "mfcc_nofl" if top_db is None else f"mfcc_db{top_db:g}")
    metrics = {
        "linmag": lambda a, ref=None: _linmag(a),
        mfcc_key: mc.make_mfcc(dev, window="hann", log="db", top_db=top_db,
                               eps=args.mfcc_eps,
                               mel_norm="slaney", mel_scale="slaney"),
        "mfcc03": mc.make_mfcc(dev, window="hann", log="pow", gamma=0.3,
                               mel_norm="slaney", mel_scale="slaney"),
    }

    override = {}
    if args.ood_dir:
        override["ood_dir"] = args.ood_dir
    if args.id_dir:
        override["id_dir"] = args.id_dir

    def trained_on(arm: str) -> set[str] | None:
        """Basenames in an arm's OWN training split, reproduced from its config.

        A separate load_arm call, which re-seeds, so it reconstructs the split
        the run actually trained on rather than whatever the override produced.
        Built on cpu: only the file lists are wanted, no audio is read and no
        forward pass runs.
        """
        m, c, dm_, _ = pb.load_arm(os.path.join(args.root, arm), args.ckpt,
                                   "cpu", args.batch_size)
        if m is None:
            return None
        tt = c.data.get("train_type", "ood")
        sets = []
        if tt in ("ood", "mixed"):
            sets.append(dm_.ood_datasets["train"])
        if tt in ("id", "mixed"):
            sets.append(dm_.id_datasets["train"])
        return {os.path.basename(f) for s in sets for f in source_files(s)}

    per_arm = {}
    groups = None
    for arm in args.arms:
        d = os.path.join(args.root, arm)
        model, cfg, dm, note = pb.load_arm(d, args.ckpt, dev, args.batch_size,
                                           data_override=override or None)
        if model is None:
            print(f"{arm:<24} skipped: {note}")
            continue
        vset = (dm.ood_datasets if args.domain == "ood"
                else dm.id_datasets)[args.split]
        files = source_files(vset)

        if override:
            # WITHOUT THIS THE TABLE CAN BE TRAIN ERROR. These weights were
            # fitted on some set of clips; the override points them at another
            # one, and nothing guarantees the two are disjoint -- an arm
            # trained on all of NSynth has seen keyboard clips, and a
            # class-specific holdout drawn from the same archive can be made of
            # exactly those. The overlap is a property of the two directories
            # and the two splits, so it is computed rather than reasoned about.
            tr = trained_on(arm)
            ev = {os.path.basename(f) for f in files}
            sha = pb.split_sha1(vset)
            if tr is None:
                print(f"{arm:<24} eval {len(ev)} clips  split {sha[:12]}  "
                      f"(training split unavailable, overlap unknown)")
            else:
                n = len(ev & tr)
                print(f"{arm:<24} eval {len(ev)} clips  split {sha[:12]}  "
                      f"seen in training: {n} ({100.0 * n / max(len(ev), 1):.1f}%)"
                      + ("   <-- NOT A HELD-OUT SET" if n else ""))

        if groups is None:
            groups = defaultdict(list)
            for i, f in enumerate(files):
                fam, src = categorize(f)
                if args.group_by in ("family", "both", "cross"):
                    groups[fam].append(i)
                if args.group_by in ("source", "both", "cross"):
                    groups[src].append(i)
                if args.group_by == "cross":
                    groups[f"{fam}_{src}"].append(i)
                groups["ALL"].append(i)
            groups = {k: v for k, v in groups.items()
                      if len(v) >= args.min_n and (not args.only or k in args.only
                                                   or k == "ALL")}
            print(f"{args.domain} {args.split}: {len(files)} clips")
            for k in sorted(groups, key=lambda k: -len(groups[k])):
                print(f"  {k:<14}{len(groups[k]):>6}")
            print()

        # NORMALISED BY AN UNRELATED PAIR OF THE SAME FAMILY. A raw L1 on
        # MFCCs is on whatever scale that family's MFCCs happen to occupy, so
        # a family with more spectral variation scores worse at equal relative
        # fit and the column cannot be read across rows. Dividing by the same
        # distance between two UNRELATED clips of the same group gives
        # "fraction of the distance between two clips of this kind": 0 is
        # exact, 1 is no better than picking another clip at random. Same
        # denominator the plate work uses as `saturation`.
        #
        # It also cancels most of the silence dilution -- a family whose clips
        # are two-thirds over deflates numerator and denominator together --
        # though not exactly, since note lengths vary within a family too,
        # which is what --active-db cleans up.
        scores = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
        for gname, idxs in groups.items():
            loader = DataLoader(Subset(vset, idxs), batch_size=args.batch_size,
                                num_workers=0)
            for batch in loader:
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                with torch.no_grad():
                    out, _ = model(batch)
                tgt = batch["audio"]
                if tgt.shape[0] < 2:
                    continue          # no partner for the denominator
                # The unrelated partner is the batch rolled by one. Group
                # members are contiguous in the Subset, so a roll pairs clips
                # of the SAME family -- which is the point: the denominator has
                # to be that family's own spread, not the whole set's.
                oth = tgt.roll(1, dims=0)

                m = None
                if args.active_db > 0:
                    # Mask from the TARGET only. Taking it from each arm's
                    # output would grade an arm that under-synthesises on
                    # fewer frames, biasing the comparison in the direction
                    # under dispute.
                    fe = _linmag(tgt).sum(dim=1)
                    m = fe >= fe.amax(dim=1, keepdim=True) * 10.0 ** (
                        -args.active_db / 20.0)

                for mname, fn in metrics.items():
                    a, b, c = fn(tgt), fn(out), fn(oth)
                    mm = m
                    if mm is not None and mm.shape[-1] != a.shape[-1]:
                        # MFCC and the raw spectrogram can differ by a frame
                        # of padding; index rather than assume they match.
                        j = (torch.arange(a.shape[-1], device=dev)
                             * mm.shape[-1] // a.shape[-1])
                        mm = mm[:, j]
                    if mm is None:
                        num = float((a - b).abs().sum())
                        den = float((a - c).abs().sum())
                        k = a.numel()
                    else:
                        w = mm[:, None, :].expand_as(a)
                        num = float(((a - b).abs() * w).sum())
                        den = float(((a - c).abs() * w).sum())
                        k = float(w.sum())
                    e = scores[gname][mname]
                    e[0] += num
                    e[1] += den
                    e[2] += k
        per_arm[arm] = {g: {m: ((e[0] / e[1] if e[1] else float("nan"))
                                if args.norm == "sat"
                                else (e[0] / e[2] if e[2] else float("nan")))
                            for m, e in ms.items()}
                        for g, ms in scores.items()}
        print(f"{arm:<24} {note}")

    if not per_arm:
        raise SystemExit("no arm produced results")

    arms = list(per_arm)
    if args.delta and len(arms) % 2:
        raise SystemExit(f"--delta needs an even number of arms, got {len(arms)}")

    arms = list(per_arm)
    for mname in metrics:
        cap = ("fraction of the distance to an unrelated clip of the SAME "
               "group; 0 exact, 1 no better than random" if args.norm == "sat"
               else "plain L1, the scale val_ood logs at; NOT comparable "
                    "across groups")
        print(f"\n=== {args.domain} {args.split} / {mname}   ({cap})")
        w = max(14, max(len(g) for g in groups) + 2)
        print(f"{'group':<{w}}{'n':>6}" + "".join(f"{a:>22}" for a in arms)
              + f"{'winner':>22}")
        for g in sorted(groups, key=lambda k: -len(groups[k])):
            vals = {a: per_arm[a][g][mname] for a in arms if g in per_arm[a]}
            best = min(vals, key=vals.get)
            print(f"{g:<{w}}{len(groups[g]):>6}"
                  + "".join(f"{vals[a]:>22.4f}" for a in arms)
                  + f"{best:>22}")

    if args.delta:
        h = len(arms) // 2
        pairs = list(zip(arms[:h], arms[h:]))
        for mname in metrics:
            print(f"\n=== IMPROVEMENT on {mname}: baseline minus final, so "
                  f"POSITIVE is better")
            w = max(14, max(len(g) for g in groups) + 2)
            lbl = [f"{b.split('_')[0]}->{f}" for b, f in pairs]
            print(f"{'group':<{w}}{'n':>6}" + "".join(f"{l:>26}" for l in lbl)
                  + f"{'most improved':>26}")
            for g in sorted(groups, key=lambda k: -len(groups[k])):
                d = {f: per_arm[b][g][mname] - per_arm[f][g][mname]
                     for b, f in pairs if g in per_arm[b] and g in per_arm[f]}
                if not d:
                    continue
                best = max(d, key=d.get)
                print(f"{g:<{w}}{len(groups[g]):>6}"
                      + "".join(f"{d[f]:>26.4f}" for _, f in pairs)
                      + f"{best:>26}")
        print("\n  The endpoint tables and this one answer different questions.")
        print("  An endpoint says which model is better on that class; this says")
        print("  which LOSS learned more from the real data there, with the")
        print("  unequal starting points divided out. They disagree: magx has the")
        print("  best endpoint on four classes but the largest improvement on")
        print("  more than that, because it starts behind on nearly all of them.")

    print("\n  Read DOWN a column to rank how representable each group is by "
          "this\n  synthesizer, and ACROSS a row to rank the losses on it. Both "
          "are now\n  on one scale -- each number is a fraction of the distance "
          "between two\n  unrelated clips of that same group -- so 0.30 for "
          "mallet and 0.30 for\n  vocal mean the same thing, which the raw L1 "
          "column did not.")
    print("  A ranking that flips between ALL and a subgroup is a fact about "
          "the\n  evaluation set, not about the losses. Read the group sizes: a "
          "flip on\n  a group of 30 against a set of 2000 is noise until shown "
          "otherwise.")


if __name__ == "__main__":
    main()
