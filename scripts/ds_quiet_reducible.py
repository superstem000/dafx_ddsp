"""Is a group's quiet region fittable, or does a log loss weight it for nothing?

    python scripts/ds_quiet_reducible.py --lin real_magx_halfw --log real_logx_halfw
    python scripts/ds_quiet_reducible.py --lin ... --log ... --group-by cross

WHY TWO ARMS. One model cannot tell "did not try to fit the quiet bins" apart
from "could not". A linear-trained arm puts almost no weight below -40 dB, so
its quiet-band residual is what is left when nothing tried. A log-trained arm
weights those bins heavily. The RATIO between them is the measurement:

  residual_log / residual_lin  ~ 1.0   the log arm weighted those bins hard and
                                       got nowhere. Unreachable or meaningless.
                                       Compression's weight is WASTED there,
                                       which is where a linear loss should win.
  well below 1.0                       the quiet region is real, reachable and
                                       informative. Compression buys something.
                                       (This is what the chorus task looks like.)

Paired, so shared capacity and per-family difficulty cancel: both arms saw the
same data under the same schedule and differ only in the loss.

WHAT THIS REPLACES. Ranking families by the SHARE of bins below a threshold.
That counts how much quiet there is, not whether it carries anything, and on
this data the count turned out uncorrelated with which loss wins: linear won on
groups spanning 3% to 19% quiet bins, while the largest hybrid win of all was
string_acoustic at 2.1% -- the least quiet group in the set. Quantity was never
the variable; reducibility is.

THE CAVEAT, stated because it bounds what the answer means. Both arms were
trained on all of NSynth jointly, so a per-family number is partly about how
capacity got allocated across families rather than about the audio alone. It is
a property of these two models on this audio. The model-free version is to fit
the synth to each clip directly under each loss and compare achieved residuals;
this is the cheap screen that says whether that is worth doing.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "external", "diffsynth"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import torch                                              # noqa: E402
from torch.utils.data import DataLoader, Subset           # noqa: E402

import ds_param_breakdown as pb                           # noqa: E402
from ds_ood_subset import source_files, categorize        # noqa: E402
from src.analysis.band_sensitivity import DB_BANDS        # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--lin", default="real_magx_halfw", help="Linear-trained arm")
    p.add_argument("--log", default="real_logx_halfw", help="Log-trained arm")
    p.add_argument("--ckpt", default="latest.ckpt")
    p.add_argument("--split", default="valid", choices=("valid", "test", "train"))
    p.add_argument("--group-by", default="cross",
                   choices=("family", "source", "both", "cross"))
    p.add_argument("--min-n", type=int, default=10)
    p.add_argument("--active-db", type=float, default=40.0,
                   help="Score frames within this many dB of the target's "
                        "loudest frame. A silent frame is silent in both "
                        "target and resynthesis and contributes nothing while "
                        "still counting, so without this the numbers rank "
                        "partly by note length.")
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    win = torch.hann_window(args.n_fft, device=dev)

    def mag(x):
        return torch.stft(x, args.n_fft, hop_length=args.hop, window=win,
                          center=True, return_complex=True).abs()

    per_arm, groups, files = {}, None, None
    for tag, arm in (("lin", args.lin), ("log", args.log)):
        d = os.path.join(args.root, arm)
        model, cfg, dm, note = pb.load_arm(d, args.ckpt, dev, args.batch_size)
        if model is None:
            raise SystemExit(f"{arm}: {note}")
        vset = dm.ood_datasets[args.split]
        if groups is None:
            files = source_files(vset)
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
            groups = {k: v for k, v in groups.items() if len(v) >= args.min_n}

        # Residual per band, summed, plus the target's own band energy as the
        # denominator -- so a band is reported as "how much of what is there is
        # still wrong", which is comparable across bands and across families.
        res = {g: torch.zeros(len(DB_BANDS), dtype=torch.float64) for g in groups}
        ref = {g: torch.zeros(len(DB_BANDS), dtype=torch.float64) for g in groups}
        for gname, idxs in groups.items():
            for batch in DataLoader(Subset(vset, idxs),
                                    batch_size=args.batch_size, num_workers=0):
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                with torch.no_grad():
                    out, _ = model(batch)
                A, B = mag(batch["audio"]).double(), mag(out).double()
                if args.active_db > 0:
                    fe = A.sum(dim=1)
                    keep = fe >= fe.amax(dim=1, keepdim=True) * 10.0 ** (
                        -args.active_db / 20.0)
                    m3 = keep[:, None, :].expand_as(A)
                else:
                    m3 = torch.ones_like(A, dtype=torch.bool)
                # Bands from the TARGET, per frame, so the partition is the
                # same for both arms and is a property of the audio.
                r = A / A.amax(dim=1, keepdim=True).clamp(min=1e-30)
                db = (20.0 * torch.log10(r.clamp(min=1e-300))
                      ).clamp(min=-(float(DB_BANDS[-1][1]) - 1e-3))
                diff = (A - B).abs()
                for i, (lo, hi) in enumerate(DB_BANDS):
                    sel = (db <= -float(lo)) & (db > -float(hi)) & m3
                    res[gname][i] += float(diff[sel].sum())
                    ref[gname][i] += float(A[sel].sum())
        per_arm[tag] = {g: (res[g] / ref[g].clamp(min=1e-30)) for g in groups}
        print(f"{tag:<5}{arm:<24} {note}")

    hdr = "".join(f"{f'{lo}-{hi}':>10}" for lo, hi in DB_BANDS)
    for tag, title in (("lin", f"{args.lin}  residual / target energy, per band"),
                       ("log", f"{args.log}  residual / target energy, per band")):
        print(f"\n=== {title}")
        print(f"{'group':<24}{'n':>6}{hdr}")
        for g in sorted(groups, key=lambda k: -len(groups[k])):
            print(f"{g:<24}{len(groups[g]):>6}"
                  + "".join(f"{v:>10.3f}" for v in per_arm[tag][g]))

    print(f"\n=== RATIO  log / linear   (~1 = the log arm weighted those bins "
          f"and gained nothing)")
    print(f"{'group':<24}{'n':>6}{hdr}")
    for g in sorted(groups, key=lambda k: -len(groups[k])):
        r = per_arm["log"][g] / per_arm["lin"][g].clamp(min=1e-30)
        print(f"{g:<24}{len(groups[g]):>6}"
              + "".join(f"{v:>10.3f}" for v in r))

    print("\n  Read the 40-100 dB columns. A ratio near or above 1.0 there means")
    print("  the log-trained arm weighted that band heavily and did not fit it")
    print("  any better -- the quiet region is UNINFORMATIVE, compression's")
    print("  extra weight is wasted, and a linear loss should win on that group.")
    print("  Well below 1.0 means the quiet region is real and reachable, and")
    print("  compression is buying something there.")
    print("  Above 1.0 is not a bug: the log arm can fit quiet bands WORSE than")
    print("  the linear arm if chasing them costs it the loud bands it needs.")


if __name__ == "__main__":
    main()
