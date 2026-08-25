"""Does any statistic of the AUDIO predict which loss wins on it?

    python scripts/ds_predict_margin.py --lin real_magx_halfw --hyb real_hybridx

Every selection criterion tried so far was proposed and then checked against the
outcome afterwards, which is how three of them survived long enough to be
recommended. Bin-share below a threshold turned out uncorrelated -- linear won
on groups spanning 3% to 19% while the largest hybrid win in the table was
string_acoustic at 2.1%, the least quiet group in the set. The log/linear
residual ratio was the right direction and had two clear counterexamples.

So: compute every candidate statistic, compute the win margin, and correlate
them across all groups in one place. The answer is either a validated predictor
or the knowledge that no audio statistic reaches this -- and either is worth
more than another criterion nobody scored.

THE STATISTICS, all from the TARGET audio alone, no model and no parameters:

  eng_lt<N>   share of spectral ENERGY more than N dB below the frame peak.
              Not bin share: a linear loss weights by energy and a log loss by
              count, so energy-below-threshold is "how much real content is
              down there" while bin-count is "how much weight compression moves
              onto it". The earlier sweep measured the second and read it as
              the first.
  bins_lt<N>  bin share, kept for comparison since it is the one already shown
              to fail.
  flatness    geometric mean over arithmetic mean of the frame spectrum,
              averaged. Near 0 is a few loud partials, near 1 is noise-like.
  eff99       fraction of bins needed to hold 99% of a frame's energy.

THE MARGIN is score_lin - score_hyb on the chosen metric, so positive means
hybrid wins. Scores are normalised by the distance between two unrelated clips
of the same group, as in ds_ood_subset, so they are comparable across groups.

PRE-SPECIFY WHAT COUNTS. Every statistic is reported against the margin with
Pearson and Spearman and a p-value. Reading the strongest of eight and quoting
it as the predictor is the same mistake in a new coat: with 34 groups and 8
statistics, one crossing p<0.05 by chance is expected. The correction is
printed alongside.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "external", "diffsynth"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import torch                                              # noqa: E402
from torch.utils.data import DataLoader, Subset           # noqa: E402

import ds_param_breakdown as pb                           # noqa: E402
import ds_mfcc_check as mc                                # noqa: E402
from ds_ood_subset import source_files, categorize        # noqa: E402


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan"), float("nan")
    r = sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)
    r = max(-0.999999, min(0.999999, r))
    # Two-sided p from the t statistic, normal-approximated. Good enough to
    # separate "obviously nothing" from "worth a closer look" at n ~ 34; it is
    # not a substitute for a proper test if this ends up in the paper.
    t = r * math.sqrt((n - 2) / (1 - r * r))
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return r, p


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--lin", default="real_magx_halfw")
    p.add_argument("--hyb", default="real_hybridx")
    p.add_argument("--ckpt", default="latest.ckpt")
    p.add_argument("--split", default="valid", choices=("valid", "test", "train"))
    p.add_argument("--metric", default="mfcc", choices=("mfcc", "mfcc03", "linmag"))
    p.add_argument("--group-by", default="cross",
                   choices=("family", "source", "both", "cross"))
    p.add_argument("--min-n", type=int, default=10)
    p.add_argument("--active-db", type=float, default=40.0)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[40.0, 60.0, 80.0, 100.0])
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

    metric = {
        "linmag": lambda a, ref=None: mag(a),
        "mfcc": mc.make_mfcc(dev, window="hann", log="db", top_db=80.0,
                             mel_norm="slaney", mel_scale="slaney"),
        "mfcc03": mc.make_mfcc(dev, window="hann", log="pow", gamma=0.3,
                               mel_norm="slaney", mel_scale="slaney"),
    }[args.metric]

    score, groups, stats = {}, None, None
    for tag, arm in (("lin", args.lin), ("hyb", args.hyb)):
        model, cfg, dm, note = pb.load_arm(os.path.join(args.root, arm),
                                           args.ckpt, dev, args.batch_size)
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
            stats = {g: defaultdict(float) for g in groups}
            nclip = {g: 0 for g in groups}

        acc = {g: [0.0, 0.0] for g in groups}
        for gname, idxs in groups.items():
            for batch in DataLoader(Subset(vset, idxs),
                                    batch_size=args.batch_size, num_workers=0):
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                with torch.no_grad():
                    out, _ = model(batch)
                tgt = batch["audio"]
                if tgt.shape[0] < 2:
                    continue
                oth = tgt.roll(1, dims=0)
                A = mag(tgt).double()
                fe = A.sum(dim=1)
                keep = (fe >= fe.amax(dim=1, keepdim=True)
                        * 10.0 ** (-args.active_db / 20.0)
                        ) if args.active_db > 0 else torch.ones_like(fe, dtype=torch.bool)

                a, b, c = metric(tgt), metric(out), metric(oth)
                mm = keep
                if mm.shape[-1] != a.shape[-1]:
                    j = (torch.arange(a.shape[-1], device=dev)
                         * mm.shape[-1] // a.shape[-1])
                    mm = mm[:, j]
                w = mm[:, None, :].expand_as(a)
                acc[gname][0] += float(((a - b).abs() * w).sum())
                acc[gname][1] += float(((a - c).abs() * w).sum())

                if tag != "lin":
                    continue
                # AUDIO STATISTICS, target only, computed once.
                Aa = A * keep[:, None, :]
                pk = A.amax(dim=1, keepdim=True).clamp(min=1e-30)
                db = 20.0 * torch.log10((A / pk).clamp(min=1e-300))
                tot_e = Aa.sum().clamp(min=1e-30)
                tot_b = keep.sum() * A.shape[1]
                for t in args.thresholds:
                    sel = (db < -t) & keep[:, None, :]
                    stats[gname][f"eng_lt{t:g}"] += float(A[sel].sum() / tot_e)
                    stats[gname][f"bins_lt{t:g}"] += float(sel.sum() / tot_b)
                # Flatness: geometric over arithmetic mean, per active frame.
                lg = torch.log(A.clamp(min=1e-12))
                gm = torch.exp(lg.mean(dim=1))
                am = A.mean(dim=1).clamp(min=1e-30)
                fl = (gm / am)[keep]
                stats[gname]["flatness"] += float(fl.mean()) if fl.numel() else 0.0
                # eff99: fraction of bins holding 99% of a frame's energy.
                srt = A.sort(dim=1, descending=True).values
                cum = srt.cumsum(dim=1) / srt.sum(dim=1, keepdim=True).clamp(min=1e-30)
                k99 = (cum < 0.99).sum(dim=1).double() / A.shape[1]
                e9 = k99[keep]
                stats[gname]["eff99"] += float(e9.mean()) if e9.numel() else 0.0
                nclip[gname] += 1
        score[tag] = {g: (acc[g][0] / acc[g][1] if acc[g][1] else float("nan"))
                      for g in groups}
        print(f"{tag:<5}{arm:<24} {note}")

    for g in groups:
        for k in stats[g]:
            stats[g][k] /= max(nclip[g], 1)

    names = sorted(next(iter(stats.values())).keys())
    order = sorted((g for g in groups if g != "ALL"),
                   key=lambda g: score["lin"][g] - score["hyb"][g])
    print(f"\n=== per group   (margin = {args.metric} lin - hyb; "
          f"POSITIVE means hybrid wins)")
    print(f"{'group':<24}{'n':>6}{'margin':>9}" + "".join(f"{n:>11}" for n in names))
    for g in order:
        m = score["lin"][g] - score["hyb"][g]
        print(f"{g:<24}{len(groups[g]):>6}{m:>9.4f}"
              + "".join(f"{stats[g][n]:>11.4f}" for n in names))

    y = [score["lin"][g] - score["hyb"][g] for g in order]
    print(f"\n=== correlation with the margin, n = {len(order)} groups")
    print(f"{'statistic':<14}{'pearson':>10}{'p':>10}{'spearman':>11}{'p':>10}")
    res = []
    for n in names:
        x = [stats[g][n] for g in order]
        r, pv = _pearson(x, y)
        rs, ps = _pearson(_rank(x), _rank(y))
        res.append((n, r, pv, rs, ps))
        print(f"{n:<14}{r:>10.3f}{pv:>10.4f}{rs:>11.3f}{ps:>10.4f}")

    k = len(names)
    best = min(res, key=lambda t: t[2])
    print(f"\n  {k} statistics tested, so the Bonferroni threshold is "
          f"p < {0.05 / k:.4f}, not 0.05.")
    print(f"  Strongest: {best[0]} at r = {best[1]:+.3f}, p = {best[2]:.4f} -- "
          f"{'SURVIVES' if best[2] < 0.05 / k else 'does NOT survive'} it.")
    print("  A statistic that only clears the uncorrected 0.05 after eight were")
    print("  tried is what picking the best of eight looks like when nothing is")
    print("  there. If none survives, no audio statistic here predicts which")
    print("  loss wins, and a family cannot be selected on mechanism -- which is")
    print("  a result about the mechanism, not a failure of the search.")


if __name__ == "__main__":
    main()
