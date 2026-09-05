"""Which predicted parameter is an arm's advantage actually made of?

    python scripts/ds_param_swap.py --dirs data/juno/moog-mt-pad \
        --arms r13_synth_magx_halfw r13_synth_hybridx \
        --match='-(4[89]|5[0-5])-\\d+\\.' --n 50 --folder-peak 0.5 \
        --force-f0-re='-(\\d+)-\\d+\\.[A-Za-z0-9]+$' --device cuda:3

WHY. On mt-pad magx scores 2.72 against hybridx's 3.87 and the reason would
not come out of the parameter table. Level was ruled out by --match-level
(the gap held at 1.21 with the level term removed). Cutoff was ruled out by
arithmetic: logx sits 1000 Hz darker than hybridx and scores 0.11 worse, so
the 220 Hz between magx and hybridx cannot buy 1.21. osc_mix does not track
the ordering. That left the amplitude envelope's SHAPE and q, both of which
happen to be monotone in the score across three arms -- which is three points
and exactly the co-variation that made share2, mix1 and mix2 unreadable
earlier.

So stop inferring. Take arm A's predictions, replace ONE parameter with arm
B's, render, and score. The score moves by however much that parameter was
worth. Every other parameter is held at A's own value, so nothing co-varies.

WHAT IS AND IS NOT CONTROLLED. The swap is exact for the parameter itself:
the dag entry is overwritten after fill_params, so the renderer sees B's
tensor and A's everything-else. What it cannot control is INTERACTION -- an
amplitude curve that suits A's cutoff may not suit B's, so the parts need not
sum to the whole. The residual is reported for that reason rather than left
for the reader to compute: if the single swaps account for most of the gap,
the decomposition is trustworthy; if they do not, the arms differ in a
combination and no single parameter is the answer.

BOTH DIRECTIONS ARE RUN. A-with-B's-x and B-with-A's-x are different
questions -- the first asks what A loses by giving up its value, the second
what B gains by receiving it -- and a parameter that matters should show up
in both. One-directional swaps are how a difference gets attributed to
whichever arm happened to be the donor.

The metric is ds_eval_folder's: MFCC against the same saturation denominator,
the batch rolled by one, so a number here is comparable to one printed there.
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

import ds_mfcc_check as mc                               # noqa: E402
from ds_eval_folder import (audio_files, load_clip,      # noqa: E402
                            load_model, _harmor_of)


def dag_of(model, x: torch.Tensor, f0_hz=None):
    """fill_params output for this batch, with conditioning supplied."""
    cond = {"audio": x}
    proc, conn = _harmor_of(model.synth)
    fixed = set(model.synth.fixed_param_names)
    for k in model.synth.fixed_param_names:
        if getattr(model.synth, k) is not None:
            continue
        if k == conn["f0_hz"] and f0_hz is not None:
            cond[k] = f0_hz.to(x.device, x.dtype).view(x.shape[0], 1, 1)
        else:
            raise SystemExit(
                f"{k} is conditioning for this synth and no value was given; "
                f"pass --force-f0-re, or use arms that predict it.")
    return model.synth.fill_params(model.estimate_param(cond), cond), cond


def score(metric, out, tgt, oth):
    """arm/saturation on one batch, ds_eval_folder's definition."""
    a, b, c = metric(tgt), metric(out), metric(oth)
    num = float((a - b).abs().sum())
    den = float((a - c).abs().sum())
    return num, den


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dirs", nargs="+", required=True, metavar="DIR")
    p.add_argument("--arms", nargs=2, required=True, metavar=("A", "B"))
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--ckpt", default="latest.ckpt")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--length", type=float, default=4.0)
    p.add_argument("--match", default=None, metavar="REGEX")
    p.add_argument("--folder-peak", type=float, default=None, metavar="P")
    p.add_argument("--force-f0-re", default=None, metavar="REGEX")
    p.add_argument("--force-f0-semis", type=float, default=0.0, metavar="S")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    rng = random.Random(args.seed)
    metric = mc.make_mfcc(args.device, window="hann", log="db", top_db=None,
                          mel_norm="slaney", mel_scale="slaney", sr=args.sr)

    for d in args.dirs:
        files = audio_files(d)
        if args.match:
            files = [f for f in files
                     if re.search(args.match, os.path.basename(f))]
        if args.n and len(files) > args.n:
            files = rng.sample(files, args.n)
        files.sort()
        raw = [load_clip(f, args.sr, args.length) for f in files]
        gain = 1.0
        if args.folder_peak is not None:
            nz = [r[2] for r in raw if r[2] > 0]
            med = float(np.median(nz)) if nz else 0.0
            gain = args.folder_peak / med if med > 0 else 1.0
        x = np.stack([r[0] for r in raw]) * gain
        g = os.path.basename(os.path.normpath(d))
        print(f"\n{g}: {len(files)} clips, gain {gain:.3f}")

        f0 = None
        if args.force_f0_re:
            hz = []
            for f in files:
                m = re.search(args.force_f0_re, os.path.basename(f))
                if not m:
                    raise SystemExit(f"--force-f0-re did not match {f}")
                hz.append(440.0 * 2.0 ** ((float(m.group(1))
                                           + args.force_f0_semis - 69.0) / 12.0))
            f0 = torch.tensor(hz, device=args.device)
            print(f"  --force-f0 {min(hz):.1f}-{max(hz):.1f} Hz")

        tgt = torch.from_numpy(x).float().to(args.device)
        oth = tgt.roll(1, dims=0)

        models, dags = {}, {}
        for arm in args.arms:
            m, _cfg, note = load_model(os.path.join(args.root, arm),
                                       args.ckpt, args.device)
            if m is None:
                raise SystemExit(f"{arm}: {note}")
            models[arm] = m
            with torch.no_grad():
                dags[arm], _ = dag_of(m, tgt, f0)
            print(f"  {arm:<26}{note}")

        A, B = args.arms
        keys = [k for k in dags[A] if k in dags[B]
                and torch.is_tensor(dags[A][k])
                and dags[A][k].shape == dags[B][k].shape
                and dags[A][k].numel() > 0]
        rev = {v: k for k, v in models[A].synth.dag_summary.items()}

        base = {}
        for arm in args.arms:
            with torch.no_grad():
                out, _ = models[arm].synth(dags[arm], tgt.shape[1])
            n, dn = score(metric, out, tgt, oth)
            base[arm] = n / dn
        gap = base[B] - base[A]
        print(f"\n  baseline   {A} {base[A]:.4f}   {B} {base[B]:.4f}"
              f"   gap {gap:+.4f}")

        print(f"\n{'swapped parameter':<24}{A[:20] + ' <- ' + B[:6]:>28}"
              f"{B[:20] + ' <- ' + A[:6]:>28}{'explains':>10}")
        rows = []
        for k in keys:
            cells = []
            for host, donor in ((A, B), (B, A)):
                d = dict(dags[host])
                d[k] = dags[donor][k]
                with torch.no_grad():
                    out, _ = models[host].synth(d, tgt.shape[1])
                n, dn = score(metric, out, tgt, oth)
                cells.append(n / dn)
            # How much of the gap this parameter moves, averaged over the two
            # directions: A should get worse and B should get better, and a
            # parameter that does only one of those is suspect.
            moved = ((cells[0] - base[A]) + (base[B] - cells[1])) / 2.0
            rows.append((abs(moved), rev.get(k, k), cells[0], cells[1], moved))
        for _mag, name, ca, cb, moved in sorted(rows, reverse=True):
            pct = 100.0 * moved / gap if abs(gap) > 1e-9 else float("nan")
            print(f"{name:<24}{ca:>28.4f}{cb:>28.4f}{pct:>9.0f}%")

        tot = sum(r[4] for r in rows)
        print(f"\n  single swaps account for {100.0 * tot / gap:.0f}% of the "
              f"gap ({tot:+.4f} of {gap:+.4f}).")
        print("  Over 100% or well under it means the arms differ in a\n"
              "  COMBINATION -- the parameters interact, and no single one is\n"
              "  the answer. Near 100% means the decomposition holds and the\n"
              "  largest row is the mechanism.\n"
              "  'explains' is the average of what the host loses and what the\n"
              "  donor gains, so a parameter has to move both directions.")


if __name__ == "__main__":
    main()
