"""Per-band identifiability on the plate. See src/analysis/band_identifiability.

    python -m src.ddsp.diag_band_identifiability --fixed-mode-grid 60,185
    PLATE_PARAM_SPACE=quiet7 python -m src.ddsp.diag_band_identifiability \
        --fixed-mode-grid 60,185 --n 24 --k 32

    # the resolution question: three sets in ONE run, sharing the renders, so
    # the tables differ in the STFT and in nothing else
    PLATE_PARAM_SPACE=emt14 python -m src.ddsp.diag_band_identifiability \
        --fixed-mode-grid 205,411 --duration 1.0 \
        --n-fft 4096 512,1024,2048,4096,8192 1024,2048,4096,8192

This is the counterpart of scripts/ds_band_identifiability, mirroring it flag
for flag -- --per-param, --vary/--rest, --pin, --draw, --log-radius,
--radius-decades, --hard-ratio, the --max-rel ladder and the --n-fft set all
mean the same thing in both -- so a plate table and a diffsynth table differ in
the system and in nothing else. The analysis underneath is the same module.

PARAMETER CONVENTION. The plate's normalized space is [-1, 1] while diffsynth's
is [0, 1], so everything here is sampled on [0, 1] and mapped at render time.
That is not cosmetic: --max-rel, --draw, --pin and --rest then carry identical
meanings across the two scripts -- a fraction of range, and a position in range
-- and a radius of 0.3 is the same fraction of the search box on both.

--fixed-mode-grid IS LOAD-BEARING, exactly as in diag_param_sensitivity. E,
rho, h, T0 and nu all change the mode COUNT, so an unpinned grid follows the
batch maximum and a candidate renders a different number of modes than its
target. That difference lands in the quietest bins, which is precisely the
column being read, and it would show up as spurious identifiability down there.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from src.plate.SevenParamPlate import BatchedModalPlateTorch
from src.cmaes.fit_7param_norm_es import (
    PARAM_KEYS, PARAM_SPACE, norm_to_physical, physical_to_plate14_tensor)
from src.analysis.band_sensitivity import EPS, stft_mag
from src.analysis import band_identifiability as bi


def _grid(text):
    a, b = text.split(",")
    return int(a), int(b)


def _sets(tokens, what):
    """['4096', '512,1024,4096'] -> [[4096], [512, 1024, 4096]]."""
    out = []
    for t in tokens:
        vals = [int(v) for v in t.split(",") if v.strip()]
        if not vals:
            raise SystemExit(f"{what}: empty set in {t!r}")
        out.append(vals)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=24, help="Targets")
    p.add_argument("--k", type=int, default=32, help="Candidates per target")
    p.add_argument("--max-rel", type=float, nargs="+", default=[0.30],
                   metavar="R",
                   help="Radii uniform in (0, this] as a fraction of range, "
                        "along random directions. A spread, not a fixed "
                        "radius -- concordance needs candidates at different "
                        "true distances to have anything to rank. Takes a "
                        "LADDER: concordance is not a constant of a parameter, "
                        "it is a function of how far apart the candidates are, "
                        "and a loss can order distant candidates well while "
                        "being noise on close ones -- which is the regime an "
                        "optimiser near a minimum actually sits in. Read "
                        "whether the ranking of parameters survives the "
                        "ladder, not one row of it.")
    p.add_argument("--pin", nargs="+", default=None, metavar="NAME=V",
                   help="Hold a parameter at V in both target and candidates, "
                        "so it contributes no distance and no difference. V is "
                        "in [0,1] as a POSITION IN RANGE, matching "
                        "ds_band_identifiability -- 0.5 is the middle of the "
                        "searched interval, not the physical value 0.5.")
    p.add_argument("--vary", nargs="+", default=None, metavar="NAME",
                   help="Search ONLY these parameters; every other one is held "
                        "at --rest for both target and candidates. This defines "
                        "a TASK the way PLATE_PARAM_SPACE does, at finer grain, "
                        "and the task is what the decomposition is a property "
                        "of -- a plate does not have one answer, a chosen set "
                        "of searched parameters does. Combine with --pin to put "
                        "the held parameters somewhere other than --rest.")
    p.add_argument("--rest", type=float, default=0.5, metavar="V",
                   help="Where --vary holds the unsearched parameters, in [0,1] "
                        "of range.")
    p.add_argument("--per-param", action="store_true",
                   help="One row per searched parameter instead of one band "
                        "table. Candidates differ from the target in that "
                        "parameter ONLY, so concordance answers 'can this loss "
                        "tell which candidate is closer in THIS parameter', "
                        "while the rest of the target stays randomly drawn per "
                        "target. That is the difference from --vary, which pins "
                        "the background at --rest and so reports a property of "
                        "one operating point. Every parameter is measured on "
                        "the same targets, so the rows are paired and the ORDER "
                        "is the result.")
    p.add_argument("--log-radius", action="store_true",
                   help="Draw candidate radii LOG-uniform over "
                        "--radius-decades below --max-rel instead of uniform "
                        "on (0, max-rel]. Uniform puts one candidate in fifteen "
                        "inside 2% of range at the default, so the pairs that "
                        "decide whether a minimum is resolvable are a rounding "
                        "error in the count.")
    p.add_argument("--radius-decades", type=float, default=2.0, metavar="D",
                   help="How many decades below --max-rel --log-radius spans. "
                        "2 means radii from max_rel/100 to max_rel.")
    p.add_argument("--hard-ratio", type=float, default=None, metavar="F",
                   help="ALSO report concordance over pairs whose true "
                        "distances are within a factor F of each other, e.g. "
                        "1.5. A pair at 0.02 against one at 0.28 is ordered "
                        "correctly by anything and says nothing about resolving "
                        "a minimum; the hard pairs are the ones an optimiser "
                        "near one actually faces, and they are a minority of "
                        "the pair count unless asked for separately.")
    p.add_argument("--draw", nargs="+", default=None, metavar="NAME=LO:HI",
                   help="Sample this parameter on [LO, HI] of its range "
                        "(0-1) instead of all of it, and read --max-rel as a "
                        "fraction of THAT span. Use it when the dataset a "
                        "checkpoint was trained on restricts the draw without "
                        "changing the parameter space, so the full range would "
                        "ask about a family the model never saw and inflate "
                        "every radius by the ratio of the two spans.")
    p.add_argument("--list", action="store_true",
                   help="Print the searched parameters and exit.")
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--n-fft", nargs="+", default=["4096"], metavar="SET",
                   help="One or more RESOLUTION SETS, each a comma-joined list "
                        "of STFT sizes: 4096 is the L1_STFT family, "
                        "512,1024,2048,4096,8192 is the _m5 arms. A set with "
                        "more than one size measures a MULTI-RESOLUTION loss "
                        "rather than one spectrogram -- every bin is weighted "
                        "exactly as _make_stft_l1 weights it, mean over bins "
                        "within a resolution then mean over resolutions -- so "
                        "the reported concordance is that loss's concordance. "
                        "SEVERAL SETS IN ONE RUN SHARE THE RENDERS. Targets and "
                        "candidates depend on --seed and not on the STFT, so "
                        "separate runs would re-render identical audio; here "
                        "each size is transformed once per target and every set "
                        "that names it reuses the result. The comparison is "
                        "exactly paired, which is the whole point -- the "
                        "difference between two sets' tables is the resolution "
                        "and nothing else.")
    p.add_argument("--hop", nargs="+", default=None, metavar="SET",
                   help="One comma-joined set per --n-fft set, same shape. "
                        "Default n_fft//4 throughout, which is the 75% overlap "
                        "the losses use.")
    p.add_argument("--floor-db", type=float, default=None,
                   help="Set the log measure's floor this far below each "
                        "target's peak instead of at the absolute eps 1e-7. At "
                        "the default the floor sits ~160 dB down, forty below "
                        "where this float32 modal sum stops being physics. With "
                        "several resolutions the floor is set per resolution, "
                        "against that resolution's own peak -- the STFT is "
                        "unnormalized, so one absolute eps sits at a different "
                        "percentile of the bin distribution on every rung and a "
                        "resolution ladder would confound the floor moving with "
                        "the resolution changing.")
    p.add_argument("--fixed-mode-grid", type=_grid, default=None, metavar="DDX,DDY")
    p.add_argument("--mode-bucket", type=int, default=1024)
    # The modal sum allocates [B, n_modes, chunk] where chunk = chunk_elems /
    # (B * n_modes), so the transient is chunk_elems * 4 bytes per tensor and
    # three of them are live at once. The 1e9 the datasets are generated with
    # is 4 GB a tensor -- fine on an idle card, an instant OOM beside a
    # training job. 5e7 is 200 MB, which fits in what a busy card has left.
    # This is a DIAGNOSTIC, not a target render: chunk_elems changes speed and
    # nothing else here, so it does not have to match the generation contract.
    p.add_argument("--chunk-elems", type=int, default=50_000_000)
    p.add_argument("--render-batch", type=int, default=8, metavar="K",
                   help="Render this many candidates at a time. Candidates are "
                        "a batch dimension of the modal sum, so K=32 in one "
                        "call is a 4x larger transient than four calls of 8 "
                        "for identical output. Lower it further to share a "
                        "card with a training job.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    nffts = _sets(args.n_fft, "--n-fft")
    hops = (_sets(args.hop, "--hop") if args.hop
            else [[nf // 4 for nf in S] for S in nffts])
    if len(hops) != len(nffts) or any(len(h) != len(s) for h, s in zip(hops, nffts)):
        raise SystemExit(
            f"--hop must mirror --n-fft exactly: "
            f"{[len(s) for s in nffts]} sizes against {[len(h) for h in hops]} hops")
    # Every (size, hop) pair appearing in ANY set, transformed once per target.
    # The sets overlap heavily by design -- 4096 is in all three of the
    # comparison that motivated this -- and recomputing a shared size per set
    # is the same waste as re-rendering, one level down.
    uniq = sorted({(nf, hp) for S, H in zip(nffts, hops) for nf, hp in zip(S, H)})
    tag = ["+".join(str(n) for n in S) for S in nffts]

    label = list(PARAM_KEYS)
    P = len(label)
    if args.list:
        print(f"{PARAM_SPACE}   {P} searched parameters:")
        for l in label:
            print(f"  {l}")
        return

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    plate = BatchedModalPlateTorch(
        device=dev, batched_modal_sum=True, compile_modal_sum=False,
        chunk_elems=args.chunk_elems, mode_bucket=args.mode_bucket)
    plate.fixed_mode_grid = args.fixed_mode_grid
    if args.fixed_mode_grid is None:
        print("WARNING: no --fixed-mode-grid. Candidates and targets will sum "
              "different\n  numbers of modes, and that difference lands in the "
              "quiet bins this tool reads.\n")

    # --vary first so an explicit --pin can override where a held parameter sits.
    pins = {}
    if args.vary:
        bad = set(args.vary) - set(label)
        if bad:
            raise SystemExit(f"unknown: {', '.join(sorted(bad))}; "
                             f"have: {', '.join(label)}")
        pins = {i: args.rest for i, l in enumerate(label) if l not in args.vary}
    for item in args.pin or []:
        k, v = item.split("=")
        if k not in label:
            raise SystemExit(f"unknown parameter {k!r}; have: {', '.join(label)}")
        pins[label.index(k)] = float(v)

    draw = {}
    for item in args.draw or []:
        k, _, span = item.partition("=")
        if k not in label:
            raise SystemExit(f"unknown parameter {k!r}; have: {', '.join(label)}")
        lo, _, hi = span.partition(":")
        lo, hi = float(lo), float(hi)
        if not 0.0 <= lo < hi <= 1.0:
            raise SystemExit(f"--draw {k}: need 0 <= LO < HI <= 1, got {lo}:{hi}")
        draw[label.index(k)] = (lo, hi)

    searched = [l for i, l in enumerate(label) if i not in pins]
    if not searched:
        raise SystemExit("every parameter is held; nothing is being searched")
    print(f"plate   space {PARAM_SPACE}   {P} parameters, {len(searched)} "
          f"searched   {args.n} targets   {args.k} candidates each   "
          f"radii (0, {', '.join(f'{r:g}' for r in args.max_rel)}] of range")
    print(f"searching: {', '.join(searched)}")
    print(f"resolution sets ({len(nffts)}, sharing one set of renders):")
    for S, H in zip(nffts, hops):
        print(f"  n_fft {' '.join(str(n) for n in S)}"
              f"   hop {' '.join(str(h) for h in H)}"
              + ("   (bins weighted as the multi-resolution loss weights them)"
                 if len(S) > 1 else ""))
    if draw:
        print("draw ranges: " + ", ".join(
            f"{label[i]}=[{lo:g},{hi:g}] (radii are a fraction of {hi - lo:g})"
            for i, (lo, hi) in sorted(draw.items())))
    if pins:
        print(f"held: {', '.join(f'{label[i]}={v:g}' for i, v in sorted(pins.items()))}")

    def render(u: torch.Tensor) -> torch.Tensor:
        """u is [B, P] on [0,1]; the plate's own normalized space is [-1,1]."""
        norm = (u.numpy().astype(np.float64) * 2.0) - 1.0
        out = []
        for i in range(0, norm.shape[0], args.render_batch):
            p14 = physical_to_plate14_tensor(
                norm_to_physical(norm[i:i + args.render_batch]), dev)
            with torch.no_grad():
                out.append(plate.forward(p14, args.duration, normalize=False))
        return torch.cat(out, dim=0)

    def _mag(n, room, gen):
        """Offset magnitudes in (0, room], uniform or log-uniform.

        LOG-UNIFORM EXISTS BECAUSE UNIFORM UNDER-SAMPLES THE FINE END. At
        max_rel 0.3 only one candidate in fifteen lands within 2% of range, so
        the pair set is dominated by candidates far apart -- which any loss
        orders correctly -- and the near ones that decide whether a minimum is
        resolvable are a rounding error. Lowering max_rel does not fix that; it
        slides the whole cloud inward and leaves the same shape. Log-uniform
        spends equal numbers of candidates on every decade of separation.
        """
        u = torch.rand(n, generator=gen)
        if args.log_radius:
            return room * 10.0 ** (-u * args.radius_decades)
        return room * u

    def sweep(axis=None, max_rel=0.30):
        # SAME SEED FOR EVERY AXIS. Each parameter is measured on the identical
        # targets, so the per-parameter rows are paired rather than separate
        # experiments -- a difference between two rows is the parameter, not a
        # different draw of backgrounds.
        g = torch.Generator().manual_seed(args.seed)
        rows = [[] for _ in nffts]
        marg = [[] for _ in nffts]
        dropped = 0
        for _ in range(args.n):
            tgt = torch.rand(P, generator=g)
            for i, (lo, hi) in draw.items():
                tgt[i] = tgt[i] * (hi - lo) + lo
            for i, v in pins.items():
                tgt[i] = v

            if axis is None:
                d = torch.randn((args.k, P), generator=g)
                d /= d.norm(dim=1, keepdim=True).clamp(min=1e-30)
                r = _mag(args.k, max_rel, g)[:, None]
                cand = (tgt[None, :] + d * r).clamp(0.0, 1.0)
                for i, (lo, hi) in draw.items():
                    cand[:, i] = cand[:, i].clamp(lo, hi)
            else:
                # ONE AXIS, RANDOM BACKGROUND. The candidates differ from the
                # target in this parameter and nothing else, so dist IS |delta p|
                # and the concordance is "can this loss tell which candidate is
                # closer in p". The rest of the target stays randomly drawn per
                # target, which is what --vary gives up: it pins the background
                # at --rest, so every target sits at one operating point and the
                # answer is a property of that point rather than of the parameter.
                #
                # DRAWN INSIDE THE BOUNDS, NOT CLAMPED TO THEM. A clamp maps
                # every over-the-edge candidate onto the boundary VALUE, so they
                # become identical patches with identical losses -- ties, counted
                # at 0.5, dragging concordance toward the coin flip for exactly
                # the targets sitting near an edge.  Instead: pick a side with
                # probability proportional to the room on that side and draw the
                # magnitude within it, which is uniform over the feasible offsets
                # and produces no duplicates.
                p0 = float(tgt[axis])
                blo, bhi = draw.get(axis, (0.0, 1.0))
                reach = max_rel * (bhi - blo)
                down, up = min(reach, p0 - blo), min(reach, bhi - p0)
                if down + up <= 0.0:
                    continue
                left = torch.rand(args.k, generator=g) * (down + up) < down
                room = torch.where(left, torch.full((args.k,), down),
                                   torch.full((args.k,), up))
                off = _mag(args.k, room, g) * torch.where(
                    left, -torch.ones(args.k), torch.ones(args.k))
                cand = tgt[None, :].repeat(args.k, 1)
                cand[:, axis] = p0 + off
            for i, v in pins.items():
                cand[:, i] = v
            # After the bounds handling and after the pins, so a candidate whose
            # only movement was in a pinned parameter is labelled with the
            # distance it actually has rather than the one it was drawn at.
            dist = (cand - tgt[None, :]).norm(dim=1)

            x_ref = render(tgt[None, :])[0]
            x_can = render(cand)
            ok = torch.isfinite(x_can).all(dim=-1)
            if not bool(ok.all()):
                dropped += int((~ok).sum())
                x_can, dist = x_can[ok], dist[ok.cpu()]
            if x_can.shape[0] < 4 or not torch.isfinite(x_ref).all():
                continue

            # Once per (size, hop), not once per set: the sets share sizes.
            cache = {(nf, hp): (stft_mag(x_ref[None, :], nf, hp, True)[0],
                                stft_mag(x_can, nf, hp, True))
                     for nf, hp in uniq}
            dt = dist.to(next(iter(cache.values()))[0].device)
            for si, (S, H) in enumerate(zip(nffts, hops)):
                A_ref = [cache[(nf, hp)][0] for nf, hp in zip(S, H)]
                A_can = [cache[(nf, hp)][1] for nf, hp in zip(S, H)]
                eps = (EPS if args.floor_db is None
                       else [float(A.max()) * 10.0 ** (-args.floor_db / 20.0)
                             for A in A_ref])
                rows[si].append(bi.probe(A_ref, A_can, dt, eps))
                marg[si].append(bi.marginal(A_ref, A_can, dt, eps,
                                            args.hard_ratio))
        return rows, marg, dropped

    if args.per_param:
        if args.vary:
            raise SystemExit(
                "--per-param and --vary are incompatible: --vary pins the "
                "unsearched parameters at --rest, which is the single operating "
                "point --per-param exists to average over. Use --pin for "
                "parameters that are genuinely held.")
        for mr in args.max_rel:
            how = (f"log-uniform over {args.radius_decades:g} decades "
                   f"below {mr:g}" if args.log_radius
                   else f"uniform on (0, {mr:g}]")
            pairs = (f"pairs within {args.hard_ratio:g}x in distance"
                     if args.hard_ratio else "all pairs")
            print(f"\n=== PER PARAMETER   radii {how}   {pairs}\n"
                  f"    candidates differ in ONE parameter, background "
                  f"redrawn per target")
            outs = [[] for _ in nffts]
            for l in searched:
                rows, marg, dropped = sweep(label.index(l), mr)
                for si, mset in enumerate(marg):
                    live = [m for m in mset if m]
                    if not live:
                        continue
                    # PAIRED PER TARGET. id_lin and id_log for one target are
                    # computed on the SAME candidates, so their difference is
                    # paired and its spread over targets is the error bar that
                    # matters. Two separate means with no dispersion cannot tell
                    # +0.006 from +0.047, which is the whole question when the
                    # differences are this small.
                    kl = "id_lin_hard" if args.hard_ratio else "id_lin"
                    kg = "id_log_hard" if args.hard_ratio else "id_log"
                    live = [m for m in live if m.get(kl, 0.0) == m.get(kl, 0.0)]
                    if not live:
                        continue
                    fl = sum(m[kl] for m in live) / len(live)
                    fg = sum(m[kg] for m in live) / len(live)
                    d = [m[kl] - m[kg] for m in live]
                    n = len(d)
                    mu = sum(d) / n
                    var = sum((x - mu) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
                    se = (var / n) ** 0.5
                    wins = sum(1 for x in d if x > 0) / n
                    outs[si].append((mu, l, fl, fg, se, wins, n))
            for si, out in enumerate(outs):
                if not out:
                    continue
                w = max(10, max(len(l) for _m, l, *_r in out) + 2)
                print(f"\n  n_fft {tag[si]}")
                print(f"{'param':<{w}}{'id_lin':>9}{'id_log':>9}{'lin-log':>10}"
                      f"{'se':>8}{'t':>7}{'lin wins':>10}{'n':>6}")
                for mu, l, fl, fg, se, wins, n in sorted(out, reverse=True):
                    t = mu / se if se > 0 else float("nan")
                    ts = "     -" if t != t else f"{t:>7.1f}"
                    print(f"{l:<{w}}{fl:>9.3f}{fg:>9.3f}{mu:>+10.4f}{se:>8.4f}{ts}"
                          f"{100 * wins:>9.0f}%{n:>6}")
        print("\n  id_lin/id_log  concordance: given two candidates differing "
              "ONLY in this\n"
              "                 parameter, how often does that loss put the "
              "closer one\n"
              "                 first. 0.5 is a coin flip, below 0.5 is "
              "systematically wrong\n"
              "  lin-log        the PAIRED difference. Both losses score the "
              "same candidates\n"
              "                 on the same target, so this is a per-target "
              "quantity and its\n"
              "                 spread over targets is the error bar\n"
              "  se, t          standard error of that difference, and mu/se. "
              "|t| under ~2 is\n"
              "                 a difference this many targets cannot resolve, "
              "whatever its sign\n"
              "  lin wins       share of targets where linear ranked better. "
              "Near 50% with a\n"
              "                 nonzero mean means a few targets carry it, "
              "which is not the\n"
              "                 same finding as a consistent small edge")
        return

    for mr in args.max_rel:
        rows, marg, dropped = sweep(None, mr)
        if not any(rows):
            raise SystemExit("no usable targets")
        if dropped:
            print(f"  {dropped} non-finite candidate renders dropped")
        if args.floor_db is not None:
            print(f"  log floor: {args.floor_db:g} dB below each target's peak")
        # Every set below is scored on the SAME targets and the SAME candidates
        # -- they came out of one sweep -- so differences across these tables
        # carry no sampling noise at all and are the resolution, exactly.
        for si, (rset, mset) in enumerate(zip(rows, marg)):
            if not rset:
                continue
            bi.report(bi.accumulate(rset),
                      title=f"plate / {PARAM_SPACE}   n_fft {tag[si]}   "
                            f"radii <= {mr:g}   {len(rset)} targets")
            bi.report_marginal(mset, title=f"n_fft {tag[si]}")


if __name__ == "__main__":
    main()
