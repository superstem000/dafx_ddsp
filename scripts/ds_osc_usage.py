"""What each arm DOES with its two oscillators: ratio, balance, waveform.

    python scripts/ds_osc_usage.py --dirs data/juno/moog-minitaur \
        --arms oct_synth_magx_halfw oct_synth_hybridx oct_synth_logx_halfw \
        --n 50 --device cuda:3 --folder-peak 0.5

This asks a different question from every other script here. Not "how good is
the output" but "what did the model build". Listening to the oct arms it
sounded like only the linear one was using BOTH oscillators at an octave while
the others collapsed onto one -- and that is directly readable from the
predicted parameters, so it does not have to stay an impression.

harmor pins osc 1 to f0 and puts osc 2 at f0*MULT, with a per-oscillator
amplitude (sep_amp) and a per-oscillator saw<->square blend. So three things
say how the pair is being used:

  MULT      where osc 2 sits. h2of_fifth draws it from {1.5, 2, 3} -- a
            fifth, an octave, a twelfth, 702 / 1200 / 1902 cents apart -- so
            the p_ columns are the fractions of clips the arm places within
            --tol-cents of each. A model that learned the quantized structure
            puts nearly all its mass there; one that did not shows up in
            `other`. Set --ratios to whatever the dataset actually used
  share2    osc 2's share of the mean control amplitude. Near 0 means the
            second oscillator is switched off and the arm is really a
            one-oscillator model whatever MULT says -- which is the thing to
            check before believing any MULT number. Near 0.5 is a balanced
            pair
  both      per cent of clips where BOTH oscillators carry real level
            (share2 between --both-lo and 1-(--both-lo)) AND MULT is not
            unison. That is the "actually using two oscillators an octave
            apart" count, one number
  mix1/2    each oscillator's saw<->square blend, 0 = saw, 1 = square

Amplitudes are the CONTROL level, not rendered power: harmor's saw and square
profiles do not sum to the same total, and the filter is applied afterwards, so
share2 indicates the split rather than measuring it.

MULT and osc_mix are static_params, so fill_params slices them to the last
frame and they are one value per clip. amplitudes is per-frame and is averaged
over frames, the same convention ds_pitch_error uses for its share column.
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

from diffsynth.processor import SCALE_FNS                # noqa: E402
from ds_eval_folder import audio_files, load_clip, load_model   # noqa: E402
from ds_pitch_error import harmor_of                     # noqa: E402


def predict(model, x: torch.Tensor):
    """(mult, share2, mix1, mix2) per clip, scaled to physical units.

    Scaled through harmor's own param_desc and the same SCALE_FNS table
    Processor.process uses, so a change to the MULT range cannot desync this
    from what the renderer actually does.

    A CONDITIONED SYNTH still needs a value for every parameter it declares as
    conditioning, or fill_params dereferences None. The placeholder below is
    inert rather than merely convenient: estimate_param reads only
    conditioning['audio'], a conditioned parameter fills its OWN key and no
    other, and nothing here renders audio -- so the value cannot reach any
    column this script reports.
    """
    cond = {"audio": x}
    for k in model.synth.fixed_param_names:
        if getattr(model.synth, k) is None:
            cond[k] = torch.ones(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
    est = model.estimate_param(cond)
    dag_in = model.synth.fill_params(est, cond)
    proc, conn = harmor_of(model.synth)

    d = proc.param_desc["f0_mult"]
    mult = SCALE_FNS[d["type"]](dag_in[conn["f0_mult"]],
                                d["range"][0], d["range"][1])
    mult = mult[:, -1, 0].detach().cpu().numpy()

    d = proc.param_desc["osc_mix"]
    mix = SCALE_FNS[d["type"]](dag_in[conn["osc_mix"]],
                               d["range"][0], d["range"][1])
    mix = mix[:, -1, :].detach().cpu().numpy()

    a = dag_in[conn["amplitudes"]].mean(dim=1).detach().cpu().numpy()
    share2 = a[:, 1] / np.maximum(a.sum(axis=1), 1e-12)
    return mult, share2, mix[:, 0], mix[:, 1]


def all_params(model, x: torch.Tensor):
    """{label: per-clip value} for EVERY parameter the synth takes.

    predict() reports four static quantities. In the h2of family static_params
    is [BFRQ, M_OSC, MULT, Q_FILT], so AMP and CUTOFF are emitted once per
    FRAME and appear nowhere in that table -- which is why arms can differ on
    real audio with nothing in it to explain the difference. This walks the dag
    instead and reports all of them, scaled through each processor's own
    param_desc, frame-wise ones summarised as their mean over frames.
    """
    cond = {"audio": x}
    for k in model.synth.fixed_param_names:
        if getattr(model.synth, k) is None:
            cond[k] = torch.ones(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
    dag_in = model.synth.fill_params(model.estimate_param(cond), cond)

    out: dict[str, "np.ndarray"] = {}
    for proc, conn in model.synth.dag:
        for name, key in conn.items():
            d = proc.param_desc.get(name)
            if d is None or d["type"] == "raw" or key not in dag_in:
                continue
            v = dag_in[key]
            if not torch.is_tensor(v):
                continue
            v = SCALE_FNS[d["type"]](v, d["range"][0], d["range"][1])
            v = v.detach().cpu().numpy()
            if v.ndim < 3 or v.shape[-1] == 0:
                continue
            # Mean over frames: static params were already sliced to one frame
            # by fill_params, so this is a no-op for them and the average of
            # the curve for AMP and CUTOFF.
            v = v.mean(axis=1)
            for c in range(v.shape[-1]):
                lab = name if v.shape[-1] == 1 else f"{name}[{c}]"
                out[lab] = v[:, c]
    return out


def spread_lines(name, v, edges, fmt="{:g}"):
    """Percentiles and a bin histogram for one predicted quantity.

    A median hides the shape, and the shape is the question: three arms with
    medians of 1.63, 2.00 and 1.78 could be three tight distributions in
    different places or three wide ones sitting on top of each other, and those
    mean opposite things about whether an ordering is real.
    """
    v = np.asarray(v, dtype=float)
    qs = [5, 10, 25, 50, 75, 90, 95]
    out = [f"    {name:<8}" + "".join(f"p{q}{'':<1}" for q in qs)]
    out[-1] = (f"    {name:<8}" + "".join(f"{'p'+str(q):>8}" for q in qs)
               + f"{'IQR':>9}")
    pc = [np.percentile(v, q) for q in qs]
    out.append(f"    {'':<8}" + "".join(f"{x:>8.2f}" for x in pc)
               + f"{pc[4] - pc[2]:>9.2f}")
    n = len(v)
    cells = []
    for i in range(len(edges) - 1):
        c = int(((v >= edges[i]) & (v < edges[i + 1])).sum())
        cells.append(f"{100.0 * c / max(n, 1):>6.0f}%")
    out.append(f"    {'bins':<8}" + "".join(
        f"{fmt.format(edges[i]) + '-' + fmt.format(edges[i+1]):>7}"
        for i in range(len(edges) - 1)))
    out.append(f"    {'':<8}" + "".join(c.rjust(7) for c in cells))
    return out


def true_mult(files, base_dir, lo, hi):
    """The generator's own MULT for each clip, in natural units, or None.

    WHY THIS EXISTS. The in-domain f0_mult error averages over MULT from 1 to
    8, and everything above 3 is a ratio no instrument plays -- so an arm could
    be better on all realistic material and still lose the aggregate. That is
    exactly the ambiguity between logx sitting HIGHER than magx in domain
    (median 4.91 against 4.56, true 4.5) and LOWER on the Moog (1.84 against
    2.26, true 1.5): either its error points the useful way by luck, or it is
    genuinely better at the low end and the mean is dominated by a region that
    does not matter. Binning by the true value separates those.

    The saved value is the raw 0..1 draw -- Synthesizer.forward seeds `outputs`
    with the dag inputs and never overwrites them -- so it is scaled here the
    same way harmor would.
    """
    out = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        pt = os.path.join(base_dir, "param", stem + ".pt")
        if not os.path.exists(pt):
            return None
        v = torch.load(pt, map_location="cpu", weights_only=False)
        if "harmor_f0_mult" not in v:
            return None
        out.append(float(v["harmor_f0_mult"].reshape(-1)[-1]) * (hi - lo) + lo)
    return np.asarray(out)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dirs", nargs="+", required=True, metavar="DIR")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--ckpt", default="latest.ckpt")
    p.add_argument("--n", type=int, default=50, help="Clips per folder; 0 = all.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--length", type=float, default=4.0)
    p.add_argument("--match", default=None, metavar="REGEX")
    p.add_argument("--folder-peak", type=float, default=None, metavar="P")
    p.add_argument("--ratios", type=float, nargs="+", default=[1.5, 2.0, 3.0],
                   metavar="R",
                   help="The ratios the generator actually draws, so the p_ "
                        "columns match the dataset. h2of_fifth is 1.5 2 3; "
                        "h2of_oct was 1 2 4; pass nothing meaningful for the "
                        "published uniform data, where every p_ column and "
                        "`other` are noise and only share2 and both apply.")
    p.add_argument("--tol-cents", type=float, default=50.0, metavar="C",
                   help="How close to a listed ratio a predicted MULT has to "
                        "be to count as it. The fifth, octave and twelfth are "
                        "702 / 1200 / 1902 cents apart, so 50 separates them "
                        "with room to spare.")
    p.add_argument("--spread", action="store_true",
                   help="Percentiles and a histogram of the predicted MULT and "
                        "share2 per arm, instead of medians alone. Three arms "
                        "with medians 1.63, 2.00 and 1.78 could be three tight "
                        "distributions or three wide overlapping ones, and "
                        "those mean opposite things about the ordering.")
    p.add_argument("--true-mult", action="store_true",
                   help="For a GENERATED folder with a param/ sibling: read "
                        "each clip's true MULT and report the error against "
                        "it, binned by that true value. The aggregate f0_mult "
                        "error averages over 1..8, and everything above 3 is a "
                        "ratio no instrument plays -- so an arm can be better "
                        "on all realistic material and still lose the mean.")
    p.add_argument("--mult-bins", type=float, nargs="+",
                   default=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0], metavar="EDGE",
                   help="Bin edges on the TRUE ratio. The default splits out "
                        "unison-to-fifth, fifth-to-octave and octave-to-twelfth "
                        "-- where every real interval lives -- from the rest.")
    p.add_argument("--both-lo", type=float, default=0.2, metavar="F",
                   help="Minimum share for an oscillator to count as in use, "
                        "for the `both` column.")
    p.add_argument("--dump", action="store_true",
                   help="Print EVERY parameter the synth takes, per arm, "
                        "median over clips and in physical units -- including "
                        "AMP and CUTOFF, which are per-frame and so appear in "
                        "no other column here.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--csv", default=None, metavar="PATH")
    args = p.parse_args()

    rng = random.Random(args.seed)
    folders = {}
    truth: dict[str, "np.ndarray"] = {}
    for d in args.dirs:
        files = audio_files(d)
        if args.match:
            files = [f for f in files if re.search(args.match, os.path.basename(f))]
        if args.n and len(files) > args.n:
            files = rng.sample(files, args.n)
        files.sort()
        raw = [load_clip(f, args.sr, args.length) for f in files]
        gain = 1.0
        if args.folder_peak is not None:
            nz = [r[2] for r in raw if r[2] > 0]
            med = float(np.median(nz)) if nz else 0.0
            gain = args.folder_peak / med if med > 0 else 1.0
        g = os.path.basename(os.path.normpath(d))
        folders[g] = (files, np.stack([r[0] for r in raw]) * gain)
        print(f"{g:<26}{len(files):>4} clips   gain {gain:.3f}")
        if args.true_mult:
            tm = true_mult(files, d, 1.0, 8.0)
            if tm is None:
                raise SystemExit(
                    f"--true-mult needs {d}/param/<stem>.pt carrying "
                    f"harmor_f0_mult; this looks like a real-audio folder.")
            truth[g] = tm
            print(f"{'':<26}true MULT median {np.median(tm):.2f}, "
                  f"range {tm.min():.2f}-{tm.max():.2f}")

    heads = "".join(f"{'p_' + f'{r:g}':>7}" for r in args.ratios)
    print(f"\n{'folder':<20}{'arm':<24}{'n':>4}{'MULT_med':>10}{heads}"
          f"{'other':>7}{'share2':>9}{'both':>7}{'mix1':>7}{'mix2':>7}")
    rows = []
    dumps: dict[str, dict[str, dict[str, float]]] = {}
    bins_acc: dict[str, dict[int, "np.ndarray"]] = {}
    # Counted once from the targets, not inside the arm loop -- every arm sees
    # the same clips, so accumulating there multiplied each count by len(arms).
    bins_n: dict[int, int] = {}
    for _tm in truth.values():
        for _i in range(len(args.mult_bins) - 1):
            _m = ((_tm >= args.mult_bins[_i]) & (_tm < args.mult_bins[_i + 1]))
            bins_n[_i] = bins_n.get(_i, 0) + int(_m.sum())
    for arm in args.arms:
        model, _cfg, note = load_model(os.path.join(args.root, arm),
                                       args.ckpt, args.device)
        if model is None:
            print(f"{'':<20}{arm:<24}  skipped -- {note}")
            continue
        _, _conn = harmor_of(model.synth)
        if _conn["f0_mult"] in model.synth.fixed_param_names:
            print(f"{'':<20}{arm:<24}  MULT is CONDITIONING for this synth -- "
                  f"the MULT and p_ columns would report the placeholder, not "
                  f"a prediction. Skipped.")
            continue
        for g, (files, x) in folders.items():
            with torch.no_grad():
                mult, share2, mix1, mix2 = predict(
                    model, torch.from_numpy(x).float().to(args.device))
            near, claimed = [], np.zeros(len(mult), dtype=bool)
            for r in args.ratios:
                m = np.abs(1200.0 * np.log2(mult / r)) <= args.tol_cents
                near.append(100.0 * float(m.mean()))
                claimed |= m
            other = 100.0 * float((~claimed).mean())
            both = 100.0 * float(((share2 >= args.both_lo)
                                  & (share2 <= 1.0 - args.both_lo)
                                  & (mult >= 1.5)).mean())
            if args.dump:
                with torch.no_grad():
                    ap = all_params(model, torch.from_numpy(x).float()
                                    .to(args.device))
                dumps.setdefault(g, {})[arm] = {k: float(np.median(v))
                                                for k, v in ap.items()}
            print(f"{g[:20]:<20}{arm:<24}{len(mult):>4}{np.median(mult):>10.2f}"
                  + "".join(f"{v:>6.0f}%" for v in near)
                  + f"{other:>6.0f}%{np.median(share2):>9.2f}{both:>6.0f}%"
                  f"{np.median(mix1):>7.2f}{np.median(mix2):>7.2f}")
            if args.true_mult and g in truth:
                tm = truth[g]
                cents = 1200.0 * np.log2(np.maximum(mult, 1e-9) / tm)
                per = bins_acc.setdefault(arm, {})
                for i in range(len(args.mult_bins) - 1):
                    m = ((tm >= args.mult_bins[i])
                         & (tm < args.mult_bins[i + 1]))
                    if m.any():
                        per[i] = np.concatenate([per.get(i, np.array([])),
                                                 cents[m]])
            if args.spread:
                print(f"  {arm} / {g[:24]}   predicted MULT")
                for ln in spread_lines("MULT", mult,
                                       [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6, 8]):
                    print(ln)
                for ln in spread_lines("share2", share2,
                                       [0, .05, .1, .2, .3, .5, .7, .8, .9, 1.0],
                                       fmt="{:.2f}"):
                    print(ln)
                print()
            for i, f in enumerate(files):
                rows.append([arm, g, os.path.basename(f), f"{mult[i]:.3f}",
                             f"{share2[i]:.3f}", f"{mix1[i]:.3f}",
                             f"{mix2[i]:.3f}"])

    for g, per_arm in dumps.items():
        keys = sorted({k for d in per_arm.values() for k in d})
        w = max(9, max(len(k) for k in keys) + 2)
        print(f"\n=== every predicted parameter, median over clips, {g}")
        print(f"{'arm':<24}" + "".join(f"{k:>{w}}" for k in keys))
        for arm, d in per_arm.items():
            print(f"{arm:<24}" + "".join(
                f"{d.get(k, float('nan')):>{w}.3f}" for k in keys))
        print("  Physical units, through each processor's own param_desc.\n"
              "  amplitudes and cutoff are per-frame in this synth and are\n"
              "  averaged over frames; the static ones are already one value\n"
              "  per clip, so the average is a no-op for them.")

    print("\n  MULT_med  median predicted ratio of osc 2 to osc 1\n"
          "  p_<r>     per cent of clips within --tol-cents of each --ratios value.\n"
          "            The quantized datasets contain only those, so a trained\n"
          "            arm should spread evenly over them with little `other`\n"
          "  share2    osc 2's share of the mean control amplitude. Near 0 means\n"
          "            the second oscillator is OFF and the arm is really a\n"
          "            one-oscillator model whatever MULT says -- check this\n"
          "            before reading any MULT column\n"
          "  both      per cent of clips using both oscillators (each above\n"
          "            --both-lo) at a non-unison ratio. The one number for\n"
          "            'is this arm actually building an octave pair'\n"
          "  mix1/2    saw<->square blend per oscillator, 0 = saw, 1 = square\n"
          "  Control level, not rendered power: the saw and square profiles do\n"
          "  not sum alike and the filter comes after, so share2 indicates the\n"
          "  split rather than measuring it.")

    if args.true_mult and bins_acc:
        print(f"\n=== MULT against the TRUE value, by true ratio "
              f"(median |error| in cents; 702 = a fifth out)")
        edges = args.mult_bins
        labs = [f"{edges[i]:g}-{edges[i+1]:g}" for i in range(len(edges) - 1)]
        print(f"{'arm':<24}" + "".join(f"{l:>11}" for l in labs) + f"{'all':>11}")
        for arm, per in bins_acc.items():
            cells = []
            for i in range(len(edges) - 1):
                v = per.get(i)
                cells.append(f"{np.median(np.abs(v)):>11.0f}" if v is not None
                             and len(v) else f"{'-':>11}")
            allv = np.concatenate([v for v in per.values() if v is not None
                                   and len(v)]) if per else np.array([])
            cells.append(f"{np.median(np.abs(allv)):>11.0f}"
                         if allv.size else f"{'-':>11}")
            print(f"{arm:<24}" + "".join(cells))
        print(f"{'n per bin':<24}" + "".join(
            f"{bins_n.get(i, 0):>11}" for i in range(len(edges) - 1))
            + f"{sum(bins_n.values()):>11}")
        print("\n  Every interval a real synth plays is below 3 -- unison,\n"
              "  fifth (1.5), octave (2), twelfth (3). If one arm wins the low\n"
              "  bins and loses the high ones, the aggregate f0_mult error is\n"
              "  dominated by ratios nothing plays and the range is the problem,\n"
              "  not the loss. If one arm wins everywhere, it simply wins.")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["arm", "folder", "file", "mult", "share2", "mix1", "mix2"])
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
