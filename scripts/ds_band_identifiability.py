"""Per-band identifiability on diffsynth. See src/analysis/band_identifiability.

    python scripts/ds_band_identifiability.py \
        --conf external/diffsynth/configs/synth/h2of.yaml
    python scripts/ds_band_identifiability.py --conf ... --floor-db 100

The plate's counterpart is src/ddsp/diag_band_identifiability, and the analysis
underneath both is the same module, so the two tables are directly comparable.
That is the entire point of running it here: the plate has 65-71% of its bins
below -120 dB and every compressed arm collapsed to a constant, while diffsynth
has 19.4% and its compressed arms are competitive. Those two facts want a
mechanism connecting them, and the mechanism is whether the bins compression
up-weights vote correctly or at random.

Parameters are already normalized to [0,1] -- fill_params takes [B, frames, P]
in that range and it IS the search space -- so a radius here is a fraction of
range with no per-parameter bounds table and no convention to pick.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "diffsynth"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf                                    # noqa: E402
from diffsynth.modelutils import construct_synth_from_conf          # noqa: E402
from gen_dataset import draw_slots                                  # noqa: E402

from src.analysis.band_sensitivity import EPS, stft_mag             # noqa: E402
from src.analysis import band_identifiability as bi                 # noqa: E402


def _sets(tokens, what):
    """['1024', '512,1024,2048'] -> [[1024], [512, 1024, 2048]]."""
    out = []
    for t in tokens:
        vals = [int(v) for v in t.split(",") if v.strip()]
        if not vals:
            raise SystemExit(f"{what}: empty set in {t!r}")
        out.append(vals)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--conf", default=None,
                    help="A MODEL synth config. Draws come from harmor's own "
                         "param_desc and every clip renders with flat envelopes, "
                         "so prefer --dataset-conf; kept for configs that have "
                         "no generator.")
    ap.add_argument("--n", type=int, default=24, help="Targets")
    ap.add_argument("--k", type=int, default=32, help="Candidates per target")
    ap.add_argument("--max-rel", type=float, nargs="+", default=[0.30],
                    metavar="R",
                    help="Radii uniform in (0, this] as a fraction of range, "
                         "along random directions. A spread, not a fixed "
                         "radius -- concordance needs candidates at different "
                         "true distances to have anything to rank. Takes a "
                         "LADDER: concordance is not a constant of a "
                         "parameter, it is a function of how far apart the "
                         "candidates are, and a loss can order distant "
                         "candidates well while being noise on close ones -- "
                         "which is the regime an optimiser near a minimum "
                         "actually sits in. Read whether the ranking of "
                         "parameters survives the ladder, not one row of it.")
    ap.add_argument("--pin", nargs="+", default=None, metavar="NAME=V",
                    help="Hold a parameter at V in both target and candidates, "
                         "so it contributes no distance and no difference. The "
                         "operating point is a design variable here as it is in "
                         "ds_param_sensitivity.")
    ap.add_argument("--vary", nargs="+", default=None, metavar="NAME",
                    help="Search ONLY these columns; every other column is held "
                         "at --rest for both target and candidates. This defines "
                         "a TASK, the way PLATE_PARAM_SPACE does on the plate, "
                         "and the task is what the decomposition is a property "
                         "of -- a synthesizer does not have one answer, a chosen "
                         "set of searched parameters does. Combine with --pin to "
                         "put the held columns somewhere other than --rest.")
    ap.add_argument("--rest", type=float, default=0.5, metavar="V",
                    help="Where --vary holds the unsearched columns.")
    ap.add_argument("--per-param", action="store_true",
                    help="One row per searched column instead of one band "
                         "table. Candidates differ from the target in that "
                         "column ONLY, so concordance answers 'can this loss "
                         "tell which candidate is closer in THIS parameter', "
                         "while the rest of the target stays randomly drawn "
                         "per target. That is the difference from --vary, "
                         "which pins the background at --rest and so reports a "
                         "property of one operating point. Every column is "
                         "measured on the same targets, so the rows are paired "
                         "and the ORDER is the result.")
    ap.add_argument("--log-radius", action="store_true",
                    help="Draw candidate radii LOG-uniform over "
                         "--radius-decades below --max-rel instead of uniform "
                         "on (0, max-rel]. Uniform puts one candidate in "
                         "fifteen inside 2% of range at the default, so the "
                         "pairs that decide whether a minimum is resolvable "
                         "are a rounding error in the count.")
    ap.add_argument("--radius-decades", type=float, default=2.0, metavar="D",
                    help="How many decades below --max-rel --log-radius "
                         "spans. 2 means radii from max_rel/100 to max_rel.")
    ap.add_argument("--hard-ratio", type=float, default=None, metavar="F",
                    help="ALSO report concordance over pairs whose true "
                         "distances are within a factor F of each other, e.g. "
                         "1.5. A pair at 0.02 against one at 0.28 is ordered "
                         "correctly by anything and says nothing about "
                         "resolving a minimum; the hard pairs are the ones an "
                         "optimiser near one actually faces, and they are a "
                         "minority of the pair count unless asked for "
                         "separately.")
    ap.add_argument("--draw", nargs="+", default=None, metavar="NAME=LO:HI",
                    help="Sample this column on [LO, HI] (normalized 0-1) "
                         "instead of the full range, and read --max-rel as a "
                         "fraction of THAT span. The dataset configs restrict "
                         "the draw without touching the processor -- h2of_r13 "
                         "draws MULT on (1, 3) while harmor keeps (1, 8) for "
                         "every checkpoint's head -- so the full range asks "
                         "about a family the model never saw and inflates "
                         "every radius by the ratio of the two spans. MULT on "
                         "(1, 3) is --draw MULT=0:0.2857.")
    ap.add_argument("--list", action="store_true",
                    help="Print the parameter columns and exit.")
    ap.add_argument("--cond", nargs="+", default=None, metavar="NAME=V",
                    help="Value for a fixed parameter the config leaves to be "
                         "supplied at run time -- f0_hz in the _f0 chains, and "
                         "anything else whose attribute is None. These are NOT "
                         "in [0,1]: they are physical (f0_hz in Hz), which is "
                         "why fill_params skips scaling them. Unset ones get a "
                         "default and the run says which. They are held equal "
                         "across target and candidates, so they contribute no "
                         "distance and no difference -- conditioning, not a "
                         "searched parameter.")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--audio-len", type=float, default=4.0)
    ap.add_argument("--n-fft", nargs="+", default=["1024"], metavar="SET",
                    help="One or more RESOLUTION SETS, each a comma-joined list "
                         "of STFT sizes. A set with more than one size measures "
                         "a MULTI-RESOLUTION loss rather than one spectrogram: "
                         "every bin is weighted exactly as _make_stft_l1 "
                         "weights it, mean over bins within a resolution then "
                         "mean over resolutions, so the reported concordance is "
                         "that loss's concordance. SEVERAL SETS IN ONE RUN "
                         "SHARE THE RENDERS -- targets and candidates depend on "
                         "--seed and not on the STFT, so separate runs would "
                         "re-render identical audio, and the comparison across "
                         "sets is then exactly paired.")
    ap.add_argument("--hop", nargs="+", default=None, metavar="SET",
                    help="One comma-joined set per --n-fft set, same shape. "
                         "Default n_fft//4 throughout, the 75% overlap the "
                         "losses use.")
    ap.add_argument("--eps", type=float, default=None, metavar="E",
                    help="ONE ABSOLUTE floor on every rung, which is what the "
                         "loss does: a single eps applied to each resolution's "
                         "magnitudes unchanged. This is NOT the same experiment "
                         "as --floor-db, and the difference is the point: "
                         "torch.stft is unnormalized, so bins scale with the "
                         "window length and a fixed eps sits deeper on a long "
                         "window than a short one -- the short rungs of an eps "
                         "arm are closer to linear, which --floor-db normalizes "
                         "away by construction. Mutually exclusive with "
                         "--floor-db.")
    ap.add_argument("--norm", default="none", choices=("target", "self", "none"),
                    help="How the two signals are put on a common scale before "
                         "the STFT. none (default) leaves the renders alone, "
                         "which is what diffsynth trains with: SpecWaveLoss is "
                         "constructed without `norm`, so spec_norm is 1.0 and "
                         "no per-example scaling happens anywhere. self divides "
                         "each by its own peak -- what this tool used to do "
                         "unconditionally, and what reproduces tables run "
                         "before this flag existed; it deletes the level "
                         "difference between candidate and target, which is a "
                         "cue the linear term reads. target divides BOTH by the "
                         "target's peak, which is what the PLATE trains with "
                         "(peak_normalized mode='target') -- use it when the "
                         "point is to hold normalization fixed across the two "
                         "systems rather than to model each one's own loss.")
    ap.add_argument("--floor-db", type=float, default=None,
                    help="Set the log measure's floor this far below each "
                         "target's peak instead of at the absolute eps 1e-7. "
                         "With several resolutions the floor is set per "
                         "resolution, against that resolution's own peak -- the "
                         "STFT is unnormalized, so one absolute eps sits at a "
                         "different percentile of the bin distribution on every "
                         "rung and the ladder would confound the floor moving "
                         "with the resolution changing.")
    ap.add_argument("--floors", type=float, nargs="+",
                   default=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
                            90.0, 100.0, 110.0, 120.0, 400.0], metavar="DB",
                   help="Cumulative dB floors, replacing the per-band table. "
                        "Each entry restricts BOTH losses to bins within that "
                        "many dB of the reference peak and ranks candidates with "
                        "what is left, so every row is a real loss over many "
                        "bins rather than a band scored in isolation -- and the "
                        "deepest floor reproduces the full-spectrum numbers, "
                        "which is the check. A band's contribution is then the "
                        "row-to-row difference. 400 means no floor.")
    ap.add_argument("--dataset-conf", type=Path, default=None, metavar="YAML",
                    help="The DATASET config the checkpoints were trained on, "
                         "e.g. configs/synth/dataset/h2of_r13.yaml. This is the "
                         "generator, and using it makes the probe draw and "
                         "render exactly as gen_dataset.py does: the same dag "
                         "with its ADSR envelopes, the same parameters, and the "
                         "same ranges via draw_slots. A model synth instead "
                         "renders flat amplitude and cutoff -- not a member of "
                         "the data distribution -- and takes each parameter's "
                         "bounds from harmor's param_desc, so MULT reads (1, 8) "
                         "on data generated over (1, 3).")
    ap.add_argument("--render-batch", type=int, default=8, metavar="K",
                    help="Render this many candidates at a time. Candidates are "
                         "a batch dimension, so all K at once is a K/8 larger "
                         "transient for identical output. Lower it to share a "
                         "card with a training job.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.conf and not args.dataset_conf:
        raise SystemExit("give --dataset-conf (preferred) or --conf")

    nffts = _sets(args.n_fft, "--n-fft")
    hops = (_sets(args.hop, "--hop") if args.hop
            else [[nf // 4 for nf in S] for S in nffts])
    if len(hops) != len(nffts) or any(len(h) != len(s) for h, s in zip(hops, nffts)):
        raise SystemExit(
            f"--hop must mirror --n-fft exactly: "
            f"{[len(s) for s in nffts]} sizes against {[len(h) for h in hops]} hops")
    # Every (size, hop) pair appearing in ANY set, transformed once per target.
    # The sets overlap heavily by design, and recomputing a shared size per set
    # is the same waste as re-rendering, one level down.
    uniq = sorted({(nf, hp) for S, H in zip(nffts, hops) for nf, hp in zip(S, H)})
    tag = ["+".join(str(n) for n in S) for S in nffts]

    if args.eps is not None and args.floor_db is not None:
        raise SystemExit(
            "--eps and --floor-db are two different experiments and cannot "
            "both apply. --eps is one absolute floor on every rung, which is "
            "what the loss does; --floor-db puts the floor at a fixed DEPTH "
            "below each rung's own peak, which deliberately removes the "
            "resolution dependence --eps exists to expose.")
    # Where the floor actually lands. Printed rather than assumed: whether an
    # eps arm is really a log arm is the question of where eps sits in the bin
    # distribution, and that is not knowable from the eps value alone.
    scale: dict = {}

    def report_scale():
        if not scale:
            return
        print("\n=== reference scale per resolution"
              + ("   (floor = --eps, absolute, as the loss applies it)"
                 if args.eps is not None else ""))
        print(f"{'n_fft':>7}{'hop':>7}{'ref peak':>12}"
              f"{'floor below peak':>19}{'ref bins under':>16}")
        for nf, hp in sorted(scale):
            pk_l, fr_l = scale[(nf, hp)]
            pk = sum(pk_l) / len(pk_l)
            if args.eps is not None and fr_l:
                dbb = 20.0 * math.log10(max(pk, 1e-300) / args.eps)
                print(f"{nf:>7}{hp:>7}{pk:>12.4g}{dbb:>16.1f} dB"
                      f"{100 * sum(fr_l) / len(fr_l):>15.1f}%")
            else:
                print(f"{nf:>7}{hp:>7}{pk:>12.4g}{'-':>19}{'-':>16}")

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # THE GENERATOR, NOT THE MODEL SYNTH, when a dataset config is given.
    #
    # The two are different dags and the difference is the whole point. The
    # model synth consumes amplitude and cutoff as per-frame curves; the
    # generator PRODUCES those curves from ADSR controls, and it is the
    # generator that defines what the training data actually is -- which
    # parameters vary, over what ranges, and with what envelopes. Drawing from
    # the model synth instead renders every clip with a flat amplitude and a
    # flat cutoff, which is not a member of the data distribution, and reads
    # each parameter's bounds off harmor's param_desc rather than off the
    # dataset -- so MULT comes back (1, 8) on data generated over (1, 3).
    #
    # Here the draws come from draw_slots, the same function gen_dataset.py
    # uses, so the ranges are the dataset's by construction rather than by a
    # flag the caller has to remember.
    src = args.dataset_conf or args.conf
    conf = OmegaConf.merge(OmegaConf.create({"data": {"sample_rate": args.sr}}),
                           OmegaConf.load(src))
    synth = construct_synth_from_conf(conf).to(dev)
    gen_mode = args.dataset_conf is not None
    names = list(synth.ext_param_sizes.keys())
    sizes = [synth.ext_param_sizes[k] for k in names]
    label = [n if s == 1 else f"{n}[{j}]"
             for n, s in zip(names, sizes) for j in range(s)]
    P = len(label)
    n_samples = int(args.audio_len * args.sr)

    if args.list:
        print(f"{Path(src).name}   {P} columns:")
        for l in label:
            print(f"  {l}")
        return

    # --vary first so an explicit --pin can override where a held column sits.
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
    if gen_mode:
        cdict = OmegaConf.to_container(conf)
        quant = cdict.get("quantize_params") or {}
        if quant:
            raise SystemExit(
                f"{Path(src).name} quantizes {', '.join(sorted(quant))} to a "
                f"discrete SET of values. Candidates here are drawn from a "
                f"continuous interval and cannot represent that -- the hull "
                f"samples the dead ground between the legal values, a region "
                f"the data never contains. Pin it with --pin, or probe a "
                f"continuous dataset.")
        for key, offset, size, kind, payload in draw_slots(
                synth, {}, cdict.get("range_params") or {}):
            if kind != "range":
                continue
            lo, hi = payload
            for j in range(size):
                draw[offset + j] = (float(lo), float(hi))
            print(f"dataset range: {key} on {list(cdict['range_params'][key])} "
                  f"-> draws on [{lo:.4f}, {hi:.4f}] of its 0..1 space")

    for item in args.draw or []:
        k, _, interval = item.partition("=")
        if k not in label:
            raise SystemExit(f"unknown parameter {k!r}; have: {', '.join(label)}")
        lo, _, hi = interval.partition(":")
        lo, hi = float(lo), float(hi)
        if not 0.0 <= lo < hi <= 1.0:
            raise SystemExit(f"--draw {k}: need 0 <= LO < HI <= 1, got {lo}:{hi}")
        draw[label.index(k)] = (lo, hi)

    # CONDITIONED COLUMNS: drawn per target, held constant across that
    # target's candidates. The generator must sample BFRQ -- it cannot make
    # data otherwise -- but under an _f0only model the estimator never predicts
    # it, so candidates that differ in pitch are measuring a failure the task
    # design exists to remove. Which columns those are is in the MODEL config's
    # fixed_params, not the dataset's, so both configs are needed: the dataset
    # says how to generate, the model says what is estimated.
    hold = set()
    if gen_mode and args.conf:
        mconf = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create({"data": {"sample_rate": args.sr}}),
                            OmegaConf.load(args.conf)))
        for k, v in (mconf.get("fixed_params") or {}).items():
            if v is not None:
                continue
            if k not in synth.ext_param_sizes:
                raise SystemExit(
                    f"{Path(args.conf).name} conditions {k!r}, which "
                    f"{Path(src).name} does not generate. The model and the "
                    f"dataset do not match.")
            off, size = 0, synth.ext_param_sizes[k]
            for key, sz in synth.ext_param_sizes.items():
                if key == k:
                    break
                off += sz
            hold.update(range(off, off + size))
        if hold:
            print("conditioned (drawn per target, equal across its candidates): "
                  + ", ".join(label[i] for i in sorted(hold)))

    searched = [l for i, l in enumerate(label)
                if i not in pins and i not in hold]
    if not searched:
        raise SystemExit("every column is held; nothing is being searched")
    print(f"{Path(src).name}   {P} columns, {len(searched)} searched   "
          f"{args.n} targets   {args.k} candidates each   "
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

    # Some chains leave a fixed parameter to be supplied at run time -- the _f0
    # configs take f0_hz from the dataset -- and fill_params reads it out of
    # `conditioning`, which defaults to None and then raises TypeError on
    # subscript. Supply a constant for each so the chain can be measured at all.
    # Constant is the right choice here and not a shortcut: conditioning is not
    # a searched parameter, so holding it equal across target and candidates is
    # exactly what it should contribute -- nothing.
    #
    # NO SILENT DEFAULT for an unrecognized name. The first version handed 1.0
    # to anything it did not know and merely printed a warning, and BFRQ -- a
    # base frequency in Hz -- got 1.0, so two whole configs were measured with a
    # 1 Hz oscillator. Their spectra came back 92-95% below -120 dB with 80%+ of
    # the linear loss in the top band, which is what a DC signal looks like, and
    # the tables were read as a result before the warning was. A wrong physical
    # conditioning value does not fail, it just produces a plausible table about
    # a synthesizer nobody is studying. Refuse instead.
    _DEFAULT_COND = {"f0_hz": 220.0, "BFRQ": 220.0}
    need = [n for n in synth.fixed_param_names if getattr(synth, n) is None]
    given = {}
    for item in args.cond or []:
        k, v = item.split("=")
        # REFUSE A NAME THAT CONDITIONS NOTHING. fixed_param_names is
        # list(fixed_params.keys()) from the model config, so h2of_f0only needs
        # BFRQ and not f0_hz -- and an unrecognised key used to be accepted and
        # then ignored, leaving the parameter on the _DEFAULT_COND fallback.
        # That is how a run intended at 130.81 Hz was measured at 220: the value
        # was printed, and nothing read it.
        if k not in need:
            raise SystemExit(
                f"--cond {k}: {Path(src).name} does not leave {k!r} to be "
                f"supplied. It conditions "
                f"{', '.join(need) if need else 'nothing'}."
                + (f" Did you mean --cond {need[0]}={v}?" if need else ""))
        given[k] = float(v)
    missing = [n for n in need if n not in _DEFAULT_COND and n not in given]
    if missing:
        raise SystemExit(
            f"{Path(src).name} leaves {', '.join(missing)} to be supplied "
            f"at run time and there is no default for it.\nThese are PHYSICAL "
            f"values, not [0,1] -- fill_params does not scale them -- so a guess "
            f"is not safe.\nSet it explicitly, e.g. --cond {missing[0]}=220")
    cond_v = {n: given.get(n, _DEFAULT_COND.get(n)) for n in need}
    if need:
        print("conditioning: "
              + ", ".join(f"{k}={v:g}" for k, v in cond_v.items())
              + "   (held equal across target and candidates)")
        fell_back = [n for n in need if n not in given]
        if fell_back:
            print(f"  NOTE: {', '.join(fell_back)} not given, using the built-in "
                  f"default. The dataset draws it per clip, so this measures "
                  f"ONE operating point rather than the family.")

    def render(p: torch.Tensor):
        """(audio, {save_param: value}) -- the second is the training target.

        gen_dataset.py writes exactly output[dag_summary[k]] for k in
        save_params, and the trainer's param_loss compares the estimator
        against those. Returning them here is what lets parameter distance be
        measured in the space the objective uses rather than in the space the
        generator happens to be drawn from.
        """
        out, par = [], []
        for i in range(0, p.shape[0], args.render_batch):
            chunk = p[i:i + args.render_batch, None, :].to(dev)
            cond = {k: torch.full((chunk.shape[0], 1, 1), v, device=dev)
                    for k, v in cond_v.items()}
            with torch.no_grad():
                audio, o = synth(synth.fill_params(chunk, cond), n_samples)
            out.append(audio)
            par.append({k: o[synth.dag_summary[k]].detach().cpu()
                        for k in save_keys})
        return (torch.cat(out, dim=0),
                {k: torch.cat([q[k] for q in par], dim=0) for k in save_keys})

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

    # Each parameter's own draw width, so --max-rel is a fraction of THAT
    # rather than of the raw 0..1 scale. 1.0 for anything unrestricted.
    span = torch.ones(P)
    for i, (lo, hi) in draw.items():
        span[i] = hi - lo

    # save_params: what gen_dataset writes and what param_loss compares.
    save_keys = list(OmegaConf.to_container(conf).get("save_params") or [])
    if gen_mode and not save_keys:
        raise SystemExit(
            f"{Path(src).name} has no save_params, so there is no statement of "
            f"what the estimator is trained to predict and no training space to "
            f"measure distance in.")
    held_out = {label[i].split("[")[0] for i in hold}
    skip = {k for k in save_keys
            if synth.dag_summary.get(k) in held_out
            or synth.dag_summary.get(k) in synth.fixed_param_names}

    def param_dist(pc, pt):
        """param_loss between each candidate and the target, same convention.

        L1 per key, mean over everything but the batch, summed, then divided by
        the number of save_params keys -- skipped and empty entries still count
        in the denominator, exactly as EstimatorSynth.param_loss does.
        """
        n = next(iter(pc.values())).shape[0]
        tot = torch.zeros(n, dtype=torch.float64)
        for k in save_keys:
            if k in skip or pt[k].numel() == 0:
                continue
            a = pc[k].double()
            b = pt[k].double().expand_as(a)
            tot += (a - b).abs().flatten(1).mean(1)
        return tot / max(len(save_keys), 1)

    if gen_mode:
        used = [k for k in save_keys if k not in skip]
        print(f"parameter distance over save_params: {', '.join(used)}"
              + (f"   (skipped: {', '.join(sorted(skip))})" if skip else ""))


    def sweep(axis=None, max_rel=0.30):
        # SAME SEED FOR EVERY AXIS. Each parameter is measured on the identical
        # 24 targets, so the per-parameter rows are paired rather than three
        # separate experiments -- a difference between two rows is the parameter,
        # not a different draw of backgrounds.
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
                # EVERY COORDINATE MOVES BY THE SAME FRACTION OF ITS OWN RANGE.
                # --max-rel is a fraction of range, and a restricted parameter's
                # range is the restricted one. Offsetting on the raw 0..1 scale
                # and clamping afterwards -- which is what this did -- gives a
                # parameter drawn on a 0.29-wide interval a typical offset of
                # 0.19, two thirds of its whole span, so most candidates pin to
                # one of its bounds and the parameter goes near-degenerate.
                # Scaling by the span is also what the per-axis branch below
                # already does, and what the header claims is happening.
                d = torch.randn((args.k, P), generator=g)
                d /= d.norm(dim=1, keepdim=True).clamp(min=1e-30)
                r = _mag(args.k, max_rel, g)[:, None]
                cand = tgt[None, :] + d * r * span[None, :]
                for i in range(P):
                    lo, hi = draw.get(i, (0.0, 1.0))
                    cand[:, i] = cand[:, i].clamp(lo, hi)
            else:
                # ONE AXIS, RANDOM BACKGROUND. The candidates differ from the target
                # in this column and nothing else, so dist IS |delta p| and the
                # concordance is "can this loss tell which candidate is closer in p".
                # The rest of the target stays randomly drawn per target, which is
                # what --vary gives up: it pins the background at --rest, so every
                # target sits at one operating point and the answer is a property of
                # that point rather than of the parameter.
                #
                # DRAWN INSIDE THE BOUNDS, NOT CLAMPED TO THEM. A clamp maps every
                # over-the-edge candidate onto the boundary VALUE, so they become
                # identical patches with identical losses -- ties, counted at 0.5,
                # dragging concordance toward the coin flip for exactly the targets
                # sitting near an edge. Sigmoid parameters put a lot of mass there.
                # Instead: pick a side with probability proportional to the room on
                # that side and draw the magnitude within it, which is uniform over
                # the feasible offsets and produces no duplicates.
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
            for i in hold:
                cand[:, i] = tgt[i]
            x_ref, p_ref = render(tgt[None, :])
            # Strip the batch dim the single-target render carries. The old
            # call was render(...)[0] on a bare tensor; unpacking the tuple
            # kept the [1, n] shape and stft then saw a 3-D input.
            x_ref = x_ref[0]
            x_can, p_can = render(cand)
            # PARAMETER DISTANCE IN THE TRAINING SPACE, not in the space the
            # generator is drawn from. The estimator never sees PEAK_A or AT_C;
            # it predicts the rendered curves those controls produce, and
            # param_loss is an L1 over save_params divided by the key count,
            # skipping conditioned and zero-width entries. Measuring distance
            # over the draw vector instead scores the losses against a
            # different quantity from the one being optimised, with no reason
            # for the two to be monotone in each other.
            dist = param_dist(p_can, p_ref)
            ok = torch.isfinite(x_can).all(dim=-1)
            if not bool(ok.all()):
                dropped += int((~ok).sum())
                x_can, dist = x_can[ok], dist[ok.cpu()]
            if x_can.shape[0] < 4 or not torch.isfinite(x_ref).all():
                continue

            # The normalization the LOSS applies, before the transform, not
            # stft_mag's own per-signal one -- see --norm.
            tp = x_ref.abs().max().clamp(min=1e-30)
            if args.norm == "target":
                xr, xc = x_ref / tp, x_can / tp
            elif args.norm == "self":
                xr = x_ref / tp
                xc = x_can / x_can.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
            else:
                xr, xc = x_ref, x_can
            # Once per (size, hop), not once per set: the sets share sizes.
            cache = {(nf, hp): (stft_mag(xr[None, :], nf, hp, False)[0],
                                stft_mag(xc, nf, hp, False))
                     for nf, hp in uniq}
            for (nf, hp), (Ar, _Ac) in cache.items():
                st = scale.setdefault((nf, hp), ([], []))
                st[0].append(float(Ar.max()))
                if args.eps is not None:
                    st[1].append(float((Ar < args.eps).double().mean()))
            dt = dist.to(next(iter(cache.values()))[0].device)
            for si, (S, H) in enumerate(zip(nffts, hops)):
                A_ref = [cache[(nf, hp)][0] for nf, hp in zip(S, H)]
                A_can = [cache[(nf, hp)][1] for nf, hp in zip(S, H)]
                if args.eps is not None:
                    eps = [args.eps] * len(S)
                elif args.floor_db is not None:
                    eps = [float(A.max()) * 10.0 ** (-args.floor_db / 20.0)
                           for A in A_ref]
                else:
                    eps = EPS
                rows[si].append(bi.cumulative(A_ref, A_can, dt, eps, args.floors))
                marg[si].append(bi.marginal(A_ref, A_can, dt, eps,
                                            args.hard_ratio))
        return rows, marg, dropped

    if args.per_param:
        if args.vary:
            raise SystemExit(
                "--per-param and --vary are incompatible: --vary pins the "
                "unsearched columns at --rest, which is the single operating "
                "point --per-param exists to average over. Use --pin for "
                "columns that are genuinely held (conditioning, say).")
        for mr in args.max_rel:
            how = (f"log-uniform over {args.radius_decades:g} decades "
                   f"below {mr:g}" if args.log_radius
                   else f"uniform on (0, {mr:g}]")
            pairs = (f"pairs within {args.hard_ratio:g}x in distance"
                     if args.hard_ratio else "all pairs")
            print(f"\n=== PER PARAMETER   radii {how}   {pairs}\n"
                  f"    candidates differ in ONE column, background "
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
            report_scale()
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
        report_scale()
        # Every set below is scored on the SAME targets and the SAME candidates
        # -- they came out of one sweep -- so differences across these tables
        # carry no sampling noise at all and are the resolution, exactly.
        for si, (rset, mset) in enumerate(zip(rows, marg)):
            if not rset:
                continue
            bi.report_cum(bi.accumulate_cum(rset),
                          title=f"{Path(src).stem}   n_fft {tag[si]}   "
                            f"radii <= {mr:g}   {len(rset)} targets")
            bi.report_marginal(mset, title=f"n_fft {tag[si]}")


if __name__ == "__main__":
    main()
