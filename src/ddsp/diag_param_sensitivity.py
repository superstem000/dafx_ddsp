"""What each plate parameter does to the audio, and in which level region.

    python -m src.ddsp.diag_param_sensitivity --data-dir data/val-p99 --n 64
    python -m src.ddsp.diag_param_sensitivity --detail --only T60_DC T0 rho

WHY. The compression argument is about WHERE a loss puts its weight: linear
weights absolute error, so it is dominated by the loudest bins; log weights
relative error, so a bin 80 dB down counts as much as the peak. That is a claim
about the loss. Whether it matters depends on something the loss knows nothing
about -- where each PARAMETER's signature actually lives.

A parameter whose effect sits in the top decile is recoverable under any
compression, and the ladder cannot separate losses on it. A parameter whose
effect sits in the quiet bins is one only a compressed loss can see, and if the
fitted set contains none of those then the whole comparison has been run on the
axis where it is least likely to bite. This measures that per parameter instead
of inferring it from an aggregate.

Seven of the fourteen are fitted (E, rho, h, Ly, T0, op_x, op_y) and seven are
pinned at dataset-generation time (Lx, nu, T60_DC, T60_F1, loss_F1, fp_x, fp_y).
Both are swept: the pinned ones are the candidates for a wider estimation task,
and the question of which to unpin is exactly the question this answers.

  THE THREE DAMPING CONSTANTS ARE THE INTERESTING CASE. T60_DC, T60_F1 and
  loss_F1 set alpha and beta in ModalPlate's frequency-dependent damping, i.e.
  they control how each mode DECAYS. Every fitted parameter controls where modes
  sit or how loud they are; nothing fitted controls decay. In a decaying IR the
  quiet bins are disproportionately the late ones, so if any parameter's
  signature lives where compression matters it should be these.

WHAT IS MEASURED, per parameter, against an unperturbed render of the same IRs:

  total change      L1 on STFT magnitude as a % of saturation, so it is
                    comparable across parameters and against the gt_loss floor.
                    Reported BOTH peak-normalized and raw: the training loss
                    normalizes (--peak-normalize target), but rho, h and Lx act
                    largely THROUGH level -- gain is 1/(0.25*mu*Lx*Ly) -- and
                    normalizing deletes exactly that. One number would hide half
                    of what those parameters do.

  decile centroid   Bins sorted by REFERENCE magnitude into ten equal-count
                    buckets; the centroid is the share-weighted mean bucket, 1
                    quietest to 10 loudest. Computed twice, on |a - b| and on
                    |log(a+eps) - log(b+eps)| -- the first is what a linear loss
                    sees, the second what a log loss sees. The GAP between them
                    is the point: a parameter with a high linear centroid and a
                    low log centroid is one a log loss chases in the wrong place.

  dB band           The same split by fixed distance below the reference peak
                    (0-20, 20-40, 40-60, 60-80, 80+), because deciles are
                    population-relative and the evaluation floor ladder is
                    denominated in dB.

The decomposition follows diag_gt_floor and confirm_f32_gt, which used it for
the numpy-vs-torch gap: same 4096-point Hann STFT at hop 1024, same
log(x + 1e-7). Reusing it means a parameter's signature and a renderer's
disagreement are read on one axis.

PERTURBATION IS A LADDER, default 0.5 / 2 / 5 / 20 / 50%, both signs averaged.
One step size measures a local derivative and an encoder does not start near the
answer, so whether the local picture applies is itself part of the question.

  --step value  a fraction of the parameter's own value; the only convention
                defined for all fourteen, since the pinned seven have no range.
  --step range  a fraction of (hi - lo) for the fitted seven, the space the
                encoder actually searches and the only convention under which
                totals compare across parameters -- T0 spans five decades and Ly
                a factor of 3.6. Pinned parameters fall back to value, marked *.
                Large steps walk off the dataset's bounds and are clamped, with
                the count reported: at 50% every fitted parameter clamps exactly
                half its draws, because one sign is always out of range, so that
                column is a one-sided step and does not compare with the rest.

MEASURED, and it changed what this script is for. log dec is NOT step-invariant
for the damping constants -- T60_DC runs 2.33 / 2.55 / 2.94 / 4.23 / 5.15 across
the ladder while every geometry parameter sits flat near 5.5. A small change in
decay barely alters the attack and compounds over time, so it lands in late quiet
bins; a large change reshapes the envelope including the attack. So "damping
lives where compression helps" is a LOCAL property, true only near the solution.
Far from it -- which is where training starts -- damping looks like everything
else, which is why a compressed loss can only matter after a parameter-loss
handover has already brought the encoder close.

T0 is the exception: 3.33 -> 3.93 across two orders of magnitude of step, low and
flat, and the weakest fitted parameter by a wide margin.

READ log dec AGAINST 5.5, NOT AGAINST 10. A change spread evenly across all ten
deciles gives a centroid of exactly 5.5, so that is the neutral value. The
geometry parameters sit at 5.4-5.9 -- they are UNIFORM, not loud-biased, and both
losses can see them, which is why the compression ladder cannot separate losses
on them. Only T0 (3.4-4.0 at every step) and damping at perturbations too small
to learn from (T60_DC 2.55 at 2%) sit meaningfully below it.

Which settles the question this script was built to ask. Damping at a learnable
step size -- x2/÷2, where it moves the audio 11-20% -- reads 5.4 to 6.05, i.e.
neutral. There is no damping range that is simultaneously wide enough to train on
and narrow enough to keep the signature where only a compressed loss reaches. The
one parameter that IS quiet-biased at a meaningful step is T0, and T0 is already
in the fitted seven.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from src.plate.SevenParamPlate import BatchedModalPlateTorch

# The fitted seven, from fit_7param_norm_es. Only membership is used -- the
# sweep is value-relative so all fourteen are on one footing -- but the bounds
# are kept beside it because "which of these has a search range at all" is the
# question the pinned half exists to answer.
FITTED_BOUNDS = {
    "E": (6.7e10, 2.2e11),
    "rho": (2430.0, 21230.0),
    "h": (0.001, 0.005),
    "Ly": (1.1, 4.0),
    "T0": (0.01, 1000.0),
    "op_x": (0.51, 1.0),
    "op_y": (0.51, 1.0),
}
# Split past 80 dB. A single 80-400 band is unbounded, and in a peak-normalized
# spectrogram of a decaying IR it holds most of the bins -- so a SUM over it wins
# on count alone and every parameter reads ~99% there, which is what made the
# first version of this table unreadable.
DB_BANDS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120), (120, 400)]
EPS = 1e-7


def _mode_grid(text: str):
    """Parse "DDX,DDY". Local rather than imported from make_dataset, which pulls
    in the fitter stack to parse two integers."""
    try:
        x, y = (int(v) for v in text.split(","))
    except Exception:
        raise argparse.ArgumentTypeError(f"expected DDX,DDY (e.g. 86,282), got {text!r}")
    if x < 1 or y < 1:
        raise argparse.ArgumentTypeError("both grid dimensions must be positive")
    return (x, y)


def stft_mag(x: torch.Tensor, n_fft: int, hop: int, normalize: bool) -> torch.Tensor:
    """Peak-normalized or raw magnitude spectrogram, matching diag_gt_floor."""
    if normalize:
        x = x / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    w = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft, hop, window=w, return_complex=True).abs()


def decompose(ref: torch.Tensor, per: torch.Tensor, eps: float = EPS):
    """(linear centroid, log centroid, linear bands, log bands, totals).

    Bins are ordered by the REFERENCE magnitude, not the perturbed one, so the
    buckets mean the same thing for every parameter and every step size.
    """
    a, b = ref.flatten(), per.flatten()
    lin = (a - b).abs()
    # eps is the floor of the log measure: two bins both far below it clamp to
    # log(eps) and register no disagreement at all. At the default 1e-7 that
    # floor sits ~160 dB under the reference peak, i.e. forty dB below where
    # this plate's float32 modal sum stops being physics -- so the log column
    # counts arithmetic. --floor-db sets it where the signal actually is.
    log = (torch.log(a + eps) - torch.log(b + eps)).abs()

    # RELATIVE change per bin -- the direct answer to "does this perturbation
    # move the loud parts or the quiet parts", in percent, with no centroid and
    # no share-of-total in the way. A share says how the change is distributed
    # given how many bins are where; this says how much each bin actually moved.
    rel = lin / (a + eps)

    # NEUTRAL FOR THIS FLOOR. A change that is uniform in RELATIVE terms -- every
    # bin scaled by the same factor -- contributes a*delta/(a+eps) per bin, which
    # is flat only while eps is far below every bin. Raise the floor and the
    # quiet bins are suppressed, so a uniform change no longer scores 5.5 and the
    # centroid drifts upward on its own. This computes where neutral actually
    # sits, from the reference and the eps in force, so the log dec column stays
    # readable at any floor. At the default eps it comes out at 5.5.
    w = a / (a + eps)

    order = torch.argsort(a)
    lin_s, log_s, w_s = lin[order], log[order], w[order]
    n = lin_s.numel()
    edges = [round(i * n / 10) for i in range(11)]
    lin_d = torch.tensor([lin_s[edges[i]:edges[i + 1]].sum() for i in range(10)])
    log_d = torch.tensor([log_s[edges[i]:edges[i + 1]].sum() for i in range(10)])
    w_d = torch.tensor([w_s[edges[i]:edges[i + 1]].sum() for i in range(10)])

    def centroid(d):
        t = d.sum()
        # 1 = quietest decile, 10 = loudest. nan when the parameter did nothing
        # at all, which is itself the answer and must not read as decile 1.
        return float((d * torch.arange(1, 11)).sum() / t) if t > 0 else float("nan")

    rel_db = 20.0 * torch.log10((a / a.max().clamp(min=1e-30)).clamp(min=1e-30))
    lin_b, log_b, cnt_b, rel_b = [], [], [], []
    for lo, hi in DB_BANDS:
        m = (-rel_db >= lo) & (-rel_db < hi)
        lin_b.append(float(lin[m].sum()))
        log_b.append(float(log[m].sum()))
        rel_b.append(float(rel[m].mean()) if int(m.sum()) else 0.0)
        # Bin count per band, so the shares above can be read per bin rather
        # than per band. Bands hold wildly different numbers of bins -- the
        # deciles do not, which is why the centroid was legible while this was
        # not -- and a sum over a band is a statement about how many bins are in
        # it as much as about where a parameter's signature lives.
        cnt_b.append(int(m.sum()))
    return (centroid(lin_d), centroid(log_d), lin_b, log_b,
            float(lin.sum()), float(log.sum()), cnt_b, rel_b, centroid(w_d))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, default=Path("data/val-p99"))
    p.add_argument("--n", type=int, default=64, help="Reference IRs")
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument(
        "--rel", type=float, nargs="+",
        default=[0.005, 0.02, 0.05, 0.2, 0.5],
        help="Perturbation sizes, read according to --step. A LADDER rather "
             "than one value: a single small step measures a local derivative, "
             "and whether the estimation problem is local is itself in "
             "question. Two things to read off it -- dnorm%% should grow "
             "roughly linearly with rel while the response is local and bend "
             "over as it saturates toward the unrelated-IR distance, and "
             "log dec should NOT move, since where a parameter's signature "
             "lives is meant to be a property of the parameter rather than of "
             "how hard it was poked.")
    p.add_argument(
        "--step", default="value", choices=("value", "range", "mul"),
        help="value: --rel as a fraction of the parameter's own value. range: "
             "--rel as a fraction of (hi - lo) for the fitted seven, which is "
             "the space the encoder searches and the only convention under "
             "which totals compare across parameters -- T0 spans five decades "
             "and Ly spans a factor of 3.6, so a 2%% value step means wildly "
             "different fractions of what is actually being estimated. Pinned "
             "parameters have no range and fall back to value, marked * in the "
             "output. mul: GEOMETRIC, v*(1+rel) and v/(1+rel). The only "
             "convention valid past rel=1 for a positive-only quantity -- "
             "value's minus branch reaches zero at rel=1 and goes negative "
             "beyond it, and a negative T60 is a decay time that grows. It is "
             "also the natural step for anything spanning decades, which T0 "
             "and the T60s do.")
    p.add_argument("--n-fft", type=int, default=4096)
    p.add_argument("--hop", type=int, default=1024)
    p.add_argument("--only", nargs="+", default=None, metavar="PARAM")
    p.add_argument("--detail", action="store_true",
                   help="Per-parameter decile and dB-band tables, not just the "
                        "summary row")
    p.add_argument(
        "--vary", nargs="+", default=None, metavar="NAME=LO:HI",
        help="Build the reference family by SAMPLING these ranges with every "
             "other parameter pinned, instead of reading val-p99. This is the "
             "measurement that decides whether a proposed task is learnable, "
             "and it is the one a val-p99 sweep cannot make: dnorm%% there is a "
             "fraction of the distance between IRs differing across five "
             "decades of T0 and the whole geometry space, so a damping nudge is "
             "small against it BY CONSTRUCTION. What matters is the change "
             "relative to the spread the encoder actually has to resolve, which "
             "means saturation computed WITHIN the proposed family. Ranges "
             "spanning more than 20x are sampled log-uniform.")
    p.add_argument(
        "--ratio", default=None, metavar="LO:HI",
        help="Sample T60_F1 = r * T60_DC with r in [LO, HI], rather than "
             "sampling it independently. beta ~ (1/T60_F1 - 1/T60_DC), so "
             "T60_F1 > T60_DC flips beta negative and the high modes grow "
             "without bound -- r < 1 makes that unreachable by construction "
             "instead of by rejection sampling. Perturbing T60_F1 at fixed "
             "T60_DC IS perturbing r, so the ladder needs no special case.")
    p.add_argument(
        "--direction", default=None, metavar="NAME=W,NAME=W",
        help="Probe a COMBINATION of search coordinates instead of one at a "
             "time, e.g. T60_DC=1,T60_ratio=0.6. Weights are normalized to a "
             "unit vector and each coordinate moves (hi-lo)*rel*w, so a "
             "single-coordinate direction reproduces that coordinate's row "
             "exactly. This exists because an encoder's failure is a shrinkage "
             "along a DIRECTION, not a parameter: for one unresolved unit "
             "vector u, spread_k = sqrt(1 - u_k^2), so the logged per-parameter "
             "spreads solve directly for u. Poking that u is the only way to "
             "ask whether the hard direction is the quiet one.")
    p.add_argument(
        "--floor-db", type=float, default=None, metavar="DB",
        help="Floor the log measure DB below the reference peak, instead of at "
             "the default absolute eps of 1e-7 (~160 dB down). This decides "
             "whether a low log dec means the signature is in the quiet region "
             "or merely below the synthesizer's own arithmetic: this plate's "
             "modal sum is float32, and swapping the reduction kernel moves "
             "bins below about -120 dB while leaving everything above -100 dB "
             "alone. Run at -100 and -120 and compare against the default -- if "
             "log dec rises to neutral once the junk is excluded, the quiet "
             "region being measured was numerical.")
    p.add_argument(
        "--pin", default=None, metavar="NAME=V,NAME=V",
        help="Override plate parameters in the reference before anything is "
             "measured, e.g. nu=0.15. Sensitivity is not a property of a "
             "parameter on its own -- it is measured at whatever the other "
             "thirteen are, and moving one moves everyone else's band "
             "distribution. Combine with --vary to ask the design question "
             "directly: at which operating point are the searched parameters "
             "most starved in the loud bands, which is where a compression "
             "ladder has something to separate.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--fixed-mode-grid", type=_mode_grid, default=None, metavar="DDX,DDY",
        help="Pin the modal grid, as the datasets are rendered with. Load-bearing "
             "for any parameter that changes the mode COUNT -- E, rho, h, T0, nu. "
             "Unpinned, the grid follows the batch maximum, so a perturbation that "
             "shifts that maximum makes the reference and the perturbed render sum "
             "a different number of modes in a different order. That lands in the "
             "quietest bins, which is exactly where this tool is reading.")
    p.add_argument("--mode-bucket", type=int, default=1024)
    p.add_argument("--chunk-elems", type=int, default=1_000_000_000)
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    plate = BatchedModalPlateTorch(
        device=dev, batched_modal_sum=True, compile_modal_sum=False,
        chunk_elems=args.chunk_elems, mode_bucket=args.mode_bucket,
    )
    plate.fixed_mode_grid = (None if args.fixed_mode_grid is None
                             else (int(args.fixed_mode_grid[0]), int(args.fixed_mode_grid[1])))
    if args.fixed_mode_grid is None:
        print("WARNING: no --fixed-mode-grid. Any parameter that changes the mode "
              "count\n  renders its reference and its perturbation on different "
              "grids, and that\n  difference lands in the quiet bins this tool "
              "reads. Pin it.\n")

    csvs = sorted(args.data_dir.glob("random_IR_params_*.csv"))[: args.n]
    if not csvs:
        raise SystemExit(f"no random_IR_params_*.csv under {args.data_dir}")
    dicts = [pd.read_csv(c).iloc[0].to_dict() for c in csvs]

    # OVERRIDE THE OPERATING POINT. Where a parameter's signature lives is not a
    # property of that parameter alone -- it is measured at whatever the other
    # thirteen happen to be, and moving one of them moves everyone else's
    # distribution. So the PINNED half is a design variable too: it sets how
    # loud-starved the searched half is, which is the property that decides
    # whether a compression ladder can separate anything at all. This makes that
    # variable addressable, so an operating point can be chosen by measurement
    # rather than inherited from whichever IR happened to be first in the
    # directory.
    if args.pin:
        pins = {}
        for item in args.pin.split(","):
            k, v = item.split("=")
            pins[k.strip()] = float(v)
        bad = [k for k in pins if k not in BatchedModalPlateTorch.PARAM_ORDER]
        if bad:
            raise SystemExit(f"not plate parameters: {', '.join(bad)}")
        dicts = [{**d, **pins} for d in dicts]
        print("operating point: " + ", ".join(f"{k}={v:g}" for k, v in pins.items())
              + "   (overriding the dataset)\n")

    vary_bounds = {}
    if args.vary or args.ratio:
        spec = {}
        for item in args.vary or []:
            k, rng = item.split("=")
            lo, hi = (float(v) for v in rng.split(":"))
            spec[k] = (lo, hi)
        rlo, rhi = ((float(v) for v in args.ratio.split(":")) if args.ratio
                    else (None, None))
        base = dict(dicts[0])
        g = torch.Generator().manual_seed(args.seed)
        fam = []
        for _ in range(args.n):
            d = dict(base)
            for k, (lo, hi) in spec.items():
                u = float(torch.rand(1, generator=g))
                # Log-uniform past 20x: a uniform draw over five decades puts
                # 90% of the mass in the top decade and the low end is never
                # sampled, which is how T0's range behaves in the existing set.
                d[k] = lo * (hi / lo) ** u if hi / lo > 20 else lo + u * (hi - lo)
            if rlo is not None:
                # After T60_DC, since it multiplies it.
                r = rlo + float(torch.rand(1, generator=g)) * (rhi - rlo)
                d["T60_F1"] = r * d["T60_DC"]
            fam.append(d)
        dicts = fam
        # Range-stepping uses what the family actually spans, which also gives
        # T60_F1 a range implied by the ratio without special-casing it.
        for k in list(spec) + (["T60_F1"] if rlo is not None else []):
            vals = [d[k] for d in dicts]
            vary_bounds[k] = (min(vals), max(vals))
        print("varying: " + ", ".join(
            f"{k} [{lo:.4g}, {hi:.4g}]" for k, (lo, hi) in vary_bounds.items())
            + "   everything else pinned")
    how = "of range (fitted only, * = of value)" if args.step == "range" else "of value"
    print(f"{args.data_dir}   {len(dicts)} IRs   perturbations "
          + ", ".join(f"{100*r:g}%" for r in args.rel) + f" {how}, both signs")

    with torch.no_grad():
        ref14 = BatchedModalPlateTorch.params_dicts_to_tensor(dicts, dev)
        # normalize=False: forward() peak-normalizes BY DEFAULT, so leaving
        # it on made the raw and normalized columns identical and silently
        # deleted every level effect this script exists to show.
        x_ref = plate.forward(ref14, args.duration, normalize=False)
        # Saturation: the same L1 against UNRELATED IRs, so a parameter's effect
        # can be read as a fraction of the distance between two different
        # plates. Without it "0.004" is a number with no scale.
        # Seeded. Unseeded, the pairing changed every run and dnorm% moved with
        # it -- 1.6% and 5.1% between runs of the same command, applied as a
        # common factor to every parameter because they share this denominator.
        # That is larger than several of the effects being compared.
        perm = torch.randperm(x_ref.shape[0],
                              generator=torch.Generator().manual_seed(args.seed)).to(dev)
        A_n = stft_mag(x_ref, args.n_fft, args.hop, True)
        A_r = stft_mag(x_ref, args.n_fft, args.hop, False)
        sat_n = float((A_n - A_n[perm]).abs().sum())
        sat_r = float((A_r - A_r[perm]).abs().sum())

    # The log measure's floor, set once from the reference so every probe is
    # measured against the same one. Referenced to the peak rather than given
    # as an absolute, because "how far below the loudest bin" is the question --
    # an absolute eps means different things for a signal at a different level.
    if args.floor_db is None:
        eps_n, eps_r = EPS, EPS
    else:
        f = 10.0 ** (-args.floor_db / 20.0)
        eps_n, eps_r = float(A_n.max()) * f, float(A_r.max()) * f
        print(f"log floor: {args.floor_db:g} dB below the reference peak "
              f"(eps {eps_n:.3g} normalized, {eps_r:.3g} raw). Bins below it "
              f"clamp on both sides and register no disagreement.")

    # LOG-DOMAIN SATURATION, so what a COMPRESSED loss has to work with can be
    # stated in the same units as what a linear one does. dnorm% has always been
    # the linear answer and there was no counterpart, which left "does
    # compression have more to go on here" to be inferred from band ratios --
    # and a ratio rewards a parameter for being invisible everywhere, which is
    # how a window too short to contain any decay came out looking like a good
    # quiet-region task.
    sat_g = float((torch.log(A_n + eps_n) - torch.log(A_n[perm] + eps_n)).abs().sum())

    BOUNDS = dict(FITTED_BOUNDS)
    BOUNDS.update(vary_bounds)
    # The denominator, printed because it is the whole point of --vary. A
    # damping nudge is small against a geometry-varying dataset by construction;
    # what decides learnability is its size against the spread the encoder has
    # to resolve, which is this.
    print(f"saturation within this family: {sat_n:.5g} normalized, "
          f"{sat_r:.5g} raw -- every dnorm% below is a fraction of it\n")
    # PERTURB IN THE SEARCH BASIS, NOT THE PLATE'S COLUMNS. Under --ratio the
    # family holds T60_F1 = r * T60_DC, so no two IRs in it differ in T60_DC
    # alone -- that direction is not in the data, and the encoder never searches
    # it. Poking the plate column by itself therefore measures a direction that
    # does not exist, for the one parameter the ranges were chosen on. Holding r
    # fixed instead scales T60_F1 with it, which at fixed r rescales the whole
    # damping profile in time, since both alpha and beta go as 1/T60_DC.
    #
    # The other two rows need no such treatment: T0 is independent, and
    # perturbing T60_F1 at fixed T60_DC IS perturbing r, which is exactly the
    # third search coordinate.
    I_DC = BatchedModalPlateTorch.PARAM_ORDER.index("T60_DC")
    I_F1 = BatchedModalPlateTorch.PARAM_ORDER.index("T60_F1")
    ratio_of = None
    if args.ratio:
        ratio_of = ref14[:, I_F1] / ref14[:, I_DC].clamp(min=1e-30)
        print("search basis: T60_DC perturbed at FIXED RATIO (T60_F1 scales with "
              "it); T60_F1 perturbed at fixed T60_DC, which is the ratio "
              "coordinate itself\n")
    names = [k for k in BatchedModalPlateTorch.PARAM_ORDER
             if (args.only and k in args.only)
             or (not args.only and (k in vary_bounds if vary_bounds else True))]

    # T60_ratio as a FIRST-CLASS ROW, and arbitrary directions in the search
    # basis. The single-coordinate rows answer "what does one parameter do",
    # which is not the question when the encoder's failure is a shrinkage along
    # a COMBINATION -- spread_k = sqrt(1 - u_k^2) for an unresolved unit
    # direction u, and the observed spreads solve to one consistent u. Poking
    # that u directly is the only way to ask whether the hard direction is quiet.
    #
    # Steps are (hi - lo) * rel * u_k per coordinate with |u| = 1, so a direction
    # that is a single coordinate reproduces that coordinate's row exactly and
    # the numbers stay comparable.
    ratio_bounds = None
    if args.ratio:
        ratio_bounds = tuple(float(v) for v in args.ratio.split(":"))
        # T60_ratio REPLACES the T60_F1 column rather than joining it. The
        # coupling below rewrites T60_F1 from r and T60_DC after every probe, so
        # a probe that moved the T60_F1 column would have its perturbation
        # overwritten and silently report zero. The ratio row measures the same
        # direction and measures it in the coordinate the encoder emits.
        names = [k for k in names if k != "T60_F1"] + ["T60_ratio"]
    dir_u = {}
    if args.direction:
        for item in args.direction.split(","):
            k, w = item.split("=")
            dir_u[k.strip()] = float(w)
        nrm = math.sqrt(sum(w * w for w in dir_u.values())) or 1.0
        dir_u = {k: w / nrm for k, w in dir_u.items()}
        dir_name = "dir(" + ",".join(f"{k}{v:+.2f}" for k, v in dir_u.items()) + ")"
        names = names + [dir_name]
        print("direction probe: " + "  ".join(f"{k} {v:+.3f}" for k, v in dir_u.items())
              + "   (unit vector in normalized search coordinates)\n")
    else:
        dir_name = None

    def search_bounds(k):
        """Bounds of a SEARCH coordinate, which T60_ratio has and no plate column does."""
        if k == "T60_ratio":
            return ratio_bounds
        return BOUNDS.get(k)

    def perturb(p14, k, delta_frac, sign):
        """Move search coordinate k by sign*delta_frac of its range, in place.

        Returns the number of IRs clamped. T60_ratio and T60_DC both write
        T60_F1, so the caller applies the ratio coupling once, after all
        coordinates have moved.
        """
        b = search_bounds(k)
        if k == "T60_ratio":
            if new_r is None:
                raise SystemExit("T60_ratio is a coordinate only under --ratio")
            lo, hi = b
            v = ratio_of + sign * delta_frac * (hi - lo)
            n = int((v.clamp(lo, hi) != v).sum())
            new_r.copy_(v.clamp(lo, hi))
            return n
        j = BatchedModalPlateTorch.PARAM_ORDER.index(k)
        if args.step == "mul":
            f = 1.0 + delta_frac
            p14[:, j] = p14[:, j] * (f if sign > 0 else 1.0 / f)
            return 0
        if args.step == "range" and b is not None:
            lo, hi = b
            v = p14[:, j] + sign * delta_frac * (hi - lo)
            # A large step walks off the range the dataset was drawn from, which
            # would render plates no target ever contained. Clamped, and counted,
            # so the ladder's top rungs are read knowing how much hit the wall.
            n = int((v.clamp(lo, hi) != v).sum())
            p14[:, j] = v.clamp(lo, hi)
            return n
        p14[:, j] = p14[:, j] * (1.0 + sign * delta_frac)
        return 0

    res, rows, clamped, nonfinite, neutral = {}, [], {}, {}, []
    onesided = set()
    for name in names:
      is_dir = name == dir_name
      for rel in args.rel:
        # Both signs, averaged. A parameter with an asymmetric response would
        # otherwise report whichever direction happened to be tried.
        norm_runs, raw_runs, n_clamp = [], [], 0
        with torch.no_grad():
            for sign in (+1.0, -1.0):
                p14 = ref14.clone()
                new_r = ratio_of.clone() if ratio_of is not None else None
                for k, w in (dir_u.items() if is_dir else ((name, 1.0),)):
                    n_clamp += perturb(p14, k, rel * w, sign)
                if ratio_of is not None:
                    # Applied once, AFTER every coordinate has moved and been
                    # clamped, so T60_F1 follows the values actually used. This
                    # is the whole point of the search basis: T60_F1 is not a
                    # coordinate, it is r times T60_DC.
                    p14[:, I_F1] = new_r * p14[:, I_DC]
                x_p = plate.forward(p14, args.duration, normalize=False)
                # A perturbation can leave the physical range -- T60 <= 0 is a
                # decay time that grows without bound. Counted and excluded
                # rather than averaged in, where one inf would take the whole
                # cell to nan and read as "this parameter does nothing".
                ok = torch.isfinite(x_p).all(dim=-1)
                bad = int((~ok).sum())
                if bad:
                    nonfinite[(name, rel)] = nonfinite.get((name, rel), 0) + bad
                if bad == x_p.shape[0]:
                    # Every IR in this branch left the physical range. Dropping
                    # the branch is the only honest option: zeroing and
                    # averaging turned "unphysical" into a LARGE change --
                    # T60_DC at x5 read 80% of the unrelated-IR distance, which
                    # was the zeroed render, not sensitivity.
                    continue
                norm_runs.append(decompose(A_n[ok], stft_mag(x_p[ok], args.n_fft, args.hop, True), eps_n))
                raw_runs.append(decompose(A_r[ok], stft_mag(x_p[ok], args.n_fft, args.hop, False), eps_r))

        def mean_of(runs, k):
            v = [r[k] for r in runs]
            if isinstance(v[0], list):
                return [sum(c) / len(c) for c in zip(*v)]
            return sum(v) / len(v)

        if not norm_runs:
            res[(name, rel)] = (float("nan"),) * 4
            continue
        if len(norm_runs) == 1:
            onesided.add((name, rel))
        cg = mean_of(norm_runs, 1)
        lb, gb = mean_of(norm_runs, 2), mean_of(norm_runs, 3)
        dn = 100.0 * mean_of(norm_runs, 4) / max(sat_n, 1e-30)
        gn = 100.0 * mean_of(norm_runs, 5) / max(sat_g, 1e-30)
        # gain = how much more a compressed loss has to work with than a linear
        # one, both as a fraction of their own unrelated-IR distance. Above 1
        # means compression sees more of this parameter; that is the whole
        # hypothesis, stated per parameter in one number.
        res[(name, rel)] = (dn, cg, gn, gn / dn if dn > 0 else float("nan"))
        if n_clamp:
            clamped[(name, rel)] = n_clamp
        # Every rel, not just the first: whether a signature MIGRATES between
        # bands as the poke grows is exactly what a single centroid cannot say.
        rows.append((name, rel, lb, gb, mean_of(norm_runs, 6), mean_of(norm_runs, 7)))
        neutral.append(mean_of(norm_runs, 8))

    w_name = max(10, max(len(n) for n in names) + 2)

    def table(title, key, fmt):
        print(f"\n=== {title}")
        print(f"{'param':<{w_name}}{'fit':>5}" +
              "".join(f"{100*r:>10.3g}%" for r in args.rel))
        for name in names:
            fit = ("v" if name in vary_bounds or name == "T60_ratio" or name == dir_name
                   else "y" if name in FITTED_BOUNDS else "-")
            # A direction and the ratio both have a range to step; only a plate
            # column with no bounds falls back to a fraction of its own value.
            has_range = (name == dir_name or search_bounds(name) is not None)
            mark = "*" if args.step == "range" and not has_range else ""
            print(f"{name + mark:<{w_name}}{fit:>5}" +
                  "".join(fmt.format(res[(name, r)][key])
                          + ("!" if (name, r) in onesided else " ")
                          for r in args.rel))

    table("dnorm% -- total change as a percentage of the unrelated-IR distance",
          0, "{:>10.3f}")
    table("gnorm% -- the same, in the LOG domain: what a compressed loss sees",
          2, "{:>10.3f}")
    table("gain -- gnorm%/dnorm%. >1 = compression has more of this parameter "
          "to work with", 3, "{:>10.2f}")
    table("log dec -- mean decile of |log(a+e)-log(b+e)|, 1 quietest to 10 loudest",
          1, "{:>10.2f}")
    if neutral:
        nz = sum(neutral) / len(neutral)
        print(f"\n  NEUTRAL FOR THIS FLOOR: {nz:.2f}  -- where a change that is "
              f"uniform in relative terms lands.")
        print("  Compare every log dec above against THIS, not against 5.5. Raising the")
        print("  floor suppresses the quiet deciles, so the centroid drifts up on its own")
        print("  and a fixed 5.5 reference would read that drift as 'loud-biased'.")
    if nonfinite:
        print("\n! = one sign branch dropped entirely, so that cell is a "
              "ONE-SIDED step and\n    does not compare with the rest of its "
              "row.\nNON-FINITE renders, excluded (not zeroed): a perturbation "
              "left the physical\nrange. beta ~ (1/T60_F1 - 1/T60_DC), so "
              "T60_F1 > T60_DC flips its sign and\ndamping DECREASES with "
              "frequency -- the high modes then grow without bound.\n"
              + ", ".join(f"{n}@{100*r:g}%={c}" for (n, r), c in sorted(nonfinite.items())))
    if clamped:
        print("\nclamped to the dataset's bounds (IRs, of "
              f"{2 * len(dicts)} per cell): " +
              ", ".join(f"{n}@{100*r:g}%={c}" for (n, r), c in sorted(clamped.items())))

    print("\ndnorm%          L1 on STFT magnitude as a percentage of the "
          "distance between\n                UNRELATED IRs, peak-normalized. "
          "Grows ~linearly with rel while the\n                response is "
          "local; bending over means it is saturating toward the\n            "
          "    distance between two different plates.")
    print("log dec         the informative one. Mean decile of "
          "|log(a+e)-log(b+e)|, 1\n                quietest to 10 loudest.")
    print("                5.5 IS NEUTRAL -- a change spread evenly across all "
          "ten deciles\n                gives exactly 5.5, so that is the "
          "reference, not the top of the\n                scale. Below it the "
          "signature is concentrated in quiet bins and only\n                a "
          "compressed loss sees it; above it in loud bins, where linear has\n"
          "                the advantage; at it, both losses see the parameter "
          "and the\n                compression ladder cannot separate them "
          "on it.")

    if args.detail:
        for name, rel, lb, gb, cb, rb in rows:
            print(f"\n=== {name}   where the change lives, by dB below peak "
                  f"(at {100*rel:g}%)")
            print(f"{'band':>10}{'bins':>8}{'linear':>9}{'lin/bin':>9}"
                  f"{'log':>9}{'log/bin':>9}{'rel%':>9}")
            sl, sg = max(sum(lb), 1e-30), max(sum(gb), 1e-30)
            sc = max(sum(cb), 1)
            for (lo, hi), a, b, c, r in zip(DB_BANDS, lb, gb, cb, rb):
                # share of the change, over share of the bins. 1.00 = this band
                # carries exactly its numerical weight; >1 concentrated here,
                # <1 depleted. The only column that compares bands to each other.
                fb = c / sc
                kl = (a / sl) / fb if fb > 0 else float("nan")
                kg = (b / sg) / fb if fb > 0 else float("nan")
                print(f"{f'{lo}-{hi}':>10}{100*fb:>7.1f}%{100*a/sl:>8.1f}%"
                      f"{kl:>8.2f}x{100*b/sg:>8.1f}%{kg:>8.2f}x{100*r:>8.2f}%")
        print("\n  bins       share of all time-frequency bins falling in that band")
        print("  linear/log share of the TOTAL change the band accounts for")
        print("  /bin       that share divided by the band's share of bins. This is the")
        print("             comparable column: 1.00x means the band carries exactly its")
        print("             numerical weight, above means the signature is concentrated")
        print("             there, below means depleted. Raw shares cannot be compared")
        print("             across bands -- the deep bands hold most of the bins, so a")
        print("             sum over them is large whatever the parameter does.")
        print("  rel%       mean |a-b|/(a+eps) over the band's bins: how much the signal")
        print("             in that band MOVED, in percent. This is the direct answer to")
        print("             'does the perturbation change the loud parts or the quiet")
        print("             parts', and unlike the share columns it needs no correction")
        print("             for how many bins a band holds. Rising with depth means a")
        print("             compressed loss has something to gain there.")


if __name__ == "__main__":
    main()
