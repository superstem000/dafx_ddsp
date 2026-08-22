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
"""

from __future__ import annotations

import argparse
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
DB_BANDS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 400)]
EPS = 1e-7


def stft_mag(x: torch.Tensor, n_fft: int, hop: int, normalize: bool) -> torch.Tensor:
    """Peak-normalized or raw magnitude spectrogram, matching diag_gt_floor."""
    if normalize:
        x = x / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    w = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft, hop, window=w, return_complex=True).abs()


def decompose(ref: torch.Tensor, per: torch.Tensor):
    """(linear centroid, log centroid, linear bands, log bands, totals).

    Bins are ordered by the REFERENCE magnitude, not the perturbed one, so the
    buckets mean the same thing for every parameter and every step size.
    """
    a, b = ref.flatten(), per.flatten()
    lin = (a - b).abs()
    log = (torch.log(a + EPS) - torch.log(b + EPS)).abs()

    order = torch.argsort(a)
    lin_s, log_s = lin[order], log[order]
    n = lin_s.numel()
    edges = [round(i * n / 10) for i in range(11)]
    lin_d = torch.tensor([lin_s[edges[i]:edges[i + 1]].sum() for i in range(10)])
    log_d = torch.tensor([log_s[edges[i]:edges[i + 1]].sum() for i in range(10)])

    def centroid(d):
        t = d.sum()
        # 1 = quietest decile, 10 = loudest. nan when the parameter did nothing
        # at all, which is itself the answer and must not read as decile 1.
        return float((d * torch.arange(1, 11)).sum() / t) if t > 0 else float("nan")

    rel_db = 20.0 * torch.log10((a / a.max().clamp(min=1e-30)).clamp(min=1e-30))
    lin_b, log_b = [], []
    for lo, hi in DB_BANDS:
        m = (-rel_db >= lo) & (-rel_db < hi)
        lin_b.append(float(lin[m].sum()))
        log_b.append(float(log[m].sum()))
    return (centroid(lin_d), centroid(log_d), lin_b, log_b,
            float(lin.sum()), float(log.sum()))


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
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode-bucket", type=int, default=1024)
    p.add_argument("--chunk-elems", type=int, default=1_000_000_000)
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    plate = BatchedModalPlateTorch(
        device=dev, batched_modal_sum=True, compile_modal_sum=False,
        chunk_elems=args.chunk_elems, mode_bucket=args.mode_bucket,
    )

    csvs = sorted(args.data_dir.glob("random_IR_params_*.csv"))[: args.n]
    if not csvs:
        raise SystemExit(f"no random_IR_params_*.csv under {args.data_dir}")
    dicts = [pd.read_csv(c).iloc[0].to_dict() for c in csvs]
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
        perm = torch.randperm(x_ref.shape[0], device=dev)
        A_n = stft_mag(x_ref, args.n_fft, args.hop, True)
        A_r = stft_mag(x_ref, args.n_fft, args.hop, False)
        sat_n = float((A_n - A_n[perm]).abs().sum())
        sat_r = float((A_r - A_r[perm]).abs().sum())

    names = [k for k in BatchedModalPlateTorch.PARAM_ORDER
             if not args.only or k in args.only]
    res, rows, clamped, nonfinite = {}, [], {}, {}
    onesided = set()
    for name in names:
      i = BatchedModalPlateTorch.PARAM_ORDER.index(name)
      for rel in args.rel:
        # Both signs, averaged. A parameter with an asymmetric response would
        # otherwise report whichever direction happened to be tried.
        norm_runs, raw_runs, n_clamp = [], [], 0
        with torch.no_grad():
            for sign in (+1.0, -1.0):
                p14 = ref14.clone()
                if args.step == "mul":
                    f = (1.0 + rel)
                    p14[:, i] = p14[:, i] * (f if sign > 0 else 1.0 / f)
                elif args.step == "range" and name in FITTED_BOUNDS:
                    lo, hi = FITTED_BOUNDS[name]
                    v = p14[:, i] + sign * rel * (hi - lo)
                    # A large step walks off the range the dataset was drawn
                    # from, which would render plates no target ever contained.
                    # Clamped, and the count reported, so the ladder's top rungs
                    # are read knowing how much of the batch hit the wall.
                    n_clamp += int((v.clamp(lo, hi) != v).sum())
                    p14[:, i] = v.clamp(lo, hi)
                else:
                    p14[:, i] = p14[:, i] * (1.0 + sign * rel)
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
                norm_runs.append(decompose(A_n[ok], stft_mag(x_p[ok], args.n_fft, args.hop, True)))
                raw_runs.append(decompose(A_r[ok], stft_mag(x_p[ok], args.n_fft, args.hop, False)))

        def mean_of(runs, k):
            v = [r[k] for r in runs]
            if isinstance(v[0], list):
                return [sum(c) / len(c) for c in zip(*v)]
            return sum(v) / len(v)

        if not norm_runs:
            res[(name, rel)] = (float("nan"), float("nan"))
            continue
        if len(norm_runs) == 1:
            onesided.add((name, rel))
        cg = mean_of(norm_runs, 1)
        lb, gb = mean_of(norm_runs, 2), mean_of(norm_runs, 3)
        dn = 100.0 * mean_of(norm_runs, 4) / max(sat_n, 1e-30)
        res[(name, rel)] = (dn, cg)
        if n_clamp:
            clamped[(name, rel)] = n_clamp
        if rel == args.rel[0]:
            rows.append((name, lb, gb))

    def table(title, key, fmt):
        print(f"\n=== {title}")
        print(f"{'param':<10}{'fit':>5}" +
              "".join(f"{100*r:>10.3g}%" for r in args.rel))
        for name in names:
            fit = "y" if name in FITTED_BOUNDS else "-"
            mark = "*" if args.step == "range" and name not in FITTED_BOUNDS else ""
            print(f"{name + mark:<10}{fit:>5}" +
                  "".join(fmt.format(res[(name, r)][key])
                          + ("!" if (name, r) in onesided else " ")
                          for r in args.rel))

    table("dnorm% -- total change as a percentage of the unrelated-IR distance",
          0, "{:>10.3f}")
    table("log dec -- mean decile of |log(a+e)-log(b+e)|, 1 quietest to 10 loudest",
          1, "{:>10.2f}")
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
          "|log(a+e)-log(b+e)|, 1\n                quietest to 10 loudest. LOW "
          "means the parameter's RELATIVE signature\n                lives in "
          "quiet bins -- the region only a compressed loss weights, and\n      "
          "          the region a dB floor throws away.")

    if args.detail:
        for name, lb, gb in rows:
            print(f"\n=== {name}   share of total change by dB below peak")
            print(f"{'band':>12}{'linear':>12}{'log':>12}")
            sl, sg = max(sum(lb), 1e-30), max(sum(gb), 1e-30)
            for (lo, hi), a, b in zip(DB_BANDS, lb, gb):
                print(f"{f'{lo}-{hi}':>12}{100*a/sl:>11.1f}%{100*b/sg:>11.1f}%")


if __name__ == "__main__":
    main()
