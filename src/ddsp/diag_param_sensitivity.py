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

PERTURBATION. --rel is a fraction of the parameter's own VALUE, default 2%,
applied to all fourteen. Range-relative (delta * (hi - lo)) would be the more
meaningful step for the fitted seven, since normalized space is what the encoder
actually searches -- but it is undefined for the pinned seven, and putting all
fourteen on one footing is the point of the exercise. The `fit` column marks
which seven have bounds at all, so a reader knows which numbers carry that
caveat. Both signs are rendered and averaged, so an asymmetric response does not
report a misleading single number.
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
    p.add_argument("--rel", type=float, default=0.02,
                   help="Perturbation size, read according to --step")
    p.add_argument(
        "--step", default="value", choices=("value", "range"),
        help="value: --rel as a fraction of the parameter's own value. range: "
             "--rel as a fraction of (hi - lo) for the fitted seven, which is "
             "the space the encoder searches and the only convention under "
             "which totals compare across parameters -- T0 spans five decades "
             "and Ly spans a factor of 3.6, so a 2%% value step means wildly "
             "different fractions of what is actually being estimated. Pinned "
             "parameters have no range and fall back to value, marked * in the "
             "output.")
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
    print(f"{args.data_dir}   {len(dicts)} IRs   perturbation {100*args.rel:g}% "
          f"of value, both signs\n")

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
    print(f"{'param':<10}{'fit':>5}{'dnorm%':>10}{'draw%':>10}"
          f"{'lin dec':>9}{'log dec':>9}   top band lin / log")
    rows = []
    for name in names:
        i = BatchedModalPlateTorch.PARAM_ORDER.index(name)
        # Both signs, averaged. A parameter with an asymmetric response would
        # otherwise report whichever direction happened to be tried.
        norm_runs, raw_runs = [], []
        with torch.no_grad():
            for sign in (+1.0, -1.0):
                p14 = ref14.clone()
                if args.step == "range" and name in FITTED_BOUNDS:
                    lo, hi = FITTED_BOUNDS[name]
                    p14[:, i] = p14[:, i] + sign * args.rel * (hi - lo)
                else:
                    p14[:, i] = p14[:, i] * (1.0 + sign * args.rel)
                x_p = plate.forward(p14, args.duration, normalize=False)
                norm_runs.append(decompose(A_n, stft_mag(x_p, args.n_fft, args.hop, True)))
                raw_runs.append(decompose(A_r, stft_mag(x_p, args.n_fft, args.hop, False)))

        def mean_of(runs, k):
            v = [r[k] for r in runs]
            if isinstance(v[0], list):
                return [sum(c) / len(c) for c in zip(*v)]
            return sum(v) / len(v)

        cl, cg = mean_of(norm_runs, 0), mean_of(norm_runs, 1)
        lb, gb = mean_of(norm_runs, 2), mean_of(norm_runs, 3)
        ltot = mean_of(norm_runs, 4)
        ltot_r = mean_of(raw_runs, 4)
        dn = 100.0 * ltot / max(sat_n, 1e-30)
        dr = 100.0 * ltot_r / max(sat_r, 1e-30)
        top_l = DB_BANDS[max(range(len(lb)), key=lambda k: lb[k])]
        top_g = DB_BANDS[max(range(len(gb)), key=lambda k: gb[k])]
        fit = "y" if name in FITTED_BOUNDS else "-"
        mark = "*" if args.step == "range" and name not in FITTED_BOUNDS else ""
        print(f"{name + mark:<10}{fit:>5}{dn:>10.3f}{dr:>10.3f}{cl:>9.2f}"
              f"{cg:>9.2f}   {top_l[0]}-{top_l[1]} / {top_g[0]}-{top_g[1]} dB")
        rows.append((name, lb, gb))

    print("\ndnorm% / draw%  L1 on STFT magnitude as a percentage of the "
          "distance between\n                UNRELATED IRs, peak-normalized "
          "and raw. raw >> norm means the\n                parameter acts "
          "mostly through level, which normalizing deletes.")
    print("lin dec         share-weighted mean decile of |a-b|. Pins near 10 for "
          "EVERY\n                parameter and carries almost nothing: "
          "|a-b| <= max(a,b), so absolute\n                error concentrates "
          "in the loudest bins whatever changed. Printed to\n                "
          "show that, not as a result.")
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
