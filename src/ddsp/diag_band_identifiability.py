"""Per-band identifiability on the plate. See src/analysis/band_identifiability.

    python -m src.ddsp.diag_band_identifiability --fixed-mode-grid 60,185
    PLATE_PARAM_SPACE=quiet7 python -m src.ddsp.diag_band_identifiability \
        --fixed-mode-grid 60,185 --n 24 --k 32

Targets and candidates are drawn from the ACTIVE parameter space -- whatever
PLATE_PARAM_SPACE selects -- because the question is about the task the encoder
was given, not about the plate in general. quiet7's answer and raw7's answer
are allowed to differ and it is interesting if they do.

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=24, help="Targets")
    p.add_argument("--k", type=int, default=32, help="Candidates per target")
    p.add_argument("--max-rel", type=float, default=0.30,
                   help="Candidates are drawn at radii uniform in (0, this] as "
                        "a fraction of the normalized range, along random "
                        "directions. A SPREAD rather than a fixed radius: "
                        "concordance needs candidates at different true "
                        "distances to have anything to rank.")
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--n-fft", type=int, default=4096)
    p.add_argument("--hop", type=int, default=1024)
    p.add_argument("--floor-db", type=float, default=None,
                   help="Set the log measure's floor this far below each "
                        "target's peak instead of at the absolute eps 1e-7. At "
                        "the default the floor sits ~160 dB down, forty below "
                        "where this float32 modal sum stops being physics.")
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

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    plate = BatchedModalPlateTorch(
        device=dev, batched_modal_sum=True, compile_modal_sum=False,
        chunk_elems=args.chunk_elems, mode_bucket=args.mode_bucket)
    plate.fixed_mode_grid = args.fixed_mode_grid
    if args.fixed_mode_grid is None:
        print("WARNING: no --fixed-mode-grid. Candidates and targets will sum "
              "different\n  numbers of modes, and that difference lands in the "
              "quiet bins this tool reads.\n")

    P = len(PARAM_KEYS)
    print(f"plate   space {PARAM_SPACE}   {P} searched   {args.n} targets   "
          f"{args.k} candidates each   radii (0, {args.max_rel:g}] of range")

    def render(norm: np.ndarray) -> torch.Tensor:
        out = []
        for i in range(0, norm.shape[0], args.render_batch):
            p14 = physical_to_plate14_tensor(
                norm_to_physical(norm[i:i + args.render_batch]), dev)
            with torch.no_grad():
                out.append(plate.forward(p14, args.duration, normalize=False))
        return torch.cat(out, dim=0)

    g = torch.Generator().manual_seed(args.seed)
    rows, dropped = [], 0
    for _ in range(args.n):
        tgt = (torch.rand(P, generator=g) * 2.0 - 1.0).numpy().astype(np.float64)

        # Random direction, random radius, then CLAMP and recompute the true
        # distance from what survived the clamp. Using the intended radius
        # would mislabel every candidate that hit a bound, and near a corner of
        # the box most of them do.
        d = torch.randn((args.k, P), generator=g).numpy()
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
        r = (torch.rand((args.k, 1), generator=g).numpy() * args.max_rel)
        cand = np.clip(tgt[None, :] + d * r * 2.0, -1.0, 1.0)
        dist = np.linalg.norm(cand - tgt[None, :], axis=1) / 2.0

        x_ref = render(tgt[None, :])[0]
        x_can = render(cand)
        ok = torch.isfinite(x_can).all(dim=-1)
        if not bool(ok.all()):
            dropped += int((~ok).sum())
            x_can, dist = x_can[ok], dist[ok.cpu().numpy()]
        if x_can.shape[0] < 4 or not torch.isfinite(x_ref).all():
            continue

        A_ref = stft_mag(x_ref[None, :], args.n_fft, args.hop, True)[0]
        A_can = stft_mag(x_can, args.n_fft, args.hop, True)
        eps = (EPS if args.floor_db is None
               else float(A_ref.max()) * 10.0 ** (-args.floor_db / 20.0))
        rows.append(bi.probe(A_ref, A_can,
                             torch.tensor(dist, device=A_ref.device), eps))

    if not rows:
        raise SystemExit("no usable targets")
    if dropped:
        print(f"  {dropped} non-finite candidate renders dropped")
    if args.floor_db is not None:
        print(f"  log floor: {args.floor_db:g} dB below each target's peak")
    bi.report(bi.accumulate(rows), title=f"plate / {PARAM_SPACE}   "
              f"{len(rows)} targets")


if __name__ == "__main__":
    main()
