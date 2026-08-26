"""How far above zero is the loss at the true parameters, and why.

For a linear loss the target/candidate disagreement is a fixed relative error
and disappears into the noise: the L1_STFT run ended with a training loss of
0.152 against a gt_loss of ~1e-5. For log(x + 1e-7) that reasoning fails. The
float32 disagreement is roughly an absolute floor spread across bins spanning
seven decades, so in the quiet bins -- where the value approaches eps -- it
becomes an O(1) discrepancy in log space. confirm_f32_gt.py already showed this
shape for the old numpy-vs-torch gap: the log error concentrated in the
quietest energy deciles while the linear error did not.

That matters for the sweep more than for any single run. If the log arm's floor
sits near its saturation level, then "log compression is bad for parameter
estimation" is unattributable -- it could just be reporting that our targets
and our synthesis disagree in exactly the bins log weights most.

Three things now differ between how a target was rendered and how the same
parameters are synthesized during training, and this measures each separately:

  params   make_dataset builds the 14-vector straight from the CSV in float32;
           training goes CSV -> gt_z (float64) -> stored float32 ->
           norm_to_physical_torch -> plate14.
  path     make_dataset uses the plate's defaults; training runs
           --batched-plate --compile-plate --chunk-elems 1e9.
  batch    torch.sum over the mode axis reduces in blocks whose grouping
           depends on n_modes = max_DDx * max_DDy for the batch. Padded modes
           are exactly zero but still change the tree shape, so the nonzero
           terms group differently and an IR's rendering depends on the batch
           it was in. Training batches are random, so this one cannot be
           matched by any generation scheme -- only bounded.

    python -m src.ddsp.diag_gt_floor --n-val 64

EVERY NUMERIC FLAG MUST MATCH THE CAMPAIGN, or this measures a different plate
and reports the difference as arithmetic noise. For emt7 that is all of them:

    PLATE_PARAM_SPACE=emt7 python -m src.ddsp.diag_gt_floor \
        --data-dir data/val-emt7 --duration 1.0 \
        --fmax 12000 --fixed-mode-grid 100,220 \
        --chunk-elems 400000000 --batch-size 64 --compile-plate \
        --losses L1_STFT L1_STFT_hyb1e2 L1_STFT_eps1e2 --report-grid

Run with the defaults instead and it loads raw7's 0.25 s val set at a 10 kHz
ceiling with no pin, which is a real measurement of something and tells you
nothing whatsoever about the campaign you are about to spend a day on.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.make_dataset import parse_mode_grid, render as md_render
from src.ddsp.train_encoder import load_dataset, peak_normalized
from src.gd.graddescent import (
    Raw7Space, norm_to_physical_torch, physical_to_plate14_torch,
)
from src.loss.loss_selector import select_loss_function

LOSSES = ("L1_STFT", "L1_STFT_pow", "L1_STFT_c2", "L1_STFT_log")


def build_space(dev, batched, chunk, bucket, compile_plate=False, grid=None, fmax=None):
    # Raw7Space reads PARAM_KEYS/BOUNDS from fit_7param_norm_es, which selects on
    # PLATE_PARAM_SPACE -- so this measures whatever space the environment names,
    # exactly as the training job does. The class name is historical.
    space = Raw7Space(dev, torch.float32, normalize=False)
    space.configure_plate(chunk, False, batched, compile_plate, bucket, grid, fmax)
    return space


def synth(space, z, duration, batch):
    out = []
    with torch.no_grad():
        for i in range(0, z.shape[0], batch):
            out.append(space.forward(z[i : i + batch], None, duration).float())
    return torch.cat(out, dim=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, default=Path("data/val-1000-0.25s"))
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--chunk-elems", type=int, default=20_000_000)
    p.add_argument("--mode-bucket", type=int, default=1024)
    p.add_argument("--compile-plate", action="store_true",
                   help="Match the training run's --compile-plate; a fused kernel changes\nthe arithmetic, so leaving it off measures a floor training does not have")
    p.add_argument(
        "--fixed-mode-grid", type=parse_mode_grid, default=None, metavar="DDX,DDY",
        help="Match the pin the dataset was rendered with. Without it this builds an "
             "unpinned plate, so every variant disagrees with a pinned target by the same "
             "amount and the batch term is unmeasurable -- a harder test than training "
             "faces, and not the one that answers whether the pin worked.",
    )
    p.add_argument(
        "--fmax", type=float, default=None,
        help="Match the ceiling the dataset was rendered with. None keeps "
             "BatchedModalPlateTorch's 10000.0, which every raw7 dataset used. A "
             "mismatch here is not a floor measurement at all -- it renders a "
             "different plate and reports the difference as arithmetic noise.",
    )
    p.add_argument(
        "--losses", nargs="+", default=list(LOSSES), metavar="NAME",
        help="Which losses to price the floor in. The default four are the raw7 "
             "ladder; pass the arms a campaign actually trains, since a floor for "
             "L1_STFT_log says nothing about L1_STFT_eps1e2.",
    )
    p.add_argument("--report-grid", action="store_true",
                   help="Price pinning n_modes: global max grid vs the typical batch max")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    space = build_space(dev, True, args.chunk_elems, args.mode_bucket,
                        args.compile_plate, args.fixed_mode_grid, args.fmax)
    z, x_tgt = load_dataset(space, args.data_dir, args.duration, dev, args.n_val)
    print(f"{args.data_dir}   {x_tgt.shape[0]} IRs\n")

    # The training loss is the registry entry wrapped in target-peak
    # normalization, so that is the form the floor has to be read in.
    fns = {n: peak_normalized(select_loss_function(n, sample_rate=44100, device=dev), "target")
           for n in args.losses}

    # The make_dataset path: plate14 built straight from the CSV, never through
    # a float32 z. This is how the targets on disk were rendered, so it is the
    # decisive comparison -- if it alone scores ~0 while every z-path variant
    # sits at the floor, the round-trip is the whole story and regenerating
    # through space.forward() fixes it.
    csvs = sorted(args.data_dir.glob("random_IR_params_*.csv"))[: x_tgt.shape[0]]
    dicts = [pd.read_csv(c).iloc[0].to_dict() for c in csvs]
    md = [md_render(space.plate, dicts[i : i + args.batch_size], args.duration)
          for i in range(0, len(dicts), args.batch_size)]
    x_md = torch.as_tensor(np.concatenate(md, axis=0), dtype=torch.float32, device=dev)

    # The condition training actually faces: batches drawn at random, so an IR
    # sits with different companions every step. Before pinning, n_modes was the
    # batch maximum and therefore depended on those companions, so this is the
    # row that would expose it -- the sorted-batch row above cannot, since it
    # reproduces the batching generation used. Rendered shuffled, then put back
    # in order to compare against the targets.
    g = torch.Generator().manual_seed(1234)
    order = torch.randperm(z.shape[0], generator=g).to(z.device)
    inv = torch.argsort(order)
    x_shuf = synth(space, z[order], args.duration, args.batch_size)[inv]

    variants = {
        "training path (batched, batch=N)": synth(space, z, args.duration, args.batch_size),
        "training path, SHUFFLED batches": x_shuf,
        "same path, batch=1": synth(space, z, args.duration, 1),
        "unbatched modal sum": synth(build_space(dev, False, args.chunk_elems, args.mode_bucket,
                                                 args.compile_plate, args.fixed_mode_grid,
                                                 args.fmax),
                                     z, args.duration, args.batch_size),
        "make_dataset path (no float32 z)": x_md,
    }

    perm = torch.randperm(x_tgt.shape[0], generator=torch.Generator().manual_seed(0))
    print(f"{'':34s} " + "  ".join(f"{n:>13s}" for n in args.losses))
    with torch.no_grad():
        sat = {n: float(fns[n](x_tgt, x_tgt[perm]).mean()) for n in args.losses}
        print(f"{'saturation (unrelated IRs)':34s} " +
              "  ".join(f"{sat[n]:13.4e}" for n in args.losses))
        print()
        for tag, x_syn in variants.items():
            vals = {n: float(fns[n](x_tgt, x_syn).mean()) for n in args.losses}
            print(f"{tag:34s} " + "  ".join(f"{vals[n]:13.4e}" for n in args.losses))
            print(f"{'  as % of saturation':34s} " +
                  "  ".join(f"{100*vals[n]/max(sat[n],1e-30):12.4f}%" for n in args.losses))
        print()

    # Where the log disagreement lives, by target-magnitude decile -- the same
    # decomposition confirm_f32_gt.py used for the numpy gap.
    x_syn = variants["training path (batched, batch=N)"]
    w = torch.hann_window(4096, device=dev)

    def mag(v):
        v = v / v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
        return torch.stft(v, 4096, 1024, window=w, return_complex=True).abs()

    with torch.no_grad():
        A, B = mag(x_tgt), mag(x_syn)
        a, b = A.flatten(), B.flatten()
        lerr = (torch.log(a + 1e-7) - torch.log(b + 1e-7)).abs()
        linerr = (a - b).abs()
        order = torch.argsort(a)
    groups = np.array_split(order.cpu().numpy(), 10)
    print("disagreement share by target-magnitude decile (1 = quietest):")
    print(f"  {'decile':>7s} {'tgt mag':>11s} {'log share':>10s} {'linear share':>13s}")
    lt, nt = float(lerr.sum()), float(linerr.sum())
    if lt == 0.0 and nt == 0.0:
        # Not a degenerate case to guard around: an exactly zero disagreement is
        # the goal. Targets and training synthesis are bit-identical, so there
        # is no error left to decompose.
        print("  target and training synthesis agree bit-for-bit; nothing to decompose")
    else:
        for i, g in enumerate(groups, 1):
            idx = torch.as_tensor(g, device=dev)
            print(f"  {i:7d} {float(a[idx].mean()):11.3e} "
                  f"{100*float(lerr[idx].sum())/max(lt,1e-300):9.1f}% "
                  f"{100*float(linerr[idx].sum())/max(nt,1e-300):12.1f}%")
    if args.report_grid:
        # The last term is that n_modes is the batch maximum, so an IR renders
        # differently depending on which batch it was in. Pinning it to a global
        # constant removes that, at the cost of every batch paying the worst
        # case -- which is exactly the batch-max effect that made early training
        # 2.5x faster. This prices it before committing.
        # Go through the SAME conversion the renderer does rather than
        # re-deriving it. The previous version un-normalized z linearly and then
        # unpacked columns as (E, rho, h, Ly, T0) with nu hardcoded to 0.25 --
        # raw7's key order and raw7's Poisson ratio, both wrong for every other
        # space. Under emt7, whose keys are (Ly, h, T0, rho, E, T60_DC, loss_F1),
        # it read Ly as E and rho as Ly and reported DDx 0, DDy 274 against a
        # true (100, 220). It also ignored LOG_PARAMS, so T0 was wrong even for
        # raw7. physical_to_plate14_torch is the single definition of the
        # column order; taking nu from the vector means a space that changes it
        # cannot silently desynchronize this again.
        pl = space.plate
        ddx, ddy = [], []
        with torch.no_grad():
            for i in range(0, z.shape[0], args.batch_size):
                zz = z[i : i + args.batch_size]
                p14 = physical_to_plate14_torch(
                    norm_to_physical_torch(zz, space._lo, space._hi)).cpu().numpy()
                Lx, Ly, h, T0, rho, E, nu = (p14[:, j] for j in range(7))
                D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
                inner = np.sqrt(np.maximum(T0 ** 2 + 4.0 * (pl.max_omega ** 2) * rho * h * D, 0.0))
                disc = np.maximum((-T0 + inner) / (2.0 * D), 0.0)
                ddx.append(np.floor(Lx / np.pi * np.sqrt(disc)))
                ddy.append(np.floor(Ly / np.pi * np.sqrt(disc)))
        bx = np.array([b.max() for b in ddx]); by = np.array([b.max() for b in ddy])
        allx, ally = np.concatenate(ddx), np.concatenate(ddy)
        print(f"\nmode grid, batches of {args.batch_size}:")
        print(f"  per-batch max   DDx median {np.median(bx):.0f}  DDy median {np.median(by):.0f}"
              f"   -> {np.median(bx*by):,.0f} modes")
        print(f"  global max      DDx {allx.max():.0f}  DDy {ally.max():.0f}"
              f"   -> {allx.max()*ally.max():,.0f} modes")
        print(f"  pinning costs   {allx.max()*ally.max()/max(np.median(bx*by),1):.2f}x the "
              f"typical batch's modal work")
        if args.fixed_mode_grid is not None:
            px, py = args.fixed_mode_grid
            over = int(((allx > px) | (ally > py)).sum())
            print(f"  pin {px},{py}      " +
                  (f"OK, covers all {allx.size} draws"
                   if not over else
                   f"TRUNCATES {over} of {allx.size} draws -- the targets are not "
                   f"the plate's output at their own parameters"))

    print("\n  A log floor concentrated in the quiet deciles is the failure mode: it")
    print("  means the log arm's error is partly our own target/synthesis disagreement")
    print("  rather than what compression does to the terrain. Read the floors above")
    print("  against saturation -- linear will be ~1e-3 %; log needs to be small too.")


if __name__ == "__main__":
    main()
