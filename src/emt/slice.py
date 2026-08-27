"""Is the spectral loss a smooth basin around the truth, or a staircase?

    PLATE_PARAM_SPACE=emt8 python -m src.emt.slice
    PLATE_PARAM_SPACE=emt8 python -m src.emt.slice --param h Ly T0 --fmax 22000 12000

No encoder, no dataset, no training. It renders a target at a known parameter
vector, walks ONE coordinate through it, and reports the loss at every step.

THE QUESTION. emt8's arms end ABOVE their own initialisation on their own
objective: the parameter-only base reaches L1_STFT 0.40-0.45 with spec_w 0
throughout, and the arm trained to minimise L1_STFT bottoms at 0.53 and sits at
0.68. The value is demonstrably reachable, so either the optimiser cannot get
there or the surface between here and there is not something gradient descent
can walk. This measures the surface.

WHY THE CEILING IS THE SUSPECT. Three campaigns, and only the newest fails:

    quiet7   11,100 modes   fmax 10 kHz    45% of Nyquist   fine
    emt7     22,000 modes   fmax 12 kHz    54%              linear arm ratio 0.14
    emt8     43,875 modes   fmax 22 kHz   100%              linear arm ratio 3.06

emt7 shares emt8's step count, crossfade length, dataset size, duration, epoch
count and unhinged head -- and a WORSE handover seam -- and did not do this. The
schedule hypotheses all die on that comparison. What is left is the ceiling.

WHAT ACTUALLY MAKES IT A STAIRCASE, and there are two hard gates in
SevenParamPlate.forward, neither of them differentiable:

    inner = sqrt(T0^2 + 4*max_omega^2*rho*h*D)
    DDx_f = floor(Lx/pi * sqrt((-T0 + inner)/(2D)))      <- INTEGER FLOOR
    DDy_f = floor(Ly/pi * sqrt(...))
    valid_mask = (m <= DDx_f) & (n <= DDy_f) & (om <= max_omega)   <- HARD

sqrt_disc grows with max_omega, so DDx_f and DDy_f are LARGER at a higher
ceiling and the same fractional change in h or Ly steps the floor more times.
Separately, a 2-D mode grid has more modes per Hz at high frequency than at low,
so the population sitting within one step of `om <= max_omega` is largest
exactly when the ceiling is highest. Every mode that crosses either gate appears
or disappears in full, which puts a step discontinuity in the loss.

At fmax 22000 and sr 44100 the top modes sit at om/sr = 3.134 rad/sample against
pi = 3.1416 -- 99.8% of Nyquist, about 2.005 samples per cycle.

RESOLUTION. The sweep half-width is in z, the coordinate the encoder emits, so
it is directly comparable to what the optimiser does: Adam's update is ~lr per
step, so at lr 3e-4 the default +-0.05 over 401 points makes each sample about
ONE Adam step. A surface that is smooth at this spacing is one gradient descent
can follow; one that is not, is not.

READ THE OUTPUT AS. `min at` should be 0.000 -- the truth. `local minima`
should be 1. `wrong-way` is the fraction of steps that move away from the truth
while pointing at it, so 0% is a clean basin and anything large is a surface
with no usable gradient. `grid steps` and `mode jumps` are the mechanism: they
count how many times the two hard gates fired across the same sweep.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.gd.graddescent import (                                  # noqa: E402
    PARAM_KEYS, Raw7Space, norm_to_physical_torch, physical_to_plate14_torch,
)
from src.loss.loss_selector import select_loss_function            # noqa: E402
from src.ddsp.train_encoder import peak_normalized                 # noqa: E402

SAMPLE_RATE = 44100


def mode_gates(p14: torch.Tensor, max_omega: float, grid):
    """(DDx_f, DDy_f, n_valid) exactly as SevenParamPlate.forward computes them.

    Replicated rather than called because the plate returns audio, not its own
    masks, and the whole point here is to see the masks move.
    """
    Lx, Ly, h, T0, rho, E, nu = [p14[:, i] for i in range(7)]
    D = E * h.pow(3) / (12.0 * (1.0 - nu.pow(2)))
    inner = torch.sqrt(torch.clamp(T0.pow(2) + 4.0 * (max_omega ** 2) * rho * h * D, min=0.0))
    sqrt_disc = torch.sqrt(torch.clamp((-T0 + inner) / (2.0 * D), min=0.0))
    DDx_f = torch.floor(Lx / math.pi * sqrt_disc)
    DDy_f = torch.floor(Ly / math.pi * sqrt_disc)

    max_DDx, max_DDy = grid
    dev, dt = p14.device, p14.dtype
    m = (torch.arange(1, max_DDx + 1, device=dev, dtype=dt)
         .view(-1, 1).repeat(1, max_DDy).flatten().unsqueeze(0))
    n = (torch.arange(1, max_DDy + 1, device=dev, dtype=dt)
         .view(1, -1).repeat(max_DDx, 1).flatten().unsqueeze(0))
    g1 = (m * (math.pi / Lx.unsqueeze(1))).pow(2) + (n * (math.pi / Ly.unsqueeze(1))).pow(2)
    om = torch.sqrt(torch.clamp(
        (T0 / (rho * h)).unsqueeze(1) * g1 + (D / (rho * h)).unsqueeze(1) * g1.pow(2), min=0.0))
    valid = (m <= DDx_f.unsqueeze(1)) & (n <= DDy_f.unsqueeze(1)) & (om <= max_omega)
    return DDx_f, DDy_f, valid.sum(dim=1)


def render(space, z: torch.Tensor, duration: float) -> torch.Tensor:
    phys = norm_to_physical_torch(z, space._lo, space._hi)
    with torch.no_grad():
        return space.plate(physical_to_plate14_torch(phys), duration=duration,
                           normalize=False)


def roughness(loss: torch.Tensor, i_truth: int):
    """(argmin offset, n local minima, wrong-way fraction) for one slice."""
    v = loss.tolist()
    i_min = min(range(len(v)), key=lambda i: v[i])
    n_min = sum(1 for i in range(1, len(v) - 1) if v[i] < v[i - 1] and v[i] < v[i + 1])
    # Walking toward the truth from either side, how often does the loss go the
    # wrong way? This is the number that says whether a gradient is usable.
    wrong = tot = 0
    for i in range(i_truth):                     # left of truth: should fall as i grows
        tot += 1
        wrong += v[i + 1] > v[i]
    for i in range(i_truth, len(v) - 1):         # right of truth: should rise
        tot += 1
        wrong += v[i + 1] < v[i]
    return i_min, n_min, wrong / max(1, tot)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--param", nargs="+", default=["h"],
                   help=f"Coordinates to sweep. Available: {PARAM_KEYS}")
    p.add_argument("--fmax", type=float, nargs="+", default=[22000.0, 12000.0],
                   help="Ceilings to compare. THE MODE GRID IS HELD FIXED across "
                        "them, so max_omega is the only thing that differs and "
                        "the comparison is single-variable.")
    p.add_argument("--fixed-mode-grid", default="125,351",
                   help="Pinned grid, emt8's. Held across every --fmax.")
    p.add_argument("--half-width", type=float, default=0.05,
                   help="Sweep half-width in z. 0.05 at lr 3e-4 is ~167 Adam steps.")
    p.add_argument("--points", type=int, default=401,
                   help="Odd, so the truth is sampled exactly. 401 over the "
                        "default half-width puts one sample per Adam step.")
    p.add_argument("--truth-seed", type=int, default=0,
                   help="Seed for the truth z. 0 uses the box centre.")
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--losses", nargs="+",
                   default=["L1_STFT", "L1_STFT_hyb1e2", "L1_STFT_eps1e2"])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--chunk-elems", type=int, default=800_000_000)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.points % 2 == 0:
        args.points += 1
    grid = tuple(int(v) for v in args.fixed_mode_grid.split(","))
    dev = torch.device(args.device)
    space = Raw7Space(dev, torch.float32, normalize=False)
    nk = len(PARAM_KEYS)

    if args.truth_seed:
        g = torch.Generator().manual_seed(args.truth_seed)
        z0 = (torch.rand(nk, generator=g) * 2 - 1) * 0.6      # keep off the walls
    else:
        z0 = torch.zeros(nk)
    z0 = z0.to(dev)

    print(f"space {os.environ.get('PLATE_PARAM_SPACE', 'raw7')}   keys {PARAM_KEYS}")
    print(f"grid {grid} ({grid[0]*grid[1]:,} modes)   duration {args.duration}s   "
          f"sr {SAMPLE_RATE}")
    print(f"truth z = [" + ", ".join(f"{v:+.3f}" for v in z0.tolist()) + "]")
    print(f"sweep +-{args.half_width} in z over {args.points} points "
          f"(step {2*args.half_width/(args.points-1):.2e}, ~1 Adam step at lr 3e-4)\n")

    losses = {n: peak_normalized(select_loss_function(n, sample_rate=SAMPLE_RATE,
                                                      device=dev), "target")
              for n in args.losses}
    i_truth = args.points // 2
    offs = torch.linspace(-args.half_width, args.half_width, args.points, device=dev)

    for key in args.param:
        j = PARAM_KEYS.index(key)
        print(f"=== {key}   (z {z0[j].item():+.3f} -> physical range swept)")
        hdr = (f"  {'fmax':>7}{'Nyq%':>6}{'DDx':>6}{'DDy':>6}{'modes':>9}"
               f"{'grid steps':>12}{'mode jumps':>12}")
        for n in args.losses:
            hdr += f"{n.replace('L1_STFT', 'lin').replace('_', ''):>10}{'min at':>9}{'minima':>8}{'wrong-way':>11}"
        print(hdr)

        for fmax in args.fmax:
            space.configure_plate(args.chunk_elems, False, True, not args.no_compile,
                                  1024, grid, fmax)
            max_omega = 2.0 * math.pi * fmax

            zs = z0.unsqueeze(0).repeat(args.points, 1)
            zs[:, j] = z0[j] + offs

            p14 = physical_to_plate14_torch(
                norm_to_physical_torch(zs, space._lo, space._hi))
            DDx, DDy, nval = mode_gates(p14, max_omega, grid)
            # How many times each hard gate fires across the sweep.
            grid_steps = int(((DDx[1:] != DDx[:-1]) | (DDy[1:] != DDy[:-1])).sum())
            mode_jumps = int((nval[1:] != nval[:-1]).sum())

            tgt = render(space, zs[i_truth:i_truth + 1], args.duration)
            curves = {n: [] for n in args.losses}
            for s in range(0, args.points, args.batch):
                cand = render(space, zs[s:s + args.batch], args.duration)
                t = tgt.expand_as(cand)
                for n, fn in losses.items():
                    curves[n].append(fn(t, cand).detach().cpu())
            curves = {n: torch.cat(v) for n, v in curves.items()}

            row = (f"  {fmax:>7,.0f}{100*fmax/(SAMPLE_RATE/2):>5.0f}%"
                   f"{int(DDx[i_truth]):>6}{int(DDy[i_truth]):>6}"
                   f"{int(nval[i_truth]):>9,}{grid_steps:>12}{mode_jumps:>12}")
            for n in args.losses:
                i_min, n_min, wrong = roughness(curves[n], i_truth)
                row += (f"{curves[n][i_truth].item():>10.4f}"
                        f"{offs[i_min].item():>+9.4f}{n_min:>8}{wrong:>10.0%}")
            print(row)
            del p14, zs, tgt
            torch.cuda.empty_cache()
        print()

    print("  min at 0.0000, minima 1, wrong-way 0% is a clean basin: the truth is")
    print("  the minimum and every step points at it. Many minima with a large")
    print("  wrong-way fraction is a surface gradient descent cannot walk, and")
    print("  'grid steps' / 'mode jumps' say whether the two hard gates are why.")
    print("  Compare the SAME loss across fmax rows -- the mode grid is pinned, so")
    print("  max_omega is the only difference between them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
