"""Time the modal-sum kernel alone across n_pad. Is the cost a slope or a step?

    python -m src.emt.bench_modal
    TORCH_LOGS=recompiles python -m src.emt.bench_modal --n-pad 22528 44032
    TORCH_LOGS=output_code python -m src.emt.bench_modal --n-pad 22528 44032 2>&1 \\
        | tee /tmp/triton.txt

THE QUESTION. Training at n_pad 44032 runs 3.6x slower than at 22528, but the
arithmetic -- B * n_pad * Ts -- is only 1.955x. Roughly 1.8x is unaccounted for,
and it is NOT launch count, memory footprint or per-mode re-read traffic: 4e8
and 8e8 chunk_elems give 313 and 156 launches, 8.7 and 14 GB, and identical
step rates.

WHAT THIS SEPARATES. Time per ELEMENT, ns / (B * n_pad * Ts):

  flat across n_pad          cost is linear in work and the slowdown is
                             elsewhere in the training step, not in the kernel.
                             MEASURED: flat. 0.0108 ns/elem at 16384 rising only
                             to 0.0120 at 44032, with 49152 (1.05x L2) FASTER
                             than 44032 (0.94x). No step anywhere. Both the L2
                             and the persistent-reduction hypotheses are dead,
                             and chunk_elems is confirmed irrelevant.
                             But that run was under no_grad -- see --grad.
  smooth rise                tiling/occupancy degrades gradually with shape.
  a STEP at some n_pad       a threshold was crossed. Two candidates:

    L2 CAPACITY. The kernel is sum(P * env * osc, dim=1) on [B,N,1] x [1,1,C],
    so sig/om/den/P are BROADCAST reads -- every output column reads the same
    [b,n] values. The reused working set is 4*B*n_pad*4 bytes: 23.1 MB at 22528
    against 45.1 MB at 44032, on an L4 whose L2 is 48 MB. Reads that hit cache
    at the smaller shape miss at the larger one. This is independent of
    chunk_elems, which matches the 4e8-vs-8e8 result.

    KERNEL SELECTION. Inductor picks a persistent reduction when rnumel is small
    enough to hold in registers and a looped one otherwise. Its thresholds are
    order 1e3, and both shapes here are 20-40x above that, so both should
    already be looped -- but the tiling heuristics still differ per shape and
    only the generated Triton settles it.

READ IT WITH THE OTHER TWO LOGS. TORCH_LOGS=output_code dumps the Triton for
each shape; diff the @triton.jit bodies for RBLOCK as a constexpr covering the
whole reduction versus a loop over roffset. TORCH_LOGS=recompiles shows whether
Dynamo is respecializing -- the plate raises the limit from its default of 8 to
128 for this reason, since dynamic=False means every distinct (N, C) is a fresh
compile and the final chunk of each forward has its own C.

Forward only, no autograd, no encoder, no dataset. It times the kernel and
nothing else.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.plate.SevenParamPlate import (                    # noqa: E402
    _get_modal_chunk_kernel_batched,
)

L2_MB = {"L4": 48, "A100": 40, "H100": 50, "L40S": 96}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-pad", type=int, nargs="+",
                   default=[16384, 22528, 24576, 28672, 32768, 36864, 40960,
                            44032, 49152, 57344])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ts", type=int, default=44100)
    p.add_argument("--chunk-elems", type=int, nargs="+",
                   default=[400_000_000, 800_000_000])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument(
        "--grad", action="store_true",
        help="Run with autograd LIVE, as training does, instead of under "
             "no_grad. torch.compile builds a different graph for the training "
             "forward -- it must save residuals for backward, which can change "
             "Inductor's fusion decisions entirely. The plate trains with "
             "--no-grad-checkpoint, so nothing is recomputed either. If ns/elem "
             "jumps under --grad but is flat without it, the cost is in the "
             "training-mode graph and not in the arithmetic.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device)
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    l2 = next((v for kk, v in L2_MB.items() if kk in name), None)
    print(f"{name}   L2 {l2 or '?'} MB   batch {args.batch}   Ts {args.ts}   "
          f"compile {not args.no_compile}")
    print("\n  per-mode working set = 4 tensors x batch x n_pad x 4 bytes; the")
    print("  broadcast reads that hit L2 below the line and miss above it.\n")
    kernel = _get_modal_chunk_kernel_batched(not args.no_compile)

    hdr = f"  {'n_pad':>7}{'set MB':>9}{'vs L2':>8}"
    for ce in args.chunk_elems:
        hdr += f"{f'{ce/1e6:.0f}M: chunk':>16}{'ms':>9}{'ns/elem':>10}"
    print(hdr)

    base = {}
    for n_pad in args.n_pad:
        ws = 4 * args.batch * n_pad * 4 / 1e6
        row = f"  {n_pad:>7,}{ws:>9.1f}{(ws/l2 if l2 else 0):>7.2f}x"
        for ce in args.chunk_elems:
            chunk = max(64, ce // (args.batch * n_pad))
            g = torch.Generator(device="cpu").manual_seed(0)
            mk = lambda lo, hi: (lo + (hi - lo) * torch.rand(
                (args.batch, n_pad, 1), generator=g)).to(dev)
            sig = mk(1.0, 50.0)
            om = mk(100.0, 1.2e5)
            den = mk(0.1, 1.0)
            P = mk(-1e-3, 1e-3)
            if args.grad:
                # Only P carries a gradient in training -- it is what the
                # encoder's parameters flow through -- but that is enough to put
                # every chunk on the training-forward path.
                P.requires_grad_(True)
            k = 1.0 / 44100.0

            def once():
                out = []
                for s in range(0, args.ts, chunk):
                    e = min(s + chunk, args.ts)
                    t = torch.arange(s, e, device=dev,
                                     dtype=torch.float32).view(1, 1, -1)
                    out.append(kernel(sig, om, den, P, t, k))
                return torch.cat(out, dim=1)

            ctx = torch.enable_grad() if args.grad else torch.no_grad()
            with ctx:
                for _ in range(args.warmup):
                    once()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.reps):
                    y = once()
                    if args.grad:
                        # Realise the graph the way training does; the base
                        # never calls backward through it, so neither do we.
                        del y
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / args.reps * 1e3
            elems = args.batch * n_pad * args.ts
            ns = ms * 1e6 / elems
            base.setdefault(ce, ns)
            row += f"{chunk:>16}{ms:>9.1f}{ns:>10.4f}"
            del sig, om, den, P
            torch.cuda.empty_cache()
        print(row)

    print("\n  ns/elem is the number to read. Flat = cost is linear in work and")
    print("  the training slowdown is NOT in this kernel. A step between two")
    print("  adjacent n_pad = a threshold, and the 'vs L2' column says whether")
    print("  it lands where the working set crosses the cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
