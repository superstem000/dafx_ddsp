"""Per-IR PSO success-rate diagnostics vs population size.

This script mirrors src.diagnostics.success_rate output conventions but swaps
CMA-ES with PSO in normalized [-1, 1] 7D search space.

A run is considered successful if best_loss < success_threshold (default 0.01).
For each IR, outputs:
- CSV: success-rate summary per population size
- PNG: success-rate vs population size plot
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

from src.cmaes.fit_7param_norm import (
    NORM_LO_NP,
    available_loss_names,
    load_target_ir_from_npz,
    norm_to_physical,
    physical_to_plate14_tensor,
    select_loss_function,
)
from src.plate.SevenParamPlate import BatchedModalPlateTorch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAMPLE_RATE = 44100


def run_pso_once(
    target_ir: np.ndarray,
    duration: float,
    synth: BatchedModalPlateTorch,
    loss_fn,
    popsize: int,
    budget: int,
    seed: int,
    dtype: torch.dtype,
    device: str,
    w: float,
    c1: float,
    c2: float,
) -> float:
    target_t = torch.tensor(target_ir, dtype=dtype, device=device).unsqueeze(0)
    dim = len(NORM_LO_NP)

    rng = np.random.default_rng(seed)

    particles = rng.uniform(-1.0, 1.0, size=(popsize, dim)).astype(np.float64)
    velocities = rng.uniform(-0.2, 0.2, size=(popsize, dim)).astype(np.float64)

    def evaluate(norm_batch: np.ndarray) -> np.ndarray:
        phys = norm_to_physical(norm_batch)
        plate_params = physical_to_plate14_tensor(phys, device).to(dtype=dtype)
        with torch.no_grad():
            audios = synth(plate_params, duration)
            losses = loss_fn(target_t.expand(norm_batch.shape[0], -1), audios).detach().cpu().numpy()
        return np.nan_to_num(losses, nan=1e6, posinf=1e6, neginf=1e6)

    pbest = particles.copy()
    pbest_scores = evaluate(particles)
    g_idx = int(np.argmin(pbest_scores))
    gbest = pbest[g_idx].copy()
    gbest_score = float(pbest_scores[g_idx])

    if budget < popsize:
        max_iter = 0
    else:
        max_iter = max(0, (budget // popsize) - 1)

    for _ in range(max_iter):
        r1 = rng.random(size=(popsize, dim))
        r2 = rng.random(size=(popsize, dim))

        velocities = w * velocities + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest[np.newaxis, :] - particles)
        particles = np.clip(particles + velocities, -1.0, 1.0)

        scores = evaluate(particles)

        improved = scores < pbest_scores
        pbest[improved] = particles[improved]
        pbest_scores[improved] = scores[improved]

        g_idx = int(np.argmin(pbest_scores))
        if float(pbest_scores[g_idx]) < gbest_score:
            gbest = pbest[g_idx].copy()
            gbest_score = float(pbest_scores[g_idx])

    return gbest_score


def save_ir_csv(rows: list[dict], out_csv: Path) -> None:
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ir_name",
        "popsize",
        "n_runs",
        "n_success",
        "success_rate",
        "successful_run_indices",
        "success_threshold",
        "mean_best_loss",
        "std_best_loss",
        "min_best_loss",
        "max_best_loss",
        "total_runtime_s",
        "avg_runtime_s",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_ir_plot(rows: list[dict], out_png: Path, ir_name: str) -> None:
    if not rows:
        return
    pop = np.array([r["popsize"] for r in rows], dtype=np.int64)
    sr = np.array([r["success_rate"] for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(pop, sr, marker="o", linewidth=1.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Population Size")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Success Rate vs Popsize | {ir_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def run(args):
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = torch.float32

    loss_fn = select_loss_function(args.loss, sample_rate=SAMPLE_RATE, device=device)

    print(f"PSO success-rate diagnostics on {device.upper()} dtype={dtype}")
    print(f"  Loss: {args.loss}")
    print(f"  Available losses: {', '.join(available_loss_names())}")
    print(f"  Pop sizes: {args.popsizes}")
    print(f"  Runs per popsize: {args.n_runs}")
    print(f"  Success threshold: {args.success_threshold}")
    print(f"  PSO (w,c1,c2)=({args.w},{args.c1},{args.c2})")

    synth = BatchedModalPlateTorch(device=device, dtype=dtype).to(device)

    dset_root = Path(args.dset_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(dset_root.glob("*.npz"))
    if args.n_samples is not None:
        npz_files = npz_files[: args.n_samples]

    base_rng = np.random.default_rng(args.seed)

    for ir_idx, npz_file in enumerate(npz_files):
        ir_name = npz_file.stem
        print(f"\n[{ir_idx + 1}/{len(npz_files)}] {npz_file.name}")

        target_ir = load_target_ir_from_npz(npz_file, args.duration, SAMPLE_RATE)
        rows = []

        for pop in args.popsizes:
            best_losses = []
            runtimes = []
            successes = 0
            success_indices: list[int] = []

            for run_idx in range(args.n_runs):
                seed = int(base_rng.integers(0, 2_147_483_647))

                t0 = time.time()
                best_loss = run_pso_once(
                    target_ir=target_ir,
                    duration=args.duration,
                    synth=synth,
                    loss_fn=loss_fn,
                    popsize=int(pop),
                    budget=int(args.budget),
                    seed=seed,
                    dtype=dtype,
                    device=device,
                    w=float(args.w),
                    c1=float(args.c1),
                    c2=float(args.c2),
                )
                dt = time.time() - t0

                best_losses.append(best_loss)
                runtimes.append(dt)
                if best_loss < args.success_threshold:
                    successes += 1
                    success_indices.append(run_idx)

            best_losses_np = np.asarray(best_losses, dtype=np.float64)
            runtimes_np = np.asarray(runtimes, dtype=np.float64)
            row = {
                "ir_name": ir_name,
                "popsize": int(pop),
                "n_runs": int(args.n_runs),
                "n_success": int(successes),
                "success_rate": float(successes / max(1, args.n_runs)),
                "successful_run_indices": ";".join(str(i) for i in success_indices),
                "success_threshold": float(args.success_threshold),
                "mean_best_loss": float(np.mean(best_losses_np)),
                "std_best_loss": float(np.std(best_losses_np)),
                "min_best_loss": float(np.min(best_losses_np)),
                "max_best_loss": float(np.max(best_losses_np)),
                "total_runtime_s": float(np.sum(runtimes_np)),
                "avg_runtime_s": float(np.mean(runtimes_np)),
            }
            rows.append(row)
            print(
                f"  pop={pop:4d} success={row['n_success']:2d}/{args.n_runs:2d} "
                f"rate={row['success_rate']:.3f} mean_loss={row['mean_best_loss']:.5f}"
            )

        save_ir_csv(rows, out_dir / f"{ir_name}_success_rate.csv")
        save_ir_plot(rows, out_dir / f"{ir_name}_success_rate.png", ir_name)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dset_root", type=str, default="data/random-IR-100-1.0s")
    p.add_argument("--output_dir", type=str, default="results/diagnostics/success_rate_pso")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--loss", type=str, default="L1_STFT")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])
    p.add_argument("--budget", type=int, default=25000)
    p.add_argument("--n_runs", type=int, default=20, help="Runs per population size")
    p.add_argument(
        "--popsizes",
        type=int,
        nargs="+",
        default=[10, 30, 50, 100],
        help="Population sizes to evaluate",
    )
    p.add_argument("--success_threshold", type=float, default=0.01)
    p.add_argument("--w", type=float, default=0.2)
    p.add_argument("--c1", type=float, default=2.0)
    p.add_argument("--c2", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
