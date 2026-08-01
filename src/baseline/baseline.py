"""Baseline-style PSO optimization for seven-parameter plate fitting.

This mirrors the output structure of src/cmaes/fit_7param.py while using:
- Optimizer: PSO (baseline style)
- Loss: configurable from src/loss/losses.py
- Synth: src/plate/SevenParamPlate.BatchedModalPlateTorch
- Targets: random_IR_XXXX.npz

Run as:
    python -m src.baseline.baseline
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

from src.loss.loss_selector import select_loss_function
from src.plate.SevenParamPlate import BatchedModalPlateTorch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100
NU = 0.25

PARAM_KEYS = ["E", "rho", "h", "Ly", "T0", "op_x", "op_y"]
PARAM_BOUNDS = {
    "E": (6.7e10, 2.2e11),
    "rho": (2430.0, 21230.0),
    "h": (0.001, 0.005),
    "Ly": (1.1, 4.0),
    "T0": (0.01, 1000.0),
    "op_x": (0.51, 1.0),
    "op_y": (0.51, 1.0),
}

BOUNDS_LO_NP = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS], dtype=np.float64)
BOUNDS_HI_NP = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS], dtype=np.float64)
BOUNDS_RANGE_NP = BOUNDS_HI_NP - BOUNDS_LO_NP

COMPOSITE_KEYS = ["mu", "D_mu", "T0_mu", "Ly", "op_x", "op_y"]
COMPOSITE_BOUNDS = {
    "mu": (2.43, 106.15),
    "D_mu": (0.28052546, 201.188843),
    "T0_mu": (0.000094206, 411.522634),
    "Ly": (1.1, 4.0),
    "op_x": (0.51, 1.0),
    "op_y": (0.51, 1.0),
}

FIXED_PLATE_PARAMS = {
    "Lx": 1.0,
    "nu": 0.25,
    "T60_DC": 6.0,
    "T60_F1": 2.0,
    "loss_F1": 500.0,
    "fp_x": 0.335,
    "fp_y": 0.467,
}


def to_composite(p: dict) -> dict:
    mu = p["rho"] * p["h"]
    D = (p["E"] * p["h"] ** 3) / (12 * (1 - NU**2))
    return {
        "mu": mu,
        "D_mu": D / mu,
        "T0_mu": p["T0"] / mu,
        "Ly": p["Ly"],
        "op_x": p["op_x"],
        "op_y": p["op_y"],
    }


def compute_nmse_6d(est_7: dict, gt_7: dict) -> float:
    e6, g6 = to_composite(est_7), to_composite(gt_7)
    errs = []
    for key in COMPOSITE_KEYS:
        lo, hi = COMPOSITE_BOUNDS[key]
        e = (e6[key] - lo) / (hi - lo)
        g = (g6[key] - lo) / (hi - lo)
        errs.append((e - g) ** 2)
    return float(np.mean(errs))


def params_np_to_dict(params_row: np.ndarray) -> dict:
    return {k: float(v) for k, v in zip(PARAM_KEYS, params_row)}


def load_target_ir_from_npz(npz_path: Path, duration_s: float, expected_sr: int) -> np.ndarray:
    with np.load(npz_path) as data:
        if "ir" in data:
            ir = np.asarray(data["ir"])
        else:
            chosen = None
            for key in ("audio", "y", "signal", "target"):
                if key in data:
                    chosen = key
                    break
            if chosen is None:
                arr_keys = [k for k in data.files if np.asarray(data[k]).ndim >= 1]
                if not arr_keys:
                    raise ValueError(f"No 1D/2D array field found in {npz_path}")
                chosen = arr_keys[0]
            ir = np.asarray(data[chosen])

        if ir.ndim > 1:
            ir = np.mean(ir, axis=-1)

        sr = int(np.asarray(data["sample_rate"]).item()) if "sample_rate" in data else expected_sr

    if sr != expected_sr:
        raise ValueError(f"Sample rate mismatch in {npz_path.name}: {sr} (expected {expected_sr})")

    ir = ir.astype(np.float64, copy=False)
    ir = ir[: int(duration_s * expected_sr)]
    ir /= max(np.max(np.abs(ir)), 1e-15)
    return ir


def physical_to_plate14_tensor(phys_batch_np: np.ndarray, device: str | torch.device) -> torch.Tensor:
    E = phys_batch_np[:, 0]
    rho = phys_batch_np[:, 1]
    h = phys_batch_np[:, 2]
    Ly = phys_batch_np[:, 3]
    T0 = phys_batch_np[:, 4]
    op_x = phys_batch_np[:, 5]
    op_y = phys_batch_np[:, 6]

    cols = np.stack(
        [
            np.full_like(E, FIXED_PLATE_PARAMS["Lx"]),
            Ly,
            h,
            T0,
            rho,
            E,
            np.full_like(E, FIXED_PLATE_PARAMS["nu"]),
            np.full_like(E, FIXED_PLATE_PARAMS["T60_DC"]),
            np.full_like(E, FIXED_PLATE_PARAMS["T60_F1"]),
            np.full_like(E, FIXED_PLATE_PARAMS["loss_F1"]),
            np.full_like(E, FIXED_PLATE_PARAMS["fp_x"]),
            np.full_like(E, FIXED_PLATE_PARAMS["fp_y"]),
            op_x,
            op_y,
        ],
        axis=1,
    )
    return torch.tensor(cols, dtype=torch.float32, device=device)


def _normalized_to_physical(x_norm: np.ndarray) -> np.ndarray:
    return BOUNDS_LO_NP + x_norm * BOUNDS_RANGE_NP


def _evaluate_batch(
    x_norm_batch: np.ndarray,
    target_t: torch.Tensor,
    duration: float,
    synth: BatchedModalPlateTorch,
    loss_fn,
    device: str,
) -> np.ndarray:
    phys = _normalized_to_physical(x_norm_batch)
    plate_params = physical_to_plate14_tensor(phys, device)
    with torch.no_grad():
        audios = synth(plate_params, duration)
        losses = loss_fn(target_t.expand(phys.shape[0], -1), audios).detach().cpu().numpy()
    losses = np.nan_to_num(losses, nan=1e6, posinf=1e6, neginf=1e6)
    return losses


def run_pso(
    target_ir: np.ndarray,
    duration: float,
    synth: BatchedModalPlateTorch,
    loss_fn,
    device: str,
    num_particles: int,
    max_iter: int,
    w: float,
    c1: float,
    c2: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    target_t = torch.tensor(target_ir, dtype=torch.float32, device=device).unsqueeze(0)

    dim = len(PARAM_KEYS)
    particles = rng.uniform(0.0, 1.0, size=(num_particles, dim))
    velocities = rng.uniform(-0.1, 0.1, size=(num_particles, dim))

    personal_best = particles.copy()
    personal_best_scores = _evaluate_batch(particles, target_t, duration, synth, loss_fn, device)

    best_idx = int(np.argmin(personal_best_scores))
    global_best = personal_best[best_idx].copy()
    global_best_score = float(personal_best_scores[best_idx])

    # Keep dense history similar to CMA-ES traces (all particle losses each iteration).
    loss_history = personal_best_scores.tolist()

    for _iter in range(max_iter):
        r1 = rng.random(size=(num_particles, dim))
        r2 = rng.random(size=(num_particles, dim))

        velocities = (
            w * velocities
            + c1 * r1 * (personal_best - particles)
            + c2 * r2 * (global_best[np.newaxis, :] - particles)
        )
        particles = np.clip(particles + velocities, 0.0, 1.0)

        scores = _evaluate_batch(particles, target_t, duration, synth, loss_fn, device)
        loss_history.extend(scores.tolist())

        improved = scores < personal_best_scores
        personal_best[improved] = particles[improved]
        personal_best_scores[improved] = scores[improved]

        best_idx = int(np.argmin(personal_best_scores))
        if personal_best_scores[best_idx] < global_best_score:
            global_best = personal_best[best_idx].copy()
            global_best_score = float(personal_best_scores[best_idx])

    best_phys = _normalized_to_physical(global_best[np.newaxis, :])[0]
    return params_np_to_dict(best_phys), global_best_score, loss_history


def plot_per_ir(stem, est_7, gt_7, loss_hist, elapsed, out_path, gt_loss=None):
    est_6 = to_composite(est_7)
    gt_6 = to_composite(gt_7) if gt_7 else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if loss_hist:
        ax.plot(loss_hist, lw=0.3, alpha=0.3, color="blue", label="all evals")
        ax.plot(np.minimum.accumulate(loss_hist), lw=1.5, color="red", label="best so far")
        if gt_loss is not None:
            ax.axhline(gt_loss, color="green", linestyle="--", alpha=0.5, label=f"GT loss={gt_loss:.4f}")
    ax.set_xlabel("Eval")
    ax.set_ylabel("Loss")
    ax.set_title("MSS Convergence (PSO)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(len(COMPOSITE_KEYS))
    width = 0.35
    e_norm = [
        (est_6[k] - COMPOSITE_BOUNDS[k][0]) / (COMPOSITE_BOUNDS[k][1] - COMPOSITE_BOUNDS[k][0])
        for k in COMPOSITE_KEYS
    ]
    ax.bar(x - width / 2, e_norm, width, label="Est", color="steelblue")
    if gt_6:
        g_norm = [
            (gt_6[k] - COMPOSITE_BOUNDS[k][0]) / (COMPOSITE_BOUNDS[k][1] - COMPOSITE_BOUNDS[k][0])
            for k in COMPOSITE_KEYS
        ]
        ax.bar(x + width / 2, g_norm, width, label="GT", color="coral")
        ax.set_title(f"6 Params (NMSE={compute_nmse_6d(est_7, gt_7):.4f})")
    else:
        ax.set_title("6 Params")
    ax.set_xticks(x)
    ax.set_xticklabels(COMPOSITE_KEYS, rotation=45)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"{stem} ({elapsed:.0f}s)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path / f"{stem}_diagnostic.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_summary(results: list, out_path: Path):
    if not results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [r["filename"][:16] for r in results]
    losses = [r["best_loss"] for r in results]
    axes[0].barh(names, losses, color="steelblue")
    axes[0].set_xlabel("Loss")
    axes[0].set_title("Loss per IR")
    axes[0].invert_yaxis()

    if results[0].get("gt_available", False):
        errs = {k: [] for k in COMPOSITE_KEYS}
        for r in results:
            est_7 = {k: r[k] for k in PARAM_KEYS}
            gt_7 = {k: r[f"gt_{k}"] for k in PARAM_KEYS}
            e6, g6 = to_composite(est_7), to_composite(gt_7)
            for k in COMPOSITE_KEYS:
                errs[k].append(abs(e6[k] - g6[k]) / (COMPOSITE_BOUNDS[k][1] - COMPOSITE_BOUNDS[k][0]))

        x = np.arange(len(COMPOSITE_KEYS))
        axes[1].bar(
            x,
            [np.mean(errs[k]) for k in COMPOSITE_KEYS],
            yerr=[np.std(errs[k]) for k in COMPOSITE_KEYS],
            color="coral",
            capsize=4,
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(COMPOSITE_KEYS, rotation=45)
        axes[1].set_ylabel("Absolute Norm Error")
        axes[1].set_title("Average Error")

    plt.tight_layout()
    plt.savefig(out_path / "summary_diagnostic.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(args):
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    loss_fn = select_loss_function(args.loss, sample_rate=SAMPLE_RATE, device=device)

    print(f"Baseline PSO + {args.loss} on {device.upper()}")
    print(f"  Particles: {args.num_particles}")
    print(f"  Iterations: {args.max_iter}")
    print(f"  PSO (w, c1, c2): ({args.w}, {args.c1}, {args.c2})")
    print(f"  Duration: {args.duration}s")

    synth = BatchedModalPlateTorch(device=device).to(device)

    target_path = Path(args.dset_root)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(target_path.glob("*.npz"))[: args.n_samples]
    all_results = []

    for idx, npz_file in enumerate(npz_files):
        result_path = output_path / f"result_{npz_file.stem}.csv"
        if result_path.exists() and not args.overwrite:
            continue

        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(npz_files)}] {npz_file.name}")
        target_ir = load_target_ir_from_npz(npz_file, args.duration, SAMPLE_RATE)

        gt_loss, gt_7 = None, None
        gt_csv = target_path / npz_file.name.replace("random_IR_", "random_IR_params_").replace(".npz", ".csv")
        if gt_csv.exists():
            try:
                gt_7 = {k: float(pd.read_csv(gt_csv).iloc[0][k]) for k in PARAM_KEYS}
                with torch.no_grad():
                    gt_tensor7 = np.array([[gt_7[k] for k in PARAM_KEYS]], dtype=np.float64)
                    gt_plate = physical_to_plate14_tensor(gt_tensor7, device)
                    gt_audio = synth(gt_plate, args.duration)
                    gt_target = torch.tensor(target_ir, device=device).float().unsqueeze(0)
                    gt_loss = float(loss_fn(gt_target, gt_audio).item())
                print(f"  GT loss: {gt_loss:.6f}")
            except Exception as e:
                print(f"  Could not calc GT: {e}")

        t0 = time.time()
        best_params, best_loss, loss_hist = run_pso(
            target_ir=target_ir,
            duration=args.duration,
            synth=synth,
            loss_fn=loss_fn,
            device=device,
            num_particles=args.num_particles,
            max_iter=args.max_iter,
            w=args.w,
            c1=args.c1,
            c2=args.c2,
            seed=args.seed + idx,
        )
        elapsed = time.time() - t0

        comp = to_composite(best_params)
        print(f"  Winner loss={best_loss:.6f} ({elapsed:.0f}s)")
        for k in COMPOSITE_KEYS:
            print(f"    {k}: {comp[k]:.6f}")
        if gt_7:
            print(f"    NMSE: {compute_nmse_6d(best_params, gt_7):.6f}")

        row = {
            "filename": npz_file.name,
            "target_npz": npz_file.name,
            "runtime": round(elapsed, 3),
            "best_loss": round(float(best_loss), 8),
            "optimizer": "PSO",
            "loss_name": args.loss,
            "num_particles": int(args.num_particles),
            "max_iter": int(args.max_iter),
            "w": float(args.w),
            "c1": float(args.c1),
            "c2": float(args.c2),
            "gt_loss": round(gt_loss, 8) if gt_loss is not None else None,
            "nmse": round(compute_nmse_6d(best_params, gt_7), 8) if gt_7 else None,
            "gt_available": gt_7 is not None,
            "mu": round(comp["mu"], 8),
            "D_mu": round(comp["D_mu"], 8),
            "T0_mu": round(comp["T0_mu"], 8),
            **{k: round(float(v), 8) for k, v in best_params.items()},
        }
        if gt_7:
            for k, v in gt_7.items():
                row[f"gt_{k}"] = round(float(v), 8)

        all_results.append(row)

        pd.DataFrame([row]).to_csv(result_path, index=False)
        plot_per_ir(npz_file.stem, best_params, gt_7, loss_hist, elapsed, output_path, gt_loss)
        print(f"  [Saved] {npz_file.stem}")

    if all_results:
        pd.DataFrame(all_results).to_csv(output_path / "summary.csv", index=False)
        plot_summary(all_results, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dset_root", type=str, default="data/random-IR-100-1.0s")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="results/cmaes/baseline_pso_mss")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loss", type=str, default="MSS")
    parser.add_argument("--num_particles", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=5)
    parser.add_argument("--w", type=float, default=0.2)
    parser.add_argument("--c1", type=float, default=2.0)
    parser.add_argument("--c2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
