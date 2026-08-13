"""PSO for mu-only fitting on SixParamPlate.

Run as:
    python -m src.baseline.fit_mu6_pso --loss L1_STFT ...
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
import numpy as np
import pandas as pd
import torch

from src.loss.loss_selector import available_loss_names, select_loss_function
from src.plate.SixParamPlate import BatchedModalPlateTorch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100

H_MIN, H_MAX = 0.001, 0.005
RHO_MIN, RHO_MAX = 2430.0, 21230.0
T0_MIN, T0_MAX = 0.01, 1000.0
E_MIN, E_MAX = 6.7e10, 22.0e10
LY_MIN, LY_MAX = 1.1, 4.0
OP_X_MIN, OP_X_MAX = 0.51, 1.0
OP_Y_MIN, OP_Y_MAX = 0.51, 1.0
NU = 0.25

_D_SCALE = 12.0 * (1.0 - NU**2)
D_MIN = E_MIN * (H_MIN**3) / _D_SCALE
D_MAX = E_MAX * (H_MAX**3) / _D_SCALE
MU_MIN = RHO_MIN * H_MIN
MU_MAX = RHO_MAX * H_MAX
D_DIV_MU_MIN = D_MIN / MU_MAX
D_DIV_MU_MAX = D_MAX / MU_MIN
T0_DIV_MU_MIN = T0_MIN / MU_MAX
T0_DIV_MU_MAX = T0_MAX / MU_MIN

SIX_KEYS: Sequence[str] = ("mu", "D_div_mu", "T0_div_mu", "Ly", "op_x", "op_y")
SIX_BOUNDS: Dict[str, Sequence[float]] = {
    "mu": (MU_MIN, MU_MAX),
    "D_div_mu": (D_DIV_MU_MIN, D_DIV_MU_MAX),
    "T0_div_mu": (T0_DIV_MU_MIN, T0_DIV_MU_MAX),
    "Ly": (LY_MIN, LY_MAX),
    "op_x": (OP_X_MIN, OP_X_MAX),
    "op_y": (OP_Y_MIN, OP_Y_MAX),
}


def _collect_param_csvs(dataset_dir: Path) -> List[Path]:
    return sorted(dataset_dir.glob("random_IR_params_*.csv"))


def _id_from_params_path(params_path: Path) -> str:
    return params_path.stem.split("_")[-1]


def _read_params_csv(csv_path: Path) -> Dict[str, float]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return {k: float(v) for k, v in row.items()}


def _load_target_ir_from_npz(npz_path: Path, duration_s: float, expected_sr: int) -> np.ndarray:
    with np.load(npz_path) as data:
        if "ir" not in data.files:
            raise KeyError(f"Expected key 'ir' in {npz_path}, found keys: {data.files}")
        ir = np.asarray(data["ir"])
        sr = int(np.asarray(data["sample_rate"]).item()) if "sample_rate" in data else expected_sr

    if sr != expected_sr:
        raise ValueError(f"Sample rate mismatch in {npz_path.name}: {sr} (expected {expected_sr})")

    if ir.ndim > 1:
        ir = np.mean(ir, axis=-1)
    ir = ir.astype(np.float64, copy=False)
    ir = ir[: int(duration_s * expected_sr)]
    return ir


def _full_to_six(full_params: Dict[str, float]) -> Dict[str, float]:
    rho = float(full_params["rho"])
    h = float(full_params["h"])
    E = float(full_params["E"])
    T0 = float(full_params["T0"])
    Ly = float(full_params["Ly"])
    op_x = float(full_params["op_x"])
    op_y = float(full_params["op_y"])
    nu = float(full_params["nu"])

    mu = rho * h
    D = E * (h**3) / (12.0 * (1.0 - nu**2))
    return {
        "mu": mu,
        "D_div_mu": D / mu,
        "T0_div_mu": T0 / mu,
        "Ly": Ly,
        "op_x": op_x,
        "op_y": op_y,
    }


def _nmse_6d(est_6: Dict[str, float], gt_6: Dict[str, float]) -> float:
    errs = []
    for key in SIX_KEYS:
        lo, hi = SIX_BOUNDS[key]
        e = (est_6[key] - lo) / (hi - lo)
        g = (gt_6[key] - lo) / (hi - lo)
        errs.append((e - g) ** 2)
    return float(np.mean(errs))


def _evaluate_mu_batch(
    mu_batch: np.ndarray,
    gt_6: Dict[str, float],
    target_t: torch.Tensor,
    duration: float,
    synth,
    loss_fn,
    device: str,
    dtype: torch.dtype,
) -> np.ndarray:
    mu_batch = np.clip(mu_batch, MU_MIN, MU_MAX)
    six = np.tile(
        np.array([gt_6["mu"], gt_6["D_div_mu"], gt_6["T0_div_mu"], gt_6["Ly"], gt_6["op_x"], gt_6["op_y"]], dtype=np.float64),
        (len(mu_batch), 1),
    )
    six[:, 0] = mu_batch
    six_t = torch.tensor(six, dtype=dtype, device=device)
    with torch.no_grad():
        pred = synth(six_t, duration, normalize=False)
        losses = loss_fn(target_t.expand(len(mu_batch), -1), pred).detach().cpu().numpy()
    return np.nan_to_num(losses, nan=1e6, posinf=1e6, neginf=1e6)


def run_pso_mu_only(
    target_ir: np.ndarray,
    gt_6: Dict[str, float],
    duration: float,
    synth,
    loss_fn,
    device: str,
    dtype: torch.dtype,
    num_particles: int,
    max_iter: int,
    w: float,
    c1: float,
    c2: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    target_t = torch.tensor(target_ir, dtype=dtype, device=device).unsqueeze(0)

    particles = rng.uniform(MU_MIN, MU_MAX, size=(num_particles,))
    velocities = rng.uniform(-0.1 * (MU_MAX - MU_MIN), 0.1 * (MU_MAX - MU_MIN), size=(num_particles,))

    pbest = particles.copy()
    pbest_scores = _evaluate_mu_batch(particles, gt_6, target_t, duration, synth, loss_fn, device, dtype)

    gidx = int(np.argmin(pbest_scores))
    gbest = float(pbest[gidx])
    gbest_score = float(pbest_scores[gidx])

    loss_history = pbest_scores.tolist()

    for _ in range(max_iter):
        r1 = rng.random(size=(num_particles,))
        r2 = rng.random(size=(num_particles,))
        velocities = w * velocities + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest - particles)
        particles = np.clip(particles + velocities, MU_MIN, MU_MAX)

        scores = _evaluate_mu_batch(particles, gt_6, target_t, duration, synth, loss_fn, device, dtype)
        loss_history.extend(scores.tolist())

        improved = scores < pbest_scores
        pbest[improved] = particles[improved]
        pbest_scores[improved] = scores[improved]

        gidx = int(np.argmin(pbest_scores))
        if float(pbest_scores[gidx]) < gbest_score:
            gbest = float(pbest[gidx])
            gbest_score = float(pbest_scores[gidx])

    return gbest, gbest_score, loss_history


def _plot_per_ir(stem: str, est_6: Dict[str, float], gt_6: Dict[str, float], loss_hist: List[float], elapsed: float, out_path: Path, gt_loss=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if loss_hist:
        ax.plot(loss_hist, lw=0.3, alpha=0.3, color="blue", label="all evals")
        ax.plot(np.minimum.accumulate(loss_hist), lw=1.5, color="red", label="best so far")
        if gt_loss is not None:
            ax.axhline(gt_loss, color="green", linestyle="--", alpha=0.5, label=f"GT loss={gt_loss:.4f}")
    ax.set_xlabel("Eval")
    ax.set_ylabel("Loss")
    ax.set_title("Optimization Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(len(SIX_KEYS))
    width = 0.35
    e_norm = [(est_6[k] - SIX_BOUNDS[k][0]) / (SIX_BOUNDS[k][1] - SIX_BOUNDS[k][0]) for k in SIX_KEYS]
    g_norm = [(gt_6[k] - SIX_BOUNDS[k][0]) / (SIX_BOUNDS[k][1] - SIX_BOUNDS[k][0]) for k in SIX_KEYS]
    ax.bar(x - width / 2, e_norm, width, label="Est", color="steelblue")
    ax.bar(x + width / 2, g_norm, width, label="GT", color="coral")
    ax.set_title(f"6 Params (NMSE={_nmse_6d(est_6, gt_6):.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(SIX_KEYS, rotation=45)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"{stem} ({elapsed:.0f}s)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path / f"{stem}_diagnostic.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(results: list, out_path: Path):
    if not results:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [r["filename"][:16] for r in results]
    losses = [r["best_loss"] for r in results]
    axes[0].barh(names, losses, color="steelblue")
    axes[0].set_xlabel("Loss")
    axes[0].set_title("Loss per IR")
    axes[0].invert_yaxis()

    mu_errs = []
    mu_span = SIX_BOUNDS["mu"][1] - SIX_BOUNDS["mu"][0]
    for r in results:
        mu_errs.append(abs(float(r["mu"]) - float(r["gt_mu"])) / mu_span)

    axes[1].plot(np.arange(len(mu_errs)), mu_errs, marker="o", linewidth=1.0)
    axes[1].set_xlabel("IR index")
    axes[1].set_ylabel("Abs. normalized mu error")
    axes[1].set_title("Mu Estimation Error")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path / "summary_diagnostic.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(args):
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    loss_fn = select_loss_function(args.loss, sample_rate=SAMPLE_RATE, device=device)

    print(f"PSO mu-only (SixParamPlate) on {device.upper()}")
    print(f"  Loss: {args.loss}")
    print(f"  Available losses: {', '.join(available_loss_names())}")
    print(f"  Dtype: {dtype}")
    print(f"  Particles: {args.num_particles}")
    print(f"  Iterations: {args.max_iter}")
    print(f"  Duration: {args.duration}s")

    synth = BatchedModalPlateTorch(sample_rate=SAMPLE_RATE, device=device, dtype=dtype, drop_sub_20hz_modes=False).to(device)

    dataset_dir = Path(args.dset_root)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    param_files = _collect_param_csvs(dataset_dir)
    if args.n_samples is not None:
        param_files = param_files[: args.n_samples]

    all_results = []

    for idx, params_csv in enumerate(param_files):
        rid = _id_from_params_path(params_csv)
        npz_file = dataset_dir / f"random_IR_{rid}.npz"
        if not npz_file.exists():
            continue

        result_path = output_path / f"result_random_IR_{rid}.csv"
        if result_path.exists() and not args.overwrite:
            continue

        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(param_files)}] {npz_file.name}")
        target_ir = _load_target_ir_from_npz(npz_file, args.duration, SAMPLE_RATE)

        gt_full = _read_params_csv(params_csv)
        gt_6 = _full_to_six(gt_full)

        gt_loss = None
        try:
            with torch.no_grad():
                gt_batch = torch.tensor([[gt_6[k] for k in SIX_KEYS]], dtype=dtype, device=device)
                gt_audio = synth(gt_batch, args.duration, normalize=False)
                gt_target = torch.tensor(target_ir, dtype=dtype, device=device).unsqueeze(0)
                gt_loss = float(loss_fn(gt_target, gt_audio).item())
            print(f"  GT loss: {gt_loss:.6f}")
        except Exception as e:
            print(f"  Could not calc GT: {e}")

        t0 = time.time()
        best_mu, best_loss, loss_hist = run_pso_mu_only(
            target_ir=target_ir,
            gt_6=gt_6,
            duration=args.duration,
            synth=synth,
            loss_fn=loss_fn,
            device=device,
            dtype=dtype,
            num_particles=args.num_particles,
            max_iter=args.max_iter,
            w=args.w,
            c1=args.c1,
            c2=args.c2,
            seed=args.seed + idx,
        )
        elapsed = time.time() - t0

        est_6 = dict(gt_6)
        est_6["mu"] = best_mu

        print(f"  Winner loss={best_loss:.6f} ({elapsed:.0f}s)")
        print(f"    mu: {best_mu:.6f}")
        print(f"    NMSE: {_nmse_6d(est_6, gt_6):.6f}")

        row = {
            "filename": npz_file.name,
            "target_npz": npz_file.name,
            "runtime": round(float(elapsed), 3),
            "best_loss": round(float(best_loss), 8),
            "optimizer": "PSO_MU_ONLY",
            "loss_name": args.loss,
            "num_particles": int(args.num_particles),
            "max_iter": int(args.max_iter),
            "w": float(args.w),
            "c1": float(args.c1),
            "c2": float(args.c2),
            "gt_loss": round(gt_loss, 8) if gt_loss is not None else None,
            "nmse": round(_nmse_6d(est_6, gt_6), 8),
            "mu": round(float(best_mu), 8),
            "gt_mu": round(float(gt_6["mu"]), 8),
            "mu_abs_error": round(abs(float(best_mu) - float(gt_6["mu"])), 8),
            "mu_rel_error": round(abs(float(best_mu) - float(gt_6["mu"])) / max(abs(float(gt_6["mu"])), 1e-18), 8),
        }
        for k in SIX_KEYS:
            row[f"est_{k}"] = round(float(est_6[k]), 8)
            row[f"gt_{k}"] = round(float(gt_6[k]), 8)

        all_results.append(row)

        pd.DataFrame([row]).to_csv(result_path, index=False)
        _plot_per_ir(npz_file.stem, est_6, gt_6, loss_hist, elapsed, output_path, gt_loss=gt_loss)
        print(f"  [Saved] {npz_file.stem}")

    if all_results:
        pd.DataFrame(all_results).to_csv(output_path / "summary.csv", index=False)
        _plot_summary(all_results, output_path)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dset_root", type=str, default="data/random-IR-100-1.0s")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="results/pso/mu_only")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--loss", type=str, default="L1_STFT")
    parser.add_argument("--num_particles", type=int, default=40)
    parser.add_argument("--max_iter", type=int, default=300)
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
