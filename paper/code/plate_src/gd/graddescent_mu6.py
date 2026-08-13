import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.loss.loss_selector import select_loss_function
from src.plate.SixParamPlate import BatchedModalPlateTorch

SAMPLE_RATE = 44100

# Bounds from challenge ranges.
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


class OOMTrialAbort(RuntimeError):
    pass


def _collect_param_csvs(dataset_dir: Path) -> List[Path]:
    return sorted(dataset_dir.glob("random_IR_params_*.csv"))


def _id_from_params_path(params_path: Path) -> str:
    return params_path.stem.split("_")[-1]


def _read_params_csv(csv_path: Path) -> Dict[str, float]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return {k: float(v) for k, v in row.items()}


def _load_npz_ir(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "ir" not in data.files:
        raise KeyError(f"Expected key 'ir' in {npz_path}, found keys: {data.files}")
    return np.asarray(data["ir"], dtype=np.float64)


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


def _compute_nmse_six(target_six: Dict[str, float], est_six: Dict[str, float]) -> float:
    errs = []
    for k in SIX_KEYS:
        lo, hi = SIX_BOUNDS[k]
        span = max(hi - lo, 1e-18)
        t = (target_six[k] - lo) / span
        e = (est_six[k] - lo) / span
        errs.append((e - t) ** 2)
    return float(np.mean(errs))


def _plot_loss_curve(histories: Sequence[Sequence[float]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for run_idx, hist in enumerate(histories, start=1):
        y = np.asarray(hist, dtype=np.float64)
        x = np.arange(len(y), dtype=np.int64)
        ax.plot(x, y, linewidth=1.2, alpha=0.85, label=f"run {run_idx}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(histories) > 1:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_per_ir_diagnostic(
    ir_id: str,
    histories: Sequence[Sequence[float]],
    target_six: Dict[str, float],
    est_six: Dict[str, float],
    elapsed_s: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for run_idx, hist in enumerate(histories, start=1):
        y = np.asarray(hist, dtype=np.float64)
        ax.plot(np.arange(len(y)), y, linewidth=0.8, alpha=0.6, label=f"run {run_idx}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Optimization Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    keys = list(SIX_KEYS)
    x = np.arange(len(keys), dtype=np.float64)
    w = 0.35
    lo = np.asarray([SIX_BOUNDS[k][0] for k in keys], dtype=np.float64)
    hi = np.asarray([SIX_BOUNDS[k][1] for k in keys], dtype=np.float64)
    span = np.maximum(hi - lo, 1e-18)
    tgt = np.asarray([target_six[k] for k in keys], dtype=np.float64)
    est = np.asarray([est_six[k] for k in keys], dtype=np.float64)
    tgt_n = np.clip((tgt - lo) / span, 0.0, 1.0)
    est_n = np.clip((est - lo) / span, 0.0, 1.0)
    ax.bar(x - w / 2, est_n, width=w, label="Est", color="steelblue")
    ax.bar(x + w / 2, tgt_n, width=w, label="GT", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalized value")
    ax.set_title(f"6 Params (NMSE={_compute_nmse_six(target_six, est_six):.4f})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.suptitle(f"random_IR_{ir_id} ({elapsed_s:.0f}s)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_summary_diagnostic(rows: Sequence[Dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [str(r.get("filename", f"IR_{r['id']}"))[:16] for r in rows]
    losses = [float(r["best_loss"]) for r in rows]
    axes[0].barh(names, losses, color="steelblue")
    axes[0].set_xlabel("Loss")
    axes[0].set_title("Loss per IR")
    axes[0].invert_yaxis()

    mu_err = []
    lo, hi = SIX_BOUNDS["mu"]
    span = max(hi - lo, 1e-18)
    for r in rows:
        mu_err.append(abs(float(r["est_mu"]) - float(r["target_mu"])) / span)
    axes[1].plot(np.arange(len(mu_err)), mu_err, marker="o", linewidth=1.0)
    axes[1].set_xlabel("IR index")
    axes[1].set_ylabel("Abs. normalized mu error")
    axes[1].set_title("Mu Estimation Error")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def _optimise_mu_only(target_ir_np, gt_six, args, plate, loss_fn, device, dtype, init_mu):
    n_samples = int(target_ir_np.shape[0])
    duration = n_samples / float(SAMPLE_RATE)

    target = torch.as_tensor(target_ir_np, dtype=dtype, device=device)
    if args.normalize_target:
        target = target / torch.clamp(target.abs().max(), min=1e-12)

    mu = nn.Parameter(torch.tensor(float(init_mu), dtype=dtype, device=device))
    optim = torch.optim.Adam([mu], lr=args.lr, betas=(args.adam_beta1, 0.999))

    hist = []
    best_loss = float("inf")
    best_mu = float(init_mu)

    try:
        for epoch in range(args.n_epochs):
            optim.zero_grad()
            mu_clamped = torch.clamp(mu, MU_MIN, MU_MAX)
            six_batch = torch.tensor(
                [[gt_six["mu"], gt_six["D_div_mu"], gt_six["T0_div_mu"], gt_six["Ly"], gt_six["op_x"], gt_six["op_y"]]],
                dtype=dtype,
                device=device,
            )
            six_batch[0, 0] = mu_clamped
            pred = plate(six_batch, duration=duration, vel_calc=args.vel_calc, normalize=False)[0]
            if args.normalize_pred:
                pred = pred / torch.clamp(pred.abs().max(), min=1e-12)

            loss = loss_fn(target.float().unsqueeze(0), pred.float().unsqueeze(0))[0]
            loss.backward()

            if args.grad_clip_type == "norm":
                torch.nn.utils.clip_grad_norm_([mu], max_norm=args.grad_clip_value)
            else:
                torch.nn.utils.clip_grad_value_([mu], clip_value=args.grad_clip_value)

            optim.step()
            with torch.no_grad():
                mu.clamp_(MU_MIN, MU_MAX)

            lv = float(loss.item())
            hist.append(lv)
            if lv < best_loss:
                best_loss = lv
                best_mu = float(mu.detach().item())

            if epoch % max(1, args.n_epochs // 20) == 0:
                print(f"      iter {epoch:4d}/{args.n_epochs} loss={lv:.9f} mu={float(mu.item()):.6f}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise OOMTrialAbort("CUDA OOM during mu-only GD") from e
        raise

    return best_mu, best_loss, hist


def run(args):
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    req = str(args.device).strip().lower()
    if req == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(req)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Loss: {args.loss}")

    loss_fn = select_loss_function(args.loss, sample_rate=SAMPLE_RATE, device=device)
    plate = BatchedModalPlateTorch(sample_rate=SAMPLE_RATE, device=device, dtype=dtype, drop_sub_20hz_modes=False)

    param_files = _collect_param_csvs(dataset_dir)
    if not param_files:
        raise FileNotFoundError(f"No random_IR_params_*.csv found in {dataset_dir}")
    if args.num is not None:
        if args.num <= 0:
            raise ValueError("--num must be positive")
        param_files = param_files[: args.num]

    # Deterministic starts across the full mu range.
    starts = np.linspace(MU_MIN, MU_MAX, num=max(1, args.n_runs), dtype=np.float64)

    rows = []
    print(f"Processing {len(param_files)} IR(s) from {dataset_dir}")

    for i, params_csv in enumerate(param_files, start=1):
        rid = _id_from_params_path(params_csv)
        npz_path = dataset_dir / f"random_IR_{rid}.npz"
        if not npz_path.exists():
            print(f"[{i}/{len(param_files)}] missing {npz_path.name}, skipping")
            continue

        full_gt = _read_params_csv(params_csv)
        target_six = _full_to_six(full_gt)
        target_ir = _load_npz_ir(npz_path)

        print(f"[{i}/{len(param_files)}] random_IR_{rid}")
        t0 = time.time()

        best_loss = float("inf")
        best_mu = float(target_six["mu"])
        best_run = 0
        run_histories = []

        for run_idx, init_mu in enumerate(starts, start=1):
            print(f"    run {run_idx:02d}/{len(starts):02d} | init_mu={float(init_mu):.6f}")
            try:
                est_mu, loss_val, hist = _optimise_mu_only(
                    target_ir_np=target_ir,
                    gt_six=target_six,
                    args=args,
                    plate=plate,
                    loss_fn=loss_fn,
                    device=device,
                    dtype=dtype,
                    init_mu=float(init_mu),
                )
            except OOMTrialAbort:
                print(f"      [OOM] run {run_idx:02d} aborted")
                continue

            run_histories.append(hist)
            print(f"      run {run_idx:02d} best loss={loss_val:.9e} est_mu={est_mu:.6f}")
            if loss_val < best_loss:
                best_loss = loss_val
                best_mu = est_mu
                best_run = run_idx

        if not run_histories:
            print("      all runs failed (OOM), skipping IR")
            continue

        elapsed = time.time() - t0
        est_six = dict(target_six)
        est_six["mu"] = float(best_mu)

        _plot_loss_curve(
            run_histories,
            out_dir / f"loss_random_IR_{rid}.png",
            title=f"random_IR_{rid} | best loss={best_loss:.6e} (best run {best_run})",
        )
        _plot_per_ir_diagnostic(
            ir_id=rid,
            histories=run_histories,
            target_six=target_six,
            est_six=est_six,
            elapsed_s=elapsed,
            out_path=out_dir / f"random_IR_{rid}_diagnostic.png",
        )

        row = {
            "id": rid,
            "filename": f"random_IR_{rid}.npz",
            "best_loss": float(best_loss),
            "best_run": int(best_run),
            "runtime_s": float(elapsed),
            "iterations": int(args.n_epochs),
            "n_runs": int(args.n_runs),
            "loss_name": args.loss,
            "lr": float(args.lr),
            "adam_beta1": float(args.adam_beta1),
            "grad_clip_value": float(args.grad_clip_value),
            "target_mu": float(target_six["mu"]),
            "est_mu": float(best_mu),
            "mu_abs_error": float(abs(best_mu - target_six["mu"])),
            "mu_rel_error": float(abs(best_mu - target_six["mu"]) / max(abs(target_six["mu"]), 1e-18)),
        }
        for k in SIX_KEYS:
            row[f"target_{k}"] = float(target_six[k])
            row[f"est_{k}"] = float(est_six[k])
        for run_idx, hist in enumerate(run_histories, start=1):
            row[f"run{run_idx}_best_loss"] = float(np.min(np.asarray(hist, dtype=np.float64)))

        rows.append(row)
        pd.DataFrame([row]).to_csv(out_dir / f"result_random_IR_{rid}.csv", index=False)
        print(f"      best loss={best_loss:.9e} runtime={elapsed:.2f}s (best run {best_run})")

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
        _plot_summary_diagnostic(rows, out_dir / "summary_diagnostic.png")
    print(f"Done. Outputs written to {out_dir}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "random-IR-100-1.0s",
        help="Directory containing random_IR_params_*.csv and random_IR_*.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "gd" / "mu_only_6param",
        help="Directory for CSV outputs and plots",
    )
    parser.add_argument("--loss", type=str, default="L1_STFT", help="Loss name from src/loss/losses.py registry")
    parser.add_argument("--num", type=int, default=10, help="Number of IRs to optimize")

    parser.add_argument("--n-runs", type=int, default=10, help="Number of restart runs with different init mu")
    parser.add_argument("--n-epochs", type=int, default=300, help="Optimization steps per run")
    parser.add_argument("--lr", type=float, default=1e-2, help="Fixed Adam learning rate")
    parser.add_argument("--adam-beta1", dest="adam_beta1", type=float, default=0.9, help="Fixed Adam beta1")
    parser.add_argument("--grad-clip-value", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--grad-clip-type", type=str, default="norm", choices=["norm", "value"], help="Gradient clipping mode")

    parser.add_argument("--vel-calc", action="store_true", help="Use velocity output instead of displacement")
    parser.add_argument("--normalize-target", action="store_true", help="Peak-normalize target IR")
    parser.add_argument("--normalize-pred", action="store_true", help="Peak-normalize predicted IR")
    parser.add_argument("--device", type=str, default="auto", help="Torch device: auto, cpu, cuda, cuda:<idx>")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Torch dtype")
    parser.add_argument("--double", action="store_true", help="Convenience flag to force float64")

    args = parser.parse_args()
    if args.double:
        args.dtype = "float64"
    run(args)


if __name__ == "__main__":
    main()
