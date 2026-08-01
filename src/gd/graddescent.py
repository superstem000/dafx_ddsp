import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats.qmc import LatinHypercube

from src.loss.loss_selector import select_loss_function
from src.plate.SevenParamPlate import BatchedModalPlateTorch

optuna.logging.set_verbosity(optuna.logging.WARNING)

SAMPLE_RATE = 44100

# Fixed constants from challenge specification.
Lx = 1.0
NU = 0.25
T60_DC = 6.0
T60_F1 = 2.0
LOSS_F1 = 500.0
FP_X = 0.335
FP_Y = 0.467

# Bounds from challenge ranges.
H_MIN, H_MAX = 0.001, 0.005
RHO_MIN, RHO_MAX = 2430.0, 21230.0
T0_MIN, T0_MAX = 0.01, 1000.0
E_MIN, E_MAX = 6.7e10, 22.0e10
LY_MIN, LY_MAX = 1.1, 4.0
OP_X_MIN, OP_X_MAX = 0.51, 1.0
OP_Y_MIN, OP_Y_MAX = 0.51, 1.0

_D_SCALE = 12.0 * (1.0 - NU**2)
D_MIN = E_MIN * (H_MIN**3) / _D_SCALE
D_MAX = E_MAX * (H_MAX**3) / _D_SCALE

MU_MIN = RHO_MIN * H_MIN
MU_MAX = RHO_MAX * H_MAX

D_DIV_MU_MIN = D_MIN / MU_MAX
D_DIV_MU_MAX = D_MAX / MU_MIN
T0_DIV_MU_MIN = T0_MIN / MU_MAX
T0_DIV_MU_MAX = T0_MAX / MU_MIN

RAW7_KEYS: Sequence[str] = ("Ly", "h", "T0", "rho", "E", "op_x", "op_y")
RAW7_BOUNDS: Dict[str, Sequence[float]] = {
    "Ly": (LY_MIN, LY_MAX),
    "h": (H_MIN, H_MAX),
    "T0": (T0_MIN, T0_MAX),
    "rho": (RHO_MIN, RHO_MAX),
    "E": (E_MIN, E_MAX),
    "op_x": (OP_X_MIN, OP_X_MAX),
    "op_y": (OP_Y_MIN, OP_Y_MAX),
}

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
    """Signal that the current Optuna trial should abort due to CUDA OOM."""


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


def _raw7_from_full(full_params: Dict[str, float]) -> Dict[str, float]:
    return {k: float(full_params[k]) for k in RAW7_KEYS}


def _raw7_to_six(raw7: Dict[str, float]) -> Dict[str, float]:
    rho = float(raw7["rho"])
    h = float(raw7["h"])
    E = float(raw7["E"])
    T0 = float(raw7["T0"])
    Ly = float(raw7["Ly"])
    op_x = float(raw7["op_x"])
    op_y = float(raw7["op_y"])

    mu = rho * h
    D = E * (h**3) / (12.0 * (1.0 - NU**2))
    return {
        "mu": mu,
        "D_div_mu": D / mu,
        "T0_div_mu": T0 / mu,
        "Ly": Ly,
        "op_x": op_x,
        "op_y": op_y,
    }


def _linear_to_raw7_params(free_values: torch.Tensor, free_keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    return {k: free_values[i] for i, k in enumerate(free_keys)}


def _raw7_to_plate_batch(raw7: Dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    full_vec = torch.stack(
        [
            torch.as_tensor(Lx, dtype=dtype, device=device),
            raw7["Ly"],
            raw7["h"],
            raw7["T0"],
            raw7["rho"],
            raw7["E"],
            torch.as_tensor(NU, dtype=dtype, device=device),
            torch.as_tensor(T60_DC, dtype=dtype, device=device),
            torch.as_tensor(T60_F1, dtype=dtype, device=device),
            torch.as_tensor(LOSS_F1, dtype=dtype, device=device),
            torch.as_tensor(FP_X, dtype=dtype, device=device),
            torch.as_tensor(FP_Y, dtype=dtype, device=device),
            raw7["op_x"],
            raw7["op_y"],
        ]
    )
    return full_vec.unsqueeze(0)


def _raw7_tensor_to_float_dict(raw7_params: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().item()) for k, v in raw7_params.items()}


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


def _compute_nmse_six(target_six: Dict[str, float], est_six: Dict[str, float]) -> float:
    errs = []
    for k in SIX_KEYS:
        lo, hi = SIX_BOUNDS[k]
        span = max(hi - lo, 1e-18)
        t = (target_six[k] - lo) / span
        e = (est_six[k] - lo) / span
        errs.append((e - t) ** 2)
    return float(np.mean(errs))


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
        ax.plot(np.arange(len(y)), y, linewidth=0.6, alpha=0.35, label=f"run {run_idx}")
    if histories:
        best_curve = np.minimum.reduce([np.asarray(h, dtype=np.float64) for h in histories])
        ax.plot(np.arange(len(best_curve)), best_curve, linewidth=1.8, color="red", label="best per iter")
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

    errs = {k: [] for k in SIX_KEYS}
    for r in rows:
        for k in SIX_KEYS:
            lo, hi = SIX_BOUNDS[k]
            span = max(hi - lo, 1e-18)
            tv = float(r[f"target_{k}"])
            ev = float(r[f"est_{k}"])
            errs[k].append(abs(ev - tv) / span)

    x = np.arange(len(SIX_KEYS))
    axes[1].bar(
        x,
        [np.mean(errs[k]) for k in SIX_KEYS],
        yerr=[np.std(errs[k]) for k in SIX_KEYS],
        color="coral",
        capsize=4,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(SIX_KEYS, rotation=20)
    axes[1].set_ylabel("Absolute Norm Error")
    axes[1].set_title("Average Error")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def _random_reset_out_of_bounds(free_values: torch.Tensor, free_keys: Sequence[str]) -> None:
    inner_guard_keys = {"rho", "h", "E", "T0"}
    inner_coverage = 0.995
    half_trim = 0.5 * (1.0 - inner_coverage)

    for i, k in enumerate(free_keys):
        lo, hi = RAW7_BOUNDS[k]
        if k in inner_guard_keys:
            span = hi - lo
            reset_lo = lo + half_trim * span
            reset_hi = hi - half_trim * span
        else:
            reset_lo = lo
            reset_hi = hi
        v = free_values[i]
        if bool((v < reset_lo) or (v > reset_hi)):
            free_values[i] = (
                torch.as_tensor(reset_lo, dtype=free_values.dtype, device=free_values.device)
                + torch.rand((), dtype=free_values.dtype, device=free_values.device)
                * torch.as_tensor(reset_hi - reset_lo, dtype=free_values.dtype, device=free_values.device)
            )


def _generate_lhs_starts(n_starts: int, seed: int) -> np.ndarray:
    sampler = LatinHypercube(d=len(RAW7_KEYS), seed=seed)
    unit = sampler.random(n=n_starts)
    lo = np.array([RAW7_BOUNDS[k][0] for k in RAW7_KEYS], dtype=np.float64)
    hi = np.array([RAW7_BOUNDS[k][1] for k in RAW7_KEYS], dtype=np.float64)
    return lo + unit * (hi - lo)


def _optimise_one(
    target_ir_np: np.ndarray,
    init_raw7: Dict[str, float],
    alt_starts: Sequence[Dict[str, float]],
    args,
    plate,
    loss_fn,
    device: torch.device,
    dtype: torch.dtype,
    trial=None,
) -> Tuple[Dict[str, float], float, List[List[float]], int]:
    n_samples = int(target_ir_np.shape[0])
    duration = n_samples / float(SAMPLE_RATE)

    target = torch.as_tensor(target_ir_np, dtype=dtype, device=device)
    if args.normalize_target:
        target = target / torch.clamp(target.abs().max(), min=1e-12)

    free_keys = list(RAW7_KEYS)

    def scalar_loss(pred_1d: torch.Tensor, target_1d: torch.Tensor) -> torch.Tensor:
        pred_b = pred_1d.float().unsqueeze(0)
        target_b = target_1d.float().unsqueeze(0)
        return loss_fn(target_b, pred_b)[0]

    starts = [init_raw7] + list(alt_starts)
    best_loss_global = float("inf")
    best_raw7_global: Dict[str, float] = {}
    best_restart_idx = 0
    all_histories: List[List[float]] = []

    for restart_idx, start in enumerate(starts, start=1):
        free_values = nn.Parameter(torch.tensor([start[k] for k in free_keys], dtype=dtype, device=device))
        optim = torch.optim.Adam([free_values], lr=args.lr, betas=(args.adam_beta1, 0.999))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=args.patience, factor=0.5)

        best_loss_restart = float("inf")
        best_free_restart = free_values.detach().clone()
        hist: List[float] = []

        try:
            for epoch in range(args.n_epochs):
                optim.zero_grad()
                raw7 = _linear_to_raw7_params(free_values, free_keys)
                batch = _raw7_to_plate_batch(raw7, dtype=dtype, device=device)
                pred = plate(batch, duration=duration, vel_calc=args.vel_calc, normalize=False)[0]
                if args.normalize_pred:
                    pred = pred / torch.clamp(pred.abs().max(), min=1e-12)

                loss = scalar_loss(pred, target)
                loss.backward()

                if args.grad_clip_type == "norm":
                    torch.nn.utils.clip_grad_norm_([free_values], max_norm=args.grad_clip_value)
                else:
                    torch.nn.utils.clip_grad_value_([free_values], clip_value=args.grad_clip_value)

                optim.step()
                with torch.no_grad():
                    _random_reset_out_of_bounds(free_values, free_keys)

                lv = float(loss.item())
                sched.step(lv)
                hist.append(lv)

                if lv < best_loss_restart:
                    best_loss_restart = lv
                    best_free_restart = free_values.detach().clone()

                if trial is not None:
                    trial.report(best_loss_restart, step=epoch + (restart_idx - 1) * args.n_epochs)
                if epoch % max(1, args.n_epochs // 20) == 0:
                    print(
                        f"      restart {restart_idx:02d}/{len(starts):02d} "
                        f"iter {epoch:4d}/{args.n_epochs} loss={lv:.9f}"
                    )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"      [OOM] restart {restart_idx} encountered CUDA OOM; aborting current trial")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise OOMTrialAbort("CUDA OOM in current trial; moving to next Optuna trial") from e
            raise

        all_histories.append(hist)
        with torch.no_grad():
            best_raw7 = _linear_to_raw7_params(best_free_restart, free_keys)
        best_raw7_dict = _raw7_tensor_to_float_dict(best_raw7)

        print(f"      restart {restart_idx:02d} best loss={best_loss_restart:.9e}")
        if best_loss_restart < best_loss_global:
            best_loss_global = best_loss_restart
            best_raw7_global = best_raw7_dict
            best_restart_idx = restart_idx

        if trial is not None and trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        break

    if not best_raw7_global:
        return {k: float(init_raw7[k]) for k in RAW7_KEYS}, 1e6, all_histories if all_histories else [[1e6]], 0
    return best_raw7_global, best_loss_global, all_histories, best_restart_idx


def create_robust_storage(db_name: str = "gd_optuna.db"):
    import sqlite3
    from sqlalchemy import event
    from sqlalchemy.pool import StaticPool

    for attempt in range(10):
        try:
            conn = sqlite3.connect(db_name, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=60000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.close()
            break
        except sqlite3.OperationalError:
            time.sleep(2 + attempt)

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_name}",
        engine_kwargs={
            "connect_args": {"timeout": 60, "check_same_thread": False},
            "poolclass": StaticPool,
        },
    )

    @event.listens_for(storage.engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        del connection_record
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return storage


def run(args):
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_device = str(args.device).strip().lower()
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Loss: {args.loss}")
    print(f"Optuna trials per IR: {args.n_trials}")
    print(f"LHS seed: {args.lhs_seed}")
    print(f"OOM reset limit: {args.oom_reset_limit}")

    if device.type == "cuda":
        cuda_idx = 0 if device.index is None else int(device.index)
        print(f"CUDA selected: {torch.cuda.get_device_name(cuda_idx)} (index {cuda_idx})")

    loss_fn = select_loss_function(args.loss, sample_rate=SAMPLE_RATE, device=device)

    plate = BatchedModalPlateTorch(
        sample_rate=SAMPLE_RATE,
        device=device,
        dtype=dtype,
        drop_sub_20hz_modes=False,
    )

    param_files = _collect_param_csvs(dataset_dir)
    if not param_files:
        raise FileNotFoundError(f"No random_IR_params_*.csv found in {dataset_dir}")

    if args.num is not None:
        if args.num <= 0:
            raise ValueError("--num must be a positive integer")
        param_files = param_files[: args.num]

    lhs_starts = _generate_lhs_starts(args.n_trials, seed=args.lhs_seed)

    rows = []
    print(f"Processing {len(param_files)} IR(s) from {dataset_dir}")

    for i, params_csv in enumerate(param_files, start=1):
        rid = _id_from_params_path(params_csv)
        npz_path = dataset_dir / f"random_IR_{rid}.npz"
        if not npz_path.exists():
            print(f"[{i}/{len(param_files)}] missing {npz_path.name}, skipping")
            continue

        full_gt = _read_params_csv(params_csv)
        target_raw7 = _raw7_from_full(full_gt)
        target_six = _full_to_six(full_gt)
        target_ir = _load_npz_ir(npz_path)

        print(f"[{i}/{len(param_files)}] random_IR_{rid}")
        storage = create_robust_storage(args.storage)
        study = optuna.create_study(
            study_name=f"gd_{args.loss}_{rid}",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=max(5, args.n_epochs // 10), reduction_factor=2),
        )

        t0 = time.time()

        def objective(trial):
            trial_idx = trial.number % len(lhs_starts)
            x0 = lhs_starts[trial_idx]
            init_raw7 = {k: float(v) for k, v in zip(RAW7_KEYS, x0)}

            alt = []
            for k in range(args.oom_reset_limit):
                j = (trial_idx + k + 1) % len(lhs_starts)
                alt.append({kk: float(vv) for kk, vv in zip(RAW7_KEYS, lhs_starts[j])})

            try:
                est_raw7, best_loss, run_histories, best_restart = _optimise_one(
                    target_ir_np=target_ir,
                    init_raw7=init_raw7,
                    alt_starts=alt,
                    args=args,
                    plate=plate,
                    loss_fn=loss_fn,
                    device=device,
                    dtype=dtype,
                    trial=trial,
                )
            except OOMTrialAbort:
                raise optuna.exceptions.TrialPruned("Trial aborted due to CUDA OOM")
            trial.set_user_attr("est_raw7", est_raw7)
            trial.set_user_attr("run_histories", run_histories)
            trial.set_user_attr("best_restart", best_restart)
            return best_loss

        study.optimize(objective, n_trials=args.n_trials)

        best = study.best_trial
        est_raw7 = best.user_attrs["est_raw7"]
        run_histories = best.user_attrs["run_histories"]
        best_restart = int(best.user_attrs["best_restart"])
        best_loss = float(best.value)

        elapsed = time.time() - t0
        est_six = _raw7_to_six(est_raw7)

        # _plot_loss_curve(
        #     run_histories,
        #     out_dir / f"loss_random_IR_{rid}.png",
        #     title=f"random_IR_{rid} | best loss={best_loss:.6e} (best restart {best_restart})",
        # )
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
            "loss": best_loss,
            "best_loss": best_loss,
            "best_restart": best_restart,
            "optuna_trial": int(best.number),
            "runtime_s": elapsed,
            "iterations": args.n_epochs,
            "n_trials": args.n_trials,
            "loss_name": args.loss,
            "lr": float(args.lr),
            "adam_beta1": float(args.adam_beta1),
            "grad_clip_value": float(args.grad_clip_value),
        }

        for run_idx, hist in enumerate(run_histories, start=1):
            row[f"run{run_idx}_best_loss"] = float(np.min(np.asarray(hist, dtype=np.float64)))

        for k in RAW7_KEYS:
            row[f"target_raw_{k}"] = target_raw7[k]
            row[f"est_raw_{k}"] = est_raw7[k]
        for k in SIX_KEYS:
            row[f"target_{k}"] = target_six[k]
            row[f"est_{k}"] = est_six[k]

        rows.append(row)
        pd.DataFrame([row]).to_csv(out_dir / f"result_random_IR_{rid}.csv", index=False)
        print(f"      best loss={best_loss:.9e} runtime={elapsed:.2f}s (best restart {best_restart})")

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
        _plot_summary_diagnostic(rows, out_dir / "summary_diagnostic.png")
    print(f"Done. Outputs written to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Optuna-guided gradient descent over seven raw plate parameters with LHS restart starts "
            "and CUDA OOM reset handling"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "random-IR-100-1.0s",
        help="Directory containing random_IR_params_*.csv and random_IR_*.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "gd" / "graddescent",
        help="Directory for CSV outputs and plots",
    )
    parser.add_argument("--loss", type=str, default="L1_STFT", help="Loss name from src/loss/losses.py registry")
    parser.add_argument("--num", type=int, default=1, help="Number of IRs to optimize")

    parser.add_argument("--n-trials", type=int, default=40, help="Optuna trials per IR")
    parser.add_argument("--storage", type=str, default="gd_optuna.db", help="Optuna SQLite DB path")
    parser.add_argument("--lhs-seed", type=int, default=42, help="LHS seed for restart starts")
    parser.add_argument("--oom-reset-limit", type=int, default=3, help="Max extra LHS restarts on CUDA OOM")

    parser.add_argument("--n-epochs", type=int, default=300, help="Optimization steps per trial")
    parser.add_argument("--lr", type=float, default=1e-2, help="Fixed Adam learning rate (not optimized by Optuna)")
    parser.add_argument("--adam-beta1", dest="adam_beta1", type=float, default=0.9, help="Fixed Adam beta1 (not optimized by Optuna)")
    parser.add_argument(
        "--grad-clip-value",
        type=float,
        default=1.0,
        help="Fixed gradient clipping value (not optimized by Optuna)",
    )

    parser.add_argument("--patience", type=int, default=20, help="ReduceLROnPlateau patience in epochs")
    parser.add_argument("--grad-clip-type", type=str, default="norm", choices=["norm", "value"], help="Gradient clipping mode")
    parser.add_argument("--vel-calc", action="store_true", help="Use velocity output instead of displacement during synthesis")
    parser.add_argument("--normalize-target", action="store_true", help="Peak-normalize target IR before optimization")
    parser.add_argument("--normalize-pred", action="store_true", help="Peak-normalize predicted IR before loss")
    parser.add_argument("--device", type=str, default="auto", help="Torch device: auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Torch dtype")
    parser.add_argument("--double", action="store_true", help="Convenience flag to force float64")

    args = parser.parse_args()
    if args.double:
        args.dtype = "float64"
    run(args)


if __name__ == "__main__":
    main()
