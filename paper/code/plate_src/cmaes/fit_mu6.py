"""CMA-ES + Optuna for mu/T0_div_mu fitting on SixParamPlate.

Run as:
    python -m src.cmaes.fit_mu6 --loss L1_STFT ...
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Sequence

import cma
import matplotlib
import numpy as np
import optuna
import pandas as pd
import torch
from scipy.stats.qmc import LatinHypercube

from src.loss.loss_selector import available_loss_names, select_loss_function
from src.plate.SixParamPlate import BatchedModalPlateTorch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    """Signal that current trial should abort due to CUDA OOM."""


def create_robust_storage(db_name: str = "cmaes_mu6.db"):
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
    ir /= max(np.max(np.abs(ir)), 1e-15)
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


def _generate_lhs_mu_t0_starts(n_trials: int, seed: int = 42) -> np.ndarray:
    sampler = LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n=n_trials)
    lo = np.array([MU_MIN, T0_DIV_MU_MIN], dtype=np.float64)
    hi = np.array([MU_MAX, T0_DIV_MU_MAX], dtype=np.float64)
    return lo + unit * (hi - lo)


def _run_cmaes_mu_t0(
    trial,
    target_ir: np.ndarray,
    duration: float,
    synth: BatchedModalPlateTorch,
    loss_fn,
    popsize: int,
    budget: int,
    seed: int,
    mu0: float,
    t0_div_mu0: float,
    gt_six: Dict[str, float],
    device: str,
):
    target_t = torch.tensor(target_ir, dtype=torch.float32, device=device).unsqueeze(0)

    sigma_mu = max((MU_MAX - MU_MIN) * 0.2, 1e-6)
    sigma_t0_div_mu = max((T0_DIV_MU_MAX - T0_DIV_MU_MIN) * 0.2, 1e-6)
    opts = {
        "maxfevals": budget,
        "popsize": popsize,
        "bounds": [[MU_MIN, T0_DIV_MU_MIN], [MU_MAX, T0_DIV_MU_MAX]],
        "seed": seed,
        "verb_disp": 0,
        "tolfun": 5e-3,
        "tolfunhist": 5e-3,
        "CMA_stds": [sigma_mu, sigma_t0_div_mu],
    }

    es = cma.CMAEvolutionStrategy([float(mu0), float(t0_div_mu0)], 1.0, opts)
    eval_count = 0
    loss_history = []
    trial_start = time.time()

    while not es.stop():
        gen_start = time.time()
        solutions = es.ask()
        sol = np.asarray(solutions, dtype=np.float64)
        mu_values = sol[:, 0]
        t0_div_mu_values = sol[:, 1]

        six_batch = np.tile(
            np.array(
                [
                    gt_six["mu"],
                    gt_six["D_div_mu"],
                    gt_six["T0_div_mu"],
                    gt_six["Ly"],
                    gt_six["op_x"],
                    gt_six["op_y"],
                ],
                dtype=np.float64,
            ),
            (len(mu_values), 1),
        )
        six_batch[:, 0] = np.clip(mu_values, MU_MIN, MU_MAX)
        six_batch[:, 2] = np.clip(t0_div_mu_values, T0_DIV_MU_MIN, T0_DIV_MU_MAX)
        six_batch_t = torch.tensor(six_batch, dtype=torch.float32, device=device)

        try:
            with torch.no_grad():
                audios = synth(six_batch_t, duration)
                fitness = loss_fn(target_t.expand(len(mu_values), -1), audios).detach().cpu().numpy()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
                raise OOMTrialAbort("CUDA OOM during CMA-ES mu+T0_div_mu trial") from e
            raise

        es.tell(sol.tolist(), fitness.tolist())
        eval_count += len(mu_values)
        loss_history.extend(fitness.tolist())

        current_best = es.result.fbest
        gen = eval_count // popsize

        if time.time() - gen_start > 300:
            print(f"        [TIMEOUT] Trial {trial.number} gen {gen} took >300s, stopping")
            break
        if time.time() - trial_start > 600:
            print(f"        [TRIAL TIMEOUT] Trial {trial.number} total >600s, stopping")
            break

        trial.report(current_best, step=gen)
        if trial.should_prune():
            print(f"        [Pruned] Trial {trial.number} at gen {gen} (loss={current_best:.4f})")
            raise optuna.exceptions.TrialPruned()

        if gen % 20 == 0:
            print(f"        [Trial {trial.number}] gen {gen:4d}  evals={eval_count:5d}  best={current_best:.4f}")

    best_mu = float(es.result.xbest[0])
    best_t0_div_mu = float(es.result.xbest[1])
    best_loss = float(es.result.fbest)
    print(
        f"        [Done] Trial {trial.number} best={best_loss:.4f} "
        f"mu={best_mu:.6f} T0_div_mu={best_t0_div_mu:.6f}"
    )
    return best_mu, best_t0_div_mu, best_loss, loss_history


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
    t0_div_mu_errs = []
    mu_span = SIX_BOUNDS["mu"][1] - SIX_BOUNDS["mu"][0]
    t0_span = SIX_BOUNDS["T0_div_mu"][1] - SIX_BOUNDS["T0_div_mu"][0]
    for r in results:
        mu_errs.append(abs(float(r["mu"]) - float(r["gt_mu"])) / mu_span)
        t0_div_mu_errs.append(abs(float(r["T0_div_mu"]) - float(r["gt_T0_div_mu"])) / t0_span)

    x = np.arange(2)
    means = [float(np.mean(mu_errs)), float(np.mean(t0_div_mu_errs))]
    stds = [float(np.std(mu_errs)), float(np.std(t0_div_mu_errs))]
    axes[1].bar(x, means, yerr=stds, color=["#4C78A8", "#F58518"], capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["mu", "T0_div_mu"])
    axes[1].set_ylabel("Abs. normalized error")
    axes[1].set_title("Average Parameter Error")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path / "summary_diagnostic.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(args):
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    loss_fn = select_loss_function(args.loss, sample_rate=SAMPLE_RATE, device=device)
    available = ", ".join(available_loss_names())

    print(f"CMA-ES mu+T0_div_mu (SixParamPlate) on {device.upper()}")
    print(f"  Loss: {args.loss}")
    print(f"  Available losses: {available}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Duration: {args.duration}s")

    synth = BatchedModalPlateTorch(sample_rate=SAMPLE_RATE, device=device, dtype=torch.float32, drop_sub_20hz_modes=False).to(device)

    dataset_dir = Path(args.dset_root)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    param_files = _collect_param_csvs(dataset_dir)
    if args.n_samples is not None:
        param_files = param_files[: args.n_samples]

    all_results = []
    lhs_starts = _generate_lhs_mu_t0_starts(args.n_trials, seed=args.lhs_seed)

    def stop_when_finished(study, trial, n_target):
        del trial
        finished = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]))
        if finished >= n_target:
            study.stop()

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
                gt_batch = torch.tensor([[gt_6[k] for k in SIX_KEYS]], dtype=torch.float32, device=device)
                gt_audio = synth(gt_batch, args.duration)
                gt_target = torch.tensor(target_ir, device=device).float().unsqueeze(0)
                gt_loss = float(loss_fn(gt_target, gt_audio).item())
            print(f"  GT loss: {gt_loss:.6f}")
        except Exception as e:
            print(f"  Could not calc GT: {e}")

        storage = create_robust_storage(args.storage)
        study = optuna.create_study(
            study_name=f"mu6_{npz_file.stem}",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=15, reduction_factor=3),
        )

        t0 = time.time()

        def objective(trial):
            trial_idx = trial.number % len(lhs_starts)
            mu0 = float(lhs_starts[trial_idx, 0])
            t0_div_mu0 = float(lhs_starts[trial_idx, 1])
            popsize = trial.suggest_int("popsize", args.popsize_min, args.popsize_max)
            seed = trial.suggest_int("seed", 0, 100000)
            print(
                f"    --> Trial {trial.number} | pop={popsize} seed={seed} "
                f"mu0={mu0:.6f} t0_div_mu0={t0_div_mu0:.6f}"
            )
            try:
                best_mu, best_t0_div_mu, best_loss, loss_hist = _run_cmaes_mu_t0(
                    trial,
                    target_ir,
                    args.duration,
                    synth,
                    loss_fn,
                    popsize,
                    args.budget,
                    seed,
                    mu0,
                    t0_div_mu0,
                    gt_6,
                    device,
                )
            except OOMTrialAbort:
                raise optuna.exceptions.TrialPruned("Trial aborted due to CUDA OOM")

            trial.set_user_attr("best_mu", best_mu)
            trial.set_user_attr("best_T0_div_mu", best_t0_div_mu)
            trial.set_user_attr("loss_history", loss_hist)
            return best_loss

        study.optimize(
            objective,
            n_trials=args.n_trials,
            callbacks=[lambda study, trial: stop_when_finished(study, trial, args.n_trials)],
        )

        elapsed = time.time() - t0
        best = study.best_trial
        best_mu = float(best.user_attrs["best_mu"])
        best_t0_div_mu = float(best.user_attrs["best_T0_div_mu"])
        best_loss_hist = best.user_attrs["loss_history"]
        best_loss = float(best.value)

        est_6 = dict(gt_6)
        est_6["mu"] = best_mu
        est_6["T0_div_mu"] = best_t0_div_mu

        print(f"\n  Winner: Trial {best.number}, loss={best_loss:.6f} ({elapsed:.0f}s)")
        print(f"    mu: {best_mu:.6f}")
        print(f"    T0_div_mu: {best_t0_div_mu:.6f}")
        print(f"    NMSE: {_nmse_6d(est_6, gt_6):.6f}")

        row = {
            "filename": npz_file.name,
            "target_npz": npz_file.name,
            "runtime": round(elapsed, 3),
            "best_loss": round(best_loss, 8),
            "best_popsize": int(best.params["popsize"]),
            "loss_name": args.loss,
            "gt_loss": round(gt_loss, 8) if gt_loss is not None else None,
            "nmse": round(_nmse_6d(est_6, gt_6), 8),
            "mu": round(best_mu, 8),
            "T0_div_mu": round(best_t0_div_mu, 8),
            "gt_mu": round(gt_6["mu"], 8),
            "gt_T0_div_mu": round(gt_6["T0_div_mu"], 8),
            "mu_abs_error": round(abs(best_mu - gt_6["mu"]), 8),
            "mu_rel_error": round(abs(best_mu - gt_6["mu"]) / max(abs(gt_6["mu"]), 1e-18), 8),
            "T0_div_mu_abs_error": round(abs(best_t0_div_mu - gt_6["T0_div_mu"]), 8),
            "T0_div_mu_rel_error": round(
                abs(best_t0_div_mu - gt_6["T0_div_mu"]) / max(abs(gt_6["T0_div_mu"]), 1e-18), 8
            ),
            "optuna_trial": int(best.number),
        }
        for k in SIX_KEYS:
            row[f"est_{k}"] = round(est_6[k], 8)
            row[f"gt_{k}"] = round(gt_6[k], 8)

        all_results.append(row)

        pd.DataFrame([row]).to_csv(result_path, index=False)
        _plot_per_ir(npz_file.stem, est_6, gt_6, best_loss_hist, elapsed, output_path, gt_loss=gt_loss)
        print(f"  [Saved] {npz_file.stem}")

    if all_results:
        pd.DataFrame(all_results).to_csv(output_path / "summary.csv", index=False)
        _plot_summary(all_results, output_path)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dset_root", type=str, default="data/random-IR-100-1.0s")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="results/cmaes/mu_only")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loss", type=str, default="L1_STFT")
    parser.add_argument("--storage", type=str, default="cmaes_mu6.db")
    parser.add_argument("--budget", type=int, default=25000)
    parser.add_argument("--lhs_seed", type=int, default=42)
    parser.add_argument("--popsize_min", type=int, default=30)
    parser.add_argument("--popsize_max", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
