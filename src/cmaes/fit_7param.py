"""CMA-ES + Optuna (Hyperband) for seven-parameter plate fitting.

Run as:
    python -m src.cmaes.fit_7param --loss CQT+LogDec ...
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cma
import matplotlib
import numpy as np
import optuna
import pandas as pd
import torch
from scipy.stats.qmc import LatinHypercube

from src.loss.loss_selector import available_loss_names, select_loss_function
from src.plate.SevenParamPlate import BatchedModalPlateTorch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

BOUNDS_LO_NP = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS])
BOUNDS_HI_NP = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS])
BOUNDS_RANGE_NP = BOUNDS_HI_NP - BOUNDS_LO_NP
CMA_STDS = (BOUNDS_RANGE_NP * 0.2).tolist()

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


def create_robust_storage(db_name: str = "cmaes_lhs.db"):
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


def params_tensor_to_dict(params_row: torch.Tensor) -> dict:
    return {k: float(v.item()) for k, v in zip(PARAM_KEYS, params_row)}


def generate_lhs_starts(n_trials: int, seed: int = 42) -> np.ndarray:
    sampler = LatinHypercube(d=7, seed=seed)
    unit_samples = sampler.random(n=n_trials)
    return BOUNDS_LO_NP + unit_samples * BOUNDS_RANGE_NP


def load_target_ir_from_npz(npz_path: Path, duration_s: float, expected_sr: int) -> np.ndarray:
    """Load and normalize IR target from dataset NPZ."""
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
    """Map [B,7] {E,rho,h,Ly,T0,op_x,op_y} into SevenParamPlate's [B,14]."""
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


def run_cmaes(
    trial,
    target_ir: np.ndarray,
    duration: float,
    synth: BatchedModalPlateTorch,
    loss_fn,
    popsize: int,
    budget: int,
    seed: int,
    x0: np.ndarray,
    device: str,
):
    target_t = torch.tensor(target_ir, dtype=torch.float32, device=device).unsqueeze(0)

    opts = {
        "maxfevals": budget,
        "popsize": popsize,
        "bounds": [BOUNDS_LO_NP.tolist(), BOUNDS_HI_NP.tolist()],
        "seed": seed,
        "verb_disp": 0,
        "tolfun": 5e-3,
        "tolfunhist": 5e-3,
        "CMA_stds": CMA_STDS,
    }

    es = cma.CMAEvolutionStrategy(x0.tolist(), 1.0, opts)
    eval_count = 0
    loss_history = []
    trial_start = time.time()

    while not es.stop():
        gen_start = time.time()
        solutions = es.ask()
        plate_params = physical_to_plate14_tensor(np.array(solutions), device)

        with torch.no_grad():
            audios = synth(plate_params, duration)
            fitness = loss_fn(target_t.expand(len(solutions), -1), audios).detach().cpu().numpy()

        es.tell(solutions, fitness.tolist())
        eval_count += len(solutions)
        loss_history.extend(fitness.tolist())

        current_best = es.result.fbest
        gen = eval_count // popsize

        if time.time() - gen_start > 300:
            print(f"        [TIMEOUT] Trial {trial.number} gen {gen} took >300s, stopping")
            break

        if time.time() - trial_start > 600:
            print(f"        [TRIAL TIMEOUT] Trial {trial.number} total >600s, stopping")
            break

        try:
            trial.report(current_best, step=gen)
            if trial.should_prune():
                print(f"        [Pruned] Trial {trial.number} at gen {gen} (loss={current_best:.4f})")
                raise optuna.exceptions.TrialPruned()
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"        [DB retry] {e}")
            time.sleep(1)

        if gen % 20 == 0:
            print(f"        [Trial {trial.number}] gen {gen:4d}  evals={eval_count:5d}  best={current_best:.4f}")

    best_p = params_tensor_to_dict(torch.tensor(es.result.xbest, dtype=torch.float32, device=device))
    print(f"        [Done] Trial {trial.number} best={es.result.fbest:.4f}")
    return best_p, es.result.fbest, loss_history


def plot_per_ir(stem, est_7, gt_7, loss_hist, best_loss, elapsed, out_path, gt_loss=None):
    del best_loss
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
    ax.set_title("Optimization Convergence")
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
    available = ", ".join(available_loss_names())

    print(f"CMA-ES with LHS + Physical Space on {device.upper()}")
    print(f"  Loss: {args.loss}")
    print(f"  Available losses: {available}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Duration: {args.duration}s")
    print("  Search: physical space with CMA_stds per-dim scaling")

    synth = BatchedModalPlateTorch(device=device).to(device)

    target_path = Path(args.dset_root)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(target_path.glob("*.npz"))[: args.n_samples]
    all_results = []

    lhs_starts = generate_lhs_starts(args.n_trials, seed=args.lhs_seed)

    def stop_when_finished(study, trial, n_target):
        del trial
        finished = len(
            study.get_trials(states=[optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED])
        )
        if finished >= n_target:
            study.stop()

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

        storage = create_robust_storage(args.storage)
        study = optuna.create_study(
            study_name=f"lhs_{npz_file.stem}",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=15, reduction_factor=3),
        )

        t0_time = time.time()

        def objective(trial):
            trial_idx = trial.number % len(lhs_starts)
            x0 = lhs_starts[trial_idx]
            popsize = trial.suggest_int("popsize", args.popsize_min, args.popsize_max)
            seed = trial.suggest_int("seed", 0, 100000)
            print(f"    --> Trial {trial.number} | pop={popsize} seed={seed}")
            bp, bl, lh = run_cmaes(
                trial,
                target_ir,
                args.duration,
                synth,
                loss_fn,
                popsize,
                args.budget,
                seed,
                x0,
                device,
            )
            trial.set_user_attr("best_params", bp)
            trial.set_user_attr("loss_history", lh)
            return bl

        study.optimize(
            objective,
            n_trials=args.n_trials,
            callbacks=[lambda study, trial: stop_when_finished(study, trial, args.n_trials)],
        )

        elapsed = time.time() - t0_time
        best = study.best_trial
        best_params = best.user_attrs["best_params"]
        best_loss_hist = best.user_attrs["loss_history"]
        best_loss = best.value

        comp = to_composite(best_params)
        print(f"\n  Winner: Trial {best.number}, loss={best_loss:.6f} ({elapsed:.0f}s)")
        for k in COMPOSITE_KEYS:
            print(f"    {k}: {comp[k]:.6f}")
        if gt_7:
            print(f"    NMSE: {compute_nmse_6d(best_params, gt_7):.6f}")

        final_6 = to_composite(best_params)
        row = {
            "filename": npz_file.name,
            "target_npz": npz_file.name,
            "runtime": round(elapsed, 3),
            "best_loss": round(best_loss, 8),
            "best_popsize": best.params["popsize"],
            "loss_name": args.loss,
            "gt_loss": round(gt_loss, 8) if gt_loss is not None else None,
            "nmse": round(compute_nmse_6d(best_params, gt_7), 8) if gt_7 else None,
            "gt_available": gt_7 is not None,
            "mu": round(final_6["mu"], 8),
            "D_mu": round(final_6["D_mu"], 8),
            "T0_mu": round(final_6["T0_mu"], 8),
            **{k: round(float(v), 8) for k, v in best_params.items()},
        }
        if gt_7:
            for k, v in gt_7.items():
                row[f"gt_{k}"] = round(float(v), 8)

        all_results.append(row)

        pd.DataFrame([row]).to_csv(result_path, index=False)
        plot_per_ir(npz_file.stem, best_params, gt_7, best_loss_hist, best_loss, elapsed, output_path, gt_loss)
        print(f"  [Saved] {npz_file.stem}")

    summary_path = output_path / "summary.csv"
    if all_results:
        pd.DataFrame(all_results).to_csv(summary_path, index=False)
        plot_summary(all_results, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dset_root", type=str, default="random-IR-10-1.0s")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="results_lhs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loss", type=str, default="CQT+LogDec")
    parser.add_argument("--storage", type=str, default="cmaes_lhs.db")
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
