"""CMA-ES + Optuna in normalized space for six-parameter plate fitting.

Stage-2 incremental ablation:
- optimize in normalized 6D space [-1, 1]
- synthesize with src.plate.SixParamPlate
- LHS restarts are sampled in 7D raw space, then converted to 6D starts

Run as:
    python -m src.cmaes.fit_6param_norm --loss L1_STFT ...
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
from src.plate.SixParamPlate import BatchedModalPlateTorch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)

SAMPLE_RATE = 44100
NU = 0.25

RAW7_KEYS = ["E", "rho", "h", "Ly", "T0", "op_x", "op_y"]
RAW7_BOUNDS = {
    "E": (6.7e10, 2.2e11),
    "rho": (2430.0, 21230.0),
    "h": (0.001, 0.005),
    "Ly": (1.1, 4.0),
    "T0": (0.01, 1000.0),
    "op_x": (0.51, 1.0),
    "op_y": (0.51, 1.0),
}

RAW7_LO = np.array([RAW7_BOUNDS[k][0] for k in RAW7_KEYS], dtype=np.float64)
RAW7_HI = np.array([RAW7_BOUNDS[k][1] for k in RAW7_KEYS], dtype=np.float64)
RAW7_RANGE = RAW7_HI - RAW7_LO

_D_SCALE = 12.0 * (1.0 - NU**2)
D_MIN = RAW7_BOUNDS["E"][0] * (RAW7_BOUNDS["h"][0] ** 3) / _D_SCALE
D_MAX = RAW7_BOUNDS["E"][1] * (RAW7_BOUNDS["h"][1] ** 3) / _D_SCALE
MU_MIN = RAW7_BOUNDS["rho"][0] * RAW7_BOUNDS["h"][0]
MU_MAX = RAW7_BOUNDS["rho"][1] * RAW7_BOUNDS["h"][1]

SIX_KEYS = ["mu", "D_div_mu", "T0_div_mu", "Ly", "op_x", "op_y"]
SIX_BOUNDS = {
    "mu": (MU_MIN, MU_MAX),
    "D_div_mu": (D_MIN / MU_MAX, D_MAX / MU_MIN),
    "T0_div_mu": (RAW7_BOUNDS["T0"][0] / MU_MAX, RAW7_BOUNDS["T0"][1] / MU_MIN),
    "Ly": RAW7_BOUNDS["Ly"],
    "op_x": RAW7_BOUNDS["op_x"],
    "op_y": RAW7_BOUNDS["op_y"],
}

SIX_LO = np.array([SIX_BOUNDS[k][0] for k in SIX_KEYS], dtype=np.float64)
SIX_HI = np.array([SIX_BOUNDS[k][1] for k in SIX_KEYS], dtype=np.float64)
SIX_RANGE = SIX_HI - SIX_LO

NORM6_LO = -np.ones(len(SIX_KEYS), dtype=np.float64)
NORM6_HI = np.ones(len(SIX_KEYS), dtype=np.float64)
NORM6_RANGE = NORM6_HI - NORM6_LO


def create_robust_storage(db_name: str = "cmaes_lhs_6_norm.db"):
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


def to_composite_6(p6: dict) -> dict:
    return {k: float(p6[k]) for k in SIX_KEYS}


def compute_nmse_6d(est_6: dict, gt_6: dict) -> float:
    errs = []
    for key in SIX_KEYS:
        lo, hi = SIX_BOUNDS[key]
        e = (est_6[key] - lo) / (hi - lo)
        g = (gt_6[key] - lo) / (hi - lo)
        errs.append((e - g) ** 2)
    return float(np.mean(errs))


def norm6_to_physical6(norm6_batch_np: np.ndarray) -> np.ndarray:
    return SIX_LO + ((norm6_batch_np - NORM6_LO) / NORM6_RANGE) * SIX_RANGE


def physical6_to_norm6(phys6_batch_np: np.ndarray) -> np.ndarray:
    return NORM6_LO + ((phys6_batch_np - SIX_LO) / SIX_RANGE) * NORM6_RANGE


def raw7_phys_to_six_phys(raw7_batch_np: np.ndarray) -> np.ndarray:
    E = raw7_batch_np[:, 0]
    rho = raw7_batch_np[:, 1]
    h = raw7_batch_np[:, 2]
    Ly = raw7_batch_np[:, 3]
    T0 = raw7_batch_np[:, 4]
    op_x = raw7_batch_np[:, 5]
    op_y = raw7_batch_np[:, 6]

    mu = rho * h
    D = (E * (h**3)) / (12.0 * (1.0 - NU**2))
    out = np.stack([mu, D / mu, T0 / mu, Ly, op_x, op_y], axis=1)
    return out


def generate_lhs_starts_norm6_from_raw7(n_trials: int, seed: int = 42) -> np.ndarray:
    sampler = LatinHypercube(d=len(RAW7_KEYS), seed=seed)
    unit_samples = sampler.random(n=n_trials)
    raw7_phys = RAW7_LO + unit_samples * RAW7_RANGE
    six_phys = raw7_phys_to_six_phys(raw7_phys)
    six_norm = physical6_to_norm6(six_phys)
    return np.clip(six_norm, -1.0, 1.0)


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


def full7_row_to_six_dict(row7: dict) -> dict:
    E = float(row7["E"])
    rho = float(row7["rho"])
    h = float(row7["h"])
    Ly = float(row7["Ly"])
    T0 = float(row7["T0"])
    op_x = float(row7["op_x"])
    op_y = float(row7["op_y"])
    mu = rho * h
    D = (E * (h**3)) / (12.0 * (1.0 - NU**2))
    return {
        "mu": mu,
        "D_div_mu": D / mu,
        "T0_div_mu": T0 / mu,
        "Ly": Ly,
        "op_x": op_x,
        "op_y": op_y,
    }


def run_cmaes(
    trial,
    target_ir: np.ndarray,
    duration: float,
    synth: BatchedModalPlateTorch,
    loss_fn,
    popsize: int,
    budget: int,
    seed: int,
    x0_norm6: np.ndarray,
    sigma0: float,
    device: str,
):
    if np.any(x0_norm6 < NORM6_LO) or np.any(x0_norm6 > NORM6_HI):
        raise ValueError(f"x0 is out of normalized bounds [-1, 1]. Got x0={x0_norm6}")

    target_t = torch.tensor(target_ir, dtype=torch.float32, device=device).unsqueeze(0)

    opts = {
        "maxfevals": budget,
        "popsize": popsize,
        "bounds": [NORM6_LO.tolist(), NORM6_HI.tolist()],
        "seed": seed,
        "verb_disp": 0,
        "tolfun": 5e-3,
        "tolfunhist": 5e-3,
    }

    es = cma.CMAEvolutionStrategy(x0_norm6.tolist(), float(sigma0), opts)
    eval_count = 0
    loss_history = []
    trial_start = time.time()

    while not es.stop():
        gen_start = time.time()
        norm_solutions = np.array(es.ask(), dtype=np.float64)
        phys_solutions = norm6_to_physical6(norm_solutions)
        plate_params = torch.tensor(phys_solutions, dtype=torch.float32, device=device)

        with torch.no_grad():
            audios = synth(plate_params, duration)
            fitness = loss_fn(target_t.expand(len(norm_solutions), -1), audios).detach().cpu().numpy()

        es.tell(norm_solutions.tolist(), fitness.tolist())
        eval_count += len(norm_solutions)
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

    best_norm = np.asarray(es.result.xbest, dtype=np.float64).reshape(1, -1)
    best_phys = norm6_to_physical6(best_norm)[0]
    best_p = {k: float(v) for k, v in zip(SIX_KEYS, best_phys)}
    print(f"        [Done] Trial {trial.number} best={es.result.fbest:.4f}")
    return best_p, es.result.fbest, loss_history


def plot_per_ir(stem, est_6, gt_6, loss_hist, best_loss, elapsed, out_path, gt_loss=None):
    del best_loss

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
    ax.bar(x - width / 2, e_norm, width, label="Est", color="steelblue")
    if gt_6:
        g_norm = [(gt_6[k] - SIX_BOUNDS[k][0]) / (SIX_BOUNDS[k][1] - SIX_BOUNDS[k][0]) for k in SIX_KEYS]
        ax.bar(x + width / 2, g_norm, width, label="GT", color="coral")
        ax.set_title(f"6 Params (NMSE={compute_nmse_6d(est_6, gt_6):.4f})")
    else:
        ax.set_title("6 Params")
    ax.set_xticks(x)
    ax.set_xticklabels(SIX_KEYS, rotation=45)
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
        errs = {k: [] for k in SIX_KEYS}
        for r in results:
            est_6 = {k: r[k] for k in SIX_KEYS}
            gt_6 = {k: r[f"gt_{k}"] for k in SIX_KEYS}
            for k in SIX_KEYS:
                errs[k].append(abs(est_6[k] - gt_6[k]) / (SIX_BOUNDS[k][1] - SIX_BOUNDS[k][0]))

        x = np.arange(len(SIX_KEYS))
        axes[1].bar(
            x,
            [np.mean(errs[k]) for k in SIX_KEYS],
            yerr=[np.std(errs[k]) for k in SIX_KEYS],
            color="coral",
            capsize=4,
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(SIX_KEYS, rotation=45)
        axes[1].set_ylabel("Absolute Norm Error")
        axes[1].set_title("Average Error")

    plt.tight_layout()
    plt.savefig(out_path / "summary_diagnostic.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(args):
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    loss_fn = select_loss_function(args.loss, sample_rate=SAMPLE_RATE, device=device)
    available = ", ".join(available_loss_names())

    print(f"CMA-ES with LHS(raw7)->norm6 starts + SixParamPlate on {device.upper()}")
    print(f"  Loss: {args.loss}")
    print(f"  Available losses: {available}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Duration: {args.duration}s")
    print("  Search: normalized 6D [-1, 1] space; expanded to six-parameter physical bounds")
    print("  Restarts: LHS in raw7, converted to six-parameter starts")
    print(f"  sigma0: {args.sigma0}")

    synth = BatchedModalPlateTorch(device=device).to(device)

    target_path = Path(args.dset_root)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(target_path.glob("*.npz"))[: args.n_samples]
    all_results = []

    lhs_starts_norm6 = generate_lhs_starts_norm6_from_raw7(args.n_trials, seed=args.lhs_seed)

    def stop_when_finished(study, trial, n_target):
        del trial
        finished = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]))
        if finished >= n_target:
            study.stop()

    for idx, npz_file in enumerate(npz_files):
        result_path = output_path / f"result_{npz_file.stem}.csv"
        if result_path.exists() and not args.overwrite:
            continue

        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(npz_files)}] {npz_file.name}")
        target_ir = load_target_ir_from_npz(npz_file, args.duration, SAMPLE_RATE)

        gt_loss, gt_6 = None, None
        gt_csv = target_path / npz_file.name.replace("random_IR_", "random_IR_params_").replace(".npz", ".csv")
        if gt_csv.exists():
            try:
                gt_row = pd.read_csv(gt_csv).iloc[0].to_dict()
                gt_6 = full7_row_to_six_dict(gt_row)
                with torch.no_grad():
                    gt_tensor6 = np.array([[gt_6[k] for k in SIX_KEYS]], dtype=np.float64)
                    gt_plate = torch.tensor(gt_tensor6, dtype=torch.float32, device=device)
                    gt_audio = synth(gt_plate, args.duration)
                    gt_target = torch.tensor(target_ir, device=device).float().unsqueeze(0)
                    gt_loss = float(loss_fn(gt_target, gt_audio).item())
                print(f"  GT loss: {gt_loss:.6f}")
            except Exception as e:
                print(f"  Could not calc GT: {e}")

        storage = create_robust_storage(args.storage)
        study = optuna.create_study(
            study_name=f"norm6_lhs_{npz_file.stem}",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=15, reduction_factor=3),
        )

        t0_time = time.time()

        def objective(trial):
            trial_idx = trial.number % len(lhs_starts_norm6)
            x0 = lhs_starts_norm6[trial_idx]
            popsize = trial.suggest_int("popsize", args.popsize_min, args.popsize_max)
            seed = trial.suggest_int("seed", 0, 100000)
            print(
                f"    --> Trial {trial.number} | pop={popsize} seed={seed} "
                f"x0_norm_min={float(np.min(x0)):.3f} x0_norm_max={float(np.max(x0)):.3f}"
            )
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
                args.sigma0,
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

        print(f"\n  Winner: Trial {best.number}, loss={best_loss:.6f} ({elapsed:.0f}s)")
        if gt_6:
            print(f"    NMSE: {compute_nmse_6d(best_params, gt_6):.6f}")

        row = {
            "filename": npz_file.name,
            "target_npz": npz_file.name,
            "runtime": round(elapsed, 3),
            "best_loss": round(best_loss, 8),
            "best_popsize": best.params["popsize"],
            "loss_name": args.loss,
            "gt_loss": round(gt_loss, 8) if gt_loss is not None else None,
            "nmse": round(compute_nmse_6d(best_params, gt_6), 8) if gt_6 else None,
            "gt_available": gt_6 is not None,
            **{k: round(float(best_params[k]), 8) for k in SIX_KEYS},
        }
        if gt_6:
            for k, v in gt_6.items():
                row[f"gt_{k}"] = round(float(v), 8)

        all_results.append(row)

        pd.DataFrame([row]).to_csv(result_path, index=False)
        plot_per_ir(npz_file.stem, best_params, gt_6, best_loss_hist, best_loss, elapsed, output_path, gt_loss)
        print(f"  [Saved] {npz_file.stem}")

    if all_results:
        pd.DataFrame(all_results).to_csv(output_path / "summary.csv", index=False)
        plot_summary(all_results, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dset_root", type=str, default="data/random-IR-100-1.0s")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="results/cmaes/incremental_ablation/2_6_norm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loss", type=str, default="L1_STFT")
    parser.add_argument("--storage", type=str, default="cmaes_lhs_6_norm.db")
    parser.add_argument("--budget", type=int, default=25000)
    parser.add_argument("--sigma0", type=float, default=0.6, help="CMA-ES initial step-size in normalized 6D space")
    parser.add_argument("--lhs_seed", type=int, default=42)
    parser.add_argument("--popsize_min", type=int, default=30)
    parser.add_argument("--popsize_max", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
