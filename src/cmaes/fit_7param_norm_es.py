"""CMA-ES + Optuna (Hyperband) for seven-parameter plate fitting.

Run as:
    python -m src.cmaes.fit_7param_norm_es --loss CQT+LogDec ...
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

BOUNDS_LO_NP = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS], dtype=np.float64)
BOUNDS_HI_NP = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS], dtype=np.float64)
BOUNDS_RANGE_NP = BOUNDS_HI_NP - BOUNDS_LO_NP

NORM_LO_NP = -np.ones(len(PARAM_KEYS), dtype=np.float64)
NORM_HI_NP = np.ones(len(PARAM_KEYS), dtype=np.float64)
NORM_RANGE_NP = NORM_HI_NP - NORM_LO_NP

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


def params_np_to_dict(params_row: np.ndarray) -> dict:
    return {k: float(v) for k, v in zip(PARAM_KEYS, params_row)}


def norm_to_physical(norm_batch_np: np.ndarray) -> np.ndarray:
    """Map normalized parameters in [-1, 1] to physical PARAM_BOUNDS."""
    return BOUNDS_LO_NP + ((norm_batch_np - NORM_LO_NP) / NORM_RANGE_NP) * BOUNDS_RANGE_NP


def generate_lhs_starts_norm(n_trials: int, seed: int = 42) -> np.ndarray:
    sampler = LatinHypercube(d=7, seed=seed)
    unit_samples = sampler.random(n=n_trials)
    return NORM_LO_NP + unit_samples * NORM_RANGE_NP


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
    sigma0: float,
    tolfun: float,
    tolfunhist: float,
    dtype: torch.dtype,
    device: str,
):
    if np.any(x0 < NORM_LO_NP) or np.any(x0 > NORM_HI_NP):
        raise ValueError(f"x0 is out of normalized bounds [-1, 1]. Got x0={x0}")

    target_t = torch.tensor(target_ir, dtype=dtype, device=device).unsqueeze(0)

    opts = {
        "maxfevals": budget,
        "popsize": popsize,
        "bounds": [NORM_LO_NP.tolist(), NORM_HI_NP.tolist()],
        "seed": seed,
        "verb_disp": 0,
        "tolfun": float(tolfun),
        "tolfunhist": float(tolfunhist),
    }

    es = cma.CMAEvolutionStrategy(x0.tolist(), float(sigma0), opts)
    eval_count = 0
    cmaes_gen_count = 0
    loss_history = []
    trial_start = time.time()

    while not es.stop():
        gen_start = time.time()
        solutions = es.ask()
        norm_solutions = np.array(solutions, dtype=np.float64)
        phys_solutions = norm_to_physical(norm_solutions)
        plate_params = physical_to_plate14_tensor(phys_solutions, device).to(dtype=dtype)

        with torch.no_grad():
            audios = synth(plate_params, duration)
            cmaes_gen_count += 1
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

    best_norm = np.asarray(es.result.xbest, dtype=np.float64).reshape(1, -1)
    best_phys = norm_to_physical(best_norm)[0]
    best_p = params_np_to_dict(best_phys)
    print(f"        [Done] Trial {trial.number} best={es.result.fbest:.4f}")
    return best_p, es.result.fbest, loss_history, eval_count, cmaes_gen_count


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

    print(f"CMA-ES with LHS + Normalized Space on {device.upper()}")
    print(f"  Loss: {args.loss}")
    print(f"  Available losses: {available}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Duration: {args.duration}s")
    print("  Search: normalized [-1, 1] space; expanded to physical bounds for synthesis")
    print(f"  sigma0: {args.sigma0}")
    print(f"  tolfun: {args.tolfun}")
    print(f"  tolfunhist: {args.tolfunhist}")
    print(f"  pruner_min_resource: {args.pruner_min_resource}")
    print(f"  pruner_reduction_factor: {args.pruner_reduction_factor}")

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = torch.float32
    synth = BatchedModalPlateTorch(device=device, dtype=dtype).to(device)

    target_path = Path(args.dset_root)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(target_path.glob("*.npz"))[: args.n_samples]
    all_results = []

    lhs_starts = generate_lhs_starts_norm(args.n_trials, seed=args.lhs_seed)

    def stop_when_finished(study, trial, n_target):
        del trial
        finished = len(
            study.get_trials(states=[optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED])
        )
        if finished >= n_target:
            study.stop()

    def stop_on_success(study, trial, threshold: float):
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            if float(trial.value) < float(threshold):
                print(
                    f"    --> Early stop triggered: trial {trial.number} "
                    f"reached loss={float(trial.value):.6f} < {threshold:.6f}"
                )
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
                    gt_plate = physical_to_plate14_tensor(gt_tensor7, device).to(dtype=dtype)
                    gt_audio = synth(gt_plate, args.duration)
                    gt_target = torch.tensor(target_ir, dtype=dtype, device=device).unsqueeze(0)
                    gt_loss = float(loss_fn(gt_target, gt_audio).item())
                print(f"  GT loss: {gt_loss:.6f}")
            except Exception as e:
                print(f"  Could not calc GT: {e}")

        # --- adaptive early-stop threshold: gt + alpha*(terrain_mean - gt) ---
        # Places the stop threshold at the same fractional position (alpha) along
        # each loss's [gt_loss .. random-terrain] range, so the stopping criterion
        # is consistent across losses of different scale/floor. Computed live per IR
        # (no hard-coded table), so any new loss self-calibrates.
        early_threshold = args.early_stop_loss  # fallback if GT unavailable
        terrain_mean = None
        if gt_loss is not None:
            try:
                rng_t = np.random.default_rng(args.terrain_seed + idx)
                pts = rng_t.uniform(
                    NORM_LO_NP, NORM_HI_NP, size=(args.terrain_k, len(PARAM_KEYS))
                )
                tgt_t = torch.tensor(target_ir, dtype=dtype, device=device).unsqueeze(0)
                tv = []
                with torch.no_grad():
                    for s in range(0, args.terrain_k, args.terrain_batch):
                        phys = norm_to_physical(pts[s:s + args.terrain_batch])
                        plate = physical_to_plate14_tensor(phys, device).to(dtype=dtype)
                        aud = synth(plate, args.duration)
                        tv.append(
                            loss_fn(tgt_t.expand(aud.shape[0], -1), aud)
                            .detach().cpu().numpy()
                        )
                terrain_mean = float(np.concatenate(tv).mean())
                early_threshold = gt_loss + args.early_stop_alpha * (terrain_mean - gt_loss)
                print(
                    f"  terrain_mean={terrain_mean:.6f}  "
                    f"early_threshold={early_threshold:.6f}  (alpha={args.early_stop_alpha})"
                )
            except Exception as e:
                print(f"  Could not calc terrain (using --early_stop_loss={early_threshold}): {e}")

        # --- scale-relative CMA-ES tolfun -------------------------------------
        # CMA-ES tolfun/tolfunhist are ABSOLUTE thresholds on the objective range
        # within a population, so a fixed value is NOT comparable across losses
        # whose numeric ranges differ by orders of magnitude.  For L1_STFT
        # (terrain-gt ~ 1.0) tolfun=5e-3 is a tight 0.5% criterion; for a loss
        # like SOT (terrain-gt ~ 1e-4) it EXCEEDS the entire usable range, so
        # CMA-ES self-terminates before it can descend at all.
        #
        # We therefore set tolfun to the same FRACTION of each loss own
        # [gt_loss .. terrain] range, mirroring how early_stop_alpha already
        # normalises the early-stop threshold.  --tolfun_frac defaults to the
        # value reproducing the historical L1_STFT behaviour (5e-3 / ~0.96).
        # Falls back to the absolute --tolfun when GT/terrain is unavailable.
        tolfun_eff = float(args.tolfun)
        tolfunhist_eff = float(args.tolfunhist)
        if terrain_mean is not None and gt_loss is not None and args.tolfun_frac > 0:
            loss_range = float(terrain_mean) - float(gt_loss)
            if loss_range > 0:
                tolfun_eff = args.tolfun_frac * loss_range
                tolfunhist_eff = args.tolfun_frac * loss_range
                print(
                    f"  tolfun={tolfun_eff:.3g} tolfunhist={tolfunhist_eff:.3g}  "
                    f"(frac={args.tolfun_frac} of range={loss_range:.3g})"
                )

        storage = create_robust_storage(args.storage)
        study = optuna.create_study(
            study_name=f"norm_lhs_{npz_file.stem}",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=args.lhs_seed),
            pruner=optuna.pruners.SuccessiveHalvingPruner(
                min_resource=args.pruner_min_resource,
                reduction_factor=args.pruner_reduction_factor,
            ),
        )

        t0_time = time.time()

        def objective(trial):
            trial_idx = trial.number % len(lhs_starts)
            x0 = lhs_starts[trial_idx]
            popsize = trial.suggest_int("popsize", args.popsize_min, args.popsize_max)
            seed = trial.suggest_int("seed", 0, 100000)
            print(
                f"    --> Trial {trial.number} | pop={popsize} seed={seed} "
                f"x0_norm_min={float(np.min(x0)):.3f} x0_norm_max={float(np.max(x0)):.3f}"
            )
            bp, bl, lh, eval_count, cmaes_gen_count = run_cmaes(
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
                tolfun_eff,
                tolfunhist_eff,
                dtype,
                device,
            )
            trial.set_user_attr("best_params", bp)
            trial.set_user_attr("loss_history", lh)
            trial.set_user_attr("eval_count", int(eval_count))
            trial.set_user_attr("cmaes_gen_count", int(cmaes_gen_count))
            return bl

        study.optimize(
            objective,
            n_trials=args.n_trials,
            callbacks=[
                lambda study, trial: stop_when_finished(study, trial, args.n_trials),
                lambda study, trial: stop_on_success(study, trial, early_threshold),
            ],
        )

        elapsed = time.time() - t0_time
        trials = study.get_trials(deepcopy=False)
        terminal_trials = [
            t
            for t in trials
            if t.state
            in (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL,
            )
        ]
        n_restarts_run = len(
            terminal_trials
        )
        total_evaluations = int(
            sum(int(t.user_attrs.get("eval_count", 0)) for t in terminal_trials)
        )
        cmaes_gen_count = int(
            sum(int(t.user_attrs.get("cmaes_gen_count", 0)) for t in terminal_trials)
        )
        best = study.best_trial
        best_params = best.user_attrs["best_params"]
        best_loss_hist = best.user_attrs["loss_history"]
        best_loss = best.value

        comp = to_composite(best_params)
        print(f"\n  Winner: Trial {best.number}, loss={best_loss:.6f} ({elapsed:.0f}s)")
        print(f"    Restarts run: {n_restarts_run}")
        for k in COMPOSITE_KEYS:
            print(f"    {k}: {comp[k]:.6f}")
        if gt_7:
            print(f"    NMSE: {compute_nmse_6d(best_params, gt_7):.6f}")

        final_6 = to_composite(best_params)
        row = {
            "filename": npz_file.name,
            "target_npz": npz_file.name,
            "runtime": float(elapsed),
            "best_loss": float(best_loss),
            "n_restarts_run": int(n_restarts_run),
            "total_evaluations": total_evaluations,
            "cmaes_gen_count": cmaes_gen_count,
            "best_popsize": best.params["popsize"],
            "loss_name": args.loss,
            "gt_loss": float(gt_loss) if gt_loss is not None else None,
            "terrain_mean": float(terrain_mean) if terrain_mean is not None else None,
            "early_threshold": float(early_threshold),
            "tolfun_eff": float(tolfun_eff),
            "nmse": float(compute_nmse_6d(best_params, gt_7)) if gt_7 else None,
            "gt_available": gt_7 is not None,
            "mu": float(final_6["mu"]),
            "D_mu": float(final_6["D_mu"]),
            "T0_mu": float(final_6["T0_mu"]),
            **{k: float(v) for k, v in best_params.items()},
        }
        if gt_7:
            for k, v in gt_7.items():
                row[f"gt_{k}"] = float(v)

        all_results.append(row)

        pd.DataFrame([row]).to_csv(result_path, index=False, float_format="%.16g")
        plot_per_ir(npz_file.stem, best_params, gt_7, best_loss_hist, best_loss, elapsed, output_path, gt_loss)
        print(f"  [Saved] {npz_file.stem}")

    summary_path = output_path / "summary.csv"
    if all_results:
        pd.DataFrame(all_results).to_csv(summary_path, index=False, float_format="%.16g")
        plot_summary(all_results, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dset_root", type=str, default="random-IR-10-1.0s")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=400)
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="results_lhs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])
    parser.add_argument("--loss", type=str, default="CQT+LogDec")
    parser.add_argument("--storage", type=str, default="cmaes_lhs_norm.db")
    parser.add_argument("--budget", type=int, default=25000)
    parser.add_argument("--sigma0", type=float, default=0.6, help="CMA-ES initial step-size in normalized space")
    parser.add_argument("--tolfun_frac", type=float, default=0.0052,
                        help="CMA-ES tolfun as a FRACTION of each loss own "
                             "(terrain_mean - gt_loss) range, making the termination "
                             "criterion scale-invariant across losses. Default 0.0052 "
                             "reproduces the historical L1_STFT calibration (5e-3/~0.96). "
                             "Set 0 to use the absolute --tolfun instead.")
    parser.add_argument("--tolfun", type=float, default=5e-3, help="CMA-ES tolfun absolute fallback")
    parser.add_argument("--tolfunhist", type=float, default=5e-3, help="CMA-ES tolfunhist termination threshold")
    parser.add_argument(
        "--pruner_min_resource",
        type=int,
        default=15,
        help="Optuna SuccessiveHalving min_resource",
    )
    parser.add_argument(
        "--pruner_reduction_factor",
        type=int,
        default=3,
        help="Optuna SuccessiveHalving reduction_factor",
    )
    parser.add_argument(
        "--early_stop_loss",
        type=float,
        default=0.01,
        help="Fallback absolute early-stop loss, used only if GT (and thus the adaptive threshold) is unavailable",
    )
    parser.add_argument(
        "--early_stop_alpha",
        type=float,
        default=0.0095,
        help="Adaptive early-stop: stop when loss < gt_loss + alpha*(terrain_mean - gt_loss). "
             "alpha=0.0095 reproduces the old 0.01 threshold for L1_STFT.",
    )
    parser.add_argument("--terrain_k", type=int, default=200,
                        help="random points sampled per IR to estimate terrain_mean")
    parser.add_argument("--terrain_batch", type=int, default=64,
                        help="synth batch size for terrain sampling")
    parser.add_argument("--terrain_seed", type=int, default=0,
                        help="base seed for terrain sampling (seed = terrain_seed + IR index)")
    parser.add_argument("--lhs_seed", type=int, default=42)
    parser.add_argument("--popsize_min", type=int, default=30)
    parser.add_argument("--popsize_max", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
