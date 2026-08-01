"""PSO + Optuna baseline with study-level early-stop callback.

This is a fork of `src/baseline/baseline_plus.py` with two additions:

  1. `--early_stop_loss` CLI flag (default 0.01). Mirrors the equivalent flag
     in `fit_7param_norm_es.py`.
  2. A `stop_on_success` Optuna callback that ends the study once any COMPLETE
     trial's value drops below the threshold.

Also adds:

  - `n_restarts_run`, `total_evals_used` to the summary row, mirroring the
     CMA-ES side, so analysis CSVs are structurally symmetric.
  - A per-IR `trials_<stem>.csv` with one row per Optuna trial
     (trial_number, final_loss, evals_used, particles, w, c1, c2, seed)
     so "evals to threshold" analytics don't require cracking the Optuna DB.

Run as:
    python -m experiment_pruning_es.src.baseline_plus_es \\
        --dset_root data/random-IR-100-1.0s \\
        --n_samples 100 \\
        --budget 5000 \\
        --n_trials 40 \\
        --early_stop_loss 0.01 \\
        --loss L1_STFT \\
        --output_dir results/experiment_pruning_es/pso_es_5k

Imports the PSO inner loop and shared plate/loss helpers from the existing
`src/baseline/baseline_torch.py` (i.e. `core`).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import optuna
import pandas as pd
import torch

from src.baseline import baseline_torch as core

optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_robust_storage(db_name: str = "baseline_plus_es.db"):
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


def _iters_from_budget(budget: int, num_particles: int) -> tuple[int, int]:
    if num_particles <= 0:
        raise ValueError("num_particles must be > 0")
    if budget < num_particles:
        raise ValueError(f"budget ({budget}) must be >= num_particles ({num_particles})")
    max_iter = (budget // num_particles) - 1
    realized = num_particles * (max_iter + 1)
    return max_iter, realized


def run(args):
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    loss_fn = core.select_loss_function(args.loss, sample_rate=core.SAMPLE_RATE, device=device)

    print("=" * 70)
    print(f"PSO + Optuna + study-level Early Stop  ({args.loss}) on {device.upper()}")
    print("=" * 70)
    print(f"  Optuna trials per IR: {args.n_trials}")
    print(f"  PSO particle range: [{args.num_particles_min}, {args.num_particles_max}]")
    print(f"  Budget target (evals/trial): {args.budget}")
    print(f"  Duration: {args.duration}s")
    print(f"  Pruner: SHP(3, 2) registered (PSO has no `trial.report` so it stays inactive)")
    print(f"  Study early-stop: loss < {args.early_stop_loss}")
    print()

    synth = core.BatchedModalPlateTorch(device=device).to(device)

    target_path = Path(args.dset_root)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(target_path.glob("*.npz"))[: args.n_samples]
    all_results = []

    def stop_when_finished(study, trial, n_target):
        del trial
        finished = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]))
        if finished >= n_target:
            study.stop()

    def stop_on_success(study, trial, threshold: float):
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            if float(trial.value) < float(threshold):
                print(
                    f"    --> Early stop: trial {trial.number} reached "
                    f"loss={float(trial.value):.6f} < {threshold:.6f}"
                )
                study.stop()

    for idx, npz_file in enumerate(npz_files):
        result_path = output_path / f"result_{npz_file.stem}.csv"
        if result_path.exists() and not args.overwrite:
            continue

        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(npz_files)}] {npz_file.name}")
        target_ir = core.load_target_ir_from_npz(npz_file, args.duration, core.SAMPLE_RATE)

        gt_loss, gt_7 = None, None
        gt_csv = target_path / npz_file.name.replace("random_IR_", "random_IR_params_").replace(".npz", ".csv")
        if gt_csv.exists():
            try:
                gt_7 = {k: float(pd.read_csv(gt_csv).iloc[0][k]) for k in core.PARAM_KEYS}
                with torch.no_grad():
                    gt_tensor7 = torch.tensor([[gt_7[k] for k in core.PARAM_KEYS]], dtype=torch.float32, device=device)
                    gt_plate = core.physical7_to_plate14_tensor(gt_tensor7, device)
                    gt_audio = synth(gt_plate, args.duration)
                    gt_target = torch.tensor(target_ir, device=device).float().unsqueeze(0)
                    gt_loss = float(loss_fn(gt_target, gt_audio).item())
                print(f"  GT loss: {gt_loss:.6f}")
            except Exception as e:
                print(f"  Could not calc GT: {e}")

        storage = create_robust_storage(args.storage)
        study = optuna.create_study(
            study_name=f"baseline_plus_es_{npz_file.stem}",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=3, reduction_factor=2),
        )

        t0 = time.time()
        per_trial_records: list[dict] = []

        def objective(trial):
            num_particles = trial.suggest_int("num_particles", args.num_particles_min, args.num_particles_max)
            w = trial.suggest_float("w", args.w_min, args.w_max)
            c1 = trial.suggest_float("c1", args.c1_min, args.c1_max)
            c2 = trial.suggest_float("c2", args.c2_min, args.c2_max)
            seed = trial.suggest_int("seed", 0, 100000)

            max_iter, realized = _iters_from_budget(args.budget, num_particles)
            print(
                f"    --> Trial {trial.number} | particles={num_particles} iter={max_iter} "
                f"(evals={realized}) w={w:.3f} c1={c1:.3f} c2={c2:.3f} seed={seed}"
            )

            bp, bl, lh = core.run_pso_torch(
                target_ir=target_ir,
                duration=args.duration,
                synth=synth,
                loss_fn=loss_fn,
                device=device,
                num_particles=num_particles,
                max_iter=max_iter,
                w=w,
                c1=c1,
                c2=c2,
                seed=seed,
            )
            trial.set_user_attr("best_params", bp)
            trial.set_user_attr("loss_history", lh)
            trial.set_user_attr("max_iter", max_iter)
            trial.set_user_attr("realized_budget", realized)

            per_trial_records.append(
                {
                    "trial_number": int(trial.number),
                    "final_loss": float(bl),
                    "evals_used": int(realized),
                    "num_particles": int(num_particles),
                    "max_iter": int(max_iter),
                    "w": float(w),
                    "c1": float(c1),
                    "c2": float(c2),
                    "seed": int(seed),
                }
            )
            return bl

        study.optimize(
            objective,
            n_trials=args.n_trials,
            callbacks=[
                lambda study, trial: stop_when_finished(study, trial, args.n_trials),
                lambda study, trial: stop_on_success(study, trial, args.early_stop_loss),
            ],
        )

        elapsed = time.time() - t0
        trials = study.get_trials(deepcopy=False)
        n_restarts_run = len(
            [
                t
                for t in trials
                if t.state
                in (
                    optuna.trial.TrialState.COMPLETE,
                    optuna.trial.TrialState.PRUNED,
                    optuna.trial.TrialState.FAIL,
                )
            ]
        )
        total_evals_used = int(
            sum(
                int(t.user_attrs.get("realized_budget", 0))
                for t in trials
                if t.state == optuna.trial.TrialState.COMPLETE
            )
        )
        best = study.best_trial
        best_params = best.user_attrs["best_params"]
        best_loss_hist = best.user_attrs["loss_history"]
        best_loss = best.value

        comp = core.to_composite(best_params)
        print(f"\n  Winner: Trial {best.number}, loss={best_loss:.6f} ({elapsed:.0f}s)")
        print(f"    Restarts run: {n_restarts_run} / {args.n_trials}")
        print(f"    Total evals used: {total_evals_used}")
        for k in core.COMPOSITE_KEYS:
            print(f"    {k}: {comp[k]:.6f}")
        if gt_7:
            print(f"    NMSE: {core.compute_nmse_6d(best_params, gt_7):.6f}")

        if per_trial_records:
            pd.DataFrame(per_trial_records).to_csv(
                output_path / f"trials_{npz_file.stem}.csv", index=False
            )

        row = {
            "filename": npz_file.name,
            "target_npz": npz_file.name,
            "runtime": round(elapsed, 3),
            "best_loss": round(float(best_loss), 8),
            "n_restarts_run": int(n_restarts_run),
            "total_evals_used": total_evals_used,
            "optimizer": "PSO+Optuna_ES",
            "loss_name": args.loss,
            "budget": int(args.budget),
            "num_particles": int(best.params["num_particles"]),
            "max_iter": int(best.user_attrs["max_iter"]),
            "realized_budget": int(best.user_attrs["realized_budget"]),
            "w": float(best.params["w"]),
            "c1": float(best.params["c1"]),
            "c2": float(best.params["c2"]),
            "seed": int(best.params["seed"]),
            "optuna_trial": int(best.number),
            "gt_loss": round(gt_loss, 8) if gt_loss is not None else None,
            "nmse": round(core.compute_nmse_6d(best_params, gt_7), 8) if gt_7 else None,
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
        core.plot_per_ir(npz_file.stem, best_params, gt_7, best_loss_hist, elapsed, output_path, gt_loss)
        print(f"  [Saved] {npz_file.stem}")

    if all_results:
        pd.DataFrame(all_results).to_csv(output_path / "summary.csv", index=False)
        core.plot_summary(all_results, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dset_root", type=str, default="data/random-IR-100-1.0s")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="results/experiment_pruning_es/pso_es_5k")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loss", type=str, default="L1_STFT")
    parser.add_argument(
        "--storage",
        type=str,
        default="results/experiment_pruning_es/pso_es_5k/baseline_plus_es.db",
    )
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument(
        "--early_stop_loss",
        type=float,
        default=0.01,
        help="Stop remaining Optuna trials for an IR once any COMPLETE trial goes below this loss",
    )

    parser.add_argument("--num_particles_min", type=int, default=30)
    parser.add_argument("--num_particles_max", type=int, default=60)
    parser.add_argument("--w_min", type=float, default=0.1)
    parser.add_argument("--w_max", type=float, default=0.9)
    parser.add_argument("--c1_min", type=float, default=0.5)
    parser.add_argument("--c1_max", type=float, default=3.0)
    parser.add_argument("--c2_min", type=float, default=0.5)
    parser.add_argument("--c2_max", type=float, default=3.0)

    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
