#!/usr/bin/env python3
"""
DAFx MASTER EXPERIMENT RUNNER: 3-WAY HYBRID LOSS BATTLE
======================================================
Strategy: 32 Particles / 125 Iterations (2 Waves per 16-core CPU)
Metrics: Hybrid Loss (Schroeder EDC + Log-MSS)
"""

import subprocess
import sys
import time
import pandas as pd
from pathlib import Path

# ===========================
# MASTER CONFIGURATION
# ===========================
TARGET_FOLDER = "random-IR-10-1.0s"
RANDOM_SEED = 42
POP = "32"
ITER = "125"

# Output Directories
PSO_DIR = "results_pso_32_125"
GWO_DIR = "results_gwo_32_125"
CMA_DIR = "results_cma_32_125"

def run_command(cmd_list, description):
    print(f"\n>>> STARTING: {description}")
    print(f"Running: {' '.join(cmd_list)}")
    start = time.time()
    try:
        subprocess.run(cmd_list, check=True)
        duration = (time.time() - start) / 60
        print(f">>> SUCCESS: {description} finished in {duration:.2f} minutes.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[!] ERROR: {description} failed with exit code {e.returncode}")
        return False

def extract_metrics(csv_path):
    """Helper to pull the key metrics from the eval CSV."""
    if not Path(csv_path).exists():
        return None
    df = pd.read_csv(csv_path)
    # Assumes eval.py outputs a CSV with columns ['metric', 'value']
    return dict(zip(df['metric'], df['value']))

if __name__ == "__main__":
    print("="*60)
    print("DAFx MASTER RUNNER: DEEP SEARCH HEAD-TO-HEAD (32/125)")
    print("="*60)
    
    overall_start = time.time()

    # 1. RUN PSO OPTIMIZATION
    pso_ok = run_command([
        sys.executable, "TaskA/baseline.py", TARGET_FOLDER, 
        "--pop", POP, "--iter", ITER, "--out", PSO_DIR, "--seed", str(RANDOM_SEED)
    ], "PSO Optimization")

    # 2. RUN GWO OPTIMIZATION
    gwo_ok = run_command([
        sys.executable, "TaskA/baseline_gwo.py", TARGET_FOLDER, 
        "--pop", POP, "--iter", ITER, "--out", GWO_DIR, "--seed", str(RANDOM_SEED)
    ], "GWO Optimization")

    # 3. RUN CMA-ES OPTIMIZATION
    cma_ok = run_command([
        sys.executable, "TaskA/baseline_cmaes.py", TARGET_FOLDER, 
        "--pop", POP, "--iter", ITER, "--out", CMA_DIR, "--seed", str(RANDOM_SEED)
    ], "CMA-ES Optimization")

    # 4. RUN EVALUATIONS
    # We run evaluation on each folder to compute NMSE against ground truth
    eval_tasks = [
        (pso_ok, PSO_DIR, "PSO"),
        (gwo_ok, GWO_DIR, "GWO"),
        (cma_ok, CMA_DIR, "CMA")
    ]
    
    for ok, directory, name in eval_tasks:
        if ok:
            run_command([
                sys.executable, "TaskA/eval.py", directory, TARGET_FOLDER
            ], f"{name} Evaluation")

    # 5. FINAL COMPARISON TABLE
    print("\n" + "="*80)
    print("FINAL PERFORMANCE SUMMARY (32 Pop / 125 Iterations)")
    print("="*80)

    results_data = []
    for name, directory in [("PSO", PSO_DIR), ("GWO", GWO_DIR), ("CMA", CMA_DIR)]:
        metrics = extract_metrics(f"{directory}/evaluation_summary.csv")
        if metrics:
            results_data.append({
                "Method": name,
                "Param NMSE (Mean)": metrics.get('parameter_nmse_mean', 0),
                "Spectral MSE (Mean)": metrics.get('spectral_mse_mean', 0)
            })

    if results_data:
        comp_df = pd.DataFrame(results_data)
        pd.options.display.float_format = '{:.6f}'.format
        print(comp_df.to_string(index=False))
        
        # Identify Winners
        best_phys = comp_df.loc[comp_df['Param NMSE (Mean)'].idxmin(), 'Method']
        best_sound = comp_df.loc[comp_df['Spectral MSE (Mean)'].idxmin(), 'Method']
        
        print("\n" + "-"*80)
        print(f"WINNER (Physical Parameters): {best_phys}")
        print(f"WINNER (Acoustic Match):      {best_sound}")
        print("-"*80)
    else:
        print("Error: Missing evaluation summaries. Check logs for failures.")

    total_duration = (time.time() - overall_start) / 60
    print(f"\nTotal Master Run Time: {total_duration:.2f} minutes.")
    print("="*80)