#!/usr/bin/env python3
"""
Baseline PSO Script for Task A
===============================
Optimized for 32 particles / 125 iterations / Hybrid Loss
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import soundfile as sf
import platform
import multiprocessing

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from TaskA.pso import PSO
from ModalPlate.ModalPlate import ModalPlate
from ModalPlate.ParamRange import (params as plate_params, 
                                   get_variable_params, 
                                   get_fixed_params,
                                   variable_params_to_full_params)
from TaskA.hybrid_loss import hybrid_loss
import TaskA.logger as logger

# ===========================
# CONFIGURATION
# ===========================
STFT_CONFIGS = [(512, 128), (2048, 512), (8192, 2048)]
SAMPLE_RATE = 44100
MAX_WORKERS = None 

# PSO Hyperparameters (Cognitive boost for small populations)
PSO_W = 0.8
PSO_C1 = 2.8
PSO_C2 = 0.5

# ===========================
# UTILITY FUNCTIONS
# ===========================

def load_target_files(target_folder):
    target_path = Path(target_folder)
    if not target_path.exists():
        raise ValueError(f"Target folder {target_folder} does not exist")
    wav_files = list(target_path.glob("*.wav"))
    if len(wav_files) == 0:
        raise ValueError(f"No *.wav files found in {target_folder}")
    
    targets = []
    print(f"Loading target files from {target_folder}")
    for wav_file in sorted(wav_files):
        try:
            audio, sr = librosa.load(wav_file, sr=SAMPLE_RATE)
            targets.append((audio, wav_file.name))
            print(f"    Loaded {wav_file.name}: {len(audio)} samples")
        except Exception as e:
            print(f"Error loading {wav_file}: {e}")
    return targets

def synthesize_plate(param_dict, duration, sample_rate=SAMPLE_RATE):
    return ModalPlate.synthesize_from_params(
        param_dict, 
        duration=duration, 
        method='velocity',
        sample_rate=sample_rate
    )

class PlateCostFunction:
    def __init__(self, target_audio, target_duration):
        self.target_audio = target_audio
        self.target_duration = target_duration

    def __call__(self, variable_params):
        param_dict = variable_params_to_full_params(variable_params)
        candidate = synthesize_plate(param_dict, duration=self.target_duration)
        # 30% Physics (EDC) / 70% Sound (Log-MSS)
        return hybrid_loss(self.target_audio, candidate, STFT_CONFIGS, alpha=0.3)

# ===========================
# MAIN OPTIMIZATION FUNCTION
# ===========================

def run_baseline_experiment(target_folder, particles, iterations, output_dir, seed=42):
    print("=" * 60)
    print(f"PSO EXPERIMENT: {output_dir}")
    print("=" * 60)
    
    if platform.system() == "Darwin":
        max_workers = 1
    elif MAX_WORKERS is None:
        max_workers = min(multiprocessing.cpu_count(), 16)
    else:
        max_workers = MAX_WORKERS

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    logger.initialize_logging(output_dir)
    
    targets = load_target_files(target_folder)
    results = []
    
    for i, (target_audio, filename) in enumerate(targets):
        print(f"\nTarget {i+1}/{len(targets)}: {filename}")
        target_duration = len(target_audio) / SAMPLE_RATE
        
        cost_func_instance = PlateCostFunction(target_audio, target_duration)
        variable_params = get_variable_params()
        bounds = [(v.low, v.high) for v in variable_params.values()]
        
        optimizer = PSO(
            cost_func_instance, bounds, 
            num_particles=particles, 
            max_iter=iterations,
            w=PSO_W, c1=PSO_C1, c2=PSO_C2, 
            max_workers=max_workers,
            seed=seed
        )
        
        best_v_params, best_loss, _, _, total_time, _ = optimizer.optimize()
        best_params = variable_params_to_full_params(best_v_params)
        
        base_name = Path(filename).stem
        target_index = base_name.split('_')[-1] if 'random_IR_' in base_name else f"{i+1:04d}"
        
        sf.write(output_path / f"best_audio_{target_index}.wav", 
                 synthesize_plate(best_params, target_duration), SAMPLE_RATE)
        pd.DataFrame([best_params]).to_csv(output_path / f"best_params_{target_index}.csv", index=False)
        
        results.append({
            'target_file': filename,
            'target_index': target_index,
            'best_loss': best_loss,
            'optimization_time': total_time,
            'iterations': particles * iterations
        })

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_path / "experiment_summary.csv", index=False)
    return results

def main():
    parser = argparse.ArgumentParser(description="DAFx PSO Runner")
    parser.add_argument('target_folder', nargs='?', default='random-IR-10-1.0s')
    parser.add_argument('--pop', type=int, default=32)
    parser.add_argument('--iter', type=int, default=125)
    parser.add_argument('--out', type=str, default='results_pso_32_125')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    run_baseline_experiment(args.target_folder, args.pop, args.iter, args.out, args.seed)

if __name__ == "__main__":
    main()