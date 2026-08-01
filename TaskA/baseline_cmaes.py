#!/usr/bin/env python3
"""
Baseline CMA-ES Script for Task A (Production Version)
======================================================
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
from TaskA.cmaes import CMAES
from ModalPlate.ModalPlate import ModalPlate
from ModalPlate.ParamRange import (params as plate_params, 
                                   get_variable_params, 
                                   get_fixed_params,
                                   variable_params_to_full_params)
from TaskA.mss_loss import multi_scale_spectral_loss
import TaskA.logger as logger

# ===========================
# CONFIGURATION
# ===========================
STFT_CONFIGS = [(512, 128), (2048, 512), (8192, 2048)]
SAMPLE_RATE = 44100
MAX_WORKERS = None  # Auto-detect (limit to 16 in code)

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

from TaskA.hybrid_loss import hybrid_loss # Assuming you saved it as hybrid_loss.py

class PlateCostFunction:
    def __init__(self, target_audio, target_duration):
        self.target_audio = target_audio
        self.target_duration = target_duration

    def __call__(self, variable_params):
        # 1. Generate candidate parameters
        param_dict = variable_params_to_full_params(variable_params)
        
        # 2. Synthesize audio (Treating as an empirical measurement) [cite: 219]
        candidate = synthesize_plate(param_dict, duration=self.target_duration)
        
        # 3. Use Hybrid Loss (30% Schroeder EDC / 70% Log-Spectral)
        # STFT_CONFIGS = [(512, 128), (2048, 512), (8192, 2048)]
        return hybrid_loss(self.target_audio, candidate, STFT_CONFIGS, alpha=0.3)

# ===========================
# MAIN OPTIMIZATION FUNCTION
# ===========================

def run_baseline_experiment(target_folder, particles, iterations, output_dir, seed=42):
    """Run CMA-ES with dynamic population, iterations, and output path."""
    print("=" * 60)
    print(f"CMA-ES EXPERIMENT: {output_dir}")
    print("=" * 60)
    
    # Multiprocessing Setup
    if platform.system() == "Darwin":
        max_workers = 1
    elif MAX_WORKERS is None:
        max_workers = min(multiprocessing.cpu_count(), 16)
    else:
        max_workers = MAX_WORKERS

    # Create dynamic output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize logging in the specific output folder
    logger.initialize_logging(output_dir)
    
    targets = load_target_files(target_folder)
    results = []
    
    for i, (target_audio, filename) in enumerate(targets):
        print(f"\nTarget {i+1}/{len(targets)}: {filename}")
        target_duration = len(target_audio) / SAMPLE_RATE
        
        variable_params = get_variable_params()
        bounds = [(v.low, v.high) for v in variable_params.values()]
        
        cost_func_instance = PlateCostFunction(target_audio, target_duration)
        
        # Pass the dynamic particles and iterations
        optimizer = CMAES(
            cost_func_instance, bounds, 
            num_particles=particles, 
            max_iter=iterations,
            max_workers=max_workers,
            seed=seed
        )
        
        best_v_params, best_loss, _, _, total_time, _ = optimizer.optimize()
        best_params = variable_params_to_full_params(best_v_params)
        
        # Save audio and params in the dynamic folder
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

    # Save summary CSV
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_path / "experiment_summary.csv", index=False)
    print(f"\nAll results saved to: {output_path.absolute()}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Deep Search CMA-ES")
    parser.add_argument('target_folder', nargs='?', default='random-IR-10-1.0s')
    parser.add_argument('--pop', type=int, default=32, help='Population size')
    parser.add_argument('--iter', type=int, default=125, help='Number of iterations')
    parser.add_argument('--out', type=str, default='results_cma_32_125', help='Output folder')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seeds
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    run_baseline_experiment(
        args.target_folder, 
        args.pop, 
        args.iter, 
        args.out, 
        args.seed
    )

if __name__ == "__main__":
    main()