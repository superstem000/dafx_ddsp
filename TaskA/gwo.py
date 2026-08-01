# Real-valued GWO (Grey Wolf Optimizer) with boundary handling
# Drop-in replacement for pso.py

import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
import psutil
import os
import logger

def set_worker_priority(worker_seed):

    """Forces each worker process to High Priority as soon as it spawns."""
    import numpy as np
    import random
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception:
        pass

class GWO:
    def __init__(self, cost_func, bounds, num_particles=32, max_iter=125, max_workers=None, seed=42, **kwargs):
        self.seed = seed
        self.original_cost_func = cost_func
        self.original_bounds = np.array(bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.dim = len(bounds)
        self.max_workers = max_workers
        
        self.param_mins = self.original_bounds[:, 0]
        self.param_ranges = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        self.normalized_bounds = np.array([[0.0, 1.0] for _ in range(self.dim)])
    
    def normalize_params(self, params):
        return (params - self.param_mins) / self.param_ranges
    
    def denormalize_params(self, normalized_params):
        return normalized_params * self.param_ranges + self.param_mins
    
    def normalized_cost_function(self, normalized_params):
        original_params = self.denormalize_params(normalized_params)
        return self.original_cost_func(original_params)

    def optimize(self):
        start_time = time.time()
        
        print(f"Starting GWO optimization...")
        print(f"Parameters: {self.num_particles} wolves, {self.max_iter} iterations")
        print(f"Using parallel evaluation with {self.max_workers} workers")
        
        half_pack = self.num_particles // 2
        pos1 = np.random.uniform(0.0, 1.0, (half_pack, self.dim))
        pos2 = 1.0 - pos1 
        positions = np.vstack([pos1, pos2])
        
        alpha_pos, alpha_score = np.zeros(self.dim), float('inf')
        beta_pos, beta_score = np.zeros(self.dim), float('inf')
        delta_pos, delta_score = np.zeros(self.dim), float('inf')
        
        last_alpha_score = float('inf')
        stagnation_counter = 0
        iteration_times = []

        use_pool = self.max_workers is None or self.max_workers > 1
        executor = ProcessPoolExecutor(
            max_workers=self.max_workers, 
            initializer=set_worker_priority,
            initargs=(self.seed,)
        ) if use_pool else None

        try:
            for i in range(self.max_iter):
                iter_start_time = time.time()
                
                if executor:
                    scores = np.array(list(executor.map(self.normalized_cost_function, positions)))
                else:
                    scores = np.array([self.normalized_cost_function(p) for p in positions])
                
                for j in range(self.num_particles):
                    fitness = scores[j]
                    if fitness < alpha_score:
                        delta_score, delta_pos = beta_score, beta_pos.copy()
                        beta_score, beta_pos = alpha_score, alpha_pos.copy()
                        alpha_score, alpha_pos = fitness, positions[j].copy()
                    elif fitness < beta_score:
                        delta_score, delta_pos = beta_score, beta_pos.copy()
                        beta_score, beta_pos = fitness, positions[j].copy()
                    elif fitness < delta_score:
                        delta_score, delta_pos = fitness, positions[j].copy()

                if alpha_score >= last_alpha_score:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                last_alpha_score = alpha_score

                a = 2.0 - 2.0 * (i / self.max_iter)**2
                
                eps = 1e-8
                w1 = 1.0 / (alpha_score + eps)
                w2 = 1.0 / (beta_score + eps)
                w3 = 1.0 / (delta_score + eps)
                total_w = w1 + w2 + w3
                w1, w2, w3 = w1/total_w, w2/total_w, w3/total_w

                for j in range(self.num_particles):
                    if not np.array_equal(positions[j], alpha_pos):
                        for d in range(self.dim):
                            r1_a, r2_a = np.random.rand(), np.random.rand()
                            X1 = alpha_pos[d] - (2*a*r1_a-a) * abs(2*r2_a * alpha_pos[d] - positions[j,d])
                            
                            r1_b, r2_b = np.random.rand(), np.random.rand()
                            X2 = beta_pos[d] - (2*a*r1_b-a) * abs(2*r2_b * beta_pos[d] - positions[j,d])
                            
                            r1_d, r2_d = np.random.rand(), np.random.rand()
                            X3 = delta_pos[d] - (2*a*r1_d-a) * abs(2*r2_d * delta_pos[d] - positions[j,d])
                            
                            positions[j, d] = (w1*X1) + (w2*X2) + (w3*X3)

                        if stagnation_counter >= 3 and np.random.rand() < 0.30:
                            positions[j] = np.random.uniform(0.0, 1.0, self.dim)
                            
                    else:
                        if stagnation_counter >= 2:
                            noise = np.random.normal(0, 0.01, self.dim)
                            positions[j] = np.clip(positions[j] + noise, 0.0, 1.0)
                    
                    positions[j] = np.clip(positions[j], 0.0, 1.0)

                iteration_times.append(time.time() - iter_start_time)
                print(f"Iter {i+1}/{self.max_iter}, Best Loss: {alpha_score:.6f} {'[STUCK]' if stagnation_counter >= 3 else ''}")
                
        finally:
            if executor:
                executor.shutdown(wait=True)
        
        return self.denormalize_params(alpha_pos), alpha_score, None, None, time.time()-start_time, np.mean(iteration_times)