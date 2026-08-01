# CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
# Drop-in replacement for pso.py and gwo.py
# Requires: pip install cma

import numpy as np
import cma
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

class CMAES:
    def __init__(self, cost_func, bounds, num_particles=32, max_iter=125, max_workers=None, **kwargs):
        self.original_cost_func = cost_func
        self.original_bounds = np.array(bounds)
        self.num_particles = num_particles  # Referred to as 'lambda' in CMA-ES
        self.max_iter = max_iter
        self.dim = len(bounds)
        self.max_workers = max_workers
        
        # Capture the seed here so it's available in optimize()
        self.seed = kwargs.get('seed', np.random.randint(10000))
        
        # Calculate normalization parameters
        self.param_mins = self.original_bounds[:, 0]
        self.param_ranges = self.original_bounds[:, 1] - self.original_bounds[:, 0]
    
    def normalize_params(self, params):
        return (params - self.param_mins) / self.param_ranges
    
    def denormalize_params(self, normalized_params):
        return normalized_params * self.param_ranges + self.param_mins
    
    def normalized_cost_function(self, normalized_params):
        original_params = self.denormalize_params(normalized_params)
        return self.original_cost_func(original_params)

    def optimize(self):
        start_time = time.time()
        
        print(f"Starting CMA-ES optimization...")
        print(f"Parameters: {self.num_particles} population, {self.max_iter} iterations")
        print(f"Using parallel evaluation with {self.max_workers} workers")
        
        # --- 1. CMA-ES INITIALIZATION ---
        # Start in the center of the normalized space [0.5, ..., 0.5]
        initial_mean = np.full(self.dim, 0.5)
        sigma0 = 0.25
        
        # Configure CMA-ES
        cma_opts = {
            'bounds': [0.0, 1.0],
            'popsize': self.num_particles,
            'CMA_active': True,      # Pushes away from high-loss areas
            'seed': int(self.seed),
            'tolx': 1e-12,           # Replaces tolsigma: stops when cloud is microscopic
            'tolfun': 1e-12,         # Stops when loss improvements are microscopic
            'verbose': -9
        }
        
        es = cma.CMAEvolutionStrategy(initial_mean, sigma0, cma_opts)
        
        iteration_times = []
        global_best_pos = None
        global_best_score = float('inf')

        # 2. START PERSISTENT POOL
        use_pool = self.max_workers is None or self.max_workers > 1
        executor = ProcessPoolExecutor(
            max_workers=self.max_workers, 
            initializer=set_worker_priority,
            initargs=(self.seed,)
        ) if use_pool else None

        try:
            for i in range(self.max_iter):
                iter_start_time = time.time()
                
                # 3. ASK: Get a generation of candidate solutions
                candidates = es.ask()
                candidates = np.array(candidates)
                
                # 4. EVALUATE FITNESS
                if executor:
                    scores = np.array(list(executor.map(self.normalized_cost_function, candidates)))
                else:
                    scores = np.array([self.normalized_cost_function(p) for p in candidates])
                
                # 5. TELL: Update the covariance matrix
                es.tell(candidates, scores)
                
                # Track best score
                current_best_idx = np.argmin(scores)
                if scores[current_best_idx] < global_best_score:
                    global_best_score = scores[current_best_idx]
                    global_best_pos = candidates[current_best_idx].copy()

                iteration_times.append(time.time() - iter_start_time)
                print(f"Iter {i+1}/{self.max_iter}, Best Loss: {global_best_score:.6f}")
                
                if es.stop():
                    break

        finally:
            if executor:
                executor.shutdown(wait=True)
        
        return self.denormalize_params(global_best_pos), global_best_score, None, None, time.time()-start_time, np.mean(iteration_times)
