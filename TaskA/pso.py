# Real-valued PSO with boundary handling (param ranges) and parallelized cost function evaluation
# The optimize() function contains the whole optimization loop

import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
import psutil
import os
# Import logger to ensure global print override is active
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

class PSO:
    def __init__(self, cost_func, bounds, num_particles=50, max_iter=200, w=0.7, c1=1.5, c2=1.5, max_workers=None, seed=42):
        self.seed = seed
        self.original_cost_func = cost_func
        self.original_bounds = np.array(bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.dim = len(bounds)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_workers = max_workers
        
        # Calculate normalization parameters
        self.param_mins = self.original_bounds[:, 0]
        self.param_ranges = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        
        # PSO will work in normalized [0,1] space
        self.normalized_bounds = np.array([[0.0, 1.0] for _ in range(self.dim)])
    
    def normalize_params(self, params):
        """Convert from original parameter space to [0,1] normalized space."""
        return (params - self.param_mins) / self.param_ranges
    
    def denormalize_params(self, normalized_params):
        """Convert from [0,1] normalized space back to original parameter space."""
        return normalized_params * self.param_ranges + self.param_mins
    
    def normalized_cost_function(self, normalized_params):
        """Wrapper that denormalizes params before calling original cost function."""
        original_params = self.denormalize_params(normalized_params)
        return self.original_cost_func(original_params)


    def optimize(self):
        start_time = time.time()
        
        print(f"Starting PSO optimization...")
        print(f"Parameters: {self.num_particles} particles, {self.max_iter} iterations")
        print(f"Using parallel evaluation with {self.max_workers} workers")
        
        # --- 1. MIRROR INITIALIZATION (OBL) ---
        half_pack = self.num_particles // 2
        pos1 = np.random.uniform(0.0, 1.0, (half_pack, self.dim))
        pos2 = 1.0 - pos1 
        particles = np.vstack([pos1, pos2])

        # Initialize velocities randomly for all 80
        velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))
        personal_best = particles.copy()
        
        # Track min/max explored values
        original_particles = np.array([self.denormalize_params(p) for p in particles])
        min_explored = np.min(original_particles, axis=0).copy()
        max_explored = np.max(original_particles, axis=0).copy()

        # 2. START THE POOL ONCE (Crucial: Do this before initial evaluation)
        use_pool = self.max_workers is None or self.max_workers > 1
        # initializer=set_worker_priority is what locks them to High Priority
        executor = ProcessPoolExecutor(
            max_workers=self.max_workers, 
            initializer=set_worker_priority,
            initargs=(self.seed,)
        ) if use_pool else None

        try:
            # 3. INITIAL EVALUATION (Use the persistent executor)
            if executor:
                personal_best_scores = np.array(list(executor.map(self.normalized_cost_function, particles)))
            else:
                personal_best_scores = np.array([self.normalized_cost_function(p) for p in particles])
            
            global_best = personal_best[np.argmin(personal_best_scores)].copy()
            global_best_score = np.min(personal_best_scores)

            iteration_times = []

            # 4. THE MAIN LOOP
            for i in range(self.max_iter):
                iter_start_time = time.time()

                # Update inertia
                current_w = 0.9 - (i / self.max_iter) * (0.9 - 0.4)
                
                # Update positions
                for j in range(self.num_particles):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[j] = (current_w * velocities[j] +
                                    self.c1 * r1 * (personal_best[j] - particles[j]) +
                                    self.c2 * r2 * (global_best - particles[j]))
                    particles[j] = np.clip(particles[j] + velocities[j], 0.0, 1.0)
                
                # 5. EVALUATE ITERATION (Use the SAME persistent executor)
                if executor:
                    # No 'with' statement here! Just use the object.
                    scores = np.array(list(executor.map(self.normalized_cost_function, particles)))
                else:
                    scores = np.array([self.normalized_cost_function(p) for p in particles])
                
                # Update Best scores
                for j in range(self.num_particles):
                    if scores[j] < personal_best_scores[j]:
                        personal_best[j] = particles[j].copy()
                        personal_best_scores[j] = scores[j]
                
                best_idx = np.argmin(personal_best_scores)
                if personal_best_scores[best_idx] < global_best_score:
                    global_best = personal_best[best_idx].copy()
                    global_best_score = personal_best_scores[best_idx]
                
                iteration_times.append(time.time() - iter_start_time)
                print(f"Iter {i+1}/{self.max_iter}, Best Loss: {global_best_score:.6f}")

        finally:
            # 6. CLEAN SHUTDOWN (The processes only die here)
            if executor:
                executor.shutdown(wait=True)
        
        return self.denormalize_params(global_best), global_best_score, min_explored, max_explored, time.time() - start_time, np.mean(iteration_times)
