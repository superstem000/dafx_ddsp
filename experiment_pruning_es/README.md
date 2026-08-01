# experiment_pruning_es

Tests whether SHP pruning hurts CMA-ES, and whether CMA-ES still beats PSO when both have study-level early stop at loss < 0.01.

## Run

```bash
bash experiment_pruning_es/scripts/cmaes_noprune.sh
bash experiment_pruning_es/scripts/pso_es.sh
```

Both write to `results/experiment_pruning_es/`. Single seed (LHS=42). Estimated total wall clock ~10h.

## Analyze

```bash
python -m experiment_pruning_es.analysis.compare_runs \
    --runs \
        cmaes_norm_es=results/cmaes_norm_es/l1_stft \
        cmaes_noprune=results/experiment_pruning_es/cmaes_noprune_5k \
        pso_plusplus=results/baseline_pso_plusplus \
        pso_es=results/experiment_pruning_es/pso_es_5k \
    --analysis_dir results/experiment_pruning_es/analysis
```

Outputs go to `results/experiment_pruning_es/analysis/`: `summary_table.csv`, `compute_vs_nmse.png`, `convergence_rate.png`, `paired_nmse_*.png`, `failure_overlap.csv`.