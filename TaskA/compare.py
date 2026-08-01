import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION ---
METHODS = {
    "PSO": "experiment_results_taskAmss",
    "GWO": "experiment_results_taskA_gwomss",
    "CMA-ES": "experiment_results_taskA_cmaesmss"
}
OUTPUT_VIS_DIR = "comparison_visuals"
Path(OUTPUT_VIS_DIR).mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

def standardize_id(val):
    """
    Standardizes 'random_IR_0001.wav', '0001', or '1' into a padded '0001'.
    This ensures the 'Intersection' logic can actually find matching files.
    """
    val = str(val)
    match = re.search(r'(\d+)', val)
    if match:
        return match.group(1).zfill(4)
    return val

def parse_detailed_log(log_path):
    """Parses log to get individual convergence lines for all targets."""
    if not os.path.exists(log_path):
        return None
    
    target_histories = {}
    current_target = None
    
    target_pattern = re.compile(r"Target \d+/\d+: ([\w\.]+)")
    iter_pattern = re.compile(r"Iter (\d+)/\d+, Best Loss: ([\d\.]+)")
    
    with open(log_path, 'r') as f:
        for line in f:
            t_match = target_pattern.search(line)
            if t_match:
                # Standardize the key immediately
                current_target = standardize_id(t_match.group(1))
                target_histories[current_target] = []
                continue
            
            i_match = iter_pattern.search(line)
            if i_match and current_target:
                target_histories[current_target].append(float(i_match.group(2)))
                
    return target_histories

def generate_visuals():
    method_dataframes = {}
    method_histories = {}
    
    print(">>> Phase 1: Loading and Standardizing Data...")
    
    # 1. LOAD ALL DATA AND NORMALIZE COLUMN NAMES
    for name, folder in METHODS.items():
        folder_path = Path(folder)
        res_path = folder_path / "evaluation_results.csv"
        exp_path = folder_path / "experiment_summary.csv"
        log_path = folder_path / "experiment.log"

        if res_path.exists() and exp_path.exists():
            df_eval = pd.read_csv(res_path)
            df_exp = pd.read_csv(exp_path)
            
            # Unify Column Names
            rename_map = {
                'file_id': 'target_id', 
                'filename': 'target_id', 
                'target': 'target_id',
                'target_file': 'target_id'
            }
            df_eval = df_eval.rename(columns=rename_map)
            df_exp = df_exp.rename(columns=rename_map)

            # Apply ID Standardization ('random_IR_0001.wav' -> '0001')
            df_eval['target_id'] = df_eval['target_id'].apply(standardize_id)
            df_exp['target_id'] = df_exp['target_id'].apply(standardize_id)

            # Merge to ensure we only use rows where we have both Loss and NMSE
            merged = pd.merge(df_eval, df_exp, on='target_id')
            
            if not merged.empty:
                method_dataframes[name] = merged
                method_histories[name] = parse_detailed_log(log_path)
            else:
                print(f"[!] Warning: Merge failed for {name}. Check ID formats.")

    if not method_dataframes:
        print("[!] No valid results found. Check your folder names and repair summaries.")
        return

    # 2. FIND THE FAIR INTERSECTION
    common_targets = set.intersection(*(set(df['target_id']) for df in method_dataframes.values()))
    common_targets = sorted(list(common_targets))
    
    if not common_targets:
        print("[!] Intersection is empty. Algorithms processed different files.")
        return

    print(f">>> Found {len(common_targets)} common targets: {common_targets}")

    # 3. FILTER DATA AND BUILD PLOT LISTS
    filtered_scatter_list = []
    filtered_summary_list = []
    
    for name, df in method_dataframes.items():
        df_filtered = df[df['target_id'].isin(common_targets)]
        
        for _, row in df_filtered.iterrows():
            filtered_scatter_list.append({
                "Method": name,
                "Hybrid_Loss": row['best_loss'],
                "NMSE": row['parameter_nmse']
            })
            
        filtered_summary_list.append({
            "Method": name,
            "Mean_NMSE": df_filtered['parameter_nmse'].mean()
        })

    # --- PLOT 1: INDIVIDUAL CONVERGENCE (10 lines per algorithm) ---
    print(">>> Phase 2: Generating Individual Convergence Plots...")
    for name, histories in method_histories.items():
        if histories:
            plt.figure(figsize=(10, 6))
            valid_his = []
            for target in common_targets:
                if target in histories and len(histories[target]) > 0:
                    plt.plot(range(1, len(histories[target])+1), histories[target], alpha=0.4, color='gray')
                    valid_his.append(histories[target])
            
            if valid_his:
                mean_loss = np.mean(valid_his, axis=0)
                plt.plot(range(1, len(mean_loss)+1), mean_loss, color='blue', lw=3, label=f'{name} Mean')
            
            plt.yscale('log')
            plt.title(f"{name}: Convergence Stability (N={len(common_targets)})", fontsize=14)
            plt.xlabel("Iteration")
            plt.ylabel("Hybrid Loss (Log Scale)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_VIS_DIR}/convergence_{name.lower()}.png")
            plt.close()

    # --- PLOT 2: MASTER CONVERGENCE COMPARISON (Fair Subset Only) ---
    print(">>> Phase 3: Generating Master Convergence Comparison...")
    plt.figure(figsize=(10, 6))
    for name, histories in method_histories.items():
        valid_his = [histories[t] for t in common_targets if t in histories and len(histories[t]) > 0]
        if valid_his:
            mean_loss = np.mean(valid_his, axis=0)
            plt.plot(range(1, len(mean_loss)+1), mean_loss, label=name, lw=2.5)
    
    plt.yscale('log')
    plt.title(f"Master Convergence (Averaged over {len(common_targets)} Targets)", fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Hybrid Loss (Log Scale)")
    plt.legend()
    plt.savefig(f"{OUTPUT_VIS_DIR}/convergence_master.png")
    plt.close()

    # --- PLOT 3: HYBRID LOSS VS PARAMETER NMSE (The Honesty Plot) ---
    print(">>> Phase 4: Generating Correlation Scatter Plot...")
    if filtered_scatter_list:
        df_scatter = pd.DataFrame(filtered_scatter_list)
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=df_scatter, x="Hybrid_Loss", y="NMSE", hue="Method", style="Method", s=150)
        
        # Global Trend Line
        sns.regplot(data=df_scatter, x="Hybrid_Loss", y="NMSE", scatter=False, color='black', 
                    line_kws={"linestyle":"--", "alpha":0.4}, label="System-wide Trend")
        
        plt.title("Correlation: Optimization Loss vs. Ground Truth Accuracy", fontsize=14)
        plt.xlabel("Final Hybrid Loss (Lower is Better)")
        plt.ylabel("Parameter NMSE (Lower is Better)")
        plt.legend()
        plt.savefig(f"{OUTPUT_VIS_DIR}/loss_vs_nmse_correlation.png")
        plt.close()

    # --- PLOT 4: NMSE BAR CHART (Zoomed Range) ---
    print(">>> Phase 5: Generating Zoomed Accuracy Bar Chart...")
    if filtered_summary_list:
        df_sum = pd.DataFrame(filtered_summary_list)
        plt.figure(figsize=(8, 6))
        
        # Modern Seaborn syntax to avoid warnings
        ax = sns.barplot(x="Method", y="Mean_NMSE", hue="Method", data=df_sum, palette="muted", legend=False)
        
        # Calculation for Detail Zoom
        y_min, y_max = df_sum['Mean_NMSE'].min(), df_sum['Mean_NMSE'].max()
        padding = (y_max - y_min) * 0.3 if y_max != y_min else 0.01
        plt.ylim(max(0, y_min - padding), y_max + padding)
        
        plt.title(f"Mean Parameter NMSE (Detail Zoom, N={len(common_targets)})", fontsize=14)
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.5f}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontweight='bold')
        
        plt.savefig(f"{OUTPUT_VIS_DIR}/nmse_comparison_zoomed.png")
        plt.close()

if __name__ == "__main__":
    print("="*60)
    print("DAFx ACADEMIC VISUALIZATION ENGINE")
    print("="*60)
    generate_visuals()
    print("\n[SUCCESS] Visualizations saved to the 'comparison_visuals' folder.")