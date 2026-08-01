"""
Gradient-Alignment ELA (finite differences, batched, lazy-loaded)
==================================================================

For each (audio loss, target IR), measures along straight-line paths from
random LHS-style starts to GT:

  1. Alignment: cos θ between -∇L(x) (descent direction) and (x_GT - x)
     (direction toward GT).
  2. Slope: ‖∇L(x)‖, gradient magnitude in normalized parameter space.

Method:
  - Forward-difference gradient. For each sample point x ∈ [0,1]^7:
      ∂L/∂x_i ≈ [L(x + ε·e_i) - L(x)] / ε         (i = 1..7)
    Total 8 forward-only evaluations per gradient (1 base + 7 axis perturbations).
    No autograd — much smaller memory footprint than backprop.
  - Batched: stack all (8 × n_batch) evaluations into one synth+loss forward
    pass per chunk.
  - Lazy-loaded: only one loss in VRAM at a time.

Why finite differences (vs autograd):
  - 6GB consumer GPUs cannot fit autograd graphs through this synth at
    duration=0.25 (each retained intermediate is [num_modes, 11025]).
  - Forward-only computation has no graph retention; all temporaries free
    after each forward pass. Allows much larger batch sizes.
  - Loss-agnostic: works for losses with non-differentiable ops (phase wrap
    modulo in GroupDelay/InstFreq, etc.) by averaging over ε-scale
    perturbations.

Conceptual grounding:
  - Gradient direction relative to known optimum as "search direction
    quality": Müller (2023, arxiv 2308.09556).
  - GT-anchored landscape feature in the FDC tradition (Jones & Forrest 1995),
    differing by being trajectory-based and continuous-derivative rather
    than rank-correlation aggregate.

Usage:
  python -m src.cmaes.ela_gradient
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.stats.qmc import Sobol

import torch

from src.loss.loss_selector import select_loss_function
from src.plate.SevenParamPlate import BatchedModalPlateTorch

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

SAMPLE_RATE = 44100
N_PARAMS = 7
EVALS_PER_GRADIENT = N_PARAMS + 1  # 7 perturbations + 1 base

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

BOUNDS_LO_NP = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS])
BOUNDS_HI_NP = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS])
BOUNDS_RANGE_NP = BOUNDS_HI_NP - BOUNDS_LO_NP

FIXED_PLATE_PARAMS = {
    "Lx": 1.0, "nu": 0.25, "T60_DC": 6.0, "T60_F1": 2.0,
    "loss_F1": 500.0, "fp_x": 0.335, "fp_y": 0.467,
}

# Full 14-loss list — finite differences works for non-differentiable losses too
DEFAULT_LOSSES = [
    "L1_STFT", "ESR", "GroupDelay", "ComplexSTFT", "Gammatone",
    "CQT+LogDec", "SC+LogMag", "VQT", "Envelope", "InstFreq",
    "MSS", "Dispersion", "CQT_L1", "Mel",
]


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def _is_oom_error(e):
    if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return True
    return False


def _safe_empty_cache(device):
    if device != "cuda":
        return True
    try:
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        if _is_oom_error(e):
            return False
        raise


def _gpu_memory_str(device):
    if device != "cuda":
        return ""
    try:
        free, total = torch.cuda.mem_get_info()
        return f"({(total-free)/1e9:.1f}/{total/1e9:.1f} GB used)"
    except Exception:
        return ""


def normalized_to_physical_np(x_norm):
    """[..., 7] normalized → [..., 7] physical, numpy."""
    return BOUNDS_LO_NP + x_norm * BOUNDS_RANGE_NP


def physical_to_plate14_np(phys_np):
    """[B, 7] physical (numpy) → [B, 14] plate array in PARAM_ORDER.
    No autograd needed; converted to torch outside."""
    B = phys_np.shape[0]
    E = phys_np[:, 0]
    rho = phys_np[:, 1]
    h = phys_np[:, 2]
    Ly = phys_np[:, 3]
    T0 = phys_np[:, 4]
    op_x = phys_np[:, 5]
    op_y = phys_np[:, 6]

    out = np.zeros((B, 14), dtype=np.float32)
    out[:, 0] = FIXED_PLATE_PARAMS["Lx"]
    out[:, 1] = Ly
    out[:, 2] = h
    out[:, 3] = T0
    out[:, 4] = rho
    out[:, 5] = E
    out[:, 6] = FIXED_PLATE_PARAMS["nu"]
    out[:, 7] = FIXED_PLATE_PARAMS["T60_DC"]
    out[:, 8] = FIXED_PLATE_PARAMS["T60_F1"]
    out[:, 9] = FIXED_PLATE_PARAMS["loss_F1"]
    out[:, 10] = FIXED_PLATE_PARAMS["fp_x"]
    out[:, 11] = FIXED_PLATE_PARAMS["fp_y"]
    out[:, 12] = op_x
    out[:, 13] = op_y
    return out


def normalize_audio(x_t):
    amax = x_t.abs().amax(dim=-1, keepdim=True)
    amax = torch.clamp(amax, min=1e-12)
    return x_t / amax


def generate_sobol_targets(n_targets, seed=42):
    sampler = Sobol(d=7, scramble=True, seed=seed)
    n_pow2 = 1
    while n_pow2 < n_targets + 1:
        n_pow2 *= 2
    raw = sampler.random(n_pow2)
    raw = np.clip(raw[1: n_targets + 1], 0.05, 0.95)
    return BOUNDS_LO_NP + raw * BOUNDS_RANGE_NP


def synthesize_target_ir(synth, gt_phys_np, duration, device, dtype):
    plate14 = physical_to_plate14_np(gt_phys_np.reshape(1, -1))
    plate14_t = torch.tensor(plate14, device=device, dtype=dtype)
    with torch.no_grad():
        ir = synth(plate14_t, duration=duration, normalize=False)[0]
    return ir


# ═══════════════════════════════════════════════════════════════════
# CORE: finite-difference gradient computation
# ═══════════════════════════════════════════════════════════════════

def build_path_points(gt_norm, n_paths, n_steps, rng):
    """All sample points across all paths.

    Returns:
        pts: [P, 7] sample points in normalized space, P = n_paths × (n_steps - 1)
        d_to_gt_unit: [P, 7] unit direction toward GT
        distances: [P] euclidean distance to GT
    """
    starts = rng.random((n_paths, 7)).astype(np.float32)
    alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)[:-1]

    pts = starts[:, None, :] + alphas[None, :, None] * (
        gt_norm[None, None, :] - starts[:, None, :]
    )
    pts = pts.reshape(-1, 7)
    pts = np.clip(pts, 0.0, 1.0)

    d_to_gt = gt_norm[None, :] - pts
    distances = np.linalg.norm(d_to_gt, axis=1)
    safe_d = np.where(distances[:, None] > 1e-12, d_to_gt, 0.0)
    safe_norm = np.where(distances > 1e-12, distances, 1.0)
    d_to_gt_unit = safe_d / safe_norm[:, None]

    return pts, d_to_gt_unit, distances


def expand_for_finite_difference(pts, eps):
    """Given P sample points, build [P*8, 7] expanded array with each base
    point followed by 7 axis-perturbed versions.

    Layout (for sample i ∈ [0, P)):
        row 8*i + 0: pts[i]                  (base)
        row 8*i + 1: pts[i] + eps * e_1
        row 8*i + 2: pts[i] + eps * e_2
        ...
        row 8*i + 7: pts[i] + eps * e_7

    Returns expanded [P*8, 7] with values clipped to [0, 1].
    """
    P = pts.shape[0]
    expanded = np.repeat(pts[:, None, :], EVALS_PER_GRADIENT, axis=1)  # [P, 8, 7]
    # Add eps to the i-th coordinate of the i-th perturbation row
    for i in range(N_PARAMS):
        expanded[:, i + 1, i] += eps
    expanded = expanded.reshape(-1, 7)
    expanded = np.clip(expanded, 0.0, 1.0)
    return expanded


def evaluate_loss_batched(
    synth, loss_fn, expanded_pts_np, target_normed,
    duration, device, dtype, batch_size,
):
    """Forward-only: evaluate loss at each row of expanded_pts.
    Adaptive batch sizing on OOM. Returns [N] loss values or raises
    PoisonedCudaError."""
    N = expanded_pts_np.shape[0]
    out = np.full(N, np.nan, dtype=np.float32)
    current_batch = batch_size
    pos = 0

    while pos < N:
        end = min(pos + current_batch, N)
        chunk_np = expanded_pts_np[pos:end]
        b = chunk_np.shape[0]

        try:
            with torch.no_grad():
                phys_np = normalized_to_physical_np(chunk_np)
                plate14 = physical_to_plate14_np(phys_np)
                plate14_t = torch.tensor(plate14, device=device, dtype=dtype)
                pred = synth(plate14_t, duration=duration, normalize=False)
                pred_normed = normalize_audio(pred)
                target_b = target_normed.expand(b, -1)
                losses = loss_fn(target_b, pred_normed)
                if losses.ndim == 0:
                    return None  # scalar loss; abort
                losses_np = losses.detach().cpu().numpy()
                out[pos:end] = losses_np
                del plate14_t, pred, pred_normed, losses
        except Exception as e:
            if _is_oom_error(e):
                cache_ok = _safe_empty_cache(device)
                if not cache_ok:
                    raise PoisonedCudaError("empty_cache OOM; CUDA context poisoned")
                if current_batch <= 1:
                    raise PoisonedCudaError("OOM at batch_size=1")
                current_batch = max(1, current_batch // 2)
                continue
            else:
                raise

        pos = end

    return out, current_batch


def compute_gradients_fd(
    synth, loss_fn, gt_norm, target_ir, duration, device, dtype,
    n_paths, n_steps, rng, batch_size, eps,
):
    """Compute finite-difference gradients at all sample points.
    Returns (cos_alignments, slopes, distances, working_batch_size)."""
    pts_np, d_to_gt_unit_np, distances_np = build_path_points(
        gt_norm, n_paths, n_steps, rng
    )
    P = pts_np.shape[0]

    # Expand for FD: P points × 8 evaluations each
    expanded_np = expand_for_finite_difference(pts_np, eps)  # [P*8, 7]

    target_normed = normalize_audio(target_ir.unsqueeze(0))

    # Effective batch size in points = batch_size / 8 (since each point needs 8 evals)
    # Pass the FULL evaluation count batch to evaluate_loss_batched, which batches by row.
    result = evaluate_loss_batched(
        synth, loss_fn, expanded_np, target_normed,
        duration, device, dtype, batch_size,
    )
    if result is None:
        return None, None, None, batch_size
    losses_flat, working_bs = result  # [P*8]

    # Reshape: [P, 8]
    losses_per_point = losses_flat.reshape(P, EVALS_PER_GRADIENT)

    # Forward-difference gradient
    L_base = losses_per_point[:, 0:1]                   # [P, 1]
    L_perturbed = losses_per_point[:, 1:]               # [P, 7]
    grads = (L_perturbed - L_base) / eps                # [P, 7]

    # Validity: finite, non-trivial gradient
    finite_grad = np.all(np.isfinite(grads), axis=1)
    grad_mags = np.linalg.norm(grads, axis=1)
    nontrivial = grad_mags > 1e-12
    finite_base = np.isfinite(L_base[:, 0])
    usable = finite_grad & nontrivial & finite_base

    # Cosine alignment: descent direction (-grad/|grad|) vs direction-to-GT
    cos_alignments = np.full(P, np.nan, dtype=np.float32)
    if usable.any():
        descent_unit = -grads[usable] / grad_mags[usable, None]
        cos_a = np.einsum('bi,bi->b', descent_unit, d_to_gt_unit_np[usable])
        cos_alignments[usable] = np.clip(cos_a, -1.0, 1.0).astype(np.float32)

    return (
        cos_alignments,
        grad_mags.astype(np.float32),
        distances_np.astype(np.float32),
        working_bs,
    )


class PoisonedCudaError(Exception):
    pass


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_targets", type=int, default=20)
    p.add_argument("--n_paths", type=int, default=20)
    p.add_argument("--n_steps", type=int, default=25)
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--eps", type=float, default=0.01,
                   help="Finite-difference perturbation in normalized [0,1]^7 space.")
    p.add_argument("--batch_size", type=int, default=128,
                   help="Forward-only batch (rows in flattened FD evaluation).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "float64"])
    p.add_argument("--output", type=str, default="ela_gradient_results")
    p.add_argument("--ela_summary", type=str, default="ela_results/ela_summary.csv")
    p.add_argument("--conv_summary", type=str,
                   default="results/loss_comparison/summary.csv")
    p.add_argument("--losses", nargs="+", default=DEFAULT_LOSSES)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  GRADIENT-ALIGNMENT ELA (finite differences, lazy-loaded)")
    print("=" * 60)
    print(f"  Device: {device} {_gpu_memory_str(device)}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Targets: {args.n_targets}")
    print(f"  Paths × steps per IR: {args.n_paths} × {args.n_steps - 1} = "
          f"{args.n_paths * (args.n_steps - 1)} points/loss/IR")
    print(f"  FD evals per gradient: {EVALS_PER_GRADIENT} (1 base + 7 axis perturbs)")
    print(f"  ε (normalized): {args.eps}")
    print(f"  Initial batch size: {args.batch_size} (adaptive)")
    print(f"  Losses ({len(args.losses)}): {args.losses}")

    synth = BatchedModalPlateTorch(
        sample_rate=SAMPLE_RATE, device=device, dtype=dtype,
        drop_sub_20hz_modes=False,
    )

    print(f"\n  Generating {args.n_targets} target IRs...")
    gt_phys_all = generate_sobol_targets(args.n_targets, seed=42)
    gt_norm_all = ((gt_phys_all - BOUNDS_LO_NP) / BOUNDS_RANGE_NP).astype(np.float32)
    target_irs = []
    for gt_phys in gt_phys_all:
        ir = synthesize_target_ir(synth, gt_phys, args.duration, device, dtype)
        target_irs.append(ir)

    # Per-loss accumulators
    per_loss_align_median = {name: [] for name in args.losses}
    per_loss_align_mean = {name: [] for name in args.losses}
    per_loss_align_q1 = {name: [] for name in args.losses}
    per_loss_align_q3 = {name: [] for name in args.losses}
    per_loss_slope_median = {name: [] for name in args.losses}
    per_loss_slope_q1 = {name: [] for name in args.losses}
    per_loss_slope_q3 = {name: [] for name in args.losses}
    per_loss_frac_pos = {name: [] for name in args.losses}
    per_loss_n_valid = {name: [] for name in args.losses}
    per_loss_total_time = {name: 0.0 for name in args.losses}
    per_loss_status = {name: "pending" for name in args.losses}
    per_loss_final_bs = {name: args.batch_size for name in args.losses}

    distance_bin_edges = np.linspace(0.0, np.sqrt(7.0), 11)
    align_by_dist_bin = {name: [[] for _ in range(len(distance_bin_edges) - 1)]
                          for name in args.losses}

    t_global_start = time.time()

    # ═══════════════════════════════════════════════════════════════
    # OUTER LOOP: lazy load one loss at a time
    # ═══════════════════════════════════════════════════════════════

    for loss_idx, loss_name in enumerate(args.losses):
        print(f"\n{'═' * 60}")
        print(f"  LOSS {loss_idx + 1}/{len(args.losses)}: {loss_name}")
        print(f"  Pre-load VRAM: {_gpu_memory_str(device)}")
        print(f"{'═' * 60}")

        try:
            loss_fn = select_loss_function(
                loss_name, sample_rate=SAMPLE_RATE, device=device,
            )
        except Exception as e:
            print(f"    Failed to load: {type(e).__name__}: {str(e)[:100]}")
            per_loss_status[loss_name] = "load_failed"
            continue

        print(f"  Loaded. VRAM: {_gpu_memory_str(device)}")

        current_bs = args.batch_size
        loss_t_start = time.time()
        loss_failed = False
        oom_streak = 0

        for t_idx in range(args.n_targets):
            if loss_failed:
                break

            gt_norm = gt_norm_all[t_idx]
            target_ir = target_irs[t_idx]
            ir_rng = np.random.default_rng(seed=t_idx * 1000)
            ir_t_start = time.time()

            try:
                cos_a, slopes, distances, working_bs = compute_gradients_fd(
                    synth, loss_fn, gt_norm, target_ir, args.duration,
                    device, dtype, args.n_paths, args.n_steps, ir_rng,
                    batch_size=current_bs, eps=args.eps,
                )
                current_bs = working_bs
                oom_streak = 0
            except PoisonedCudaError as e:
                print(f"    IR {t_idx + 1}: CUDA context unrecoverable ({e}). Aborting loss.")
                loss_failed = True
                per_loss_status[loss_name] = "cuda_poisoned"
                break
            except Exception as e:
                if _is_oom_error(e):
                    oom_streak += 1
                    print(f"    IR {t_idx + 1}: OOM (streak={oom_streak}); bs={current_bs}")
                    if oom_streak >= 3:
                        loss_failed = True
                        per_loss_status[loss_name] = "persistent_oom"
                        break
                    _safe_empty_cache(device)
                    continue
                else:
                    print(f"    IR {t_idx + 1}: ERROR ({type(e).__name__}: {str(e)[:80]})")
                    loss_failed = True
                    per_loss_status[loss_name] = "runtime_error"
                    break

            ir_elapsed = time.time() - ir_t_start

            if cos_a is None:
                print(f"    IR {t_idx + 1}: scalar loss returned, aborting loss")
                loss_failed = True
                per_loss_status[loss_name] = "scalar_loss"
                break

            valid = ~np.isnan(cos_a)
            n_valid = int(valid.sum())

            if n_valid < 10:
                per_loss_align_median[loss_name].append(np.nan)
                per_loss_align_mean[loss_name].append(np.nan)
                per_loss_align_q1[loss_name].append(np.nan)
                per_loss_align_q3[loss_name].append(np.nan)
                per_loss_slope_median[loss_name].append(np.nan)
                per_loss_slope_q1[loss_name].append(np.nan)
                per_loss_slope_q3[loss_name].append(np.nan)
                per_loss_frac_pos[loss_name].append(np.nan)
                per_loss_n_valid[loss_name].append(n_valid)
                print(f"    IR {t_idx + 1:>2d}/{args.n_targets}: "
                      f"only {n_valid} valid pts ({ir_elapsed:.1f}s, bs={current_bs})")
                continue

            cos_a_v = cos_a[valid]
            slopes_v = slopes[valid]
            distances_v = distances[valid]

            per_loss_align_median[loss_name].append(np.median(cos_a_v))
            per_loss_align_mean[loss_name].append(np.mean(cos_a_v))
            per_loss_align_q1[loss_name].append(np.percentile(cos_a_v, 25))
            per_loss_align_q3[loss_name].append(np.percentile(cos_a_v, 75))
            per_loss_slope_median[loss_name].append(np.median(slopes_v))
            per_loss_slope_q1[loss_name].append(np.percentile(slopes_v, 25))
            per_loss_slope_q3[loss_name].append(np.percentile(slopes_v, 75))
            per_loss_frac_pos[loss_name].append(float((cos_a_v > 0).mean()))
            per_loss_n_valid[loss_name].append(n_valid)

            bin_idx = np.digitize(distances_v, distance_bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, len(distance_bin_edges) - 2)
            for bb in range(len(distance_bin_edges) - 1):
                in_bin = bin_idx == bb
                if in_bin.any():
                    align_by_dist_bin[loss_name][bb].extend(
                        cos_a_v[in_bin].tolist()
                    )

            print(f"    IR {t_idx + 1:>2d}/{args.n_targets}: "
                  f"align={np.median(cos_a_v):+.3f} "
                  f"slope={np.median(slopes_v):.2e} "
                  f"({ir_elapsed:.1f}s, bs={current_bs})")

        loss_total = time.time() - loss_t_start
        per_loss_total_time[loss_name] = loss_total
        per_loss_final_bs[loss_name] = current_bs
        if not loss_failed:
            per_loss_status[loss_name] = "ok"

        del loss_fn
        gc.collect()
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        elapsed_global = time.time() - t_global_start
        n_done = loss_idx + 1
        avg_per_loss = elapsed_global / n_done
        remaining = avg_per_loss * (len(args.losses) - n_done)
        print(f"\n  {loss_name} done in {loss_total/60:.1f}m | "
              f"Status: {per_loss_status[loss_name]} | bs={current_bs} | "
              f"Total: {elapsed_global/60:.1f}m, ~{remaining/60:.1f}m remaining")
        print(f"  Post-unload VRAM: {_gpu_memory_str(device)}")

    # ═══════════════════════════════════════════════════════════════
    # AGGREGATE
    # ═══════════════════════════════════════════════════════════════

    summary_rows = []
    for name in args.losses:
        if not per_loss_align_median[name]:
            continue
        summary_rows.append({
            "loss": name,
            "alignment_median": np.nanmedian(per_loss_align_median[name]),
            "alignment_mean": np.nanmedian(per_loss_align_mean[name]),
            "alignment_q1": np.nanmedian(per_loss_align_q1[name]),
            "alignment_q3": np.nanmedian(per_loss_align_q3[name]),
            "slope_median": np.nanmedian(per_loss_slope_median[name]),
            "slope_q1": np.nanmedian(per_loss_slope_q1[name]),
            "slope_q3": np.nanmedian(per_loss_slope_q3[name]),
            "frac_positive_alignment": np.nanmedian(per_loss_frac_pos[name]),
            "n_valid_per_ir_median": int(np.median(per_loss_n_valid[name])),
            "n_irs_completed": len(per_loss_align_median[name]),
            "final_batch_size": per_loss_final_bs[name],
            "status": per_loss_status[name],
            "total_seconds": per_loss_total_time[name],
        })
    sdf = pd.DataFrame(summary_rows).sort_values("alignment_median", ascending=False)
    sdf.to_csv(out_dir / "gradient_summary.csv", index=False)

    print(f"\n{'═' * 95}")
    print(f"  RESULTS (sorted by alignment_median)")
    print(f"{'═' * 95}")
    print(f"  {'Loss':>15s} | {'Align':>6s} {'AQ1':>7s} {'AQ3':>7s} | "
          f"{'Slope':>10s} | {'IRs':>3s} | {'BS':>3s} | {'Time':>7s} | {'Status':>10s}")
    print("  " + "-" * 100)
    for _, r in sdf.iterrows():
        print(f"  {r['loss']:>15s} | {r['alignment_median']:>+6.3f} "
              f"{r['alignment_q1']:>+7.3f} {r['alignment_q3']:>+7.3f} | "
              f"{r['slope_median']:>10.3e} | {r['n_irs_completed']:>3d} | "
              f"{r['final_batch_size']:>3d} | {r['total_seconds']:>6.1f}s | "
              f"{r['status']:>10s}")

    failed = [n for n in args.losses if per_loss_status[n] not in ("ok", "pending")]
    if failed:
        print(f"\n  Failed/incomplete losses:")
        for n in failed:
            print(f"    {n}: {per_loss_status[n]}")

    # ═══════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════

    if len(sdf) == 0:
        print("\n  No successful losses — skipping plots.")
        return

    loss_names = sdf["loss"].tolist()
    n_losses = len(loss_names)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_losses))

    # ── Figure 1: alignment & slope bars ──
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_losses * 0.4)))
    for ax, key, title in [
        (axes[0], "alignment_median",
         "Median cos(θ) — descent direction vs direction-to-GT\n"
         "(+1 = aligned, 0 = perpendicular, -1 = anti-aligned)"),
        (axes[1], "slope_median",
         "Median ‖∇L‖ in normalized parameter space\n(higher = steeper)"),
    ]:
        vals = [sdf[sdf["loss"] == ln][key].values[0] for ln in loss_names]
        ax.barh(range(n_losses), vals, color=colors)
        ax.set_yticks(range(n_losses))
        ax.set_yticklabels(loss_names, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")
        if key == "alignment_median":
            ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                fmt = f" {v:+.3f}" if key == "alignment_median" else f" {v:.2e}"
                ax.text(v, i, fmt, va="center", fontsize=7)

    plt.suptitle("Gradient-Alignment ELA (FD) — descent quality along paths to GT",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "gradient_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: gradient_metrics.png")

    def match_field(loss_name, df, field):
        if df is None:
            return None
        ln = str(loss_name).lower().replace("_", "").replace("+", "")
        for _, row in df.iterrows():
            cn = str(row["loss"]).lower().replace("_", "").replace("+", "")
            if cn == ln:
                v = row.get(field)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    return v
        return None

    ela_path = Path(args.ela_summary)
    ela_df = pd.read_csv(ela_path) if ela_path.exists() else None
    conv_path = Path(args.conv_summary)
    conv_df = pd.read_csv(conv_path) if conv_path.exists() else None

    # ── Figure 2: alignment & slope vs monotonicity ──
    if ela_df is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, key, label in [
            (axes[0], "alignment_median", "Median cos(θ)"),
            (axes[1], "slope_median", "Median ‖∇L‖"),
        ]:
            xs, ys, names_plot = [], [], []
            for _, r in sdf.iterrows():
                mono = match_field(r["loss"], ela_df, "monotonicity")
                if mono is not None:
                    xs.append(r[key])
                    ys.append(mono)
                    names_plot.append(r["loss"])

            ax.scatter(xs, ys, s=70, zorder=5, color="steelblue", edgecolor="black")
            for x, y, n in zip(xs, ys, names_plot):
                ax.annotate(n, (x, y), textcoords="offset points",
                            xytext=(5, 5), fontsize=8)
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel("Monotonicity (existing ELA)", fontsize=10)
            ax.grid(True, alpha=0.3)
            if key == "alignment_median":
                ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

            if len(xs) >= 4:
                r_p, p_p = scipy_stats.pearsonr(xs, ys)
                r_s, p_s = scipy_stats.spearmanr(xs, ys)
                ax.set_title(f"{label} vs Monotonicity\n"
                             f"Pearson r={r_p:+.3f} (p={p_p:.3f}), "
                             f"Spearman ρ={r_s:+.3f} (p={p_s:.3f})",
                             fontweight="bold", fontsize=10)
            else:
                ax.set_title(f"{label} vs Monotonicity",
                             fontweight="bold", fontsize=10)

        plt.tight_layout()
        plt.savefig(out_dir / "gradient_vs_monotonicity.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved: gradient_vs_monotonicity.png")

    # ── Figure 3: alignment & slope vs convergence rate ──
    if conv_df is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, key, label in [
            (axes[0], "alignment_median", "Median cos(θ)"),
            (axes[1], "slope_median", "Median ‖∇L‖"),
        ]:
            xs, ys, names_plot = [], [], []
            for _, r in sdf.iterrows():
                cr = match_field(r["loss"], conv_df, "conv_rate")
                if cr is not None:
                    xs.append(r[key])
                    ys.append(cr)
                    names_plot.append(r["loss"])

            ax.scatter(xs, ys, s=70, zorder=5, color="darkorange", edgecolor="black")
            for x, y, n in zip(xs, ys, names_plot):
                ax.annotate(n, (x, y), textcoords="offset points",
                            xytext=(5, 5), fontsize=8)
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel("CMA-ES Convergence Rate", fontsize=10)
            ax.grid(True, alpha=0.3)
            if key == "alignment_median":
                ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

            if len(xs) >= 4:
                r_p, p_p = scipy_stats.pearsonr(xs, ys)
                r_s, p_s = scipy_stats.spearmanr(xs, ys)
                ax.set_title(f"{label} vs Convergence Rate\n"
                             f"Pearson r={r_p:+.3f} (p={p_p:.3f}), "
                             f"Spearman ρ={r_s:+.3f} (p={p_s:.3f})",
                             fontweight="bold", fontsize=10)
            else:
                ax.set_title(f"{label} vs Convergence Rate",
                             fontweight="bold", fontsize=10)

        plt.tight_layout()
        plt.savefig(out_dir / "gradient_vs_convergence.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved: gradient_vs_convergence.png")

    # ── Figure 4: alignment vs distance-to-GT ──
    bin_centers = 0.5 * (distance_bin_edges[:-1] + distance_bin_edges[1:])
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(loss_names):
        bins = align_by_dist_bin[name]
        bin_meds, bin_xs = [], []
        for bb, vals in enumerate(bins):
            if len(vals) >= 5:
                bin_meds.append(np.median(vals))
                bin_xs.append(bin_centers[bb])
        if bin_xs:
            ax.plot(bin_xs, bin_meds, marker="o", color=colors[i], label=name,
                    linewidth=1.5, markersize=4)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Distance from current point to GT (normalized)", fontsize=10)
    ax.set_ylabel("Median cos(θ) of descent direction with direction-to-GT", fontsize=10)
    ax.set_title("Gradient Alignment vs Distance to GT", fontweight="bold")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "alignment_along_path.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: alignment_along_path.png")

    # ── Figure 5: alignment vs slope per loss ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, ln in enumerate(loss_names):
        r = sdf[sdf["loss"] == ln].iloc[0]
        ax.scatter(r["alignment_median"], r["slope_median"], s=80, color=colors[i],
                   edgecolor="black", zorder=5)
        ax.annotate(ln, (r["alignment_median"], r["slope_median"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Median cos(θ)", fontsize=10)
    ax.set_ylabel("Median ‖∇L‖", fontsize=10)
    ax.set_yscale("log")
    ax.set_title("Alignment vs Slope per Loss", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "alignment_vs_slope.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: alignment_vs_slope.png")

    elapsed = time.time() - t_global_start
    print(f"\n  Total time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
    print(f"  All outputs in: {out_dir}/")


if __name__ == "__main__":
    main()