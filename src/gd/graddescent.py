"""Gradient-descent fitting of the modal plate in the identifiable 6-parameter space.

Design notes (see git history for the previous raw-7-parameter version):

1. Search space.  The synthesized IR is an exact function of only
   S(P) = {mu, D_div_mu, T0_div_mu, Ly, op_x, op_y}; this is also the
   submission format (TaskA_submission/submission.csv).  The raw seven
   parameters carry an exact one-dimensional symmetry,

       (E, rho, h) -> (c^3 E, c rho, h / c)

   which leaves mu = rho*h and D/mu = E h^2 / (12(1-nu^2) rho) invariant, so the
   IR is unchanged.  Along that direction the gradient is exactly zero, and Adam
   -- which divides each coordinate by its own RMS gradient -- rescales the
   remaining floating-point noise back up to a full lr-sized step.  The iterate
   then random-walks along the flat valley.  Optimizing the composite space
   removes the degeneracy entirely.

2. mu is profiled out, not searched.  In SixParamPlate.forward, mu enters in
   exactly one place: ms = 0.25*mu*Lx*Ly, feeding P ∝ 1/ms.  Everything else
   (om_sq, sig, r, the DDx/DDy mode bounds, the couplings) depends only on the
   five shape parameters.  Hence

       y(mu) = y(mu_ref) * (mu_ref / mu)      exactly.

   So the optimal mu can be found by a 1-D search over a *rescaling* of one
   cached IR, with no re-synthesis.  This is variable projection: it is at least
   as good as optimizing mu jointly, and better conditioned.

3. No peak normalization by default.  The dataset npz stores the unnormalized
   IR precisely so that the amplitude carries mu (see ModalPlate/DatasetGen.py
   and the docstring of ternary_mu.load_target_ir_from_npz).  The CMA-ES stage-1
   normalizes and therefore needs stage-2 ternary_mu to recover mu; here mu
   comes out of the same run for free.

4. Bounds and the NMSE metric are imported from ternary_mu so the numbers are
   directly comparable to the CMA-ES pipeline.  (The previous version of this
   file recomputed D_div_mu bounds as D_MIN/MU_MAX .. D_MAX/MU_MIN, which is
   ~25x too wide because D and mu share h; that made its reported NMSE
   incomparable to, and far more flattering than, the CMA-ES numbers.)

5. Restarts are batched.  K Latin-hypercube starts are optimized as a single
   [K, 5] tensor under one Adam, since the losses in src/loss are already
   batched.  Combined with the opt-in gradient checkpointing in SixParamPlate,
   this keeps peak memory roughly independent of K.
"""

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats.qmc import LatinHypercube

from src.loss.loss_selector import select_loss_function
from src.mu_optimization.ternary_mu import (
    COMPOSITE_BOUNDS,
    COMPOSITE_KEYS,
    MU_MAX,
    MU_MIN,
    load_target_ir_from_npz,
    nmse_6d,
    seven_to_six,
)
from src.plate.SixParamPlate import BatchedModalPlateTorch as SixParamPlate

SAMPLE_RATE = 44100
NU = 0.25

# Raw seven-parameter bounds, used only to draw physically feasible starts.
RAW7_KEYS: Sequence[str] = ("E", "rho", "h", "Ly", "T0", "op_x", "op_y")
RAW7_BOUNDS: Dict[str, Tuple[float, float]] = {
    "E": (6.7e10, 22.0e10),
    "rho": (2430.0, 21230.0),
    "h": (0.001, 0.005),
    "Ly": (1.1, 4.0),
    "T0": (0.01, 1000.0),
    "op_x": (0.51, 1.0),
    "op_y": (0.51, 1.0),
}

# The five parameters Adam actually searches; mu is profiled out (see module docstring).
SHAPE_KEYS: Sequence[str] = ("D_div_mu", "T0_div_mu", "Ly", "op_x", "op_y")

# Coordinates spanning multiple decades are searched in log space so that a step
# is a relative change: D_div_mu spans ~2.9 decades, T0_div_mu ~6.6.
LOG_KEYS = {"mu", "D_div_mu", "T0_div_mu"}


class NormBox:
    """Maps a normalized vector z in [0, 1]^P to physical parameters.

    Linear coordinates use lo + z*(hi-lo); log coordinates use
    exp(log lo + z*(log hi - log lo)).  All bounds are strictly positive, so the
    log branch is always finite and torch.where cannot introduce NaN gradients.
    """

    def __init__(self, keys: Sequence[str], device: torch.device, dtype: torch.dtype):
        self.keys = list(keys)
        lo = np.array([COMPOSITE_BOUNDS[k][0] for k in self.keys], dtype=np.float64)
        hi = np.array([COMPOSITE_BOUNDS[k][1] for k in self.keys], dtype=np.float64)
        if np.any(lo <= 0.0):
            raise ValueError("NormBox assumes strictly positive bounds")

        self.lo_np, self.hi_np = lo, hi
        self.is_log_np = np.array([k in LOG_KEYS for k in self.keys], dtype=bool)

        self.lo = torch.as_tensor(lo, device=device, dtype=dtype)
        self.hi = torch.as_tensor(hi, device=device, dtype=dtype)
        self.log_lo = torch.as_tensor(np.log(lo), device=device, dtype=dtype)
        self.log_hi = torch.as_tensor(np.log(hi), device=device, dtype=dtype)
        self.is_log = torch.as_tensor(self.is_log_np, device=device)

    def to_physical(self, z: torch.Tensor) -> torch.Tensor:
        lin = self.lo + z * (self.hi - self.lo)
        log = torch.exp(self.log_lo + z * (self.log_hi - self.log_lo))
        return torch.where(self.is_log, log, lin)

    def to_unit_np(self, v: np.ndarray) -> np.ndarray:
        """Inverse map, in numpy, with clipping into [0, 1]."""
        v = np.asarray(v, dtype=np.float64)
        lin = (v - self.lo_np) / np.maximum(self.hi_np - self.lo_np, 1e-300)
        with np.errstate(divide="ignore", invalid="ignore"):
            log = (np.log(np.maximum(v, 1e-300)) - np.log(self.lo_np)) / (
                np.log(self.hi_np) - np.log(self.lo_np)
            )
        return np.clip(np.where(self.is_log_np, log, lin), 0.0, 1.0)


def _is_oom(exc: BaseException) -> bool:
    """CUDA OOM is a plain RuntimeError before torch 1.13, and typed after."""
    typed = getattr(torch.cuda, "OutOfMemoryError", None)
    return (typed is not None and isinstance(exc, typed)) or "out of memory" in str(exc).lower()


def _collect_param_csvs(dataset_dir: Path) -> List[Path]:
    return sorted(dataset_dir.glob("random_IR_params_*.csv"))


def _id_from_params_path(params_path: Path) -> str:
    return params_path.stem.split("_")[-1]


def _read_params_csv(csv_path: Path) -> Dict[str, float]:
    with csv_path.open("r", newline="") as f:
        row = next(csv.DictReader(f))
    return {k: float(v) for k, v in row.items()}


def feasible_h_interval(mu: float, d_div_mu: float) -> Optional[Tuple[float, float]]:
    """Range of plate thickness h consistent with a (mu, D_div_mu) pair.

    The composite bounding box is strictly larger than the image of the raw-7
    box, because for a fixed mu the choice of h pins rho = mu/h and then
    E = 12(1-nu^2) * D_div_mu * mu / h^3.  Returns None when no h satisfies both
    the rho and E bounds, i.e. when the composite point is not physically
    realizable.
    """
    rho_lo, rho_hi = RAW7_BOUNDS["rho"]
    e_lo, e_hi = RAW7_BOUNDS["E"]
    h_lo, h_hi = RAW7_BOUNDS["h"]
    scale = 12.0 * (1.0 - NU**2)

    # rho = mu / h within bounds  ->  h within [mu/rho_hi, mu/rho_lo]
    lo = max(h_lo, mu / rho_hi)
    hi = min(h_hi, mu / rho_lo)

    # E = scale * D_div_mu * mu / h^3 within bounds -> h^3 within [.../e_hi, .../e_lo]
    num = scale * d_div_mu * mu
    lo = max(lo, (num / e_hi) ** (1.0 / 3.0))
    hi = min(hi, (num / e_lo) ** (1.0 / 3.0))

    return (lo, hi) if lo <= hi else None


def composite_to_raw7(six: Dict[str, float]) -> Optional[Dict[str, float]]:
    """Pick one representative raw-7 witness for a composite point, or None."""
    interval = feasible_h_interval(float(six["mu"]), float(six["D_div_mu"]))
    if interval is None:
        return None
    h = math.sqrt(interval[0] * interval[1])  # geometric middle of the valid range
    mu = float(six["mu"])
    rho = mu / h
    E = 12.0 * (1.0 - NU**2) * float(six["D_div_mu"]) * mu / (h**3)
    T0 = float(six["T0_div_mu"]) * mu
    if not (RAW7_BOUNDS["T0"][0] <= T0 <= RAW7_BOUNDS["T0"][1]):
        return None
    return {
        "E": E,
        "rho": rho,
        "h": h,
        "Ly": float(six["Ly"]),
        "T0": T0,
        "op_x": float(six["op_x"]),
        "op_y": float(six["op_y"]),
    }


def generate_starts(n_starts: int, seed: int, init_space: str, box: NormBox) -> np.ndarray:
    """Latin-hypercube starts, returned as normalized [n_starts, 5] shape vectors.

    init_space='raw7' draws uniformly in the raw seven-parameter box and maps
    through to composite coordinates.  Every start is then physically feasible
    by construction, and the start distribution matches the CMA-ES prior.

    init_space='composite' samples the composite box directly (uniform in the
    normalized, i.e. log, coordinate).  This covers a strictly larger region
    than any real plate can reach.
    """
    if init_space == "composite":
        sampler = LatinHypercube(d=len(SHAPE_KEYS), seed=seed)
        return sampler.random(n=n_starts)

    sampler = LatinHypercube(d=len(RAW7_KEYS), seed=seed)
    unit = sampler.random(n=n_starts)
    lo = np.array([RAW7_BOUNDS[k][0] for k in RAW7_KEYS], dtype=np.float64)
    hi = np.array([RAW7_BOUNDS[k][1] for k in RAW7_KEYS], dtype=np.float64)
    raw = lo + unit * (hi - lo)

    rows = []
    for r in raw:
        p7 = {k: float(v) for k, v in zip(RAW7_KEYS, r)}
        six = seven_to_six(p7)
        rows.append(box.to_unit_np(np.array([six[k] for k in SHAPE_KEYS])))
    return np.asarray(rows, dtype=np.float64)


def solve_mu_by_scale(
    pred_ref: torch.Tensor,
    mu_ref: torch.Tensor,
    target: torch.Tensor,
    loss_fn,
    n_iters: int,
    loss_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact per-row mu solve by ternary search over an output rescaling.

    Because y(mu) = y(mu_ref) * (mu_ref/mu), searching mu is the same as
    searching a scalar multiplier s applied to a cached IR, with mu = mu_ref/s.
    No re-synthesis is performed, so this costs a handful of loss evaluations.

    Args:
        pred_ref: [K, T] IRs synthesized at mu_ref.
        mu_ref:   [K] mu values those IRs were synthesized at.
        target:   [1, T] target IR.

    Returns:
        (mu [K], loss [K]) at the optimum.
    """
    K = pred_ref.shape[0]
    tgt = target.to(loss_dtype).expand(K, -1)
    ref = pred_ref.to(loss_dtype)

    # mu in [MU_MIN, MU_MAX]  <->  s = mu_ref/mu in [mu_ref/MU_MAX, mu_ref/MU_MIN]
    log_ref = torch.log(mu_ref.to(loss_dtype))
    lo = log_ref - math.log(MU_MAX)
    hi = log_ref - math.log(MU_MIN)

    def loss_at(log_s: torch.Tensor) -> torch.Tensor:
        out = loss_fn(tgt, torch.exp(log_s).unsqueeze(1) * ref)
        return torch.nan_to_num(out, nan=1e12, posinf=1e12, neginf=1e12)

    with torch.no_grad():
        for _ in range(n_iters):
            third = (hi - lo) / 3.0
            m1, m2 = lo + third, hi - third
            take_left = loss_at(m1) < loss_at(m2)
            hi = torch.where(take_left, m2, hi)
            lo = torch.where(take_left, lo, m1)

        log_s = 0.5 * (lo + hi)
        best_loss = loss_at(log_s)

    mu = torch.clamp(mu_ref / torch.exp(log_s).to(mu_ref.dtype), MU_MIN, MU_MAX)
    return mu, best_loss.to(torch.float64)


def _synthesize(
    plate: SixParamPlate,
    z: torch.Tensor,
    mu: torch.Tensor,
    box: NormBox,
    duration: float,
    args,
) -> torch.Tensor:
    """Synthesize [K, T] IRs from normalized shape vectors z and explicit mu."""
    shape = box.to_physical(z)
    six = torch.cat([mu.unsqueeze(1), shape], dim=1)
    pred = plate(six, duration=duration, vel_calc=args.vel_calc, normalize=False)
    if args.normalize_pred:
        peak = torch.clamp(pred.abs().amax(dim=1, keepdim=True), min=1e-12)
        pred = pred / peak
    return pred


def _fit_batch(
    z0: np.ndarray,
    target: torch.Tensor,
    duration: float,
    args,
    plate: SixParamPlate,
    box: NormBox,
    loss_fn,
    device: torch.device,
    dtype: torch.dtype,
    loss_dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[float]]]:
    """Run batched multi-start Adam over one sub-batch of starts.

    Returns (best_z [K,5], best_mu [K], best_loss [K], per-restart loss history).
    """
    K = z0.shape[0]
    z = nn.Parameter(torch.as_tensor(z0, device=device, dtype=dtype))
    optim = torch.optim.Adam([z], lr=args.lr, betas=(args.adam_beta1, 0.999))

    if args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    elif args.lr_schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=args.patience, factor=0.5)
    else:
        sched = None

    mu = torch.full((K,), math.sqrt(MU_MIN * MU_MAX), device=device, dtype=dtype)
    best_loss = torch.full((K,), float("inf"), device=device, dtype=torch.float64)
    best_z = z.detach().clone()
    best_mu = mu.clone()
    hist: List[List[float]] = [[] for _ in range(K)]
    stale = 0

    for epoch in range(args.n_epochs):
        # Profile mu out periodically so the shape gradient is evaluated at a
        # near-optimal amplitude instead of being swamped by amplitude error.
        if args.mu_mode == "profile" and (epoch % args.mu_polish_every == 0):
            with torch.no_grad():
                ref = _synthesize(plate, z.detach(), mu, box, duration, args)
                mu, _ = solve_mu_by_scale(
                    ref, mu, target, loss_fn, args.mu_ternary_iters, loss_dtype
                )

        optim.zero_grad(set_to_none=True)
        pred = _synthesize(plate, z, mu, box, duration, args)
        loss = loss_fn(target.to(loss_dtype).expand(K, -1), pred.to(loss_dtype))

        # Keep a bad row from poisoning the whole batch: its contribution to the
        # summed objective is dropped, and its gradient is zeroed below.
        finite = torch.isfinite(loss)
        torch.where(finite, loss, torch.zeros_like(loss)).sum().backward()

        if z.grad is None:  # every row was non-finite; nothing to step on
            print(f"      iter {epoch}: no finite gradients, stopping this batch")
            break

        with torch.no_grad():
            g = z.grad
            bad = ~torch.isfinite(g).all(dim=1) | ~finite
            g[bad] = 0.0

            # Per-restart clipping; a single clip_grad_norm_ over the whole
            # tensor would couple independent restarts together.
            if args.grad_clip_type == "norm":
                gn = g.norm(dim=1, keepdim=True)
                g.mul_((args.grad_clip_value / (gn + 1e-12)).clamp(max=1.0))
            else:
                g.clamp_(-args.grad_clip_value, args.grad_clip_value)

        # Snapshot the parameters that actually produced `loss`, before stepping,
        # so the recorded best_z and best_loss refer to the same point.
        z_eval = z.detach().clone()

        optim.step()
        with torch.no_grad():
            z.clamp_(0.0, 1.0)  # projected step: no teleports, momentum stays valid

        lv = torch.where(finite, loss, torch.full_like(loss, float("inf"))).detach().to(torch.float64)
        improved = lv < best_loss
        best_loss = torch.where(improved, lv, best_loss)
        with torch.no_grad():
            best_z[improved] = z_eval[improved]
            best_mu[improved] = mu[improved]

        for i, v in enumerate(lv.tolist()):
            hist[i].append(v)

        if args.lr_schedule == "plateau":
            sched.step(float(lv.min()))
        elif sched is not None:
            sched.step()

        if epoch % max(1, args.n_epochs // 10) == 0:
            print(
                f"      iter {epoch:4d}/{args.n_epochs}  "
                f"best={float(best_loss.min()):.6e}  median={float(lv.median()):.6e}"
            )

        stale = 0 if bool(improved.any()) else stale + 1
        if float(best_loss.min()) <= args.early_stop_loss:
            print(f"      early stop at iter {epoch}: reached {float(best_loss.min()):.6e}")
            break
        if stale >= args.patience and args.lr_schedule != "plateau":
            print(f"      early stop at iter {epoch}: no improvement for {stale} iters")
            break

    # Final exact mu solve at each restart's best shape.
    if args.mu_mode != "fixed":
        with torch.no_grad():
            ref = _synthesize(plate, best_z, best_mu, box, duration, args)
            best_mu, best_loss = solve_mu_by_scale(
                ref, best_mu, target, loss_fn, args.mu_ternary_iters, loss_dtype
            )

    return (
        best_z.detach().cpu().numpy(),
        best_mu.detach().cpu().numpy(),
        best_loss.detach().cpu().numpy(),
        hist,
    )


def fit_one_ir(
    target_np: np.ndarray,
    starts: np.ndarray,
    args,
    plate: SixParamPlate,
    box: NormBox,
    loss_fn,
    device: torch.device,
    dtype: torch.dtype,
    loss_dtype: torch.dtype,
) -> Tuple[Dict[str, float], float, List[List[float]]]:
    """Fit all restarts for one IR, splitting into sub-batches and backing off on OOM."""
    duration = target_np.shape[0] / float(SAMPLE_RATE)
    target = torch.as_tensor(target_np, device=device, dtype=dtype).unsqueeze(0)
    if args.normalize_target:
        target = target / torch.clamp(target.abs().max(), min=1e-12)

    all_z, all_mu, all_loss, all_hist = [], [], [], []
    batch = min(args.restart_batch, starts.shape[0])
    idx = 0
    while idx < starts.shape[0]:
        chunk = starts[idx : idx + batch]
        try:
            z_b, mu_b, loss_b, hist_b = _fit_batch(
                chunk, target, duration, args, plate, box, loss_fn, device, dtype, loss_dtype
            )
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if batch == 1:
                print("      [OOM] cannot fit even a single restart; skipping remaining starts")
                break
            batch = max(1, batch // 2)
            print(f"      [OOM] retrying with restart_batch={batch}")
            continue

        all_z.append(z_b)
        all_mu.append(mu_b)
        all_loss.append(loss_b)
        all_hist.extend(hist_b)
        idx += chunk.shape[0]

    if not all_loss:
        raise RuntimeError("all restarts failed (out of memory)")

    z = np.concatenate(all_z, axis=0)
    mu = np.concatenate(all_mu, axis=0)
    loss = np.concatenate(all_loss, axis=0)
    winner = int(np.argmin(loss))

    z_t = torch.as_tensor(z[winner : winner + 1], device=device, dtype=dtype)
    shape = box.to_physical(z_t)[0].detach().cpu().numpy()
    est = {"mu": float(mu[winner])}
    est.update({k: float(v) for k, v in zip(SHAPE_KEYS, shape)})
    return est, float(loss[winner]), all_hist


def _plot_per_ir(
    ir_id: str,
    hist: Sequence[Sequence[float]],
    gt6: Dict[str, float],
    est6: Dict[str, float],
    elapsed: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for h in hist:
        ax.plot(np.arange(len(h)), h, linewidth=0.6, alpha=0.3)
    lengths = {len(h) for h in hist}
    if hist and len(lengths) == 1:
        ax.plot(
            np.arange(len(hist[0])),
            np.minimum.reduce([np.asarray(h, dtype=np.float64) for h in hist]),
            linewidth=1.8,
            color="red",
            label="best per iter",
        )
        ax.legend(loc="best", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"Convergence ({len(hist)} restarts)")
    ax.grid(True, alpha=0.3)

    # Error is shown in the same normalized coordinates the search uses, so that
    # coordinates spanning several decades are not visually flattened.
    ax = axes[1]
    keys = list(COMPOSITE_KEYS)
    x = np.arange(len(keys), dtype=np.float64)
    w = 0.35

    def unit(d: Dict[str, float]) -> np.ndarray:
        out = []
        for k in keys:
            lo, hi = COMPOSITE_BOUNDS[k]
            v = min(max(float(d[k]), lo), hi)
            if k in LOG_KEYS:
                out.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)))
            else:
                out.append((v - lo) / (hi - lo))
        return np.asarray(out)

    ax.bar(x - w / 2, unit(est6), width=w, label="Est", color="steelblue")
    ax.bar(x + w / 2, unit(gt6), width=w, label="GT", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalized value (log where applicable)")
    ax.set_title(f"NMSE_6d={nmse_6d(est6, gt6):.3e}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.suptitle(f"random_IR_{ir_id} ({elapsed:.0f}s)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_summary(rows: Sequence[Dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].barh([str(r["id"]) for r in rows], [r["best_loss"] for r in rows], color="steelblue")
    axes[0].set_xlabel("Loss")
    axes[0].set_title("Final loss per IR")
    axes[0].invert_yaxis()

    nm = np.asarray([r["nmse_6d"] for r in rows], dtype=np.float64)
    axes[1].semilogy(np.arange(len(nm)), np.maximum(nm, 1e-16), marker="o", linewidth=1.0)
    axes[1].set_xlabel("IR index")
    axes[1].set_ylabel("NMSE (6d)")
    axes[1].set_title(f"NMSE_6d  (median={np.median(nm):.3e})")
    axes[1].grid(True, alpha=0.3)

    rel = np.asarray([r["mu_rel_error"] for r in rows], dtype=np.float64)
    axes[2].semilogy(np.arange(len(rel)), np.maximum(rel, 1e-16), marker="o", color="coral")
    axes[2].set_xlabel("IR index")
    axes[2].set_ylabel("|mu_est - mu_gt| / mu_gt")
    axes[2].set_title(f"mu relative error (median={np.median(rel):.3e})")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def run(args) -> None:
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    req = str(args.device).strip().lower()
    if req == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(req)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    loss_dtype = torch.float64 if args.loss_dtype == "float64" else torch.float32

    if args.normalize_pred != args.normalize_target:
        print(
            "WARNING: normalize-pred and normalize-target differ; the amplitude "
            "comparison that identifies mu is then meaningless."
        )
    if args.normalize_pred and args.mu_mode != "fixed":
        print(
            "WARNING: peak normalization makes mu unidentifiable (it is a pure "
            "output scale). Falling back to --mu-mode fixed; recover mu with "
            "src.mu_optimization.ternary_mu instead."
        )
        args.mu_mode = "fixed"

    print(f"Device: {device}   dtype: {dtype}   loss dtype: {loss_dtype}")
    print(f"Loss: {args.loss}")
    print(f"Search space: {list(SHAPE_KEYS)} (log: {sorted(LOG_KEYS & set(SHAPE_KEYS))})")
    print(f"mu mode: {args.mu_mode}   restarts: {args.n_restarts} (batch {args.restart_batch})")
    print(f"Init space: {args.init_space}   seed: {args.seed}")
    if device.type == "cuda":
        print(f"CUDA: {torch.cuda.get_device_name(0 if device.index is None else device.index)}")

    loss_fn = select_loss_function(args.loss, sample_rate=SAMPLE_RATE, device=device)
    plate = SixParamPlate(
        sample_rate=SAMPLE_RATE,
        device=device,
        dtype=dtype,
        drop_sub_20hz_modes=False,
        chunk_elems=args.chunk_elems,
        grad_checkpoint=not args.no_grad_checkpoint,
    )
    box = NormBox(SHAPE_KEYS, device=device, dtype=dtype)

    param_files = _collect_param_csvs(dataset_dir)
    if not param_files:
        raise FileNotFoundError(f"No random_IR_params_*.csv found in {dataset_dir}")
    if args.num is not None:
        if args.num <= 0:
            raise ValueError("--num must be a positive integer")
        param_files = param_files[: args.num]

    starts = generate_starts(args.n_restarts, args.seed, args.init_space, box)

    rows: List[Dict[str, float]] = []
    print(f"Processing {len(param_files)} IR(s) from {dataset_dir}")

    for i, params_csv in enumerate(param_files, start=1):
        rid = _id_from_params_path(params_csv)
        npz_path = dataset_dir / f"random_IR_{rid}.npz"
        if not npz_path.exists():
            print(f"[{i}/{len(param_files)}] missing {npz_path.name}, skipping")
            continue

        gt6 = seven_to_six(_read_params_csv(params_csv))
        target_np = load_target_ir_from_npz(npz_path, args.duration, SAMPLE_RATE)

        print(f"[{i}/{len(param_files)}] random_IR_{rid}  ({target_np.shape[0]} samples)")
        t0 = time.time()
        try:
            est6, best_loss, hist = fit_one_ir(
                target_np, starts, args, plate, box, loss_fn, device, dtype, loss_dtype
            )
        except RuntimeError as e:
            print(f"      FAILED: {e}")
            continue
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        elapsed = time.time() - t0

        raw7 = composite_to_raw7(est6)
        row: Dict[str, float] = {
            "id": rid,
            "filename": f"random_IR_{rid}.npz",
            "best_loss": best_loss,
            "nmse_6d": nmse_6d(est6, gt6),
            "mu_rel_error": abs(est6["mu"] - gt6["mu"]) / max(abs(gt6["mu"]), 1e-18),
            "runtime_s": elapsed,
            "n_restarts": int(args.n_restarts),
            "n_epochs": int(args.n_epochs),
            "loss_name": args.loss,
            "mu_mode": args.mu_mode,
            "raw7_feasible": raw7 is not None,
        }
        for k in COMPOSITE_KEYS:
            row[f"est_{k}"] = float(est6[k])
            row[f"gt_{k}"] = float(gt6[k])
        if raw7 is not None:
            for k, v in raw7.items():
                row[f"est_raw_{k}"] = float(v)

        rows.append(row)
        pd.DataFrame([row]).to_csv(out_dir / f"result_random_IR_{rid}.csv", index=False)
        print(
            f"      loss={best_loss:.6e}  NMSE_6d={row['nmse_6d']:.3e}  "
            f"mu_rel_err={row['mu_rel_error']:.3e}  {elapsed:.1f}s"
        )

        if not args.no_plots:
            _plot_per_ir(rid, hist, gt6, est6, elapsed, out_dir / f"random_IR_{rid}_diagnostic.png")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "summary.csv", index=False)
        df[["filename", "est_mu", "est_D_div_mu", "est_T0_div_mu", "est_Ly", "est_op_x", "est_op_y"]].rename(
            columns={
                "est_mu": "mu",
                "est_D_div_mu": "D_mu",
                "est_T0_div_mu": "T0_mu",
                "est_Ly": "Ly",
                "est_op_x": "op_x",
                "est_op_y": "op_y",
            }
        ).to_csv(out_dir / "submission.csv", index=False)
        if not args.no_plots:
            _plot_summary(rows, out_dir / "summary_diagnostic.png")
        print(
            f"\nMedian NMSE_6d = {df['nmse_6d'].median():.3e} | "
            f"median mu rel err = {df['mu_rel_error'].median():.3e} | "
            f"total {df['runtime_s'].sum():.0f}s over {len(df)} IRs"
        )
        if not df["raw7_feasible"].all():
            n = int((~df["raw7_feasible"]).sum())
            print(f"WARNING: {n} solution(s) lie outside the raw-7 feasible set")
    print(f"Done. Outputs written to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Batched multi-start gradient descent over the identifiable 6-parameter "
            "plate space, with mu profiled out analytically"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    repo = Path(__file__).resolve().parents[2]
    p.add_argument("--dataset-dir", type=Path, default=repo / "data" / "random-IR-100-1.0s")
    p.add_argument("--output-dir", type=Path, default=repo / "results" / "gd" / "graddescent")
    p.add_argument("--loss", type=str, default="L1_STFT", help="Loss name from src/loss/losses.py")
    p.add_argument("--num", type=int, default=1, help="Number of IRs to fit")
    p.add_argument("--duration", type=float, default=0.25, help="Seconds of IR to fit")

    p.add_argument(
        "--n-restarts", "--n-trials", dest="n_restarts", type=int, default=16,
        help="Latin-hypercube restarts per IR, optimized as one batch",
    )
    p.add_argument(
        "--restart-batch", type=int, default=16,
        help="Restarts held on the device at once; halved automatically on CUDA OOM",
    )
    p.add_argument("--seed", "--lhs-seed", dest="seed", type=int, default=42)
    p.add_argument(
        "--init-space", type=str, default="raw7", choices=["raw7", "composite"],
        help="raw7 keeps starts physically feasible and matches the CMA-ES prior",
    )

    p.add_argument("--n-epochs", type=int, default=400, help="Adam steps per restart")
    p.add_argument("--lr", type=float, default=0.02, help="Adam lr, in normalized [0,1] units")
    p.add_argument("--adam-beta1", dest="adam_beta1", type=float, default=0.9)
    p.add_argument("--grad-clip-value", type=float, default=1.0)
    p.add_argument("--grad-clip-type", type=str, default="norm", choices=["norm", "value"])
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    p.add_argument("--patience", type=int, default=60, help="Stop after this many non-improving steps")
    p.add_argument("--early-stop-loss", type=float, default=0.0, help="Stop once loss drops below this")

    p.add_argument(
        "--mu-mode", type=str, default="profile", choices=["profile", "fixed"],
        help="profile solves mu exactly by rescaling (no re-synthesis); fixed leaves it to ternary_mu",
    )
    p.add_argument("--mu-polish-every", type=int, default=10, help="Steps between mu solves")
    p.add_argument("--mu-ternary-iters", type=int, default=24, help="Ternary iterations per mu solve")

    p.add_argument(
        "--normalize-target", action="store_true",
        help="Peak-normalize the target. Discards the amplitude that identifies mu.",
    )
    p.add_argument("--normalize-pred", action="store_true", help="Peak-normalize the prediction")
    p.add_argument("--vel-calc", action="store_true", help="Use the velocity output branch")

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument(
        "--loss-dtype", type=str, default="float32", choices=["float32", "float64"],
        help="Loss precision; MSS allocates a float32 accumulator and fails in float64",
    )
    p.add_argument("--double", action="store_true", help="Convenience flag to force float64 synthesis")
    p.add_argument("--chunk-elems", type=int, default=8_000_000, help="Plate modal-sum chunk budget")
    p.add_argument("--no-grad-checkpoint", action="store_true", help="Disable plate gradient checkpointing")
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.double:
        args.dtype = "float64"
    args.mu_polish_every = max(1, int(args.mu_polish_every))
    run(args)


if __name__ == "__main__":
    main()
