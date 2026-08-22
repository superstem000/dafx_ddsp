"""Gradient descent on the same terrain the CMA-ES sweep searches.

The default search space is identical to src.cmaes.fit_7param_norm_es: the seven
raw parameters {E, rho, h, Ly, T0, op_x, op_y}, mapped linearly from [-1,1] onto
PARAM_BOUNDS, synthesized through SevenParamPlate.  The bounds, the linear map,
the [B,14] plate packing and the Latin-hypercube starts are all imported from
that module rather than restated here, and the differentiable copy of the plate
packing is checked against it at startup, so the two cannot silently diverge.

Exactly one thing is deliberately different: peak normalization is off.

run_cmaes peak-normalizes both signals -- load_target_ir_from_npz divides the
target by its max, and synth() is called without normalize=, which
BatchedModalPlateTorch.forward defaults to True.  That discards the absolute
amplitude, and since mu enters the synthesis only as an overall 1/mu scale, mu
becomes unidentifiable and has to be recovered afterwards by
src.mu_optimization.ternary_mu.  Fitting the un-normalized IR instead lets mu
fall out of the same run, so there is no second stage.  Pass --normalize to put
that difference back and reproduce the CMA-ES objective exactly.

An earlier version of this file searched an "identifiable" 5-coordinate
composite space with log scaling and mu solved separately.  Those were three
unforced deviations from a terrain already known to work -- CMA-ES recovers many
of these IRs within a single restart -- and they are kept only as the
--space composite6 ablation, not as the default.

Note on the raw-7 space: it carries an exact symmetry,
(E, rho, h) -> (c^3 E, c rho, h/c), which leaves the IR unchanged.  Gradients
along it are zero, so Adam random-walks that direction.  It is harmless for
scoring -- every one of the six submitted composites is invariant under it -- and
CMA-ES searches the same space regardless.
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

from src.cmaes.fit_7param_norm_es import (
    BOUNDS_HI_NP,
    BOUNDS_LO_NP,
    FIXED_PLATE_PARAMS,
    IS_LOG_NP,
    PARAM_KEYS,
    PRODUCT_PLATE_PARAMS,
    generate_lhs_starts_norm,
    norm_to_physical,
    physical_to_norm,
    physical_to_plate14_tensor,
)
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
from src.plate.SevenParamPlate import BatchedModalPlateTorch as SevenParamPlate
from src.plate.SixParamPlate import BatchedModalPlateTorch as SixParamPlate

SAMPLE_RATE = 44100

# Composite-space ablation only; the default raw-7 space uses PARAM_BOUNDS.
SHAPE_KEYS: Sequence[str] = ("D_div_mu", "T0_div_mu", "Ly", "op_x", "op_y")
LOG_KEYS = {"mu", "D_div_mu", "T0_div_mu"}


def _is_oom(exc: BaseException) -> bool:
    typed = getattr(torch.cuda, "OutOfMemoryError", None)
    return (typed is not None and isinstance(exc, typed)) or "out of memory" in str(exc).lower()


def _read_params_csv(csv_path: Path) -> Dict[str, float]:
    with csv_path.open("r", newline="") as f:
        row = next(csv.DictReader(f))
    return {k: float(v) for k, v in row.items()}


def nmse_7d(est7: Optional[Dict[str, float]], gt7: Dict[str, float]) -> float:
    """Mean squared error over the raw seven, normalized by PARAM_BOUNDS.

    Same convention as compute_nmse_6d, but in raw space. The two differ in a
    way worth reporting: every one of the six composites is invariant under
    (E, rho, h) -> (c^3 E, c rho, h/c), so a solution displaced along that
    symmetry scores zero in 6d while 7d sees it. Reporting both says whether a
    fit landed on the true parameters or merely on the manifold through them.
    """
    if est7 is None:
        return float("nan")
    errs = []
    for i, k in enumerate(PARAM_KEYS):
        if IS_LOG_NP[i]:
            # Normalized by the same coordinate z uses. Dividing a decades-wide
            # parameter's error by its linear span would score a factor-of-two
            # miss at the bottom of the range as ~0, which is precisely the part
            # of the range this space exists to make resolvable.
            span = np.log(BOUNDS_HI_NP[i] / BOUNDS_LO_NP[i])
            e = np.log(max(est7[k], 1e-300) / max(gt7[k], 1e-300)) / span
        else:
            e = (est7[k] - gt7[k]) / (BOUNDS_HI_NP[i] - BOUNDS_LO_NP[i])
        errs.append(e ** 2)
    return float(np.mean(errs))


# --------------------------------------------------------------------------
# Raw-7 space: byte-for-byte the CMA-ES terrain, made differentiable.
# --------------------------------------------------------------------------


def norm_to_physical_torch(z: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Differentiable copy of fit_7param_norm_es.norm_to_physical ([-1,1] -> bounds).

    Log-spaced for LOG_PARAMS: a parameter spanning decades gets a z whose steps
    are constant in ratio, so the bottom decade is not compressed into the last
    1% of the coordinate the encoder has to emit.
    """
    u = (z + 1.0) / 2.0
    lin = lo + u * (hi - lo)
    if not IS_LOG_NP.any():
        return lin
    is_log = torch.as_tensor(IS_LOG_NP, device=z.device)
    log = lo * torch.pow(hi / lo, u)
    return torch.where(is_log, log, lin)


def physical_to_plate14_torch(phys: torch.Tensor) -> torch.Tensor:
    """Differentiable copy of fit_7param_norm_es.physical_to_plate14_tensor."""
    named = {k: phys[:, i] for i, k in enumerate(PARAM_KEYS)}
    for out, (a, b) in PRODUCT_PLATE_PARAMS.items():
        named[out] = named[a] * named[b]
    ones = torch.ones_like(phys[:, 0])
    return torch.stack(
        [named[k] if k in named else ones * FIXED_PLATE_PARAMS[k]
         for k in SevenParamPlate.PARAM_ORDER],
        dim=1,
    )


def verify_mapping_matches_cmaes(device: torch.device) -> None:
    """Fail loudly if the differentiable packing drifts from the CMA-ES one."""
    rng = np.random.default_rng(0)
    z = rng.uniform(-1.0, 1.0, size=(8, len(PARAM_KEYS)))
    # float32 on both sides before packing. A space with a derived column
    # (T60_F1 = ratio * T60_DC) computes a product here, and a float64 product
    # rounded afterwards is not always the float32 product -- comparing those
    # would fail this check by one ulp on a mapping that is in fact identical.
    # Both real paths pack from a float32 z, so float32 is also the honest
    # comparison. For a space with no derived column every column is a copy and
    # this changes nothing.
    phys = norm_to_physical(z).astype(np.float32)
    expect = physical_to_plate14_tensor(phys, device)
    got = physical_to_plate14_torch(torch.as_tensor(phys, device=device, dtype=torch.float32))
    err = float((expect - got).abs().max())
    if err > 0.0:
        raise RuntimeError(f"plate14 packing diverges from fit_7param_norm_es (max abs diff {err:g})")


class Raw7Space:
    """Seven raw parameters, linear from [-1,1]: the CMA-ES search space."""

    name = "raw7"
    keys = list(PARAM_KEYS)
    lo, hi = -1.0, 1.0
    profiles_mu = False

    def __init__(self, device, dtype, normalize: bool):
        self.device, self.dtype, self.normalize = device, dtype, normalize
        self.plate = SevenParamPlate(
            sample_rate=SAMPLE_RATE, device=device, dtype=dtype, drop_sub_20hz_modes=False
        )
        self._lo = torch.as_tensor(BOUNDS_LO_NP, device=device, dtype=dtype)
        self._hi = torch.as_tensor(BOUNDS_HI_NP, device=device, dtype=dtype)

    def configure_plate(
        self, chunk_elems: int, grad_checkpoint: bool, batched: bool = False,
        compile_modal_sum: bool = False, mode_bucket: int = 1024,
        fixed_mode_grid=None,
    ) -> None:
        self.plate.chunk_elems = chunk_elems
        self.plate.grad_checkpoint = grad_checkpoint
        self.plate.batched_modal_sum = batched
        self.plate.compile_modal_sum = compile_modal_sum
        self.plate.mode_bucket = mode_bucket
        # Held constant so synthesis stops depending on the batch it happened
        # in; see SevenParamPlate.fixed_mode_grid.
        self.plate.fixed_mode_grid = (
            None if fixed_mode_grid is None
            else (int(fixed_mode_grid[0]), int(fixed_mode_grid[1]))
        )

    def lhs(self, n_starts: int, seed: int) -> np.ndarray:
        # The very generator CMA-ES uses for its restart starts.
        return generate_lhs_starts_norm(n_starts, seed=seed)

    def forward(self, z: torch.Tensor, mu: Optional[torch.Tensor], duration: float) -> torch.Tensor:
        del mu  # mu is rho*h, determined by the parameters themselves
        phys = norm_to_physical_torch(z, self._lo, self._hi)
        return self.plate(physical_to_plate14_torch(phys), duration=duration, normalize=self.normalize)

    def to_six(self, z_row: np.ndarray) -> Dict[str, float]:
        return seven_to_six(self.to_raw7(z_row))

    def to_raw7(self, z_row: np.ndarray) -> Dict[str, float]:
        phys = norm_to_physical(np.asarray(z_row, dtype=np.float64)[None, :])[0]
        return {k: float(v) for k, v in zip(PARAM_KEYS, phys)}

    def gt_z(self, gt7: Dict[str, float]) -> np.ndarray:
        phys = np.array([gt7[k] for k in PARAM_KEYS], dtype=np.float64)
        return physical_to_norm(phys[None, :])[0]


# --------------------------------------------------------------------------
# Composite-6 space: retained as an ablation, not the default.
# --------------------------------------------------------------------------


class NormBox:
    """Maps z in [0,1]^P to physical values, log-scaled for keys spanning decades."""

    def __init__(self, keys: Sequence[str], device: torch.device, dtype: torch.dtype):
        self.keys = list(keys)
        lo = np.array([COMPOSITE_BOUNDS[k][0] for k in self.keys], dtype=np.float64)
        hi = np.array([COMPOSITE_BOUNDS[k][1] for k in self.keys], dtype=np.float64)
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
        v = np.asarray(v, dtype=np.float64)
        lin = (v - self.lo_np) / np.maximum(self.hi_np - self.lo_np, 1e-300)
        log = (np.log(np.maximum(v, 1e-300)) - np.log(self.lo_np)) / (
            np.log(self.hi_np) - np.log(self.lo_np)
        )
        return np.clip(np.where(self.is_log_np, log, lin), 0.0, 1.0)


class Composite6Space:
    """Five shape coordinates with mu profiled out exactly (ablation)."""

    name = "composite6"
    keys = list(SHAPE_KEYS)
    lo, hi = 0.0, 1.0

    def __init__(self, device, dtype, normalize: bool):
        self.device, self.dtype, self.normalize = device, dtype, normalize
        self.plate = SixParamPlate(
            sample_rate=SAMPLE_RATE, device=device, dtype=dtype, drop_sub_20hz_modes=False
        )
        self.box = NormBox(SHAPE_KEYS, device=device, dtype=dtype)
        self.mu_ref = float(np.sqrt(MU_MIN * MU_MAX))
        # Peak normalization erases the amplitude mu is carried by.
        self.profiles_mu = not normalize

    def configure_plate(
        self, chunk_elems: int, grad_checkpoint: bool, batched: bool = False,
        compile_modal_sum: bool = False, mode_bucket: int = 1024,
    ) -> None:
        self.plate.chunk_elems = chunk_elems
        self.plate.grad_checkpoint = grad_checkpoint
        self.plate.batched_modal_sum = batched
        self.plate.compile_modal_sum = compile_modal_sum
        self.plate.mode_bucket = mode_bucket

    def lhs(self, n_starts: int, seed: int) -> np.ndarray:
        # Draw in raw-7 and map through, so starts stay physically realizable.
        raw = generate_lhs_starts_norm(n_starts, seed=seed)
        phys = BOUNDS_LO_NP + ((raw + 1.0) / 2.0) * (BOUNDS_HI_NP - BOUNDS_LO_NP)
        out = []
        for row in phys:
            six = seven_to_six({k: float(v) for k, v in zip(PARAM_KEYS, row)})
            out.append(self.box.to_unit_np(np.array([six[k] for k in SHAPE_KEYS])))
        return np.asarray(out, dtype=np.float64)

    def forward(self, z: torch.Tensor, mu: Optional[torch.Tensor], duration: float) -> torch.Tensor:
        six = torch.cat([mu.unsqueeze(1), self.box.to_physical(z)], dim=1)
        return self.plate(six, duration=duration, normalize=self.normalize)

    def to_six(self, z_row: np.ndarray) -> Dict[str, float]:
        zt = torch.as_tensor(z_row, device=self.device, dtype=self.dtype).unsqueeze(0)
        vals = self.box.to_physical(zt)[0].detach().cpu().numpy()
        return {k: float(v) for k, v in zip(SHAPE_KEYS, vals)}

    def to_raw7(self, z_row: np.ndarray) -> Optional[Dict[str, float]]:
        return None

    def gt_z(self, gt7: Dict[str, float]) -> np.ndarray:
        six = seven_to_six(gt7)
        return self.box.to_unit_np(np.array([six[k] for k in SHAPE_KEYS]))


def solve_mu_by_scale(pred_ref, mu_ref, target, loss_fn, n_iters, loss_dtype):
    """Exact per-row mu solve by ternary search over an output rescaling.

    mu enters synthesis only through ms = 0.25*mu*Lx*Ly, so
    y(mu) = y(mu_ref)*(mu_ref/mu) exactly and no re-synthesis is needed.
    """
    K = pred_ref.shape[0]
    tgt = target.to(loss_dtype).expand(K, -1)
    ref = pred_ref.to(loss_dtype)
    log_ref = torch.log(mu_ref.to(loss_dtype))
    lo = log_ref - math.log(MU_MAX)
    hi = log_ref - math.log(MU_MIN)

    def loss_at(log_s):
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
        best = loss_at(log_s)

    mu = torch.clamp(mu_ref / torch.exp(log_s).to(mu_ref.dtype), MU_MIN, MU_MAX)
    return mu, best.to(torch.float64)


def _fit_batch(z0, target, duration, args, space, loss_fn, device, dtype, loss_dtype, n_epochs):
    """Batched multi-start Adam over one wave of starts.

    Returns (best_z, best_mu, best_loss, history, epochs_run); epochs_run is
    what the caller charges against the compute budget when a wave stops early.
    """
    K = z0.shape[0]
    z = nn.Parameter(torch.as_tensor(z0, device=device, dtype=dtype))
    # eps must sit well below the gradient magnitude; un-normalized plate IRs
    # peak near 1e-8, which is exactly torch's default eps.
    optim = torch.optim.Adam([z], lr=args.lr, betas=(args.adam_beta1, 0.999), eps=args.adam_eps)

    if args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)
    elif args.lr_schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=args.patience, factor=0.5)
    else:
        sched = None

    mu = torch.full((K,), float(np.sqrt(MU_MIN * MU_MAX)), device=device, dtype=dtype)
    best_loss = torch.full((K,), float("inf"), device=device, dtype=torch.float64)
    best_z = z.detach().clone()
    best_mu = mu.clone()
    hist: List[List[float]] = [[] for _ in range(K)]
    loss_scale: Optional[torch.Tensor] = None
    stale = 0

    epochs_run = 0
    for epoch in range(n_epochs):
        epochs_run = epoch + 1
        if space.profiles_mu and (epoch % args.mu_polish_every == 0):
            with torch.no_grad():
                ref = space.forward(z.detach(), mu, duration)
                mu, _ = solve_mu_by_scale(ref, mu, target, loss_fn, args.mu_ternary_iters, loss_dtype)

        optim.zero_grad(set_to_none=True)
        pred = space.forward(z, mu, duration)
        loss = loss_fn(target.to(loss_dtype).expand(K, -1), pred.to(loss_dtype))

        finite = torch.isfinite(loss)
        objective = torch.where(finite, loss, torch.zeros_like(loss))
        if args.loss_scale == "auto":
            if loss_scale is None:
                loss_scale = (
                    objective[finite].median().detach().abs().clamp(min=1e-30)
                    if bool(finite.any())
                    else torch.ones((), device=objective.device, dtype=objective.dtype)
                )
                print(f"      loss scale (fixed for this batch): {float(loss_scale):.4e}")
            objective = objective / loss_scale
        objective.sum().backward()

        if z.grad is None:
            print(f"      iter {epoch}: no finite gradients, stopping this batch")
            break

        with torch.no_grad():
            g = z.grad
            g[~torch.isfinite(g).all(dim=1) | ~finite] = 0.0
            if args.grad_clip_type == "norm":
                gn = g.norm(dim=1, keepdim=True)
                g.mul_((args.grad_clip_value / (gn + 1e-12)).clamp(max=1.0))
            else:
                g.clamp_(-args.grad_clip_value, args.grad_clip_value)

        z_eval = z.detach().clone()  # the point that produced `loss`
        optim.step()
        with torch.no_grad():
            z.clamp_(space.lo, space.hi)

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

        if epoch % max(1, n_epochs // 10) == 0:
            print(f"      iter {epoch:4d}/{n_epochs}  best={float(best_loss.min()):.6e}")

        stale = 0 if bool(improved.any()) else stale + 1
        if float(best_loss.min()) <= args.early_stop_loss:
            print(f"      early stop at iter {epoch}")
            break
        if args.patience > 0 and stale >= args.patience and args.lr_schedule != "plateau":
            print(f"      early stop at iter {epoch}: {stale} iters without improvement")
            break

    if space.profiles_mu:
        with torch.no_grad():
            ref = space.forward(best_z, best_mu, duration)
            best_mu, best_loss = solve_mu_by_scale(
                ref, best_mu, target, loss_fn, args.mu_ternary_iters, loss_dtype
            )

    return (
        best_z.detach().cpu().numpy(),
        best_mu.detach().cpu().numpy(),
        best_loss.detach().cpu().numpy(),
        hist,
        epochs_run,
    )


def fit_one_ir(target_np, args, space, loss_fn, device, dtype, loss_dtype):
    """Fit one IR under a fixed compute budget, in waves of fresh starts.

    Budget is counted in row-steps (starts x Adam iterations), the same unit as
    the CMA-ES --budget of function evaluations, so the two are comparable. A
    wave that stops early hands its unspent budget to the next wave rather than
    ending the fit, which is the point of budgeting rather than fixing the
    iteration count: a stalled set of starts becomes fresh starts instead of a
    truncated run.
    """
    duration = target_np.shape[0] / float(SAMPLE_RATE)
    target = torch.as_tensor(target_np, device=device, dtype=dtype).unsqueeze(0)
    if args.normalize:
        target = target / torch.clamp(target.abs().max(), min=1e-12)

    budget = args.budget if args.budget > 0 else args.n_restarts * args.n_epochs
    batch = min(args.restart_batch, args.n_restarts)
    spent, wave = 0, 0
    best = None  # (loss, z_row, mu_row)
    all_hist: List[List[float]] = []
    # Every start's endpoint, not just the winner's. The winner alone cannot say
    # whether reaching the loss floor is what recovers the parameters; sixteen
    # (final loss, final parameters) pairs can.
    all_finals: List[Tuple[float, np.ndarray, float]] = []

    while spent < budget and wave < args.max_waves:
        epochs = min(args.n_epochs, (budget - spent) // max(1, batch))
        if epochs < args.min_wave_epochs:
            break
        # A distinct seed per wave, so a wave that stalls is followed by
        # genuinely different starts rather than the same ones again.
        starts = space.lhs(batch, args.seed + 1000 * wave)
        try:
            z_b, mu_b, loss_b, hist_b, ran = _fit_batch(
                starts, target, duration, args, space, loss_fn, device, dtype, loss_dtype, epochs
            )
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if batch == 1:
                print("      [OOM] cannot fit a single start; stopping")
                break
            batch = max(1, batch // 2)
            print(f"      [OOM] retrying with restart_batch={batch}")
            continue

        spent += batch * ran
        all_hist.extend(hist_b)
        for k in range(len(loss_b)):
            all_finals.append((float(loss_b[k]), z_b[k].copy(), float(mu_b[k])))
        w = int(np.argmin(loss_b))
        if best is None or loss_b[w] < best[0]:
            best = (float(loss_b[w]), z_b[w].copy(), float(mu_b[w]))
        wave += 1
        print(
            f"      wave {wave}: {batch} starts x {ran} iters, "
            f"best so far {best[0]:.6e}, budget {spent}/{budget}"
        )

    if best is None:
        raise RuntimeError("no wave completed")

    est6 = space.to_six(best[1])
    if space.profiles_mu:
        est6["mu"] = best[2]
    return est6, space.to_raw7(best[1]), best[0], all_hist, all_finals, spent, wave


def loss_at_ground_truth(gt7, gt6, target, duration, space, loss_fn, device, dtype, loss_dtype):
    """Loss evaluated at the true parameters.

    The single most informative number to sit beside the achieved loss: if the
    fitter's loss is far above this, the optimizer failed to reach an optimum
    the loss does define. If it is at or below it, the loss prefers some other
    point and no optimizer would have recovered the true parameters.
    """
    z = torch.as_tensor(space.gt_z(gt7), device=device, dtype=dtype).unsqueeze(0)
    mu = torch.full((1,), float(gt6["mu"]), device=device, dtype=dtype)
    with torch.no_grad():
        pred = space.forward(z, mu, duration)
        lv = loss_fn(target.to(loss_dtype).expand(1, -1), pred.to(loss_dtype))
    return float(lv[0])


def _plot_per_ir(rid, hist, gt6, est6, elapsed, out_path, gt_loss=None, per_restart=None):
    """Per-restart convergence, the parameter fit, and how the two relate.

    The third panel is the one that carries an argument rather than a summary.
    A restart's final loss is observable without ground truth; its final
    NMSE_6d is not. Plotting them against each other over every start says
    whether driving the loss to its floor is what recovers the parameters, or
    whether the loss bottoms out somewhere the parameters do not follow -- and
    a single best-restart number cannot distinguish those.
    """
    ncol = 3 if per_restart else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6.5 * ncol, 5))
    ax = axes[0]
    for h in hist:
        ax.plot(np.arange(len(h)), h, linewidth=0.6, alpha=0.35)
    if gt_loss is not None and gt_loss > 0:
        # Without this line the reader cannot tell a converged run from a stalled
        # one: both are just curves that flatten. This is the floor they are
        # flattening against.
        ax.axhline(gt_loss, color="k", ls="--", lw=1.0,
                   label=f"loss at true params ({gt_loss:.2e})")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"Convergence ({len(hist)} restarts)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    keys = list(COMPOSITE_KEYS)
    x = np.arange(len(keys), dtype=np.float64)

    def unit(d):
        out = []
        for k in keys:
            lo, hi = COMPOSITE_BOUNDS[k]
            out.append((min(max(float(d[k]), lo), hi) - lo) / (hi - lo))
        return np.asarray(out)

    ax.bar(x - 0.175, unit(est6), width=0.35, label="Est", color="steelblue")
    ax.bar(x + 0.175, unit(gt6), width=0.35, label="GT", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"NMSE_6d={nmse_6d(est6, gt6):.3e}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    if per_restart:
        ax = axes[2]
        fl = np.array([max(p[0], 1e-300) for p in per_restart])
        fn = np.array([max(p[1], 1e-300) for p in per_restart])
        ax.scatter(fl, fn, s=46, color="steelblue", edgecolor="k",
                   linewidth=0.5, zorder=3)
        if gt_loss is not None and gt_loss > 0:
            ax.axvline(gt_loss, color="k", ls="--", lw=1.0, label="loss at true params")
            ax.legend(fontsize=8, loc="lower right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("final loss")
        ax.set_ylabel("final NMSE_6d")
        ax.set_title(f"{len(per_restart)} restarts: loss reached vs parameters recovered")
        ax.grid(True, alpha=0.3, which="both")

    plt.suptitle(f"random_IR_{rid} ({elapsed:.0f}s)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def run(args) -> None:
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    req = str(args.device).strip().lower()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if req == "auto"
        else torch.device(req)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    loss_dtype = torch.float64 if args.loss_dtype == "float64" else torch.float32

    verify_mapping_matches_cmaes(device)

    space_cls = Raw7Space if args.space == "raw7" else Composite6Space
    space = space_cls(device, dtype, args.normalize)
    space.configure_plate(
        args.chunk_elems, not args.no_grad_checkpoint, args.batched_plate,
        args.compile_plate, args.mode_bucket,
    )

    print(f"Device: {device}   dtype: {dtype}   loss dtype: {loss_dtype}")
    print(f"Loss: {args.loss}   space: {space.name}   coords: {space.keys}")
    print(f"normalize: {args.normalize}   mu profiled: {space.profiles_mu}")
    print(f"restarts: {args.n_restarts} (batch {args.restart_batch})   seed: {args.seed}")
    if args.normalize:
        print("NOTE: --normalize reproduces the CMA-ES objective; mu is then")
        print("      unidentifiable and must be recovered with ternary_mu.")
    if device.type == "cuda":
        print(f"CUDA: {torch.cuda.get_device_name(0 if device.index is None else device.index)}")

    loss_fn = select_loss_function(args.loss, sample_rate=SAMPLE_RATE, device=device)

    param_files = sorted(dataset_dir.glob("random_IR_params_*.csv"))
    if not param_files:
        raise FileNotFoundError(f"No random_IR_params_*.csv found in {dataset_dir}")
    if args.num:
        param_files = param_files[: args.num]

    rows: List[Dict] = []
    print(f"Processing {len(param_files)} IR(s) from {dataset_dir}")

    for i, params_csv in enumerate(param_files, start=1):
        rid = params_csv.stem.split("_")[-1]
        npz_path = dataset_dir / f"random_IR_{rid}.npz"
        if not npz_path.exists():
            print(f"[{i}/{len(param_files)}] missing {npz_path.name}, skipping")
            continue

        gt7 = _read_params_csv(params_csv)
        gt6 = seven_to_six(gt7)
        target_np = load_target_ir_from_npz(npz_path, args.duration, SAMPLE_RATE)

        print(f"[{i}/{len(param_files)}] random_IR_{rid}  ({target_np.shape[0]} samples)")
        t0 = time.time()
        try:
            est6, est7, best_loss, hist, finals, spent, waves = fit_one_ir(
                target_np, args, space, loss_fn, device, dtype, loss_dtype
            )
        except RuntimeError as e:
            print(f"      FAILED: {e}")
            continue
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        elapsed = time.time() - t0

        target_t = torch.as_tensor(target_np, device=device, dtype=dtype).unsqueeze(0)
        if args.normalize:
            target_t = target_t / torch.clamp(target_t.abs().max(), min=1e-12)
        gt_loss = loss_at_ground_truth(
            gt7, gt6, target_t, target_np.shape[0] / float(SAMPLE_RATE),
            space, loss_fn, device, dtype, loss_dtype,
        )

        row = {
            "id": rid,
            "filename": f"random_IR_{rid}.npz",
            "best_loss": best_loss,
            "gt_loss": gt_loss,
            "loss_ratio": best_loss / max(gt_loss, 1e-300),
            "nmse_6d": nmse_6d(est6, gt6),
            "nmse_7d": nmse_7d(est7, gt7),
            "evals_spent": int(spent),
            "waves": int(waves),
            "mu_rel_error": abs(est6["mu"] - gt6["mu"]) / max(abs(gt6["mu"]), 1e-18),
            "runtime_s": elapsed,
            "space": space.name,
            "normalize": args.normalize,
            "n_restarts": int(args.n_restarts),
            "n_epochs": int(args.n_epochs),
            "loss_name": args.loss,
        }
        for k in COMPOSITE_KEYS:
            row[f"est_{k}"] = float(est6[k])
            row[f"gt_{k}"] = float(gt6[k])
        if est7:
            for k, v in est7.items():
                row[f"est_raw_{k}"] = float(v)
                row[f"gt_raw_{k}"] = float(gt7[k])

        rows.append(row)
        pd.DataFrame([row]).to_csv(out_dir / f"result_random_IR_{rid}.csv", index=False)
        # Only call it a loss problem when the fit is genuinely elsewhere: at the
        # bottom of the well the two differ by float32 noise, and reading that as
        # "the loss prefers another point" gets the successful case exactly wrong.
        ratio = row["loss_ratio"]
        if row["nmse_6d"] < 1e-8:
            verdict = "converged to GT"
        elif 0.1 < ratio <= 1.0:
            verdict = "at GT loss level, parameters differ"
        elif ratio <= 0.1:
            verdict = "loss prefers elsewhere -- not an optimizer problem"
        else:
            verdict = f"optimizer short of GT by {ratio:.1f}x"
        print(
            f"      loss={best_loss:.6e}  gt_loss={gt_loss:.6e}  ({verdict})\n"
            f"      NMSE_6d={row['nmse_6d']:.3e}  NMSE_7d={row['nmse_7d']:.3e}  "
            f"mu_rel_err={row['mu_rel_error']:.3e}  {elapsed:.1f}s"
        )
        # Score every start, not just the winner, and keep the trace on disk.
        # Without this the loss curves live only in memory, so any change to the
        # figure costs a full refit -- 379s for one IR at 16x3000.
        per_restart = []
        for f_loss, f_z, f_mu in finals:
            e6 = space.to_six(f_z)
            if space.profiles_mu:
                e6["mu"] = f_mu
            per_restart.append((f_loss, nmse_6d(e6, gt6)))
        if hist:
            width = max(len(h) for h in hist)
            curves = np.full((len(hist), width), np.nan)
            for j, h in enumerate(hist):
                curves[j, : len(h)] = h
            np.savez_compressed(
                out_dir / f"trace_random_IR_{rid}.npz",
                curves=curves,
                final_loss=np.array([p[0] for p in per_restart]),
                final_nmse_6d=np.array([p[1] for p in per_restart]),
                gt_loss=np.float64(gt_loss),
            )
        if not args.no_plots:
            _plot_per_ir(rid, hist, gt6, est6, elapsed,
                         out_dir / f"random_IR_{rid}_diagnostic.png",
                         gt_loss=gt_loss, per_restart=per_restart)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "summary.csv", index=False)
        df[["filename"] + [f"est_{k}" for k in COMPOSITE_KEYS]].rename(
            columns={f"est_{k}": k for k in COMPOSITE_KEYS}
        ).to_csv(out_dir / "submission.csv", index=False)
        print(
            f"\nMedian NMSE_6d = {df['nmse_6d'].median():.3e} | "
            f"NMSE_7d = {df['nmse_7d'].median():.3e} | "
            f"median mu rel err = {df['mu_rel_error'].median():.3e} | "
            f"{df['runtime_s'].sum():.0f}s over {len(df)} IRs"
        )
    print(f"Done. Outputs written to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gradient descent on the CMA-ES search terrain, without peak normalization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    repo = Path(__file__).resolve().parents[2]
    p.add_argument("--dataset-dir", type=Path, default=repo / "random-IR-200-0.2s")
    p.add_argument("--output-dir", type=Path, default=repo / "results" / "gd" / "graddescent")
    p.add_argument("--loss", type=str, default="L1_STFT")
    p.add_argument("--num", type=int, default=1, help="Number of IRs to fit")
    p.add_argument("--duration", type=float, default=0.25)

    p.add_argument(
        "--space", type=str, default="raw7", choices=["raw7", "composite6"],
        help="raw7 is the CMA-ES search space; composite6 is the ablation",
    )
    p.add_argument(
        "--normalize", action="store_true",
        help="Peak-normalize both signals as run_cmaes does; makes mu unidentifiable",
    )

    p.add_argument("--n-restarts", "--n-trials", dest="n_restarts", type=int, default=16)
    p.add_argument("--restart-batch", type=int, default=16)
    p.add_argument("--seed", "--lhs-seed", dest="seed", type=int, default=42)

    p.add_argument("--n-epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.02, help="Adam lr in normalized units")
    p.add_argument("--adam-beta1", dest="adam_beta1", type=float, default=0.9)
    p.add_argument("--adam-eps", type=float, default=1e-16)
    p.add_argument("--grad-clip-value", type=float, default=1.0)
    p.add_argument("--grad-clip-type", type=str, default="norm", choices=["norm", "value"])
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    p.add_argument(
        "--patience", type=int, default=0,
        help="Stop a wave after this many non-improving steps; 0 disables. "
             "Was 60, which killed runs that later recovered.",
    )
    p.add_argument(
        "--budget", type=int, default=0,
        help="Total row-steps (starts x iterations), comparable to the CMA-ES "
             "--budget of evaluations. 0 means n_restarts * n_epochs, one wave.",
    )
    p.add_argument("--max-waves", type=int, default=64, help="Cap on waves of fresh starts")
    p.add_argument("--min-wave-epochs", type=int, default=100, help="Do not start a wave shorter than this")
    p.add_argument("--early-stop-loss", type=float, default=0.0)
    p.add_argument("--loss-scale", type=str, default="auto", choices=["auto", "none"])

    p.add_argument("--mu-polish-every", type=int, default=10)
    p.add_argument("--mu-ternary-iters", type=int, default=24)

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--loss-dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--chunk-elems", type=int, default=8_000_000)
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument(
        "--batched-plate", action="store_true",
        help="Sum modes for the whole batch at once instead of looping over it",
    )
    p.add_argument(
        "--compile-plate", action="store_true",
        help="torch.compile the modal sum so the elementwise chain fuses into the reduction",
    )
    p.add_argument(
        "--mode-bucket", type=int, default=1024,
        help="Pad the mode axis to a multiple of this so compiled shapes stay static",
    )
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.mu_polish_every = max(1, int(args.mu_polish_every))
    run(args)


if __name__ == "__main__":
    main()
