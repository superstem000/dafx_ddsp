"""Compute gradient of L1_STFT loss w.r.t. the 7 plate parameters.

Primary path is autograd through SevenParamPlate + the loss function.
graddescent.py confirms this works end-to-end.

Caveats:
- The Python-int mode-grid sizing (max_DDx, max_DDy) and the boolean valid_mask
  in SevenParamPlate cause "mode-appearing/disappearing" boundary effects to be
  invisible to autograd. At any non-boundary point the gradient is exact for
  the smooth-within-active-mode-set function. For finite-difference reference
  points this matters in extreme corners only.

Run as:
    python -m src.diagnostics.loss_gradient
to execute the self-test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.loss.loss_selector import select_loss_function
from src.plate.SevenParamPlate import BatchedModalPlateTorch

SAMPLE_RATE = 44100

PARAM_KEYS: Sequence[str] = ("E", "rho", "h", "Ly", "T0", "op_x", "op_y")
PARAM_BOUNDS = {
    "E": (6.7e10, 2.2e11),
    "rho": (2430.0, 21230.0),
    "h": (0.001, 0.005),
    "Ly": (1.1, 4.0),
    "T0": (0.01, 1000.0),
    "op_x": (0.51, 1.0),
    "op_y": (0.51, 1.0),
}
BOUNDS_LO = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS], dtype=np.float64)
BOUNDS_HI = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS], dtype=np.float64)
BOUNDS_RANGE = BOUNDS_HI - BOUNDS_LO

FIXED = {
    "Lx": 1.0, "nu": 0.25,
    "T60_DC": 6.0, "T60_F1": 2.0, "loss_F1": 500.0,
    "fp_x": 0.335, "fp_y": 0.467,
}

_PLATE: BatchedModalPlateTorch | None = None
_LOSS_FN_CACHE: dict = {}


def _plate(device: str, dtype: torch.dtype) -> BatchedModalPlateTorch:
    global _PLATE
    if _PLATE is None or _PLATE.device != torch.device(device) or _PLATE.dtype != dtype:
        _PLATE = BatchedModalPlateTorch(device=device, dtype=dtype).to(device)
    return _PLATE


def _loss(name: str, device: str):
    key = (name, device)
    if key not in _LOSS_FN_CACHE:
        _LOSS_FN_CACHE[key] = select_loss_function(name, sample_rate=SAMPLE_RATE, device=device)
    return _LOSS_FN_CACHE[key]


def _build_plate14_tensor(phys7: torch.Tensor) -> torch.Tensor:
    """Map [B, 7] (E, rho, h, Ly, T0, op_x, op_y) to SevenParamPlate's [B, 14],
    keeping the autograd graph intact."""
    B = phys7.shape[0]
    dtype, device = phys7.dtype, phys7.device
    E, rho, h, Ly, T0, op_x, op_y = [phys7[:, i] for i in range(7)]
    const = lambda v: torch.full((B,), float(v), dtype=dtype, device=device)
    return torch.stack(
        [
            const(FIXED["Lx"]), Ly, h, T0, rho, E,
            const(FIXED["nu"]),
            const(FIXED["T60_DC"]), const(FIXED["T60_F1"]), const(FIXED["loss_F1"]),
            const(FIXED["fp_x"]), const(FIXED["fp_y"]),
            op_x, op_y,
        ],
        dim=1,
    )


def compute_gradient(
    params_norm: np.ndarray,
    target_ir: torch.Tensor,
    loss_name: str = "L1_STFT",
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    duration: float | None = None,
    use_autograd: bool = True,
    fd_eps: float = 1e-2,
) -> np.ndarray:
    """Return ∇L(x) at x = params_norm (in [0, 1]^7 space).

    Autograd path differentiates through SevenParamPlate.forward(...,
    normalize=False) and the configured loss function. Falls back to forward
    finite differences (eps in normalized units) if `use_autograd=False`.

    Returns a length-7 numpy array in PARAM_KEYS order.
    """
    if params_norm.shape != (7,):
        raise ValueError(f"params_norm must be shape (7,), got {params_norm.shape}")
    if not np.all((params_norm >= -1e-6) & (params_norm <= 1.0 + 1e-6)):
        raise ValueError(f"params_norm must be in [0, 1]; got {params_norm}")

    plate = _plate(device, dtype)
    loss_fn = _loss(loss_name, device)

    target_t = target_ir.to(device=device, dtype=dtype)
    if target_t.ndim == 1:
        target_t = target_t.unsqueeze(0)
    elif target_t.ndim != 2 or target_t.shape[0] != 1:
        raise ValueError(f"target_ir must be 1-D or [1, N]; got {tuple(target_t.shape)}")

    n_samples = int(target_t.shape[-1])
    dur = float(n_samples / SAMPLE_RATE) if duration is None else float(duration)

    lo = torch.as_tensor(BOUNDS_LO, dtype=dtype, device=device)
    rng = torch.as_tensor(BOUNDS_RANGE, dtype=dtype, device=device)

    def loss_at(xn: torch.Tensor) -> torch.Tensor:
        phys7 = (lo + xn * rng).unsqueeze(0)
        plate14 = _build_plate14_tensor(phys7)
        pred = plate(plate14, duration=dur, normalize=False)
        return loss_fn(target_t.expand_as(pred), pred)[0]

    if use_autograd:
        xn = torch.as_tensor(params_norm, dtype=dtype, device=device).clone()
        xn.requires_grad_(True)
        L = loss_at(xn)
        if not L.requires_grad:
            raise RuntimeError(
                "Autograd graph was severed somewhere in synth+loss; "
                "rerun with use_autograd=False."
            )
        L.backward()
        if xn.grad is None:
            raise RuntimeError("xn.grad is None after backward(); autograd path failed.")
        return xn.grad.detach().cpu().numpy().astype(np.float64)

    # Forward finite differences in normalized space.
    grad = np.zeros(7, dtype=np.float64)
    with torch.no_grad():
        x0 = torch.as_tensor(params_norm, dtype=dtype, device=device)
        L0 = float(loss_at(x0).item())
        for i in range(7):
            xp = x0.clone()
            xp[i] = torch.clamp(xp[i] + fd_eps, 0.0, 1.0)
            actual_eps = float(xp[i].item() - x0[i].item())
            if abs(actual_eps) < 1e-12:
                xp[i] = x0[i] - fd_eps
                actual_eps = float(xp[i].item() - x0[i].item())
            Lp = float(loss_at(xp).item())
            grad[i] = (Lp - L0) / actual_eps
    return grad


# ---- self-test --------------------------------------------------------------

def _run_self_test() -> None:
    import csv
    repo_root = Path(__file__).resolve().parents[2]
    params_csv = repo_root / "data" / "random-IR-100-1.0s" / "random_IR_params_0001.csv"
    with params_csv.open() as f:
        gt_full = next(csv.DictReader(f))
    gt_phys = np.array([float(gt_full[k]) for k in PARAM_KEYS], dtype=np.float64)
    gt_norm = (gt_phys - BOUNDS_LO) / BOUNDS_RANGE
    print(f"[test] GT norm:      {np.array2string(gt_norm, precision=4)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    plate = _plate(device, dtype)
    duration = 0.25

    # Synthetic target = synth(GT) so loss(GT) ≈ 0.
    with torch.no_grad():
        gt_t = torch.as_tensor(gt_phys, dtype=dtype, device=device).unsqueeze(0)
        target = plate(_build_plate14_tensor(gt_t), duration=duration, normalize=False)[0]

    # Sanity: loss at GT.
    L_gt = float(_loss("L1_STFT", device)(target.unsqueeze(0), target.unsqueeze(0))[0].item())
    print(f"[test] L1_STFT at GT (target == pred): {L_gt:.3e}  (expect ~0)")

    # Perturb away from GT in one dimension at a time; check sign of grad.
    rng_fd = 1e-3  # use small eps so FD agrees with autograd
    for i, key in enumerate(PARAM_KEYS):
        x_pert = gt_norm.copy()
        delta = 0.05
        x_pert[i] = float(np.clip(x_pert[i] + delta, 0.0, 1.0))
        actual_delta = x_pert[i] - gt_norm[i]
        sign_back = -np.sign(actual_delta)  # direction back toward GT (negative if we moved up)

        g_auto = compute_gradient(
            x_pert, target, "L1_STFT",
            device=device, dtype=dtype, duration=duration,
            use_autograd=True,
        )
        g_fd = compute_gradient(
            x_pert, target, "L1_STFT",
            device=device, dtype=dtype, duration=duration,
            use_autograd=False, fd_eps=rng_fd,
        )

        # Descent direction = -grad. Want -g[i] to have the same sign as
        # (gt[i] - pert[i]) = -actual_delta, i.e. grad[i] should have sign +actual_delta.
        # Equivalently g_auto[i] * sign(actual_delta) > 0 for "points back".
        points_back = (g_auto[i] * np.sign(actual_delta)) > 0
        # Cosine similarity vs FD on the full 7-vector for sanity.
        cos = float(np.dot(g_auto, g_fd) / (np.linalg.norm(g_auto) * np.linalg.norm(g_fd) + 1e-30))
        rel_err = float(np.linalg.norm(g_auto - g_fd) / (np.linalg.norm(g_fd) + 1e-30))
        print(
            f"[test] perturb {key:>5s} +{actual_delta:+.3f} | "
            f"g_auto[{i}]={g_auto[i]:+.3e}  g_fd[{i}]={g_fd[i]:+.3e}  "
            f"points_back={points_back}  cos(auto, fd)={cos:.4f}  "
            f"rel||auto-fd||={rel_err:.3f}"
        )


if __name__ == "__main__":
    _run_self_test()
