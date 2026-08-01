import math
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import torch


TensorOrFloat = Union[torch.Tensor, float]


class BatchedModalPlateTorch(torch.nn.Module):
    """Batched PyTorch port of ModalPlate/ModalPlate.py.

    This keeps the same modal physics as the NumPy implementation:
    - mode shapes use sin() coupling for simply-supported boundaries
    - modal gain uses the extra 4/(Lx * Ly) scaling from ModalPlate.py
    - recurrence is solved analytically in closed form for stable gradients

    Notes:
    - This is intentionally *not* the DifferentiablePlate G1/G2->(r,Omega)
      round-trip approach.
    - By default, modes below 20 Hz are kept to match ModalPlate.py.
    """

    PARAM_ORDER: Sequence[str] = (
        "Lx",
        "Ly",
        "h",
        "T0",
        "rho",
        "E",
        "nu",
        "T60_DC",
        "T60_F1",
        "loss_F1",
        "fp_x",
        "fp_y",
        "op_x",
        "op_y",
    )

    def __init__(
        self,
        sample_rate: int = 44100,
        fmax: float = 10000.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        drop_sub_20hz_modes: bool = False,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.fmax = float(fmax)
        self.max_omega = 2.0 * math.pi * self.fmax
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.drop_sub_20hz_modes = bool(drop_sub_20hz_modes)

    def _as_tensor(self, value: TensorOrFloat) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=self.dtype)
        return torch.tensor(value, device=self.device, dtype=self.dtype)

    @classmethod
    def params_dicts_to_tensor(
        cls,
        param_dicts: Iterable[Dict[str, float]],
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        rows = []
        for p in param_dicts:
            rows.append([float(p[k]) for k in cls.PARAM_ORDER])
        out = torch.tensor(rows, dtype=dtype)
        if device is not None:
            out = out.to(device)
        return out

    def forward(
        self,
        params_batch: torch.Tensor,
        duration: float,
        vel_calc: bool = False,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Synthesize a batch of impulse responses.

        Args:
            params_batch: Tensor [B, 14] in PARAM_ORDER.
            duration: IR duration in seconds.
            vel_calc: Match ModalPlate.py velocity branch if True.
            normalize: Per-example max-abs normalization if True.

        Returns:
            Tensor [B, Ts]
        """
        if params_batch.ndim != 2 or params_batch.shape[1] != len(self.PARAM_ORDER):
            raise ValueError(
                f"params_batch must have shape [B, {len(self.PARAM_ORDER)}], "
                f"got {tuple(params_batch.shape)}"
            )

        p = params_batch.to(device=self.device, dtype=self.dtype)
        B = p.shape[0]
        Ts = int(self.sample_rate * float(duration))
        k = 1.0 / float(self.sample_rate)

        Lx, Ly, h, T0, rho, E, nu, T60_DC, T60_F1, loss_F1, fp_x, fp_y, op_x, op_y = [
            p[:, i] for i in range(14)
        ]

        # Plate stiffness and modal mass exactly as in ModalPlate.py
        D = E * h.pow(3) / (12.0 * (1.0 - nu.pow(2)))
        ms = 0.25 * rho * h * Lx * Ly

        OmDamp1 = torch.zeros_like(loss_F1)
        OmDamp2 = 2.0 * math.pi * loss_F1
        dOmSq = OmDamp2.pow(2) - OmDamp1.pow(2)
        alpha = 3.0 * math.log(10.0) / dOmSq * (OmDamp2.pow(2) / T60_DC - OmDamp1.pow(2) / T60_F1)
        beta = 3.0 * math.log(10.0) / dOmSq * (1.0 / T60_F1 - 1.0 / T60_DC)

        # Batch-wise mode-grid upper bounds (same DDx/DDy logic as NumPy code).
        inner = torch.sqrt(torch.clamp(T0.pow(2) + 4.0 * (self.max_omega ** 2) * rho * h * D, min=0.0))
        disc = torch.clamp((-T0 + inner) / (2.0 * D), min=0.0)
        sqrt_disc = torch.sqrt(disc)
        DDx_f = torch.floor(Lx / math.pi * sqrt_disc)
        DDy_f = torch.floor(Ly / math.pi * sqrt_disc)
        max_DDx = max(1, int(torch.max(DDx_f).item()))
        max_DDy = max(1, int(torch.max(DDy_f).item()))

        m_vals = (
            torch.arange(1, max_DDx + 1, device=self.device, dtype=self.dtype)
            .view(-1, 1)
            .repeat(1, max_DDy)
            .flatten()
        )
        n_vals = (
            torch.arange(1, max_DDy + 1, device=self.device, dtype=self.dtype)
            .view(1, -1)
            .repeat(max_DDx, 1)
            .flatten()
        )
        m_idx = m_vals.unsqueeze(0).expand(B, -1)
        n_idx = n_vals.unsqueeze(0).expand(B, -1)

        # Keep only modes within each example's DDx/DDy limits.
        ddx_mask = m_idx <= DDx_f.unsqueeze(1)
        ddy_mask = n_idx <= DDy_f.unsqueeze(1)
        dn_mask = ddx_mask & ddy_mask

        g1 = (m_idx * (math.pi / Lx.unsqueeze(1))).pow(2) + (n_idx * (math.pi / Ly.unsqueeze(1))).pow(2)
        g2 = g1.pow(2)
        om_sq = (T0 / (rho * h)).unsqueeze(1) * g1 + (D / (rho * h)).unsqueeze(1) * g2
        om = torch.sqrt(torch.clamp(om_sq, min=0.0))

        max_om_mask = om <= self.max_omega
        valid_mask = dn_mask & max_om_mask
        if self.drop_sub_20hz_modes:
            valid_mask = valid_mask & (om >= (20.0 * 2.0 * math.pi))

        sig = alpha.unsqueeze(1) + beta.unsqueeze(1) * om.pow(2)
        r = torch.exp(-sig * k)

        # Match ModalPlate.py coupling terms (sin, not cos).
        in_weight = torch.sin(fp_x.unsqueeze(1) * math.pi * m_idx) * torch.sin(fp_y.unsqueeze(1) * math.pi * n_idx)
        out_weight = torch.sin(op_x.unsqueeze(1) * math.pi * m_idx) * torch.sin(op_y.unsqueeze(1) * math.pi * n_idx)

        # Match ModalPlate.py Pvec scaling: 4 * ... / (ms * Lx * Ly)
        P = 4.0 * out_weight * in_weight * (k * k) * r / (ms.unsqueeze(1) * Lx.unsqueeze(1) * Ly.unsqueeze(1))
        P = P * valid_mask.to(self.dtype)

        denom = torch.sin(om * k)
        denom = torch.where(torch.abs(denom) < 1e-12, torch.full_like(denom, 1e-12), denom)

        y_raw = torch.zeros((B, Ts), device=self.device, dtype=self.dtype)
        for b in range(B):
            vi = valid_mask[b].nonzero(as_tuple=True)[0]
            if vi.numel() == 0:
                continue

            sig_b = sig[b, vi].unsqueeze(1)
            om_b = om[b, vi].unsqueeze(1)
            den_b = denom[b, vi].unsqueeze(1)
            P_b = P[b, vi].unsqueeze(1)

            chunk_size = max(1000, 50_000_000 // max(1, vi.numel()))
            for start in range(0, Ts, chunk_size):
                end = min(start + chunk_size, Ts)
                t = torch.arange(start, end, device=self.device, dtype=self.dtype).view(1, -1)
                env = torch.exp(-sig_b * k * t)
                osc = torch.sin(om_b * k * (t + 1.0)) / den_b
                y_raw[b, start:end] = torch.sum(P_b * env * osc, dim=0)

        # NumPy recursion outputs y[n] = sum(q1 before update), i.e. one-sample delay.
        y = torch.zeros_like(y_raw)
        y[:, 1:] = y_raw[:, :-1]

        if vel_calc:
            v = torch.zeros_like(y)
            v[:, 1:] = (y[:, 1:] - y[:, :-1]) / k
            y = v

        if normalize:
            amax = torch.max(torch.abs(y), dim=1, keepdim=True).values
            amax = torch.where(amax < 1e-15, torch.ones_like(amax), amax)
            y = y / amax

        return y


def single_from_dict(
    param_dict: Dict[str, float],
    duration: float,
    sample_rate: int = 44100,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    vel_calc: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """Convenience wrapper for one parameter dictionary."""
    model = BatchedModalPlateTorch(
        sample_rate=sample_rate,
        device=device,
        dtype=dtype,
        drop_sub_20hz_modes=False,
    )
    batch = BatchedModalPlateTorch.params_dicts_to_tensor([param_dict], device=model.device, dtype=dtype)
    with torch.no_grad():
        y = model(batch, duration=duration, vel_calc=vel_calc, normalize=normalize)
    return y[0].detach().cpu().numpy()
