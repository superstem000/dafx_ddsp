import math
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import torch
import torch.utils.checkpoint


TensorOrFloat = Union[torch.Tensor, float]


class BatchedModalPlateTorch(torch.nn.Module):
    """Batched six-parameter PyTorch plate model.

    This variant operates directly in the identifiable Task-A space:
        S(P) = {mu, D_div_mu, T0_div_mu, Ly, op_x, op_y}
    and does not reconstruct raw E/rho/h/T0 parameters.
    """

    PARAM_ORDER: Sequence[str] = (
        "mu",
        "D_div_mu",
        "T0_div_mu",
        "Ly",
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
        Lx: float = 1.0,
        T60_DC: float = 6.0,
        T60_F1: float = 2.0,
        loss_F1: float = 500.0,
        fp_x: float = 0.335,
        fp_y: float = 0.467,
        chunk_elems: int = 50_000_000,
        grad_checkpoint: bool = False,
        batched_modal_sum: bool = False,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.fmax = float(fmax)
        self.max_omega = 2.0 * math.pi * self.fmax
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.drop_sub_20hz_modes = bool(drop_sub_20hz_modes)

        # Time-chunking budget for the modal sum, in (modes x samples) elements.
        # Under autograd every chunk's intermediates are retained, so chunking
        # only bounds memory when grad_checkpoint is on; then peak memory is
        # ~chunk_elems instead of ~(n_modes * Ts).
        self.chunk_elems = int(chunk_elems)
        self.grad_checkpoint = bool(grad_checkpoint)
        # Sum all modes for the whole batch at once instead of looping over the
        # batch. Trades redundant work on examples with few active modes for
        # batch parallelism. Off by default: verify equivalence before enabling.
        self.batched_modal_sum = bool(batched_modal_sum)

        # Fixed challenge constants.
        self.Lx = float(Lx)
        self.T60_DC = float(T60_DC)
        self.T60_F1 = float(T60_F1)
        self.loss_F1 = float(loss_F1)
        self.fp_x = float(fp_x)
        self.fp_y = float(fp_y)

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

    def _modal_sum_chunk(
        self,
        sig_b: torch.Tensor,
        om_b: torch.Tensor,
        den_b: torch.Tensor,
        P_b: torch.Tensor,
        k: float,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Sum the modal series over samples [start, end) for one example."""

        def _inner(sig_b, om_b, den_b, P_b):
            t = torch.arange(start, end, device=self.device, dtype=self.dtype).view(1, -1)
            env = torch.exp(-sig_b * k * t)
            osc = torch.sin(om_b * k * (t + 1.0)) / den_b
            return torch.sum(P_b * env * osc, dim=0)

        if self.grad_checkpoint and torch.is_grad_enabled() and P_b.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                _inner, sig_b, om_b, den_b, P_b, use_reentrant=False
            )
        return _inner(sig_b, om_b, den_b, P_b)

    def _modal_sum_batched(self, sig, om, denom, P, k, Ts):
        """Modal series for the whole batch at once.

        P has already been multiplied by valid_mask, so modes outside an
        example's own limits contribute exactly zero and the per-example
        nonzero() selection is an optimization rather than a requirement.
        Invalid modes are safe to evaluate: sig = alpha + beta*om^2 with both
        constants positive, so exp(-sig*k*t) decays to zero and cannot overflow.
        """
        B, M = P.shape
        # Memory is now B*M*chunk rather than n_valid*chunk.
        chunk = max(64, self.chunk_elems // max(1, B * M))
        pieces = []

        def _inner(sig, om, denom, P, start=0, end=0):
            t = torch.arange(start, end, device=self.device, dtype=self.dtype).view(1, 1, -1)
            env = torch.exp(-sig.unsqueeze(2) * k * t)
            osc = torch.sin(om.unsqueeze(2) * k * (t + 1.0)) / denom.unsqueeze(2)
            return torch.sum(P.unsqueeze(2) * env * osc, dim=1)

        for start in range(0, Ts, chunk):
            end = min(start + chunk, Ts)
            fn = (lambda s, o, d, p, _s=start, _e=end: _inner(s, o, d, p, _s, _e))
            if self.grad_checkpoint and torch.is_grad_enabled() and P.requires_grad:
                pieces.append(
                    torch.utils.checkpoint.checkpoint(fn, sig, om, denom, P, use_reentrant=False)
                )
            else:
                pieces.append(fn(sig, om, denom, P))
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        params_batch: torch.Tensor,
        duration: float,
        vel_calc: bool = False,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Synthesize a batch of impulse responses.

        Args:
            params_batch: Tensor [B, 6] in PARAM_ORDER.
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

        mu, D_div_mu, T0_div_mu, Ly, op_x, op_y = [p[:, i] for i in range(6)]

        Lx = torch.as_tensor(self.Lx, dtype=self.dtype, device=self.device).expand(B)
        T60_DC = torch.as_tensor(self.T60_DC, dtype=self.dtype, device=self.device).expand(B)
        T60_F1 = torch.as_tensor(self.T60_F1, dtype=self.dtype, device=self.device).expand(B)
        loss_F1 = torch.as_tensor(self.loss_F1, dtype=self.dtype, device=self.device).expand(B)
        fp_x = torch.as_tensor(self.fp_x, dtype=self.dtype, device=self.device).expand(B)
        fp_y = torch.as_tensor(self.fp_y, dtype=self.dtype, device=self.device).expand(B)

        # Modal mass can be expressed directly with mu = rho*h.
        ms = 0.25 * mu * Lx * Ly

        OmDamp1 = torch.zeros_like(loss_F1)
        OmDamp2 = 2.0 * math.pi * loss_F1
        dOmSq = OmDamp2.pow(2) - OmDamp1.pow(2)
        alpha = 3.0 * math.log(10.0) / dOmSq * (OmDamp2.pow(2) / T60_DC - OmDamp1.pow(2) / T60_F1)
        beta = 3.0 * math.log(10.0) / dOmSq * (1.0 / T60_F1 - 1.0 / T60_DC)

        # DDx/DDy upper bounds entirely in terms of D/mu and T0/mu:
        # disc = (-T0 + sqrt(T0^2 + 4*omega^2*mu*D)) / (2*D)
        #      = (-T0/mu + sqrt((T0/mu)^2 + 4*omega^2*(D/mu))) / (2*(D/mu))
        inner = torch.sqrt(
            torch.clamp(T0_div_mu.pow(2) + 4.0 * (self.max_omega ** 2) * D_div_mu, min=0.0)
        )
        disc = torch.clamp((-T0_div_mu + inner) / (2.0 * D_div_mu), min=0.0)
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

        ddx_mask = m_idx <= DDx_f.unsqueeze(1)
        ddy_mask = n_idx <= DDy_f.unsqueeze(1)
        dn_mask = ddx_mask & ddy_mask

        g1 = (m_idx * (math.pi / Lx.unsqueeze(1))).pow(2) + (n_idx * (math.pi / Ly.unsqueeze(1))).pow(2)
        g2 = g1.pow(2)

        # Also direct six-parameter form:
        # om_sq = (T0/mu)*g1 + (D/mu)*g2
        om_sq = T0_div_mu.unsqueeze(1) * g1 + D_div_mu.unsqueeze(1) * g2
        om = torch.sqrt(torch.clamp(om_sq, min=0.0))

        max_om_mask = om <= self.max_omega
        valid_mask = dn_mask & max_om_mask
        if self.drop_sub_20hz_modes:
            valid_mask = valid_mask & (om >= (20.0 * 2.0 * math.pi))

        sig = alpha.unsqueeze(1) + beta.unsqueeze(1) * om.pow(2)
        r = torch.exp(-sig * k)

        in_weight = torch.sin(fp_x.unsqueeze(1) * math.pi * m_idx) * torch.sin(fp_y.unsqueeze(1) * math.pi * n_idx)
        out_weight = torch.sin(op_x.unsqueeze(1) * math.pi * m_idx) * torch.sin(op_y.unsqueeze(1) * math.pi * n_idx)

        P = 4.0 * out_weight * in_weight * (k * k) * r / (ms.unsqueeze(1) * Lx.unsqueeze(1) * Ly.unsqueeze(1))
        P = P * valid_mask.to(self.dtype)

        denom = torch.sin(om * k)
        denom = torch.where(torch.abs(denom) < 1e-12, torch.full_like(denom, 1e-12), denom)

        if self.batched_modal_sum:
            y_raw = self._modal_sum_batched(sig, om, denom, P, k, Ts)
        else:
            y_raw = torch.zeros((B, Ts), device=self.device, dtype=self.dtype)
            for b in range(B):
                vi = valid_mask[b].nonzero(as_tuple=True)[0]
                if vi.numel() == 0:
                    continue

                sig_b = sig[b, vi].unsqueeze(1)
                om_b = om[b, vi].unsqueeze(1)
                den_b = denom[b, vi].unsqueeze(1)
                P_b = P[b, vi].unsqueeze(1)

                chunk_size = max(1000, self.chunk_elems // max(1, vi.numel()))
                for start in range(0, Ts, chunk_size):
                    end = min(start + chunk_size, Ts)
                    y_raw[b, start:end] = self._modal_sum_chunk(
                        sig_b, om_b, den_b, P_b, k, start, end
                    )

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
    """Convenience wrapper for one six-parameter dictionary."""
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
