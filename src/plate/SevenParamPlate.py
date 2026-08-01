import math
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import torch
import torch.utils.checkpoint


TensorOrFloat = Union[torch.Tensor, float]


def _modal_chunk_kernel(
    sig: torch.Tensor,
    om: torch.Tensor,
    den: torch.Tensor,
    P: torch.Tensor,
    t: torch.Tensor,
    k: float,
) -> torch.Tensor:
    """Damped-sinusoid sum over modes: [N,1] mode terms against [1,C] times.

    Written as a single expression so torch.compile can fuse the elementwise
    chain into the reduction. Unfused, `env` and `osc` are each materialized at
    the full [N,C] size, which is what makes the modal sum memory-bound.
    """
    env = torch.exp(-sig * k * t)
    osc = torch.sin(om * k * (t + 1.0)) / den
    return torch.sum(P * env * osc, dim=0)


def _modal_chunk_kernel_batched(
    sig: torch.Tensor,
    om: torch.Tensor,
    den: torch.Tensor,
    P: torch.Tensor,
    t: torch.Tensor,
    k: float,
) -> torch.Tensor:
    """Batched form of the modal sum: [B,N,1] mode terms against [1,1,C] times."""
    env = torch.exp(-sig * k * t)
    osc = torch.sin(om * k * (t + 1.0)) / den
    return torch.sum(P * env * osc, dim=1)


_COMPILED_MODAL_CHUNK = None
_COMPILED_MODAL_CHUNK_BATCHED = None


def _raise_recompile_limit() -> None:
    """Dynamo defaults to 8 specializations, past which it silently falls back."""
    try:
        import torch._dynamo as _dynamo

        for name in ("recompile_limit", "cache_size_limit"):
            if hasattr(_dynamo.config, name):
                setattr(_dynamo.config, name, max(getattr(_dynamo.config, name), 128))
    except Exception:
        pass


def _get_modal_chunk_kernel(compile_it: bool):
    """Compile lazily and once; dynamic=False specializes per padded shape.

    Each padded mode count crosses with a full-length and a remainder time chunk,
    so a run legitimately needs a few dozen specializations. Dynamo's default
    recompile limit is 8, past which it silently falls back to eager and the
    fusion is lost for every shape after the eighth -- raise it rather than
    coarsen the bucketing, which would only add padding waste.
    """
    global _COMPILED_MODAL_CHUNK
    if not compile_it or not hasattr(torch, "compile"):
        return _modal_chunk_kernel
    if _COMPILED_MODAL_CHUNK is None:
        _raise_recompile_limit()
        _COMPILED_MODAL_CHUNK = torch.compile(_modal_chunk_kernel, dynamic=False)
    return _COMPILED_MODAL_CHUNK


def _get_modal_chunk_kernel_batched(compile_it: bool):
    """Same, for the batch-parallel form, so the two paths compare like for like."""
    global _COMPILED_MODAL_CHUNK_BATCHED
    if not compile_it or not hasattr(torch, "compile"):
        return _modal_chunk_kernel_batched
    if _COMPILED_MODAL_CHUNK_BATCHED is None:
        _raise_recompile_limit()
        _COMPILED_MODAL_CHUNK_BATCHED = torch.compile(_modal_chunk_kernel_batched, dynamic=False)
    return _COMPILED_MODAL_CHUNK_BATCHED


def _pad_modes(x: torch.Tensor, n_pad: int, value: float) -> torch.Tensor:
    """Right-pad the mode axis to a fixed length so shapes stay static.

    Padding is exact, not approximate: P pads with 0 and den with 1, so a padded
    slot contributes P*exp(0)*sin(0)/1 = 0 to the sum and 0 to every gradient.
    """
    # F.pad acts on the last dim, which is the mode axis for both the 1-D
    # per-example form [modes] and the 2-D batched form [batch, modes].
    if x.shape[-1] == n_pad:
        return x
    return torch.nn.functional.pad(x, (0, n_pad - x.shape[-1]), value=value)


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
        chunk_elems: int = 50_000_000,
        grad_checkpoint: bool = False,
        batched_modal_sum: bool = False,
        compile_modal_sum: bool = False,
        mode_bucket: int = 1024,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.fmax = float(fmax)
        self.max_omega = 2.0 * math.pi * self.fmax
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.drop_sub_20hz_modes = bool(drop_sub_20hz_modes)

        # Time-chunking budget for the modal sum, in elements. Under autograd
        # every chunk's intermediates are retained, so chunking only bounds
        # memory when grad_checkpoint is on.
        self.chunk_elems = int(chunk_elems)
        self.grad_checkpoint = bool(grad_checkpoint)
        # Sum all modes for the whole batch at once instead of looping over the
        # batch. Trades redundant work on examples with few active modes for
        # batch parallelism. Off by default: verify equivalence before enabling.
        self.batched_modal_sum = bool(batched_modal_sum)
        # Fuse the modal sum with torch.compile. The mode axis is padded up to a
        # multiple of mode_bucket so the compiled kernel sees a handful of static
        # shapes instead of a new one every step as the mode count drifts.
        self.compile_modal_sum = bool(compile_modal_sum)
        self.mode_bucket = int(mode_bucket)

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

    def _maybe_checkpoint(self, fn, *tensors):
        needs_grad = any(t.requires_grad for t in tensors)
        if self.grad_checkpoint and torch.is_grad_enabled() and needs_grad:
            return torch.utils.checkpoint.checkpoint(fn, *tensors, use_reentrant=False)
        return fn(*tensors)

    def _modal_sum_single(self, sig_b, om_b, den_b, P_b, k, start, end):
        """Modal series over samples [start, end) for one example: [n_modes,1] inputs."""

        def _inner(sig_b, om_b, den_b, P_b):
            t = torch.arange(start, end, device=self.device, dtype=self.dtype).view(1, -1)
            env = torch.exp(-sig_b * k * t)
            osc = torch.sin(om_b * k * (t + 1.0)) / den_b
            return torch.sum(P_b * env * osc, dim=0)

        return self._maybe_checkpoint(_inner, sig_b, om_b, den_b, P_b)

    def _modal_sum_batched(self, sig, om, denom, P, k, Ts):
        """Modal series for the whole batch at once, one kernel instead of B.

        P has already been multiplied by valid_mask, so modes outside an
        example's own limits contribute exactly zero and the per-example
        nonzero() selection is an optimization rather than a requirement. This
        trades redundant work on examples with few active modes for batch
        parallelism, which is the right trade only when the device is otherwise
        idle between launches.

        Padding is exact: P pads with 0 and den with 1, so a padded slot adds
        P*exp(0)*sin(0)/1 = 0 to the sum and 0 to every gradient.
        """
        B, M = P.shape
        if self.compile_modal_sum:
            bucket = max(1, self.mode_bucket)
            n_pad = ((M + bucket - 1) // bucket) * bucket
        else:
            n_pad = M

        sig_b = _pad_modes(sig, n_pad, 0.0).unsqueeze(2)
        om_b = _pad_modes(om, n_pad, 0.0).unsqueeze(2)
        den_b = _pad_modes(denom, n_pad, 1.0).unsqueeze(2)
        P_b = _pad_modes(P, n_pad, 0.0).unsqueeze(2)

        # Memory is B*n_pad*chunk here rather than n_valid*chunk per example.
        chunk = max(64, self.chunk_elems // max(1, B * n_pad))
        kernel = _get_modal_chunk_kernel_batched(self.compile_modal_sum)
        pieces = []
        for start in range(0, Ts, chunk):
            end = min(start + chunk, Ts)
            t = torch.arange(start, end, device=self.device, dtype=self.dtype).view(1, 1, -1)
            if self.grad_checkpoint and torch.is_grad_enabled() and P_b.requires_grad:
                pieces.append(
                    torch.utils.checkpoint.checkpoint(
                        kernel, sig_b, om_b, den_b, P_b, t, k, use_reentrant=False
                    )
                )
            else:
                pieces.append(kernel(sig_b, om_b, den_b, P_b, t, k))
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

        if self.batched_modal_sum:
            y_raw = self._modal_sum_batched(sig, om, denom, P, k, Ts)
        else:
            kernel = _get_modal_chunk_kernel(self.compile_modal_sum)
            rows = []
            for b in range(B):
                vi = valid_mask[b].nonzero(as_tuple=True)[0]
                if vi.numel() == 0:
                    rows.append(torch.zeros(Ts, device=self.device, dtype=self.dtype))
                    continue

                n = int(vi.numel())
                if self.compile_modal_sum:
                    bucket = max(1, self.mode_bucket)
                    n_pad = ((n + bucket - 1) // bucket) * bucket
                else:
                    n_pad = n

                sig_b = _pad_modes(sig[b, vi], n_pad, 0.0).unsqueeze(1)
                om_b = _pad_modes(om[b, vi], n_pad, 0.0).unsqueeze(1)
                den_b = _pad_modes(denom[b, vi], n_pad, 1.0).unsqueeze(1)
                P_b = _pad_modes(P[b, vi], n_pad, 0.0).unsqueeze(1)

                chunk_size = max(256, self.chunk_elems // max(1, n_pad))
                pieces = []
                for start in range(0, Ts, chunk_size):
                    end = min(start + chunk_size, Ts)
                    t = torch.arange(start, end, device=self.device, dtype=self.dtype).view(1, -1)
                    if self.grad_checkpoint and torch.is_grad_enabled() and P_b.requires_grad:
                        pieces.append(
                            torch.utils.checkpoint.checkpoint(
                                kernel, sig_b, om_b, den_b, P_b, t, k, use_reentrant=False
                            )
                        )
                    else:
                        pieces.append(kernel(sig_b, om_b, den_b, P_b, t, k))
                rows.append(torch.cat(pieces, dim=0))
            y_raw = torch.stack(rows, dim=0)

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
