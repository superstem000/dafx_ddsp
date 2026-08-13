"""Factory-based candidate loss functions.

Each factory returns a callable with signature:
    loss_fn(candidate, target) -> torch.Tensor

Input tensors may be [N] or [B, N]. Returned tensor is [B].
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

try:
    from nnAudio import features as nnfeatures

    HAS_NNAUDIO = True
except Exception:
    HAS_NNAUDIO = False


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"Expected [N] or [B,N], got shape {tuple(x.shape)}")
    return x


def _match_pair(candidate: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = _ensure_2d(candidate)
    t = _ensure_2d(target)
    if c.shape != t.shape:
        raise ValueError(f"candidate and target must have same shape, got {tuple(c.shape)} vs {tuple(t.shape)}")
    return c, t


def _stft_complex(x: torch.Tensor, n_fft: int, hop_length: int, win_length: Optional[int], window: str) -> torch.Tensor:
    n_fft_eff = min(int(n_fft), x.shape[-1])
    if n_fft_eff < 2:
        raise ValueError("n_fft is too small for input length")
    hop = max(1, min(int(hop_length), n_fft_eff - 1))
    win_len = n_fft_eff if win_length is None else max(1, min(int(win_length), n_fft_eff))
    if window == "hann":
        win = torch.hann_window(win_len, device=x.device, dtype=x.dtype)
    elif window == "hamming":
        win = torch.hamming_window(win_len, device=x.device, dtype=x.dtype)
    else:
        raise ValueError(f"Unsupported window: {window}")
    return torch.stft(x, n_fft=n_fft_eff, hop_length=hop, win_length=win_len, window=win, return_complex=True)


def make_waveform_l1_loss() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def loss_fn(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c, t = _match_pair(candidate, target)
        return torch.mean(torch.abs(c - t), dim=1)

    return loss_fn


def make_waveform_l2_loss() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def loss_fn(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c, t = _match_pair(candidate, target)
        return torch.mean((c - t) ** 2, dim=1)

    return loss_fn


def make_l1_stft_loss(
    *,
    n_fft: int = 4096,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = "hann",
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    hop = n_fft // 4 if hop_length is None else hop_length

    def loss_fn(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c, t = _match_pair(candidate, target)
        c_spec = torch.abs(_stft_complex(c, n_fft=n_fft, hop_length=hop, win_length=win_length, window=window))
        t_spec = torch.abs(_stft_complex(t, n_fft=n_fft, hop_length=hop, win_length=win_length, window=window))
        return torch.mean(torch.abs(c_spec - t_spec), dim=(1, 2))

    return loss_fn


def make_fft_mag_l1_loss(*, n_fft: Optional[int] = None) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def loss_fn(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c, t = _match_pair(candidate, target)
        if n_fft is None:
            n = c.shape[-1]
        else:
            n = int(n_fft)
        c_fft = torch.fft.rfft(c, n=n)
        t_fft = torch.fft.rfft(t, n=n)
        return torch.mean(torch.abs(torch.abs(c_fft) - torch.abs(t_fft)), dim=1)

    return loss_fn


def make_group_delay_loss(
    *,
    n_fft: int = 4096,
    hop_length: int = 1024,
    win_length: Optional[int] = None,
    window: str = "hann",
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def loss_fn(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c, t = _match_pair(candidate, target)
        c_z = _stft_complex(c, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        t_z = _stft_complex(t, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)

        c_phase = torch.angle(c_z)
        t_phase = torch.angle(t_z)
        c_gd = -(c_phase[:, 1:, :] - c_phase[:, :-1, :])
        t_gd = -(t_phase[:, 1:, :] - t_phase[:, :-1, :])

        c_gd = (c_gd + torch.pi) % (2 * torch.pi) - torch.pi
        t_gd = (t_gd + torch.pi) % (2 * torch.pi) - torch.pi

        mag_w = (torch.abs(c_z[:, :-1, :]) + torch.abs(t_z[:, :-1, :])) / 2 + 1e-10
        gd_diff = torch.abs(c_gd - t_gd)
        return torch.mean(gd_diff * mag_w / (torch.mean(mag_w, dim=(1, 2), keepdim=True) + 1e-10), dim=(1, 2))

    return loss_fn


def make_cqt_l1_loss(
    *,
    sample_rate: int = 44100,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: float = 32.7,
    min_input_len: int = 33000,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if not HAS_NNAUDIO:
        raise RuntimeError("nnAudio is required for CQT loss")

    layer = nnfeatures.CQT1992v2(
        sr=int(sample_rate),
        hop_length=int(hop_length),
        n_bins=int(n_bins),
        bins_per_octave=int(bins_per_octave),
        fmin=float(fmin),
    )

    def loss_fn(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c, t = _match_pair(candidate, target)
        layer_dev = next(layer.parameters(), None)
        dev = layer_dev.device if layer_dev is not None else c.device
        layer.to(dev)
        c = c.to(dev)
        t = t.to(dev)

        if c.shape[-1] < int(min_input_len):
            pad = int(min_input_len) - c.shape[-1]
            c = F.pad(c, (0, pad))
            t = F.pad(t, (0, pad))

        c_spec = layer(c)
        t_spec = layer(t)
        return torch.mean(torch.abs(c_spec - t_spec), dim=(1, 2))

    return loss_fn


def make_gammatone_loss(
    *,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bins: int = 64,
    fmin: float = 50.0,
    fmax: Optional[float] = None,
    min_input_len: int = 33000,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if not HAS_NNAUDIO:
        raise RuntimeError("nnAudio is required for Gammatone loss")

    if fmax is None:
        fmax = sample_rate / 2

    layer = nnfeatures.Gammatonegram(
        sr=int(sample_rate),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        n_bins=int(n_bins),
        fmin=float(fmin),
        fmax=float(fmax),
    )

    def loss_fn(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c, t = _match_pair(candidate, target)
        layer_dev = next(layer.parameters(), None)
        dev = layer_dev.device if layer_dev is not None else c.device
        layer.to(dev)
        c = c.to(dev)
        t = t.to(dev)

        if c.shape[-1] < int(min_input_len):
            pad = int(min_input_len) - c.shape[-1]
            c = F.pad(c, (0, pad))
            t = F.pad(t, (0, pad))

        c_spec = layer(c)
        t_spec = layer(t)
        return torch.mean(torch.abs(c_spec - t_spec), dim=(1, 2))

    return loss_fn


def make_loss_factory(name: str, **kwargs) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    key = "".join(ch for ch in name.lower() if ch.isalnum())
    table = {
        "l1stft": make_l1_stft_loss,
        "l1cqt": make_cqt_l1_loss,
        "waveforml2": make_waveform_l2_loss,
        "waveforml1": make_waveform_l1_loss,
        "groupdelay": make_group_delay_loss,
        "gammatone": make_gammatone_loss,
        "fft": make_fft_mag_l1_loss,
        "fftloss": make_fft_mag_l1_loss,
    }
    if key not in table:
        raise ValueError(f"Unknown candidate loss '{name}'")
    return table[key](**kwargs)

