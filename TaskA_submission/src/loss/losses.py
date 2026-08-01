"""Loss component library for seven-parameter CMA-ES fitting.

These losses are ported from:
`dafxchal/cmaes_customloss/landscapes/landscape_v5.py`

All functions take tensors shaped [B, N] and return [B].
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

try:
    from nnAudio import features as nnfeatures

    HAS_NNAUDIO = True
except ImportError:
    HAS_NNAUDIO = False

try:
    from kymatio.torch import Scattering1D as KymatioScattering1D

    HAS_SCAT1D = True
except ImportError:
    HAS_SCAT1D = False

try:
    from tslearn.metrics import soft_dtw

    HAS_SOFTDTW = True
except ImportError:
    HAS_SOFTDTW = False


_RUNTIME_DEVICE = torch.device("cpu")
_RUNTIME_SR = 44100

_cqt_layer = None
_vqt_layer = None
_gamma_layer = None
_mr_cqt_layers = None
_scat1d_op = None
_mel_fbs = {}


def configure_loss_runtime(sample_rate: int = 44100, device: str | torch.device = "cpu") -> None:
    """Set runtime context used by loss helper factories and caches."""
    global _RUNTIME_DEVICE, _RUNTIME_SR
    global _cqt_layer, _vqt_layer, _gamma_layer, _mr_cqt_layers, _scat1d_op, _mel_fbs

    dev = torch.device(device)
    if dev != _RUNTIME_DEVICE or int(sample_rate) != _RUNTIME_SR:
        _cqt_layer = None
        _vqt_layer = None
        _gamma_layer = None
        _mr_cqt_layers = None
        _scat1d_op = None
        _mel_fbs = {}

    _RUNTIME_DEVICE = dev
    _RUNTIME_SR = int(sample_rate)


def _nan_like_batch(batch: torch.Tensor) -> torch.Tensor:
    return torch.full((batch.shape[0],), float("nan"), device=batch.device, dtype=batch.dtype)


def _stft_mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    n_fft = min(n_fft, x.shape[-1])
    hop = min(hop, n_fft - 1)
    window = torch.hann_window(n_fft, device=x.device)
    return torch.abs(torch.stft(x, n_fft, hop, window=window, return_complex=True))


def _stft_complex(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    n_fft = min(n_fft, x.shape[-1])
    hop = min(hop, n_fft - 1)
    window = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft, hop, window=window, return_complex=True)


def _get_cqt():
    global _cqt_layer
    if _cqt_layer is None and HAS_NNAUDIO:
        _cqt_layer = nnfeatures.CQT1992v2(
            sr=_RUNTIME_SR, hop_length=512, fmin=32.7, n_bins=84, bins_per_octave=12
        ).to(_RUNTIME_DEVICE)
    return _cqt_layer


def _get_vqt():
    global _vqt_layer
    if _vqt_layer is None and HAS_NNAUDIO:
        try:
            _vqt_layer = nnfeatures.VQT(
                sr=_RUNTIME_SR, hop_length=512, fmin=32.7, n_bins=84, bins_per_octave=12
            ).to(_RUNTIME_DEVICE)
        except Exception:
            _vqt_layer = "FAILED"
    return _vqt_layer


def _get_gamma():
    global _gamma_layer
    if _gamma_layer is None and HAS_NNAUDIO:
        _gamma_layer = nnfeatures.Gammatonegram(
            sr=_RUNTIME_SR,
            n_fft=2048,
            n_bins=64,
            hop_length=512,
            fmin=50.0,
            fmax=_RUNTIME_SR // 2,
        ).to(_RUNTIME_DEVICE)
    return _gamma_layer


def _get_mr_cqt():
    global _mr_cqt_layers
    if _mr_cqt_layers is None and HAS_NNAUDIO:
        configs = [
            dict(hop_length=256, n_bins=96, bins_per_octave=12, fmin=27.5),
            dict(hop_length=512, n_bins=84, bins_per_octave=12, fmin=32.7),
            dict(hop_length=1024, n_bins=72, bins_per_octave=24, fmin=55.0),
        ]
        _mr_cqt_layers = [nnfeatures.CQT1992v2(sr=_RUNTIME_SR, **cfg).to(_RUNTIME_DEVICE) for cfg in configs]
    return _mr_cqt_layers


def _get_scat1d(N: int):
    global _scat1d_op
    if _scat1d_op is None and HAS_SCAT1D:
        try:
            _scat1d_op = KymatioScattering1D(J=8, shape=(N,), Q=12, max_order=2).to(_RUNTIME_DEVICE)
        except Exception:
            _scat1d_op = "FAILED"
    return _scat1d_op


def _get_mel_fb(n_fft: int, n_mels: int) -> torch.Tensor:
    key = (n_fft, n_mels, str(_RUNTIME_DEVICE), _RUNTIME_SR)
    if key not in _mel_fbs:
        import librosa

        fb = librosa.filters.mel(sr=_RUNTIME_SR, n_fft=n_fft, n_mels=n_mels)
        _mel_fbs[key] = torch.from_numpy(fb).float().to(_RUNTIME_DEVICE)
    return _mel_fbs[key]


def loss_sc_logmag(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    configs = [(512, 128), (2048, 512), (min(8192, target.shape[-1]), 2048)]
    eps = 1e-7
    sc_total = torch.zeros(target.shape[0], device=target.device)
    lm_total = torch.zeros(target.shape[0], device=target.device)
    for n_fft, hop in configs:
        n_fft = min(n_fft, target.shape[-1])
        hop = min(hop, n_fft - 1)
        t_spec = _stft_mag(target, n_fft, hop)
        c_spec = _stft_mag(candidate, n_fft, hop)
        sc_total += torch.norm(t_spec - c_spec, p="fro", dim=(1, 2)) / (torch.norm(t_spec, p="fro", dim=(1, 2)) + eps)
        lm_total += torch.mean(torch.abs(torch.log(t_spec + eps) - torch.log(c_spec + eps)), dim=(1, 2))
    return sc_total / 3 + 0.5 * lm_total / 3


def loss_cqt_l1(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if not HAS_NNAUDIO:
        return _nan_like_batch(target)
    layer = _get_cqt()
    min_len = 33000
    B, N = target.shape
    t, c = target, candidate
    if N < min_len:
        t = F.pad(t, (0, min_len - N))
        c = F.pad(c, (0, min_len - N))
    t_cqt = layer(t)
    c_cqt = layer(c)
    return torch.mean(torch.abs(t_cqt - c_cqt), dim=(1, 2))


def loss_cqt_logdecay(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if not HAS_NNAUDIO:
        return _nan_like_batch(target)
    layer = _get_cqt()
    min_len = 33000
    B, N = target.shape
    t, c = target, candidate
    if N < min_len:
        t = F.pad(t, (0, min_len - N))
        c = F.pad(c, (0, min_len - N))
    t_cqt = layer(t)
    c_cqt = layer(c)
    eps = 1e-7
    l1 = torch.mean(torch.abs(t_cqt - c_cqt), dim=(1, 2))
    n_bins = t_cqt.shape[1]
    n_groups, group_size = 6, n_bins // 6
    decay_diff = torch.zeros(B, device=target.device)
    for g in range(n_groups):
        t_band = t_cqt[:, g * group_size : (g + 1) * group_size, :]
        c_band = c_cqt[:, g * group_size : (g + 1) * group_size, :]
        t_energy = torch.sum(t_band**2, dim=1) + eps
        c_energy = torch.sum(c_band**2, dim=1) + eps
        t_log = 10 * torch.log10(t_energy / (t_energy[:, 0:1] + eps))
        c_log = 10 * torch.log10(c_energy / (c_energy[:, 0:1] + eps))
        decay_diff += torch.mean(torch.abs(t_log - c_log), dim=1)
    return l1 + 0.1 * decay_diff / n_groups


def loss_mel(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    configs = [(512, 128, 64), (2048, 512, 128), (min(8192, target.shape[-1]), 2048, 256)]
    eps = 1e-7
    B = target.shape[0]
    total = torch.zeros(B, device=target.device)
    for n_fft, hop, n_mels in configs:
        n_fft = min(n_fft, target.shape[-1])
        hop = min(hop, n_fft - 1)
        mel_fb = _get_mel_fb(n_fft, n_mels)
        t_spec = _stft_mag(target, n_fft, hop)
        c_spec = _stft_mag(candidate, n_fft, hop)
        t_mel = torch.matmul(mel_fb.unsqueeze(0), t_spec)
        c_mel = torch.matmul(mel_fb.unsqueeze(0), c_spec)
        total += torch.mean(torch.abs(torch.log(t_mel + eps) - torch.log(c_mel + eps)), dim=(1, 2))
    return total / len(configs)


def loss_lsd(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    eps = 1e-7
    n_fft = min(4096, target.shape[-1])
    t_spec = _stft_mag(target, n_fft, n_fft // 4)
    c_spec = _stft_mag(candidate, n_fft, n_fft // 4)
    t_log = 20 * torch.log10(t_spec + eps)
    c_log = 20 * torch.log10(c_spec + eps)
    return torch.sqrt(torch.mean((t_log - c_log) ** 2, dim=(1, 2)))


def loss_l1_stft(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    n_fft = min(4096, target.shape[-1])
    t_spec = _stft_mag(target, n_fft, n_fft // 4)
    c_spec = _stft_mag(candidate, n_fft, n_fft // 4)
    return torch.mean(torch.abs(t_spec - c_spec), dim=(1, 2))

def loss_l1_stft_1024(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    n_fft = min(1024, target.shape[-1])
    t_spec = _stft_mag(target, n_fft, n_fft // 4)
    c_spec = _stft_mag(candidate, n_fft, n_fft // 4)
    return torch.mean(torch.abs(t_spec - c_spec), dim=(1, 2))

def loss_l1_waveform(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(target - candidate), dim=1)

def loss_l2_waveform(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    return torch.mean((target - candidate) ** 2, dim=1)

def loss_envelope(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    frame_len = 512
    eps = 1e-10
    B, N = target.shape
    pad_len = frame_len - (N % frame_len)
    if pad_len == frame_len:
        pad_len = 0
    t_pad = F.pad(target, (0, pad_len))
    c_pad = F.pad(candidate, (0, pad_len))
    t_rms = torch.sqrt(torch.mean(t_pad.view(B, -1, frame_len) ** 2, dim=2) + eps)
    c_rms = torch.sqrt(torch.mean(c_pad.view(B, -1, frame_len) ** 2, dim=2) + eps)
    return torch.mean(torch.abs(torch.log(t_rms) - torch.log(c_rms)), dim=1)


def loss_esr(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    return torch.sum((target - candidate) ** 2, dim=1) / (torch.sum(target**2, dim=1) + 1e-10)


def loss_freq_decay(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    n_fft = min(4096, target.shape[-1])
    hop = min(2048, n_fft - 1)
    t_spec = _stft_mag(target, n_fft, hop)
    c_spec = _stft_mag(candidate, n_fft, hop)
    n_freq = t_spec.shape[1]
    freqs = torch.linspace(0, _RUNTIME_SR / 2, n_freq, device=target.device)
    eps = 1e-10
    bands = [(50, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    B = target.shape[0]
    total = torch.zeros(B, device=target.device)
    for f_lo, f_hi in bands:
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if mask.sum() == 0:
            continue
        t_energy = torch.sum(t_spec[:, mask, :] ** 2, dim=1) + eps
        c_energy = torch.sum(c_spec[:, mask, :] ** 2, dim=1) + eps
        t_log = torch.clamp(10 * torch.log10(t_energy / (t_energy[:, 0:1] + eps)), -100, 100)
        c_log = torch.clamp(10 * torch.log10(c_energy / (c_energy[:, 0:1] + eps)), -100, 100)
        total += torch.mean(torch.abs(t_log - c_log), dim=1)
    return total / len(bands)


def loss_subband(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    n_fft = min(8192, target.shape[-1])
    hop = n_fft // 4
    eps = 1e-7
    t_spec = _stft_mag(target, n_fft, hop)
    c_spec = _stft_mag(candidate, n_fft, hop)
    n_freq = t_spec.shape[1]
    freqs = torch.linspace(0, _RUNTIME_SR / 2, n_freq, device=target.device)
    bands = [(20, 200, 3.0), (200, 2000, 1.0), (2000, 10000, 2.0)]
    B = target.shape[0]
    total = torch.zeros(B, device=target.device)
    for f_lo, f_hi, weight in bands:
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if mask.sum() == 0:
            continue
        t_band = t_spec[:, mask, :]
        c_band = c_spec[:, mask, :]
        sc = torch.norm(t_band - c_band, p="fro", dim=(1, 2)) / (torch.norm(t_band, p="fro", dim=(1, 2)) + eps)
        logmag = torch.mean(torch.abs(20 * torch.log10(t_band + eps) - 20 * torch.log10(c_band + eps)), dim=(1, 2))
        total += weight * (sc + logmag / 50.0)
    return total


def loss_modal_density(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    n_fft = min(16384, target.shape[-1])
    hop = n_fft // 4
    t_spec = _stft_mag(target, n_fft, hop).mean(dim=2)
    c_spec = _stft_mag(candidate, n_fft, hop).mean(dim=2)
    n_freq = t_spec.shape[1]
    freqs = torch.linspace(0, _RUNTIME_SR / 2, n_freq, device=target.device)
    bands = [(50, 500), (500, 2000), (2000, 5000), (5000, 10000)]
    B = target.shape[0]
    total = torch.zeros(B, device=target.device)
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() < 10:
            continue
        t_band = t_spec[:, mask]
        c_band = c_spec[:, mask]
        t_smooth = F.avg_pool1d(t_band.unsqueeze(1), kernel_size=5, stride=1, padding=2)
        c_smooth = F.avg_pool1d(c_band.unsqueeze(1), kernel_size=5, stride=1, padding=2)
        t_peaks = F.max_pool1d(t_smooth, kernel_size=3, stride=1, padding=1).squeeze(1)
        c_peaks = F.max_pool1d(c_smooth, kernel_size=3, stride=1, padding=1).squeeze(1)
        t_sq = t_smooth.squeeze(1)
        c_sq = c_smooth.squeeze(1)
        t_prom = torch.clamp(torch.std(t_sq, dim=1, keepdim=True) * 0.3, min=1e-7)
        c_prom = torch.clamp(torch.std(c_sq, dim=1, keepdim=True) * 0.3, min=1e-7)
        tk = ((t_sq == t_peaks) & (t_sq > t_prom)).sum(1).float()
        ck = ((c_sq == c_peaks) & (c_sq > c_prom)).sum(1).float()
        bandwidth = hi - lo
        t_density = tk / bandwidth * 1000
        c_density = ck / bandwidth * 1000
        total += torch.abs(t_density - c_density) / torch.max(
            torch.max(t_density, c_density), torch.tensor(1.0, device=target.device)
        )
    return total / len(bands)


def loss_group_delay(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    n_fft = min(4096, target.shape[-1])
    hop = min(1024, n_fft - 1)
    t_z = _stft_complex(target, n_fft, hop)
    c_z = _stft_complex(candidate, n_fft, hop)
    t_phase = torch.angle(t_z)
    c_phase = torch.angle(c_z)
    t_gd = -(t_phase[:, 1:, :] - t_phase[:, :-1, :])
    c_gd = -(c_phase[:, 1:, :] - c_phase[:, :-1, :])
    t_gd = (t_gd + torch.pi) % (2 * torch.pi) - torch.pi
    c_gd = (c_gd + torch.pi) % (2 * torch.pi) - torch.pi
    mag_wt = (torch.abs(t_z[:, :-1, :]) + torch.abs(c_z[:, :-1, :])) / 2 + 1e-10
    gd_diff = torch.clamp(torch.abs(t_gd - c_gd), 0, 100)
    return torch.mean(gd_diff * mag_wt / (torch.mean(mag_wt, dim=(1, 2), keepdim=True) + 1e-10), dim=(1, 2))


def loss_complex_stft(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    configs = [(512, 128), (2048, 512), (min(8192, target.shape[-1]), 2048)]
    B = target.shape[0]
    total = torch.zeros(B, device=target.device)
    for n_fft, hop in configs:
        n_fft = min(n_fft, target.shape[-1])
        hop = min(hop, n_fft - 1)
        t_z = _stft_complex(target, n_fft, hop)
        c_z = _stft_complex(candidate, n_fft, hop)
        total += torch.mean(torch.abs(t_z.real - c_z.real) + torch.abs(t_z.imag - c_z.imag), dim=(1, 2))
    return total / len(configs)


def loss_inst_freq(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    n_fft = min(4096, target.shape[-1])
    hop = min(1024, n_fft - 1)
    t_z = _stft_complex(target, n_fft, hop)
    c_z = _stft_complex(candidate, n_fft, hop)
    t_phase = torch.angle(t_z)
    c_phase = torch.angle(c_z)
    t_if = t_phase[:, :, 1:] - t_phase[:, :, :-1]
    c_if = c_phase[:, :, 1:] - c_phase[:, :, :-1]
    t_if = (t_if + torch.pi) % (2 * torch.pi) - torch.pi
    c_if = (c_if + torch.pi) % (2 * torch.pi) - torch.pi
    mag_wt = (torch.abs(t_z[:, :, :-1]) + torch.abs(c_z[:, :, :-1])) / 2 + 1e-10
    if_diff = torch.clamp(torch.abs(t_if - c_if), 0, 100)
    return torch.mean(if_diff * mag_wt / (torch.mean(mag_wt, dim=(1, 2), keepdim=True) + 1e-10), dim=(1, 2))


def loss_dispersion(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    n_fft = min(2048, target.shape[-1])
    hop = min(256, n_fft - 1)
    eps = 1e-7
    t_spec = _stft_mag(target, n_fft, hop)
    c_spec = _stft_mag(candidate, n_fft, hop)
    n_freq = t_spec.shape[1]
    freqs = torch.linspace(0, _RUNTIME_SR / 2, n_freq, device=target.device).view(1, -1, 1)
    t_power = t_spec**2
    c_power = c_spec**2
    t_cent = torch.sum(freqs * t_power, dim=1) / (torch.sum(t_power, dim=1) + eps)
    c_cent = torch.sum(freqs * c_power, dim=1) / (torch.sum(c_power, dim=1) + eps)
    t_norm = t_cent / (_RUNTIME_SR / 2)
    c_norm = c_cent / (_RUNTIME_SR / 2)
    cent_diff = torch.mean(torch.abs(t_norm - c_norm), dim=1)
    deriv_diff = torch.mean(torch.abs(torch.diff(t_norm, dim=1) - torch.diff(c_norm, dim=1)), dim=1)
    return cent_diff * 10.0 + deriv_diff * 20.0


def loss_centroid_traj(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    n_fft = min(4096, target.shape[-1])
    hop = min(1024, n_fft - 1)
    eps = 1e-7
    t_spec = _stft_mag(target, n_fft, hop)
    c_spec = _stft_mag(candidate, n_fft, hop)
    n_freq = t_spec.shape[1]
    freqs = torch.linspace(0, _RUNTIME_SR / 2, n_freq, device=target.device).view(1, -1, 1)
    t_cent = torch.sum(freqs * t_spec**2, dim=1) / (torch.sum(t_spec**2, dim=1) + eps)
    c_cent = torch.sum(freqs * c_spec**2, dim=1) / (torch.sum(c_spec**2, dim=1) + eps)
    kernel = 7
    t_smooth = F.avg_pool1d(t_cent.unsqueeze(1) / (_RUNTIME_SR / 2), kernel_size=kernel, stride=1, padding=kernel // 2).squeeze(1)
    c_smooth = F.avg_pool1d(c_cent.unsqueeze(1) / (_RUNTIME_SR / 2), kernel_size=kernel, stride=1, padding=kernel // 2).squeeze(1)
    return torch.mean(torch.abs(t_smooth - c_smooth), dim=1) * 20.0


def loss_scattering1d(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if not HAS_SCAT1D:
        return _nan_like_batch(target)
    B, N = target.shape
    op = _get_scat1d(N)
    if op == "FAILED":
        return _nan_like_batch(target)
    with torch.no_grad():
        t_scat = op(target)
        c_scat = op(candidate)
    return torch.mean(torch.abs(t_scat - c_scat), dim=(1, 2))


def loss_vqt(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if not HAS_NNAUDIO:
        return _nan_like_batch(target)
    layer = _get_vqt()
    if layer == "FAILED":
        return _nan_like_batch(target)
    B, N = target.shape
    min_len = 33000
    t, c = target, candidate
    if N < min_len:
        t = F.pad(t, (0, min_len - N))
        c = F.pad(c, (0, min_len - N))
    with torch.no_grad():
        t_spec = layer(t)
        c_spec = layer(c)
    return torch.mean(torch.abs(t_spec - c_spec), dim=(1, 2))


def loss_gammatone(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if not HAS_NNAUDIO:
        return _nan_like_batch(target)
    layer = _get_gamma()
    B, N = target.shape
    min_len = 33000
    t, c = target, candidate
    if N < min_len:
        t = F.pad(t, (0, min_len - N))
        c = F.pad(c, (0, min_len - N))
    with torch.no_grad():
        t_spec = layer(t)
        c_spec = layer(c)
    return torch.mean(torch.abs(t_spec - c_spec), dim=(1, 2))


def loss_multi_res_cqt(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if not HAS_NNAUDIO:
        return _nan_like_batch(target)
    layers = _get_mr_cqt()
    B, N = target.shape
    min_len = 33000
    t, c = target, candidate
    if N < min_len:
        t = F.pad(t, (0, min_len - N))
        c = F.pad(c, (0, min_len - N))
    total = torch.zeros(B, device=target.device)
    with torch.no_grad():
        for layer in layers:
            total += torch.mean(torch.abs(layer(t) - layer(c)), dim=(1, 2))
    return total / len(layers)


def loss_onset_disp(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    n_fft = min(2048, target.shape[-1])
    hop = min(64, n_fft - 1)
    t_spec = _stft_mag(target, n_fft, hop)
    c_spec = _stft_mag(candidate, n_fft, hop)
    n_freq = t_spec.shape[1]
    freqs = torch.linspace(0, _RUNTIME_SR / 2, n_freq, device=target.device)
    bands = [(100, 300), (300, 700), (700, 1500), (1500, 3000), (3000, 6000), (6000, 10000)]
    B = target.shape[0]

    def get_onsets(spec: torch.Tensor) -> torch.Tensor:
        onsets = []
        for f_lo, f_hi in bands:
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            if mask.sum() == 0:
                onsets.append(torch.zeros(B, device=target.device))
                continue
            band_energy = torch.sum(spec[:, mask, :] ** 2, dim=1)
            peak_energy = torch.max(band_energy, dim=1, keepdim=True).values
            threshold = 0.1 * peak_energy
            above = (band_energy > threshold).float()
            first_above = torch.argmax(above, dim=1).float()
            onset_time = first_above * hop / _RUNTIME_SR
            onsets.append(onset_time)
        return torch.stack(onsets, dim=1)

    t_onsets = get_onsets(t_spec)
    c_onsets = get_onsets(c_spec)
    t_rel = t_onsets - t_onsets.min(dim=1, keepdim=True).values
    c_rel = c_onsets - c_onsets.min(dim=1, keepdim=True).values
    return torch.mean(torch.abs(t_rel - c_rel), dim=1) * 1000.0


def loss_null_pattern(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    n_fft = min(16384, target.shape[-1])
    hop = n_fft // 4
    eps = 1e-7
    t_spec = _stft_mag(target, n_fft, hop).mean(dim=2)
    c_spec = _stft_mag(candidate, n_fft, hop).mean(dim=2)
    n_freq = t_spec.shape[1]
    freqs = torch.linspace(0, _RUNTIME_SR / 2, n_freq, device=target.device)

    mask = (freqs >= 50) & (freqs <= 8000)
    t_sub = t_spec[:, mask]
    c_sub = c_spec[:, mask]

    sigma_bins = 20
    kernel = sigma_bins * 2 + 1
    t_smooth = F.avg_pool1d(t_sub.unsqueeze(1), kernel_size=kernel, stride=1, padding=kernel // 2).squeeze(1)
    c_smooth = F.avg_pool1d(c_sub.unsqueeze(1), kernel_size=kernel, stride=1, padding=kernel // 2).squeeze(1)

    t_ratio = t_sub / (t_smooth + eps)
    c_ratio = c_sub / (c_smooth + eps)

    ratio_diff = torch.mean(torch.abs(t_ratio - c_ratio), dim=1)

    t_null_count = (t_ratio < 0.5).sum(dim=1).float()
    c_null_count = (c_ratio < 0.5).sum(dim=1).float()
    null_diff = torch.abs(t_null_count - c_null_count) / torch.clamp(torch.max(t_null_count, c_null_count), min=1.0)

    return ratio_diff * 5.0 + null_diff * 2.0


def loss_softdtw_mel(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if not HAS_SOFTDTW:
        return _nan_like_batch(target)
    import librosa

    B = target.shape[0]
    n_fft = min(4096, target.shape[-1])
    hop = min(2048, n_fft - 1)
    n_mels = 64
    results = []
    for i in range(B):
        t_np = target[i].detach().cpu().numpy()
        c_np = candidate[i].detach().cpu().numpy()
        t_mel = librosa.feature.melspectrogram(y=t_np, sr=_RUNTIME_SR, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
        c_mel = librosa.feature.melspectrogram(y=c_np, sr=_RUNTIME_SR, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
        t_log = np.log(t_mel + 1e-7).T
        c_log = np.log(c_mel + 1e-7).T
        ml = min(t_log.shape[0], c_log.shape[0])
        try:
            val = float(soft_dtw(t_log[:ml], c_log[:ml], gamma=1.0))
        except Exception:
            val = float("nan")
        results.append(val)
    return torch.tensor(results, device=target.device, dtype=target.dtype)


def loss_mss(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    configs = [(512, 128), (2048, 512), (min(8192, target.shape[-1]), 2048)]
    eps = 1e-7
    B = target.shape[0]
    total = torch.zeros(B, device=target.device)
    for n_fft, hop in configs:
        n_fft = min(n_fft, target.shape[-1])
        hop = min(hop, n_fft - 1)
        t_spec = _stft_mag(target, n_fft, hop)
        c_spec = _stft_mag(candidate, n_fft, hop)
        sc = torch.norm(t_spec - c_spec, p="fro", dim=(1, 2)) / (torch.norm(t_spec, p="fro", dim=(1, 2)) + eps)
        logmag = torch.mean(torch.abs(torch.log(t_spec + eps) - torch.log(c_spec + eps)), dim=(1, 2))
        total += sc + logmag
    return total / len(configs)


LOSS_COMPONENTS: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "MSS": loss_mss,
    "SC+LogMag": loss_sc_logmag,
    "CQT_L1": loss_cqt_l1,
    "CQT+LogDec": loss_cqt_logdecay,
    "Mel": loss_mel,
    "LSD": loss_lsd,
    "L1_STFT": loss_l1_stft,
    "L1_STFT_1024": loss_l1_stft_1024,
    "MultiResCQT": loss_multi_res_cqt,
    "VQT": loss_vqt,
    "Gammatone": loss_gammatone,
    "Envelope": loss_envelope,
    "ESR": loss_esr,
    "FreqDecay": loss_freq_decay,
    "Subband": loss_subband,
    "ModalDensity": loss_modal_density,
    "Dispersion": loss_dispersion,
    "CentroidTraj": loss_centroid_traj,
    "OnsetDisp": loss_onset_disp,
    "NullPattern": loss_null_pattern,
    "GroupDelay": loss_group_delay,
    "ComplexSTFT": loss_complex_stft,
    "InstFreq": loss_inst_freq,
    "Scattering1D": loss_scattering1d,
    "SoftDTW_Mel": loss_softdtw_mel,
    "L1": loss_l1_waveform,
    "L2": loss_l2_waveform,
}


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


LOSS_NAME_ALIASES: Dict[str, str] = {}
for canonical in LOSS_COMPONENTS:
    LOSS_NAME_ALIASES[_normalize_name(canonical)] = canonical

_EXTRA_ALIASES = {
    "cqtlogdec": "CQT+LogDec",
    "cqtlogdecay": "CQT+LogDec",
    "cqtl1": "CQT_L1",
    "sclogmag": "SC+LogMag",
    "l1stft": "L1_STFT",
    "multirescqt": "MultiResCQT",
    "softdtw": "SoftDTW_Mel",
    "softdtwmel": "SoftDTW_Mel",
}
LOSS_NAME_ALIASES.update(_EXTRA_ALIASES)


def available_losses() -> List[str]:
    return list(LOSS_COMPONENTS.keys())
