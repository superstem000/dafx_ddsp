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


# ============================================================================
# Decomposition variants: isolate each way MSS differs from L1_STFT.
#
# L1_STFT (winner) = single resolution {4096}, LINEAR magnitude, NO spectral-conv.
# MSS              = multi resolution {512,2048,8192}, LOG magnitude, +spectral-conv.
#
# Three axes:  resolution (single/multi) x magnitude (linear/log) x SC term (off/on).
# All variants are built from the SAME two primitives (_stft_mag + mean-abs), so they
# differ ONLY in the toggled flag -> clean attribution of which component hurts.
# The SC term is always computed on LINEAR magnitude (as in loss_mss), independent of
# the log flag, so the magnitude axis toggles only the L1 term.
# ============================================================================

_MSS_RES = [512, 2048, 8192]   # the multi-resolution set used by loss_mss


def _compress_mag(x: torch.Tensor, comp: str, eps: float = 1e-7, gamma: float = 0.3):
    """Magnitude compression modes, ordered by strength at the low-energy end.

    "linear" : x                 (C0) no compression        -- our winner
    "pow"    : x ** gamma        (power-law, gamma=0.3)     -- speech-enhancement standard
    "c2"     : log(1 + x)        (Schwar & Muller's fix)    -- ~linear near 0, log for large x
    "c1"     : log(x + eps)      (the standard MSS log)     -- unbounded below, floor at log(eps)
    """
    if comp == "linear":
        return x
    if comp == "pow":
        return torch.pow(x + 1e-12, gamma)
    if comp == "c2":
        return torch.log1p(x)
    if comp == "c1":
        return torch.log(x + eps)
    raise ValueError(f"unknown compression mode: {comp!r}")


def _make_stft_l1(n_ffts, log: bool = False, sc: bool = False, eps: float = 1e-7,
                  comp: str = None, gamma: float = 0.3):
    """Build a pure magnitude-L1 STFT loss with optional compression and SC term.

    (single=[4096], log=False, sc=False)                 == loss_l1_stft
    (n_ffts=[512,2048,8192], log=True, sc=True)          == loss_mss

    `comp`, when given, overrides `log` and selects the compression explicitly:
    "linear" | "pow" | "c2" | "c1".  `log=True` is equivalent to comp="c1".
    Everything else (window, hop, Hann, L1 distance) is held fixed, so a change
    of `comp` alone isolates the compression variable against loss_l1_stft.
    """
    if comp is None:
        comp = "c1" if log else "linear"

    def _loss(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        total = None
        for nf in n_ffts:
            nf_ = min(nf, target.shape[-1])
            hop = nf_ // 4
            t = _stft_mag(target, nf_, hop)
            c = _stft_mag(candidate, nf_, hop)
            t_c = _compress_mag(t, comp, eps, gamma)
            c_c = _compress_mag(c, comp, eps, gamma)
            mag_term = torch.mean(torch.abs(t_c - c_c), dim=(1, 2))
            term = mag_term
            if sc:
                term = term + torch.norm(t - c, p="fro", dim=(1, 2)) / (
                    torch.norm(t, p="fro", dim=(1, 2)) + eps
                )
            total = term if total is None else total + term
        return total / len(n_ffts)

    return _loss


_DECOMP_LOSSES: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    # --- window-size ladder: single-res, linear, no SC (1024 & 4096 already exist) ---
    "L1_STFT_512":  _make_stft_l1([512]),
    "L1_STFT_2048": _make_stft_l1([2048]),
    "L1_STFT_8192": _make_stft_l1([8192]),
    # --- single-factor flips from the L1_STFT winner (4096, linear, no SC) ---
    "L1_STFT_multi": _make_stft_l1(_MSS_RES, log=False, sc=False),  # flip: resolution
    "L1_STFT_log":   _make_stft_l1([4096],  log=True,  sc=False),   # flip: magnitude (log)
    "L1_STFT_sc":    _make_stft_l1([4096],  log=False, sc=True),    # flip: SC term
    # --- two-factor cells (fill the factorial toward MSS) ---
    "L1_STFT_log_multi": _make_stft_l1(_MSS_RES, log=True,  sc=False),  # multi + log
    "L1_STFT_sc_multi":  _make_stft_l1(_MSS_RES, log=False, sc=True),   # multi + SC
    "L1_STFT_log_sc":    _make_stft_l1([4096],  log=True,  sc=True),    # single + log + SC
}

LOSS_COMPONENTS.update(_DECOMP_LOSSES)
for _canon in _DECOMP_LOSSES:
    LOSS_NAME_ALIASES[_normalize_name(_canon)] = _canon


# ============================================================================
# Schwar & Muller (2023), "Multi-Scale Spectral Loss Revisited" (IEEE SPL).
# Their recommended "Smooth MSS" configuration = (WF, S5, C2, D2):
#   WF : flat-top window          -> very low sidelobes, wide mainlobe
#   S5 : prime window sizes       -> z rarely coincides w/ DFT bins at 2+ scales
#   C2 : p(x) = log(1 + gamma*x)  -> the log FIX (>=0, no large log(eps) floor)
#   D2 : d(Y,Yhat) = ||Y - Yhat||_2^2   (squared L2 over the spectrogram matrix)
#
# This is the paper's GENERALIZED MSS (eq. 2): a matrix distance between
# (compressed) magnitude spectrograms summed over window sizes -- NOT the
# auraloss "spectral-convergence + log-magnitude" pair used by loss_mss above.
# Hop is H = N/2 as in the paper.  gamma = 1 for comparability with C1.
# ============================================================================

# Standard flat-top window coefficients (general-cosine form; matches scipy).
_FLATTOP_A = (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368)
_flattop_windows: Dict = {}


def _flattop_window(n: int, device: torch.device) -> torch.Tensor:
    """Flat-top window (WF), cached per (length, device). Length n == n_fft."""
    key = (int(n), str(device))
    w = _flattop_windows.get(key)
    if w is None:
        if n <= 1:
            w = torch.ones(max(int(n), 1), device=device, dtype=torch.float32)
        else:
            k = torch.arange(n, device=device, dtype=torch.float32)
            phase = 2.0 * np.pi * k / (n - 1)
            w = torch.zeros(n, device=device, dtype=torch.float32)
            for i, a in enumerate(_FLATTOP_A):
                w = w + ((-1.0) ** i) * a * torch.cos(i * phase)
        _flattop_windows[key] = w
    return w


_MSS_PRIMES_S5 = [67, 127, 257, 509, 1021, 2053]  # paper S5


def _compress_c2(x: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """C2 compression: log(1 + gamma*x). Always >= 0, no log(eps) floor."""
    return torch.log1p(gamma * x)


def _make_smooth_mss(n_ffts=_MSS_PRIMES_S5, gamma: float = 1.0):
    """Build the paper's Smooth MSS loss (WF, S5, C2, D2). Returns [B]."""
    def _loss(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        total = None
        for nf in n_ffts:
            nf_ = min(int(nf), target.shape[-1])
            hop = max(1, nf_ // 2)                       # H = N/2
            win = _flattop_window(nf_, target.device)    # WF
            t = torch.abs(torch.stft(target, nf_, hop, window=win, return_complex=True))
            c = torch.abs(torch.stft(candidate, nf_, hop, window=win, return_complex=True))
            t = _compress_c2(t, gamma)                   # C2
            c = _compress_c2(c, gamma)
            d = torch.sum((t - c) ** 2, dim=(1, 2))      # D2: squared L2
            total = d if total is None else total + d
        return total
    return _loss


loss_smooth_mss = _make_smooth_mss()


LOSS_COMPONENTS["SmoothMSS"] = loss_smooth_mss
LOSS_NAME_ALIASES[_normalize_name("SmoothMSS")] = "SmoothMSS"
for _alias in ("smooth_mss", "mss_smooth", "mss_fix", "mssfix", "smoothmss"):
    LOSS_NAME_ALIASES[_normalize_name(_alias)] = "SmoothMSS"


# ---------------------------------------------------------------------------
# Resolution-matched Smooth MSS: same machinery (WF, C2, D2, hop N/2), but the
# window sizes are matched to our MSS's {512, 2048, 8192} -> nearest primes
# {509, 2053, 8191}. Purpose: test whether Smooth MSS still lags L1_STFT at
# resolutions we've already shown converge fine (2048->0.96, 8192->0.96), i.e.
# remove the "windows too small" confound from the SmoothMSS-vs-L1 comparison.
# ---------------------------------------------------------------------------
_MSS_RES_PRIMES = [509, 2053, 8191]   # nearest primes to {512, 2048, 8192}

loss_smooth_mss_resmatch = _make_smooth_mss(n_ffts=_MSS_RES_PRIMES)

LOSS_COMPONENTS["SmoothMSS_resmatch"] = loss_smooth_mss_resmatch
LOSS_NAME_ALIASES[_normalize_name("SmoothMSS_resmatch")] = "SmoothMSS_resmatch"
for _alias in ("smoothmss_resmatch", "smooth_mss_resmatch", "smoothmss_match"):
    LOSS_NAME_ALIASES[_normalize_name(_alias)] = "SmoothMSS_resmatch"


# ===========================================================================
# Compression-isolation cells: single-res 4096, Hann, L1 -- EXACTLY the
# loss_l1_stft config with ONLY the magnitude compression changed.  Together
# with L1_STFT (linear) and L1_STFT_log (C1) these give a single-variable
# dose-response ladder across compression strength:
#
#     linear  ->  pow(0.3)  ->  log(1+x)  ->  log(x+eps)
#      (C0)        (POW)        (C2)          (C1)
#
# Prediction from the floor-up-weighting mechanism: floor-weighting, minimum
# relocation, and recovery failure should all increase monotonically along
# this ladder.
# ===========================================================================
_DECOMP_LOSSES["L1_STFT_c2"] = _make_stft_l1([4096], comp="c2")
_DECOMP_LOSSES["L1_STFT_pow"] = _make_stft_l1([4096], comp="pow", gamma=0.3)


# ===========================================================================
# Spectral Optimal Transport (Torres, Peeters & Richard, ICASSP 2024)
# "Unsupervised Harmonic Parameter Estimation Using DDSP and Spectral OT"
#
# Horizontal (energy-displacement) loss: per frame, normalise the power
# spectrum to a probability distribution over frequency, then take the
# 1-D p-Wasserstein distance along the frequency axis; average over frames.
#
# W_p^p is computed by the quantile-function formula (as in POT / their code):
#     W_p(a,b)^p = sum_i |F_a^-1(r_i) - F_b^-1(r_i)|^p * (r_i - r_{i-1})
# with r the merged, sorted set of CDF levels of both measures.
#
# Config follows their best variant SOT-2048: flat-top window, gamma = 2048,
# squared-magnitude (power) spectra, p = 2, no root (W_p^p is more stable).
# ===========================================================================
def _wasserstein_1d_freq(a: torch.Tensor, b: torch.Tensor,
                         pos: torch.Tensor, p: int = 2) -> torch.Tensor:
    """Batched 1-D W_p^p between distributions on a shared support.

    a, b : [M, F] non-negative weights, each row summing to 1.
    pos  : [F]    bin positions (frequency axis, normalised to [0, 1]).
    returns [M].
    """
    cws_a = torch.cumsum(a, dim=-1).contiguous()
    cws_b = torch.cumsum(b, dim=-1).contiguous()

    # merged quantile levels r (sorted), and their spacing dr
    qs = torch.sort(torch.cat([cws_a, cws_b], dim=-1), dim=-1).values
    dr = qs - F.pad(qs[..., :-1], (1, 0))                       # r_i - r_{i-1}

    Fbins = a.shape[-1]
    x = pos.unsqueeze(0).expand(a.shape[0], -1)                 # [M, F]

    idx_a = torch.searchsorted(cws_a, qs.contiguous()).clamp(0, Fbins - 1)
    idx_b = torch.searchsorted(cws_b, qs.contiguous()).clamp(0, Fbins - 1)
    icdf_a = torch.take_along_dim(x, idx_a, dim=-1)             # F_a^-1(r)
    icdf_b = torch.take_along_dim(x, idx_b, dim=-1)             # F_b^-1(r)

    return torch.sum(torch.abs(icdf_a - icdf_b) ** p * dr, dim=-1)


def _make_sot(n_fft: int = 2048, p: int = 2, lam: float = 0.0,
              lin_ffts: List[int] = None, eps: float = 1e-8):
    """Spectral Optimal Transport loss.

    lam > 0 adds Torres et al.'s vertical stabiliser: L_SOT + lam * L_lin,
    where L_lin is the *linear* (uncompressed) multi-scale magnitude L1.
    Their paper uses lam = 0.05 with Gamma0 = {2048,...,64}.
    """
    if lin_ffts is None:
        lin_ffts = [2048, 1024, 512, 256, 128, 64]
    lin_loss = _make_stft_l1(lin_ffts, comp="linear") if lam > 0 else None

    def _loss(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        nf = min(n_fft, target.shape[-1])
        hop = nf // 4
        win = _flattop_window(nf, target.device)                # WF, as they use
        t = torch.abs(torch.stft(target, nf, hop, win_length=nf,
                                 window=win, return_complex=True)) ** 2
        c = torch.abs(torch.stft(candidate, nf, hop, win_length=nf,
                                 window=win, return_complex=True)) ** 2
        B, Fb, T = t.shape

        # [B, F, T] -> [B*T, F]; each frame a distribution over frequency
        t_f = t.permute(0, 2, 1).reshape(-1, Fb)
        c_f = c.permute(0, 2, 1).reshape(-1, Fb)
        t_f = t_f / (t_f.sum(dim=-1, keepdim=True) + eps)
        c_f = c_f / (c_f.sum(dim=-1, keepdim=True) + eps)

        pos = torch.linspace(0.0, 1.0, Fb, device=target.device, dtype=t_f.dtype)
        w = _wasserstein_1d_freq(t_f, c_f, pos, p=p)            # [B*T]
        out = w.reshape(B, T).mean(dim=-1)                      # temporal mean

        if lin_loss is not None:
            out = out + lam * lin_loss(target, candidate)
        return out

    return _loss


# --- lambda: scale-matched, NOT the paper's literal 0.05 ---------------------
# Torres et al. use lam = 0.05 with 16 kHz / 4096-sample harmonic signals.  That
# constant is NOT scale-free: it fixes the RATIO of the two terms only at their
# scales.  Measured on our 44.1 kHz modal plate IRs (terrain - gt, 50 IRs):
#
#     bare SOT range          ~ 1.3e-4
#     linear MSS range        ~ 4.9e-1     (Gamma0 = {2048..64})
#
# so lam = 0.05 makes the linear term ~200x larger than the transport term --
# i.e. "SOT_lin" would be numerically ~99.5% linear MSS, and the transport
# formulation we are supposedly evaluating contributes almost nothing.
#
# Their framing is explicit that the vertical term is a STABILISER -- it exists
# "to penalize W2 minima with low spectral overlap" and to fix "instability
# issues" from the normalisation.  It is a corrective, not a co-equal objective:
# in their paper transport does the frequency localisation.  A 1:1 split would
# overweight the linear term relative to that intent (and, since we already know
# linear magnitude works well here, a 1:1 result would be uninterpretable -- we
# could not tell whether transport or linear did the work).
#
# We therefore choose lam so that transport leads ~3:1 over the stabiliser --
# transport clearly dominates the objective (~75%), but the linear term still
# carries enough weight to actually veto the degenerate minima it exists to
# catch (a near-vestigial 10:1 term likely could not):
#
#     lam = range_SOT / (3 * range_lin) ~ 1.3e-4 / (3 * 4.9e-1) ~ 8.7e-5
#
# Together with bare SOT (lam = 0) this gives the two informative points:
# pure transport, and transport with a light magnitude stabiliser.
_SOT_LAMBDA = 8.7e-5      # transport-dominant, ~3:1 over the linear stabiliser
_SOT_LAMBDA_PAPER = 0.05  # Torres et al.'s literal value (linear-dominated here)

_DECOMP_LOSSES["SOT"] = _make_sot(n_fft=2048, p=2, lam=0.0)
_DECOMP_LOSSES["SOT_lin"] = _make_sot(n_fft=2048, p=2, lam=_SOT_LAMBDA)
_DECOMP_LOSSES["SOT_lin_paper"] = _make_sot(n_fft=2048, p=2, lam=_SOT_LAMBDA_PAPER)


for _name, _fn in (
    ("L1_STFT_c2", _DECOMP_LOSSES["L1_STFT_c2"]),
    ("L1_STFT_pow", _DECOMP_LOSSES["L1_STFT_pow"]),
    ("SOT", _DECOMP_LOSSES["SOT"]),
    ("SOT_lin", _DECOMP_LOSSES["SOT_lin"]),
    ("SOT_lin_paper", _DECOMP_LOSSES["SOT_lin_paper"]),
):
    LOSS_COMPONENTS[_name] = _fn
    LOSS_NAME_ALIASES[_normalize_name(_name)] = _name

for _alias, _canon in (
    ("l1stftc2", "L1_STFT_c2"), ("l1_stft_log1p", "L1_STFT_c2"),
    ("l1stftpow", "L1_STFT_pow"), ("l1_stft_power", "L1_STFT_pow"),
    ("sot_mss", "SOT_lin"), ("sotlin", "SOT_lin"),
):
    LOSS_NAME_ALIASES[_normalize_name(_alias)] = _canon


# ===========================================================================
# Mel_linear : the C0 counterpart of loss_mel (line ~211).
#
# loss_mel applies log(x + 1e-7) to the mel-filtered magnitudes.  Mel_linear is
# IDENTICAL in every other respect -- same three (n_fft, hop, n_mels) configs,
# same librosa mel filterbanks, same L1 distance -- and only drops the log.
#
# (Mel, Mel_linear) is therefore a controlled compression flip on a SECOND
# representation, independent of the STFT ladder.  If compression degrades
# recovery here too, the effect is representation-independent rather than an
# artefact of the linear-frequency STFT.
# ===========================================================================
def loss_mel_linear(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    configs = [(512, 128, 64), (2048, 512, 128), (min(8192, target.shape[-1]), 2048, 256)]
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
        total += torch.mean(torch.abs(t_mel - c_mel), dim=(1, 2))   # <-- NO LOG
    return total / len(configs)


# ===========================================================================
# Energy-masked STFT losses -- the direct test of the floor-up-weighting mechanism.
#
# Fig. 1 shows C1 (log) spends ~95% of its loss mass on the lowest-energy 50%
# of TF bins, while linear spends ~40%.  The mechanism claims it is precisely
# that floor mass that relocates the minimum off the true parameters.
#
# If so, then masking the floor OUT while KEEPING the log compression should
# restore convergence.  That is L1_STFT_log_masked.
#
# The mask keeps the top `keep_top` fraction of bins ranked by TARGET magnitude.
# It depends only on the target signal -- no ground-truth parameter information
# leaks in -- so this is a deployable loss, not merely a diagnostic probe.
#
# L1_STFT_masked (linear + same mask) is the necessary CONTROL: it shows whether
# masking by itself changes anything.  Without it, a positive result for
# log_masked could be dismissed as "masking helps any loss".
# ===========================================================================
def _make_stft_l1_masked(n_ffts, comp: str = "c1", keep_top: float = 0.5,
                         sc: bool = False, eps: float = 1e-7, gamma: float = 0.3):
    def _loss(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        total = None
        for nf in n_ffts:
            nf_ = min(nf, target.shape[-1])
            hop = nf_ // 4
            t = _stft_mag(target, nf_, hop)                 # [B, F, T]
            c = _stft_mag(candidate, nf_, hop)
            B = t.shape[0]

            # threshold from the TARGET only: keep the top `keep_top` of bins
            tf = t.reshape(B, -1)
            n_bins = tf.shape[1]
            k = max(1, min(n_bins, int(round(n_bins * (1.0 - keep_top)))))
            thresh = torch.kthvalue(tf, k, dim=1, keepdim=True).values      # [B, 1]
            mask = (tf >= thresh).to(t.dtype).reshape(t.shape)              # [B, F, T]

            t_c = _compress_mag(t, comp, eps, gamma)
            c_c = _compress_mag(c, comp, eps, gamma)
            diff = torch.abs(t_c - c_c) * mask
            denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
            term = diff.sum(dim=(1, 2)) / denom             # mean over KEPT bins

            if sc:
                term = term + torch.norm((t - c) * mask, p="fro", dim=(1, 2)) / (
                    torch.norm(t * mask, p="fro", dim=(1, 2)) + eps
                )
            total = term if total is None else total + term
        return total / len(n_ffts)

    return _loss


# single-res 4096 / Hann / L1 -- same config as loss_l1_stft, only the magnitude
# transform and the mask differ.
_DECOMP_LOSSES["L1_STFT_log_masked"] = _make_stft_l1_masked([4096], comp="c1", keep_top=0.5)
_DECOMP_LOSSES["L1_STFT_masked"]     = _make_stft_l1_masked([4096], comp="linear", keep_top=0.5)


LOSS_COMPONENTS["Mel_linear"] = loss_mel_linear
LOSS_COMPONENTS["L1_STFT_log_masked"] = _DECOMP_LOSSES["L1_STFT_log_masked"]
LOSS_COMPONENTS["L1_STFT_masked"] = _DECOMP_LOSSES["L1_STFT_masked"]

for _name in ("Mel_linear", "L1_STFT_log_masked", "L1_STFT_masked"):
    LOSS_NAME_ALIASES[_normalize_name(_name)] = _name

for _alias, _canon in (
    ("mel_lin", "Mel_linear"), ("mellinear", "Mel_linear"), ("mel_c0", "Mel_linear"),
    ("l1_stft_masked_log", "L1_STFT_log_masked"), ("logmasked", "L1_STFT_log_masked"),
    ("linmasked", "L1_STFT_masked"),
):
    LOSS_NAME_ALIASES[_normalize_name(_alias)] = _canon


# ═══════════════════════════════════════════════════════════════════════════
# Representation-diversity additions (SI-SDR, EDC/EDR, MFCC) + level-preserving
# spectral OT control. Appended after the decomp/SOT block so the OT control can
# reuse the EXACT SOT machinery (_wasserstein_1d_freq, _flattop_window).
# Interface matches the rest of the file: inputs [B, N], output [B], lower=better.
# ═══════════════════════════════════════════════════════════════════════════


_dct_mats: Dict[tuple, torch.Tensor] = {}


def _get_dct(n_mfcc: int, n_mels: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix [n_mfcc, n_mels] (type-2, 'ortho' norm) --
    the standard MFCC transform (librosa / torchaudio). Verified identical to
    scipy.fft.dct(type=2, norm='ortho') to machine precision."""
    key = (n_mfcc, n_mels, str(_RUNTIME_DEVICE))
    if key not in _dct_mats:
        n = torch.arange(float(n_mels)).unsqueeze(0)            # [1, n_mels]
        k = torch.arange(float(n_mfcc)).unsqueeze(1)            # [n_mfcc, 1]
        d = torch.cos(np.pi / float(n_mels) * (n + 0.5) * k)    # [n_mfcc, n_mels]
        d[0] *= 1.0 / np.sqrt(2.0)
        d *= np.sqrt(2.0 / float(n_mels))
        _dct_mats[key] = d.float().to(_RUNTIME_DEVICE)
    return _dct_mats[key]


def loss_sisdr(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Negative scale-invariant SDR (Le Roux et al., ICASSP 2019). Zero-mean;
    project estimate onto target; target-energy / residual-energy ratio in dB,
    negated so lower = better. NOTE: returns dB, so this loss can be negative
    (unlike the >=0 magnitude losses) -- fine for rank-based CMA-ES."""
    eps = 1e-8
    t = target - target.mean(dim=1, keepdim=True)
    c = candidate - candidate.mean(dim=1, keepdim=True)
    alpha = torch.sum(c * t, dim=1, keepdim=True) / (torch.sum(t * t, dim=1, keepdim=True) + eps)
    s_target = alpha * t
    e_noise = c - s_target
    ratio = (torch.sum(s_target ** 2, dim=1) + eps) / (torch.sum(e_noise ** 2, dim=1) + eps)
    return -10.0 * torch.log10(ratio + eps)


def loss_edc(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Broadband energy-decay-curve loss: Schroeder backward integration
    EDC(t) = sum_{tau>=t} h(tau)^2, normalized to 0 dB at t=0, L2 in dB.
    Decay-domain, orthogonal to STFT magnitude."""
    eps = 1e-10
    t_edc = torch.flip(torch.cumsum(torch.flip(target ** 2, dims=[1]), dim=1), dims=[1])
    c_edc = torch.flip(torch.cumsum(torch.flip(candidate ** 2, dims=[1]), dim=1), dims=[1])
    t_db = 10.0 * torch.log10(t_edc / (t_edc[:, 0:1] + eps) + eps)
    c_db = 10.0 * torch.log10(c_edc / (c_edc[:, 0:1] + eps) + eps)
    return torch.mean((t_db - c_db) ** 2, dim=1)


def loss_edr(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Energy-decay-relief loss (Jot): per-frequency-band Schroeder integration
    of STFT power, L2 in dB. Frequency-resolved decay (decay x frequency)."""
    eps = 1e-10
    n_fft = min(1024, target.shape[-1])
    hop = min(n_fft // 4, n_fft - 1)
    t_pow = _stft_mag(target, n_fft, hop) ** 2
    c_pow = _stft_mag(candidate, n_fft, hop) ** 2
    t_edr = torch.flip(torch.cumsum(torch.flip(t_pow, dims=[2]), dim=2), dims=[2])
    c_edr = torch.flip(torch.cumsum(torch.flip(c_pow, dims=[2]), dim=2), dims=[2])
    t_db = 10.0 * torch.log10(t_edr / (t_edr[:, :, 0:1] + eps) + eps)
    c_db = 10.0 * torch.log10(c_edr / (c_edr[:, :, 0:1] + eps) + eps)
    return torch.mean((t_db - c_db) ** 2, dim=(1, 2))


def loss_mfcc(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """MFCC L1 distance. Standard pipeline: power mel-spectrogram ->
    power-to-dB (10*log10) -> orthonormal DCT-II -> first n_mfcc coefficients."""
    eps = 1e-10
    n_fft = min(2048, target.shape[-1])
    hop = min(n_fft // 4, n_fft - 1)
    n_mels, n_mfcc = 128, 20
    mel_fb = _get_mel_fb(n_fft, n_mels)                 # [n_mels, n_fft//2+1]
    dct = _get_dct(n_mfcc, n_mels)                      # [n_mfcc, n_mels]
    t_pow = _stft_mag(target, n_fft, hop) ** 2
    c_pow = _stft_mag(candidate, n_fft, hop) ** 2
    t_mel = torch.matmul(mel_fb.unsqueeze(0), t_pow)    # [B, n_mels, T]
    c_mel = torch.matmul(mel_fb.unsqueeze(0), c_pow)
    t_db = 10.0 * torch.log10(t_mel + eps)
    c_db = 10.0 * torch.log10(c_mel + eps)
    t_mfcc = torch.matmul(dct.unsqueeze(0), t_db)       # [B, n_mfcc, T]
    c_mfcc = torch.matmul(dct.unsqueeze(0), c_db)
    return torch.mean(torch.abs(t_mfcc - c_mfcc), dim=(1, 2))


def _make_sot_lp(n_fft: int = 2048, p: int = 2, eps: float = 1e-8):
    """Level-preserving Spectral Optimal Transport -- the controlled contrast
    for SOT. IDENTICAL to _make_sot(lam=0) in every respect (flat-top window,
    n_fft, power spectra, per-frame-normalized 1-D W_p via _wasserstein_1d_freq)
    EXCEPT the temporal aggregation: instead of a flat mean over frames, each
    frame's transport cost is weighted by that frame's share of total (target)
    energy. This restores relative inter-frame level -- the single variable that
    differs from SOT -- without adding an absolute-magnitude term (which would
    make it SOT_lin). So SOT / SpectralOT_LP / SOT_lin is a clean 3-point ladder:
    no level -> relative level (via transport) -> absolute level (via +L_lin)."""
    def _loss(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        nf = min(n_fft, target.shape[-1])
        hop = nf // 4
        win = _flattop_window(nf, target.device)
        t = torch.abs(torch.stft(target, nf, hop, win_length=nf,
                                 window=win, return_complex=True)) ** 2
        c = torch.abs(torch.stft(candidate, nf, hop, win_length=nf,
                                 window=win, return_complex=True)) ** 2
        B, Fb, T = t.shape
        t_f = t.permute(0, 2, 1).reshape(-1, Fb)
        c_f = c.permute(0, 2, 1).reshape(-1, Fb)
        frame_mass = t_f.sum(dim=-1)                            # [B*T] target frame energy
        t_f = t_f / (t_f.sum(dim=-1, keepdim=True) + eps)
        c_f = c_f / (c_f.sum(dim=-1, keepdim=True) + eps)
        pos = torch.linspace(0.0, 1.0, Fb, device=target.device, dtype=t_f.dtype)
        w = _wasserstein_1d_freq(t_f, c_f, pos, p=p).reshape(B, T)
        fm = frame_mass.reshape(B, T)
        weight = fm / (fm.sum(dim=-1, keepdim=True) + eps)      # energy-share weights
        return (w * weight).sum(dim=-1)
    return _loss


loss_spectral_ot_lp = _make_sot_lp(n_fft=2048, p=2)


LOSS_COMPONENTS["SI-SDR"] = loss_sisdr
LOSS_COMPONENTS["EDC"] = loss_edc
LOSS_COMPONENTS["EDR"] = loss_edr
LOSS_COMPONENTS["MFCC"] = loss_mfcc
LOSS_COMPONENTS["SpectralOT_LP"] = loss_spectral_ot_lp

for _name in ("SI-SDR", "EDC", "EDR", "MFCC", "SpectralOT_LP"):
    LOSS_NAME_ALIASES[_normalize_name(_name)] = _name

for _alias, _canon in (
    ("sisdr", "SI-SDR"), ("sdr", "SI-SDR"), ("si_sdr", "SI-SDR"),
    ("mfccs", "MFCC"),
    ("otlp", "SpectralOT_LP"), ("sotlp", "SpectralOT_LP"),
    ("sot_lp", "SpectralOT_LP"), ("spectralotlp", "SpectralOT_LP"),
    ("levelpreservingot", "SpectralOT_LP"), ("sot_levelpreserving", "SpectralOT_LP"),
):
    LOSS_NAME_ALIASES[_normalize_name(_alias)] = _canon
