"""Band-limit audio to the renderer's ceiling. One definition, two callers.

The plate renders nothing above `fmax` -- modes past it are masked out of the
modal sum entirely -- so a real recording carries content in a band the model
cannot enter, and that asymmetry corrupts two different things at once:

  THE METRIC. loss_mfcc uses librosa's Slaney mel bank, 128 bands to 22.05 kHz.
  12 kHz sits at 51.1 of 59.99 on that axis, so roughly 19 of the 128 bands are
  above the ceiling. In every render they are floored at log10(1e-10); in both
  targets of the saturation reference they carry signal. Every arm/saturation
  number therefore includes a constant penalty for a band nothing can reach.

  THE ENCODER'S INPUT. Worse, because the encoder is on a LINEAR frequency axis:
  1025 bins over 22.05 kHz, of which 467 -- 46% -- sit above 12 kHz. Every
  training render is silent there, so across nearly half the input width the
  network only ever saw the constant log floor. A real IR at 44.1 kHz puts live
  signal in all of it. That is a distribution shift over 46% of the input, and
  saturated activations emitting a constant is exactly what it should produce --
  which is what emt7's encoders did (six of seven parameters identical across
  fifteen audibly different IRs).

Brick wall rather than a designed filter, because the renderer's own ceiling IS
a brick wall: `max_om_mask = om <= self.max_omega` drops modes, it does not roll
them off. Matching that is the point. Pre-ringing is irrelevant against a band
already 20+ dB down.
"""

from __future__ import annotations

import torch


def brickwall_lowpass(x: torch.Tensor, fc: float | None, sr: float) -> torch.Tensor:
    """Zero every rFFT bin at or above `fc`. Operates on the last axis.

    None, or a cutoff at/above Nyquist, returns x untouched -- so a caller can
    pass its flag straight through without branching.
    """
    if fc is None or fc >= sr / 2.0:
        return x
    n = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)
    f = torch.fft.rfftfreq(n, 1.0 / sr, device=x.device)
    keep = (f < fc).to(X.real.dtype)
    return torch.fft.irfft(X * keep, n=n, dim=-1)
