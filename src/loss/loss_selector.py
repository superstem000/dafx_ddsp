"""Loss selection helpers for CMA-ES CLI scripts."""

from __future__ import annotations

from typing import Callable, List

import torch

from .losses import LOSS_COMPONENTS, LOSS_NAME_ALIASES, available_losses, configure_loss_runtime


def available_loss_names() -> List[str]:
    return available_losses()


def select_loss_function(
    name: str,
    *,
    sample_rate: int = 44100,
    device: str | torch.device = "cpu",
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a loss function by name/alias and configure loss runtime context."""
    configure_loss_runtime(sample_rate=sample_rate, device=device)

    canonical = LOSS_NAME_ALIASES.get("".join(ch for ch in name.lower() if ch.isalnum()))
    if canonical is None:
        choices = ", ".join(available_losses())
        raise ValueError(f"Unknown loss '{name}'. Available losses: {choices}")
    return LOSS_COMPONENTS[canonical]
