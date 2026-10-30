"""EMA target encoder with the BYOL / I-JEPA cosine momentum schedule."""
import math

import torch
from torch import nn


def ema_momentum(step: int, total_steps: int, tau_start: float = 0.996, tau_end: float = 1.0) -> float:
    """τ_s = τ_end - (τ_end - τ_start) * (cos(π s / S) + 1) / 2, i.e. τ_start -> τ_end."""
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return tau_end - (tau_end - tau_start) * (math.cos(math.pi * progress) + 1.0) / 2.0


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """ξ <- τ ξ + (1 - τ) θ for parameters (fused multi-tensor ops); buffers are copied."""
    p_t = list(target.parameters())
    p_o = [p.detach() for p in online.parameters()]
    assert len(p_t) == len(p_o)
    torch._foreach_mul_(p_t, tau)
    torch._foreach_add_(p_t, p_o, alpha=1.0 - tau)
    for b_t, b_o in zip(target.buffers(), online.buffers()):
        b_t.copy_(b_o)
