"""Collapse diagnostics: per-dimension std and effective rank (RankMe)."""
from typing import Dict

import torch
from torch import Tensor


@torch.no_grad()
def per_dim_std(x: Tensor) -> float:
    """Mean over dimensions of the across-batch standard deviation."""
    if x.size(0) < 2:
        return float("nan")
    return x.float().std(dim=0).mean().item()


@torch.no_grad()
def rankme(x: Tensor, eps: float = 1e-7) -> float:
    """Effective rank exp(H(p)), p = normalised singular values of the (batch x d) matrix."""
    if x.size(0) < 2:
        return float("nan")
    s = torch.linalg.svdvals(x.float())
    p = s / (s.sum() + eps) + eps
    return torch.exp(-(p * torch.log(p)).sum()).item()


@torch.no_grad()
def collapse_report(z_ctx: Tensor, targets: Dict[str, Tensor], counts: Dict[str, Tensor]) -> Dict[str, float]:
    out = {"z_std": per_dim_std(z_ctx), "z_rankme": rankme(z_ctx)}
    for k, y in targets.items():
        nonempty = counts[k] > 0
        y = torch.nn.functional.layer_norm(y[nonempty], y.shape[-1:])
        out[f"y_std_{k}"] = per_dim_std(y)
        out[f"y_rankme_{k}"] = rankme(y)
    return out
