"""RelJEPA objective.

    y_k     = LayerNorm(mean_{u in T, type(u)=k} H[u])      target per node type k (no gradient)
    e_k     = 1[no type-k rows in (t, t + Δ]]              empty-window indicator
    ŷ_k,ê_k = g_φ(z_ctx, emb(Δ), emb(k))                    predictor
    L_JEPA  = Σ_k mean_i ω_i [(1 - e_ik) ℓ(ŷ_ik, sg(y_ik)) + BCE(ê_ik, e_ik)]

ω_i reweights empty vs non-empty examples inversely to their frequency (normalised to mean 1),
mirroring TVE's reweighting of null task vectors. ``reg_weight`` and ``bce_weight`` scale the two
terms; both are 1 except in the empty-head ablation, which sets one to 0.
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Predictor(nn.Module):
    """g_φ: 2-layer MLP of width 2d on [z_ctx, emb(Δ), emb(k)] -> (ŷ_k, ê_k logit)."""

    def __init__(self, dim: int, target_types: List[str], num_horizons: int = 1):
        super().__init__()
        self.target_types = list(target_types)
        self.type_emb = nn.Embedding(len(self.target_types), dim)
        self.horizon_emb = nn.Embedding(num_horizons, dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim),
            nn.LayerNorm(2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim + 1),
        )

    def forward(self, z_ctx: Tensor, horizon: int = 0) -> Dict[str, Tensor]:
        n = z_ctx.size(0)
        h = self.horizon_emb.weight[horizon].expand(n, -1)
        out = {}
        for i, k in enumerate(self.target_types):
            t = self.type_emb.weight[i].expand(n, -1)
            out[k] = self.mlp(torch.cat([z_ctx, h, t], dim=-1))
        return out


def regression_loss(pred: Tensor, target: Tensor, kind: str) -> Tensor:
    """Per-example loss between prediction and (LayerNorm-ed) target."""
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    if kind == "cosine":
        return 1.0 - F.cosine_similarity(pred, target, dim=-1)
    if kind == "mse":
        return F.mse_loss(pred, target, reduction="none").mean(dim=-1)
    raise ValueError(f"unknown regression loss {kind!r}")


def empty_weights(empty: Tensor, p_empty: Optional[float]) -> Tensor:
    """Inverse-frequency weights for empty (1) vs non-empty (0) examples, mean weight 1."""
    if p_empty is None or not 0.0 < p_empty < 1.0:
        return torch.ones_like(empty)
    w_empty = 0.5 / p_empty
    w_full = 0.5 / (1.0 - p_empty)
    return torch.where(empty > 0.5, torch.full_like(empty, w_empty), torch.full_like(empty, w_full))


def jepa_loss(
    preds: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    counts: Dict[str, Tensor],
    kind: str = "smooth_l1",
    p_empty: Optional[Dict[str, float]] = None,
    stop_grad: bool = True,
    reg_weight: float = 1.0,
    bce_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """Compute L_JEPA and its parts. ``targets[k]`` are pooled means (before LayerNorm).

    ``stop_grad=False`` is only for the shared-encoder variants (SIGReg / no anti-collapse).
    """
    total = 0.0
    logs: Dict[str, Tensor] = {}
    for k, out in preds.items():
        y_hat, e_logit = out[..., :-1], out[..., -1]
        empty = (counts[k] == 0).float()
        y = F.layer_norm(targets[k], targets[k].shape[-1:])
        if stop_grad:
            y = y.detach()                                                  # sg(y_k)
        w = empty_weights(empty, None if p_empty is None else p_empty.get(k))
        reg = regression_loss(y_hat, y, kind) * (1.0 - empty)
        bce = F.binary_cross_entropy_with_logits(e_logit, empty, reduction="none")
        loss_k = (w * (reg_weight * reg + bce_weight * bce)).sum() / w.sum()
        total = total + loss_k
        logs[f"reg_{k}"] = ((w * reg).sum() / (w * (1.0 - empty)).sum().clamp(min=1e-8)).detach()
        logs[f"bce_{k}"] = ((w * bce).sum() / w.sum()).detach()
        logs[f"empty_rate_{k}"] = empty.mean().detach()
    logs["jepa_loss"] = total.detach() if isinstance(total, Tensor) else torch.tensor(total)
    return {"loss": total, **logs}
