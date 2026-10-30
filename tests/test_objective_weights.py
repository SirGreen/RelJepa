"""Empty-head ablation: reg_weight / bce_weight switch off one term of L_JEPA."""
import torch

from reljepa.objective import jepa_loss


def _batch(seed=0, n=16, d=8):
    g = torch.Generator().manual_seed(seed)
    preds = {"transactions": torch.randn(n, d + 1, generator=g, requires_grad=True)}
    targets = {"transactions": torch.randn(n, d, generator=g)}
    counts = {"transactions": torch.tensor([0, 3] * (n // 2))}
    return preds, targets, counts


def test_default_weights_match_sum_of_terms():
    preds, targets, counts = _batch()
    p_empty = {"transactions": 0.5}
    full = jepa_loss(preds, targets, counts, kind="cosine", p_empty=p_empty)["loss"]
    reg = jepa_loss(preds, targets, counts, kind="cosine", p_empty=p_empty, bce_weight=0.0)["loss"]
    bce = jepa_loss(preds, targets, counts, kind="cosine", p_empty=p_empty, reg_weight=0.0)["loss"]
    assert torch.allclose(full, reg + bce)


def test_zero_weight_removes_the_gradient_of_that_term():
    preds, targets, counts = _batch()
    jepa_loss(preds, targets, counts, kind="cosine", bce_weight=0.0)["loss"].backward()
    grad = preds["transactions"].grad
    assert torch.all(grad[:, -1] == 0)          # emptiness logit gets no gradient
    assert grad[:, :-1].abs().sum() > 0
    preds, targets, counts = _batch()
    jepa_loss(preds, targets, counts, kind="cosine", reg_weight=0.0)["loss"].backward()
    grad = preds["transactions"].grad
    assert torch.all(grad[:, :-1] == 0)         # predicted target gets no gradient
    assert grad[:, -1].abs().sum() > 0
