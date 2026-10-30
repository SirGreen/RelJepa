"""Train/eval gap diagnostic for RelJEPA.

At diagnostic steps the same (context, target) batch is re-scored three ways, without gradients:
  * "train"   : training mode (input masking + dropout), as in the optimisation step,
  * "nomask"  : training mode with input masking disabled (dropout only),
  * "eval"    : eval mode (no masking, no dropout).
The regression part of L_JEPA is reported for each, together with a "mean" reference: the loss of
predicting the batch-mean target for every non-empty example. This tells whether the context
embedding carries next-window information and how much of it the corruptions hide.
"""
from typing import Dict

import torch
import torch.nn.functional as F

from reljepa.objective import regression_loss


@torch.no_grad()
def _reg_losses(out, kind: str) -> Dict[str, float]:
    res = {}
    for k, pred in out["preds"].items():
        nonempty = out["counts"][k] > 0
        if nonempty.sum() < 2:
            continue
        y = F.layer_norm(out["targets"][k][nonempty], out["targets"][k].shape[-1:])
        y_hat = pred[nonempty][..., :-1]
        res[k] = regression_loss(y_hat, y, kind).mean().item()
        res[f"{k}_meanref"] = regression_loss(y.mean(0, keepdim=True).expand_as(y), y, kind).mean().item()
    return res


@torch.no_grad()
def mode_gap_report(model, ctx, tgt, root: str, delta: int, kind: str) -> Dict[str, float]:
    """Re-score clones of one batch in three modes; leaves the model in training mode."""
    report = {}
    was_training = model.training
    rate = model.mask_rate
    try:
        model.train()
        out = model(ctx.clone(), tgt.clone(), root, delta)
        report.update({f"gap_train_{k}": v for k, v in _reg_losses(out, kind).items()})
        model.mask_rate = 0.0
        out = model(ctx.clone(), tgt.clone(), root, delta)
        report.update({f"gap_nomask_{k}": v for k, v in _reg_losses(out, kind).items() if "meanref" not in k})
        model.mask_rate = rate
        model.eval()
        out = model(ctx.clone(), tgt.clone(), root, delta)
        report.update({f"gap_eval_{k}": v for k, v in _reg_losses(out, kind).items() if "meanref" not in k})
    finally:
        model.mask_rate = rate
        model.train(was_training)
    return report
