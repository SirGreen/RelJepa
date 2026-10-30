"""Fine-tuning additions for the RelJEPA study (all switched by environment variables so that
TVE's default behaviour is unchanged):

- FREEZE_BACKBONE=1: linear probing. The pretrained backbone (encoder, temporal encoder, GNN,
  shallow embeddings) is frozen and kept in eval mode; only the linear head is trained (TVE's
  linear-probing protocol).
- Per-example predictions at each seed's best-validation epoch are saved for bootstrap intervals
, together with the val/test labels.
- SAVE_FT_CHECKPOINTS=0: skip writing fine-tuned model checkpoints (disk space on the laptop).
"""
import os

import numpy as np

_BACKBONE = ("encoder", "temporal_encoder", "gnn", "embedding_dict")


def probing_enabled() -> bool:
    return os.environ.get("FREEZE_BACKBONE", "0") == "1"


def save_checkpoints_enabled() -> bool:
    return os.environ.get("SAVE_FT_CHECKPOINTS", "1") == "1"


def freeze_backbone(model):
    """Freeze the backbone and keep it in eval mode even when model.train() is called."""
    frozen = [getattr(model, name) for name in _BACKBONE]
    for module in frozen:
        for p in module.parameters():
            p.requires_grad_(False)
    original_train = model.train

    def train(mode: bool = True):
        original_train(mode)
        for module in frozen:
            module.eval()
        return model

    model.train = train
    return model


def save_labels(log_dir, task, data_dict):
    """Save val/test labels once per run directory (same order as the predictions)."""
    for split in ("val", "test"):
        path = os.path.join(log_dir, f"{split}-labels.npy")
        if not os.path.exists(path):
            np.save(path, data_dict[f"{split}_table"].df[task.target_col].to_numpy())


def save_predictions(log_dir, run, predictions):
    """predictions: {'val': array, 'test': array} from the best-validation epoch."""
    for split, pred in predictions.items():
        if pred is not None:
            np.save(os.path.join(log_dir, f"run{run+1}-best-{split}-pred.npy"), np.asarray(pred))
