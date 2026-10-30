"""Per-user recommendation metrics for paired bootstrap intervals.

RelBench 1.1.0 computes MAP@k as the mean, over users with at least one true item, of
    AP_u = sum_j precision@j * hit_j / min(#true items of u, k)
(`relbench.metrics.link_prediction_map` after `_filter`). `per_user_ap` returns those AP_u in table
order, so `per_user_ap(...).mean()` equals RelBench's MAP and the same users are kept for every
method (the filter depends on the labels only).
"""
from typing import Sequence

import numpy as np


def per_user_ap(true_items: Sequence, pred: np.ndarray) -> np.ndarray:
    """AP@k per user with >= 1 true item. ``true_items[i]``: array-like of item ids of user i;
    ``pred``: [n_users, k] predicted item ids (best first), aligned with ``true_items``."""
    pred = np.asarray(pred)
    if len(true_items) != pred.shape[0]:
        raise ValueError(f"{len(true_items)} label rows vs {pred.shape[0]} prediction rows")
    isin = np.stack([np.isin(p, np.asarray(t)) for t, p in zip(true_items, pred)])
    count = np.array([len(t) for t in true_items])
    keep = count > 0
    isin, count = isin[keep], count[keep]
    k = isin.shape[1]
    precision = np.cumsum(isin, axis=1) / (np.arange(k) + 1)
    return (precision * isin).sum(axis=1) / np.minimum(count, k)
