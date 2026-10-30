"""Target view of RelJEPA.

For a seed (v, t) the target view T(v, t, Δ) contains the fact rows reachable from v whose timestamps
lie in (t, t + Δ], plus the dimension rows they link to. We build it from a mini-batch sampled by the
RelBench temporal neighbour sampler at seed time t + Δ (so only rows with timestamp <= t + Δ are ever
sampled) and then

1. drop fact-table nodes (node types with a ``time`` attribute) whose timestamp is <= t,
2. drop dimension nodes that are no longer linked to a kept fact node,
3. cut every edge incident to the seed node v itself, so that the root's own features cannot flow
   into the target representations (the predictor must not be able to copy v's attributes).

The seed nodes stay in the node set (isolated) so that ``seed_time`` and the root node store keep
their sizes; their outputs are never pooled into targets.
"""
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

SECONDS_PER_DAY = 24 * 60 * 60


def in_window(node_time: Tensor, window_end: Tensor, delta_seconds: int) -> Tensor:
    """Mask of rows with window_end - Δ < time <= window_end (window_end = t + Δ per row)."""
    return (node_time > window_end - delta_seconds) & (node_time <= window_end)


def _subset_node_store(store, keep: Tensor) -> None:
    """Index every per-node attribute of a node store in place."""
    num_nodes = keep.numel()
    for key in list(store.keys()):
        value = store[key]
        if key in ("batch_size", "seed_time", "input_id"):
            continue
        if isinstance(value, Tensor) and value.dim() > 0 and value.size(0) == num_nodes:
            store[key] = value[keep]
        elif hasattr(value, "num_rows") and value.num_rows == num_nodes:   # torch_frame.TensorFrame
            store[key] = value[keep.nonzero().view(-1)]


def make_target_view(
    batch: HeteroData,
    root: str,
    delta_seconds: int,
) -> Tuple[HeteroData, Dict[str, Tensor]]:
    """Filter a batch sampled at seed time t + Δ down to the target view.

    Args:
        batch: mini-batch from ``NeighborLoader`` with ``input_time = t + Δ`` (disjoint temporal
            sampling, so every node store has a ``batch`` vector mapping nodes to seeds).
        root: node type of the seed entities.
        delta_seconds: window length Δ in seconds.

    Returns:
        The filtered batch (modified in place and returned) and, per node type, the boolean keep
        mask over the original nodes (for diagnostics and tests).
    """
    batch_size = batch[root].batch_size
    window_end = batch[root].seed_time  # = t + Δ, shape [batch_size]
    seed_ids = torch.arange(batch_size, device=window_end.device)

    keep: Dict[str, Tensor] = {}
    # 1. fact rows: keep only the window (t, t + Δ].
    for node_type in batch.node_types:
        store = batch[node_type]
        n = store.num_nodes
        if node_type == root:
            keep[node_type] = torch.ones(n, dtype=torch.bool, device=window_end.device)
        elif "time" in store:
            keep[node_type] = in_window(store.time, window_end[store.batch], delta_seconds)
        else:
            keep[node_type] = torch.zeros(n, dtype=torch.bool, device=window_end.device)

    # 2. dimension rows: keep those linked to a kept fact row (edges in either direction).
    fact_types = {nt for nt in batch.node_types if nt != root and "time" in batch[nt]}
    for (src, _, dst), edge_index in batch.edge_index_dict.items():
        if src in fact_types and dst not in fact_types and dst != root:
            linked = edge_index[1][keep[src][edge_index[0]]]
            keep[dst][linked] = True
        elif dst in fact_types and src not in fact_types and src != root:
            linked = edge_index[0][keep[dst][edge_index[1]]]
            keep[src][linked] = True

    # 3. remap edges to the kept nodes and cut edges touching the seed nodes.
    new_index: Dict[str, Tensor] = {}
    for node_type, mask in keep.items():
        mapping = torch.full((mask.numel(),), -1, dtype=torch.long, device=mask.device)
        mapping[mask] = torch.arange(int(mask.sum()), device=mask.device)
        new_index[node_type] = mapping
    is_seed = torch.zeros(batch[root].num_nodes, dtype=torch.bool, device=window_end.device)
    is_seed[seed_ids] = True
    for edge_type in list(batch.edge_types):
        src, _, dst = edge_type
        edge_index = batch[edge_type].edge_index
        ok = keep[src][edge_index[0]] & keep[dst][edge_index[1]]
        if src == root:
            ok &= ~is_seed[edge_index[0]]
        if dst == root:
            ok &= ~is_seed[edge_index[1]]
        edge_index = edge_index[:, ok]
        batch[edge_type].edge_index = torch.stack(
            [new_index[src][edge_index[0]], new_index[dst][edge_index[1]]], dim=0
        )
        for key in list(batch[edge_type].keys()):   # drop per-edge attributes we did not remap
            if key != "edge_index":
                del batch[edge_type][key]

    for node_type, mask in keep.items():
        if node_type != root:
            _subset_node_store(batch[node_type], mask)
    return batch, keep


def pool_targets(
    h_dict: Dict[str, Tensor],
    batch: HeteroData,
    root: str,
    target_types,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Mean-pool target-encoder outputs per seed and node type.

    Returns ``means[k]`` of shape [batch_size, d] (zeros for empty windows) and ``counts[k]`` of
    shape [batch_size] (number of in-window rows of type k per seed).
    """
    batch_size = batch[root].batch_size
    means, counts = {}, {}
    for k in target_types:
        h = h_dict[k]
        idx = batch[k].batch
        total = torch.zeros(batch_size, h.size(-1), device=h.device, dtype=h.dtype)
        total.index_add_(0, idx, h)
        cnt = torch.zeros(batch_size, device=h.device, dtype=h.dtype)
        cnt.index_add_(0, idx, torch.ones_like(idx, dtype=h.dtype))
        means[k] = total / cnt.clamp(min=1).unsqueeze(-1)
        counts[k] = cnt
    return means, counts
