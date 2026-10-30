"""Paired loaders and train/eval loops for RelJEPA."""
import csv
import math
import os
import time
from typing import Dict, Optional

import torch
from torch_geometric.loader import NeighborLoader
from tqdm.rich import tqdm

from src.helpers.database_helper import get_node_train_table_input
from src.utils.io_utils import get_rank, get_world_size

from reljepa.diagnostics import collapse_report
from reljepa.ema import ema_momentum, ema_update
from reljepa.mode_gap import mode_gap_report
from reljepa.sigreg import SIGReg
from reljepa.objective import jepa_loss
from reljepa.target_view import SECONDS_PER_DAY


class CollapseError(RuntimeError):
    """Raised when the context embeddings collapse."""


def get_paired_loaders(cfg, data_dict):
    """For each split, a context loader (seed time t) and a target loader (seed time t + Δ).

    Both iterate the same seeds in the same order (identically seeded samplers); the training loop
    asserts this on every batch.
    """
    task = data_dict["task"]
    data = data_dict["data"]
    delta = int(cfg.jepa.delta_days * SECONDS_PER_DAY)
    loaders = {}
    for split in ["train", "val", "test"]:
        table_input = get_node_train_table_input(table=data_dict[f"{split}_table"], task=task)
        shuffle = split == "train"

        def make(input_time, strategy, transform, num_workers):
            sampler = torch.utils.data.distributed.DistributedSampler(
                table_input.nodes[1], num_replicas=get_world_size(), rank=get_rank(),
                shuffle=shuffle, seed=cfg.seed,
            )
            return NeighborLoader(
                data,
                num_neighbors=cfg.train.num_neighbors,
                time_attr="time",
                input_nodes=table_input.nodes,
                input_time=input_time,
                subgraph_type=cfg.train.subgraph_type,
                transform=transform,
                batch_size=cfg.train.batch_size,
                temporal_strategy=strategy,
                sampler=sampler,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
            )

        # Val/test loaders run only every few epochs: fewer (default: no) workers saves RAM.
        ctx_w = cfg.train.num_workers if shuffle else int(os.environ.get("EVAL_NUM_WORKERS", 0))
        tgt_w = cfg.jepa.target_num_workers if shuffle else int(os.environ.get("EVAL_NUM_WORKERS", 0))
        ctx = make(table_input.time, cfg.train.temporal_strategy, table_input.transform, ctx_w)
        tgt = make(table_input.time + delta, cfg.jepa.target_temporal_strategy, None, tgt_w)
        loaders[split] = (ctx, tgt)
    return loaders, delta


def _check_pair(ctx, tgt, root, delta):
    bs = ctx[root].batch_size
    assert tgt[root].batch_size == bs
    assert torch.equal(ctx[root].n_id[:bs], tgt[root].n_id[:bs]), "context/target seeds differ"
    assert torch.equal(tgt[root].seed_time, ctx[root].seed_time + delta), "target seed time != t + Δ"


_SIGREG = {}


def compute_losses(cfg, out, ctx, root, tve_loss_fn, p_empty):
    losses = jepa_loss(
        out["preds"], out["targets"], out["counts"],
        kind=cfg.jepa.loss, p_empty=p_empty, stop_grad=cfg.jepa.use_ema,
        reg_weight=float(cfg.jepa.get("reg_weight", 1.0)), bce_weight=float(cfg.jepa.get("bce_weight", 1.0)),
    )
    loss = losses["loss"]
    lam = float(cfg.jepa.get("lambda_sigreg", 0.0))
    if lam > 0:   # RelJEPA-SIGReg: (1 - λ) L_JEPA + λ SIGReg(z_ctx)
        device = out["z_ctx"].device
        if device not in _SIGREG:
            _SIGREG[device] = SIGReg().to(device)
        sig = _SIGREG[device](out["z_ctx"])
        losses["sigreg"] = sig.detach()
        loss = (1.0 - lam) * loss + lam * sig
    if cfg.jepa.beta_tve > 0:
        tve = tve_loss_fn(out["tve"], ctx[root].y.float())
        losses["tve_loss"] = tve.detach()
        loss = loss + cfg.jepa.beta_tve * tve
    losses["total_loss"] = loss.detach()
    return loss, losses


def set_epoch(loaders, epoch):
    for loader in loaders:
        if isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
            loader.sampler.set_epoch(epoch)


def train_one_epoch(cfg, model, task, loaders, optimizer, tve_loss_fn, p_empty, delta, device,
                    state: Dict, diag_path: Optional[str] = None):
    """One epoch of RelJEPA. ``state`` carries the global step and total steps for the EMA schedule."""
    model.train()
    root = task.entity_table
    ctx_loader, tgt_loader = loaders
    total_steps_epoch = min(len(ctx_loader), cfg.train.max_steps_per_epoch)
    sums: Dict[str, float] = {}
    count = 0
    steps = 0
    t0 = time.time()
    for ctx, tgt in tqdm(zip(ctx_loader, tgt_loader), total=total_steps_epoch, desc="RelJEPA training"):
        _check_pair(ctx, tgt, root, delta)
        bs = ctx[root].batch_size
        ctx, tgt = ctx.to(device), tgt.to(device)
        diag_step = (state["step"] + 1) % cfg.jepa.diag_every == 0 or steps == 0
        if diag_step:   # keep untouched copies for the train/eval gap diagnostic
            ctx_copy, tgt_copy = ctx.clone(), tgt.clone()
        out = model(ctx, tgt, root, delta)
        loss, losses = compute_losses(cfg, out, ctx, root, tve_loss_fn, p_empty)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state["step"] += 1
        tau = ema_momentum(state["step"], state["total_steps"], cfg.jepa.tau_start, 1.0)
        if model.target is not None:
            ema_update(model.target, torch.nn.ModuleDict(model.context_modules()), tau)

        # Accumulate on the GPU; host syncs only at diagnostic steps (keeps CPU/GPU overlap).
        for k, v in losses.items():
            if k != "loss":
                sums[k] = sums.get(k, 0.0) + v * bs
        count += bs
        steps += 1

        if diag_step:
            report = collapse_report(out["z_ctx"].detach(), {k: v.detach() for k, v in out["targets"].items()},
                                     out["counts"])
            report.update(mode_gap_report(model, ctx_copy, tgt_copy, root, delta, cfg.jepa.loss))
            del ctx_copy, tgt_copy
            row = {"step": state["step"], "tau": tau, "sec_per_step": (time.time() - t0) / steps,
                   **{k: float(v) for k, v in losses.items() if k != "loss"}, **report}
            if diag_path is not None:
                new = not os.path.exists(diag_path)
                with open(diag_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if new:
                        writer.writeheader()
                    writer.writerow(row)
            if report["z_std"] < cfg.jepa.collapse_std:
                raise CollapseError(f"collapse at step {state['step']}: mean per-dim std(z_ctx) = {report['z_std']:.2e}")

        if steps > cfg.train.max_steps_per_epoch:   # same (+1) step count as TVE's loops
            break
    return {k: float(v) / count for k, v in sums.items()}


@torch.no_grad()
def evaluate(cfg, model, task, loaders, tve_loss_fn, p_empty, delta, device):
    """Mean losses on a held-out pretraining split."""
    model.eval()
    root = task.entity_table
    ctx_loader, tgt_loader = loaders
    sums: Dict[str, float] = {}
    count = 0
    for ctx, tgt in zip(ctx_loader, tgt_loader):
        _check_pair(ctx, tgt, root, delta)
        bs = ctx[root].batch_size
        ctx, tgt = ctx.to(device), tgt.to(device)
        out = model(ctx, tgt, root, delta)
        _, losses = compute_losses(cfg, out, ctx, root, tve_loss_fn, p_empty)
        for k, v in losses.items():
            if k != "loss":
                sums[k] = sums.get(k, 0.0) + v * bs
        count += bs
    return {k: float(v) / count for k, v in sums.items()}
