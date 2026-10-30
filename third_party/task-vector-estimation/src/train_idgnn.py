"""ID-GNN fine-tuning for rel-hm user-item-purchase (port of RelBench v1.1.0 examples/idgnn_link.py
onto TVE's backbone so that pretrained backbone checkpoints can be evaluated).

Differences from the RelBench example, all shared by every method: TVE's backbone (mean
aggregation, LayerNorm, residual GraphSAGE), the optional low-label setting (a fixed random
fraction of training users), and multiple seeds per process. Best epoch is chosen by validation
MAP; test is evaluated once with that epoch's weights.

Usage (container, TVE root):
  python src/train_idgnn.py -c /workspace/configs/rel-hm/ft-user-item-purchase-128.yaml \
      --num_runs 3 --num_workers 3 --gpus '[0]' --name <n> --note <n> [--checkpoint <backbone.pth>]
"""
import copy
import json
import os
import signal

import numpy as np
import rootutils
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm.rich import tqdm

import warnings
warnings.filterwarnings("ignore")

import wandb

from relbench.base import RecommendationTask, Table
from relbench.modeling.graph import get_link_train_table_input
from relbench.modeling.loader import SparseTensor

rootutils.setup_root('.', indicator='.project-root', pythonpath=True)

from src.helpers.dataset_helper import get_data
from src.helpers.model_helper import get_conv_layer, get_hetero_encoder
from src.helpers.save_helper import load_backbone_checkpoint
from src.models.idgnn import IDGNN
from src.utils.io_utils import (
    parse_cfg_args, parse_train_args, load_config, get_device, create_log_dir, print_config_tree,
    handle_exit, init_process_group, destroy_process_group, setup_cfg, setup_wandb,
    infer_dataset_task, sanity_check,
)
from src.utils.io_utils import rank_zero_print as print
from src.utils.random_utils import set_random_seed
from src.utils.wandb_utils import set_wandb_directories

set_wandb_directories()
TUNE_METRIC = "link_prediction_map"


def subsample_users(table: Table, task: RecommendationTask, fraction: float, seed: int) -> Table:
    """Low-label setting: keep all training rows of a fixed random `fraction` of training users."""
    if fraction >= 1.0:
        return table
    users = np.sort(table.df[task.src_entity_col].unique())
    rng = np.random.default_rng(seed)
    keep = rng.choice(users, size=int(round(fraction * len(users))), replace=False)
    df = table.df[table.df[task.src_entity_col].isin(keep)].reset_index(drop=True)
    return Table(df=df, fkey_col_to_pkey_table=table.fkey_col_to_pkey_table, pkey_col=table.pkey_col,
                 time_col=table.time_col)


def make_loader(cfg, data, table, task, shuffle):
    table_input = get_link_train_table_input(table, task)
    loader = NeighborLoader(
        data,
        num_neighbors=cfg.train.num_neighbors,
        time_attr="time",
        input_nodes=table_input.src_nodes,
        input_time=table_input.src_time,
        subgraph_type="bidirectional",
        batch_size=cfg.train.batch_size,
        temporal_strategy=cfg.train.temporal_strategy,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        persistent_workers=cfg.train.num_workers > 0,
    )
    return loader, table_input


def train_epoch(cfg, model, task, loader, sparse, optimizer, device):
    model.train()
    src, dst = task.src_entity_table, task.dst_entity_table
    loss_sum = torch.zeros((), device=device)
    count = 0
    steps = 0
    total = min(len(loader), cfg.train.max_steps_per_epoch)
    for batch in tqdm(loader, total=total, desc="ID-GNN training"):
        batch = batch.to(device)
        out = model.forward_dst_readout(batch, src, dst).flatten()
        bs = batch[src].batch_size
        src_batch, dst_index = sparse[batch[src].input_id]
        target = torch.isin(batch[dst].batch + bs * batch[dst].n_id, src_batch + bs * dst_index).float()
        loss = F.binary_cross_entropy_with_logits(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach() * out.numel()
        count += out.numel()
        steps += 1
        if steps > cfg.train.max_steps_per_epoch:   # same (+1) step count as TVE's loops
            break
    return float(loss_sum) / max(count, 1)


@torch.no_grad()
def predict(model, task, loader, device):
    model.eval()
    src, dst = task.src_entity_table, task.dst_entity_table
    preds = []
    for batch in tqdm(loader, desc="ID-GNN eval"):
        batch = batch.to(device)
        out = model.forward_dst_readout(batch, src, dst).flatten()
        bs = batch[src].batch_size
        scores = torch.zeros(bs, task.num_dst_nodes, device=device)
        scores[batch[dst].batch, batch[dst].n_id] = torch.sigmoid(out)
        preds.append(torch.topk(scores, k=task.eval_k, dim=1).indices)
    return torch.cat(preds, dim=0).cpu().numpy()


def save_test_ap(log_dir, run, task, pred, metrics):
    """Per-user test AP for paired bootstrap intervals; never fails the run."""
    try:
        from reljepa.rec_metrics import per_user_ap
        table = task.get_table("test", mask_input_cols=False)
        ap = per_user_ap(table.df[task.dst_entity_col].tolist(), pred)
        np.save(os.path.join(log_dir, f"run{run + 1}-best-test-ap.npy"), ap)
        if not np.isclose(ap.mean(), metrics[TUNE_METRIC], rtol=0, atol=1e-9):
            print(f"WARNING: mean per-user AP {ap.mean():.6f} != test MAP {metrics[TUNE_METRIC]:.6f}")
    except Exception as e:  # noqa: BLE001 - a diagnostic must not lose a finished run
        print(f"WARNING: per-user test AP not saved: {e!r}")


def main(cfg, cfg_file):
    device = get_device(cfg)
    setup_cfg(cfg)
    log_dir = create_log_dir(cfg, cfg_file, "")
    setup_wandb(cfg, log_dir)
    print_config_tree(cfg, save_to_file=True, output_dir=log_dir)

    data_dict = get_data(cfg, device)
    task = data_dict["task"]
    assert isinstance(task, RecommendationTask)
    train_table = subsample_users(data_dict["train_table"], task, cfg.rec.train_user_fraction,
                                  cfg.rec.subsample_seed)
    print(f"Training rows: {len(train_table.df)} (user fraction {cfg.rec.train_user_fraction})")
    train_loader, train_input = make_loader(cfg, data_dict["data"], train_table, task, True)
    val_loader, _ = make_loader(cfg, data_dict["data"], data_dict["val_table"], task, False)
    test_loader, _ = make_loader(cfg, data_dict["data"], data_dict["test_table"], task, False)
    sparse = SparseTensor(train_input.dst_nodes[1], device=device)

    per_seed = {"val": {}, "test": {}}
    curves = []
    for run in range(cfg.num_runs):
        set_random_seed(cfg.seed + run)
        model = IDGNN(
            data=data_dict["data"], schema_graph=data_dict["schema_graph"],
            schema_line_graph=data_dict["schema_line_graph"], out_channels=1,
            col_stats_dict=data_dict["col_stats_dict"], hetero_encoder_cls=get_hetero_encoder(cfg),
            hetero_encoder_kwargs={"out_channels": cfg.model.conv.kwargs.channels, **cfg.model.hetero_encoder.kwargs},
            temporal_encoder_kwargs={"out_channels": cfg.model.conv.kwargs.channels, **cfg.model.temporal_encoder.kwargs},
            conv_cls=get_conv_layer(cfg),
            conv_kwargs={"num_layers": len(cfg.train.num_neighbors), **cfg.model.conv.kwargs},
            shallow_list=cfg.model.shallow_list,
        ).to(device)
        if cfg.checkpoint:
            load_backbone_checkpoint(model, None, cfg.checkpoint, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

        best_val, best_state, best_epoch, run_curve = -1.0, None, -1, []
        for epoch in range(cfg.train.num_epochs):
            loss = train_epoch(cfg, model, task, train_loader, sparse, optimizer, device)
            val_metrics = task.evaluate(predict(model, task, val_loader, device), data_dict["val_table"])
            run_curve.append({"epoch": epoch + 1, "train_loss": loss, **{f"val_{k}": float(v) for k, v in val_metrics.items()}})
            print(f"run {run+1} epoch {epoch+1}: loss {loss:.4f} val {val_metrics}")
            if val_metrics[TUNE_METRIC] > best_val:
                best_val, best_epoch = val_metrics[TUNE_METRIC], epoch + 1
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        val_metrics = task.evaluate(predict(model, task, val_loader, device), data_dict["val_table"])
        test_pred = predict(model, task, test_loader, device)
        test_metrics = task.evaluate(test_pred)
        save_test_ap(log_dir, run, task, test_pred, test_metrics)   # RelJEPA: paired bootstrap
        print(f"run {run+1}: best epoch {best_epoch}, val {val_metrics}, test {test_metrics}")
        for split, metrics in [("val", val_metrics), ("test", test_metrics)]:
            for k, v in metrics.items():
                per_seed[split].setdefault(k, []).append(float(v))
        curves.append({"run": run + 1, "best_epoch": best_epoch, "curve": run_curve})

    record = {
        "dataset": cfg.dataset.cls, "task": cfg.dataset.task, "name": cfg.name, "note": cfg.note,
        "git_commit": cfg.get("sha", "unknown"), "checkpoint": cfg.checkpoint, "lr": cfg.optimizer.lr,
        "channels": cfg.model.conv.kwargs.channels, "batch_size": cfg.train.batch_size,
        "num_epochs": cfg.train.num_epochs, "max_steps_per_epoch": cfg.train.max_steps_per_epoch,
        "train_user_fraction": cfg.rec.train_user_fraction, "seeds": [cfg.seed + r for r in range(cfg.num_runs)],
        "wandb_run_id": getattr(wandb.run, "id", None), "log_dir": log_dir, "per_seed": per_seed,
    }
    with open(os.path.join(log_dir, "per_seed_results.json"), "w") as f:
        json.dump(record, f, indent=1)
    with open(os.path.join(log_dir, "curves.json"), "w") as f:
        json.dump(curves, f, indent=1)
    print(json.dumps(per_seed, indent=1))
    wandb.finish()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_exit)
    args, unparsed = parse_cfg_args()
    args, vars = parse_train_args(args, unparsed)
    cfg = load_config(args.config, context=vars)
    cfg.sha = os.environ.get("GIT_COMMIT", "unknown")
    sanity_check(cfg)
    cfg.sweep = False
    cfg.dataset.cls, cfg.dataset.task = infer_dataset_task(cfg, args.config)
    cfg.model.conv.kwargs.num_layers = len(cfg.train.num_neighbors)
    init_process_group(cfg)
    main(cfg, args.config)
    destroy_process_group()
