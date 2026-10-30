"""Frozen customer embeddings for the information probes.

For each method's backbone checkpoint (or a random initialisation for "scratch"), embeds every
probe pair (customer, t) from its temporal neighbourhood at seed time t, exactly as fine-tuning
would see it, in eval mode. Saves data/probes/emb_<method>.npy (rows aligned with probe_set.parquet).

Run inside the container from the TVE root:
  python /workspace/scripts/probes/extract_embeddings.py --methods scratch tve1 reljepa ...
Checkpoint paths are read from /workspace/logs/main/logdir_pt-user-<method>.txt; for the short
ablations, method abl-<variant> reads
/workspace/logs/ablations/logdir_abl-user-reljepa-<variant>.txt.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import NeighborLoader

sys.path.insert(0, "/workspace/tests")
from conftest import load_cfg  # noqa: E402
from relbench.modeling.utils import to_unix_time  # noqa: E402
from src.helpers.dataset_helper import get_data  # noqa: E402
from src.helpers.model_helper import get_conv_layer, get_hetero_encoder  # noqa: E402
from src.helpers.save_helper import load_backbone_checkpoint  # noqa: E402
from src.models.baseline import Baseline  # noqa: E402
from src.utils.random_utils import set_random_seed  # noqa: E402

CKPT_FILE = {"mae": "epoch81.pth", "tve1": "epoch81.pth", "reljepa": "run1-final.pth", "reljepa-tve": "run1-final.pth",
             "tve1-empty": "run1-final.pth"}   # tve1-empty: TVE-1 + empty-window head control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["scratch", "mae", "tve1", "reljepa", "reljepa-tve"])
    ap.add_argument("--root", default="user")
    args = ap.parse_args()

    cfg = load_cfg(path="/workspace/configs/rel-hm/ft-user-churn-128.yaml", batch_size=512, task="user-churn")
    cfg.train.num_workers = 3
    device = torch.device("cuda")
    data_dict = get_data(cfg, device)
    probe = pd.read_parquet("data/probes/probe_set.parquet")
    nodes = torch.from_numpy(probe["customer_id"].to_numpy().astype(np.int64))
    time = torch.from_numpy(to_unix_time(probe["timestamp"]))

    for method in args.methods:
        set_random_seed(43)
        model = Baseline(
            data=data_dict["data"], schema_graph=data_dict["schema_graph"],
            schema_line_graph=data_dict["schema_line_graph"], out_channels=1,
            col_stats_dict=data_dict["col_stats_dict"], hetero_encoder_cls=get_hetero_encoder(cfg),
            hetero_encoder_kwargs={"out_channels": 128, **cfg.model.hetero_encoder.kwargs},
            temporal_encoder_kwargs={"out_channels": 128, **cfg.model.temporal_encoder.kwargs},
            conv_cls=get_conv_layer(cfg), conv_kwargs={"num_layers": 2, **cfg.model.conv.kwargs},
        ).to(device)
        if method.startswith("abl-"):
            with open(f"/workspace/logs/ablations/logdir_abl-{args.root}-reljepa-{method[4:]}.txt") as f:
                ckpt = os.path.join(f.read().strip(), "run1-final.pth")
            load_backbone_checkpoint(model, None, ckpt, device)
        elif method != "scratch":
            with open(f"/workspace/logs/main/logdir_pt-{args.root}-{method}.txt") as f:
                ckpt = os.path.join(f.read().strip(), CKPT_FILE[method])
            load_backbone_checkpoint(model, None, ckpt, device)
        model.eval()
        torch.manual_seed(0)       # same neighbour samples for every method
        loader = NeighborLoader(
            data_dict["data"], num_neighbors=cfg.train.num_neighbors, time_attr="time",
            input_nodes=("customer", nodes), input_time=time, subgraph_type=cfg.train.subgraph_type,
            batch_size=512, temporal_strategy=cfg.train.temporal_strategy, shuffle=False,
            num_workers=cfg.train.num_workers,
        )
        embs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                bs = batch["customer"].batch_size
                x = model.encoder(batch.tf_dict)
                rel = model.temporal_encoder(batch["customer"].seed_time, batch.time_dict, batch.batch_dict)
                for nt, r in rel.items():
                    x[nt] = x[nt] + r
                x = model.gnn(x, batch.edge_index_dict)
                embs.append(x["customer"][:bs].float().cpu())
        emb = torch.cat(embs).numpy()
        np.save(f"data/probes/emb_{method}.npy", emb)
        print(method, emb.shape, "mean per-dim std", float(emb.std(0).mean()))


if __name__ == "__main__":
    main()
