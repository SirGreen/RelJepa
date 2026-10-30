"""Build the shared pretraining tables used by every pretraining method.

From TVE's released *training* table of a root (e.g. rel-hm user-tve-1-hop) draw, with a fixed seed,
  - `train`: N_train random (entity, t) rows,
  - `val`, `test`: N_hold rows each, disjoint from `train` and from each other,
all from the training period (every window (t, t + Δ] ends on or before the validation cutoff).
The TVE 1-hop task-vector columns are kept so TVE, MAE, RelJEPA and RelJEPA+TVE read the same rows
(methods that do not use task vectors ignore those columns).

Run inside the container from the TVE root, e.g.
  python /workspace/scripts/data/make_pretrain_subsample.py --root user --n_train 2000000
Writes data/relbench/rel-hm/tasks/<root>-pt-tve-1-hop/{train,val,test}.parquet and a JSON manifest
to /workspace/results/raw/pretrain_subsample_<root>.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from relbench.base import Table

VAL_TS = pd.Timestamp("2020-09-07")
DELTA = pd.Timedelta(days=7)
TASK_DIR = "data/relbench/rel-hm/tasks"
ENTITY = {"user": ("customer_id", "customer"), "item": ("article_id", "article")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", choices=["user", "item"], required=True)
    parser.add_argument("--n_train", type=int, required=True)
    parser.add_argument("--n_hold", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    src = os.path.join(TASK_DIR, f"{args.root}-tve-1-hop", "train.parquet")
    # Memory-light: keep the Arrow table, check timestamps column-wise, `take` only selected rows.
    table = pq.read_table(src)
    ts = pd.to_datetime(table.column("timestamp").to_pandas())
    assert (ts + DELTA <= VAL_TS).all(), "leakage: a pretraining window ends after the validation cutoff"
    del ts

    rng = np.random.default_rng(args.seed)
    n_total = table.num_rows
    n_need = args.n_train + 2 * args.n_hold
    assert n_need <= n_total, f"requested {n_need} rows but the table has {n_total}"
    perm = rng.permutation(n_total)[:n_need]
    parts = {
        "train": np.sort(perm[: args.n_train]),
        "val": np.sort(perm[args.n_train: args.n_train + args.n_hold]),
        "test": np.sort(perm[args.n_train + args.n_hold:]),
    }

    out_dir = os.path.join(TASK_DIR, f"{args.root}-pt-tve-1-hop")
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"source": src, "source_rows": n_total, "seed": args.seed, "splits": {}}
    entity_col, entity_table = ENTITY[args.root]
    for split, idx in parts.items():
        part = table.take(idx).to_pandas()
        # Save through RelBench's Table so the parquet carries the table metadata it expects
        # (same as TVE's notebooks).
        Table(df=part, fkey_col_to_pkey_table={entity_col: entity_table}, pkey_col=None,
              time_col="timestamp").save(os.path.join(out_dir, f"{split}.parquet"))
        pts = pd.to_datetime(part["timestamp"])
        manifest["splits"][split] = {
            "rows": len(part),
            "null_rate": float((part["label"] == 1).mean()),
            "ts_min": str(pts.min()),
            "ts_max": str(pts.max()),
            "max_window_end": str(pts.max() + DELTA),
        }
        print(split, manifest["splits"][split], flush=True)

    os.makedirs("/workspace/results/raw", exist_ok=True)
    with open(f"/workspace/results/raw/pretrain_subsample_{args.root}.json", "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
