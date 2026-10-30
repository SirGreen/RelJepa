"""Inspect TVE's released rel-hm pretraining tables: size, timestamps, null-window rate, leakage.

Run inside the container from the TVE root:  python /workspace/scripts/data/inspect_pretrain_tables.py
Writes a JSON summary to /workspace/results/raw/pretrain_tables_summary.json.
"""
import json
import os

import pandas as pd
import pyarrow.parquet as pq

DATA = "data/relbench/rel-hm/tasks"
VAL_TS = pd.Timestamp("2020-09-07")
TEST_TS = pd.Timestamp("2020-09-14")
DELTA = pd.Timedelta(days=7)

summary = {}
for task in ["user-tve-1-hop", "item-tve-1-hop", "user-ssl", "item-ssl", "user-tve-2-hop", "item-tve-2-hop"]:
    for split in ["train", "val", "test"]:
        path = os.path.join(DATA, task, f"{split}.parquet")
        if not os.path.exists(path):
            continue
        meta = pq.ParquetFile(path).metadata
        cols = pq.ParquetFile(path).schema_arrow.names
        read_cols = [c for c in ["timestamp", "customer_id", "article_id", "label"] if c in cols]
        df = pd.read_parquet(path, columns=read_cols)
        ts = pd.to_datetime(df["timestamp"])
        entry = {
            "rows": int(meta.num_rows),
            "num_columns": len(cols),
            "first_columns": cols[:4],
            "ts_min": str(ts.min()),
            "ts_max": str(ts.max()),
            "num_timestamps": int(ts.nunique()),
            "max_window_end": str(ts.max() + DELTA),
            "all_windows_end_before_val": bool((ts + DELTA <= VAL_TS).all()),
            "any_window_overlaps_test": bool((ts + DELTA > TEST_TS).any()),
        }
        if "label" in df:
            entry["null_rate"] = float((df["label"] == 1).mean())
        ent_col = "customer_id" if "customer_id" in df else "article_id"
        entry["num_entities"] = int(df[ent_col].nunique())
        summary[f"{task}/{split}"] = entry
        print(f"{task}/{split}: {json.dumps(entry)}", flush=True)

os.makedirs("/workspace/results/raw", exist_ok=True)
with open("/workspace/results/raw/pretrain_tables_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
