"""Dataset and task statistics for the paper's Table 1 (results/raw/task_stats.csv, db_stats.csv).

Run inside the container from the TVE root:  python /workspace/scripts/data/task_stats.py
Counts rows of the RelBench rel-hm database (full download and the view used for message passing,
which stops at the test cutoff) and the train/val/test sizes of every task we evaluate.
"""
import csv
import glob
import os

import pandas as pd
import pyarrow.parquet as pq

DB_FULL = "data/relbench/rel-hm/db"            # RelBench download (all rows)
DB_VIEW = "data/rel-hm/dbs/default_db"         # TVE/RelBench default view (rows up to the test cutoff)
TASKS = "data/relbench/rel-hm/tasks"
OUT = "/workspace/results/raw"
TASK_LIST = ["user-churn", "item-sales", "user-churn-top50-spending", "user-churn-top100-spending",
             "item-sales-top200-spending", "user-item-purchase"]

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "db_stats.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["db", "table", "rows"])
    for name, path in [("relbench_download", DB_FULL), ("message_passing_view", DB_VIEW)]:
        total = 0
        for p in sorted(glob.glob(os.path.join(path, "*.parquet"))):
            n = pq.ParquetFile(p).metadata.num_rows
            total += n
            w.writerow([name, os.path.basename(p)[:-8], n])
            print(name, os.path.basename(p), n)
        w.writerow([name, "TOTAL", total])
        print(name, "TOTAL", total)

with open(os.path.join(OUT, "task_stats.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "split", "rows", "num_timestamps", "ts_min", "ts_max", "positive_rate_or_mean"])
    for task in TASK_LIST:
        for split in ["train", "val", "test"]:
            p = os.path.join(TASKS, task, f"{split}.parquet")
            df = pd.read_parquet(p)
            ts = pd.to_datetime(df["timestamp"])
            stat = ""
            if "churn" in df:
                stat = round(float(df["churn"].mean()), 4)
            elif "sales" in df:
                stat = round(float(df["sales"].mean()), 4)
            w.writerow([task, split, len(df), ts.nunique(), ts.min().date(), ts.max().date(), stat])
            print(task, split, len(df), ts.nunique(), ts.min().date(), ts.max().date(), stat)
