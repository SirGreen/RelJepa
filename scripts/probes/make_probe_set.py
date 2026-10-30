"""Information-probe set: what does a frozen encoder know about the next week?

Draws (customer, t) pairs from TVE's released rel-hm user pretraining table that the shared
pretraining subsample did NOT use (same seeded permutation as make_pretrain_subsample.py), all
with windows ending on or before the validation cutoff, and computes next-window targets directly
from the database:
  1. count        : number of transactions in (t, t+7d]
  2. spend_g0..9  : spend per top-10 product group (+ spend_other), cross-column combination
  3. early_share  : share of the window's transactions in its first 2 days (timing)
  4. articles     : purchased article ids (item identity; Recall@12 via a linear multi-label probe)
Customers are split 80/20 into probe-train / probe-test (no customer in both).

Run inside the container from the TVE root:
  python /workspace/scripts/probes/make_probe_set.py --n_full 20000 --n_empty 5000 --n_train_subsample 6000000
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from relbench.base import Database

VAL_TS = pd.Timestamp("2020-09-07")
DELTA = pd.Timedelta(days=7)
EARLY = pd.Timedelta(days=2)
SRC = "data/relbench/rel-hm/tasks/user-tve-1-hop/train.parquet"
OUT_DIR = "data/probes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_full", type=int, default=20000)
    ap.add_argument("--n_empty", type=int, default=5000)
    ap.add_argument("--n_train_subsample", type=int, required=True,
                    help="n_train used by make_pretrain_subsample.py for the user root")
    ap.add_argument("--n_hold", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    table = pq.read_table(SRC, columns=["timestamp", "customer_id", "label"]).to_pandas()
    n_total = len(table)
    # Rows used by the shared pretraining subsample (train + 2 hold-out splits): exclude them.
    perm = np.random.default_rng(args.seed).permutation(n_total)
    unused = perm[args.n_train_subsample + 2 * args.n_hold:]
    pool = table.iloc[unused]
    assert (pd.to_datetime(pool["timestamp"]) + DELTA <= VAL_TS).all()

    rng = np.random.default_rng(args.seed + 1)
    full = pool[pool["label"] == 0].sample(n=args.n_full, random_state=int(rng.integers(1 << 31)))
    empty = pool[pool["label"] == 1].sample(n=args.n_empty, random_state=int(rng.integers(1 << 31)))
    probe = pd.concat([full, empty]).reset_index(drop=True)
    probe["timestamp"] = pd.to_datetime(probe["timestamp"])

    db = Database.load("data/rel-hm/dbs/default_db")
    tr = db.table_dict["transactions"].df[["customer_id", "article_id", "t_dat", "price"]]
    art = db.table_dict["article"].df[["article_id", "product_group_name"]]
    tr = tr.merge(art, on="article_id", how="left")
    # Top-10 product groups by number of transactions before the validation cutoff.
    top = tr[tr["t_dat"] <= VAL_TS]["product_group_name"].value_counts().index[:10].tolist()
    tr["group"] = np.where(tr["product_group_name"].isin(top),
                           tr["product_group_name"].map({g: i for i, g in enumerate(top)}), 10)

    # Window rows per probe pair via a merge on customer and a time filter.
    win = probe[["customer_id", "timestamp"]].reset_index().merge(tr, on="customer_id", how="inner")
    win = win[(win["t_dat"] > win["timestamp"]) & (win["t_dat"] <= win["timestamp"] + DELTA)]
    g = win.groupby("index")
    probe["count"] = g.size().reindex(probe.index, fill_value=0).astype(int)
    spend = win.pivot_table(index="index", columns="group", values="price", aggfunc="sum", fill_value=0.0)
    for i in range(11):
        name = f"spend_g{i}" if i < 10 else "spend_other"
        probe[name] = spend[i].reindex(probe.index, fill_value=0.0) if i in spend.columns else 0.0
    early = win.assign(early=win["t_dat"] <= win["timestamp"] + EARLY).groupby("index")["early"].mean()
    probe["early_share"] = early.reindex(probe.index)          # NaN for empty windows
    probe["articles"] = g["article_id"].apply(lambda s: sorted(set(s.tolist()))).reindex(probe.index)
    probe["articles"] = probe["articles"].apply(lambda x: x if isinstance(x, list) else [])

    # Consistency check against TVE's null label.
    assert ((probe["count"] == 0) == (probe["label"] == 1)).all(), "count disagrees with TVE's null label"

    customers = np.sort(probe["customer_id"].unique())
    test_customers = set(np.random.default_rng(args.seed + 2).choice(
        customers, size=int(0.2 * len(customers)), replace=False))
    probe["split"] = np.where(probe["customer_id"].isin(test_customers), "test", "train")

    os.makedirs(OUT_DIR, exist_ok=True)
    probe.to_parquet(os.path.join(OUT_DIR, "probe_set.parquet"), index=False)
    meta = {"rows": len(probe), "non_empty": int((probe["count"] > 0).sum()), "top_groups": top,
            "split_counts": probe["split"].value_counts().to_dict(),
            "ts_min": str(probe["timestamp"].min()), "ts_max": str(probe["timestamp"].max())}
    os.makedirs("/workspace/results/raw", exist_ok=True)
    with open("/workspace/results/raw/probe_set_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
