"""Target-view tests.

For sampled seeds (v, t):
  * every fact row in the context view has time <= t;
  * every fact row in the target view has t < time <= t + Δ;
  * no edge in the target view touches the seed node;
  * every kept dimension row is linked to a kept fact row;
  * the number of in-window rows per seed equals an exact count computed directly from the
    database (independent of the sampler), and emptiness matches TVE's null label.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from reljepa.target_view import make_target_view

N_BATCHES = 3


@pytest.fixture(scope="module")
def batches(user_data):
    from src.helpers.reljepa_train_helper import get_paired_loaders

    cfg, data_dict = user_data
    loaders, delta = get_paired_loaders(cfg, data_dict)
    ctx_loader, tgt_loader = loaders["train"]
    out = []
    for i, (ctx, tgt) in enumerate(zip(ctx_loader, tgt_loader)):
        out.append((ctx, tgt))
        if i + 1 == N_BATCHES:
            break
    return cfg, data_dict, out, delta


def test_context_view_is_causal(batches):
    _, _, pairs, _ = batches
    for ctx, _ in pairs:
        t = ctx["customer"].seed_time
        tx = ctx["transactions"]
        assert (tx.time <= t[tx.batch]).all()


def test_target_view_window_and_isolation(batches):
    _, _, pairs, delta = batches
    for _, tgt in pairs:
        bs = tgt["customer"].batch_size
        end = tgt["customer"].seed_time.clone()                  # t + Δ
        view, keep = make_target_view(tgt.clone(), "customer", delta)
        tx = view["transactions"]
        assert (tx.time > end[tx.batch] - delta).all()
        assert (tx.time <= end[tx.batch]).all()
        for (src, _, dst), ei in view.edge_index_dict.items():
            if src == "customer":
                assert (ei[0] >= bs).all(), "edge leaves a seed node"
            if dst == "customer":
                assert (ei[1] >= bs).all(), "edge enters a seed node"
        # every kept article is linked to at least one kept transaction
        linked = torch.zeros(view["article"].num_nodes, dtype=torch.bool)
        ei = view["transactions", "f2p_article_id", "article"].edge_index
        linked[ei[1]] = True
        assert linked.all()


def test_in_window_counts_match_database(batches):
    cfg, data_dict, pairs, delta = batches
    from relbench.base import Database

    db = Database.load("data/rel-hm/dbs/default_db")
    tr = db.table_dict["transactions"].df[["customer_id", "t_dat"]]
    tr = tr.sort_values(["customer_id", "t_dat"]).reset_index(drop=True)
    by_customer = tr.groupby("customer_id")["t_dat"].apply(np.asarray)
    train_df = data_dict["train_table"].df

    for _, tgt in pairs:
        bs = tgt["customer"].batch_size
        seeds = tgt["customer"].n_id[:bs].numpy()
        ends = pd.to_datetime(tgt["customer"].seed_time.numpy(), unit="s")
        view, _ = make_target_view(tgt.clone(), "customer", delta)
        got = torch.bincount(view["transactions"].batch, minlength=bs).numpy()
        input_id = tgt["customer"].input_id.numpy()
        labels = train_df["label"].values[input_id]
        for i in range(bs):
            times = by_customer.get(seeds[i], np.array([], dtype="datetime64[ns]"))
            start = ends[i] - pd.Timedelta(seconds=delta)
            exact = int(((times > np.datetime64(start)) & (times <= np.datetime64(ends[i]))).sum())
            if exact <= cfg.train.num_neighbors[0]:      # below the sampler's fan-out the view is complete
                assert got[i] == exact, f"seed {i}: view has {got[i]} rows, database has {exact}"
            assert (exact == 0) == (labels[i] == 1), "emptiness disagrees with TVE's null label"
