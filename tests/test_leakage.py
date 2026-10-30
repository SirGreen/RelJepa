"""Leakage tests.

1. Every pretraining example (all splits of our shared subsamples) has t + Δ <= validation cutoff.
2. The graph used for message passing contains no row after the test cutoff.
3. Downstream training tables end their label windows at or before the validation cutoff.
"""
import os

import pandas as pd
import pytest
from relbench.base import Table

TASKS = "/workspace/third_party/task-vector-estimation/data/relbench/rel-hm/tasks"
VAL_TS = pd.Timestamp("2020-09-07")
TEST_TS = pd.Timestamp("2020-09-14")
DELTA = pd.Timedelta(days=7)


@pytest.mark.parametrize("task", ["user-pt-tve-1-hop", "item-pt-tve-1-hop"])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_pretraining_windows_end_before_validation(task, split):
    df = Table.load(os.path.join(TASKS, task, f"{split}.parquet")).df
    assert (pd.to_datetime(df["timestamp"]) + DELTA <= VAL_TS).all()


@pytest.mark.parametrize("task", ["user-churn", "item-sales", "user-churn-top50-spending",
                                  "user-churn-top100-spending", "item-sales-top200-spending"])
def test_downstream_train_windows_end_before_validation(task):
    df = Table.load(os.path.join(TASKS, task, "train.parquet")).df
    assert (pd.to_datetime(df["timestamp"]) + DELTA <= VAL_TS).all()


def test_message_passing_graph_stops_at_test_cutoff():
    from relbench.base import Database

    db = Database.load("/workspace/third_party/task-vector-estimation/data/rel-hm/dbs/default_db")
    assert pd.to_datetime(db.table_dict["transactions"].df["t_dat"]).max() <= TEST_TS
