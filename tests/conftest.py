"""Shared fixtures. Run inside the container from the TVE root:
    scripts/dk.sh python -m pytest -q /workspace/tests
The data fixtures load the rel-hm graph once per session (~1 min).
"""
import os
import sys

import pytest

TVE_ROOT = "/workspace/third_party/task-vector-estimation"
if TVE_ROOT not in sys.path:
    sys.path.insert(0, TVE_ROOT)

VAL_TS = "2020-09-07"
TEST_TS = "2020-09-14"
BENCH_CFG = "/workspace/configs/bench/reljepa-user.yaml"


def load_cfg(path=BENCH_CFG, batch_size=64, task=None):
    from src.utils.io_utils import load_config

    context = dict(seed=43, debug=True, fast_dev_run=False, num_runs=1, name="test", note="test",
                   project="test", checkpoint="", gpus=[], num_workers=0)
    cfg = load_config(path, context=context)
    cfg.train.batch_size = batch_size
    cfg.train.num_workers = 0
    if "jepa" in cfg:
        cfg.jepa.target_num_workers = 0
    if task is not None:
        cfg.dataset.task = task
    return cfg


@pytest.fixture(scope="session")
def user_data():
    import torch
    from src.helpers.dataset_helper import get_data

    os.chdir(TVE_ROOT)
    cfg = load_cfg()
    data_dict = get_data(cfg, torch.device("cpu"))
    return cfg, data_dict
