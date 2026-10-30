"""RelBench 1.1.0 rmse under scikit-learn >= 1.6. No data needed:
    scripts/dk.sh python -m pytest -q /workspace/tests/test_metrics_compat.py
"""
import numpy as np
import pytest
from relbench import metrics as rbm

from src.helpers.metrics_compat import patch_task_metrics, rmse


def test_rmse_value_and_name():
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=1000), rng.normal(size=1000)
    assert rmse(y, p) == pytest.approx(float(np.sqrt(np.mean((y - p) ** 2))), rel=1e-12)
    assert rmse.__name__ == "rmse"   # evaluate() keys metrics by __name__


def test_patch_replaces_only_rmse():
    class Task:
        metrics = [rbm.r2, rbm.mae, rbm.rmse]

    task = patch_task_metrics(Task())
    assert [m.__name__ for m in task.metrics] == ["r2", "mae", "rmse"]
    assert task.metrics[:2] == [rbm.r2, rbm.mae] and task.metrics[2] is rmse
    assert Task.metrics[2] is rbm.rmse   # class attribute untouched
    y, p = np.arange(5.0), np.arange(5.0) + 1.0
    assert {fn.__name__: fn(y, p) for fn in task.metrics}["rmse"] == pytest.approx(1.0)


def test_patch_leaves_other_tasks_alone():
    class Cls:
        metrics = [rbm.roc_auc, rbm.f1]

    class Ssl:
        metrics = None

    assert patch_task_metrics(Cls()).metrics == [rbm.roc_auc, rbm.f1]
    assert patch_task_metrics(Ssl()).metrics is None


def test_hm_regression_tasks_use_relbench_rmse():
    from src.tasks import hm

    for cls in (hm.ItemSalesTask, hm.ItemSalesTop200SpendingTask):
        assert rbm.rmse in cls.metrics


def test_get_task_applies_patch():
    import inspect

    from src.helpers import task_helper

    assert "patch_task_metrics" in inspect.getsource(task_helper.get_task)
