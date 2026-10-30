"""scikit-learn compatibility for RelBench 1.1.0 metrics.

`relbench.metrics.rmse` calls `mean_squared_error(..., squared=False)`. scikit-learn 1.6 removed
the `squared` argument (the container has 1.7.2), so every task whose metric list contains RelBench's
rmse (rel-hm item-sales, item-sales-top200-spending) crashed at its first evaluation.
`patch_task_metrics` swaps in an RMSE with the same name and value; all other metrics are untouched.
"""
import numpy as np
import sklearn.metrics as skm
from relbench import metrics as relbench_metrics


def rmse(true, pred) -> float:
    """Root mean squared error; same name and value as RelBench 1.1.0's `rmse` on 1-D targets."""
    return float(np.sqrt(skm.mean_squared_error(true, pred)))


def patch_task_metrics(task):
    """Replace RelBench's rmse in `task.metrics` on this instance (the task class is unchanged)."""
    metrics = getattr(task, "metrics", None)
    if metrics:
        task.metrics = [rmse if m is relbench_metrics.rmse else m for m in metrics]
    return task
