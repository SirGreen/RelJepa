"""per_user_ap must average to RelBench's own MAP."""
import numpy as np
import pytest

from reljepa.rec_metrics import per_user_ap

relbench_metrics = pytest.importorskip("relbench.metrics")


def _relbench_map(true_items, pred):
    # Same construction as relbench.base.RecommendationTask.evaluate (v1.1.0).
    isin = np.stack([np.isin(np.array(p), np.array(t)) for t, p in zip(true_items, pred)])
    count = np.array([len(t) for t in true_items])
    return relbench_metrics.link_prediction_map(isin, count)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mean_equals_relbench_map(seed):
    rng = np.random.default_rng(seed)
    n_users, n_items, k = 500, 60, 12
    # Include users with no true items (filtered by RelBench) and with more than k true items.
    true_items = [rng.choice(n_items, size=rng.integers(0, 20), replace=False) for _ in range(n_users)]
    pred = np.stack([rng.choice(n_items, size=k, replace=False) for _ in range(n_users)])
    ap = per_user_ap(true_items, pred)
    assert len(ap) == sum(len(t) > 0 for t in true_items)
    assert np.isclose(ap.mean(), _relbench_map(true_items, pred), rtol=0, atol=1e-12)
    assert np.all((ap >= 0) & (ap <= 1))


def test_perfect_and_empty_predictions():
    true_items = [np.array([3, 5]), np.array([7]), np.array([], dtype=int)]
    pred = np.array([[3, 5, 0], [1, 2, 4], [3, 5, 7]])
    ap = per_user_ap(true_items, pred)
    assert ap.tolist() == [1.0, 0.0]   # third user (no true items) is dropped, as in RelBench
