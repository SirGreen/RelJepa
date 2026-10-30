"""The bootstrap intervals of one (stage, task) must not depend on which other groups exist."""
import importlib.util
import os
import zlib

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("ar", os.path.join(ROOT, "scripts", "analyze_results.py"))
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


def _summary(base, stages, n=50):
    """Fake run directories; each stage's data depend only on the stage name."""
    rows = []
    for stage in stages:
        rng = np.random.default_rng(zlib.crc32(stage.encode()))
        labels = rng.normal(size=n)
        for method in ("tve1", "reljepa"):
            d = base / f"{stage}-{method}"
            d.mkdir(parents=True)
            np.save(d / "test-labels.npy", labels)
            for k in (1, 2):
                np.save(d / f"run{k}-best-test-pred.npy", rng.normal(size=n))
            rows.append({"stage": stage, "task": "item-sales", "method": method, "metric": "mae",
                         "log_dir": str(d)})
    return pd.DataFrame(rows)


def test_group_intervals_ignore_other_groups(tmp_path):
    both = ar.bootstrap(_summary(tmp_path / "both", ["full", "probe"]), 200)
    only = ar.bootstrap(_summary(tmp_path / "only", ["probe"]), 200)
    cols = ["test_seedmean", "ci_low", "ci_high", "diff_vs_tve1_low", "diff_vs_tve1_high"]
    pick = lambda b: b[(b["stage"] == "probe") & (b["method"] == "reljepa")][cols].to_numpy()
    np.testing.assert_allclose(pick(both), pick(only))
