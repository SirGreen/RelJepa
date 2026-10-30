"""Summarize fine-tuning results.

Reads results/raw/finetune.csv (written by scripts/collect_results.py). For every
(stage, task, method) it selects the configuration (learning rate) with the best MEAN VALIDATION
score over seeds, then reports validation and test mean ± std over seeds at that configuration.
Run names follow scripts/runs/finetune_main.sh: ft-<stage>-<task>-<method>[-<sweep>].

If per-example test predictions were saved (run<k>-best-test-pred.npy + test-labels.npy in the
run directory), it also reports a 95% bootstrap interval over test examples for the seed-mean
metric, and paired bootstrap intervals for (method - baseline) differences. For recommendation
(MAP) it uses the per-user test AP saved as run<k>-best-test-ap.npy.

Output: results/tables/summary.csv (and a printed overview).
Usage: python scripts/analyze_results.py [--csv results/raw/finetune.csv] [--boot 2000]
"""
import argparse
import os
import re
import zlib

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNE = {"roc_auc": "max", "mae": "min", "link_prediction_map": "max"}
METHODS = ["scratch", "mae", "tve1", "reljepa", "reljepa-tve", "tve1-empty"]   # tve1-empty: TVE-1 + empty-window head control
PROBE_GRID = (0.001, 0.005)   # pre-registered learning rates of the full-data probes
# rec-1pct / rec-0p1pct: recommendation with 1% / 0.1% of training users.
NAME_RE = re.compile(r"^ft-(?P<stage>limited|probe|rec-low|rec-1pct|rec-0p1pct|rec-full|full|repro)-(?P<rest>.+)$")


def parse_name(name):
    """ft-limited-user-churn-top50-spending-reljepa-tve-lr0005 -> (limited, task, method)."""
    m = NAME_RE.match(name)
    if not m:
        return None
    stage, rest = m.group("stage"), m.group("rest")
    rest = re.sub(r"-lr\d+$", "", rest)            # sweep suffix added by TVE's sweep runner
    for method in sorted(METHODS, key=len, reverse=True):
        if rest.endswith("-" + method) or rest == method:
            task = rest[: -len(method)].rstrip("-") or ("user-item-purchase" if stage.startswith("rec") else "")
            return stage, task, method
    return None


def _metric_fn(metric):
    if metric == "roc_auc":
        from sklearn.metrics import roc_auc_score
        return lambda y, p: roc_auc_score(y, p)
    if metric == "mae":
        return lambda y, p: float(np.mean(np.abs(y - p)))
    return None


def load_test_preds(log_dir):
    """Seed predictions (n_seeds x n_test) and labels saved by the fine-tuning patch, or None."""
    run_dir = os.path.join(ROOT, log_dir)
    labels = os.path.join(run_dir, "test-labels.npy")
    preds = sorted(f for f in os.listdir(run_dir) if f.endswith("-best-test-pred.npy")) if os.path.isdir(run_dir) else []
    if not preds or not os.path.exists(labels):
        return None
    return np.stack([np.load(os.path.join(run_dir, f)) for f in preds]), np.load(labels)


def load_test_ap(log_dir):
    """Per-user test AP of each seed (n_seeds x n_users) saved by train_idgnn.py, or None."""
    run_dir = os.path.join(ROOT, log_dir)
    files = sorted(f for f in os.listdir(run_dir) if f.endswith("-best-test-ap.npy")) if os.path.isdir(run_dir) else []
    return np.stack([np.load(os.path.join(run_dir, f)) for f in files]) if files else None


def _node_stat(fn, metric, preds, labels):
    """Seed-mean test metric on a resample of test rows (idx=None: all rows)."""
    def stat(idx):
        y = labels if idx is None else labels[idx]
        if metric == "roc_auc" and len(np.unique(y)) < 2:
            return np.nan
        return float(np.mean([fn(y, p if idx is None else p[idx]) for p in preds]))
    return stat


def _map_stat(ap):
    """MAP of the seed-mean per-user AP on a resample of test users (equals the seed-mean MAP)."""
    return lambda idx: float(ap.mean() if idx is None else ap[idx].mean())


def bootstrap(summary, n_boot, seed=0):
    """95% intervals of the seed-mean test metric and paired differences vs. every other method.

    Test units (rows; users for MAP) are resampled once per replicate and shared by all methods,
    so the differences are paired. Indices are drawn per replicate to keep
    memory small on the full-data test sets. Each (stage, task) has its own random stream, so its
    intervals do not change when other runs are added.
    """
    out = []
    for (stage, task), g in summary.groupby(["stage", "task"]):
        rng = np.random.default_rng([seed, zlib.crc32(f"{stage}/{task}".encode())])
        metric = g["metric"].iloc[0]
        if metric == "link_prediction_map":
            aps = {r.method: load_test_ap(r.log_dir) for r in g.itertuples()}
            aps = {m: v.mean(0) for m, v in aps.items() if v is not None}
            if not aps:
                continue
            n = len(next(iter(aps.values())))
            assert all(len(a) == n for a in aps.values()), f"{stage}/{task}: different test users"
            stats = {m: _map_stat(a) for m, a in aps.items()}
        else:
            fn = _metric_fn(metric)
            loaded = {r.method: load_test_preds(r.log_dir) for r in g.itertuples()}
            loaded = {m: v for m, v in loaded.items() if v is not None}
            if fn is None or not loaded:
                continue
            labels = next(iter(loaded.values()))[1]
            n = len(labels)
            assert all(np.array_equal(lab, labels) for _, lab in loaded.values()), f"{stage}/{task}: labels differ"
            stats = {m: _node_stat(fn, metric, preds, labels) for m, (preds, _) in loaded.items()}
        boots = {m: np.empty(n_boot) for m in stats}
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            for m, stat in stats.items():
                boots[m][b] = stat(idx)
        for m, vals in boots.items():
            row = {"stage": stage, "task": task, "method": m, "metric": metric, "n_test": n,
                   "test_seedmean": stats[m](None),
                   "ci_low": np.nanpercentile(vals, 2.5), "ci_high": np.nanpercentile(vals, 97.5)}
            for ref in boots:
                if ref != m:
                    d = vals - boots[ref]
                    row[f"diff_vs_{ref}_low"] = np.nanpercentile(d, 2.5)
                    row[f"diff_vs_{ref}_high"] = np.nanpercentile(d, 97.5)
            out.append(row)
    return pd.DataFrame(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(ROOT, "results", "raw", "finetune.csv"))
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "tables", "summary.csv"))
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    parsed = df["name"].map(parse_name)
    df = df[parsed.notna()].copy()
    df[["stage", "task", "method"]] = pd.DataFrame(parsed[parsed.notna()].tolist(), index=df.index)

    # E2/E3: select only within the pre-registered grid {1e-3, 5e-3}. The first
    # probes-full pass ran at the configs' fixed rate (item-sales 2e-3), a separate sensitivity analysis.
    full_probe = (df["stage"] == "probe") & df["task"].isin(["user-churn", "item-sales"])
    df = df[~full_probe | df["lr"].isin(PROBE_GRID)]

    rows = []
    for (stage, task, method), g in df.groupby(["stage", "task", "method"]):
        metric = next((m for m in TUNE if m in set(g["metric"])), None)
        if metric is None:
            continue
        g = g[g["metric"] == metric]
        val = g[g["split"] == "val"].groupby("log_dir")["value"].mean()
        best_dir = val.idxmax() if TUNE[metric] == "max" else val.idxmin()
        sel = g[g["log_dir"] == best_dir]
        v = sel[sel["split"] == "val"]["value"]
        t = sel[sel["split"] == "test"]["value"]
        rows.append({
            "stage": stage, "task": task, "method": method, "metric": metric,
            "selected_lr": sel["lr"].iloc[0], "n_seeds": len(t),
            "val_mean": v.mean(), "val_std": v.std(ddof=1), "test_mean": t.mean(), "test_std": t.std(ddof=1),
            "log_dir": best_dir,
        })
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.drop(columns=["log_dir"]).to_string(index=False))
    if args.boot > 0 and len(out):
        boot = bootstrap(out, args.boot)
        if len(boot):
            path = args.out.replace(".csv", "_bootstrap.csv")
            boot.to_csv(path, index=False)
            with pd.option_context("display.width", 220, "display.max_columns", 20):
                print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
