"""Paired bootstrap intervals for the information probes.

Reads results/raw/info_probes/<method>.npz written by scripts/probes/run_probes.py. For every probe
it resamples probe-test customers with replacement (the same resample for every method), recomputes
the score exactly as run_probes.py does (R^2, mean R^2 over columns for "spend", mean per-customer
Recall@12 for "items"), and reports 95% percentile intervals of each score and of the paired
difference between every pair of methods.
Output: results/tables/info_probes_bootstrap.csv.  Usage: python scripts/probes/bootstrap_probes.py [--boot 2000]
The resamples depend only on the probe-test set, so a run with other --methods (e.g. the loss
ablations) draws the same resamples as the main run.
"""
import argparse
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METHODS = ["scratch", "mae", "tve1", "reljepa", "reljepa-tve", "tve1-empty"]   # tve1-empty: TVE-1 + empty-window head control
PROBES = {  # probe -> (true key, pred key); None means a per-customer score averaged directly
    "count_r2": ("count_true", "count_pred"),
    "spend_r2": ("spend_true", "spend_pred"),
    "early_r2": ("early_true", "early_pred"),
    "item_recall12": None,
}


def r2(y, p):
    """sklearn.metrics.r2_score; for 2-D targets the mean over columns (as in run_probes.py)."""
    y = y.reshape(len(y), -1)
    p = p.reshape(len(p), -1)
    ss_res = ((y - p) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.mean(1.0 - ss_res / ss_tot))


def score(data, probe, idx=None):
    if PROBES[probe] is None:
        v = data["item_recall"]
        return float(v.mean() if idx is None else v[idx].mean())
    t, p = PROBES[probe]
    y, q = data[t], data[p]
    return r2(y, q) if idx is None else r2(y[idx], q[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "raw", "info_probes"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables", "info_probes_bootstrap.csv"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    args = ap.parse_args()

    data = {m: dict(np.load(os.path.join(args.dir, f"{m}.npz")))
            for m in args.methods if os.path.exists(os.path.join(args.dir, f"{m}.npz"))}
    if not data:
        raise SystemExit(f"no probe outputs in {args.dir}")
    ref = next(iter(data.values()))
    for m, d in data.items():   # paired resampling needs the same customers in the same order
        for probe, keys in PROBES.items():
            if keys is None:
                assert len(d["item_recall"]) == len(ref["item_recall"]), (m, probe)
            else:
                assert np.array_equal(d[keys[0]], ref[keys[0]]), f"{m}: {probe} targets differ"

    rows = []
    for probe, keys in PROBES.items():
        n = len(ref["item_recall"]) if keys is None else len(ref[keys[0]])
        rng = np.random.default_rng(0)
        boots = {m: np.empty(args.boot) for m in data}
        for b in range(args.boot):
            idx = rng.integers(0, n, size=n)
            for m, d in data.items():
                boots[m][b] = score(d, probe, idx)
        for m, d in data.items():
            row = {"probe": probe, "method": m, "n_test": n, "value": score(d, probe),
                   "ci_low": np.percentile(boots[m], 2.5), "ci_high": np.percentile(boots[m], 97.5)}
            for other in data:
                if other != m:
                    diff = boots[m] - boots[other]
                    row[f"diff_vs_{other}_low"] = np.percentile(diff, 2.5)
                    row[f"diff_vs_{other}_high"] = np.percentile(diff, 97.5)
            rows.append(row)
            print(f"{probe:14s} {m:12s} {row['value']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    first = ["probe", "method", "n_test", "value", "ci_low", "ci_high"]
    fields = first + sorted({k for r in rows for k in r} - set(first))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
