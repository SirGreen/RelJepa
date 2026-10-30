"""E2/E3 sensitivity analysis: the first full-data probe pass at one learning rate.

The first probes-full pass ran every method at the task config's fixed rate (user-churn 5e-3,
item-sales 2e-3). This script reports that pass with analyze_results.py's own selection rule and
paired bootstrap, restricted to those runs (first-pass directories, before the grid completion).
The primary E2/E3 analysis is analyze_results.py, which selects only within {1e-3, 5e-3}.
Output: results/tables/e2e3_firstpass_bootstrap.csv.  Usage: python scripts/e2e3_firstpass.py
"""
import importlib.util
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST_PASS_RATE = {"user-churn": 0.005, "item-sales": 0.002}
FIRST_PASS_END = "2026-10-25_07-05"   # probes-full finished 07:01 UTC; grid-completion runs start later

spec = importlib.util.spec_from_file_location("ar", os.path.join(ROOT, "scripts", "analyze_results.py"))
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


def main():
    df = pd.read_csv(os.path.join(ROOT, "results", "raw", "finetune.csv"))
    parsed = df["name"].map(ar.parse_name)
    df = df[parsed.notna()].copy()
    df[["stage", "task", "method"]] = pd.DataFrame(parsed[parsed.notna()].tolist(), index=df.index)
    df = df[(df["stage"] == "probe") & df["task"].isin(list(FIRST_PASS_RATE))]
    df = df[df["lr"] == df["task"].map(FIRST_PASS_RATE)]
    stamp = df["log_dir"].str.extract(r"sweep-(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})")[0]
    df = df[stamp < FIRST_PASS_END]
    rows = []
    for (stage, task, method), g in df.groupby(["stage", "task", "method"]):
        metric = next(m for m in ar.TUNE if m in set(g["metric"]))
        g = g[g["metric"] == metric]
        val = g[g["split"] == "val"].groupby("log_dir")["value"].mean()
        best = val.idxmax() if ar.TUNE[metric] == "max" else val.idxmin()
        sel = g[g["log_dir"] == best]
        v, t = sel[sel["split"] == "val"]["value"], sel[sel["split"] == "test"]["value"]
        rows.append({"stage": stage, "task": task, "method": method, "metric": metric,
                     "selected_lr": sel["lr"].iloc[0], "n_seeds": len(t), "val_mean": v.mean(),
                     "test_mean": t.mean(), "test_std": t.std(ddof=1), "log_dir": best})
    summary = pd.DataFrame(rows)
    print(summary.drop(columns="log_dir").to_string(index=False), flush=True)
    boot = ar.bootstrap(summary, 2000)
    out = os.path.join(ROOT, "results", "tables", "e2e3_firstpass_bootstrap.csv")
    boot.to_csv(out, index=False)
    print(boot.to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
