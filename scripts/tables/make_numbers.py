"""Generate paper/tables/numbers.tex: every result quoted in the text, as LaTeX lookups.

In the paper, \\R{<key>} prints a value and fails visibly (red TODO) if the key does not exist.
Keys:
  <stage>/<task>/<method>           test mean ± std over seeds    e.g. probe/user-churn/reljepa
  <stage>/<task>/<method>/mean      test mean only
  <stage>/<task>/<a>-vs-<b>         95% paired bootstrap interval of a - b
  info/<probe>/<method>[/...]       information probes (results/tables/info_probes_bootstrap.csv)
  infoabl/<probe>/<method>[/...]    the same probes on the short ablation checkpoints, abl-<variant>
                                    (info_probes_ablation_bootstrap.csv)
  firstpass/<task>/...              first E2/E3 pass at one learning rate
  gain/<churn|sales|probe>/<method> gain over the frozen random encoder (Fig. 2)
  rec/hitrate/<stage>/<method>      share of test customers with >= 1 hit in the top 12 (%)
Scales follow the tables: AUROC, MAP, Recall and R^2 x100 with 2 decimals; MAE with 4 decimals
(3 for top-200 sales). Usage: python scripts/tables/make_numbers.py
"""
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLES = os.path.join(ROOT, "results", "tables")
OUT = os.path.join(ROOT, "paper", "tables", "numbers.tex")
METHODS = ["scratch", "mae", "tve1", "reljepa", "reljepa-tve", "tve1-empty"]   # tve1-empty: TVE-1 + empty-window head control


def _fmt(metric, task):
    if metric == "mae":
        return 1, (3 if "top200" in task else 4)
    if metric.startswith("link_prediction"):
        return 100, 3   # MAP x100: seed stds and differences are ~0.01
    return 100, 2


def _num(x, dec):
    return f"{x:.{dec}f}".replace("-", "$-$")


def _signed(x, dec):
    return f"{x:+.{dec}f}".replace("+", "$+$").replace("-", "$-$")


def _boot_entries(prefix, boot, scale_of, refs=METHODS):
    out = {}
    for r in boot.to_dict("records"):   # not itertuples: it renames the hyphenated diff columns
        scale, dec = scale_of(r)
        base = f"{prefix(r)}/{r['method']}"
        mean_key = "test_seedmean" if "test_seedmean" in r else "value"
        out[f"{base}/mean"] = _num(r[mean_key] * scale, dec)
        for ref in refs:
            lo, hi = r.get(f"diff_vs_{ref}_low"), r.get(f"diff_vs_{ref}_high")
            if lo is not None and not pd.isna(lo):
                out[f"{prefix(r)}/{r['method']}-vs-{ref}"] = f"[{_signed(lo * scale, dec)}, {_signed(hi * scale, dec)}]"
    return out


def _pretrain_dir(logdir_file):
    with open(os.path.join(ROOT, "logs", logdir_file)) as f:
        path = f.read().strip()
    return os.path.join(ROOT, "third_party", "task-vector-estimation", path.split("task-vector-estimation/", 1)[-1])


def _ablations():
    """Anti-collapse and empty-head ablations: 1-seed churn probes, final diagnostics.

    abl/<variant>/churn            test AUROC x100 of the ablation probe (probe-abl-user-reljepa-*;
                                   abl/scratch/churn: the frozen random-init pilot probe)
    diag/<run>/<column>            last logged value of run1-diagnostics.csv (RankMe: 1 decimal)
    """
    out = {}
    df = pd.read_csv(os.path.join(ROOT, "results", "raw", "finetune.csv"))
    variants = ("ema10", "sigreg", "noac", "ema10-noempty", "ema10-emptyonly")
    probes = {v: f"probe-abl-user-reljepa-{v}" for v in variants}
    probes["scratch"] = "pilot-probe-scratch"   # frozen random init, same probe config
    for variant, name in probes.items():
        v = df[(df["name"] == name) & (df["split"] == "test") & (df["metric"] == "roc_auc")]["value"]
        if len(v):
            out[f"abl/{variant}/churn"] = _num(100 * v.mean(), 2)
    runs = {f"abl-{v}": f"ablations/logdir_abl-user-reljepa-{v}.txt" for v in variants
            if os.path.exists(os.path.join(ROOT, "logs", f"ablations/logdir_abl-user-reljepa-{v}.txt"))}
    runs.update({f"main-{root}-{m}": f"main/logdir_pt-{root}-{m}.txt"
                 for root in ("user", "item") for m in ("reljepa", "reljepa-tve")})
    for run, logdir_file in runs.items():
        d = pd.read_csv(os.path.join(_pretrain_dir(logdir_file), "run1-diagnostics.csv"))
        last = d.iloc[-1]
        out[f"diag/{run}/step"] = f"{int(last['step']):,}".replace(",", "{,}")
        for col in d.columns:
            if col == "z_rankme" or col.startswith("y_rankme_"):
                out[f"diag/{run}/{col}"] = _num(last[col], 1)
    return out


def _gains(summary, info):
    """Gain over the frozen random encoder ("scratch") on the frozen-encoder evaluations (Fig. 2).

    gain/churn/<method>    churn linear probe, test AUROC x100 (seed means)
    gain/sales/<method>    sales linear probe, MAE reduction (4 decimals)
    gain/<probe>/<method>  information probes, x100
    """
    out = {}
    probe = summary[summary["stage"] == "probe"]
    churn = probe[probe["task"] == "user-churn"].set_index("method")["test_mean"]
    sales = probe[probe["task"] == "item-sales"].set_index("method")["test_mean"]
    for m in churn.index:
        out[f"gain/churn/{m}"] = _num(100 * (churn[m] - churn["scratch"]), 2)
    for m in sales.index:
        out[f"gain/sales/{m}"] = _num(sales["scratch"] - sales[m], 4)
    for r in info.to_dict("records"):
        ref = info[(info["probe"] == r["probe"]) & (info["method"] == "scratch")]["value"].iloc[0]
        out[f"gain/{r['probe']}/{r['method']}"] = _num(100 * (r["value"] - ref), 2)
    return out


def _cost():
    """Pretraining wall-clock time from logs/main/summary.log.

    Start = timestamp in the run directory name, end = the job's record time (both UTC).
    cost/<root>/<method>/hours and cost/<root>/reljepa-over-tve1 (ratio of wall-clock times).
    """
    out, hours = {}, {}
    with open(os.path.join(ROOT, "logs", "main", "summary.log")) as f:
        for line in f:
            m = re.match(r"(\S+)Z pt-(user|item)-(\S+) exit=0 logdir=.*/(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", line.strip())
            if m:
                end = pd.Timestamp(m.group(1))
                start = pd.Timestamp(pd.to_datetime(m.group(4), format="%Y-%m-%d_%H-%M-%S"))
                hours[(m.group(2), m.group(3))] = (end - start).total_seconds() / 3600
    for (root, method), h in hours.items():
        out[f"cost/{root}/{method}/hours"] = _num(h, 1)
    for root in ("user", "item"):
        if (root, "reljepa") in hours and (root, "tve1") in hours:
            out[f"cost/{root}/reljepa-over-tve1"] = _num(hours[(root, "reljepa")] / hours[(root, "tve1")], 1)
    return out


def main():
    entries = {}
    summary = pd.read_csv(os.path.join(TABLES, "summary.csv"))
    for r in summary.itertuples():
        scale, dec = _fmt(r.metric, r.task)
        key = f"{r.stage}/{r.task}/{r.method}"
        entries[key] = f"{_num(r.test_mean * scale, dec)} $\\pm$ {_num(r.test_std * scale, dec)}"
        entries[f"{key}/mean"] = _num(r.test_mean * scale, dec)
    boot = pd.read_csv(os.path.join(TABLES, "summary_bootstrap.csv"))
    entries.update(_boot_entries(lambda r: f"{r['stage']}/{r['task']}", boot, lambda r: _fmt(r["metric"], r["task"])))
    first = pd.read_csv(os.path.join(TABLES, "e2e3_firstpass_bootstrap.csv"))
    entries.update(_boot_entries(lambda r: f"firstpass/{r['task']}", first, lambda r: _fmt(r["metric"], r["task"])))
    info = pd.read_csv(os.path.join(TABLES, "info_probes_bootstrap.csv"))
    entries.update(_boot_entries(lambda r: f"info/{r['probe']}", info, lambda r: (100, 2)))
    info_abl = pd.read_csv(os.path.join(TABLES, "info_probes_ablation_bootstrap.csv"))
    entries.update(_boot_entries(lambda r: f"infoabl/{r['probe']}", info_abl, lambda r: (100, 2),
                                 refs=sorted(info_abl["method"].unique())))
    for r in summary[summary["stage"].str.startswith("rec")].itertuples():
        run = os.path.join(ROOT, r.log_dir)
        files = sorted(f for f in os.listdir(run) if f.endswith("-best-test-ap.npy"))
        ap = np.stack([np.load(os.path.join(run, f)) for f in files])
        entries[f"rec/hitrate/{r.stage}/{r.method}"] = _num(100 * (ap > 0).mean(), 1)
    entries.update(_ablations())
    entries.update(_gains(summary, info))
    entries.update(_cost())

    lines = ["% Generated by scripts/tables/make_numbers.py. Do not edit.",
             "\\newcommand{\\R}[1]{\\ifcsname res@#1\\endcsname\\csname res@#1\\endcsname"
             "\\else\\TODO{missing result #1}\\fi}"]
    for k in sorted(entries):
        lines.append(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{entries[k]}}}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(entries)} values)")


if __name__ == "__main__":
    main()
