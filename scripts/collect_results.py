"""Collect per-seed fine-tuning results into results/raw/finetune.csv.

Scans every `per_seed_results.json` written by TVE's train_baseline.py (patched print_runs_stats)
and by src/train_idgnn.py, and writes one CSV row per (run directory, seed, split, metric) together
with the run metadata (commit, checkpoint, lr, channels, W&B id). Never edit the CSV by hand.

Usage (host or container):  python scripts/collect_results.py [--logs <dir>] [--out <csv>]
"""
import argparse
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOGS = os.path.join(ROOT, "third_party", "task-vector-estimation", "logs", "train")
DEFAULT_OUT = os.path.join(ROOT, "results", "raw", "finetune.csv")
FIELDS = ["log_dir", "name", "note", "dataset", "task", "checkpoint", "lr", "channels", "batch_size",
          "num_epochs", "max_steps_per_epoch", "train_user_fraction", "git_commit", "wandb_run_id",
          "seed", "split", "metric", "value"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default=DEFAULT_LOGS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.logs, "**", "per_seed_results.json"), recursive=True)):
        with open(path) as f:
            rec = json.load(f)
        meta = {k: rec.get(k) for k in FIELDS if k in rec}
        meta["log_dir"] = os.path.relpath(os.path.dirname(path), ROOT).replace("\\", "/")
        for split, metrics in rec["per_seed"].items():
            for metric, values in metrics.items():
                for seed, value in zip(rec["seeds"], values):
                    rows.append({**meta, "seed": seed, "split": split, "metric": metric, "value": value})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows from {len({r['log_dir'] for r in rows})} runs to {args.out}")


if __name__ == "__main__":
    main()
