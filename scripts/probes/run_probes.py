"""Linear information probes on frozen customer embeddings.

Probe-train customers fit the probe, probe-test customers (disjoint) evaluate it.
  count       : RidgeCV on log1p(count), all pairs                         -> test R^2
  spend       : RidgeCV on log1p(spend per top-10 group + other), non-empty -> mean test R^2
  early_share : RidgeCV on the share of purchases in the first 2 days, non-empty -> test R^2
  items       : linear multi-label probe over the 1,000 most-bought articles of probe-train,
                non-empty pairs                                             -> test Recall@12
Embeddings are standardised with probe-train statistics. Writes results/raw/info_probes.csv and,
per method, the probe-test targets/predictions and per-customer recalls to
results/raw/info_probes/<method>.npz for paired bootstrap intervals (E4/E5;
scripts/probes/bootstrap_probes.py).

Run inside the container from the TVE root:  python /workspace/scripts/probes/run_probes.py
Ablation checkpoints go to their own CSV, so the main results stay untouched:
  python /workspace/scripts/probes/run_probes.py --methods abl-ema10 ... --out /workspace/results/raw/info_probes_ablation.csv
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

METHODS = ["scratch", "mae", "tve1", "reljepa", "reljepa-tve"]
ALPHAS = np.logspace(-2, 4, 13)
OUT = "/workspace/results/raw/info_probes.csv"
OUT_DIR = "/workspace/results/raw/info_probes"


def ridge_r2(xtr, ytr, xte, yte):
    """Test R^2 (mean over columns for multi-output) and the test predictions."""
    model = RidgeCV(alphas=ALPHAS).fit(xtr, ytr)
    pred = model.predict(xte)
    if yte.ndim == 1:
        return r2_score(yte, pred), pred
    return float(np.mean([r2_score(yte[:, j], pred[:, j]) for j in range(yte.shape[1])])), pred


def item_recall(xtr, items_tr, xte, items_te, n_items=1000, k=12, epochs=60, seed=0):
    counts = pd.Series([a for s in items_tr for a in s]).value_counts()
    vocab = {a: i for i, a in enumerate(counts.index[:n_items])}
    ytr = torch.zeros(len(items_tr), len(vocab))
    for r, s in enumerate(items_tr):
        for a in s:
            if a in vocab:
                ytr[r, vocab[a]] = 1.0
    torch.manual_seed(seed)
    lin = torch.nn.Linear(xtr.shape[1], len(vocab))
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    xt = torch.from_numpy(xtr).float()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(xt), ytr)
        loss.backward()
        opt.step()
    inv = np.array(list(vocab.keys()))
    top = torch.topk(lin(torch.from_numpy(xte).float()), k, dim=1).indices.numpy()
    recalls = []
    for r, s in enumerate(items_te):
        if len(s) == 0:
            continue
        hits = len(set(inv[top[r]]) & set(s))
        recalls.append(hits / min(len(s), k))
    return np.asarray(recalls, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.methods != METHODS and args.out == OUT:
        raise SystemExit("--methods other than the main ones needs its own --out")
    probe = pd.read_parquet("data/probes/probe_set.parquet")
    tr, te = probe["split"].to_numpy() == "train", probe["split"].to_numpy() == "test"
    nonempty = probe["count"].to_numpy() > 0
    spend_cols = [c for c in probe.columns if c.startswith("spend_")]
    rows = []
    for method in args.methods:
        path = f"data/probes/emb_{method}.npy"
        if not os.path.exists(path):
            continue
        x = np.load(path)
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        x = (x - mu) / sd
        y_count = np.log1p(probe["count"].to_numpy())
        y_spend = np.log1p(probe[spend_cols].to_numpy())
        y_early = probe["early_share"].to_numpy()
        count_r2, count_pred = ridge_r2(x[tr], y_count[tr], x[te], y_count[te])
        spend_r2, spend_pred = ridge_r2(x[tr & nonempty], y_spend[tr & nonempty], x[te & nonempty], y_spend[te & nonempty])
        early_r2, early_pred = ridge_r2(x[tr & nonempty], y_early[tr & nonempty], x[te & nonempty], y_early[te & nonempty])
        recalls = item_recall(x[tr & nonempty], probe["articles"][tr & nonempty].tolist(),
                              x[te & nonempty], probe["articles"][te & nonempty].tolist())
        res = {"count_r2": count_r2, "spend_r2": spend_r2, "early_r2": early_r2,
               "item_recall12": float(np.mean(recalls))}
        os.makedirs(OUT_DIR, exist_ok=True)
        np.savez_compressed(os.path.join(OUT_DIR, f"{method}.npz"),
                            count_true=y_count[te], count_pred=count_pred,
                            spend_true=y_spend[te & nonempty], spend_pred=spend_pred,
                            early_true=y_early[te & nonempty], early_pred=early_pred,
                            item_recall=recalls)
        for k, v in res.items():
            rows.append({"method": method, "probe": k, "value": v})
        print(method, {k: round(v, 4) for k, v in res.items()}, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "probe", "value"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
