"""Fig. 3: collapse diagnostics of the anti-collapse ablations.

Reads run1-diagnostics.csv of the three 10 x 1,000-step user-root ablations (EMA, SIGReg on z_ctx,
no anti-collapse); run directories come from logs/ablations/logdir_abl-user-reljepa-*.txt.
Small multiples, one panel per embedding: effective rank (RankMe) of the context embedding z_ctx
and of the target vectors y_k for both target node types, over pretraining steps (log y axis).
Identity never relies on colour alone: each variant has its own line style and marker and is
named in the legend. Vector PDF, 7 pt text, LNCS text width.
Usage: python scripts/figures/plot_collapse.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TVE = os.path.join(ROOT, "third_party", "task-vector-estimation")
OUT = os.path.join(ROOT, "paper", "figures", "collapse.pdf")

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
ACCENT = "#2a78d6"   # categorical slot 1 (palette.md), as in plot_decomposition.py
VARIANTS = [  # (run tag, label, colour, line style, marker)
    ("ema10", "EMA target encoder", ACCENT, "-", "o"),
    ("sigreg", "Shared + SIGReg on $\\mathbf{z}_{\\mathrm{ctx}}$", MUTED, "--", "s"),
    ("noac", "Shared, no anti-collapse", INK, ":", "^"),
]
PANELS = [("z_rankme", "Context $\\mathbf{z}_{\\mathrm{ctx}}$"),
          ("y_rankme_transactions", "Target $\\mathbf{y}_k$, transactions"),
          ("y_rankme_article", "Target $\\mathbf{y}_k$, articles")]


def _run_dir(tag):
    with open(os.path.join(ROOT, "logs", "ablations", f"logdir_abl-user-reljepa-{tag}.txt")) as f:
        path = f.read().strip()
    return os.path.join(TVE, path.split("task-vector-estimation/", 1)[-1])


def main():
    data = {tag: pd.read_csv(os.path.join(_run_dir(tag), "run1-diagnostics.csv")) for tag, *_ in VARIANTS}
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.8, 1.6), sharey=True)
    for ax, (col, title) in zip(axes, PANELS):
        for tag, label, colour, style, marker in VARIANTS:
            d = data[tag]
            ax.plot(d["step"] / 1000, d[col], color=colour, linestyle=style, linewidth=1.0,
                    marker=marker, markevery=max(1, len(d) // 8), markersize=3, label=label)
        ax.set_title(title, fontsize=7, color=INK)
        ax.set_yscale("log")
        ax.set_xlabel("Pretraining step (thousands)", fontsize=6.3)
        ax.grid(color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, labelsize=6.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Effective rank (RankMe)", fontsize=6.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=6.3,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0.1, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
