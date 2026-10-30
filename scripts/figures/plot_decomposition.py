"""Fig. 2: decomposition of future signals on frozen customer embeddings.

Left: which objective components each method trains with (masked-feature reconstruction, TVE's
handcrafted statistics, the empty-window head, learned targets). Right: small multiples, one panel per
kind of future signal (metrics differ, so no shared axis), each showing the gain over a frozen
randomly initialised encoder ("Scratch"):
  activity  = churn linear probe (E2), test AUROC, mean over 3 seeds  (results/tables/summary.csv)
  volume    = next-week transaction count, R^2                        (results/raw/info_probes*.csv)
  semantics = spend per product group (E5), mean R^2
  identity  = next-week articles (E4), Recall@12
Horizontal bars with the method names as tick labels, so identity never relies on colour; methods
with learned targets have an accent fill plus 45-degree hatching, the post hoc TVE-1+head control a
neutral fill with dark 135-degree hatching and a dagger, the rest neutral gray (grayscale-safe).
Values are direct-labelled. Vector PDF, 7 pt text, width = LNCS text width (12.2 cm).
Usage: python scripts/figures/plot_decomposition.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY = os.path.join(ROOT, "results", "tables", "summary.csv")
INFO = [os.path.join(ROOT, "results", "raw", f) for f in ("info_probes.csv", "info_probes_tveempty.csv")]
OUT = os.path.join(ROOT, "paper", "figures", "decomposition.pdf")

METHODS = [("mae", "MFR"), ("tve1", "TVE-1"), ("tve1-empty", "TVE-1+head$^{\\dagger}$"),
           ("reljepa", "RelJEPA"), ("reljepa-tve", "RelJEPA+TVE")]
COMPONENTS = ["Masked", "Stats", "Empty", "Learned"]   # objective parts, matrix columns
USES = {"mae": {"Masked"}, "tve1": {"Stats"}, "tve1-empty": {"Stats", "Empty"},
        "reljepa": {"Empty", "Learned"}, "reljepa-tve": {"Stats", "Empty", "Learned"}}
PANELS = [("churn", "Activity (E2)\nchurn $\\Delta$AUROC"),
          ("count_r2", "Volume\ncount $\\Delta R^2$"),
          ("spend_r2", "Semantics (E5)\nspend/group $\\Delta R^2$"),
          ("item_recall12", "Identity (E4)\nitems $\\Delta$Recall")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
NEUTRAL, ACCENT = "#b9b8b1", "#2a78d6"          # neutral gray; categorical slot 1 (palette.md)
EMPHASIS = {"reljepa", "reljepa-tve"}           # methods with learned targets
POST_HOC = {"tve1-empty"}                       # post hoc control, not pre-registered


def gains():
    """Gain over the frozen random encoder ("scratch"), x100, per panel and method."""
    s = pd.read_csv(SUMMARY)
    churn = s[(s["stage"] == "probe") & (s["task"] == "user-churn")].set_index("method")["test_mean"]
    out = {"churn": {m: 100 * (churn[m] - churn["scratch"]) for m, _ in METHODS}}
    info = pd.concat([pd.read_csv(f) for f in INFO], ignore_index=True).set_index(["probe", "method"])["value"]
    for probe, _ in PANELS[1:]:
        out[probe] = {m: 100 * (info[(probe, m)] - info[(probe, "scratch")]) for m, _ in METHODS}
    return out


def main():
    g = gains()
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                         "hatch.linewidth": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 1 + len(PANELS), figsize=(4.8, 1.5), sharey=True,
                             gridspec_kw={"width_ratios": [0.62] + [1] * len(PANELS)})
    rows = range(len(METHODS))

    mat = axes[0]
    for i, (m, _) in enumerate(METHODS):
        for j, c in enumerate(COMPONENTS):
            used = c in USES[m]
            mat.plot(j, i, "o", ms=4.2 if used else 2.2, color=INK if used else GRID, zorder=3)
        js = [j for j, c in enumerate(COMPONENTS) if c in USES[m]]
        if len(js) > 1:
            mat.plot([min(js), max(js)], [i, i], color=INK, linewidth=0.8, zorder=2)
    mat.set_xlim(-0.6, len(COMPONENTS) - 0.4)
    mat.set_xticks(range(len(COMPONENTS)))
    mat.set_xticklabels(COMPONENTS, rotation=90, fontsize=6.3)
    mat.xaxis.tick_top()
    mat.tick_params(axis="both", length=0, labelsize=6.3)
    for side in mat.spines.values():
        side.set_visible(False)
    mat.set_yticks(list(rows))
    mat.set_yticklabels([label for _, label in METHODS])
    mat.invert_yaxis()

    for ax, (key, title) in zip(axes[1:], PANELS):
        vals = [g[key][m] for m, _ in METHODS]
        for i, ((m, _), v) in enumerate(zip(METHODS, vals)):
            if m in POST_HOC:   # hatch takes the edge colour; linewidth 0 keeps the outline off
                ax.barh(i, v, height=0.62, color=NEUTRAL, edgecolor=MUTED, hatch="\\\\\\\\", linewidth=0)
            else:
                ax.barh(i, v, height=0.62, color=ACCENT if m in EMPHASIS else NEUTRAL, edgecolor="white",
                        hatch="////" if m in EMPHASIS else None, linewidth=0.5)
            ax.text(v, i, f" {v:.2f} ".replace("-", "−"), va="center",
                    ha="left" if v >= 0 else "right", color=INK, fontsize=6.3)
        lo, hi = min(0.0, min(vals)), max(vals)
        span = hi - lo
        ax.set_xlim(lo - (0.55 * span if lo < 0 else 0), hi + 0.6 * span)
        ax.axvline(0, color=MUTED, linewidth=0.6)
        ax.set_title(title, fontsize=7, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, labelsize=6.3)
        if lo < 0:   # symmetric ticks keep "0" clear of the negative labels
            step = round(max(-lo, hi), 1)
            ax.set_xticks([-step, 0, step])
            ax.set_xticklabels([f"{-step:g}".replace("-", "−"), "0", f"{step:g}"])
        else:
            ax.locator_params(axis="x", nbins=3)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=0.5)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    for key, _ in PANELS:
        print(key, {m: round(v, 2) for m, v in g[key].items()})


if __name__ == "__main__":
    main()
