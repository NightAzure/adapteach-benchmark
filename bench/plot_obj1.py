"""
Figure 2: Objective 1 — Retrieval Component Comparison
Generates a grouped bar chart (MRR@10 + nDCG@10) for configs A–F.

When the CSV has a `dataset` column (from compute_metrics.py with per-dataset
grouping), produces a 2×N faceted subplot — one panel per dataset.
Falls back to a single chart when no `dataset` column is present.

Usage:
    # Single chart (combined / no dataset column)
    py bench/plot_obj1.py --csv bench/results/obj1_external.csv --out bench/figures/fig2_obj1.png

    # Faceted per-dataset chart
    py bench/plot_obj1.py --csv bench/results/obj1_by_dataset.csv --out bench/figures/fig2_obj1_faceted.png
"""

import argparse
import csv
import math
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Canonical results — 937-query external benchmark (CS1QA + MBPP) ─────────
CANONICAL = {
    "A": {"MRR@10": 0.000, "nDCG@10": 0.000, "ctx_precision": 0.000},
    "B": {"MRR@10": 0.962, "nDCG@10": 0.590, "ctx_precision": 0.891},
    "C": {"MRR@10": 0.888, "nDCG@10": 0.466, "ctx_precision": 0.780},
    "D": {"MRR@10": 0.931, "nDCG@10": 0.538, "ctx_precision": 0.875},
    "E": {"MRR@10": 0.842, "nDCG@10": 0.386, "ctx_precision": 0.750},
    "F": {"MRR@10": 0.931, "nDCG@10": 0.538, "ctx_precision": 0.875},
}
CANONICAL_N = 937

CONFIG_LABELS = {
    "A": "A\nBaseline",
    "B": "B\nDense",
    "C": "C\nHybrid",
    "D": "D\nAST+Dense",
    "E": "E\nFull\nGraphRAG",
    "F": "F\nGraph-as-\nContext",
}

DATASET_TITLES = {
    "custom":   "Custom (hand-crafted)",
    "cs1qa":    "CS1QA (real student Qs)",
    "mbpp":     "MBPP (code tasks)",
    "staqc":    "StaQC (Stack Overflow)",
    "combined": "Combined (all datasets)",
    "unknown":  "Unknown",
}

DATASET_ORDER = ["custom", "cs1qa", "staqc", "mbpp", "combined", "unknown"]

# Palette
COLOR_MRR  = "#2A9D8F"   # teal
COLOR_NDCG = "#E9C46A"   # amber
COLOR_CTX  = "#F4A261"   # orange


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(path: str) -> tuple[dict, bool]:
    """
    Returns (data, has_dataset_col).

    If has_dataset_col is True:
        data = { dataset: { config: {MRR, nDCG, ctx_precision, n} } }
    Else:
        data = { config: {MRR, nDCG, ctx_precision} }  (n extracted separately)
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    has_ds = "dataset" in (rows[0] if rows else {})

    if has_ds:
        data: dict = {}
        for row in rows:
            ds  = row["dataset"]
            cfg = row["config"]
            data.setdefault(ds, {})[cfg] = {
                "MRR@10":        float(row["MRR@10"]),
                "nDCG@10":       float(row["nDCG@10"]),
                "ctx_precision": float(row["context_precision"]),
                "n":             int(row["n"]),
            }
        return data, True
    else:
        data = {}
        n = 0
        for row in rows:
            cfg = row["config"]
            data[cfg] = {
                "MRR@10":        float(row["MRR@10"]),
                "nDCG@10":       float(row["nDCG@10"]),
                "ctx_precision": float(row["context_precision"]),
            }
            if "n" in row:
                n = max(n, int(row["n"]))
        return (data, n), False


# ── Single chart ──────────────────────────────────────────────────────────────

def _draw_panel(ax, data_by_cfg: dict, configs: list[str], title: str) -> None:
    """Draw one grouped bar panel onto the given axes."""
    x     = np.arange(len(configs))
    width = 0.26

    mrr  = [data_by_cfg.get(c, {}).get("MRR@10",        0) for c in configs]
    ndcg = [data_by_cfg.get(c, {}).get("nDCG@10",       0) for c in configs]
    ctx  = [data_by_cfg.get(c, {}).get("ctx_precision",  0) for c in configs]

    bars_mrr  = ax.bar(x - width, mrr,  width, color=COLOR_MRR,  edgecolor="white", linewidth=0.6)
    bars_ndcg = ax.bar(x,         ndcg, width, color=COLOR_NDCG, edgecolor="white", linewidth=0.6)
    bars_ctx  = ax.bar(x + width, ctx,  width, color=COLOR_CTX,  edgecolor="white", linewidth=0.6)

    for bars in (bars_mrr, bars_ndcg, bars_ctx):
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.013,
                    f"{h:.3f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="#333333", fontweight="semibold",
                )

    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs], fontsize=7.5)
    ax.set_ylabel("Score", fontsize=9, labelpad=6)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=8, color="#1A1A1A")
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_single(data_by_cfg: dict, n: int, out_path: str) -> None:
    """One chart for a flat (no-dataset) CSV."""
    configs = list(CONFIG_LABELS.keys())

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")

    _draw_panel(ax, data_by_cfg, configs,
                f"Objective 1 — Retrieval Component Comparison  (n={n:,} queries)")

    ax.set_xlabel("Pipeline Configuration", fontsize=10, labelpad=8)
    ax.legend(
        handles=[
            mpatches.Patch(color=COLOR_MRR,  label="MRR@10"),
            mpatches.Patch(color=COLOR_NDCG, label="nDCG@10"),
            mpatches.Patch(color=COLOR_CTX,  label="Context Precision"),
        ],
        loc="upper left", framealpha=0.85, fontsize=9, edgecolor="#CCCCCC",
    )

    plt.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_obj1] Saved -> {out_path}")


# ── Faceted chart ─────────────────────────────────────────────────────────────

def plot_faceted(data_by_ds: dict, out_path: str) -> None:
    """2×N faceted subplots — one per dataset (excludes 'combined')."""
    configs = list(CONFIG_LABELS.keys())

    # Sort datasets: custom first, then cs1qa, staqc, mbpp; skip combined for subplots
    datasets = sorted(
        [ds for ds in data_by_ds if ds != "combined"],
        key=lambda d: (DATASET_ORDER.index(d) if d in DATASET_ORDER else 99, d),
    )
    if not datasets:
        print("[plot_obj1] No datasets found for faceted chart.")
        return

    ncols = 2
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows), squeeze=False)
    fig.patch.set_facecolor("#FAFAF8")

    for i, ds in enumerate(datasets):
        row_i, col_i = divmod(i, ncols)
        ax = axes[row_i][col_i]
        ax.set_facecolor("#FAFAF8")
        n = max((data_by_ds[ds].get(c, {}).get("n", 0) for c in configs), default=0)
        title = f"{DATASET_TITLES.get(ds, ds)}  (n={n:,})"
        _draw_panel(ax, data_by_ds[ds], configs, title)
        if col_i == 0:
            ax.set_ylabel("Score", fontsize=9, labelpad=6)
        ax.set_xlabel("Pipeline Configuration", fontsize=8.5, labelpad=6)

    # Hide unused axes
    for j in range(len(datasets), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=COLOR_MRR,  label="MRR@10"),
        mpatches.Patch(color=COLOR_NDCG, label="nDCG@10"),
        mpatches.Patch(color=COLOR_CTX,  label="Context Precision"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=10,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Objective 1 — Retrieval Metrics by Dataset and Pipeline Configuration",
        fontsize=13, fontweight="bold", y=1.01, color="#1A1A1A",
    )

    plt.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_obj1] Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="Path to results CSV. If it has a 'dataset' column, produces faceted chart.")
    parser.add_argument("--out", default="bench/figures/fig2_obj1.png",
                        help="Output PNG path")
    args = parser.parse_args()

    if args.csv:
        result, has_ds = load_csv(args.csv)
        if has_ds:
            plot_faceted(result, args.out)
        else:
            data_by_cfg, n = result
            plot_single(data_by_cfg, n, args.out)
    else:
        # Default: use canonical 937-query numbers as single chart
        plot_single(CANONICAL, CANONICAL_N, args.out)


if __name__ == "__main__":
    main()
