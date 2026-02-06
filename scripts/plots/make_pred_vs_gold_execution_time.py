#!/usr/bin/env python3
"""
scripts/make_pred_vs_gold_execution_time.py

Compare predicted SQL execution time vs gold SQL execution time.

You asked:
- Barplot where X-axis groups are: Overall + each dataset
- Each X point has 4 bars (model × rdbms), same structure as previous scripts
- Bars represent average execution time (seconds)
- Show BOTH pred and gold in ONE figure, without exploding to 8 bars:
  -> Use two panels (subplots) in the same figure:
     (left) Pred avg execution time
     (right) Gold avg execution time
  This keeps the "4 bars per x point" invariant.

Outputs:
- docs/tables/pred_vs_gold_execution_time_by_dataset_model_rdbms.csv
- docs/figures/pred_vs_gold_execution_time_by_dataset_model_rdbms.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- helpers ----------

def _annotate_bars(ax: plt.Axes, bars, values: np.ndarray, fmt: str = "{:.3f}s") -> None:
    for rect, v in zip(bars, values):
        if not np.isfinite(v):
            continue
        x = rect.get_x() + rect.get_width() / 2.0
        y = rect.get_height()
        ax.text(x, y + 0.01 * max(1.0, float(np.nanmax(values))), fmt.format(v), ha="center", va="bottom", fontsize=8)


def _dynamic_ylim(ax: plt.Axes, values: np.ndarray) -> None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vmax = float(vals.max()) if vals.size else 0.0
    pad = max(0.05, 0.10 * vmax)
    ax.set_ylim(0.0, vmax + pad)


# ---------- aggregation ----------

def build_table(df: pd.DataFrame) -> pd.DataFrame:
    required = ["model", "dataset", "rdbms", "pred_execution_time_s", "gold_execution_time_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    dff = df.copy()
    dff["model"] = dff["model"].astype(str)
    dff["dataset"] = dff["dataset"].astype(str)
    dff["rdbms"] = dff["rdbms"].astype(str)

    dff["pred_execution_time_s"] = pd.to_numeric(dff["pred_execution_time_s"], errors="coerce")
    dff["gold_execution_time_s"] = pd.to_numeric(dff["gold_execution_time_s"], errors="coerce")

    # Per dataset × model × rdbms
    per = (
        dff.groupby(["dataset", "model", "rdbms"], dropna=False)
        .agg(
            n=("model", "size"),
            pred_exec_time_mean=("pred_execution_time_s", "mean"),
            gold_exec_time_mean=("gold_execution_time_s", "mean"),
            pred_exec_time_median=("pred_execution_time_s", "median"),
            gold_exec_time_median=("gold_execution_time_s", "median"),
        )
        .reset_index()
    )

    # Overall per model × rdbms (across all datasets)
    overall = (
        dff.groupby(["model", "rdbms"], dropna=False)
        .agg(
            n=("model", "size"),
            pred_exec_time_mean=("pred_execution_time_s", "mean"),
            gold_exec_time_mean=("gold_execution_time_s", "mean"),
            pred_exec_time_median=("pred_execution_time_s", "median"),
            gold_exec_time_median=("gold_execution_time_s", "median"),
        )
        .reset_index()
    )
    overall["dataset"] = "Overall"

    out = pd.concat([overall, per], ignore_index=True)

    # Round for reporting
    for c in ["pred_exec_time_mean", "gold_exec_time_mean", "pred_exec_time_median", "gold_exec_time_median"]:
        out[c] = out[c].round(6)

    out = out.sort_values(["dataset", "rdbms", "model"]).reset_index(drop=True)
    return out


# ---------- plotting ----------

def plot_two_panel_barplot(
    tbl: pd.DataFrame,
    out_png: Path,
    models: list[str],
    datasets: list[str],
    rdbms_list: list[str] | None,
    use_median: bool = False,
    log_y: bool = False,
) -> None:
    """
    Two panels in one figure:
      Left: pred execution time
      Right: gold execution time

    Each panel: X groups = Overall + datasets, 4 bars per group = model×rdbms
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    dff = tbl.copy()
    dff["model"] = dff["model"].astype(str)
    dff["dataset"] = dff["dataset"].astype(str)
    dff["rdbms"] = dff["rdbms"].astype(str)

    models_present = [m for m in models if m in set(dff["model"].unique())] if models else sorted(dff["model"].unique())
    if not models_present:
        raise ValueError("No requested models found in table.")

    if rdbms_list is None:
        rdbms_present = sorted(dff["rdbms"].unique().tolist())
    else:
        rdbms_present = [r for r in rdbms_list if r in set(dff["rdbms"].unique())]
        if not rdbms_present:
            raise ValueError("No requested rdbms found in table.")

    target_datasets = ["Overall"] + datasets

    # columns to plot
    pred_col = "pred_exec_time_median" if use_median else "pred_exec_time_mean"
    gold_col = "gold_exec_time_median" if use_median else "gold_exec_time_mean"

    # Ensure all combinations exist (missing -> 0)
    idx = pd.MultiIndex.from_product(
        [target_datasets, models_present, rdbms_present],
        names=["dataset", "model", "rdbms"],
    )
    dff = (
        dff.set_index(["dataset", "model", "rdbms"])
        .reindex(idx)
        .fillna({pred_col: 0.0, gold_col: 0.0, "n": 0})
        .reset_index()
    )

    dff["series"] = dff["model"] + " / " + dff["rdbms"]
    series_order = [f"{m} / {r}" for m in models_present for r in rdbms_present]

    pred_pivot = (
        dff.pivot(index="dataset", columns="series", values=pred_col)
        .reindex(index=target_datasets, columns=series_order)
        .fillna(0.0)
    )
    gold_pivot = (
        dff.pivot(index="dataset", columns="series", values=gold_col)
        .reindex(index=target_datasets, columns=series_order)
        .fillna(0.0)
    )

    x_labels = target_datasets
    x = np.arange(len(x_labels))

    n_series = len(series_order)
    width = 0.8 / max(1, n_series)
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * width

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), sharex=True)
    ax1, ax2 = axes

    # Panel 1: pred
    all_pred = pred_pivot.to_numpy(dtype=float).ravel()
    for i, sname in enumerate(series_order):
        y = pred_pivot[sname].to_numpy(dtype=float)
        bars = ax1.bar(x + offsets[i], y, width=width, label=sname)
        _annotate_bars(ax1, bars, y, fmt="{:.4f}s")
    ax1.set_title("Predicted SQL Avg Execution Time" + (" (median)" if use_median else " (mean)"))
    ax1.set_ylabel("Execution Time (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=0)
    if log_y:
        ax1.set_yscale("log")
        ax1.set_ylabel("Execution Time (s, log scale)")
    else:
        _dynamic_ylim(ax1, all_pred)
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel 2: gold
    all_gold = gold_pivot.to_numpy(dtype=float).ravel()
    for i, sname in enumerate(series_order):
        y = gold_pivot[sname].to_numpy(dtype=float)
        bars = ax2.bar(x + offsets[i], y, width=width, label=sname)
        _annotate_bars(ax2, bars, y, fmt="{:.4f}s")
    ax2.set_title("Gold SQL Avg Execution Time" + (" (median)" if use_median else " (mean)"))
    ax2.set_ylabel("Execution Time (s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=0)
    if log_y:
        ax2.set_yscale("log")
        ax2.set_ylabel("Execution Time (s, log scale)")
    else:
        _dynamic_ylim(ax2, all_gold)
    ax2.grid(True, axis="y", alpha=0.3)

    # One legend for both panels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=9)

    fig.suptitle("Predicted vs Gold SQL Execution Time (Overall + per dataset; model × RDBMS)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="results/master_results.csv", help="Path to master CSV")
    ap.add_argument("--docs_dir", default="docs", help="Docs directory (default: docs)")
    ap.add_argument("--models", default="gpt2xl,qwen", help="Comma-separated model names (default: gpt2xl,qwen)")
    ap.add_argument("--datasets", default="imdb,yelp,atis,advising", help="Comma-separated datasets (default: imdb,yelp,atis,advising)")
    ap.add_argument("--rdbms", default="", help="Comma-separated rdbms list (default: all found in CSV)")
    ap.add_argument("--median", action="store_true", help="Use median instead of mean (more robust)")
    ap.add_argument("--log_y", action="store_true", help="Use log scale on y axis (helpful if heavy-tailed)")
    args = ap.parse_args()

    master_csv = Path(args.master_csv)
    if not master_csv.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")

    docs_dir = Path(args.docs_dir)
    tables_dir = docs_dir / "tables"
    figures_dir = docs_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rdbms_list = [r.strip() for r in args.rdbms.split(",") if r.strip()] or None

    df = pd.read_csv(master_csv)

    tbl = build_table(df)
    out_table = tables_dir / "pred_vs_gold_execution_time_by_dataset_model_rdbms.csv"
    tbl.to_csv(out_table, index=False, encoding="utf-8")

    out_plot = figures_dir / "pred_vs_gold_execution_time_by_dataset_model_rdbms.png"
    plot_two_panel_barplot(
        tbl,
        out_plot,
        models=models,
        datasets=datasets,
        rdbms_list=rdbms_list,
        use_median=args.median,
        log_y=args.log_y,
    )

    print(f"✅ Wrote table: {out_table}")
    print(f"✅ Wrote plot : {out_plot}")


if __name__ == "__main__":
    main()
