#!/usr/bin/env python3
"""
scripts/make_core_outcome_metrics.py

Core Outcome Metrics:
- Reads the master CSV
- Writes aggregated tables to docs/tables/
- Writes plots to docs/figures/

Metric definitions (fixed):
- execution accuracy = pred_vs_gold_match
- pred_success = SQL executes without error
- valid_prediction = syntactically valid SQL
- successful_execution = valid_prediction AND pred_success

Figures produced by this script:
- Grouped bar chart of execution accuracy (%) by dataset, with an "Overall" group first.
  Each x-group contains bars for every (model, rdbms) combination.
- Grouped bar chart of pred_success rate (%) by dataset, with an "Overall" group first.
  Each x-group contains bars for every (model, rdbms) combination.

Notes:
- Bars are annotated with values (important when rates are very low).
- y-axis limits are set dynamically to avoid flattening tiny values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import as_bool_series


# ---------- helpers ----------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Boolean normalization
    for c in [
        "pred_success",
        "gold_success",
        "pred_vs_gold_match",
        "valid_prediction",
        "has_repairs",
        "successful_execution",
    ]:
        if c in df.columns:
            df[c] = as_bool_series(df[c])

    # Enforce definition:
    # successful_execution = valid_prediction AND pred_success
    if {"valid_prediction", "pred_success"} <= set(df.columns):
        df["successful_execution"] = (
            df["valid_prediction"].fillna(False)
            & df["pred_success"].fillna(False)
        ).astype(bool)

    # Numeric coercion
    num_cols = [
        "difficulty",
        "schema_num_tables",
        "schema_num_columns",
        "prompt_tokens",
        "gen_time_s",
        "pred_execution_time_s",
        "gold_execution_time_s",
        "num_table_repairs",
        "num_column_repairs",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _annotate_bars(ax: plt.Axes, bars, fmt: str) -> None:
    """Write value labels above each bar (works even for tiny bars)."""
    for b in bars:
        h = b.get_height()
        if not np.isfinite(h):
            continue
        x = b.get_x() + b.get_width() / 2.0
        ax.text(x, h + 0.3, fmt.format(h), ha="center", va="bottom", fontsize=8)


def _set_dynamic_percent_ylim(values: np.ndarray) -> None:
    """Set y-limits based on data (nice for low values)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vmax = float(vals.max()) if vals.size else 0.0
    pad = max(2.0, 0.10 * vmax)  # at least 2 percentage points of headroom
    plt.ylim(0.0, min(100.0, vmax + pad))


def _fmt_for_percent_values(values: np.ndarray) -> str:
    """
    Choose label precision based on magnitude.
    If values are extremely small, show more decimals.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vmax = float(vals.max()) if vals.size else 0.0
    if vmax < 0.01:
        return "{:.5f}%"
    if vmax < 0.1:
        return "{:.4f}%"
    if vmax < 1.0:
        return "{:.3f}%"
    return "{:.2f}%"


# ---------- aggregation ----------

def build_core_outcome_table(df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = ["model", "dataset", "rdbms"]
    g = df.groupby(grp_cols, dropna=False)

    summary = g.agg(
        n_queries=("row_id", "count"),
        execution_accuracy_pct=("pred_vs_gold_match", lambda s: s.mean() * 100.0),
        pred_success_rate_pct=("pred_success", lambda s: s.mean() * 100.0),
        valid_sql_rate_pct=("valid_prediction", lambda s: s.mean() * 100.0),
        successful_execution_rate_pct=("successful_execution", lambda s: s.mean() * 100.0),
        gold_success_rate_pct=("gold_success", lambda s: s.mean() * 100.0),
        has_repairs_rate_pct=("has_repairs", lambda s: s.mean() * 100.0),
        avg_gen_time_s=("gen_time_s", "mean"),
        median_gen_time_s=("gen_time_s", "median"),
        avg_prompt_tokens=("prompt_tokens", "mean"),
        avg_pred_execution_time_s=("pred_execution_time_s", "mean"),
        avg_gold_execution_time_s=("gold_execution_time_s", "mean"),
        avg_schema_num_tables=("schema_num_tables", "mean"),
        avg_schema_num_columns=("schema_num_columns", "mean"),
    ).reset_index()

    pct_cols = [c for c in summary.columns if c.endswith("_pct")]
    summary[pct_cols] = summary[pct_cols].round(6)

    float_cols = [
        "avg_gen_time_s",
        "median_gen_time_s",
        "avg_pred_execution_time_s",
        "avg_gold_execution_time_s",
    ]
    summary[float_cols] = summary[float_cols].round(6)

    summary[["avg_prompt_tokens", "avg_schema_num_tables", "avg_schema_num_columns"]] = (
        summary[["avg_prompt_tokens", "avg_schema_num_tables", "avg_schema_num_columns"]].round(2)
    )

    return summary.sort_values(["dataset", "rdbms", "model"]).reset_index(drop=True)


# ---------- plotting (generic grouped by (model, rdbms)) ----------

def _plot_rate_by_dataset_model_rdbms(
    df: pd.DataFrame,
    out_png: Path,
    metric_col: str,
    ylabel: str,
    title: str,
    models: list[str],
    datasets: list[str],
    rdbms_list: list[str] | None = None,
) -> None:
    """
    Grouped bar chart:
    X-axis: Overall, then datasets.
    Bars per x-group: one bar for each (model, rdbms) combination.
    """
    required = {"model", "dataset", "rdbms", metric_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Need columns {sorted(required)}. Missing: {missing}")

    out_png.parent.mkdir(parents=True, exist_ok=True)

    dff = df.copy()
    dff["model"] = dff["model"].astype(str)
    dff["dataset"] = dff["dataset"].astype(str)
    dff["rdbms"] = dff["rdbms"].astype(str)

    # normalize metric to bool then to percent rate
    dff[metric_col] = as_bool_series(dff[metric_col]).fillna(False).astype(bool)

    # choose models/rdbms actually present
    models_present = [m for m in models if m in set(dff["model"].unique())] if models else sorted(dff["model"].unique())
    if not models_present:
        raise ValueError(f"No requested models found. Requested={models}, found={sorted(dff['model'].unique())}")

    if rdbms_list is None:
        rdbms_present = sorted(dff["rdbms"].unique().tolist())
    else:
        rdbms_present = [r for r in rdbms_list if r in set(dff["rdbms"].unique())]
        if not rdbms_present:
            raise ValueError(f"No requested RDBMS found. Requested={rdbms_list}, found={sorted(dff['rdbms'].unique())}")

    target_datasets = ["Overall"] + datasets

    # Per (model, rdbms, dataset)
    per_mrd = (
        dff.groupby(["model", "rdbms", "dataset"], dropna=False)[metric_col]
        .mean()
        .mul(100.0)
        .reset_index(name="rate_pct")
    )

    # Overall per (model, rdbms)
    per_mr_overall = (
        dff.groupby(["model", "rdbms"], dropna=False)[metric_col]
        .mean()
        .mul(100.0)
        .reset_index(name="rate_pct")
    )
    per_mr_overall["dataset"] = "Overall"

    combined = pd.concat([per_mr_overall, per_mrd], ignore_index=True)
    combined = combined[combined["model"].isin(models_present)]
    combined = combined[combined["rdbms"].isin(rdbms_present)]
    combined = combined[combined["dataset"].isin(target_datasets)]

    # Ensure all combinations exist
    idx = pd.MultiIndex.from_product(
        [models_present, rdbms_present, target_datasets],
        names=["model", "rdbms", "dataset"],
    )
    combined = (
        combined.set_index(["model", "rdbms", "dataset"])
        .reindex(idx)
        .fillna({"rate_pct": 0.0})
        .reset_index()
    )

    # Build a "series" label = model|rdbms (so legend distinguishes all bars)
    combined["series"] = combined["model"] + " / " + combined["rdbms"]

    # Pivot to dataset rows × series cols
    series_order = [f"{m} / {r}" for m in models_present for r in rdbms_present]
    pivot = (
        combined.pivot(index="dataset", columns="series", values="rate_pct")
        .reindex(index=target_datasets, columns=series_order)
        .fillna(0.0)
    )

    x_labels = pivot.index.tolist()
    x = np.arange(len(x_labels))

    n_series = len(series_order)
    width = 0.8 / max(1, n_series)
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * width

    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    all_vals = pivot.to_numpy(dtype=float).ravel()
    fmt = _fmt_for_percent_values(all_vals)

    for i, sname in enumerate(series_order):
        y = pivot[sname].to_numpy(dtype=float)
        bars = ax.bar(x + offsets[i], y, width=width, label=sname)
        _annotate_bars(ax, bars, fmt=fmt)

    _set_dynamic_percent_ylim(all_vals)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.legend(ncol=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_execution_accuracy_by_dataset_model_rdbms(
    df: pd.DataFrame,
    out_png: Path,
    models: list[str],
    datasets: list[str],
    rdbms_list: list[str] | None = None,
) -> None:
    _plot_rate_by_dataset_model_rdbms(
        df=df,
        out_png=out_png,
        metric_col="pred_vs_gold_match",
        ylabel="Execution Accuracy (%)",
        title="Execution Accuracy by Dataset (Overall + per dataset; model × RDBMS)",
        models=models,
        datasets=datasets,
        rdbms_list=rdbms_list,
    )


def plot_pred_success_rate_by_dataset_model_rdbms(
    df: pd.DataFrame,
    out_png: Path,
    models: list[str],
    datasets: list[str],
    rdbms_list: list[str] | None = None,
) -> None:
    _plot_rate_by_dataset_model_rdbms(
        df=df,
        out_png=out_png,
        metric_col="pred_success",
        ylabel="Predicted SQL Executable Rate (%)",
        title="Predicted SQL Executable Rate by Dataset (Overall + per dataset; model × RDBMS)",
        models=models,
        datasets=datasets,
        rdbms_list=rdbms_list,
    )


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", required=True)
    ap.add_argument("--docs_dir", default="docs")
    ap.add_argument("--models", default="gpt2xl,qwen", help="Comma-separated model names (default: gpt2xl,qwen)")
    ap.add_argument("--datasets", default="imdb,yelp,atis,advising", help="Comma-separated datasets (default: imdb,yelp,atis,advising)")
    ap.add_argument("--rdbms", default="", help="Comma-separated rdbms list (default: all found in CSV)")
    args = ap.parse_args()

    master_csv = Path(args.master_csv)
    if not master_csv.exists():
        raise FileNotFoundError(master_csv)

    docs_dir = Path(args.docs_dir)
    tables_dir = docs_dir / "tables"
    figures_dir = docs_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rdbms_list = [r.strip() for r in args.rdbms.split(",") if r.strip()] or None

    df_raw = pd.read_csv(master_csv)
    df = normalize_columns(df_raw)

    # Table (unchanged)
    summary = build_core_outcome_table(df)
    out_table = tables_dir / "core_outcome_summary.csv"
    summary.to_csv(out_table, index=False)

    # Plot: execution accuracy by dataset with bars for (model, rdbms)
    out_plot_acc = figures_dir / "core_outcome" / "accuracy_by_dataset_model_rdbms.png"
    plot_execution_accuracy_by_dataset_model_rdbms(
        df, out_plot_acc, models=models, datasets=datasets, rdbms_list=rdbms_list
    )

    # Plot: pred_success rate by dataset with bars for (model, rdbms)
    out_plot_exec = figures_dir / "core_outcome" / "pred_success_by_dataset_model_rdbms.png"
    plot_pred_success_rate_by_dataset_model_rdbms(
        df, out_plot_exec, models=models, datasets=datasets, rdbms_list=rdbms_list
    )

    print(f"✅ Wrote table: {out_table}")
    print(f"✅ Wrote plot : {out_plot_acc}")
    print(f"✅ Wrote plot : {out_plot_exec}")


if __name__ == "__main__":
    main()
