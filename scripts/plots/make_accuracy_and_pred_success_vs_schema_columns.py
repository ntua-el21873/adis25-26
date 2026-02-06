#!/usr/bin/env python3
"""
scripts/make_accuracy_and_pred_success_vs_schema_columns.py

Plot 5 — Accuracy & Pred Success vs Number of Columns (schema_num_columns)
(refactored to match make_accuracy_and_pred_success_vs_prompt_tokens.py)

Aggregation:
- model × dataset × rdbms × schema_columns_bin

Binning (fixed, paper-friendly):
- 1–10, 11–20, 21–40, 41+

Faceting strategy (same as prompt_tokens script):
- One figure per dataset
- Within each figure: subplots per rdbms
- In each subplot: lines for each model across schema column bins

Outputs:
- docs/tables/accuracy_vs_schema_columns_by_model_dataset_rdbms.csv
- docs/tables/pred_success_vs_schema_columns_by_model_dataset_rdbms.csv
- docs/figures/accuracy_vs_schema_columns__<dataset>.png
- docs/figures/pred_success_vs_schema_columns__<dataset>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import as_bool_series


# ---------- helpers ----------

BIN_LABELS = ["1–10", "11–20", "21–40", "41+"]


def add_schema_columns_bins(df: pd.DataFrame, col: str = "schema_num_columns") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    df[col] = df[col].astype(int)

    bins = [-np.inf, 10, 20, 40, np.inf]
    labels = BIN_LABELS
    df["schema_columns_bin"] = pd.cut(
        df[col],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    df["schema_columns_bin"] = pd.Categorical(df["schema_columns_bin"], categories=labels, ordered=True)
    return df


def _dynamic_ylim_percent(ax, y_values: np.ndarray) -> None:
    vals = np.asarray(y_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vmax = float(vals.max()) if vals.size else 0.0
    pad = max(2.0, 0.10 * vmax)
    ax.set_ylim(0.0, min(100.0, vmax + pad))


def _annotate_points(ax, x: np.ndarray, y: np.ndarray) -> None:
    # annotate only non-zero points to reduce clutter
    for xi, yi in zip(x, y):
        if not np.isfinite(yi) or yi == 0:
            continue
        ax.annotate(
            f"{yi:.3f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            va="bottom",
            fontsize=8,
        )


# ---------- table building ----------

def build_table(df: pd.DataFrame, metric_col: str, out_value_name: str) -> pd.DataFrame:
    """
    Builds a binned table aggregated by:
      model × dataset × rdbms × schema_columns_bin

    metric_col:
      - "pred_vs_gold_match" (accuracy)
      - "pred_success"       (executable rate)
    """
    required = ["model", "dataset", "rdbms", "schema_num_columns", metric_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    df = df.copy()
    df["model"] = df["model"].astype(str)
    df["dataset"] = df["dataset"].astype(str)
    df["rdbms"] = df["rdbms"].astype(str)

    df[metric_col] = as_bool_series(df[metric_col]).fillna(False).astype(bool)
    df = add_schema_columns_bins(df, col="schema_num_columns")

    g = df.groupby(["model", "dataset", "rdbms", "schema_columns_bin"], dropna=False)

    out = g.agg(
        n_queries=(metric_col, "size"),
        p=(metric_col, "mean"),
    ).reset_index()

    out[out_value_name] = (out["p"] * 100.0).round(6)
    out = out.drop(columns=["p"])

    out = out.sort_values(["dataset", "rdbms", "schema_columns_bin", "model"]).reset_index(drop=True)
    return out


# ---------- plotting ----------

def plot_faceted_by_dataset(
    tbl: pd.DataFrame,
    figures_dir: Path,
    value_col: str,
    title_prefix: str,
    filename_prefix: str,
    annotate_points: bool = True,
) -> list[Path]:
    """
    One figure per dataset. Subplots per rdbms. Lines per model.
    X-axis is schema column bin order (1–10, 11–20, 21–40, 41+).
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    datasets = sorted(tbl["dataset"].astype(str).unique().tolist())
    models = sorted(tbl["model"].astype(str).unique().tolist())

    # x positions for the categorical bins
    x_labels = BIN_LABELS
    x_pos = np.arange(len(x_labels))

    for dataset in datasets:
        ds_tbl = tbl[tbl["dataset"].astype(str) == dataset].copy()
        rdbms_list = sorted(ds_tbl["rdbms"].astype(str).unique().tolist())

        n = len(rdbms_list)
        ncols = min(3, max(1, n))
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.6 * ncols, 4.0 * nrows),
            squeeze=False,
        )
        fig.suptitle(f"{title_prefix} vs Schema Columns (binned) — dataset={dataset}", y=1.02)

        for idx, rdbms in enumerate(rdbms_list):
            ax = axes[idx // ncols][idx % ncols]
            sub = ds_tbl[ds_tbl["rdbms"].astype(str) == rdbms].copy()

            ax.set_title(str(rdbms))
            ax.set_xlabel("Schema #Columns (binned)")
            ax.set_ylabel(f"{title_prefix} (%)")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)

            all_y = []

            for model in models:
                msub = sub[sub["model"].astype(str) == model].copy()
                if msub.empty:
                    continue

                # Ensure bin order
                msub["schema_columns_bin"] = pd.Categorical(msub["schema_columns_bin"], categories=BIN_LABELS, ordered=True)
                msub = msub.sort_values("schema_columns_bin")

                # Align on all bins (missing -> 0)
                msub = (
                    msub.set_index("schema_columns_bin")
                    .reindex(BIN_LABELS)
                    .fillna({value_col: 0.0})
                    .reset_index()
                )

                y = msub[value_col].to_numpy(dtype=float)
                all_y.append(y)

                ax.plot(x_pos, y, marker="o", label=model)

                if annotate_points:
                    _annotate_points(ax, x_pos, y)

            if all_y:
                _dynamic_ylim_percent(ax, np.concatenate(all_y))
            else:
                ax.set_ylim(0, 1)

            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=9)

        # Hide unused axes
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.tight_layout()
        out_png = figures_dir / f"{filename_prefix}" / f"{dataset}.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(out_png)

    return written


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="results/master_results.csv")
    ap.add_argument("--docs_dir", default="docs")
    ap.add_argument("--no_point_labels", action="store_true", help="Disable point value annotations")
    args = ap.parse_args()

    master_csv = Path(args.master_csv)
    if not master_csv.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")

    docs_dir = Path(args.docs_dir)
    tables_dir = docs_dir / "tables"
    figures_dir = docs_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(master_csv)

    # 1) Accuracy table + plots
    tbl_acc = build_table(
        df,
        metric_col="pred_vs_gold_match",
        out_value_name="execution_accuracy_pct",
    )
    out_table_acc = tables_dir / "accuracy_vs_schema_columns_by_model_dataset_rdbms.csv"
    tbl_acc.to_csv(out_table_acc, index=False, encoding="utf-8")

    written_acc = plot_faceted_by_dataset(
        tbl_acc,
        figures_dir=figures_dir,
        value_col="execution_accuracy_pct",
        title_prefix="Execution Accuracy",
        filename_prefix="accuracy_vs_schema_columns",
        annotate_points=(not args.no_point_labels),
    )

    # 2) Pred success table + plots
    tbl_exec = build_table(
        df,
        metric_col="pred_success",
        out_value_name="pred_success_rate_pct",
    )
    out_table_exec = tables_dir / "pred_success_vs_schema_columns_by_model_dataset_rdbms.csv"
    tbl_exec.to_csv(out_table_exec, index=False, encoding="utf-8")

    written_exec = plot_faceted_by_dataset(
        tbl_exec,
        figures_dir=figures_dir,
        value_col="pred_success_rate_pct",
        title_prefix="Predicted SQL Executable Rate",
        filename_prefix="pred_success_vs_schema_columns",
        annotate_points=(not args.no_point_labels),
    )

    print(f"✅ Wrote table: {out_table_acc}")
    for p in written_acc:
        print(f"✅ Wrote plot : {p}")

    print(f"✅ Wrote table: {out_table_exec}")
    for p in written_exec:
        print(f"✅ Wrote plot : {p}")


if __name__ == "__main__":
    main()
