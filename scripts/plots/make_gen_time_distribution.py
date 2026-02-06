#!/usr/bin/env python3
"""
scripts/make_gen_time_distribution.py

Plot 7 — Generation Time Distribution (faceted by dataset)
- One figure with subplots (one per dataset)
- In each subplot: boxplot with X=model, Y=gen_time_s
- Optional log-scale Y axis

Outputs:
- docs/tables/gen_time_summary_by_model_dataset.csv
- docs/figures/gen_time_distribution_by_model__faceted_by_dataset.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- aggregation table ----------

def build_table(df: pd.DataFrame) -> pd.DataFrame:
    required = ["model", "dataset", "gen_time_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    df = df.copy()
    df["model"] = df["model"].astype(str)
    df["dataset"] = df["dataset"].astype(str)
    df["gen_time_s"] = pd.to_numeric(df["gen_time_s"], errors="coerce")
    df = df.dropna(subset=["gen_time_s"])

    g = df.groupby(["dataset", "model"], dropna=False)["gen_time_s"]

    out = g.agg(
        n="size",
        mean_gen_time_s="mean",
        median_gen_time_s="median",
        p90_gen_time_s=lambda s: s.quantile(0.90),
        p95_gen_time_s=lambda s: s.quantile(0.95),
        max_gen_time_s="max",
    ).reset_index()

    out[["mean_gen_time_s", "median_gen_time_s", "p90_gen_time_s", "p95_gen_time_s", "max_gen_time_s"]] = (
        out[["mean_gen_time_s", "median_gen_time_s", "p90_gen_time_s", "p95_gen_time_s", "max_gen_time_s"]].round(6)
    )

    out = out.sort_values(["dataset", "median_gen_time_s", "model"], ascending=[True, True, True]).reset_index(drop=True)
    return out


# ---------- plotting ----------

def plot_faceted_boxplot_by_dataset(df: pd.DataFrame, out_png: Path, log_y: bool) -> None:
    required = ["model", "dataset", "gen_time_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    out_png.parent.mkdir(parents=True, exist_ok=True)

    dff = df.copy()
    dff["model"] = dff["model"].astype(str)
    dff["dataset"] = dff["dataset"].astype(str)
    dff["gen_time_s"] = pd.to_numeric(dff["gen_time_s"], errors="coerce")
    dff = dff.dropna(subset=["gen_time_s"])

    datasets = sorted(dff["dataset"].unique().tolist())

    n = len(datasets)
    if n == 4:
        ncols, nrows = 2, 2
    else:
        ncols = min(3, max(1, n))
        nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.6 * ncols, 4.2 * nrows),
        squeeze=False,
    )
    fig.suptitle("Generation Time Distribution by Model (faceted by dataset)", y=1.02)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        sub = dff[dff["dataset"] == dataset].copy()

        # Order models by median gen time within this dataset (stable and meaningful)
        order = sub.groupby("model")["gen_time_s"].median().sort_values().index.astype(str).tolist()
        data = [sub[sub["model"] == m]["gen_time_s"].to_numpy() for m in order]

        ax.boxplot(data, labels=order, showfliers=True)
        ax.set_title(str(dataset))
        ax.set_xlabel("Model")
        ax.set_ylabel("Generation Time (s)")
        ax.tick_params(axis="x", rotation=20)

        if log_y:
            ax.set_yscale("log")
            ax.set_ylabel("Generation Time (s, log scale)")

        ax.grid(True, axis="y", alpha=0.3)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="results/master_results.csv")
    ap.add_argument("--docs_dir", default="docs")
    ap.add_argument("--log_y", action="store_true", help="Use log-scale y axis")
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

    # Table: per dataset × model summary
    tbl = build_table(df)
    out_table = tables_dir / "gen_time_summary_by_model_dataset.csv"
    tbl.to_csv(out_table, index=False, encoding="utf-8")

    # Plot: faceted by dataset, x=model
    out_plot = figures_dir / "gen_time_distribution_by_model__faceted_by_dataset.png"
    plot_faceted_boxplot_by_dataset(df, out_plot, log_y=args.log_y)

    print(f"✅ Wrote table: {out_table}")
    print(f"✅ Wrote plot : {out_plot}")


if __name__ == "__main__":
    main()
