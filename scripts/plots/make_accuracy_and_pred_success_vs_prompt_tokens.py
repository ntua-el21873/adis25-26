#!/usr/bin/env python3
"""
scripts/make_accuracy_and_pred_success_vs_prompt_tokens.py

Plot 6 — Accuracy & Pred Success vs Prompt Tokens (binned; aggregated by model×dataset×rdbms)

You asked to aggregate by: model, dataset, rdbms (and prompt_tokens bins).

This script produces TWO analyses (and plots) using the same binning:
1) execution accuracy = pred_vs_gold_match
2) executable rate     = pred_success

Faceting strategy (readable, research-grade):
- One figure per dataset
- Within each figure: subplots per rdbms
- In each subplot: lines for each model over prompt_tokens (bin mean)

Outputs:
- docs/tables/accuracy_vs_prompt_tokens_by_model_dataset_rdbms.csv
- docs/tables/pred_success_vs_prompt_tokens_by_model_dataset_rdbms.csv
- docs/figures/accuracy_vs_prompt_tokens__<dataset>.png
- docs/figures/pred_success_vs_prompt_tokens__<dataset>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import as_bool_series, _dynamic_ylim_percent


# ---------- helpers ----------

def make_global_bins(x: pd.Series, n_bins: int) -> tuple[np.ndarray, list[str]]:
    """
    Global quantile bins across ALL rows (shared bins across model/dataset/rdbms).
    This makes comparisons fair (same x-bins across groups).
    """
    x = x.dropna()
    if x.empty:
        return np.array([0, 1]), ["0–1"]

    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x.to_numpy(), q))

    # Fallback if quantiles collapse
    if len(edges) < 3:
        lo = float(x.min())
        hi = float(x.max())
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, n_bins + 1)

    labels = [f"{int(edges[i])}–{int(edges[i+1])}" for i in range(len(edges) - 1)]
    return edges, labels


def _annotate_points(ax, x: np.ndarray, y: np.ndarray) -> None:
    """
    Light annotation: only annotate non-zero points to avoid clutter.
    """
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

def build_table(df: pd.DataFrame, n_bins: int, metric_col: str, out_value_name: str) -> pd.DataFrame:
    """
    Builds a binned table aggregated by:
      model × dataset × rdbms × prompt_tokens_bin

    metric_col:
      - "pred_vs_gold_match" (accuracy)
      - "pred_success"       (executable rate)
    """
    required = ["model", "dataset", "rdbms", "prompt_tokens", metric_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    df = df.copy()
    df["model"] = df["model"].astype(str)
    df["dataset"] = df["dataset"].astype(str)
    df["rdbms"] = df["rdbms"].astype(str)

    df[metric_col] = as_bool_series(df[metric_col]).fillna(False).astype(bool)

    df["prompt_tokens"] = pd.to_numeric(df["prompt_tokens"], errors="coerce")
    df = df.dropna(subset=["prompt_tokens"])
    df["prompt_tokens"] = df["prompt_tokens"].astype(int)

    edges, labels = make_global_bins(df["prompt_tokens"], n_bins=n_bins)

    df["prompt_tokens_bin"] = pd.cut(
        df["prompt_tokens"],
        bins=edges,
        labels=labels[: len(edges) - 1],
        include_lowest=True,
        right=True,
    )

    g = df.groupby(["model", "dataset", "rdbms", "prompt_tokens_bin"], dropna=False)

    out = g.agg(
        n_queries=(metric_col, "size"),
        p=(metric_col, "mean"),
        prompt_tokens_mean=("prompt_tokens", "mean"),
    ).reset_index()

    out[out_value_name] = (out["p"] * 100.0).round(6)
    out["prompt_tokens_mean"] = out["prompt_tokens_mean"].round(2)

    out = out.drop(columns=["p"])

    # Keep stable bin ordering
    out["prompt_tokens_bin"] = pd.Categorical(out["prompt_tokens_bin"], categories=labels, ordered=True)

    out = out.sort_values(["dataset", "rdbms", "prompt_tokens_bin", "model"]).reset_index(drop=True)
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
    X-axis is prompt_tokens_mean.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    datasets = sorted(tbl["dataset"].astype(str).unique().tolist())
    models = sorted(tbl["model"].astype(str).unique().tolist())

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
        fig.suptitle(f"{title_prefix} vs Prompt Tokens (binned) — dataset={dataset}", y=1.02)

        for idx, rdbms in enumerate(rdbms_list):
            ax = axes[idx // ncols][idx % ncols]
            sub = ds_tbl[ds_tbl["rdbms"].astype(str) == rdbms].copy()

            ax.set_title(str(rdbms))
            ax.set_xlabel("Prompt Tokens (bin mean)")
            ax.set_ylabel(f"{title_prefix} (%)")

            all_y = []

            for model in models:
                msub = sub[sub["model"].astype(str) == model].copy()
                if msub.empty:
                    continue

                msub = msub.dropna(subset=["prompt_tokens_mean"]).sort_values("prompt_tokens_mean")

                x = msub["prompt_tokens_mean"].to_numpy(dtype=float)
                y = msub[value_col].to_numpy(dtype=float)
                all_y.append(y)

                ax.plot(x, y, marker="o", label=model)

                if annotate_points:
                    _annotate_points(ax, x, y)

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
    ap.add_argument("--bins", type=int, default=8, help="Number of global quantile bins (default: 8)")
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
        n_bins=args.bins,
        metric_col="pred_vs_gold_match",
        out_value_name="execution_accuracy_pct",
    )
    out_table_acc = tables_dir / "accuracy_vs_prompt_tokens_by_model_dataset_rdbms.csv"
    tbl_acc.to_csv(out_table_acc, index=False, encoding="utf-8")

    written_acc = plot_faceted_by_dataset(
        tbl_acc,
        figures_dir=figures_dir,
        value_col="execution_accuracy_pct",
        title_prefix="Execution Accuracy",
        filename_prefix="accuracy_vs_prompt_tokens",
        annotate_points=(not args.no_point_labels),
    )

    # 2) Pred success table + plots
    tbl_exec = build_table(
        df,
        n_bins=args.bins,
        metric_col="pred_success",
        out_value_name="pred_success_rate_pct",
    )
    out_table_exec = tables_dir / "pred_success_vs_prompt_tokens_by_model_dataset_rdbms.csv"
    tbl_exec.to_csv(out_table_exec, index=False, encoding="utf-8")

    written_exec = plot_faceted_by_dataset(
        tbl_exec,
        figures_dir=figures_dir,
        value_col="pred_success_rate_pct",
        title_prefix="Predicted SQL Executable Rate",
        filename_prefix="pred_success_vs_prompt_tokens",
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
