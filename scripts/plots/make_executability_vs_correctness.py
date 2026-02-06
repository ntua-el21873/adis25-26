#!/usr/bin/env python3
"""
scripts/make_executability_vs_correctness.py

Executable vs Correctness (stacked bars), grouped like the core-outcome plots.

You asked:
- ONE diagram
- X-axis: Overall + datasets
- For each dataset: 4 bars (model × rdbms)
- Each bar is STACKED with:
  - Executable + Correct
  - Executable + Incorrect
  - Non-Executable

Definitions:
- executable_correct   = pred_success == True AND pred_vs_gold_match == True
- executable_incorrect = pred_success == True AND pred_vs_gold_match == False
- non_executable       = pred_success == False

Outputs:
- docs/tables/executability_vs_correctness_by_dataset_model_rdbms.csv
- docs/figures/executability_vs_correctness_by_dataset_model_rdbms.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import as_bool_series


# ---------- helpers ----------

def _annotate_stack(ax: plt.Axes, bars, values, fmt: str, min_show: float = 1.0) -> None:
    """
    Annotate stacked segments (only if segment >= min_show%) to avoid clutter.
    `bars` are the Rectangles returned by bar(), `values` are heights.
    """
    for rect, v in zip(bars, values):
        if not np.isfinite(v) or v < min_show:
            continue
        x = rect.get_x() + rect.get_width() / 2.0
        y = rect.get_y() + rect.get_height() / 2.0
        ax.text(x, y, fmt.format(v), ha="center", va="center", fontsize=8)


def _fmt_for_percent(values: np.ndarray) -> str:
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

def build_table(df: pd.DataFrame) -> pd.DataFrame:
    required = ["model", "dataset", "rdbms", "pred_success", "pred_vs_gold_match"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    df = df.copy()
    df["model"] = df["model"].astype(str)
    df["dataset"] = df["dataset"].astype(str)
    df["rdbms"] = df["rdbms"].astype(str)

    df["pred_success"] = as_bool_series(df["pred_success"]).fillna(False).astype(bool)
    df["pred_vs_gold_match"] = as_bool_series(df["pred_vs_gold_match"]).fillna(False).astype(bool)

    df["executable_correct"] = df["pred_success"] & df["pred_vs_gold_match"]
    df["executable_incorrect"] = df["pred_success"] & (~df["pred_vs_gold_match"])
    df["non_executable"] = ~df["pred_success"]

    # Per dataset×model×rdbms
    per = (
        df.groupby(["dataset", "model", "rdbms"], dropna=False)
        .agg(
            n_queries=("pred_success", "size"),
            n_executable_correct=("executable_correct", "sum"),
            n_executable_incorrect=("executable_incorrect", "sum"),
            n_non_executable=("non_executable", "sum"),
        )
        .reset_index()
    )

    # Overall per model×rdbms
    overall = (
        df.groupby(["model", "rdbms"], dropna=False)
        .agg(
            n_queries=("pred_success", "size"),
            n_executable_correct=("executable_correct", "sum"),
            n_executable_incorrect=("executable_incorrect", "sum"),
            n_non_executable=("non_executable", "sum"),
        )
        .reset_index()
    )
    overall["dataset"] = "Overall"

    out = pd.concat([overall, per], ignore_index=True)

    # Percentages
    denom = out["n_queries"].replace(0, np.nan).astype(float)
    out["pct_executable_correct"] = (out["n_executable_correct"] / denom * 100.0)
    out["pct_executable_incorrect"] = (out["n_executable_incorrect"] / denom * 100.0)
    out["pct_non_executable"] = (out["n_non_executable"] / denom * 100.0)
    out["pct_sum"] = (
        out["pct_executable_correct"] + out["pct_executable_incorrect"] + out["pct_non_executable"]
    )

    # Round for output readability (keep enough precision for tiny numbers)
    for c in ["pct_executable_correct", "pct_executable_incorrect", "pct_non_executable", "pct_sum"]:
        out[c] = out[c].round(6)

    out = out.sort_values(["dataset", "rdbms", "model"]).reset_index(drop=True)
    return out


# ---------- plotting ----------

def plot_stacked_by_dataset_model_rdbms(
    tbl: pd.DataFrame,
    out_png: Path,
    models: list[str],
    datasets: list[str],
    rdbms_list: list[str] | None = None,
    annotate: bool = True,
) -> None:
    """
    X-axis: Overall + datasets
    For each dataset: one stacked bar per (model, rdbms)
    Stack: correct / incorrect / non-executable
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

    # Ensure all combinations exist (missing -> 0)
    idx = pd.MultiIndex.from_product(
        [target_datasets, models_present, rdbms_present],
        names=["dataset", "model", "rdbms"],
    )
    dff = (
        dff.set_index(["dataset", "model", "rdbms"])
        .reindex(idx)
        .fillna(
            {
                "pct_executable_correct": 0.0,
                "pct_executable_incorrect": 0.0,
                "pct_non_executable": 0.0,
            }
        )
        .reset_index()
    )

    # series order: model major, rdbms minor (same as your other scripts)
    series_order = [(m, r) for m in models_present for r in rdbms_present]
    series_labels = [f"{m} / {r}" for (m, r) in series_order]

    # Build matrices shaped: (datasets, series)
    def get_matrix(col: str) -> np.ndarray:
        pivot = (
            dff.pivot(index="dataset", columns=["model", "rdbms"], values=col)
            .reindex(index=target_datasets, columns=pd.MultiIndex.from_tuples(series_order))
            .fillna(0.0)
        )
        return pivot.to_numpy(dtype=float)

    A = get_matrix("pct_executable_correct")
    B = get_matrix("pct_executable_incorrect")
    C = get_matrix("pct_non_executable")

    x_labels = target_datasets
    x = np.arange(len(x_labels))

    n_series = len(series_order)
    width = 0.8 / max(1, n_series)
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * width

    plt.figure(figsize=(15, 6))
    ax = plt.gca()

    # Decide label precision based on all segment values
    fmt = _fmt_for_percent(np.r_[A.ravel(), B.ravel(), C.ravel()])

    # Draw stacked bars per series
    for i, label in enumerate(series_labels):
        y1 = A[:, i]
        y2 = B[:, i]
        y3 = C[:, i]

        bars1 = ax.bar(x + offsets[i], y1, width=width, label=f"{label} — Exec+Correct")
        bars2 = ax.bar(x + offsets[i], y2, width=width, bottom=y1, label=f"{label} — Exec+Incorrect")
        bars3 = ax.bar(x + offsets[i], y3, width=width, bottom=y1 + y2, label=f"{label} — Non-Exec")

        if annotate:
            _annotate_stack(ax, bars1, y1, fmt=fmt, min_show=1.0)
            _annotate_stack(ax, bars2, y2, fmt=fmt, min_show=1.0)
            _annotate_stack(ax, bars3, y3, fmt=fmt, min_show=1.0)

    ax.set_ylabel("Share of Queries (%)")
    ax.set_xlabel("Dataset")
    ax.set_title("Executability vs Correctness by Dataset (Overall + per dataset; model × RDBMS)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)

    # Since stacks sum to ~100%, keep 0-100 (this is the correct scale for stacked shares)
    ax.set_ylim(0, 100)

    # Legend would explode if we label every series×segment.
    # Better: show ONE legend for segments using proxy artists, and separate series labeling via bar order.
    # We'll do a clean segment-only legend:
    from matplotlib.patches import Patch
    proxies = [
        Patch(label="Executable + Correct"),
        Patch(label="Executable + Incorrect"),
        Patch(label="Non-Executable"),
    ]
    ax.legend(handles=proxies, loc="upper right")

    # Add a small note listing the series order (so reader knows which 4 bars correspond to what)
    series_note = "Bars per dataset (left→right): " + ", ".join(series_labels)
    ax.text(0.01, -0.14, series_note, transform=ax.transAxes, ha="left", va="top", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="results/master_results.csv", help="Path to master CSV")
    ap.add_argument("--docs_dir", default="docs", help="Docs directory (default: docs)")
    ap.add_argument("--models", default="gpt2xl,qwen", help="Comma-separated model names (default: gpt2xl,qwen)")
    ap.add_argument("--datasets", default="imdb,yelp,atis,advising", help="Comma-separated datasets (default: imdb,yelp,atis,advising)")
    ap.add_argument("--rdbms", default="", help="Comma-separated rdbms list (default: all found in CSV)")
    ap.add_argument("--no_annotate", action="store_true", help="Disable in-bar percentage annotations")
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
    out_table = tables_dir / "executability_vs_correctness_by_dataset_model_rdbms.csv"
    tbl.to_csv(out_table, index=False, encoding="utf-8")

    out_plot = figures_dir / "executability_vs_correctness_by_dataset_model_rdbms.png"
    plot_stacked_by_dataset_model_rdbms(
        tbl,
        out_plot,
        models=models,
        datasets=datasets,
        rdbms_list=rdbms_list,
        annotate=(not args.no_annotate),
    )

    print(f"✅ Wrote table: {out_table}")
    print(f"✅ Wrote plot : {out_plot}")


if __name__ == "__main__":
    main()
