#!/usr/bin/env python3
"""
scripts/make_repair_effectiveness.py

Table 2 — Repair Effectiveness Summary (model × dataset × rdbms)
Plot 9  — Prediction Success Rate With vs Without Repairs (one figure per dataset, subplots per rdbms)

Outputs:
- docs/tables/repair_effectiveness_by_model_dataset_rdbms.csv
- docs/figures/prediction_success_rate_with_vs_without_repairs__<dataset>.png

Notes:
- With the current master CSV, we can’t compute "pre-repair vs post-repair" on the same query
  unless you also stored raw/pre-repair correctness separately.
- We therefore report prediction success rate conditioned on has_repairs (with/without).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import as_bool_series


# ---------- table ----------

def build_table(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "model",
        "dataset",
        "rdbms",
        "pred_success",
        "has_repairs",
        "num_table_repairs",
        "num_column_repairs",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    dff = df.copy()

    dff["model"] = dff["model"].astype(str)
    dff["dataset"] = dff["dataset"].astype(str)
    dff["rdbms"] = dff["rdbms"].astype(str)

    dff["pred_success"] = as_bool_series(dff["pred_success"]).fillna(False).astype(bool)
    dff["has_repairs"] = as_bool_series(dff["has_repairs"]).fillna(False).astype(bool)

    dff["num_table_repairs"] = pd.to_numeric(dff["num_table_repairs"], errors="coerce").fillna(0).astype(int)
    dff["num_column_repairs"] = pd.to_numeric(dff["num_column_repairs"], errors="coerce").fillna(0).astype(int)

    rows = []
    for (model, dataset, rdbms), sub in dff.groupby(["model", "dataset", "rdbms"], dropna=False):
        n = int(len(sub))

        with_rep = sub[sub["has_repairs"]]
        without_rep = sub[~sub["has_repairs"]]

        avg_table_rep = float(sub["num_table_repairs"].mean()) if n else np.nan
        avg_col_rep = float(sub["num_column_repairs"].mean()) if n else np.nan
        pct_any_repairs = float(sub["has_repairs"].mean() * 100.0) if n else np.nan

        acc_overall = float(sub["pred_success"].mean() * 100.0) if n else np.nan
        acc_with_rep = float(with_rep["pred_success"].mean() * 100.0) if len(with_rep) else np.nan
        acc_without_rep = float(without_rep["pred_success"].mean() * 100.0) if len(without_rep) else np.nan

        rows.append(
            {
                "model": str(model),
                "dataset": str(dataset),
                "rdbms": str(rdbms),
                "n_queries": n,
                "avg_num_table_repairs": round(avg_table_rep, 3),
                "avg_num_column_repairs": round(avg_col_rep, 3),
                "pct_queries_with_any_repair": round(pct_any_repairs, 2),
                "prediction_success_rate_overall_pct": round(acc_overall, 2),
                "prediction_success_rate_with_repairs_pct": round(acc_with_rep, 2) if acc_with_rep == acc_with_rep else np.nan,
                "prediction_success_rate_without_repairs_pct": round(acc_without_rep, 2) if acc_without_rep == acc_without_rep else np.nan,
                "n_with_repairs": int(len(with_rep)),
                "n_without_repairs": int(len(without_rep)),
            }
        )

    out = pd.DataFrame(rows)

    # Sort for stable readability: dataset, rdbms, overall prediction success rate desc, then fewer repairs
    out = out.sort_values(
        ["dataset", "rdbms", "prediction_success_rate_overall_pct", "pct_queries_with_any_repair", "model"],
        ascending=[True, True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    return out


# ---------- plot ----------

def plot_pred_succ_with_without_repairs_faceted_by_dataset(
    tbl: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """
    One figure per dataset.
    Within each figure: subplots per rdbms.
    Each subplot: grouped bars by model (no repairs vs with repairs).
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    datasets = sorted(tbl["dataset"].astype(str).unique().tolist())
    models_all = sorted(tbl["model"].astype(str).unique().tolist())

    for dataset in datasets:
        ds_tbl = tbl[tbl["dataset"].astype(str) == dataset].copy()
        rdbms_list = sorted(ds_tbl["rdbms"].astype(str).unique().tolist())

        n = len(rdbms_list)
        ncols = min(3, max(1, n))
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.8 * ncols, 4.0 * nrows),
            squeeze=False,
        )
        fig.suptitle(f"Prediction Success Rate With vs Without Repairs — dataset={dataset}", y=1.03)

        for idx, rdbms in enumerate(rdbms_list):
            ax = axes[idx // ncols][idx % ncols]
            sub = ds_tbl[ds_tbl["rdbms"].astype(str) == rdbms].copy()

            # Keep a consistent model order, but only those present in this subplot
            models = [m for m in models_all if (sub["model"].astype(str) == m).any()]

            acc_without = []
            acc_with = []
            for m in models:
                mrow = sub[sub["model"].astype(str) == m]
                # One row per (model,dataset,rdbms); if duplicates exist, take first deterministically
                if mrow.empty:
                    acc_without.append(np.nan)
                    acc_with.append(np.nan)
                else:
                    acc_without.append(float(mrow["prediction_success_rate_without_repairs_pct"].iloc[0]))
                    acc_with.append(float(mrow["prediction_success_rate_with_repairs_pct"].iloc[0]))

            x = np.arange(len(models))
            width = 0.40

            ax.bar(x - width / 2, acc_without, width=width, label="No repairs")
            ax.bar(x + width / 2, acc_with, width=width, label="With repairs")

            ax.set_title(str(rdbms))
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=20, ha="right")
            ax.set_ylabel("Prediction Success Rate (%)")
            ax.grid(True, axis="y", alpha=0.3)

            # Legend only once per figure to reduce clutter
            if idx == 0:
                ax.legend(fontsize=9)

        # Hide unused axes
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.tight_layout()
        out_png = figures_dir / f"pred_succ_rate_with_vs_without_repairs" / f"{dataset}.png"
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

    # Table: model×dataset×rdbms
    tbl = build_table(df)
    out_table = tables_dir / "repair_effectiveness_by_model_dataset_rdbms.csv"
    tbl.to_csv(out_table, index=False, encoding="utf-8")

    # Plots: one per dataset
    written = plot_pred_succ_with_without_repairs_faceted_by_dataset(tbl, figures_dir)

    print(f"✅ Wrote table: {out_table}")
    for p in written:
        print(f"✅ Wrote plot : {p}")


if __name__ == "__main__":
    main()
