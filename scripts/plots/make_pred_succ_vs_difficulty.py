#!/usr/bin/env python3
"""
scripts/make_pred_succ_vs_difficulty.py

Plot 3 — Prediction Success vs Difficulty (faceted by dataset, subplots by rdbms)

- Aggregation: model × dataset × rdbms × difficulty
- X: difficulty
- Y: prediction success rate (% pred_success)
- Line: model
- Subplot: rdbms (within each dataset figure)
- Output: one PNG per dataset (e.g., accuracy_vs_difficulty__imdb.png)

Outputs:
- docs/tables/pred_succ_vs_difficulty_by_model_dataset_rdbms.csv
- docs/figures/pred_succ_vs_difficulty__<dataset>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import as_bool_series


def _dynamic_ylim_percent(ax, y_values: np.ndarray) -> None:
    vals = np.asarray(y_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vmax = float(vals.max()) if vals.size else 0.0
    pad = max(2.0, 0.10 * vmax)
    ax.set_ylim(0.0, min(100.0, vmax + pad))


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    required = ["model", "dataset", "rdbms", "difficulty", "pred_success"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    df = df.copy()
    df["pred_success"] = as_bool_series(df["pred_success"]).fillna(False).astype(bool)
    df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce")

    # Drop rows without difficulty (can't plot them)
    df = df.dropna(subset=["difficulty"])
    df["difficulty"] = df["difficulty"].astype(int)

    g = df.groupby(["model", "dataset", "rdbms", "difficulty"], dropna=False)

    out = g.agg(
        n_queries=("pred_success", "size"),
        p=("pred_success", "mean"),
    ).reset_index()

    # Convert to %
    out["pred_success_rate_pct"] = (out["p"] * 100.0)

    # 95% CI (normal approximation)
    n = out["n_queries"].astype(float).replace(0, np.nan)
    se = np.sqrt((out["p"] * (1.0 - out["p"])) / n)
    out["ci95_low_pct"] = ((out["p"] - 1.96 * se).clip(0, 1) * 100.0)
    out["ci95_high_pct"] = ((out["p"] + 1.96 * se).clip(0, 1) * 100.0)

    # presentation rounding
    out["pred_success_rate_pct"] = out["pred_success_rate_pct"].round(6)
    out["ci95_low_pct"] = out["ci95_low_pct"].round(6)
    out["ci95_high_pct"] = out["ci95_high_pct"].round(6)

    out = out.drop(columns=["p"])
    out = out.sort_values(["dataset", "rdbms", "difficulty", "model"]).reset_index(drop=True)
    return out


def plot_faceted_by_dataset(tbl: pd.DataFrame, figures_dir: Path, show_ci: bool = True) -> list[Path]:
    """
    Make one figure per dataset.
    Within each figure: subplots per rdbms.
    Each subplot shows model lines vs difficulty.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    datasets = sorted(tbl["dataset"].astype(str).unique().tolist())
    models = sorted(tbl["model"].astype(str).unique().tolist())

    for dataset in datasets:
        ds_tbl = tbl[tbl["dataset"].astype(str) == dataset].copy()
        rdbms_list = sorted(ds_tbl["rdbms"].astype(str).unique().tolist())

        # Grid: up to 3 columns, as many rows as needed
        n = len(rdbms_list)
        ncols = min(3, max(1, n))
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
        fig.suptitle(f"Prediction Success Rate vs Difficulty — dataset={dataset}", y=1.02)

        for idx, rdbms in enumerate(rdbms_list):
            ax = axes[idx // ncols][idx % ncols]
            sub = ds_tbl[ds_tbl["rdbms"].astype(str) == rdbms].copy()

            difficulties = sorted(sub["difficulty"].unique().tolist())
            ax.set_title(str(rdbms))
            ax.set_xlabel("Difficulty")
            ax.set_ylabel("Prediction Success Rate (%)")
            ax.set_xticks(difficulties)

            all_y = []

            for model in models:
                msub = sub[sub["model"].astype(str) == model].sort_values("difficulty")
                if msub.empty:
                    continue

                x = msub["difficulty"].to_numpy(dtype=int)
                y = msub["pred_success_rate_pct"].to_numpy(dtype=float)
                all_y.append(y)

                ax.plot(x, y, marker="o", label=model)
                
                # Point labels: value above each point
                for xi, yi in zip(x, y):
                    ax.annotate(
                        f"{yi:.3f}",          # choose decimals
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(0, 6),        # 6px above the point
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )


                if show_ci:
                    ylo = msub["ci95_low_pct"].to_numpy(dtype=float)
                    yhi = msub["ci95_high_pct"].to_numpy(dtype=float)
                    ax.fill_between(x, ylo, yhi, alpha=0.2)

                # Optional point labels (uncomment if you want)
                # for xi, yi in zip(x, y):
                #     ax.text(xi, yi + 0.2, f"{yi:.4f}", ha="center", va="bottom", fontsize=8)

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
        out_png = figures_dir / f"pred_succ_vs_difficulty" / f"{dataset}.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(out_png)

    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="results/master_results.csv", help="Path to master CSV")
    ap.add_argument("--docs_dir", default="docs", help="Docs directory (default: docs)")
    ap.add_argument("--no_ci", action="store_true", help="Disable confidence intervals shading")
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

    tbl = build_table(df)
    out_table = tables_dir / "pred_succ_vs_difficulty_by_model_dataset_rdbms.csv"
    tbl.to_csv(out_table, index=False, encoding="utf-8")

    written = plot_faceted_by_dataset(tbl, figures_dir, show_ci=(not args.no_ci))

    print(f"✅ Wrote table: {out_table}")
    for p in written:
        print(f"✅ Wrote plot : {p}")


if __name__ == "__main__":
    main()
