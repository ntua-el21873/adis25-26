#!/usr/bin/env python3
"""
scripts/make_gen_time_vs_prompt_tokens.py

Plot 8 — Generation Time vs Prompt Tokens (refactored, aggregated like your other scripts)

You asked to align aggregation / presentation with the rest of the pipeline:
- Aggregate by model × dataset × rdbms
- Present plots in a readable faceted layout (like gen_time_distribution faceted by dataset)
- Still keep a single "main" figure (one per dataset is also reasonable, but here we do ONE figure)

This script produces:
1) Table A: Pearson correlation per (model, dataset, rdbms)
2) Figure: one figure faceted by dataset, with subplots per rdbms, scatter colored by model
   (X=prompt_tokens, Y=gen_time_s), optional log Y.

Outputs:
- docs/tables/gen_time_vs_prompt_tokens_summary_by_model_dataset_rdbms.csv
- docs/figures/gen_time_vs_prompt_tokens__faceted_by_dataset.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- table ----------

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = ["model", "dataset", "rdbms", "prompt_tokens", "gen_time_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    dff = df.copy()
    dff["model"] = dff["model"].astype(str)
    dff["dataset"] = dff["dataset"].astype(str)
    dff["rdbms"] = dff["rdbms"].astype(str)
    dff["prompt_tokens"] = pd.to_numeric(dff["prompt_tokens"], errors="coerce")
    dff["gen_time_s"] = pd.to_numeric(dff["gen_time_s"], errors="coerce")
    dff = dff.dropna(subset=["prompt_tokens", "gen_time_s"])

    rows = []
    for (model, dataset, rdbms), sub in dff.groupby(["model", "dataset", "rdbms"], dropna=False):
        n = int(len(sub))
        if n < 2:
            corr = np.nan
        else:
            # Pearson correlation
            corr = np.corrcoef(sub["prompt_tokens"].to_numpy(), sub["gen_time_s"].to_numpy())[0, 1]

        rows.append({
            "model": str(model),
            "dataset": str(dataset),
            "rdbms": str(rdbms),
            "n": n,
            "pearson_corr_prompt_tokens_vs_gen_time": float(corr) if corr == corr else np.nan,
            "mean_prompt_tokens": float(sub["prompt_tokens"].mean()),
            "median_prompt_tokens": float(sub["prompt_tokens"].median()),
            "mean_gen_time_s": float(sub["gen_time_s"].mean()),
            "median_gen_time_s": float(sub["gen_time_s"].median()),
        })

    out = pd.DataFrame(rows)
    out["pearson_corr_prompt_tokens_vs_gen_time"] = out["pearson_corr_prompt_tokens_vs_gen_time"].round(4)
    out[["mean_prompt_tokens", "median_prompt_tokens"]] = out[["mean_prompt_tokens", "median_prompt_tokens"]].round(2)
    out[["mean_gen_time_s", "median_gen_time_s"]] = out[["mean_gen_time_s", "median_gen_time_s"]].round(6)

    out = out.sort_values(
        ["dataset", "rdbms", "pearson_corr_prompt_tokens_vs_gen_time", "model"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    return out


# ---------- plot ----------
def plot_scatter_one_figure_per_dataset_with_rdbms_subplots(
    df: pd.DataFrame,
    figures_dir: Path,
    log_y: bool,
) -> list[Path]:
    """
    Output: one figure per dataset
    Each figure: subplots per rdbms (e.g., 2 subplots for mysql/mariadb)
    """
    required = ["model", "dataset", "rdbms", "prompt_tokens", "gen_time_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    figures_dir.mkdir(parents=True, exist_ok=True)

    dff = df.copy()
    dff["model"] = dff["model"].astype(str)
    dff["dataset"] = dff["dataset"].astype(str)
    dff["rdbms"] = dff["rdbms"].astype(str)
    dff["prompt_tokens"] = pd.to_numeric(dff["prompt_tokens"], errors="coerce")
    dff["gen_time_s"] = pd.to_numeric(dff["gen_time_s"], errors="coerce")
    dff = dff.dropna(subset=["prompt_tokens", "gen_time_s"])

    datasets = sorted(dff["dataset"].unique().tolist())
    models = sorted(dff["model"].unique().tolist())

    written: list[Path] = []

    for dataset in datasets:
        ds = dff[dff["dataset"] == dataset].copy()
        rdbms_list = sorted(ds["rdbms"].unique().tolist())

        ncols = len(rdbms_list)
        fig, axes = plt.subplots(
            nrows=1,
            ncols=ncols,
            figsize=(6.0 * ncols, 4.2),
            squeeze=False,
        )
        axes = axes[0]  # shape: (ncols,)

        fig.suptitle(f"Generation Time vs Prompt Tokens — dataset={dataset}", y=1.03)

        for j, rdbms in enumerate(rdbms_list):
            ax = axes[j]
            sub = ds[ds["rdbms"] == rdbms].copy()
            ax.set_title(str(rdbms))

            for model in models:
                msub = sub[sub["model"] == model]
                if msub.empty:
                    continue
                ax.scatter(
                    msub["prompt_tokens"].to_numpy(),
                    msub["gen_time_s"].to_numpy(),
                    s=10,
                    alpha=0.6,
                    label=model,
                )

            ax.set_xlabel("Prompt Tokens")
            ax.set_ylabel("Gen Time (s)")
            if log_y:
                ax.set_yscale("log")
                ax.set_ylabel("Gen Time (s, log)")

            ax.grid(True, alpha=0.3)

            # Legend only on first subplot of each dataset figure
            if j == 0:
                ax.legend(fontsize=9)

        fig.tight_layout()
        out_png = figures_dir / "gen_time_vs_prompt_tokens" / f"{dataset}.png"
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
    ap.add_argument("--log_y", action="store_true", help="Use log-scale y axis for gen_time_s")
    ap.add_argument(
        "--facet_mode",
        default="dataset_rdbms",
        help="dataset: 1 panel per dataset; dataset_rdbms: grid dataset×rdbms (default, matches your other scripts)",
    )
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

    # Table: correlation per model×dataset×rdbms
    tbl = build_summary(df)
    out_table = tables_dir / "gen_time_vs_prompt_tokens_summary_by_model_dataset_rdbms.csv"
    tbl.to_csv(out_table, index=False, encoding="utf-8")

    # Plots
    written = plot_scatter_one_figure_per_dataset_with_rdbms_subplots(df, figures_dir, log_y=args.log_y)

    print(f"✅ Wrote table: {out_table}")
    for p in written:
        print(f"✅ Wrote plot : {p}")
if __name__ == "__main__":
    main()
