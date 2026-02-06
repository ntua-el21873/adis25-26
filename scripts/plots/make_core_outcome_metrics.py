#!/usr/bin/env python3
"""
scripts/make_core_outcome_metrics.py

Core Outcome Metrics:
- Reads the master CSV
- Writes an aggregated summary table to docs/core_outcome_summary.csv
- Writes a bar plot (execution accuracy by model) to docs/figures/core_outcome_accuracy_by_model.png

Expected master CSV columns (at minimum):
model,dataset,rdbms,row_id,difficulty,query_split,question_split,schema_num_tables,schema_num_columns,
prompt_tokens,gen_time_s,pred_success,gold_success,pred_vs_gold_match,pred_execution_time_s,gold_execution_time_s,
num_table_repairs,num_column_repairs,has_repairs,valid_prediction,successful_execution
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _as_bool_series(s: pd.Series) -> pd.Series:
    """
    Robustly coerce common representations to boolean.
    Handles: True/False, 1/0, "true"/"false", "OK"/"FAIL", "yes"/"no".
    Unknowns -> NaN -> treated as False in means (via fillna(False)).
    """
    if s is None:
        return pd.Series(dtype="boolean")
    if pd.api.types.is_bool_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float).fillna(0.0).astype(int).astype(bool)

    s2 = s.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "ok": True,
        "fail": False,
        "success": True,
        "error": False,
    }
    out = s2.map(mapping)
    return out.astype("boolean")


def build_core_outcome_table(df: pd.DataFrame) -> pd.DataFrame:
    required = ["model", "dataset", "rdbms"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master CSV is missing required columns: {missing}")

    # Normalize booleans
    for col in ["pred_success", "gold_success", "pred_vs_gold_match"]:
        if col in df.columns:
            df[col] = _as_bool_series(df[col]).fillna(False).astype(bool)

    # Normalize numerics (safe)
    for col in [
        "gen_time_s",
        "prompt_tokens",
        "schema_num_tables",
        "schema_num_columns",
        "pred_execution_time_s",
        "gold_execution_time_s",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Grouped summary (model × dataset × rdbms)
    grp_cols = ["model", "dataset", "rdbms"]
    g = df.groupby(grp_cols, dropna=False)

    summary = g.agg(
        n_queries=("row_id", "count") if "row_id" in df.columns else ("model", "count"),
        exec_accuracy=("pred_vs_gold_match", "mean") if "pred_vs_gold_match" in df.columns else (df.columns[0], lambda _: np.nan),
        executable_rate=("pred_success", "mean") if "pred_success" in df.columns else (df.columns[0], lambda _: np.nan),
        gold_executable_rate=("gold_success", "mean") if "gold_success" in df.columns else (df.columns[0], lambda _: np.nan),
        avg_gen_time_s=("gen_time_s", "mean") if "gen_time_s" in df.columns else (df.columns[0], lambda _: np.nan),
        median_gen_time_s=("gen_time_s", "median") if "gen_time_s" in df.columns else (df.columns[0], lambda _: np.nan),
        avg_prompt_tokens=("prompt_tokens", "mean") if "prompt_tokens" in df.columns else (df.columns[0], lambda _: np.nan),
        avg_schema_tables=("schema_num_tables", "mean") if "schema_num_tables" in df.columns else (df.columns[0], lambda _: np.nan),
        avg_schema_columns=("schema_num_columns", "mean") if "schema_num_columns" in df.columns else (df.columns[0], lambda _: np.nan),
        avg_pred_exec_time_s=("pred_execution_time_s", "mean") if "pred_execution_time_s" in df.columns else (df.columns[0], lambda _: np.nan),
        avg_gold_exec_time_s=("gold_execution_time_s", "mean") if "gold_execution_time_s" in df.columns else (df.columns[0], lambda _: np.nan),
    ).reset_index()

    # Percent columns as percentages (0-100)
    for col in ["exec_accuracy", "executable_rate", "gold_executable_rate"]:
        if col in summary.columns:
            summary[col] = (summary[col].astype(float) * 100.0).round(2)

    # Round continuous stats
    for col in ["avg_gen_time_s", "median_gen_time_s", "avg_pred_exec_time_s", "avg_gold_exec_time_s"]:
        if col in summary.columns:
            summary[col] = summary[col].astype(float).round(6)

    for col in ["avg_prompt_tokens", "avg_schema_tables", "avg_schema_columns"]:
        if col in summary.columns:
            summary[col] = summary[col].astype(float).round(2)

    # Deterministic ordering
    summary = summary.sort_values(["dataset", "rdbms", "model"]).reset_index(drop=True)
    return summary


def plot_execution_accuracy_by_model(df: pd.DataFrame, out_png: Path) -> None:
    if "model" not in df.columns or "pred_vs_gold_match" not in df.columns:
        raise ValueError("Need columns: 'model' and 'pred_vs_gold_match' to plot accuracy by model.")

    # Per-model overall accuracy across all datasets/rdbms
    per_model = (
        df.groupby("model", dropna=False)["pred_vs_gold_match"]
        .apply(lambda s: _as_bool_series(s).fillna(False).astype(bool).mean())
        .sort_index()
    )

    # Convert to %
    per_model_pct = (per_model * 100.0).sort_values(ascending=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.bar(per_model_pct.index.astype(str), per_model_pct.values)
    plt.ylabel("Execution Accuracy (%)")
    plt.xlabel("Model")
    plt.title("Execution Accuracy by Model (Overall)")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, min(100, max(5, float(per_model_pct.max()) + 5)))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--master_csv",
        required=True,
        help="Path to the master CSV built from results/*.jsonl",
    )
    ap.add_argument(
        "--docs_dir",
        default="docs",
        help="Docs directory (default: docs)",
    )
    args = ap.parse_args()

    master_csv = Path(args.master_csv)
    if not master_csv.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")

    docs_dir = Path(args.docs_dir)
    figures_dir = docs_dir / "figures"
    docs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(master_csv)

    # Build and write summary table
    summary = build_core_outcome_table(df)
    out_table = docs_dir / "core_outcome_summary.csv"
    summary.to_csv(out_table, index=False, encoding="utf-8")

    # Plot accuracy by model
    out_plot = figures_dir / "core_outcome_accuracy_by_model.png"
    plot_execution_accuracy_by_model(df, out_plot)

    print(f"✅ Wrote table: {out_table}")
    print(f"✅ Wrote plot : {out_plot}")


if __name__ == "__main__":
    main()
