"""
scripts/plot_results.py

Read a metrics CSV (from jsonl->csv step) and generate plots into docs/figures/.

Assumptions about CSV columns (from the converter we discussed):
Core identifiers:
  - dataset
  - rdbms_mode  (optional; if missing, plots still work grouped by dataset only)

Gold/pred execution:
  - mysql_pred_success, mysql_gold_success, mysql_ex, mysql_ex_given_success, mysql_exec_time_s, mysql_gold_exec_time_s
  - mariadb_pred_success, mariadb_gold_success, mariadb_ex, mariadb_ex_given_success, mariadb_exec_time_s, mariadb_gold_exec_time_s
  - mysql_vs_mariadb_match  (only when both executed)

Generation + prompt:
  - gen_time_s
  - prompt_chars (or prompt_tokens if you added)
  - schema_tables_included
  - schema_columns_included
  - complexity_bucket  in {"simple","medium","complex"} (or empty)

Output:
  - PNG files into docs/figures/ with consistent filenames.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------
# Helpers
# --------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _safe_col(df: pd.DataFrame, col: str, default=0):
    if col in df.columns:
        return df[col]
    return default

def _as_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    # handle "True"/"False" strings or NaNs
    return df[col].fillna(False).astype(bool)

def _as_num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")

def _agg_rate(df: pd.DataFrame, flag_col: str) -> float:
    s = _as_bool_series(df, flag_col)
    return float(s.mean()) if len(s) else 0.0

def _agg_mean(df: pd.DataFrame, num_col: str) -> float:
    s = _as_num_series(df, num_col)
    return float(s.dropna().mean()) if s.notna().any() else float("nan")

def _agg_median(df: pd.DataFrame, num_col: str) -> float:
    s = _as_num_series(df, num_col)
    return float(s.dropna().median()) if s.notna().any() else float("nan")

def _agg_p95(df: pd.DataFrame, num_col: str) -> float:
    s = _as_num_series(df, num_col).dropna()
    return float(s.quantile(0.95)) if len(s) else float("nan")

def _filter_complexity(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    if "complexity_bucket" not in df.columns:
        return df.iloc[0:0]
    return df[df["complexity_bucket"].fillna("") == bucket]

def _savefig(out_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _bar_plot(series: pd.Series, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure()
    series = series.sort_values(ascending=False)
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("dataset")
    _savefig(out_path)

def _bar_plot_multi(df: pd.DataFrame, title: str, ylabel: str, out_path: Path) -> None:
    """
    Expects df indexed by dataset and columns as categories (e.g. rdbms or complexity).
    """
    plt.figure()
    df = df.sort_index()
    df.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("dataset")
    plt.legend()
    _savefig(out_path)

def _line_plot_by_bucket(df: pd.DataFrame, y_col: str, title: str, ylabel: str, out_path: Path) -> None:
    """
    Plot y (mean) across complexity buckets: simple->medium->complex.
    """
    buckets = ["simple", "medium", "complex"]
    vals = []
    for b in buckets:
        sub = _filter_complexity(df, b)
        vals.append(_agg_rate(sub, y_col))
    plt.figure()
    plt.plot(buckets, vals, marker="o")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("complexity_bucket")
    _savefig(out_path)


# --------------------------
# Metric computation
# --------------------------

def compute_metrics_by_dataset(df: pd.DataFrame, rdbms: str) -> pd.DataFrame:
    """
    rdbms in {"mysql","mariadb"}.
    Returns per-dataset aggregated metrics (one row per dataset).
    """
    assert rdbms in {"mysql", "mariadb"}
    pref = f"{rdbms}_"

    out_rows = []
    for dataset, g in df.groupby("dataset", dropna=False):
        total = len(g)

        pred_success = _agg_rate(g, pref + "pred_success")
        gold_success = _agg_rate(g, pref + "gold_success")
        ex = _agg_rate(g, pref + "ex")
        ex_given_success = _agg_rate(g, pref + "ex_given_success")

        exec_mean = _agg_mean(g, pref + "exec_time_s")
        exec_median = _agg_median(g, pref + "exec_time_s")
        exec_p95 = _agg_p95(g, pref + "exec_time_s")

        gold_exec_mean = _agg_mean(g, pref + "gold_exec_time_s")

        gen_mean = _agg_mean(g, "gen_time_s")
        gen_median = _agg_median(g, "gen_time_s")
        gen_p95 = _agg_p95(g, "gen_time_s")

        prompt_chars_mean = _agg_mean(g, "prompt_chars") if "prompt_chars" in g.columns else float("nan")
        tables_mean = _agg_mean(g, "schema_tables_included") if "schema_tables_included" in g.columns else float("nan")
        cols_mean = _agg_mean(g, "schema_columns_included") if "schema_columns_included" in g.columns else float("nan")

        out_rows.append({
            "dataset": dataset,
            "n": total,

            # Metrics (aligning to the 9 we discussed)
            "pred_exec_rate": pred_success,          # 1) Exec success rate
            "gold_exec_rate": gold_success,          # 2) Gold runnable rate
            "ex": ex,                                # 3) EX
            "ex_given_success": ex_given_success,    # 4) EX | pred success

            "pred_exec_time_mean_s": exec_mean,      # 5) Pred exec time
            "pred_exec_time_median_s": exec_median,
            "pred_exec_time_p95_s": exec_p95,

            "gold_exec_time_mean_s": gold_exec_mean, # 6) Gold exec time

            "gen_time_mean_s": gen_mean,             # 7) Generation time
            "gen_time_median_s": gen_median,
            "gen_time_p95_s": gen_p95,

            "prompt_chars_mean": prompt_chars_mean,  # 8) Prompt size proxy
            "schema_tables_mean": tables_mean,       # 9) Schema compactness proxies
            "schema_cols_mean": cols_mean,
        })

    return pd.DataFrame(out_rows).set_index("dataset").sort_index()


def compute_cross_rdbms_match_by_dataset(df: pd.DataFrame) -> pd.Series:
    """
    mysql_vs_mariadb_match is meaningful only for rows where both succeeded and were compared.
    This returns per-dataset match rate over rows where match is not null.
    """
    if "mysql_vs_mariadb_match" not in df.columns:
        return pd.Series(dtype=float)

    s = df["mysql_vs_mariadb_match"]
    # keep only rows where a comparison happened
    valid = df[s.notna()].copy()
    if valid.empty:
        return pd.Series(dtype=float)

    valid["mysql_vs_mariadb_match"] = valid["mysql_vs_mariadb_match"].astype(bool)
    return valid.groupby("dataset")["mysql_vs_mariadb_match"].mean().sort_index()


# --------------------------
# Plotting
# --------------------------

def make_plots(df: pd.DataFrame, out_dir: Path, tag: str) -> None:
    """
    Creates plots for MySQL, MariaDB, and cross-RDBMS match (if present).
    tag is used in filenames so multiple runs don't overwrite each other.
    """
    _ensure_dir(out_dir)

    # Per-RDBMS aggregated tables
    mysql_m = compute_metrics_by_dataset(df, "mysql")
    maria_m = compute_metrics_by_dataset(df, "mariadb")

    # 1) Exec success rate
    _bar_plot(mysql_m["pred_exec_rate"], f"MySQL: Exec success rate ({tag})", "rate", out_dir / f"{tag}__mysql__exec_success_rate.png")
    _bar_plot(maria_m["pred_exec_rate"], f"MariaDB: Exec success rate ({tag})", "rate", out_dir / f"{tag}__mariadb__exec_success_rate.png")

    # 2) Gold runnable rate
    _bar_plot(mysql_m["gold_exec_rate"], f"MySQL: Gold runnable rate ({tag})", "rate", out_dir / f"{tag}__mysql__gold_runnable_rate.png")
    _bar_plot(maria_m["gold_exec_rate"], f"MariaDB: Gold runnable rate ({tag})", "rate", out_dir / f"{tag}__mariadb__gold_runnable_rate.png")

    # 3) EX
    _bar_plot(mysql_m["ex"], f"MySQL: EX ({tag})", "rate", out_dir / f"{tag}__mysql__ex.png")
    _bar_plot(maria_m["ex"], f"MariaDB: EX ({tag})", "rate", out_dir / f"{tag}__mariadb__ex.png")

    # 4) EX | success
    _bar_plot(mysql_m["ex_given_success"], f"MySQL: EX | pred success ({tag})", "rate", out_dir / f"{tag}__mysql__ex_given_success.png")
    _bar_plot(maria_m["ex_given_success"], f"MariaDB: EX | pred success ({tag})", "rate", out_dir / f"{tag}__mariadb__ex_given_success.png")

    # 5) Pred exec time (median)
    _bar_plot(mysql_m["pred_exec_time_median_s"], f"MySQL: Pred exec time (median) ({tag})", "seconds", out_dir / f"{tag}__mysql__pred_exec_time_median_s.png")
    _bar_plot(maria_m["pred_exec_time_median_s"], f"MariaDB: Pred exec time (median) ({tag})", "seconds", out_dir / f"{tag}__mariadb__pred_exec_time_median_s.png")

    # 6) Gold exec time (mean)
    _bar_plot(mysql_m["gold_exec_time_mean_s"], f"MySQL: Gold exec time (mean) ({tag})", "seconds", out_dir / f"{tag}__mysql__gold_exec_time_mean_s.png")
    _bar_plot(maria_m["gold_exec_time_mean_s"], f"MariaDB: Gold exec time (mean) ({tag})", "seconds", out_dir / f"{tag}__mariadb__gold_exec_time_mean_s.png")

    # 7) Generation time (median)
    _bar_plot(mysql_m["gen_time_median_s"], f"MySQL run: Generation time (median) ({tag})", "seconds", out_dir / f"{tag}__gen_time_median_s.png")
    # (gen time is model-side; same for both, but we plot once.)

    # 8) Prompt size (mean chars)
    if "prompt_chars" in df.columns:
        _bar_plot(mysql_m["prompt_chars_mean"], f"Prompt size mean (chars) ({tag})", "chars", out_dir / f"{tag}__prompt_chars_mean.png")

    # 9) Schema compactness (mean #tables and #cols)
    compact_cols = []
    if "schema_tables_included" in df.columns:
        compact_cols.append("schema_tables_mean")
    if "schema_columns_included" in df.columns:
        compact_cols.append("schema_cols_mean")

    if compact_cols:
        # combine into a multi-bar plot using mysql_m (same prompt builder)
        compact_df = mysql_m[compact_cols].copy()
        compact_df.columns = ["mean_tables" if c == "schema_tables_mean" else "mean_columns" for c in compact_df.columns]
        _bar_plot_multi(compact_df, f"Schema compactness ({tag})", "count", out_dir / f"{tag}__schema_compactness.png")

    # Cross-RDBMS match rate (extra plot, only if present)
    match = compute_cross_rdbms_match_by_dataset(df)
    if not match.empty:
        _bar_plot(match, f"MySQL vs MariaDB: result match rate ({tag})", "rate", out_dir / f"{tag}__mysql_vs_mariadb__match_rate.png")

    # Complexity-bucket plots (optional, if complexity_bucket exists)
    if "complexity_bucket" in df.columns:
        # show EX by complexity for mysql and mariadb
        _line_plot_by_bucket(df, "mysql_ex", f"MySQL: EX by complexity ({tag})", "rate", out_dir / f"{tag}__mysql__ex_by_complexity.png")
        _line_plot_by_bucket(df, "mariadb_ex", f"MariaDB: EX by complexity ({tag})", "rate", out_dir / f"{tag}__mariadb__ex_by_complexity.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to metrics CSV (from jsonl->csv).")
    parser.add_argument("--out_dir", type=str, default="docs/figures", help="Output directory for plots.")
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Filename tag (default: basename of CSV without extension).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return 1

    out_dir = Path(args.out_dir)
    tag = args.tag or csv_path.stem

    df = pd.read_csv(csv_path)

    if "dataset" not in df.columns:
        print("❌ CSV must contain a 'dataset' column.")
        return 1

    # Normalize types for safety
    for col in [
        "mysql_pred_success", "mysql_gold_success", "mysql_ex", "mysql_ex_given_success",
        "mariadb_pred_success", "mariadb_gold_success", "mariadb_ex", "mariadb_ex_given_success",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    make_plots(df, out_dir=out_dir, tag=tag)

    print(f"✅ Plots saved to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
