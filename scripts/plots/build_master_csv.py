#!/usr/bin/env python3
"""
Build a master CSV from results/*.jsonl files.

Input files naming format:
  {model}_benchmark_{dataset}_{rdbms}.jsonl
Example:
  gpt2xl_benchmark_imdb_mysql.jsonl

Each line in each jsonl is a JSON object (one record per question).

Output:
  results/master_results.csv

Notes:
- We flatten nested execution fields like "mysql_pred.success" into normalized columns.
- We keep ONLY analysis-friendly scalar columns (no huge SQL/schema strings).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# begin from grandparent dir of this script (repo root)
os.chdir(Path(__file__).parent.parent.parent)

RESULTS_DIR = Path("results")
OUTPUT_CSV = RESULTS_DIR / "master_results.csv"


FILENAME_RE = re.compile(r"^(?P<model>.+?)_benchmark_(?P<dataset>.+?)_(?P<rdbms>.+?)\.jsonl$", re.IGNORECASE)


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _get_exec_field(record: Dict[str, Any], prefix: str, key: str) -> Any:
    """
    Supports both styles you have in your jsonl:

    Style A (namespaced with dots):
      mysql_pred.success
      mysql_pred.execution_time_s
      mysql_pred.error
      mysql_gold.success
      ...

    Style B (flattened with underscores):
      mysql_pred_success
      mysql_pred_time_s
      mysql_pred_error_msg
      mysql_gold_success
      ...
    """
    dotted = f"{prefix}.{key}"
    if dotted in record:
        return record[dotted]

    # Alternate keys depending on benchmark script version
    alt_map = {
        # success
        ("pred", "success"): [f"{prefix}_success", f"{prefix}__success"],
        ("gold", "success"): [f"{prefix}_success", f"{prefix}__success"],
        # execution time
        ("pred", "execution_time_s"): [f"{prefix}_time_s", f"{prefix}_execution_time_s"],
        ("gold", "execution_time_s"): [f"{prefix}_time_s", f"{prefix}_execution_time_s"],
        # rows
        ("pred", "rows"): [f"{prefix}_rows"],
        ("gold", "rows"): [f"{prefix}_rows"],
        # error
        ("pred", "error"): [f"{prefix}_error", f"{prefix}_error_msg"],
        ("gold", "error"): [f"{prefix}_error", f"{prefix}_error_msg"],
    }

    # Identify whether prefix is like "mysql_pred" or "mysql_gold"
    if prefix.endswith("_pred"):
        kind = "pred"
    elif prefix.endswith("_gold"):
        kind = "gold"
    else:
        kind = ""

    candidates = alt_map.get((kind, key), [])
    for c in candidates:
        if c in record:
            return record[c]

    return None


def parse_filename_fields(path: Path) -> Tuple[str, str, str]:
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return m.group("model"), m.group("dataset"), m.group("rdbms")


def build_rows_from_file(jsonl_path: Path) -> List[Dict[str, Any]]:
    model_from_name, dataset_from_name, rdbms_from_name = parse_filename_fields(jsonl_path)

    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON parse error in {jsonl_path} line {line_no}: {e}") from e

            # Prefer record fields if present; fallback to filename-derived ones.
            model = rec.get("llm", model_from_name)
            dataset = rec.get("dataset", dataset_from_name)
            rdbms = rec.get("rdbms", rdbms_from_name)

            row_id = rec.get("id")

            # Execution fields (support both schemas)
            pred_prefix = f"{rdbms}_pred"
            gold_prefix = f"{rdbms}_gold"

            pred_success = _get_exec_field(rec, pred_prefix, "success")
            gold_success = _get_exec_field(rec, gold_prefix, "success")

            pred_exec_time = _get_exec_field(rec, pred_prefix, "execution_time_s")
            gold_exec_time = _get_exec_field(rec, gold_prefix, "execution_time_s")


            # Match field
            match = rec.get(f"{rdbms}_pred_vs_gold_match")
            if match is None:
                # sometimes "mysql_pred_vs_gold_match"
                match = rec.get(f"{rdbms}_pred_vs_gold_match")

            # Repairs
            pred_repairs = rec.get("pred_repairs", [])
            pred_col_repairs = rec.get("pred_column_repairs", rec.get("pred_col_repairs", []))

            num_table_repairs = len(pred_repairs) if isinstance(pred_repairs, list) else 0
            num_column_repairs = len(pred_col_repairs) if isinstance(pred_col_repairs, list) else 0

            out: Dict[str, Any] = {
                # Core identifiers
                "model": model,
                "dataset": dataset,
                "rdbms": rdbms,
                "row_id": row_id,

                # Metadata
                "difficulty": _safe_int(rec.get("difficulty")),
                "query_split": str(rec.get("query_split", "")) if rec.get("query_split") is not None else "",
                "question_split": str(rec.get("question_split", "")) if rec.get("question_split") is not None else "",

                # Schema / prompt
                "schema_num_tables": _safe_int(rec.get("schema_num_tables")),
                "schema_num_columns": _safe_int(rec.get("schema_num_columns")),
                "prompt_tokens": _safe_int(rec.get("prompt_tokens")),

                # Generation
                "gen_time_s": _safe_float(rec.get("gen_time_s")),

                # Execution outcomes
                "pred_success": bool(pred_success) if pred_success is not None else False,
                "gold_success": bool(gold_success) if gold_success is not None else False,
                "pred_vs_gold_match": bool(match) if match is not None else False,

                # Execution timing & sizes
                "pred_execution_time_s": _safe_float(pred_exec_time),
                "gold_execution_time_s": _safe_float(gold_exec_time),

                # Repairs
                "num_table_repairs": _safe_int(num_table_repairs),
                "num_column_repairs": _safe_int(num_column_repairs),
            }

            # Optional derived columns (handy for analysis)
            out["has_repairs"] = (out["num_table_repairs"] or 0) + (out["num_column_repairs"] or 0) > 0
            out["valid_prediction"] = out["pred_success"]
            out["successful_execution"] = out["pred_success"] and out["gold_success"]

            rows.append(out)

    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(RESULTS_DIR.glob("*_benchmark_*_*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No jsonl files found in {RESULTS_DIR} matching '*_benchmark_*_*.jsonl'")

    all_rows: List[Dict[str, Any]] = []
    for p in jsonl_files:
        all_rows.extend(build_rows_from_file(p))

    # Stable column order
    fieldnames = [
        "model", "dataset", "rdbms", "row_id",
        "difficulty", "query_split", "question_split",
        "schema_num_tables", "schema_num_columns", "prompt_tokens",
        "gen_time_s",
        "pred_success", "gold_success", "pred_vs_gold_match",
        "pred_execution_time_s", "gold_execution_time_s",
        "num_table_repairs", "num_column_repairs",
        "has_repairs", "valid_prediction", "successful_execution",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"✅ Wrote {len(all_rows)} rows to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
