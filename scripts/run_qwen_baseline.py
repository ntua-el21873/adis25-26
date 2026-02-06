"""
scripts/run_qwen_baseline.py

Research-grade Qwen Text2SQL runner.
Functionally IDENTICAL to run_gpt2xl_baseline.py for fair comparison.

Differences from GPT-2 script:
- Removed prompt truncation logic (Qwen context window is large enough).
- Uses QwenAgent instead of GPT2XLAgent.
"""

import re
import argparse
import json
import sys
import time
from pathlib import Path

from nbformat import write

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.qwen_agent import QwenAgent
from database.db_manager import DatabaseManager
from typing import Optional

# We import the exact same utils as the GPT-2 baseline
from scripts.sql_utils import (
    fill_gold_sql,
    compare_results,
    repair_pred_table_names,
    repair_pred_column_names,
    parse_schema_counts,
    normalize_pred_sql,
    normalize_table_case,
    build_identifier_maps,
)

# -----------------------------------------------------------------------------
# Helper Functions (Mirrored from run_gpt2xl_baseline.py)
# -----------------------------------------------------------------------------


def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def pack_exec_fields(prefix: str, res: dict | None) -> dict:
    """
    Flattens execution results into the specific JSON format required by metrics.
    Matches GPT-2 baseline implementation exactly.
    """
    if res is None:
        return {
            f"{prefix}_success": False,
            f"{prefix}_error_msg": "Not executed",
            f"{prefix}_time_s": 0.0,
            f"{prefix}_rows": 0,
        }
    return {
        f"{prefix}_success": bool(res.get("success", False)),
        f"{prefix}_error_msg": (
            str(res.get("error", "") or "") if res.get("error") else None
        ),
        f"{prefix}_time_s": res.get("execution_time", 0.0),
        f"{prefix}_rows": res.get("rows_affected", 0),
    }


def fill_question_text(text: str, variables: dict) -> str:
    """Substitute variables into the question text (e.g. number0 -> 100)."""
    out = text
    for k, v in variables.items():
        out = out.replace(k, str(v))
    return out


def pred_vs_gold_match(pred_res: Optional[dict], gold_res: Optional[dict]) -> bool:
    """
    Execution-based equivalence.
    """
    if not pred_res or not gold_res:
        return False
    if not pred_res.get("success") or not gold_res.get("success"):
        return False

    pred_df = pred_res.get("result")
    gold_df = gold_res.get("result")
    if pred_df is not None and gold_df is not None:
        return bool(compare_results(pred_df, gold_df))

    return False


def count_prompt_tokens_effective(
    agent: QwenAgent, schema_compact: str, question: str, max_new_tokens: int
) -> int:
    """
    EXACT prompt token count as actually fed into model, including truncation logic.

    We call the agent's internal prompt constructor/truncation helper and count
    the resulting input_ids length.

    This is the correct "prompt_tokens" for prompt complexity and context pressure.
    """
    inputs = agent._make_inputs_under_limit(
        schema_compact, question, max_new_tokens=max_new_tokens
    )
    # inputs["input_ids"] is shape [1, seq_len]
    return int(inputs["input_ids"].shape[1])


def _count_from_sources(sql_upper: str) -> int:
    """
    Count number of table sources in FROM clause:
    - supports implicit joins (comma-separated)
    - supports explicit JOINs
    """
    m = re.search(
        r"\bFROM\b(.*?)(\bWHERE\b|\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|;|$)",
        sql_upper,
        flags=re.DOTALL,
    )
    if not m:
        return 0

    from_part = m.group(1)

    # Remove anything inside parentheses to avoid counting subquery FROMs as sources here
    # (we already score subqueries separately via SELECT count)
    from_part_no_parens = re.sub(r"\([^()]*\)", " ", from_part)

    # Implicit joins: tables separated by commas
    comma_sources = 0
    if from_part_no_parens.strip():
        comma_sources = from_part_no_parens.count(",") + 1

    # Explicit joins: each JOIN introduces another source
    join_sources = len(re.findall(r"\bJOIN\b", from_part_no_parens))

    # If JOIN syntax is used, sources are typically (1 + #JOIN)
    # If comma syntax is used, sources are (1 + #commas)
    # If both appear (rare), take the max to be safe.
    return max(comma_sources, 1 + join_sources if join_sources > 0 else 0)


def read_last_row_id(jsonl_path: Path) -> int:
    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        return -1

    last_id = -1
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj and isinstance(obj["id"], int):
                    last_id = max(last_id, obj["id"])
            except json.JSONDecodeError:
                # ignore malformed lines
                pass
    return last_id


def derive_out_paths(args_out: str, dataset_name: str, rdbms: str) -> Path:
    """
    If args_out is empty -> results/qwen_benchmark_<dataset>_<rdbms>.jsonl

    If args_out is provided:
      - if endswith .jsonl -> insert _<rdbms> before suffix
        (e.g. foo.jsonl -> foo_mysql.jsonl / foo_mariadb.jsonl)
      - else -> treat as a prefix and append _<rdbms>.jsonl
        (e.g. foo -> foo_mysql.jsonl)
    """
    if not args_out:
        return Path("results") / f"qwen_benchmark_{dataset_name}_{rdbms}.jsonl"

    p = Path(args_out)
    if p.suffix.lower() == ".jsonl":
        return p.with_name(f"{p.stem}_{rdbms}{p.suffix}")
    return Path(f"{args_out}_{rdbms}.jsonl")

# ----------------------------
# SQL difficulty scoring
# ----------------------------


def sql_difficulty_1to4(sql: str) -> int:
    s = sql.upper()

    # core complexity signals
    selects = len(re.findall(r"\bSELECT\b", s))
    subqueries = max(0, selects - 1)

    has_group = "GROUP BY" in s
    has_having = "HAVING" in s
    has_set_ops = any(op in s for op in ("UNION", "INTERSECT", "EXCEPT"))

    sources = _count_from_sources(s)  # implicit/explicit join proxy

    # Difficulty 4
    if has_set_ops or subqueries >= 2 or sources >= 4:
        return 4

    # Difficulty 3
    if subqueries == 1 or has_having or sources == 3:
        return 3

    # Difficulty 2
    if has_group or sources == 2:
        return 2

    # Difficulty 1
    return 1


def get_difficulty(entry: dict, sentence: dict) -> int:
    """
    Return difficulty score in {1,2,3,4}.
    Always defined.
    Priority:
      1) sentence["difficulty"]
      2) entry["difficulty"]
      3) derived from gold SQL structure
    """

    # Dataset-provided difficulty (preferred)
    if isinstance(sentence, dict) and "difficulty" in sentence:
        return int(sentence["difficulty"])
    if isinstance(entry, dict) and "difficulty" in entry:
        return int(entry["difficulty"])

    # Derive from SQL
    sql_list = entry.get("sql", [])

    if not sql_list:
        return 1  # default to easiest if no SQL
    return sql_difficulty_1to4(str(sql_list[0]))


#  -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset JSON"
    )
    parser.add_argument(
        "--rdbms", type=str, default="mysql", choices=["mysql", "mariadb", "both"]
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max entries to process (0=all)"
    )
    parser.add_argument(
        "--max_tables", type=int, default=4, help="Max tables for compact schema."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max tokens to generate for SQL.",
    )
    parser.add_argument("--out", type=str, default="", help="Custom output path")
    args = parser.parse_args()

    # Setup Paths & Data
    dataset_path = Path(args.dataset)
    dataset_name = dataset_path.stem
    llm_name = "qwen"

    try:
        data = load_dataset(dataset_path)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1

    agent = QwenAgent()
    print(f"✅ Loaded QwenAgent successfully.")

    both_mode = (args.rdbms == "both")

    if not both_mode:
        out_path = derive_out_paths(args.out, dataset_name, args.rdbms)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"🚀 Qwen Text2SQL Benchmark Runner")
        print("=" * 80)
        print(f"Dataset file: {dataset_path}")
        print(f"Dataset name (DB): {dataset_name}")
        print(f"LLM: {llm_name}")
        print(f"RDBMS: {args.rdbms}")
        print(f"Output: {out_path}")
        print(f"Question limit: {args.limit if args.limit > 0 else 'ALL'}")
        print(f"Schema max tables: {args.max_tables}")
        print(f"Max new tokens: {args.max_new_tokens}")
        print("=" * 80)

        last_id = read_last_row_id(out_path)
        resume_from = 0
        if last_id >= 0:
            resume_from = last_id + 1

        

        # Initialize Components
        
        db_manager = DatabaseManager(args.rdbms)

        # Get table names for normalization/repair (Crucial for fairness)
        schema_tables = db_manager.get_table_names(dataset_name)
        schema_num_tables = len(schema_tables)
        schema_map = db_manager.get_schema_map(database=dataset_name)
        table_map = build_identifier_maps(db_manager, dataset_name)

        row_id = 0
        questions_processed = 0
        skipped = 0

        # Processing Loop
        mode = "a" if out_path.exists() and out_path.stat().st_size > 0 else "w"
        mariadb_out_path = out_path.parent / f"qwen_{dataset_name}_mariadb.jsonl"
        with out_path.open(mode, encoding="utf-8") as f, mariadb_out_path.open("w", encoding="utf-8") as f_mariadb:
            for entry in data:
                # Metadata
                query_split = entry.get("query-split", "")
                sql_variants = entry.get("sql", [])
                if not isinstance(sql_variants, list):
                    sql_variants = [str(sql_variants)]
                gold_sql_first = sql_variants[0] if sql_variants else ""

                # Iterate over paraphrases (sentences) - EXACTLY as GPT-2 does
                sentences = entry.get("sentences", [])
                for sentence in sentences:
                    if args.limit > 0 and questions_processed >= args.limit:
                        break

                    if row_id < resume_from:
                        row_id += 1
                        continue

                    question_text = sentence.get("text", "")
                    question_vars = sentence.get("variables", {})
                    question_text_filled = fill_question_text(question_text, question_vars)

                    question_split = sentence.get("question-split", "")
                    difficulty = get_difficulty(entry, sentence)

                    schema_compact = db_manager.get_compact_schema(
                        database=dataset_name,
                        question=question_text_filled,
                        max_tables=args.max_tables,
                    )

                    schema_num_tables, schema_num_columns = parse_schema_counts(
                        schema_compact
                    )

                    prompt_tokens = count_prompt_tokens_effective(
                        agent,
                        schema_compact,
                        question_text_filled,
                        max_new_tokens=args.max_new_tokens,
                    )

                    # Prepare Gold SQL for execution
                    gold_sql_exec = fill_gold_sql(entry, sentence)
                    gold_sql_exec = normalize_table_case(gold_sql_exec, table_map)

                    # --- A. GENERATION ---
                    t0 = time.time()

                    try:
                        # Expecting tuple (sql, prompt_tokens, completion_tokens)
                        pred_sql_raw = agent.generate_sql(
                            schema=schema_compact,
                            question=question_text_filled,
                            max_new_tokens=args.max_new_tokens,
                            max_time=240.0,  # 4 minutes per query
                        )
                    except Exception as e:
                        skipped += 1
                        print(f"[{row_id}] Skipping due to generation error: {e}")
                        print(f"Current skipped count: {skipped}")
                        continue

                    gen_time_s = time.time() - t0

                    # --- B. NORMALIZATION & REPAIR ---
                    # Apply the EXACT same repair logic as GPT-2 to ensure fair scoring
                    pred_sql = normalize_pred_sql(pred_sql_raw, schema_tables)
                    pred_sql = normalize_table_case(pred_sql, table_map)
                    pred_sql, pred_repairs = repair_pred_table_names(
                        pred_sql, schema_tables
                    )
                    pred_sql_fixed, pred_col_repairs = repair_pred_column_names(
                        pred_sql, schema_map
                    )

                    # --- C. EXECUTION ---

                    db_manager.switch_database(dataset_name)
                    # Execute Prediction
                    pred_res = db_manager.execute_query(pred_sql_fixed)
                    # Execute Gold
                    gold_res = db_manager.execute_query(gold_sql_exec)

                    # --- D. COMPARISON ---
                    match = pred_vs_gold_match(pred_res, gold_res)

                    # --- E. RECORDING ---
                    # This dictionary structure matches run_gpt2xl_baseline.py exactly
                    record = {
                        "id": row_id,
                        "dataset": dataset_name,
                        "llm": "qwen",
                        "rdbms": args.rdbms,
                        # Question Info
                        "question_text": question_text,
                        "question_text_filled": question_text_filled,
                        "question_variables": question_vars,
                        "query_split": query_split,
                        "question_split": question_split,
                        "difficulty": difficulty,
                        # Gold Info
                        "gold_sql_first": gold_sql_first,
                        "gold_sql_exec": gold_sql_exec,
                        # Schema / Prompt Info
                        "schema_compact": schema_compact,
                        "schema_num_tables": schema_num_tables,
                        "schema_num_columns": schema_num_columns,
                        "prompt_tokens": prompt_tokens,
                        # Prediction Info
                        "pred_sql_raw": pred_sql_raw,
                        "pred_sql": pred_sql_fixed,
                        "pred_repairs": pred_repairs,
                        "pred_col_repairs": pred_col_repairs,
                        "gen_time_s": round(gen_time_s, 6),
                    }

                    
                    # Flatten execution results
                    record.update(pack_exec_fields(f"{args.rdbms}_pred", pred_res))
                    record.update(pack_exec_fields(f"{args.rdbms}_gold", gold_res))
                    
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
                    # Console Feedback (Formatted like GPT-2)
                    pred_ok = "OK" if pred_res.get("success") else "FAIL"
                    gold_ok = "OK" if gold_res.get("success") else "FAIL"
                    acc = "✔" if match else "✘"

                    print(
                        f"[{row_id}] qsplit={query_split or '-'} "
                        f"pred={pred_ok} gold={gold_ok} ex={acc} "
                        f"tables={schema_num_tables} prompt_tokens={prompt_tokens}"
                    )

                    row_id += 10
                    questions_processed += 1

                if args.limit > 0 and questions_processed >= args.limit:
                    break

        db_manager.close()

        print("\n" + "=" * 80)
        print("✅ Done")
        print("=" * 80)
        print(f"Questions processed: {questions_processed}")
        print(f"Wrote JSONL: {out_path}")
        return 0

    # -------------------------
    # BOTH MODE (mysql + mariadb)
    # -------------------------
    mysql_out = derive_out_paths(args.out, dataset_name, "mysql")
    mariadb_out = derive_out_paths(args.out, dataset_name, "mariadb")
    mysql_out.parent.mkdir(parents=True, exist_ok=True)
    mariadb_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Qwen Text2SQL Benchmark Runner")
    print("=" * 80)
    print(f"Dataset file: {dataset_path}")
    print(f"Dataset name (DB): {dataset_name}")
    print(f"LLM: {llm_name}")
    print(f"RDBMS: both")
    print(f"MySQL output:   {mysql_out}")
    print(f"MariaDB output: {mariadb_out}")
    print(f"Question limit: {args.limit if args.limit > 0 else 'ALL'}")
    print(f"Schema max tables: {args.max_tables}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("=" * 80)

    mysql_resume_from = read_last_row_id(mysql_out) + 1
    mariadb_resume_from = read_last_row_id(mariadb_out) + 1

    db_mysql = DatabaseManager("mysql")
    db_mariadb = DatabaseManager("mariadb")

    # Important: schema/repairs should be identical across engines for fairness.
    # We build schema_tables / schema_map / table_map from ONE engine (mysql) and reuse.
    schema_tables = db_mysql.get_table_names(dataset_name)
    schema_map = db_mysql.get_schema_map(database=dataset_name)
    table_map = build_identifier_maps(db_mysql, dataset_name)

    mysql_mode = "a" if mysql_out.exists() and mysql_out.stat().st_size > 0 else "w"
    mariadb_mode = "a" if mariadb_out.exists() and mariadb_out.stat().st_size > 0 else "w"

    row_id = 0
    questions_processed = 0
    skipped = 0

    with mysql_out.open(mysql_mode, encoding="utf-8") as f_mysql, \
         mariadb_out.open(mariadb_mode, encoding="utf-8") as f_mariadb:

        for entry in data:
            query_split = entry.get("query-split", "")
            sql_variants = entry.get("sql", [])
            if not isinstance(sql_variants, list):
                sql_variants = [str(sql_variants)]
            gold_sql_first = sql_variants[0] if sql_variants else ""

            sentences = entry.get("sentences", [])
            for sentence in sentences:
                if args.limit > 0 and questions_processed >= args.limit:
                    break

                # If BOTH files already have this id, skip quickly (like two resumed runs)
                mysql_needs = (row_id >= mysql_resume_from)
                mariadb_needs = (row_id >= mariadb_resume_from)
                if not mysql_needs and not mariadb_needs:
                    row_id += 1
                    continue

                question_text = sentence.get("text", "")
                question_vars = sentence.get("variables", {})
                question_text_filled = fill_question_text(question_text, question_vars)
                question_split = sentence.get("question-split", "")
                difficulty = get_difficulty(entry, sentence)

                # Use one engine (mysql) to build compact schema for determinism
                schema_compact = db_mysql.get_compact_schema(
                    database=dataset_name,
                    question=question_text_filled,
                    max_tables=args.max_tables,
                )
                schema_num_tables, schema_num_columns = parse_schema_counts(schema_compact)

                prompt_tokens = count_prompt_tokens_effective(
                    agent,
                    schema_compact,
                    question_text_filled,
                    max_new_tokens=args.max_new_tokens,
                )

                gold_sql_exec = fill_gold_sql(entry, sentence)
                gold_sql_exec = normalize_table_case(gold_sql_exec, table_map)

                # --- A. GENERATION (done ONCE, used for both engines) ---
                t0 = time.time()
                try:
                    pred_sql_raw = agent.generate_sql(
                        schema=schema_compact,
                        question=question_text_filled,
                        max_new_tokens=args.max_new_tokens,
                        max_time=240.0,
                    )
                except Exception as e:
                    skipped += 1
                    print(f"[{row_id}] Skipping due to generation error: {e}")
                    print(f"Current skipped count: {skipped}")
                    row_id += 1
                    continue
                gen_time_s = time.time() - t0

                # --- B. NORMALIZATION & REPAIR ---
                pred_sql = normalize_pred_sql(pred_sql_raw, schema_tables)
                pred_sql = normalize_table_case(pred_sql, table_map)
                pred_sql, pred_repairs = repair_pred_table_names(pred_sql, schema_tables)
                pred_sql_fixed, pred_col_repairs = repair_pred_column_names(pred_sql, schema_map)

                # --- C. EXECUTION (both engines) ---
                db_mysql.switch_database(dataset_name)
                db_mariadb.switch_database(dataset_name)

                pred_res_mysql = db_mysql.execute_query(pred_sql_fixed)
                gold_res_mysql = db_mysql.execute_query(gold_sql_exec)
                match_mysql = pred_vs_gold_match(pred_res_mysql, gold_res_mysql)

                pred_res_mariadb = db_mariadb.execute_query(pred_sql_fixed)
                gold_res_mariadb = db_mariadb.execute_query(gold_sql_exec)
                match_mariadb = pred_vs_gold_match(pred_res_mariadb, gold_res_mariadb)

                base_record = {
                    "id": row_id,
                    "dataset": dataset_name,
                    "llm": "qwen",
                    "question_text": question_text,
                    "question_text_filled": question_text_filled,
                    "question_variables": question_vars,
                    "query_split": query_split,
                    "question_split": question_split,
                    "difficulty": difficulty,
                    "gold_sql_first": gold_sql_first,
                    "gold_sql_exec": gold_sql_exec,
                    "schema_compact": schema_compact,
                    "schema_num_tables": schema_num_tables,
                    "schema_num_columns": schema_num_columns,
                    "prompt_tokens": prompt_tokens,
                    "pred_sql_raw": pred_sql_raw,
                    "pred_sql": pred_sql_fixed,
                    "pred_repairs": pred_repairs,
                    "pred_col_repairs": pred_col_repairs,
                    "gen_time_s": round(gen_time_s, 6),
                }

                if mysql_needs:
                    rec_mysql = dict(base_record)
                    rec_mysql["rdbms"] = "mysql"
                    rec_mysql.update(pack_exec_fields("mysql_pred", pred_res_mysql))
                    rec_mysql.update(pack_exec_fields("mysql_gold", gold_res_mysql))
                    rec_mysql["mysql_pred_vs_gold_match"] = bool(match_mysql)
                    f_mysql.write(json.dumps(rec_mysql, ensure_ascii=False) + "\n")

                if mariadb_needs:
                    rec_mariadb = dict(base_record)
                    rec_mariadb["rdbms"] = "mariadb"
                    rec_mariadb.update(pack_exec_fields("mariadb_pred", pred_res_mariadb))
                    rec_mariadb.update(pack_exec_fields("mariadb_gold", gold_res_mariadb))
                    rec_mariadb["mariadb_pred_vs_gold_match"] = bool(match_mariadb)
                    f_mariadb.write(json.dumps(rec_mariadb, ensure_ascii=False) + "\n")

                # Console feedback (show both)
                pm = "OK" if pred_res_mysql.get("success") else "FAIL"
                gm = "OK" if gold_res_mysql.get("success") else "FAIL"
                am = "✔" if match_mysql else "✘"

                pmd = "OK" if pred_res_mariadb.get("success") else "FAIL"
                gmd = "OK" if gold_res_mariadb.get("success") else "FAIL"
                amd = "✔" if match_mariadb else "✘"

                print(
                    f"[{row_id}] qsplit={query_split or '-'} "
                    f"mysql: pred={pm} gold={gm} ex={am} | "
                    f"mariadb: pred={pmd} gold={gmd} ex={amd} "
                    f"tables={schema_num_tables} prompt_tokens={prompt_tokens}"
                )

                row_id += 1
                questions_processed += 1

            if args.limit > 0 and questions_processed >= args.limit:
                break

    db_mysql.close()
    db_mariadb.close()

    print("\n" + "=" * 80)
    print("✅ Done (both)")
    print("=" * 80)
    print(f"Questions processed: {questions_processed}")
    print(f"Wrote JSONL (mysql):   {mysql_out}")
    print(f"Wrote JSONL (mariadb): {mariadb_out}")

if __name__ == "__main__":
    main()
