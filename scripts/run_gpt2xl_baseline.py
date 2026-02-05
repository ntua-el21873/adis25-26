"""
scripts/run_gpt2xl_benchmark.py

Research-grade GPT-2 XL Text2SQL runner producing JSONL rows that match the
"Evaluation Metrics & JSONL Specification" contract.

One run = one dataset + one RDBMS.
No aggregation here; metrics are derived later from CSV.

Required output fields included:
- Identifiers: id, dataset, llm, rdbms
- Question metadata: question_text, question_text_filled, question_variables, query_split, question_split, difficulty (optional)
- Gold SQL: gold_sql_first, gold_sql_exec
- Schema/prompt: schema_compact, schema_num_tables, schema_num_columns (optional), prompt_tokens
- LLM output: pred_sql_raw, pred_sql, gen_time_s
- Execution (flat namespaced): {rdbms}_pred.success/time/error, {rdbms}_gold..., {rdbms}_pred_vs_gold_match
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gpt2xl_agent import GPT2XLAgent
from database.db_manager import DatabaseManager
from sql_utils import (
    fill_gold_sql,
    normalize_pred_sql,
    compare_results,
    repair_pred_table_names,
    repair_pred_column_names,
    parse_schema_counts,
    normalize_table_case,
)


# ----------------------------
# Dataset helpers
# ----------------------------


def load_dataset(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Dataset JSON must be a list, got: {type(data)}")
    return data


def get_query_split(entry: dict) -> str:
    return str(entry.get("query-split", ""))


def get_sql_variants(entry: dict) -> List[str]:
    sql_list = entry.get("sql", [])
    if isinstance(sql_list, list):
        return [str(x) for x in sql_list]
    return [str(sql_list)] if sql_list else []


def iter_sentences(entry: dict) -> Iterable[dict]:
    sentences = entry.get("sentences", [])
    if not isinstance(sentences, list):
        return
    for s in sentences:
        if isinstance(s, dict):
            yield s


def get_sentence_text(sentence: dict) -> str:
    return str(sentence.get("text", ""))


def get_question_split(sentence: dict) -> str:
    return str(sentence.get("question-split", ""))


def get_sentence_variables(sentence: dict) -> Dict[str, Any]:
    vars_map = sentence.get("variables", {})
    return vars_map if isinstance(vars_map, dict) else {}


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



def derive_out_path(args_out: str, dataset_name: str, rdbms: str) -> Path:
    """
    If args_out is empty -> default_out_path(dataset_name, rdbms)

    If args_out is provided:
      - if endswith .jsonl -> insert _<rdbms> before suffix
        (e.g. foo.jsonl -> foo_mysql.jsonl / foo_mariadb.jsonl)
      - else -> treat as a prefix and append _<rdbms>.jsonl
        (e.g. foo -> foo_mysql.jsonl)
    """
    if not args_out or not args_out.strip():
        return default_out_path(dataset_name, rdbms)

    p = Path(args_out)
    if p.suffix.lower() == ".jsonl":
        return p.with_name(f"{p.stem}_{rdbms}{p.suffix}")
    return Path(f"{args_out}_{rdbms}.jsonl")


def print_header(
    dataset_path: Path,
    dataset_name: str,
    llm_name: str,
    rdbms: str,
    out_paths: List[Tuple[str, Path]],
    args,
) -> None:
    print("🧪 GPT-2 XL Text2SQL Benchmark Runner")
    print("=" * 80)
    print(f"Dataset file: {dataset_path}")
    print(f"Dataset name (DB): {dataset_name}")
    print(f"LLM: {llm_name}")
    print(f"RDBMS: {rdbms}")
    for label, p in out_paths:
        print(f"{label}: {p}")
    print(f"Question limit: {args.limit if args.limit > 0 else 'ALL'}")
    print(f"Schema max tables: {args.max_tables}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Entry start index: {args.entry_idx}")
    print("=" * 80)


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


# ----------------------------
# Variable substitution for question_text_filled (best-effort)
# ----------------------------


def build_identifier_maps(db: DatabaseManager, dataset_name: str):
    """
    Input: DatabaseManager + dataset name
    Output: table_map: lowercase_table -> actual_table
    """
    # Actual table names from DB
    tables = db.get_table_names(
        database=dataset_name
    )  # e.g. ["airport", "flight", ...]
    table_map = {t.lower(): t for t in tables}  # map lowercase -> actual

    # Optional: columns too, if you have/get them
    # columns = db.get_all_columns(database=dataset_name)  # you may need to add this
    # col_map = {c.lower(): c for c in columns}

    return table_map


def fill_question_text(question_text: str, variables: Dict[str, Any]) -> str:
    """
    Substitute variables in question text when placeholders appear as bare tokens,
    e.g. "airport_code0" -> "MKE".

    We replace only whole tokens using regex boundaries:
      - Not preceded by [A-Za-z0-9_]
      - Not followed by [A-Za-z0-9_]
    so we don't accidentally replace substrings.
    """
    filled = question_text
    if not variables:
        return filled

    # Replace longer keys first to avoid edge cases like var1 vs var10
    for k in sorted(variables.keys(), key=len, reverse=True):
        v = variables[k]
        val = str(v)

        # Whole-token match for identifiers (letters/digits/underscore)
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(str(k))}(?![A-Za-z0-9_])")
        filled = pattern.sub(val, filled)

    return filled


# ----------------------------
# Schema/prompt instrumentation
# ----------------------------


def count_prompt_tokens_effective(
    agent: GPT2XLAgent, schema_compact: str, question: str, max_new_tokens: int
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


_QUOTED = re.compile(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")")


def normalize_table_case(sql: str, table_map: Dict[str, str]) -> str:
    """
    Replace table names in SQL to match the *actual* case in the DB.
    - table_map: lowercase_table -> actual_table
    - avoids changing inside single/double quoted strings.
    - replaces whole tokens only.
    """
    if not sql:
        return sql

    parts = _QUOTED.split(sql)  # keeps delimiters
    for i in range(0, len(parts), 2):  # only outside quotes
        chunk = parts[i]

        # Replace longest names first (airport_service before airport)
        for key in sorted(table_map.keys(), key=len, reverse=True):
            actual = table_map[key]
            # token boundary: not surrounded by [A-Za-z0-9_]
            chunk = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])",
                actual,
                chunk,
                flags=re.IGNORECASE,  # match any case in gold/pred
            )

        parts[i] = chunk

    return "".join(parts)


# ----------------------------
# Execution result packing
# ----------------------------


def pack_exec_fields(prefix: str, exec_res: Optional[dict]) -> Dict[str, Any]:
    """
    Flat JSON fields:
      {prefix}.success
      {prefix}.execution_time_s
      {prefix}.error
    """
    if exec_res is None:
        return {
            f"{prefix}.success": False,
            f"{prefix}.execution_time_s": None,
            f"{prefix}.error": "NO_EXECUTION_ATTEMPT",
        }

    return {
        f"{prefix}.success": bool(exec_res.get("success")),
        f"{prefix}.execution_time_s": exec_res.get("execution_time"),
        f"{prefix}.error": exec_res.get("error"),
    }


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


# ----------------------------
# Output naming
# ----------------------------


def default_out_path(dataset_name: str, rdbms: str) -> Path:
    return Path("results") / f"gpt2xl_benchmark_{dataset_name}_{rdbms}.jsonl"


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset JSON file."
    )
    parser.add_argument(
        "--rdbms",
        type=str,
        required=True,
        choices=["mysql", "mariadb", "both"],
        help="One RDBMS per run (or both).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, process only first N QUESTIONS (sentences).",
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
    parser.add_argument(
        "--out", type=str, default="", help="Optional output JSONL path (or prefix)."
    )
    parser.add_argument(
        "--entry_idx",
        type=int,
        default=0,
        help="Start from this entry index in the dataset list.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return 1

    dataset_name = dataset_path.stem
    llm_name = "gpt2xl"
    both_mode = args.rdbms == "both"

    data = load_dataset(dataset_path)

    # Initialize model once
    agent = GPT2XLAgent()

    # -------------------------
    # SINGLE RDBMS MODE
    # -------------------------
    if not both_mode:
        rdbms = args.rdbms
        out_path = derive_out_path(args.out, dataset_name, rdbms)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print_header(
            dataset_path, dataset_name, llm_name, rdbms, [("Output", out_path)], args
        )

        last_id = read_last_row_id(out_path)
        resume_from = (last_id + 1) if last_id >= 0 else 0

        db = DatabaseManager(rdbms)
        db.switch_database(dataset_name)

        schema_tables = db.get_table_names(database=dataset_name)
        table_map = build_identifier_maps(db, dataset_name)
        schema_map = db.get_schema_map()

        row_id = 0
        questions_processed = 0
        skipped = 0

        mode = "a" if out_path.exists() and out_path.stat().st_size > 0 else "w"
        with out_path.open(mode, encoding="utf-8") as f:
            for entry in data[args.entry_idx :]:
                query_split = get_query_split(entry)
                sql_variants = get_sql_variants(entry)
                gold_sql_first = sql_variants[0] if sql_variants else ""

                for sentence in iter_sentences(entry):
                    if args.limit > 0 and questions_processed >= args.limit:
                        break

                    if row_id < resume_from:
                        row_id += 1
                        continue

                    question_text = get_sentence_text(sentence)
                    question_vars = get_sentence_variables(sentence)
                    question_text_filled = fill_question_text(
                        question_text, question_vars
                    )

                    question_split = get_question_split(sentence)
                    difficulty = get_difficulty(entry, sentence)

                    schema_compact = db.get_compact_schema(
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

                    gold_sql_exec = fill_gold_sql(entry, sentence)
                    gold_sql_exec = normalize_table_case(gold_sql_exec, table_map)

                    t0 = time.time()
                    try:
                        pred_sql_raw = agent.generate_sql(
                            schema=schema_compact,
                            question=question_text_filled,
                            max_new_tokens=args.max_new_tokens,
                            max_time=240.0,
                        )
                    except RuntimeError:
                        skipped += 1
                        print(
                            f"[SKIP {row_id}] RuntimeError during generation (skipped so far: {skipped})"
                        )
                        row_id += 1
                        continue
                    gen_time_s = time.time() - t0

                    pred_sql = normalize_pred_sql(pred_sql_raw, schema_tables)
                    pred_sql = normalize_table_case(pred_sql, table_map)
                    pred_sql, pred_repairs = repair_pred_table_names(
                        pred_sql, schema_tables
                    )
                    pred_sql, pred_col_repairs = repair_pred_column_names(
                        pred_sql, schema_map
                    )

                    db.switch_database(dataset_name)
                    pred_res = db.execute_query(pred_sql)
                    gold_res = db.execute_query(gold_sql_exec)

                    match = pred_vs_gold_match(pred_res, gold_res)

                    record: Dict[str, Any] = {
                        "id": row_id,
                        "dataset": dataset_name,
                        "llm": llm_name,
                        "rdbms": rdbms,
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
                        "pred_sql": pred_sql,
                        "pred_repairs": pred_repairs,
                        "pred_col_repairs": pred_col_repairs,
                        "gen_time_s": round(gen_time_s, 6),
                    }

                    record.update(pack_exec_fields(f"{rdbms}_pred", pred_res))
                    record.update(pack_exec_fields(f"{rdbms}_gold", gold_res))
                    record[f"{rdbms}_pred_vs_gold_match"] = bool(match)

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    pred_ok = "OK" if pred_res and pred_res.get("success") else "FAIL"
                    gold_ok = "OK" if gold_res and gold_res.get("success") else "FAIL"
                    acc = "✔" if match else "✘"
                    print(
                        f"[{row_id}] qsplit={query_split or '-'} ssplit={question_split or '-'} "
                        f"pred={pred_ok} gold={gold_ok} ex={acc} "
                        f"tables={schema_num_tables} prompt_tokens={prompt_tokens}"
                    )

                    row_id += 1
                    questions_processed += 1

                if args.limit > 0 and questions_processed >= args.limit:
                    break

        db.close()

        print("\n" + "=" * 80)
        print("✅ Done")
        print("=" * 80)
        print(f"Questions processed: {questions_processed}")
        print(f"Wrote JSONL: {out_path}")
        return 0

    # -------------------------
    # BOTH MODE (mysql + mariadb)
    # -------------------------
    mysql_out = derive_out_path(args.out, dataset_name, "mysql")
    mariadb_out = derive_out_path(args.out, dataset_name, "mariadb")
    mysql_out.parent.mkdir(parents=True, exist_ok=True)
    mariadb_out.parent.mkdir(parents=True, exist_ok=True)

    print_header(
        dataset_path,
        dataset_name,
        llm_name,
        "both",
        [("MySQL output", mysql_out), ("MariaDB output", mariadb_out)],
        args,
    )

    mysql_resume_from = read_last_row_id(mysql_out) + 1
    mariadb_resume_from = read_last_row_id(mariadb_out) + 1

    db_mysql = DatabaseManager("mysql")
    db_mariadb = DatabaseManager("mariadb")
    db_mysql.switch_database(dataset_name)
    db_mariadb.switch_database(dataset_name)

    # For determinism/fairness, derive schema + maps from ONE engine (mysql), reuse for both
    schema_tables = db_mysql.get_table_names(database=dataset_name)
    table_map = build_identifier_maps(db_mysql, dataset_name)
    schema_map = db_mysql.get_schema_map()

    mysql_mode = "a" if mysql_out.exists() and mysql_out.stat().st_size > 0 else "w"
    mariadb_mode = (
        "a" if mariadb_out.exists() and mariadb_out.stat().st_size > 0 else "w"
    )

    row_id = 0
    questions_processed = 0
    skipped = 0

    with mysql_out.open(mysql_mode, encoding="utf-8") as f_mysql, mariadb_out.open(
        mariadb_mode, encoding="utf-8"
    ) as f_mariadb:

        for entry in data[args.entry_idx :]:
            query_split = get_query_split(entry)
            sql_variants = get_sql_variants(entry)
            gold_sql_first = sql_variants[0] if sql_variants else ""

            for sentence in iter_sentences(entry):
                if args.limit > 0 and questions_processed >= args.limit:
                    break

                mysql_needs = row_id >= mysql_resume_from
                mariadb_needs = row_id >= mariadb_resume_from

                # If both files already have this id, skip (like two resumed runs)
                if not mysql_needs and not mariadb_needs:
                    row_id += 1
                    continue

                question_text = get_sentence_text(sentence)
                question_vars = get_sentence_variables(sentence)
                question_text_filled = fill_question_text(question_text, question_vars)

                question_split = get_question_split(sentence)
                difficulty = get_difficulty(entry, sentence)

                # Build compact schema once (use mysql for determinism)
                schema_compact = db_mysql.get_compact_schema(
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

                gold_sql_exec = fill_gold_sql(entry, sentence)
                gold_sql_exec = normalize_table_case(gold_sql_exec, table_map)

                # Generate ONCE, use for both
                t0 = time.time()
                try:
                    pred_sql_raw = agent.generate_sql(
                        schema=schema_compact,
                        question=question_text_filled,
                        max_new_tokens=args.max_new_tokens,
                        max_time=240.0,
                    )
                except RuntimeError:
                    skipped += 1
                    print(
                        f"[SKIP {row_id}] RuntimeError during generation (skipped so far: {skipped})"
                    )
                    row_id += 1
                    continue
                gen_time_s = time.time() - t0

                pred_sql = normalize_pred_sql(pred_sql_raw, schema_tables)
                pred_sql = normalize_table_case(pred_sql, table_map)
                pred_sql, pred_repairs = repair_pred_table_names(
                    pred_sql, schema_tables
                )
                pred_sql, pred_col_repairs = repair_pred_column_names(
                    pred_sql, schema_map
                )

                # Execute on both
                db_mysql.switch_database(dataset_name)
                db_mariadb.switch_database(dataset_name)

                pred_res_mysql = db_mysql.execute_query(pred_sql)
                gold_res_mysql = db_mysql.execute_query(gold_sql_exec)
                match_mysql = pred_vs_gold_match(pred_res_mysql, gold_res_mysql)

                pred_res_mariadb = db_mariadb.execute_query(pred_sql)
                gold_res_mariadb = db_mariadb.execute_query(gold_sql_exec)
                match_mariadb = pred_vs_gold_match(pred_res_mariadb, gold_res_mariadb)

                base_record: Dict[str, Any] = {
                    "id": row_id,
                    "dataset": dataset_name,
                    "llm": llm_name,
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
                    "pred_sql": pred_sql,
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
                    rec_mariadb.update(
                        pack_exec_fields("mariadb_pred", pred_res_mariadb)
                    )
                    rec_mariadb.update(
                        pack_exec_fields("mariadb_gold", gold_res_mariadb)
                    )
                    rec_mariadb["mariadb_pred_vs_gold_match"] = bool(match_mariadb)
                    f_mariadb.write(json.dumps(rec_mariadb, ensure_ascii=False) + "\n")

                pm = (
                    "OK" if pred_res_mysql and pred_res_mysql.get("success") else "FAIL"
                )
                gm = (
                    "OK" if gold_res_mysql and gold_res_mysql.get("success") else "FAIL"
                )
                am = "✔" if match_mysql else "✘"

                pmd = (
                    "OK"
                    if pred_res_mariadb and pred_res_mariadb.get("success")
                    else "FAIL"
                )
                gmd = (
                    "OK"
                    if gold_res_mariadb and gold_res_mariadb.get("success")
                    else "FAIL"
                )
                amd = "✔" if match_mariadb else "✘"

                print(
                    f"[{row_id}] qsplit={query_split or '-'} ssplit={question_split or '-'} "
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
