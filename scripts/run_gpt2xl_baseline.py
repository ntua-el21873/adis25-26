"""
scripts/run_gpt2xl_benchmark.py

Research-grade GPT-2 XL Text2SQL runner producing JSONL rows that match the
"Evaluation Metrics & JSONL Specification" contract.

One run = one dataset + one RDBMS.
No aggregation here; metrics are derived later from CSV.

Required output fields included:
- Identifiers: id, dataset, llm, rdbms
- Question metadata: question_text, question_text_filled, question_variables, query_split, question_split, difficulty (optional)
- Gold SQL: gold_sql_first, gold_sql_exec, gold_sql_variants (optional)
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
from sql_utils import fill_gold_sql, normalize_pred_sql, compare_results, repair_pred_table_names


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


def get_difficulty(entry: dict, sentence: dict) -> Any:
    # Spec: include only if present in the dataset file.
    if "difficulty" in sentence:
        return sentence.get("difficulty")
    if "difficulty" in entry:
        return entry.get("difficulty")
    
    # if not present, calculate from SQL (optional)
    gold_sql = entry["sql"]
    # now we make a score for the sql query depending on how difficult it is (simple, medium, complex)
    return None


# ----------------------------
# Variable substitution for question_text_filled (best-effort)
# ----------------------------

def build_identifier_maps(db: DatabaseManager, dataset_name: str):
    # Actual table names from DB
    tables = db.get_table_names(database=dataset_name)  # e.g. ["airport", "flight", ...]
    table_map = {t.lower(): t for t in tables}          # map lowercase -> actual

    # Optional: columns too, if you have/get them
    # columns = db.get_all_columns(database=dataset_name)  # you may need to add this
    # col_map = {c.lower(): c for c in columns}

    return table_map


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

def parse_schema_counts(schema_compact: str) -> Tuple[int, Optional[int]]:
    """
    Heuristic parsing of compact schema string to estimate:
      - schema_num_tables
      - schema_num_columns

    Adjust if your schema_compact format differs.
    """
    if not schema_compact or not schema_compact.strip():
        return 0, 0

    lines = [ln.strip() for ln in schema_compact.splitlines() if ln.strip()]
    tables: List[str] = []
    col_count = 0

    for ln in lines:
        # table(col1, col2)
        m = re.match(r"^([A-Za-z_][\w]*)\s*\((.*)\)\s*$", ln)
        if m:
            t = m.group(1)
            tables.append(t)
            cols_blob = m.group(2).strip()
            if cols_blob:
                cols = [c.strip() for c in cols_blob.split(",") if c.strip()]
                col_count += len(cols)
            continue

        # table: col1, col2
        m = re.match(r"^([A-Za-z_][\w]*)\s*:\s*(.*)\s*$", ln)
        if m:
            t = m.group(1)
            tables.append(t)
            cols_blob = m.group(2).strip()
            if cols_blob:
                cols = [c.strip() for c in cols_blob.split(",") if c.strip()]
                col_count += len(cols)
            continue

    schema_num_tables = len(dict.fromkeys(tables))
    schema_num_columns = col_count if col_count >= 0 else None
    return schema_num_tables, schema_num_columns


def count_prompt_tokens_effective(agent: GPT2XLAgent, schema_compact: str, question: str, max_new_tokens: int) -> int:
    """
    EXACT prompt token count as actually fed into model, including truncation logic.

    We call the agent's internal prompt constructor/truncation helper and count
    the resulting input_ids length.

    This is the correct "prompt_tokens" for prompt complexity and context pressure.
    """
    inputs = agent._make_inputs_under_limit(schema_compact, question, max_new_tokens=max_new_tokens)
    # inputs["input_ids"] is shape [1, seq_len]
    return int(inputs["input_ids"].shape[1])


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
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON file.")
    parser.add_argument("--rdbms", type=str, required=True, choices=["mysql", "mariadb"], help="One RDBMS per run.")
    parser.add_argument("--limit", type=int, default=1, help="If > 0, process only first N QUESTIONS (sentences).")
    parser.add_argument("--max_tables", type=int, default=12, help="Max tables for compact schema.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate for SQL.")
    parser.add_argument("--out", type=str, default="", help="Optional output JSONL path.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return 1

    dataset_name = dataset_path.stem
    rdbms = args.rdbms
    llm_name = "gpt2xl"

    out_path = Path(args.out) if args.out.strip() else default_out_path(dataset_name, rdbms)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("🧪 GPT-2 XL Text2SQL Benchmark Runner")
    print("=" * 80)
    print(f"Dataset file: {dataset_path}")
    print(f"Dataset name (DB): {dataset_name}")
    print(f"LLM: {llm_name}")
    print(f"RDBMS: {rdbms}")
    print(f"Output: {out_path}")
    print(f"Question limit: {args.limit if args.limit > 0 else 'ALL'}")
    print(f"Schema max tables: {args.max_tables}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("=" * 80)

    data = load_dataset(dataset_path)

    # Initialize model once
    agent = GPT2XLAgent()

    # Use the selected RDBMS for schema introspection + execution
    db = DatabaseManager(rdbms)
    db.switch_database(dataset_name)
    schema_tables = db.get_table_names(database=dataset_name)

    table_map = build_identifier_maps(db, dataset_name)
    row_id = 0
    questions_processed = 0

    with out_path.open("w", encoding="utf-8") as f:
        for entry in data:
            query_split = get_query_split(entry)
            sql_variants = get_sql_variants(entry)
            gold_sql_first = sql_variants[0] if sql_variants else ""

            for sentence in iter_sentences(entry):
                if args.limit > 0 and questions_processed >= args.limit:
                    break

                question_text = get_sentence_text(sentence)
                question_vars = get_sentence_variables(sentence)
                question_text_filled = fill_question_text(question_text, question_vars)

                question_split = get_question_split(sentence)
                difficulty = get_difficulty(entry, sentence)

                # Compact schema for prompt
                schema_compact = db.get_compact_schema(
                    database=dataset_name,
                    question=question_text_filled,
                    max_tables=args.max_tables,
                )
                schema_num_tables, schema_num_columns = parse_schema_counts(schema_compact)

                # Exact prompt tokens as actually fed into GPT-2 (includes truncation)
                prompt_tokens = count_prompt_tokens_effective(
                    agent, schema_compact, question_text_filled, max_new_tokens=args.max_new_tokens
                )

                # Gold SQL executable (filled)
                gold_sql_exec = fill_gold_sql(entry, sentence)
                gold_sql_exec = normalize_table_case(gold_sql_exec, table_map)

                # Generate SQL (time only generation)
                t0 = time.time()
                pred_sql_raw = agent.generate_sql(
                    schema=schema_compact,
                    question=question_text_filled,
                    max_new_tokens=args.max_new_tokens,
                )
                gen_time_s = time.time() - t0

                # Normalize prediction (table casing etc.)
                pred_sql = normalize_pred_sql(pred_sql_raw, schema_tables)
                pred_sql = normalize_table_case(pred_sql, table_map)
                pred_sql, pred_repairs = repair_pred_table_names(pred_sql, schema_tables)

                # Execute predicted + gold
                db.switch_database(dataset_name)
                pred_res = db.execute_query(pred_sql)
                gold_res = db.execute_query(gold_sql_exec)

                match = pred_vs_gold_match(pred_res, gold_res)

                record: Dict[str, Any] = {
                    # Core identifiers
                    "id": row_id,
                    "dataset": dataset_name,
                    "llm": llm_name,
                    "rdbms": rdbms,

                    # Question & dataset metadata
                    "question_text": question_text,
                    "question_text_filled": question_text_filled,
                    "question_variables": question_vars,
                    "query_split": query_split,
                    "question_split": question_split,

                    # Gold SQL
                    "gold_sql_first": gold_sql_first,
                    "gold_sql_exec": gold_sql_exec,

                    # Schema/prompt information
                    "schema_compact": schema_compact,
                    "schema_num_tables": schema_num_tables,
                    "schema_num_columns": schema_num_columns,
                    "prompt_tokens": prompt_tokens,

                    # LLM output
                    "pred_sql_raw": pred_sql_raw,
                    "pred_sql": pred_sql,
                    "gen_time_s": round(gen_time_s, 6),
                }

                if sql_variants:
                    record["gold_sql_variants"] = sql_variants
                if difficulty is not None:
                    record["difficulty"] = difficulty

                # Execution results (namespaced)
                record.update(pack_exec_fields(f"{rdbms}_pred", pred_res))
                record.update(pack_exec_fields(f"{rdbms}_gold", gold_res))

                # Execution comparison
                record[f"{rdbms}_pred_vs_gold_match"] = bool(match)

                record["pred_repairs"] = pred_repairs

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Console line
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


if __name__ == "__main__":
    raise SystemExit(main())
