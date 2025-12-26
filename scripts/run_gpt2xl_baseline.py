"""
scripts/run_gpt2xl_baseline.py

Run a simple, reproducible Text2SQL baseline using GPT-2 XL on a Text2SQL dataset
in the jkkummerfeld/text2sql-data JSON format.

What it does (per entry -> per sentence):
  1) Read dataset JSON (list of query entries)
  2) For each sentence (question) inside an entry:
      - build a compact schema (table + columns), optionally filtered by question keywords
      - generate SQL via GPT-2 XL (deterministic)
      - execute predicted SQL on MySQL and MariaDB
      - write one JSONL record per sentence to results/

Notes:
  - GPT-2 XL has a 1024-token context window; the agent should compact schema (and/or truncate).
  - This script assumes the dataset database exists in both RDBMS with the same name as the dataset file stem
    e.g., datasets_source/data/advising.json -> database "advising"
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gpt2xl_agent import GPT2XLAgent  # noqa: E402
from database.db_manager import DatabaseManager, compare_results  # noqa: E402


def get_query_split(entry: dict) -> str:
    return str(entry.get("query-split", ""))


def get_sql_variants(entry: dict) -> list[str]:
    sql_list = entry.get("sql", [])
    if isinstance(sql_list, list):
        return [str(x) for x in sql_list]
    return [str(sql_list)] if sql_list else []


def iter_sentences(entry: dict):
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


def get_sentence_variables(sentence: dict) -> dict:
    vars_map = sentence.get("variables", {})
    return vars_map if isinstance(vars_map, dict) else {}


def load_dataset(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Dataset JSON must be a list, got: {type(data)}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets_source/data/advising.json",
        help="Path to dataset JSON file (text2sql-data format).",
    )
    parser.add_argument(
        "--limit_entries",
        type=int,
        default=50,
        help="Process only the first N entries (each entry may contain multiple sentences).",
    )
    parser.add_argument(
        "--max_tables",
        type=int,
        default=12,
        help="Max tables to include in compact schema after optional filtering.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=120,
        help="Max tokens to generate for SQL.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/gpt2xl_baseline.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_path.stem  # used as DB name convention

    print("🧪 GPT-2 XL Text2SQL Baseline Runner")
    print("=" * 70)
    print(f"Dataset file: {dataset_path}")
    print(f"Dataset name (DB): {dataset_name}")
    print(f"Output: {out_path}")
    print(f"Entry limit: {args.limit_entries}")
    print(f"Schema max tables: {args.max_tables}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("=" * 70)

    data = load_dataset(dataset_path)

    # Initialize model once
    agent = GPT2XLAgent()

    # Reuse DB connections
    mysql = DatabaseManager("mysql")
    maria = DatabaseManager("mariadb")

    # Counters
    row_id = 0
    n_ok_mysql = 0
    n_ok_maria = 0
    n_both_ok = 0
    n_match = 0

    with out_path.open("w", encoding="utf-8") as f:
        for entry in data[: args.limit_entries]:
            query_split = get_query_split(entry)
            sql_variants = get_sql_variants(entry)
            gold_sql_first = sql_variants[0] if sql_variants else ""

            for sentence in iter_sentences(entry):
                question_text = get_sentence_text(sentence)
                question_split = get_question_split(sentence)
                question_vars = get_sentence_variables(sentence)

                # Build compact schema using MySQL inspector (assumes both DBs share schema)
                # IMPORTANT: your DatabaseManager must include get_compact_schema(...) as discussed.
                schema_compact = mysql.get_compact_schema(
                    database=dataset_name,
                    question=question_text,
                    max_tables=args.max_tables,
                )

                # Generate SQL
                t0 = time.time()
                pred_sql = agent.generate_sql(
                    schema=schema_compact,
                    question=question_text,
                    max_new_tokens=args.max_new_tokens,
                )
                gen_time = time.time() - t0

                # Execute on MySQL
                mysql.switch_database(dataset_name)
                mysql_res = mysql.execute_query(pred_sql)

                # Execute on MariaDB
                maria.switch_database(dataset_name)
                maria_res = maria.execute_query(pred_sql)

                # Result matching (only if both succeeded and returned tabular results)
                match = None
                if mysql_res.get("success") and maria_res.get("success"):
                    n_both_ok += 1
                    if mysql_res.get("result") is not None and maria_res.get("result") is not None:
                        match = compare_results(mysql_res["result"], maria_res["result"])
                        if match:
                            n_match += 1

                if mysql_res.get("success"):
                    n_ok_mysql += 1
                if maria_res.get("success"):
                    n_ok_maria += 1

                record = {
                    "id": row_id,
                    "dataset": dataset_name,

                    # Format-consistent metadata
                    "query_split": query_split,
                    "question_split": question_split,
                    "question_text": question_text,
                    "question_variables": question_vars,

                    # Gold SQL (variables may remain as placeholders)
                    "gold_sql_first": gold_sql_first,
                    "gold_sql_variants": sql_variants,

                    # Prompt inputs
                    "schema_compact": schema_compact,

                    # Model output + timings
                    "pred_sql": pred_sql,
                    "gen_time_s": round(gen_time, 4),

                    # DB exec results
                    "mysql": {
                        "success": mysql_res.get("success"),
                        "execution_time_s": mysql_res.get("execution_time"),
                        "rows": mysql_res.get("rows_affected"),
                        "error": mysql_res.get("error"),
                    },
                    "mariadb": {
                        "success": maria_res.get("success"),
                        "execution_time_s": maria_res.get("execution_time"),
                        "rows": maria_res.get("rows_affected"),
                        "error": maria_res.get("error"),
                    },
                    "mysql_vs_mariadb_match": match,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(
                    f"[{row_id}] qsplit={query_split or '-'} ssplit={question_split or '-'} "
                    f"mysql={'OK' if mysql_res.get('success') else 'FAIL'} "
                    f"maria={'OK' if maria_res.get('success') else 'FAIL'}"
                )

                row_id += 1

    mysql.close()
    maria.close()

    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    print(f"Total questions processed: {row_id}")
    if row_id > 0:
        print(f"MySQL success rate:   {n_ok_mysql}/{row_id} ({n_ok_mysql/row_id*100:.1f}%)")
        print(f"MariaDB success rate: {n_ok_maria}/{row_id} ({n_ok_maria/row_id*100:.1f}%)")
        print(f"Both succeeded:       {n_both_ok}/{row_id} ({n_both_ok/row_id*100:.1f}%)")
        if n_both_ok > 0:
            print(f"Result match rate:    {n_match}/{n_both_ok} ({n_match/n_both_ok*100:.1f}%)")
    print(f"\n✅ Wrote results to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
