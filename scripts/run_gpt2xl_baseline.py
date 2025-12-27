"""
scripts/run_gpt2xl_baseline.py

Run a simple, reproducible Text2SQL baseline using GPT-2 XL on a Text2SQL dataset
in the jkkummerfeld/text2sql-data JSON format.

Adds:
  --rdbms mysql|mariadb|both
  Output file name auto-derived per dataset + rdbms, saved under results/
"""

import argparse
import json
import sys
import time
from pathlib import Path

from sql_utils import fill_gold_sql, normalize_pred_sql, compare_results

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gpt2xl_agent import GPT2XLAgent
from database.db_manager import DatabaseManager


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


def _default_out_path(dataset_name: str, rdbms: str) -> Path:
    # results/gpt2xl_baseline_<dataset>_<rdbms>.jsonl
    return Path("results") / f"gpt2xl_baseline_{dataset_name}_{rdbms}.jsonl"


def _pack_exec_result(res: dict | None):
    if res is None:
        return None
    return {
        "success": res.get("success"),
        "execution_time_s": res.get("execution_time"),
        "rows": res.get("rows_affected"),
        "error": res.get("error"),
    }


def _results_match(res_a: dict | None, res_b: dict | None) -> bool | None:
    """
    Return:
      - True/False if both succeeded and have DataFrame results
      - None if not comparable (e.g., one failed, or no tabular results)
    """
    if not res_a or not res_b:
        return None
    if not res_a.get("success") or not res_b.get("success"):
        return None

    df_a = res_a.get("result")
    df_b = res_b.get("result")
    if df_a is None or df_b is None:
        return None

    return compare_results(df_a, df_b)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets_source/data/advising.json",
        help="Path to dataset JSON file (text2sql-data format).",
    )
    parser.add_argument(
        "--rdbms",
        type=str,
        default="mysql",
        choices=["mysql", "mariadb", "both"],
        help="RDBMS to use for execution.",
    )
    parser.add_argument(
        "--limit_entries",
        type=int,
        default=1,
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
        default=128,
        help="Max tokens to generate for SQL.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSONL path. If omitted, saved under results/ with an RDBMS-specific name.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return 1

    dataset_name = dataset_path.stem  # used as DB name convention

    # Decide output path
    if args.out.strip():
        out_path = Path(args.out)
    else:
        out_path = _default_out_path(dataset_name, args.rdbms)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("🧪 GPT-2 XL Text2SQL Baseline Runner")
    print("=" * 70)
    print(f"Dataset file: {dataset_path}")
    print(f"Dataset name (DB): {dataset_name}")
    print(f"RDBMS: {args.rdbms}")
    print(f"Output: {out_path}")
    print(f"Entry limit: {args.limit_entries}")
    print(f"Schema max tables: {args.max_tables}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("=" * 70)

    data = load_dataset(dataset_path)

    # Initialize model once
    agent = GPT2XLAgent()

    # We will always open a MySQL connection for schema introspection
    # (because schema is shared and you already use mysql.get_compact_schema()).
    # If you want pure-mariadb runs without mysql at all, we can switch introspection too.
    mysql_for_schema = DatabaseManager("mysql")

    mysql_db = None
    maria_db = None

    if args.rdbms in ("mysql", "both"):
        mysql_db = DatabaseManager("mysql")
    if args.rdbms in ("mariadb", "both"):
        maria_db = DatabaseManager("mariadb")

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

                # Compact schema for prompt
                schema_compact = mysql_for_schema.get_compact_schema(
                    database=dataset_name,
                    question=question_text,
                    max_tables=args.max_tables,
                )

                # Gold SQL executable (filled with values)
                gold_sql_exec = fill_gold_sql(entry, sentence)

                # Generate SQL
                t0 = time.time()
                pred_sql_raw = agent.generate_sql(
                    schema=schema_compact,
                    question=question_text,
                    max_new_tokens=args.max_new_tokens,
                )

                # Normalize prediction (table casing etc.)
                schema_tables = mysql_for_schema.get_table_names(database=dataset_name)
                pred_sql = normalize_pred_sql(pred_sql_raw, schema_tables)
                gen_time = time.time() - t0

                # Execute on selected RDBMS
                mysql_pred = mysql_gold = None
                maria_pred = maria_gold = None

                if mysql_db is not None:
                    mysql_db.switch_database(dataset_name)
                    mysql_pred = mysql_db.execute_query(pred_sql)
                    mysql_gold = mysql_db.execute_query(gold_sql_exec)  # NEW

                    if mysql_pred.get("success"):
                        n_ok_mysql += 1

                if maria_db is not None:
                    maria_db.switch_database(dataset_name)
                    maria_pred = maria_db.execute_query(pred_sql)
                    maria_gold = maria_db.execute_query(gold_sql_exec)  # NEW

                    if maria_pred.get("success"):
                        n_ok_maria += 1

                # Cross-RDBMS match only in "both" mode (predicted SQL)
                match = None
                if mysql_pred is not None and maria_pred is not None:
                    if mysql_pred.get("success") and maria_pred.get("success"):
                        n_both_ok += 1
                        if mysql_pred.get("result") is not None and maria_pred.get("result") is not None:
                            match = compare_results(mysql_pred["result"], maria_pred["result"])
                            if match:
                                n_match += 1

                # Execution accuracy: predicted vs gold per-RDBMS
                mysql_exec_match = _results_match(mysql_pred, mysql_gold)  # NEW
                maria_exec_match = _results_match(maria_pred, maria_gold)  # NEW

                record = {
                    "id": row_id,
                    "dataset": dataset_name,

                    # Dataset metadata
                    "query_split": query_split,
                    "question_split": question_split,
                    "question_text": question_text,
                    "question_variables": question_vars,

                    # Gold SQL (raw + executable)
                    "gold_sql_first": gold_sql_first,
                    "gold_sql_variants": sql_variants,
                    "gold_sql_exec": gold_sql_exec,  # executable version

                    # Prompt inputs
                    "schema_compact": schema_compact,

                    # Model output + timings
                    "pred_sql_raw": pred_sql_raw,  # unnormalized
                    "pred_sql": pred_sql,          # normalized used for execution
                    "gen_time_s": round(gen_time, 4),

                    "rdbms_mode": args.rdbms,

                    # Pred exec results
                    "mysql": _pack_exec_result(mysql_pred),
                    "mariadb": _pack_exec_result(maria_pred),

                    # Gold exec results (NEW)
                    "mysql_gold": _pack_exec_result(mysql_gold),
                    "mariadb_gold": _pack_exec_result(maria_gold),

                    # Execution match pred vs gold (NEW)
                    "mysql_pred_vs_gold_match": mysql_exec_match,
                    "mariadb_pred_vs_gold_match": maria_exec_match,

                    # Only meaningful in both-mode (predicted cross-db match)
                    "mysql_vs_mariadb_match": match,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Console line (include exec match if available)
                mysql_status = "-"
                maria_status = "-"
                if mysql_pred is not None:
                    mysql_status = "OK" if mysql_pred.get("success") else "FAIL"
                if maria_pred is not None:
                    maria_status = "OK" if maria_pred.get("success") else "FAIL"

                # NEW: show exec-accuracy status compactly
                mysql_acc = "-" if mysql_exec_match is None else ("✔" if mysql_exec_match else "✘")
                maria_acc = "-" if maria_exec_match is None else ("✔" if maria_exec_match else "✘")

                print(
                    f"[{row_id}] qsplit={query_split or '-'} ssplit={question_split or '-'} "
                    f"mysql={mysql_status}/{mysql_acc} maria={maria_status}/{maria_acc}"
                )

                row_id += 1


    # Close connections
    mysql_for_schema.close()
    if mysql_db is not None:
        mysql_db.close()
    if maria_db is not None:
        maria_db.close()

    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    print(f"Total questions processed: {row_id}")
    if row_id > 0:
        if args.rdbms in ("mysql", "both"):
            print(f"MySQL success rate:   {n_ok_mysql}/{row_id} ({n_ok_mysql/row_id*100:.1f}%)")
        if args.rdbms in ("mariadb", "both"):
            print(f"MariaDB success rate: {n_ok_maria}/{row_id} ({n_ok_maria/row_id*100:.1f}%)")
        if args.rdbms == "both":
            print(f"Both succeeded:       {n_both_ok}/{row_id} ({n_both_ok/row_id*100:.1f}%)")
            if n_both_ok > 0:
                print(f"Result match rate:    {n_match}/{n_both_ok} ({n_match/n_both_ok*100:.1f}%)")
    print(f"\n✅ Wrote results to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
