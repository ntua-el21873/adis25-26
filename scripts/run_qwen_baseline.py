"""
scripts/run_qwen_baseline.py

Run a full reproducible Text2SQL baseline using the local Qwen Agent.
Functionally equivalent to run_gpt2xl_baseline.py for fair comparison.

Features:
- Executes BOTH Predicted SQL and Gold SQL
- Calculates Accuracy (Pred Result == Gold Result)
- Handles Dataset Splits (Easy/Medium/Hard)

Adds:
  --rdbms mysql|mariadb|both
  Output file name auto-derived per dataset + rdbms, saved under results/
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.qwen_agent import QwenAgent
from database.db_manager import DatabaseManager
from scripts.sql_utils import fill_gold_sql, normalize_pred_sql, compare_results

# -----------------------------------------------------------------------------
# Helper Functions 
# -----------------------------------------------------------------------------
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
    # results/qwen_baseline_<dataset>_<rdbms>.jsonl
    return Path("results") / f"qwen_baseline_{dataset_name}_{rdbms}.jsonl"

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
      - True/False if both succeeded and have DataFrame results (Accuracy Check)
      - None if not comparable
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

# -----------------------------------------------------------------------------
# Main Execution Loop
# -----------------------------------------------------------------------------
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
        default=5,
        help="Process only the first N entries.",
    )
    parser.add_argument(
        "--max_tables",
        type=int,
        default=12,
        help="Max tables to include in compact schema.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSONL path.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return 1

    dataset_name = dataset_path.stem

    # Decide output path
    if args.out.strip():
        out_path = Path(args.out)
    else:
        out_path = _default_out_path(dataset_name, args.rdbms)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("🧪 Qwen (Local) Text2SQL Baseline Runner")
    print("=" * 70)
    print(f"Dataset file: {dataset_path}")
    print(f"Dataset name (DB): {dataset_name}")
    print(f"RDBMS: {args.rdbms}")
    print(f"Output: {out_path}")
    print(f"Entry limit: {args.limit_entries}")
    print("=" * 70)

    data = load_dataset(dataset_path)

    # Initialize Agent
    agent = QwenAgent()

    # Initialize Databases
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
    n_match = 0 # Cross-DB match count

    with out_path.open("w", encoding="utf-8") as f:
        for entry in data[: args.limit_entries]:
            query_split = get_query_split(entry)
            sql_variants = get_sql_variants(entry)
            gold_sql_first = sql_variants[0] if sql_variants else ""

            for sentence in iter_sentences(entry):
                question_text = get_sentence_text(sentence)
                question_split = get_question_split(sentence)
                question_vars = get_sentence_variables(sentence)

                # A. Get Schema
                schema_compact = mysql_for_schema.get_compact_schema(
                    database=dataset_name,
                    question=question_text,
                    max_tables=args.max_tables,
                )

                # B. Prepare Gold SQL (The Correct Answer)
                gold_sql_exec = fill_gold_sql(entry, sentence)

                # C. Generate Prediction
                # Note: QwenAgent handles max_new_tokens internally (hardcoded to 256)
                t0 = time.time()
                pred_sql_raw = agent.generate_sql(schema_compact, question_text)
                gen_time = time.time() - t0

                # Normalize 
                schema_tables = mysql_for_schema.get_table_names(database=dataset_name)
                pred_sql = normalize_pred_sql(pred_sql_raw, schema_tables)

                # D. Execute Pred vs Gold
                mysql_pred = mysql_gold = None
                maria_pred = maria_gold = None

                # --- MySQL Execution ---
                if mysql_db:
                    mysql_db.switch_database(dataset_name)
                    mysql_pred = mysql_db.execute_query(pred_sql)
                    mysql_gold = mysql_db.execute_query(gold_sql_exec)

                    if mysql_pred.get("success"):
                        n_ok_mysql += 1

                # --- MariaDB Execution ---
                if maria_db:
                    maria_db.switch_database(dataset_name)
                    maria_pred = maria_db.execute_query(pred_sql)
                    maria_gold = maria_db.execute_query(gold_sql_exec)

                    if maria_pred.get("success"):
                        n_ok_maria += 1

                # E. Compare Results (Pred vs Gold) - Accuracy
                mysql_exec_match = _results_match(mysql_pred, mysql_gold)
                maria_exec_match = _results_match(maria_pred, maria_gold)

                # F. Cross-RDBMS Comparison (Only if running BOTH)
                match = None
                if mysql_pred and maria_pred:
                    if mysql_pred.get("success") and maria_pred.get("success"):
                        n_both_ok += 1
                        if mysql_pred.get("result") is not None and maria_pred.get("result") is not None:
                            match = compare_results(mysql_pred["result"], maria_pred["result"])
                            if match:
                                n_match += 1

                # Logging
                # Status: OK/FAIL (Execution)
                # Accuracy: ✔/✘ (Correctness vs Gold)
                ms = "-"
                ma = "-"
                if mysql_pred:
                    ms = "OK" if mysql_pred.get("success") else "FAIL"
                    ma = "✔" if mysql_exec_match else ("✘" if mysql_exec_match is not None else "-")

                marias = "-"
                mariaa = "-"
                if maria_pred:
                    marias = "OK" if maria_pred.get("success") else "FAIL"
                    mariaa = "✔" if maria_exec_match else ("✘" if maria_exec_match is not None else "-")
                
                print(f"[{row_id}] qsplit={query_split or '-'} mysql={ms}/{ma} maria={marias}/{mariaa}")

                # G. Save Record
                record = {
                    "id": row_id,
                    "dataset": dataset_name,
                    "query_split": query_split,
                    "question_split": question_split,
                    "question_text": question_text,
                    "question_variables": question_vars,
                    
                    # SQLs
                    "gold_sql_first": gold_sql_first,
                    "gold_sql_exec": gold_sql_exec,
                    "pred_sql_raw": pred_sql_raw,
                    "pred_sql": pred_sql,
                    
                    "schema_compact": schema_compact,
                    "rdbms_mode": args.rdbms,
                    "gen_time_s": round(gen_time, 4),
                    
                    # Execution Results
                    "mysql": _pack_exec_result(mysql_pred),
                    "mariadb": _pack_exec_result(maria_pred),
                    "mysql_gold": _pack_exec_result(mysql_gold),
                    "mariadb_gold": _pack_exec_result(maria_gold),
                    
                    # Accuracy Booleans
                    "mysql_pred_vs_gold_match": mysql_exec_match,
                    "mariadb_pred_vs_gold_match": maria_exec_match,
                    
                    # Cross-DB Match
                    "mysql_vs_mariadb_match": match
                }
                
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                row_id += 1

    # Cleanup
    mysql_for_schema.close()
    if mysql_db: mysql_db.close()
    if maria_db: maria_db.close()

    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    print(f"Total Questions: {row_id}")
    
    if args.rdbms in ("mysql", "both") and row_id > 0:
        print(f"MySQL success rate:   {n_ok_mysql}/{row_id} ({n_ok_mysql/row_id*100:.1f}%)")
        
    if args.rdbms in ("mariadb", "both") and row_id > 0:
        print(f"MariaDB success rate: {n_ok_maria}/{row_id} ({n_ok_maria/row_id*100:.1f}%)")
        
    if args.rdbms == "both" and row_id > 0:
        print(f"Both succeeded:       {n_both_ok}/{row_id} ({n_both_ok/row_id*100:.1f}%)")
        if n_both_ok > 0:
            print(f"Result match rate:    {n_match}/{n_both_ok} ({n_match/n_both_ok*100:.1f}%)")
            
    print(f"\n✅ Wrote results to: {out_path}")
    return 0

if __name__ == "__main__":
    main()