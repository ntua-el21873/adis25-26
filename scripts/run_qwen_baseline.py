"""
scripts/run_qwen_baseline.py

Research-grade Qwen Text2SQL runner.
Functionally IDENTICAL to run_gpt2xl_baseline.py for fair comparison.

Differences from GPT-2 script:
- Removed prompt truncation logic (Qwen context window is large enough).
- Uses QwenAgent instead of GPT2XLAgent.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.qwen_agent import QwenAgent
from database.db_manager import DatabaseManager
# We import the exact same utils as the GPT-2 baseline
from scripts.sql_utils import (
    fill_gold_sql, 
    compare_results, 
    repair_pred_table_names
)

# -----------------------------------------------------------------------------
# Helper Functions (Mirrored from run_gpt2xl_baseline.py)
# -----------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data

def load_schema_from_file(dataset_name: str, rdbms: str) -> str:
    """
    Reads the pre-processed CREATE TABLE statements.
    """
    root = Path(__file__).resolve().parent.parent
    schema_path = root / "data" / "processed" / "schemas" / rdbms / f"{dataset_name}.schema.sql"
    
    if not schema_path.exists():
        # Fallback to mysql if mariadb specific file missing
        fallback = root / "data" / "processed" / "schemas" / "mysql" / f"{dataset_name}.schema.sql"
        if fallback.exists():
            return fallback.read_text(encoding="utf-8")
        print(f"⚠️ Schema not found: {schema_path}")
        return ""
    
    return schema_path.read_text(encoding="utf-8")

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
            f"{prefix}_rows": 0
        }
    return {
        f"{prefix}_success": bool(res.get("success", False)),
        f"{prefix}_error_msg": str(res.get("error", "") or "") if res.get("error") else None,
        f"{prefix}_time_s": res.get("execution_time", 0.0),
        f"{prefix}_rows": res.get("rows_affected", 0)
    }

def fill_question_text(text: str, variables: dict) -> str:
    """Substitute variables into the question text (e.g. number0 -> 100)."""
    out = text
    for k, v in variables.items():
        out = out.replace(k, str(v))
    return out

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON")
    parser.add_argument("--rdbms", type=str, default="mysql", choices=["mysql", "mariadb"])
    parser.add_argument("--limit", type=int, default=0, help="Max entries to process (0=all)")
    parser.add_argument("--out", type=str, default="", help="Custom output path")
    args = parser.parse_args()

    # 1. Setup Paths & Data
    dataset_path = Path(args.dataset)
    dataset_name = dataset_path.stem
    
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("results") / f"qwen_baseline_{dataset_name}_{args.rdbms}.jsonl"
    
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Qwen Baseline | DB: {dataset_name} | RDBMS: {args.rdbms}")
    print(f"   Output: {out_path}")

    try:
        data = load_dataset(dataset_path)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1

    # 2. Initialize Components
    agent = QwenAgent() 
    db_manager = DatabaseManager(args.rdbms)
    
    # Load Schema Text
    schema_text = load_schema_from_file(dataset_name, args.rdbms)
    
    # Get table names for normalization/repair (Crucial for fairness)
    real_tables = db_manager.get_table_names(dataset_name)
    schema_num_tables = len(real_tables)

    row_id = 0
    questions_processed = 0

    # 3. Processing Loop
    with out_path.open("w", encoding="utf-8") as f:
        for entry in data:
            # Metadata
            query_split = entry.get("query-split", "")
            difficulty = entry.get("difficulty", "unknown")
            sql_variants = entry.get("sql", [])
            if not isinstance(sql_variants, list):
                sql_variants = [str(sql_variants)]
            gold_sql_first = sql_variants[0] if sql_variants else ""

            # Iterate over paraphrases (sentences) - EXACTLY as GPT-2 does
            sentences = entry.get("sentences", [])
            for sentence in sentences:
                question_text = sentence.get("text", "")
                variables = sentence.get("variables", {})
                question_split = sentence.get("question-split", "")
                
                # Filled question
                question_text_filled = fill_question_text(question_text, variables)

                # --- A. GENERATION ---
                print(f"[{row_id}] Generating...", end=" ", flush=True)
                t0 = time.time()
                
                try:
                    # Expecting tuple (sql, prompt_tokens, completion_tokens)
                    pred_sql_raw, p_tokens, c_tokens = agent.generate_sql(schema_text, question_text)
                except Exception as e:
                    print(f"Gen Error: {e}")
                    pred_sql_raw = "SELECT 1;"
                    p_tokens, c_tokens = 0, 0
                
                gen_time_s = time.time() - t0

                # --- B. NORMALIZATION & REPAIR ---
                # Apply the EXACT same repair logic as GPT-2 to ensure fair scoring
                pred_sql_fixed, pred_repairs = repair_pred_table_names(pred_sql_raw, real_tables)

                # Prepare Gold SQL for execution
                gold_sql_exec = fill_gold_sql(entry, sentence)

                # --- C. EXECUTION ---
                db_manager.switch_database(dataset_name)
                
                # Execute Prediction
                pred_res = db_manager.execute_query(pred_sql_fixed)
                
                # Execute Gold
                gold_res = db_manager.execute_query(gold_sql_exec)

                # --- D. COMPARISON ---
                match = compare_results(pred_res.get("result"), gold_res.get("result"))

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
                    "question_variables": variables,
                    "query_split": query_split,
                    "question_split": question_split,
                    "difficulty": difficulty,
                    
                    # Gold Info
                    "gold_sql_first": gold_sql_first,
                    "gold_sql_exec": gold_sql_exec,
                    "gold_sql_variants": sql_variants,
                    
                    # Schema / Prompt Info
                    "schema_compact": schema_text[:200] + "...", 
                    "schema_num_tables": schema_num_tables,
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    
                    # Prediction Info
                    "pred_sql_raw": pred_sql_raw,
                    "pred_sql": pred_sql_fixed, 
                    "pred_repairs": pred_repairs, # ADDED: Missing in previous version
                    "gen_time_s": round(gen_time_s, 6),
                    
                    # Comparison Result
                    f"{args.rdbms}_pred_vs_gold_match": bool(match)
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
                    f"tables={schema_num_tables} prompt_tokens={p_tokens}"
                )

                row_id += 1
                questions_processed += 1
            
            if args.limit > 0 and questions_processed >= args.limit:
                break

    db_manager.close()
    print("\n" + "="*60)
    print(f"Done. Processed {questions_processed} queries.")
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()