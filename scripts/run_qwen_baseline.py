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


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset JSON"
    )
    parser.add_argument(
        "--rdbms", type=str, default="mysql", choices=["mysql", "mariadb"]
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

    # 1. Setup Paths & Data
    dataset_path = Path(args.dataset)
    dataset_name = dataset_path.stem
    rdbms = args.rdbms
    llm_name = "qwen"

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("results") / f"qwen_baseline_{dataset_name}_{args.rdbms}.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Qwen Text2SQL Benchmark Runner")
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

    try:
        data = load_dataset(dataset_path)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1

    # 2. Initialize Components
    agent = QwenAgent()
    db_manager = DatabaseManager(rdbms)

    # Get table names for normalization/repair (Crucial for fairness)
    schema_tables = db_manager.get_table_names(dataset_name)
    schema_num_tables = len(schema_tables)
    schema_map = db_manager.get_schema_map(database=dataset_name)

    table_map = build_identifier_maps(db_manager, dataset_name)

    row_id = 0
    questions_processed = 0

    # 3. Processing Loop
    with out_path.open("w", encoding="utf-8") as f:
        for entry in data:
            # Metadata
            query_split = entry.get("query-split", "")
            sql_variants = entry.get("sql", [])
            if not isinstance(sql_variants, list):
                sql_variants = [str(sql_variants)]
            gold_sql_first = sql_variants[0] if sql_variants else ""

            difficulty = entry.get("difficulty", "unknown")
            # Iterate over paraphrases (sentences) - EXACTLY as GPT-2 does
            sentences = entry.get("sentences", [])
            for sentence in sentences:
                if args.limit > 0 and questions_processed >= args.limit:
                    break
                question_text = sentence.get("text", "")
                question_vars = sentence.get("variables", {})
                question_text_filled = fill_question_text(question_text, question_vars)

                question_split = sentence.get("question-split", "")

                # Prepare compact schema
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
                    )
                except Exception as e:
                    print(f"Gen Error: {e}")
                    pred_sql_raw = "SELECT 1;"
                    p_tokens, c_tokens = 0, 0

                gen_time_s = time.time() - t0

                # --- B. NORMALIZATION & REPAIR ---
                # Apply the EXACT same repair logic as GPT-2 to ensure fair scoring
                pred_sql = normalize_pred_sql(pred_sql_raw, schema_tables)
                pred_sql = normalize_table_case(pred_sql, table_map)
                pred_sql, pred_repairs = repair_pred_table_names(pred_sql, schema_tables)
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

                record[f"{args.rdbms}_pred_vs_gold_match"] = bool(match)

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

                row_id += 1
                questions_processed += 1

            if args.limit > 0 and questions_processed >= args.limit:
                break

    db_manager.close()
    
    print("\n" + "=" * 80)
    print("✅ Done")
    print("=" * 80)
    print(f"Questions processed: {questions_processed}")
    print(f"Wrote JSONL: {out_path}")


if __name__ == "__main__":
    main()
