import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import time

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gpt2xl_agent import GPT2XLAgent
from database.db_manager import DatabaseManager


def load_dataset(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Dataset JSON must be a list, got: {type(data)}")
    return data


def iter_sentences(entry: dict):
    sentences = entry.get("sentences", [])
    if not isinstance(sentences, list):
        return
    for s in sentences:
        if isinstance(s, dict):
            yield s


def fill_question_text(question_text: str, variables: Dict[str, Any]) -> str:
    """
    Replace bare variable tokens (e.g., airport_code0) with their values.
    No quoting, no escaping.
    """
    import re

    filled = question_text
    if not variables:
        return filled

    for k in sorted(variables.keys(), key=len, reverse=True):
        v = variables[k]
        if v is None:
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(str(k))}(?![A-Za-z0-9_])")
        filled = pattern.sub(str(v), filled)

    return filled


def main():
    parser = argparse.ArgumentParser(description="Single-run GPT-2 XL prediction tester on a Text2SQL dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSON file (text2sql-data format).",
    )
    parser.add_argument(
        "--rdbms",
        type=str,
        default="mysql",
        choices=["mysql", "mariadb"],
        help="Which RDBMS to use for schema introspection (default: mysql).",
    )
    parser.add_argument(
        "--max_tables",
        type=int,
        default=10,
        help="Max tables to include in compact schema (passed to get_compact_schema).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Max tokens to generate for SQL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="How many QUESTIONS (sentences) to predict (default: 1).",
    )
    parser.add_argument(
        "--entry_idx",
        type=int,
        default=100,
        help="Start from this entry index in the dataset list (default: 0).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset_name = dataset_path.stem  # DB name convention
    data = load_dataset(dataset_path)

    agent = GPT2XLAgent()

    # Use DB manager only for compact schema construction (same as benchmark)
    db = DatabaseManager(args.rdbms)

    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Dataset file: {dataset_path}")
    print(f"Schema RDBMS: {args.rdbms}")
    print(f"Max tables: {args.max_tables}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Limit (questions): {args.limit}")
    print(f"Starting entry index: {args.entry_idx}")
    print("=" * 80)

    n_done = 0

    for entry in data[args.entry_idx :]:
        for sentence in iter_sentences(entry):
            if n_done >= args.limit:
                break

            question_text = str(sentence.get("text", ""))
            variables = sentence.get("variables", {})
            variables = variables if isinstance(variables, dict) else {}

            question_filled = fill_question_text(question_text, variables)

            schema_compact = db.get_compact_schema(
                database=dataset_name,
                question=question_filled,
                max_tables=args.max_tables,
            )
            print("\n" + "-" * 80)
            print(f"Question #{n_done}")
            print(f"question_text:        {question_text}")
            print(f"question_text_filled: {question_filled}")
            print(f"variables:            {variables}")
            print("\nSchema (compact):")
            print(schema_compact)
            print("\nPredicted SQL:")

            t0 = time.time()
            pred_sql = agent.generate_sql(
                schema=schema_compact,
                question=question_filled,
                max_new_tokens=args.max_new_tokens,
            )
            t1 = time.time()

            print(pred_sql)
            print(f"\n(Time taken: {t1 - t0:.2f} seconds)")
            print("-" * 80)

            n_done += 1

        if n_done >= args.limit:
            break

    db.close()
    print(f"\nDone. Produced {n_done} predictions.")


if __name__ == "__main__":
    main()
