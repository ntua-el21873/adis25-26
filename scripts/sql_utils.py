# scripts/sql_utils.py
"""
SQL utilities for Text2SQL evaluation.

- fill_gold_sql: materialize gold SQL with concrete values
- normalize_pred_sql: minor normalization so SQL executes reliably

"""

import re
from typing import Any, Dict, List, Tuple
from difflib import SequenceMatcher

_QUOTED = re.compile(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")")

# Capture a table identifier right after FROM/JOIN/UPDATE/INTO/DELETE FROM
# Supports optional backticks and optional db.table form.
_TABLE_POS = re.compile(
    r"""
    \b(?:
        from|
        join|
        update|
        into|
        delete\s+from
    )\b
    \s+
    (`?)([A-Za-z_][\w]*)(`?)          # table token (group 2)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Helper to compute similarity ratio
def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _best_table_match(token: str, tables: List[str], min_ratio: float = 0.86) -> Tuple[str, float, float]:
    """
    Returns (best_table, best_ratio, second_best_ratio).
    tables are actual DB table names.
    """
    t = token.lower()

    # Fast path: exact case-insensitive match
    for real in tables:
        if real.lower() == t:
            return real, 1.0, 0.0

    # Fast path: plural stripping
    if t.endswith("s"):
        singular = t[:-1]
        for real in tables:
            if real.lower() == singular:
                return real, 0.99, 0.0

    scored = []
    for real in tables:
        r = _ratio(t, real.lower())
        scored.append((r, real))
    scored.sort(reverse=True, key=lambda x: x[0])

    best_r, best = scored[0]
    second_r = scored[1][0] if len(scored) > 1 else 0.0

    if best_r < min_ratio:
        return token, best_r, second_r  # no change
    return best, best_r, second_r

def repair_pred_table_names(sql: str, actual_tables: List[str], min_ratio: float = 0.86, min_gap: float = 0.03):
    """
    Repairs predicted SQL table names by fuzzy matching to actual DB table names,
    but ONLY in table positions (FROM/JOIN/UPDATE/INTO/DELETE FROM) and ONLY outside quotes.

    Returns: (new_sql, changes)
      changes: list of dicts like {"from": "flights", "to": "flight", "ratio": 0.99}
    """
    if not sql or not actual_tables:
        return sql, []

    parts = _QUOTED.split(sql)
    changes = []

    for i in range(0, len(parts), 2):  # outside quotes only
        chunk = parts[i]

        def repl(m):
            q1, tok, q2 = m.group(1), m.group(2), m.group(3)
            best, best_r, second_r = _best_table_match(tok, actual_tables, min_ratio=min_ratio)

            # Ambiguity guard: best must beat second best by a margin
            if best.lower() != tok.lower():
                if (best_r - second_r) < min_gap and best_r < 0.99:
                    return m.group(0)  # too ambiguous, skip
                changes.append({"from": tok, "to": best, "ratio": round(best_r, 4)})

            return m.group(0).replace(tok, best)

        chunk = _TABLE_POS.sub(repl, chunk)
        parts[i] = chunk

    return "".join(parts), changes



def fill_gold_sql(entry: dict, sentence: dict) -> str:
    """
    Fill variable placeholders in the gold SQL using dataset-provided values.

    Policy (as requested):
    - Pure substitution: replace placeholder tokens with values as-is.
    - Do NOT add quotes, do NOT escape, do NOT type-cast.
    - If the dataset SQL already has quotes around the placeholder, they remain.
      Example: AIRPORT_CODE = "airport_code0" -> AIRPORT_CODE = "MKE"

    Strategy:
    - Use FIRST SQL variant
    - Prefer sentence["variables"] (question vars)
    - Fall back to entry["variables"][i]["example"] (sql-only vars)
    """

    sql_list = entry.get("sql", [])
    if not sql_list:
        return ""

    sql = str(sql_list[0])

    # Collect replacements
    replacements: Dict[str, Any] = {}

    sent_vars = sentence.get("variables", {})
    if isinstance(sent_vars, dict):
        replacements.update(sent_vars)

    for v in entry.get("variables", []):
        if not isinstance(v, dict):
            continue
        name = v.get("name")
        example = v.get("example")
        if name and name not in replacements:
            replacements[name] = example

    if not replacements:
        return sql

    # Replace longer names first to avoid var1 matching inside var10
    for name in sorted(replacements.keys(), key=len, reverse=True):
        value = replacements[name]
        if value is None:
            continue

        # Replace whole identifier token occurrences only.
        # This matches placeholders surrounded by punctuation/quotes/spaces safely.
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(str(name))}(?![A-Za-z0-9_])")
        sql = pattern.sub(str(value), sql)

    return sql



def normalize_pred_sql(pred_sql: str, schema_tables: list[str]) -> str:
    """
    Normalize predicted SQL so it matches DB schema conventions.

    Currently:
    - Fix table-name casing (MySQL/MariaDB table names are case-sensitive on Linux)
    """

    if not pred_sql:
        return pred_sql

    normalized = pred_sql

    for table in schema_tables:
        # replace whole-word table references case-insensitively
        pattern = re.compile(rf"\b{table}\b", re.IGNORECASE)
        normalized = pattern.sub(table, normalized)

    return normalized.strip()


def compare_results(result1, result2) -> bool:
    """
    Compare two SQL query results represented as pandas DataFrames.

    - Ignores row order
    - Requires same columns
    - Treats NaN / NULL consistently
    
    Returns:
        bool: True if results match, False otherwise
    """

    if result1 is None or result2 is None:
        return False

    if set(result1.columns) != set(result2.columns):
        return False

    # Reorder columns consistently
    cols = sorted(result1.columns.tolist())
    df1 = result1[cols].copy()
    df2 = result2[cols].copy()

    if df1.shape != df2.shape:
        return False

    try:
        # Normalize NaN / None
        df1 = df1.fillna("__NULL__")
        df2 = df2.fillna("__NULL__")

        # Sort rows
        df1 = df1.sort_values(by=cols).reset_index(drop=True)
        df2 = df2.sort_values(by=cols).reset_index(drop=True)

        return df1.equals(df2)
    except Exception:
        return False

def compare_db_results(mysql_result, mariadb_result):
    """
    Compare results from MySQL and MariaDB
    
    Returns:
        dict: Comparison results
    """
    both_success = (
        mysql_result['success'] and 
        mariadb_result['success']
    )
    
    if not both_success:
        return {
            'match': False,
            'reason': 'One or both queries failed',
            'mysql_success': mysql_result['success'],
            'mariadb_success': mariadb_result['success']
        }
    
    results_match = compare_results(
        mysql_result['result'],
        mariadb_result['result']
    )
    
    return {
        'match': results_match,
        'mysql_rows': mysql_result['rows_affected'],
        'mariadb_rows': mariadb_result['rows_affected'],
        'mysql_time': mysql_result['execution_time'],
        'mariadb_time': mariadb_result['execution_time'],
        'time_difference': abs(
            mysql_result['execution_time'] - 
            mariadb_result['execution_time']
        )
    }