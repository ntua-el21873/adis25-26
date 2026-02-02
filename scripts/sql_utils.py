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


# Capture FROM/JOIN table with optional AS and optional alias.
# Examples it matches:
#   FROM movie
#   FROM movie m
#   FROM movie AS m
#   JOIN actor a
#   JOIN actor AS a
_TABLE_ALIAS_POS = re.compile(
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
    (?:\s+(?:as\s+)?([A-Za-z_][\w]*))?  # optional alias (group 4)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Dotted identifier: alias_or_table.column (optional backticks around each piece)
# Examples:
#   actor.name
#   `actor`.`name`
#   a.name
#   a.`birth_year`
_DOTTED_COL = re.compile(
    r"""
    (`?)([A-Za-z_][\w]*)(`?)      # left (alias/table) => group 2
    \s*\.\s*
    (`?)([A-Za-z_][\w]*)(`?)      # right (column)     => group 5
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def extract_table_alias_map(sql: str) -> Dict[str, str]:
    """
    Best-effort alias extraction from FROM/JOIN/etc.
    Returns mapping: alias_or_table_token_lower -> real_table_token_lower

    If no alias is present, maps table->table as well.
    """
    if not sql:
        return {}

    parts = _QUOTED.split(sql)
    alias_to_table: Dict[str, str] = {}

    for i in range(0, len(parts), 2):  # outside quotes
        chunk = parts[i]
        for m in _TABLE_ALIAS_POS.finditer(chunk):
            table = m.group(2)
            alias = m.group(4)

            if table:
                alias_to_table[table.lower()] = table.lower()
            if alias:
                alias_to_table[alias.lower()] = table.lower()

    return alias_to_table

def _best_col_match(
    token: str,
    cols: List[str],
    min_ratio: float = 0.90
) -> Tuple[str, float, float]:
    """
    Returns (best_col, best_ratio, second_best_ratio).
    cols are real column names (case sensitive in DB, but we compare lower).
    """
    t = token.lower()

    # exact match
    for c in cols:
        if c.lower() == t:
            return c, 1.0, 0.0

    # plural stripping (rare for cols but harmless)
    if t.endswith("s"):
        singular = t[:-1]
        for c in cols:
            if c.lower() == singular:
                return c, 0.99, 0.0

    scored = []
    for c in cols:
        r = _ratio(t, c.lower())
        scored.append((r, c))
    scored.sort(reverse=True, key=lambda x: x[0])

    best_r, best = scored[0]
    second_r = scored[1][0] if len(scored) > 1 else 0.0

    if best_r < min_ratio:
        return token, best_r, second_r  # no change
    return best, best_r, second_r


def _infer_id_abbrev(table: str) -> str | None:
    """
    Deterministically infer the common abbreviation key for tables like:
      actor -> aid, movie -> mid, director -> did, writer -> wid, producer -> pid, tv_series -> sid
    Heuristic:
      - if table has underscores: take first letters of parts + 'id' is uncommon here,
        but for your schema it's typically first letter of the main noun + 'id' e.g. tv_series -> sid.
      - otherwise: first letter + 'id'
    """
    if not table:
        return None
    t = table.lower()
    # special-case tv_series (common in your schemas)
    if t == "tv_series":
        return "sid"
    # generic: first char + "id"
    return f"{t[0]}id"

def repair_pred_column_names(
    sql: str,
    schema_map: Dict[str, List[str]],
    alias_to_table: Dict[str, str] | None = None,
    min_ratio: float = 0.90,
    min_gap: float = 0.05,
    allow_id_abbrev: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Repairs predicted SQL column names ONLY in dotted positions (<alias_or_table>.<col>)
    and ONLY outside quotes, using schema_map for table->columns.

    Inputs:
      - sql: predicted SQL string
      - schema_map: {table: [col1, col2, ...]}
      - alias_to_table: mapping alias->real_table (lowercased). If None, we infer from sql.
      - min_ratio/min_gap: fuzzy match thresholds
      - allow_id_abbrev: enable safe mapping for id-like hallucinations (actor_id -> aid, movie_id -> mid)

    Returns: (new_sql, changes)
      changes: list of dicts like:
        {
          "table_or_alias": "a",
          "resolved_table": "actor",
          "from": "actor_id",
          "to": "aid",
          "ratio": 1.0,
          "method": "id_abbrev" | "exact" | "fuzzy"
        }
    """
    if not sql or not schema_map:
        return sql, []

    # Normalize schema_map key access (case-insensitive)
    schema_lut: Dict[str, List[str]] = {t.lower(): cols for t, cols in schema_map.items()}

    if alias_to_table is None:
        alias_to_table = extract_table_alias_map(sql)

    parts = _QUOTED.split(sql)
    changes: List[Dict[str, Any]] = []

    for i in range(0, len(parts), 2):  # outside quotes only
        chunk = parts[i]

        def repl(m: re.Match):
            lq1, left, lq2 = m.group(1), m.group(2), m.group(3)
            rq1, col, rq2 = m.group(4), m.group(5), m.group(6)

            left_l = left.lower()
            col_l = col.lower()

            # Resolve alias->table (fallback: treat left as table)
            resolved_table = alias_to_table.get(left_l, left_l)

            cols = schema_lut.get(resolved_table or "")
            if not cols:
                # Unknown table/alias -> don't touch
                return m.group(0)

            # If already valid, keep (case preserved)
            for real_c in cols:
                if real_c.lower() == col_l:
                    # Optionally normalize case to schema's column casing:
                    if real_c != col:
                        changes.append({
                            "table_or_alias": left,
                            "resolved_table": resolved_table,
                            "from": col,
                            "to": real_c,
                            "ratio": 1.0,
                            "method": "exact",
                        })
                        return f"{lq1}{left}{lq2}.{rq1}{real_c}{rq2}"
                    return m.group(0)

            # Safe id abbreviation repair (high precision for your datasets)
            if allow_id_abbrev:
                # If model says actor_id / movie_id / director_id / ... or plain id
                if col_l == "id" or col_l.endswith("_id"):
                    inferred = _infer_id_abbrev(resolved_table or "")
                    if inferred:
                        # Only apply if inferred abbrev exists in this table
                        for real_c in cols:
                            if real_c.lower() == inferred:
                                changes.append({
                                    "table_or_alias": left,
                                    "resolved_table": resolved_table,
                                    "from": col,
                                    "to": real_c,
                                    "ratio": 1.0,
                                    "method": "id_abbrev",
                                })
                                return f"{lq1}{left}{lq2}.{rq1}{real_c}{rq2}"

            # Fuzzy match within this table's columns
            best, best_r, second_r = _best_col_match(col, cols, min_ratio=min_ratio)

            if best.lower() != col_l:
                # ambiguity guard
                if (best_r - second_r) < min_gap and best_r < 0.99:
                    return m.group(0)

                changes.append({
                    "table_or_alias": left,
                    "resolved_table": resolved_table,
                    "from": col,
                    "to": best,
                    "ratio": round(best_r, 4),
                    "method": "fuzzy",
                })
                return f"{lq1}{left}{lq2}.{rq1}{best}{rq2}"

            return m.group(0)

        chunk = _DOTTED_COL.sub(repl, chunk)
        parts[i] = chunk

    return "".join(parts), changes
# -----------------------
# SQL utilities
# ----------------------

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