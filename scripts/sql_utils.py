# scripts/sql_utils.py
"""
SQL utilities for Text2SQL evaluation.

- fill_gold_sql: materialize gold SQL with concrete values
- normalize_pred_sql: minor normalization so SQL executes reliably

"""

import re
from typing import Any, Dict, List, Tuple, Optional
from difflib import SequenceMatcher
from database.db_manager import DatabaseManager

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

    Returns: (new_sql, changes)
    """
    if not sql or not schema_map:
        return sql, []

    schema_lut: Dict[str, List[str]] = {t.lower(): cols for t, cols in schema_map.items()}

    if alias_to_table is None:
        alias_to_table = extract_table_alias_map(sql)

    parts = _QUOTED.split(sql)
    changes: List[Dict[str, Any]] = []
    seen_changes: set[tuple] = set()

    def _push_change(d: Dict[str, Any]):
        key = (
            d.get("table_or_alias"),
            d.get("resolved_table"),
            d.get("from"),
            d.get("to"),
            d.get("method"),
        )
        if key not in seen_changes:
            seen_changes.add(key)
            changes.append(d)

    def _exact_col(cols: List[str], target_lower: str) -> str | None:
        for real_c in cols:
            if real_c.lower() == target_lower:
                return real_c
        return None

    def _try_prefix_strip(col_l: str, resolved_table: str, left_l: str) -> str | None:
        """
        High-precision: if col looks like '<table>_<name>' or '<alias>_<name>',
        strip the prefix and return stripped token (lowercased), else None.
        """
        # Candidates: resolved_table_, left_, plus simple plural variant
        prefixes = [resolved_table, left_l]
        if resolved_table.endswith("s"):
            prefixes.append(resolved_table[:-1])
        else:
            prefixes.append(resolved_table + "s")

        for p in prefixes:
            p = (p or "").lower()
            if p and col_l.startswith(p + "_"):
                return col_l[len(p) + 1 :]  # strip "p_"
        return None

    def _id_abbrev_allowed(col_l: str, resolved_table: str, left_l: str) -> bool:
        """
        Prevent over-eager *_id mapping:
          allow only id, <table>_id, <alias>_id.
        """
        if col_l == "id":
            return True
        if col_l == f"{resolved_table}_id":
            return True
        if col_l == f"{left_l}_id":
            return True
        return False

    for i in range(0, len(parts), 2):
        chunk = parts[i]

        def repl(m: re.Match):
            lq1, left, lq2 = m.group(1), m.group(2), m.group(3)
            rq1, col, rq2 = m.group(4), m.group(5), m.group(6)

            left_l = left.lower()
            col_l = col.lower()

            resolved_table = alias_to_table.get(left_l, left_l)
            cols = schema_lut.get(resolved_table or "")
            if not cols:
                return m.group(0)

            # 1) exact match already valid
            real = _exact_col(cols, col_l)
            if real is not None:
                if real != col:
                    _push_change({
                        "table_or_alias": left,
                        "resolved_table": resolved_table,
                        "from": col,
                        "to": real,
                        "ratio": 1.0,
                        "method": "exact",
                    })
                    return f"{lq1}{left}{lq2}.{rq1}{real}{rq2}"
                return m.group(0)

            # 2) prefix-strip repair: actor.actor_name -> actor.name
            stripped = _try_prefix_strip(col_l, resolved_table or "", left_l)
            if stripped:
                real2 = _exact_col(cols, stripped)
                if real2 is not None:
                    _push_change({
                        "table_or_alias": left,
                        "resolved_table": resolved_table,
                        "from": col,
                        "to": real2,
                        "ratio": 1.0,
                        "method": "prefix_strip",
                    })
                    return f"{lq1}{left}{lq2}.{rq1}{real2}{rq2}"

            # 3) safe id abbreviation repair (tighter trigger)
            if allow_id_abbrev and _id_abbrev_allowed(col_l, resolved_table or "", left_l):
                inferred = _infer_id_abbrev(resolved_table or "")
                if inferred:
                    real3 = _exact_col(cols, inferred.lower())
                    if real3 is not None:
                        _push_change({
                            "table_or_alias": left,
                            "resolved_table": resolved_table,
                            "from": col,
                            "to": real3,
                            "ratio": 1.0,
                            "method": "id_abbrev",
                        })
                        return f"{lq1}{left}{lq2}.{rq1}{real3}{rq2}"

            # 4) fuzzy match within table
            best, best_r, second_r = _best_col_match(col, cols, min_ratio=min_ratio)
            if best.lower() != col_l:
                if (best_r - second_r) < min_gap and best_r < 0.99:
                    return m.group(0)

                _push_change({
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


import pandas as pd

def compare_results(result1, result2) -> bool:
    """
    Compare two SQL query results represented as pandas DataFrames.

    - Ignores row order
    - Column names compared case-insensitively (and trimmed)
    - Treats NaN / NULL consistently

    Returns:
        bool: True if results match, False otherwise
    """
    if result1 is None or result2 is None:
        return False

    if not isinstance(result1, pd.DataFrame) or not isinstance(result2, pd.DataFrame):
        return False

    def normalize_col(c: str) -> str:
        return str(c).strip().lower()

    # Build normalized->original maps (and ensure no collisions after normalization)
    n1 = [normalize_col(c) for c in result1.columns]
    n2 = [normalize_col(c) for c in result2.columns]

    if len(set(n1)) != len(n1) or len(set(n2)) != len(n2):
        # e.g., columns ["A", "a"] would collide -> ambiguous
        return False

    if set(n1) != set(n2):
        return False

    # Reorder both dataframes by the same normalized column order
    norm_cols = sorted(set(n1))
    map1 = dict(zip(n1, result1.columns))
    map2 = dict(zip(n2, result2.columns))

    df1 = result1[[map1[c] for c in norm_cols]].copy()
    df2 = result2[[map2[c] for c in norm_cols]].copy()

    # Unify column names so downstream operations align
    df1.columns = norm_cols
    df2.columns = norm_cols

    if df1.shape != df2.shape:
        return False

    try:
        # Normalize NULL/NaN
        sentinel = "__NULL__"
        df1 = df1.fillna(sentinel)
        df2 = df2.fillna(sentinel)

        # Make comparison more stable across dtypes (optional but helpful)
        df1 = df1.astype(str)
        df2 = df2.astype(str)

        # Sort rows (row-order independent)
        df1 = df1.sort_values(by=norm_cols, kind="mergesort").reset_index(drop=True)
        df2 = df2.sort_values(by=norm_cols, kind="mergesort").reset_index(drop=True)

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