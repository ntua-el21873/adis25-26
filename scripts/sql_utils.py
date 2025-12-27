# scripts/sql_utils.py
"""
SQL utilities for Text2SQL evaluation.

- fill_gold_sql: materialize gold SQL with concrete values
- normalize_pred_sql: minor normalization so SQL executes reliably
"""

import re


def fill_gold_sql(entry: dict, sentence: dict) -> str:
    """
    Fill variable placeholders in the gold SQL using dataset-provided values.

    Strategy:
    - Use the FIRST SQL variant
    - Prefer sentence["variables"] for question vars
    - Fall back to entry["variables"][i]["example"] for sql-only vars
    """

    sql_list = entry.get("sql", [])
    if not sql_list:
        return ""

    sql = str(sql_list[0])

    # Collect replacements
    replacements = {}

    # Variables appearing in the question
    sent_vars = sentence.get("variables", {})
    if isinstance(sent_vars, dict):
        replacements.update(sent_vars)

    # Variables defined only in SQL
    for v in entry.get("variables", []):
        name = v.get("name")
        example = v.get("example")
        if name and name not in replacements:
            replacements[name] = example

    # Replace placeholders
    for name, value in replacements.items():
        if value is None:
            continue

        # Numeric?
        if isinstance(value, (int, float)) or str(value).isdigit():
            sql = re.sub(rf"\b{name}\b", str(value), sql)
        else:
            # String → quote
            escaped = str(value).replace('"', '\\"')
            sql = re.sub(rf"\b{name}\b", f'"{escaped}"', sql)

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