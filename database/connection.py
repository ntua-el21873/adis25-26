# database/connection.py
"""
Connection helpers for MySQL and MariaDB using SQLAlchemy.

This module exposes a single public function:

    get_engine(db_type, database=None)

- db_type: 'mysql' or 'mariadb'
- database: optional database name override

It reads credentials from the project's .env file, which is expected
to be located in the project root (one level above this 'database' package).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine


# --- Load environment variables ------------------------------------------------

# Try to load .env from the project root (../.env relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    # Fallback: load from current working directory if .env is there
    load_dotenv()


# --- Internal helpers ----------------------------------------------------------


def _build_mysql_url(database: str | None) -> str:
    """Build SQLAlchemy URL for MySQL."""
    user = os.getenv("MYSQL_USER", "text2sql_user")
    password = os.getenv("MYSQL_PASSWORD", "text2sql_pass")
    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = os.getenv("MYSQL_PORT", "3306")
    db_name = database or os.getenv("MYSQL_DATABASE", "text2sql_db")

    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"


def _build_mariadb_url(database: str | None) -> str:
    """Build SQLAlchemy URL for MariaDB (via MySQL protocol)."""
    user = os.getenv("MARIADB_USER", "text2sql_user")
    password = os.getenv("MARIADB_PASSWORD", "text2sql_pass")
    host = os.getenv("MARIADB_HOST", "127.0.0.1")
    # In your docker-compose, MariaDB is mapped to 3307 on the host
    port = os.getenv("MARIADB_PORT", "3307")
    db_name = database or os.getenv("MARIADB_DATABASE", "text2sql_db")

    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"


# --- Public API ----------------------------------------------------------------


def get_engine(db_type: str, database: str | None = None, echo: bool = False):
    """
    Return a SQLAlchemy engine for the given database type.

    Args:
        db_type: 'mysql' or 'mariadb' (case-insensitive)
        database: Optional database name override. If None, use .env defaults.
        echo: If True, SQLAlchemy will log all SQL statements.

    Returns:
        sqlalchemy.engine.Engine instance.

    Raises:
        ValueError: if db_type is not supported.
    """
    db_type = db_type.lower()

    if db_type == "mysql":
        url = _build_mysql_url(database)
    elif db_type == "mariadb":
        url = _build_mariadb_url(database)
    else:
        raise ValueError(f"Unsupported db_type: {db_type!r}. Use 'mysql' or 'mariadb'.")

    # pool_pre_ping=True helps avoid stale connections
    engine = create_engine(url, pool_pre_ping=True, echo=echo, future=True)
    return engine
