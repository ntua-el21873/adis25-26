"""
scripts/extract_schemas.py

DB Bootstrapper for Text2SQL experiments:
- Resolve SQL assets per dataset (download or local file)
- Ensure database exists (optional reset)
- Import SQL into MySQL/MariaDB (SKIP if already populated, unless forced)
- Dump schema-only snapshots (DDL) via mysqldump/mariadb-dump --no-data
- Dump data-only snapshots (DML) via mysqldump/mariadb-dump --no-create-info

Usage examples:
  # Import (skip if already imported) + dump DDL+DML
  python scripts/extract_schemas.py

  # Only MySQL
  python scripts/extract_schemas.py --only mysql

  # Reset + reimport
  python scripts/extract_schemas.py --reset-db

  # Force reimport even if already populated
  python scripts/extract_schemas.py --force-import

  # Skip dumps
  python scripts/extract_schemas.py --no-schema-dump --no-data-dump
"""

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

# -----------------------------
# Config: container names match docker-compose.yml
# -----------------------------
CONTAINERS = {
    "mysql": "text2sql-mysql",
    "mariadb": "text2sql-mariadb",
}

# -----------------------------
# Where to cache downloaded SQL assets
# -----------------------------
CACHE_DIR = Path("data/source_sql")
DDL_OUT_DIR = Path("data/processed/DDL")
DML_OUT_DIR = Path("data/processed/DML")

# -----------------------------
# SQL dump sources
# -----------------------------
DATASET_SQL_SOURCES: Dict[str, Dict] = {
    "advising": {
        "type": "direct_sql",
        "url": "https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/refs/heads/master/data/advising-db.sql",
        "out_name": "advising.sql",
    },
    "atis": {
        "type": "direct_sql",
        "url": "https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/refs/heads/master/data/atis-db.sql",
        "out_name": "atis.sql",
    },
    "imdb": {
        "type": "local_sql",
        "path": "data/source_sql/IMDB.database.sql",
    },
    "yelp": {
        "type": "local_sql",
        "path": "data/source_sql/YELP.database.sql",
    },
}

DEFAULT_DATASETS = ["advising", "atis", "imdb", "yelp"]

# 
EXPECTED_TABLES = {
    "advising": 15,
    "atis": 25,
    "imdb": 16,
    "yelp": 7,
}


@dataclass
class DbCreds:
    root_password: str


def run(cmd: List[str], *, input_path: Optional[Path] = None) -> None:
    """Run a subprocess command, optionally piping a file to stdin."""
    if input_path is None:
        subprocess.run(cmd, check=True)
    else:
        with input_path.open("rb") as f:
            subprocess.run(cmd, check=True, stdin=f)


def download_file(url: str, out_path: Path, force: bool = False) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"📦 Using cached: {out_path}")
        return out_path

    print(f"⬇️  Downloading: {url}")
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()

    with out_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"✅ Saved: {out_path}")
    return out_path


def resolve_sql_asset(dataset: str, force_download: bool) -> Optional[Path]:
    """
    Return local path to the dataset SQL file, downloading if needed.
    Returns None if no source is configured.
    """
    if dataset not in DATASET_SQL_SOURCES:
        print(f"⚠️  No SQL source configured for dataset '{dataset}'. Skipping import.")
        return None

    src = DATASET_SQL_SOURCES[dataset]
    typ = src.get("type")

    if typ == "direct_sql":
        url = src["url"]
        out_name = src.get("out_name", f"{dataset}.sql")
        return download_file(url, CACHE_DIR / out_name, force=force_download)

    if typ == "local_sql":
        p = Path(src["path"])
        if not p.exists():
            raise FileNotFoundError(f"Local SQL not found: {p}")
        print(f"📦 Using local SQL: {p}")
        return p

    raise ValueError(f"Unknown SQL source type: {typ}")


def _client_for(db_type: str) -> str:
    # Both images typically provide `mysql`; mariadb image also provides `mariadb`.
    # We'll use mariadb client for mariadb target to be explicit.
    return "mysql" if db_type == "mysql" else "mariadb"


def _dump_for(db_type: str) -> str:
    return "mysqldump" if db_type == "mysql" else "mariadb-dump"


def docker_mysql_exec(db_type: str, creds: DbCreds, sql: str) -> None:
    """Execute a one-liner SQL command inside the container."""
    container = CONTAINERS[db_type]
    client = _client_for(db_type)
    cmd = [
        "docker", "exec", "-i", container,
        client,
        "-uroot",
        f"-p{creds.root_password}",
        "-e", sql,
    ]
    run(cmd)


def docker_mysql_query_scalar(db_type: str, creds: DbCreds, sql: str) -> Optional[int]:
    """
    Execute a SQL statement inside the container and return the first cell as int.
    Uses -N -s for machine-friendly output.
    """
    container = CONTAINERS[db_type]
    client = _client_for(db_type)
    cmd = [
        "docker", "exec", "-i", container,
        client,
        "-uroot",
        f"-p{creds.root_password}",
        "-N", "-s",
        "-e", sql,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        s = out.decode("utf-8", errors="replace").strip()
        if not s:
            return None
        return int(s.splitlines()[0].strip())
    except Exception:
        return None


def db_exists(db_type: str, creds: DbCreds, db_name: str) -> bool:
    q = f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{db_name}';"
    return (docker_mysql_query_scalar(db_type, creds, q) or 0) > 0


def table_count(db_type: str, creds: DbCreds, db_name: str) -> int:
    q = (
        "SELECT COUNT(*) "
        "FROM INFORMATION_SCHEMA.TABLES "
        f"WHERE TABLE_SCHEMA = '{db_name}' AND TABLE_TYPE='BASE TABLE';"
    )
    return int(docker_mysql_query_scalar(db_type, creds, q) or 0)


def is_already_imported(db_type: str, creds: DbCreds, db_name: str, min_tables: int = 1) -> bool:
    """
    Heuristic: consider imported if DB exists and has >= min_tables base tables.
    """
    if not db_exists(db_type, creds, db_name):
        return False
    return table_count(db_type, creds, db_name) >= min_tables


def docker_mysql_import_file(db_type: str, creds: DbCreds, dataset_db: str, sql_file: Path) -> None:
    """
    Import a .sql file into a specific database using docker exec + mysql stdin.
    We call: mysql -uroot -p... <db> < file.sql
    """
    container = CONTAINERS[db_type]
    client = _client_for(db_type)
    cmd = [
        "docker", "exec", "-i", container,
        client,
        "-uroot",
        f"-p{creds.root_password}",
        dataset_db,
    ]
    run(cmd, input_path=sql_file)


def ensure_db(db_type: str, creds: DbCreds, db_name: str, reset: bool) -> None:
    if reset:
        print(f"🧹 Dropping database '{db_name}' on {db_type} (if exists)")
        docker_mysql_exec(db_type, creds, f"DROP DATABASE IF EXISTS `{db_name}`;")
    print(f"🛠️  Creating database '{db_name}' on {db_type} (if not exists)")
    docker_mysql_exec(
        db_type,
        creds,
        f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;",
    )


def extract_schema_snapshot(db_type: str, creds: DbCreds, db_name: str) -> None:
    """Dump schema-only (DDL) using mysqldump/mariadb-dump --no-data."""
    container = CONTAINERS[db_type]
    out_dir = DDL_OUT_DIR / db_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{db_name}.schema.sql"

    dump = _dump_for(db_type)
    cmd = [
        "docker", "exec", "-i", container,
        dump,
        "-uroot",
        f"-p{creds.root_password}",
        "--no-data",
        "--routines",
        "--triggers",
        "--skip-comments",
        db_name,
    ]

    print(f"🧾 Writing schema snapshot: {out_path}")
    with out_path.open("wb") as f:
        subprocess.run(cmd, check=True, stdout=f)


def extract_data_snapshot(db_type: str, creds: DbCreds, db_name: str) -> None:
    """Dump data-only (DML inserts) using mysqldump/mariadb-dump --no-create-info."""
    container = CONTAINERS[db_type]
    out_dir = DML_OUT_DIR / db_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{db_name}.data.sql"

    dump = _dump_for(db_type)
    cmd = [
        "docker", "exec", "-i", container,
        dump,
        "-uroot",
        f"-p{creds.root_password}",
        "--no-create-info",
        "--skip-triggers",
        "--skip-add-drop-table",
        "--single-transaction",
        "--quick",
        "--default-character-set=utf8mb4",
        "--skip-comments",
        db_name,
    ]

    print(f"🧾 Writing data snapshot: {out_path}")
    with out_path.open("wb") as f:
        subprocess.run(cmd, check=True, stdout=f)


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Download SQL dumps (if available), import into MySQL/MariaDB, and extract DDL/DML snapshots."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset DBs to import (default: advising atis imdb yelp)",
    )
    parser.add_argument(
        "--only",
        choices=["mysql", "mariadb"],
        default=None,
        help="Only run for one RDBMS",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Drop dataset DBs before importing (forces reimport)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload SQL assets even if cached",
    )
    parser.add_argument(
        "--force-import",
        action="store_true",
        help="Import SQL even if DB already looks populated (tables exist)",
    )
    parser.add_argument(
        "--no-schema-dump",
        action="store_true",
        help="Skip schema snapshot extraction (DDL)",
    )
    parser.add_argument(
        "--no-data-dump",
        action="store_true",
        help="Skip data-only snapshot extraction (DML)",
    )
    args = parser.parse_args()

    mysql_creds = DbCreds(root_password=os.getenv("MYSQL_ROOT_PASSWORD", "root123"))
    mariadb_creds = DbCreds(root_password=os.getenv("MARIADB_ROOT_PASSWORD", "root123"))

    targets = ["mysql", "mariadb"] if args.only is None else [args.only]

    print("🚀 DB Bootstrapper (SQL import + DDL/DML snapshots)")
    print(f"Targets: {targets}")
    print(f"Datasets: {args.datasets}")
    print(f"Reset DBs: {args.reset_db}")
    print(f"Force download: {args.force_download}")
    print(f"Force import: {args.force_import}")
    print("")

    for db_type in targets:
        creds = mysql_creds if db_type == "mysql" else mariadb_creds
        print("=" * 70)
        print(f"🔧 RDBMS: {db_type.upper()}")
        print("=" * 70)

        for dataset in args.datasets:
            print(f"\n--- Dataset: {dataset} ---")
            sql_path = resolve_sql_asset(dataset, force_download=args.force_download)
            if sql_path is None:
                continue

            ensure_db(db_type, creds, dataset, reset=args.reset_db)

            # Decide whether to import
            min_tables = EXPECTED_TABLES.get(dataset, 1)

            already = False
            if not args.reset_db:
                already = is_already_imported(db_type, creds, dataset, min_tables=min_tables)

            if already and not args.force_import:
                tc = table_count(db_type, creds, dataset)
                print(f"⏭️  Skipping import: {db_type}:{dataset} already populated (tables={tc} >= {min_tables}).")
            else:
                print(f"📥 Importing {sql_path.name} into {db_type}:{dataset} ...")
                try:
                    docker_mysql_import_file(db_type, creds, dataset, sql_path)
                    print(f"✅ Import complete: {db_type}:{dataset}")
                except subprocess.CalledProcessError:
                    print(f"❌ Import failed for {db_type}:{dataset}.")
                    print("Tip: check container logs and SQL syntax compatibility.")
                    return 1

            if not args.no_schema_dump:
                extract_schema_snapshot(db_type, creds, dataset)
            if not args.no_data_dump:
                extract_data_snapshot(db_type, creds, dataset)

    print("\n✅ Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
