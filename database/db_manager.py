"""
database/db_manager.py
Enhanced database manager with support for text2sql datasets
"""

from typing import Dict, List, Set
import pandas as pd
import time
import json
from pyparsing import Dict
from pathlib import Path
from sqlalchemy import text, inspect
from database.connection import get_engine
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple


class DatabaseManager:
    """Enhanced database manager for text2sql experiments"""

    # Dataset database mapping
    DATASET_DATABASES = {
        "advising": "advising",
        "atis": "atis",
        "imdb": "imdb",
        "yelp": "yelp",
    }

    def __init__(self, db_type, database=None):
        """
        Initialize database manager

        Args:
            db_type: 'mysql' or 'mariadb'
            database: Optional specific database
        """
        self.db_type = db_type.lower()
        self.database = database
        self.engine = get_engine(self.db_type, database)
        # Caches (per database) to keep schema filtering fast & deterministic
        self._schema_map_cache: Dict[str, Dict[str, List[str]]] = {}
        self._fk_graph_cache: Dict[str, Dict[str, Set[str]]] = {}

        print(f"✅ Connected to {self.db_type.upper()}")
        if database:
            print(f"   Database: {database}")

    def execute_query(self, sql, params=None, timeout=30):
        """
        Execute SQL query

        Args:
            sql: SQL query string
            params: Optional parameters
            timeout: Query timeout in seconds

        Returns:
            dict: {
                'success': bool,
                'result': DataFrame or None,
                'rows_affected': int,
                'execution_time': float,
                'error': str or None
            }
        """
        start_time = time.time()

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql), params or {})

                if result.returns_rows:
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    rows_affected = len(df)
                else:
                    df = None
                    rows_affected = result.rowcount

                conn.commit()

            execution_time = time.time() - start_time

            return {
                "success": True,
                "result": df,
                "rows_affected": rows_affected,
                "execution_time": execution_time,
                "error": None,
                "db_type": self.db_type,
            }

        except Exception as e:
            execution_time = time.time() - start_time

            return {
                "success": False,
                "result": None,
                "rows_affected": 0,
                "execution_time": execution_time,
                "error": str(e),
                "db_type": self.db_type,
            }

    def get_schema(self, database=None):
        """
        Get database schema as formatted string

        Args:
            database: Optional database name

        Returns:
            str: Schema in CREATE TABLE format
        """
        if database:
            self.switch_database(database)

        inspector = inspect(self.engine)
        tables = inspector.get_table_names()

        schema_lines = []

        for table in tables:
            columns = inspector.get_columns(table)
            pk = inspector.get_pk_constraint(table)
            fks = inspector.get_foreign_keys(table)

            # Build CREATE TABLE
            schema_lines.append(f"CREATE TABLE {table} (")

            col_defs = []
            for col in columns:
                col_type = str(col["type"])
                col_def = f"    {col['name']} {col_type}"

                if not col["nullable"]:
                    col_def += " NOT NULL"
                if col.get("default"):
                    col_def += f" DEFAULT {col['default']}"

                col_defs.append(col_def)

            # Add constraints
            if pk["constrained_columns"]:
                pk_cols = ", ".join(pk["constrained_columns"])
                col_defs.append(f"    PRIMARY KEY ({pk_cols})")

            for fk in fks:
                fk_cols = ", ".join(fk["constrained_columns"])
                ref_table = fk["referred_table"]
                ref_cols = ", ".join(fk["referred_columns"])
                col_defs.append(
                    f"    FOREIGN KEY ({fk_cols}) REFERENCES {ref_table}({ref_cols})"
                )

            schema_lines.append(",\n".join(col_defs))
            schema_lines.append(");\n")

        return "\n".join(schema_lines)

    def get_schema_for_dataset(self, dataset_name):
        """
        Get schema for specific text2sql dataset

        Args:
            dataset_name: 'academic', 'imdb', 'yelp', etc.

        Returns:
            str: Schema string
        """
        db_name = self.DATASET_DATABASES.get(dataset_name)
        if not db_name:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return self.get_schema(db_name)

    def get_compact_schema(
        self,
        database=None,
        include_types: bool = False,
        max_tables: int | None = None,
        question: str | None = None,
        add_fk_neighbors: bool = True,
    ) -> str:
        """
        Return compact schema:
          table(col1, col2, ...)

        Improvements:
        - Uses full schema_map + deterministic scoring for relevance
        - Optionally expands by 1-hop FK neighbors for joinability
        - Caches schema info per database
        """
        if database:
            self.switch_database(database)

        schema_map = self.get_schema_map()  # cached
        fk_graph = self.get_fk_graph() if (question and add_fk_neighbors) else None

        tables = list(schema_map.keys())

        if question and max_tables is not None:
            tables = self._select_tables(
                question=question,
                schema_map=schema_map,
                max_tables=max_tables,
                fk_graph=fk_graph,
                add_neighbors=add_fk_neighbors,
            )
        elif max_tables is not None:
            tables = sorted(tables)[:max_tables]
        else:
            tables = sorted(tables)

        inspector = inspect(self.engine)

        lines = []
        for table in tables:
            cols = inspector.get_columns(
                table
            )  # or schema_map[table], but types need inspector
            if include_types:
                col_str = ", ".join(f"{c['name']} {str(c['type'])}" for c in cols)
            else:
                col_str = ", ".join(c["name"] for c in cols)

            lines.append(f"{table}({col_str})")

        return "\n".join(lines)

        # ----------------------------

    # Schema introspection helpers (cached)
    # ----------------------------

    def get_schema_map(self, database: str | None = None) -> Dict[str, List[str]]:
        """
        Return full schema as a dict:
          {table_name: [col1, col2, ...], ...}

        Cached per database for speed/reproducibility.
        """
        if database:
            self.switch_database(database)

        db_key = self.database or ""
        if db_key in self._schema_map_cache:
            return self._schema_map_cache[db_key]

        inspector = inspect(self.engine)
        tables = inspector.get_table_names()

        schema_map: Dict[str, List[str]] = {}
        for t in tables:
            cols = inspector.get_columns(t)
            schema_map[t] = [c["name"] for c in cols]

        self._schema_map_cache[db_key] = schema_map
        return schema_map

    def get_fk_graph(self, database: str | None = None) -> Dict[str, Set[str]]:
        """
        Return a simple undirected FK neighbor graph:
          graph[table] = set(other_tables_connected_by_fk)

        Cached per database.
        """
        if database:
            self.switch_database(database)

        db_key = self.database or ""
        if db_key in self._fk_graph_cache:
            return self._fk_graph_cache[db_key]

        inspector = inspect(self.engine)
        tables = inspector.get_table_names()

        graph: Dict[str, Set[str]] = {t: set() for t in tables}

        for t in tables:
            try:
                fks = inspector.get_foreign_keys(t) or []
            except Exception:
                fks = []

            for fk in fks:
                ref = fk.get("referred_table")
                if not ref:
                    continue
                if t in graph:
                    graph[t].add(ref)
                if ref in graph:
                    graph[ref].add(t)

        self._fk_graph_cache[db_key] = graph
        return graph

    # ----------------------------
    # Improved schema filtering
    # ----------------------------

    @staticmethod
    def _split_ident(s: str) -> List[str]:
        s = s.replace("_", " ")
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
        s = re.sub(r"(\D)(\d)", r"\1 \2", s)
        s = re.sub(r"(\d)(\D)", r"\1 \2", s)
        return [t for t in re.findall(r"[A-Za-z0-9]+", s) if t]

    @staticmethod
    def _stem(tok: str) -> str:
        # tiny deterministic stemmer (plural + common suffixes)
        if len(tok) > 4 and tok.endswith("ing"):
            return tok[:-3]
        if len(tok) > 3 and tok.endswith("ed"):
            return tok[:-2]
        if len(tok) > 3 and tok.endswith("es"):
            return tok[:-2]
        if len(tok) > 2 and tok.endswith("s"):
            return tok[:-1]
        return tok

    @classmethod
    def _normalize_tokens(cls, text: str) -> List[str]:
        STOPWORDS = {
            "a",
            "an",
            "the",
            "all",
            "any",
            "from",
            "to",
            "of",
            "in",
            "on",
            "at",
            "for",
            "with",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "do",
            "does",
            "did",
            "list",
            "show",
            "give",
            "get",
            "find",
            "what",
            "which",
            "who",
            "where",
            "when",
            "how",
            "many",
            "much",
        }
        toks = []
        for raw in cls._split_ident(text.lower()):
            if raw in STOPWORDS:
                continue
            toks.append(cls._stem(raw))
        return toks

    @staticmethod
    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _rank_tables_for_question(
        self,
        question: str,
        schema_map: Dict[str, List[str]],
        table_weight: float = 3.0,
        col_weight: float = 1.0,
        fuzzy_weight: float = 0.25,
    ) -> List[Tuple[str, float]]:
        """
        Deterministic ranking:
          score = table_weight*|Q ∩ table_tokens| + col_weight*|Q ∩ col_tokens| + fuzzy_weight*best_fuzzy
        """
        q_tokens = self._normalize_tokens(question)
        q_set = set(q_tokens)

        ranked: List[Tuple[str, float]] = []

        for table, cols in schema_map.items():
            t_tokens = {self._stem(x.lower()) for x in self._split_ident(table)}
            c_tokens: Set[str] = set()
            for c in cols:
                for x in self._split_ident(c):
                    c_tokens.add(self._stem(x.lower()))

            table_hits = len(q_set & t_tokens)
            col_hits = len(q_set & c_tokens)

            score = table_weight * table_hits + col_weight * col_hits

            # fuzzy helps "flights" ~ "flight", small typos, etc.
            best_fuzzy = 0.0
            for qt in q_tokens:
                for tt in t_tokens:
                    best_fuzzy = max(best_fuzzy, self._ratio(qt, tt))
            score += fuzzy_weight * best_fuzzy

            ranked.append((table, score))

        ranked.sort(key=lambda x: (-x[1], x[0]))  # deterministic tie-break
        return ranked

    def _select_tables(
        self,
        question: str,
        schema_map: Dict[str, List[str]],
        max_tables: int,
        fk_graph: Optional[Dict[str, Set[str]]] = None,
        add_neighbors: bool = True,
    ) -> List[str]:
        ranked = self._rank_tables_for_question(question, schema_map)

        selected = [t for (t, s) in ranked if s > 0][:max_tables]

        # fallback: if question too generic, don't hide schema entirely
        if not selected:
            selected = sorted(schema_map.keys())[:max_tables]

        # Optional 1-hop FK expansion to help joins
        if fk_graph and add_neighbors and len(selected) < max_tables:
            seen = set(selected)
            base = list(selected)
            for t in base:
                for nb in sorted(fk_graph.get(t, set())):
                    if nb not in seen:
                        selected.append(nb)
                        seen.add(nb)
                        if len(selected) >= max_tables:
                            break
                if len(selected) >= max_tables:
                    break

        return selected

    def switch_database(self, database):
        """Switch to different database"""
        self.database = database
        self.engine = get_engine(self.db_type, database)

    def list_databases(self):
        """List all databases"""
        result = self.execute_query("SHOW DATABASES")
        if result["success"]:
            return result["result"]["Database"].tolist()
        return []

    def get_table_names(self, database=None):
        """Get table names"""
        if database:
            self.switch_database(database)

        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_dataset_info(self, dataset_name):
        """
        Get information about a text2sql dataset

        Returns:
            dict: {
                'database': str,
                'tables': list,
                'table_count': int,
                'available': bool
            }
        """
        db_name = self.DATASET_DATABASES.get(dataset_name)
        if not db_name:
            return {"available": False}

        databases = self.list_databases()
        available = db_name in databases

        if available:
            tables = self.get_table_names(db_name)
            return {
                "database": db_name,
                "tables": tables,
                "table_count": len(tables),
                "available": True,
            }
        else:
            return {
                "database": db_name,
                "available": False,
                "error": "Database not found",
            }

    def test_dataset_query(self, dataset_name, sql_query):
        """
        Test a query on a specific dataset

        Args:
            dataset_name: Dataset name
            sql_query: SQL query to test

        Returns:
            dict: Query execution result
        """
        db_name = self.DATASET_DATABASES.get(dataset_name)
        if not db_name:
            return {"success": False, "error": f"Unknown dataset: {dataset_name}"}

        # Switch to dataset database
        self.switch_database(db_name)

        # Execute query
        result = self.execute_query(sql_query)
        result["dataset"] = dataset_name
        result["database"] = db_name

        return result

    def close(self):
        """Close connection"""
        self.engine.dispose()
        print(f"✅ Closed connection to {self.db_type.upper()}")
