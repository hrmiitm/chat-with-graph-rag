"""
Database helpers — PostgreSQL connection pool + Apache AGE Cypher wrapper.
Uses psycopg v3 (sync) for simplicity and reliability.
"""

import json
import logging
import psycopg
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level connection pool (initialized on first use)
_pool: str = settings.database_url


@contextmanager
def get_conn():
    """Yield a database connection with AGE extensions loaded."""
    conn = psycopg.connect(_pool)
    try:
        with conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute("SET search_path = ag_catalog, \"$user\", public;")
        yield conn
    finally:
        conn.close()


def execute_sql(query: str, params: tuple = None) -> list[dict]:
    """Execute a SQL query and return rows as list of dicts."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description:
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            conn.commit()
            return []


def execute_sql_returning(query: str, params: tuple = None) -> dict | None:
    """Execute a SQL INSERT/UPDATE with RETURNING and return single row."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description:
                cols = [d.name for d in cur.description]
                row = cur.fetchone()
                conn.commit()
                return dict(zip(cols, row)) if row else None
            conn.commit()
            return None


def execute_cypher(cypher: str, graph: str = "document_graph", 
                   columns: str = "v agtype") -> list:
    """
    Execute a Cypher query via Apache AGE's cypher() SQL function.
    
    Args:
        cypher: Cypher query string (e.g. "MATCH (n) RETURN n")
        graph: Name of the AGE graph
        columns: Column definition for the result set
        
    Returns:
        List of result values (parsed from AGE's agtype format)
    """
    sql = f"SELECT * FROM cypher('{graph}', $$ {cypher} $$) as ({columns});"
    results = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql)
                if cur.description:
                    rows = cur.fetchall()
                    for row in rows:
                        parsed = []
                        for val in row:
                            parsed.append(_parse_agtype(val))
                        results.append(parsed[0] if len(parsed) == 1 else parsed)
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Cypher query failed: {e}\nQuery: {cypher}")
                raise
    return results


def _parse_agtype(val) -> any:
    """Parse AGE agtype values into Python objects."""
    if val is None:
        return None
    s = str(val)
    # AGE returns vertices/edges as JSON-like strings with ::vertex or ::edge suffix
    for suffix in ["::vertex", "::edge", "::path"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


def check_db_health() -> bool:
    """Return True if database is reachable and extensions are loaded."""
    try:
        rows = execute_sql("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'age');")
        ext_names = {r["extname"] for r in rows}
        return "vector" in ext_names and "age" in ext_names
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        return False
