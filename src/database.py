"""
Msty Admin MCP - Database Utilities

SQLite database operations with SQL injection protection.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional, List

from .constants import ALLOWED_TABLE_NAMES

logger = logging.getLogger("msty-admin-mcp")


def get_database_connection(db_path: str) -> Optional[sqlite3.Connection]:
    """Get a read-only connection to Msty database.

    Uses ``immutable=1`` URI mode so SQLite never waits for the app's write
    lock — reads succeed even while Msty Studio is running.  Falls back to
    ``mode=ro`` and then a plain connection if the immutable flag is not
    supported by the installed SQLite build.
    """
    if not db_path or not Path(db_path).exists():
        return None

    def _try_connect(uri: str) -> Optional[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error:
            return None

    # 1. Immutable snapshot — never blocks on a lock held by another process
    conn = _try_connect(f"file:{db_path}?mode=ro&immutable=1")
    if conn:
        return conn

    # 2. Read-only mode — may block briefly if WAL checkpoint is running
    conn = _try_connect(f"file:{db_path}?mode=ro")
    if conn:
        return conn

    # 3. Plain fallback (e.g. very old SQLite that doesn't support URI flags)
    try:
        conn = sqlite3.connect(db_path, timeout=5, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        return None


def query_database(db_path: str, query: str, params: tuple = ()) -> list:
    """Execute a read-only query on the Msty database"""
    conn = get_database_connection(db_path)
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"Query error: {e}")
        return []
    finally:
        conn.close()


def get_table_names(db_path: str) -> List[str]:
    """Get all table names from the database"""
    query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    results = query_database(db_path, query)
    return [r['name'] for r in results]


def is_safe_table_name(table_name: str) -> bool:
    """Validate table name against allowlist to prevent SQL injection"""
    if not table_name:
        return False
    # Must be valid identifier AND in allowlist
    return table_name.isidentifier() and table_name.lower() in ALLOWED_TABLE_NAMES


def validate_table_exists(db_path: str, table_name: str) -> bool:
    """Check if table exists in database AND is safe to query"""
    if not table_name or not table_name.isidentifier():
        return False
    existing_tables = get_table_names(db_path)
    return table_name in existing_tables


def safe_query_table(
    db_path: str,
    table_name: str,
    limit: int = 100,
    where_clause: str = None,
    params: tuple = ()
) -> list:
    """
    Safely query a table with SQL injection protection.
    Table name must exist in database and be a valid identifier.
    """
    if not validate_table_exists(db_path, table_name):
        logger.warning(f"Rejected query for non-existent or invalid table: {table_name}")
        return []

    # Build safe query - table name validated against actual database tables
    query = f"SELECT * FROM {table_name}"
    if where_clause:
        query += f" WHERE {where_clause}"
    query += " LIMIT ?"

    return query_database(db_path, query, params + (limit,))


def safe_count_table(
    db_path: str,
    table_name: str,
    where_clause: str = None,
    params: tuple = ()
) -> int:
    """Safely count rows in a table with SQL injection protection."""
    if not validate_table_exists(db_path, table_name):
        logger.warning(f"Rejected count for non-existent or invalid table: {table_name}")
        return 0

    query = f"SELECT COUNT(*) as count FROM {table_name}"
    if where_clause:
        query += f" WHERE {where_clause}"

    results = query_database(db_path, query, params)
    return results[0]['count'] if results else 0


def get_table_row_count(db_path: str, table_name: str) -> int:
    """Get row count for a specific table (SQL injection safe)"""
    return safe_count_table(db_path, table_name)


__all__ = [
    "get_database_connection",
    "query_database",
    "get_table_names",
    "is_safe_table_name",
    "validate_table_exists",
    "safe_query_table",
    "safe_count_table",
    "get_table_row_count"
]
