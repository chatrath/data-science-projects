"""Synthetic data and an independently enforced, bounded SQLite read boundary."""
from contextlib import closing, contextmanager
from datetime import date, timedelta
from pathlib import Path
import re
import sqlite3
import time

DB_PATH = Path(__file__).resolve().parent / "mock_ enterprise.db"
TABLES = {"employees", "customers", "tickets"}
MAX_ROWS = 200
# Explicit allowlist: extension loading and arbitrary SQL functions are denied.
FUNCTIONS = {"count", "sum", "avg", "min", "max", "round", "coalesce", "ifnull",
             "lower", "upper", "length", "substr", "trim", "date", "datetime",
             "strftime", "julianday", "abs", "like", "glob", "nullif"}


def initialize_database(path: Path = DB_PATH) -> None:
    """Create and seed once; subsequent runs preserve existing records."""
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary REAL)")
        conn.execute("CREATE TABLE IF NOT EXISTS customers (id INTEGER PRIMARY KEY, email TEXT, country TEXT, signup_date TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS tickets (id INTEGER PRIMARY KEY, employee_id INTEGER REFERENCES employees(id), subject TEXT, status TEXT)")
        # Supports country grouping/counts without an unindexed scan or temp sort.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_customers_country ON customers(country)")
        conn.executemany("INSERT OR IGNORE INTO employees VALUES (?, ?, ?, ?)", [
            (1, "Alex Morgan", "Engineering", 82000), (2, "Sam Patel", "Support", 54000),
            (3, "Jordan Lee", "Finance", 71000)])
        today = date.today()
        last_monday = today - timedelta(days=today.weekday() + 7)
        conn.executemany("INSERT OR IGNORE INTO customers VALUES (?, ?, ?, ?)", [
            (1, "alex@example.com", "United Kingdom", last_monday.isoformat()),
            (2, "sam@example.com", "Germany", (last_monday + timedelta(days=2)).isoformat()),
            (3, "jordan@example.com", "United States", today.isoformat()),
            (4, "casey@example.com", "France", (today - timedelta(days=30)).isoformat())])
        conn.executemany("INSERT OR IGNORE INTO tickets VALUES (?, ?, ?, ?)", [
            (1, 1, "VPN access", "open"), (2, 2, "Monitor replacement", "closed")])


def _authorize(action, arg1, arg2, database, source):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and arg1 in TABLES and database in ("main", None):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in FUNCTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


@contextmanager
def read_only_connection(path: Path = DB_PATH):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    try:
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_authorize)
        deadline = time.monotonic() + 2
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        yield conn
    finally:
        conn.close()


def _validate(sql: str) -> str:
    if not isinstance(sql, str) or len(sql) > 10000:
        raise ValueError("SQL must be a string of at most 10,000 characters.")
    sql = sql.strip().removesuffix(";").strip()
    if not re.match(r"^SELECT\b", sql, flags=re.I):
        raise ValueError("Only a single SELECT statement is supported by this MVP.")
    return sql


def get_schema(path: Path = DB_PATH) -> str:
    """Expose metadata only, without sending table rows to the model."""
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        definitions = [row[0] for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name IN (?, ?, ?) ORDER BY name",
            tuple(sorted(TABLES)))]
    return "\n".join(definitions)


def get_table_metadata(table_name: str, path: Path = DB_PATH) -> dict:
    """Trusted metadata/count lookup; identifiers are checked before SQL use."""
    name = table_name.strip().casefold()
    if name not in TABLES:
        raise ValueError("Table is outside the allowed table list.")
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=2)) as conn:
        conn.execute("PRAGMA query_only = ON")
        with closing(conn.execute(f'PRAGMA table_info("{name}")')) as cursor:
            columns = [{"name": row[1], "type": row[2], "primary_key": bool(row[5]),
                        "nullable": not bool(row[3] or row[5])} for row in cursor.fetchall()]
    if not columns:
        raise ValueError("Requested table does not exist in the database.")
    with read_only_connection(path) as conn:
        with closing(conn.execute(f'SELECT COUNT(*) FROM "{name}"')) as cursor:
            count = cursor.fetchone()[0]
    return {"table": name, "columns": columns, "row_count": count}


def dry_run(sql: str, path: Path = DB_PATH) -> None:
    """Compile the query with the same authorizer used for execution."""
    sql = _validate(sql)
    with read_only_connection(path) as conn:
        cursor = conn.execute("EXPLAIN QUERY PLAN " + sql)
        cursor.close()


def execute_query(sql: str, row_limit: int = MAX_ROWS, path: Path = DB_PATH) -> dict:
    """Return bounded rows, ordered column names, and truncation metadata."""
    if type(row_limit) is not int or not 1 <= row_limit <= MAX_ROWS:
        raise ValueError(f"Row limit must be between 1 and {MAX_ROWS}.")
    sql = _validate(sql)
    with read_only_connection(path) as conn:
        cursor = conn.execute(sql)  # execute rejects stacked statements.
        try:
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchmany(row_limit + 1)
            return {"columns": columns, "rows": rows[:row_limit], "truncated": len(rows) > row_limit}
        finally:
            cursor.close()


if __name__ == "__main__":
    initialize_database()
    print(f"Database ready: {DB_PATH}")
