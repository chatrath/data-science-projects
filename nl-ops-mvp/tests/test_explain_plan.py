"""Exercise actual SQLite plans without executing the submitted SELECT."""
from contextlib import closing
import sqlite3
from unittest.mock import Mock

import pytest

from database import initialize_database, read_only_connection
from guardrails import inspect_query_plan


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "plans.db"
    initialize_database(path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE INDEX idx_employees_department ON employees(department)")
    return path


@pytest.mark.parametrize("sql", [
    "SELECT name FROM employees WHERE id=1",
    "SELECT name FROM employees WHERE department='Engineering'",
    "SELECT 1", "SELECT name FROM employees WHERE salary > 60000",
    "SELECT department FROM employees ORDER BY department",
    "SELECT e.name, t.subject FROM tickets t JOIN employees e ON e.id=t.employee_id WHERE t.status='open'",
])
def test_indexed_or_simple_query_passes(db_path, sql):
    with read_only_connection(db_path) as conn:
        assert inspect_query_plan(conn, sql) == (True, "Plan looks optimal.")


@pytest.mark.parametrize("sql", [
    "SELECT * FROM employees", "SELECT * FROM employees LIMIT 1",
    "SELECT e.name FROM employees AS e",
    "SELECT name FROM employees /* WHERE id=1 */",
    "SELECT name FROM employees WHERE 1=1",
    "SELECT name FROM employees WHERE EXISTS (SELECT id FROM customers WHERE id=1)",
])
def test_unfiltered_core_scan_is_blocked(db_path, sql):
    with read_only_connection(db_path) as conn:
        assert inspect_query_plan(conn, sql) == (
            False, "Query rejected: Estimated plan requires an unindexed full-table scan.")


def test_temp_sort_is_blocked(db_path):
    with read_only_connection(db_path) as conn:
        allowed, message = inspect_query_plan(conn, "SELECT name FROM employees WHERE salary > 0 ORDER BY name")
    assert not allowed
    assert "temporary b-tree sort" in message


def test_only_explain_is_executed(db_path):
    with read_only_connection(db_path) as conn:
        observed = Mock(wraps=conn)
        inspect_query_plan(observed, "SELECT name FROM employees WHERE id=1")
        observed.execute.assert_called_once_with("EXPLAIN QUERY PLAN SELECT name FROM employees WHERE id=1")


@pytest.mark.parametrize("sql", ["DROP TABLE employees", "SELECT missing_column FROM employees",
                                 "SELECT * FROM secrets", "SELECT load_extension('bad')"])
def test_unsafe_or_unplannable_query_fails_closed(db_path, sql):
    with read_only_connection(db_path) as conn:
        assert not inspect_query_plan(conn, sql)[0]
