"""Enterprise recovery integration tests with real SQLite and sqlglot."""
from conftest import invoke_reviewed
from contextlib import contextmanager
import sqlite3
from unittest.mock import Mock

import pytest
import graph as workflow
from guardrails import validate_sql_security, inspect_execution_plan


@pytest.fixture
def enterprise(monkeypatch):
    from agents import SafetyDecision
    monkeypatch.setattr(workflow, "check_intent", lambda *args: SafetyDecision(allowed=True, reason="Read only"))
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO employees VALUES (1, 'Alex')")
    @contextmanager
    def connection():
        yield conn
    monkeypatch.setattr(workflow, "read_only_connection", connection)
    monkeypatch.setattr(workflow, "get_schema", lambda: "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT);")
    def execute(sql, limit):
        cursor = conn.execute(sql)
        return {"rows": cursor.fetchmany(limit), "columns": [c[0] for c in cursor.description], "truncated": False}
    monkeypatch.setattr(workflow, "execute_query", execute)
    yield
    conn.close()


@pytest.mark.parametrize("sql", ["DROP TABLE employees", "DELETE FROM employees",
                                   "UPDATE employees SET name='x'", "INSERT INTO employees VALUES(2,'x')",
                                   "ALTER TABLE employees ADD COLUMN x TEXT",
                                   "SELECT 1; DELETE FROM employees"])
def test_destructive_statements(sql):
    assert validate_sql_security(sql) == (False, "Security Violation: Destructive statement detected")


def test_plan_rejects_filtered_unindexed_scan():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE employees(id INTEGER PRIMARY KEY, name TEXT)")
        ok, reason = inspect_execution_plan(conn, "SELECT id FROM employees WHERE name='Alex'")
        assert not ok and "SCAN" in reason
        assert inspect_execution_plan(conn, "SELECT id FROM employees WHERE id=1")[0]
    finally:
        conn.close()


def test_missing_column_corrected(enterprise, monkeypatch):
    generate = Mock(side_effect=["SELECT missing FROM employees WHERE id=1", "SELECT name FROM employees WHERE id=1"])
    monkeypatch.setattr(workflow, "generate_sql", generate)
    result = invoke_reviewed(workflow.graph, {"request": "Find employee 1"})
    assert result["status"] == "completed"
    assert result["correction_attempts"] == 1
    assert "no such column" in generate.call_args.kwargs["sql_syntax_error"]
    assert result["rows"] == [["Alex"]]


def test_retry_cap(enterprise, monkeypatch):
    generate = Mock(return_value="SELECT ( FROM employees")
    monkeypatch.setattr(workflow, "generate_sql", generate)
    execute = Mock()
    monkeypatch.setattr(workflow, "execute_query", execute)
    result = invoke_reviewed(workflow.graph, {"request": "Find employee 1"})
    assert result["status"] == "error"
    assert result["correction_attempts"] == 3
    assert generate.call_count == 3
    execute.assert_not_called()


def test_execution_error_corrected(enterprise, monkeypatch):
    generate = Mock(return_value="SELECT name FROM employees WHERE id=1")
    monkeypatch.setattr(workflow, "generate_sql", generate)
    execute = Mock(side_effect=[sqlite3.OperationalError("no such column: missing"), {"rows": [["Alex"]], "columns": ["name"], "truncated": False}])
    monkeypatch.setattr(workflow, "execute_query", execute)
    result = invoke_reviewed(workflow.graph, {"request": "Find employee 1"})
    assert result["status"] == "completed"
    assert result["correction_attempts"] == 1
    assert "no such column" in generate.call_args.kwargs["sql_syntax_error"]
