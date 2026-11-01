"""Validator policy tests, including nested and stacked-statement bypasses."""
import pytest

from guardrails import validate_sql

ALLOWED = ["employees", "customers", "tickets"]


def test_valid_select_passes():
    assert validate_sql("SELECT id, name FROM employees", ALLOWED)[0]


def test_drop_table_is_blocked():
    assert not validate_sql("DROP TABLE employees", ALLOWED)[0]


def test_nonexistent_table_is_blocked():
    assert not validate_sql("SELECT * FROM secrets", ALLOWED)[0]


def test_malformed_syntax_is_caught():
    assert validate_sql("SELECT ( FROM employees", ALLOWED) == (False, "Invalid SQL syntax.")


@pytest.mark.parametrize("sql", [
    "SELECT id FROM employees; DROP TABLE employees",
    "SELECT 1; SELECT 2", "SELECT FROM employees", "", "-- comment only",
    "SELECT * FROM employees JOIN secrets ON employees.id=secrets.id",
    "SELECT * FROM employees WHERE id IN (SELECT id FROM secrets)",
    "SELECT * FROM temp.employees", "SELECT * FROM other.employees",
    "SELECT * FROM sqlite_master", "PRAGMA table_info(employees)",
    "WITH x AS (SELECT * FROM employees) SELECT * FROM x",
    "SELECT id FROM employees UNION SELECT id FROM customers",
    "SELECT * INTO other FROM employees", "SELECT * FROM json_each('[1]')",
])
def test_bypass_attempts_are_blocked(sql):
    assert not validate_sql(sql, ALLOWED)[0]


@pytest.mark.parametrize("sql", [
    "SELECT e.name, t.subject FROM employees e JOIN tickets t ON t.employee_id=e.id",
    'SELECT id FROM main."employees"',
    "SELECT id FROM employees WHERE id IN (SELECT employee_id FROM tickets)",
    "SELECT '; DROP TABLE employees' AS literal FROM customers",
    "SELECT COUNT(*) FROM employees;", "SELECT 1",
])
def test_safe_variants_pass(sql):
    assert validate_sql(sql, ALLOWED)[0]
