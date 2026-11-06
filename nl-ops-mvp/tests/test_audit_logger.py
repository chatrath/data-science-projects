"""Audit persistence, graph interceptions, cache coverage, and write errors."""
from conftest import invoke_reviewed
from datetime import datetime
import json
from unittest.mock import patch

import pytest

import audit_logger
from agents import SafetyDecision
from cache import run_cached
from database import get_schema, initialize_database, read_only_connection, execute_query
from graph import graph


@pytest.fixture
def local_graph(tmp_path):
    path = tmp_path / "enterprise.db"
    initialize_database(path)
    with patch("graph.get_schema", side_effect=lambda: get_schema(path)), \
         patch("graph.read_only_connection", side_effect=lambda: read_only_connection(path)), \
         patch("graph.execute_query", side_effect=lambda sql, limit: execute_query(sql, limit, path)):
        yield path


def test_successful_query_appends_structured_json(local_graph):
    raw = "  Show employee 1\nplease  "
    with patch("graph.check_intent", return_value=SafetyDecision(allowed=True, reason="Read only")), \
         patch("graph.generate_sql", return_value="SELECT name FROM employees WHERE id=1"):
        result = invoke_reviewed(graph, {"request": raw})
    assert result["status"] == "completed"
    lines = audit_logger.AUDIT_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert datetime.fromisoformat(record["timestamp"]).tzinfo is not None
    assert record["user_query"] == raw
    assert record["safety_status"] == "Passed"
    assert record["sql_generated"] == "SELECT name FROM employees WHERE id=1"
    assert record["guardrail_status"] == {"ast": "Passed", "explain": "Passed"}
    assert record["execution_outcome"] == "Success"
    assert record["row_count"] == 1


@pytest.mark.parametrize("sql,ast,plan", [
    ("DROP TABLE employees", "Blocked", "NotRun"),
    ("SELECT * FROM employees", "Passed", "Blocked"),
])
def test_guardrail_rejections_are_logged(local_graph, sql, ast, plan):
    with patch("graph.check_intent", return_value=SafetyDecision(allowed=True, reason="Read only")), \
         patch("graph.generate_sql", return_value=sql), patch("graph.execute_query") as execute:
        invoke_reviewed(graph, {"request": "Test rejected SQL"})
    execute.assert_not_called()
    record = audit_logger.read_recent_audit_logs()[0]
    assert record["execution_outcome"] == "Blocked-by-Guardrail"
    assert record["guardrail_status"] == {"ast": ast, "explain": plan}
    assert record["row_count"] == 0


def test_cached_query_appends_once_per_submission(local_graph, tmp_path):
    request = "  Show employee 1  "
    def live():
        return invoke_reviewed(graph, {"request": request, "row_limit": 50})
    with patch("graph.check_intent", return_value=SafetyDecision(allowed=True, reason="Read only")), \
         patch("graph.generate_sql", return_value="SELECT name FROM employees WHERE id=1"):
        run_cached(request, 50, live, local_graph, tmp_path / "results.db")
    run_cached(request, 50, lambda: pytest.fail("Cached submission must skip live work"), local_graph, tmp_path / "results.db")
    newest, original = audit_logger.read_recent_audit_logs()
    assert newest["execution_outcome"] == "Cached"
    assert original["execution_outcome"] == "Success"
    assert newest["request_id"] != original["request_id"]
    assert newest["user_query"] == request
    assert newest["row_count"] == 1


def test_safety_block_and_empty_input_are_logged(local_graph):
    with patch("graph.check_intent", return_value=SafetyDecision(allowed=False, reason="Writes blocked")):
        invoke_reviewed(graph, {"request": "Delete customers"})
    invoke_reviewed(graph, {"request": ""})
    empty, blocked = audit_logger.read_recent_audit_logs()
    assert empty["execution_outcome"] == "Invalid-Request"
    assert blocked["execution_outcome"] == "Blocked-by-Safety"
    assert blocked["safety_status"] == "Blocked"


def test_audit_write_failure_is_visible(local_graph):
    with patch("graph.append_audit_event", side_effect=OSError("Disk full")):
        result = invoke_reviewed(graph, {"request": ""})
    assert result["audit_error"]
    assert result["status"] == "error"
    assert not result["audit_written"]


def test_reader_tolerates_corrupt_line_and_returns_newest_first():
    audit_logger.append_audit_event({"request": "first"})
    with audit_logger.AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write("not JSON\n")
    audit_logger.append_audit_event({"request": "second"})
    assert [r["user_query"] for r in audit_logger.read_recent_audit_logs()] == ["second", "first"]
