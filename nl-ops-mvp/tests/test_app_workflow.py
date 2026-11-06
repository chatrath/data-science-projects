"""Exercise the real graph stream in Streamlit, including no-op node deltas."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from agents import SafetyDecision
import audit_logger
import cache
import database
import graph as workflow


@pytest.mark.parametrize("blocked", [False, True])
def test_ui_handles_noop_updates_without_false_failure_or_duplicate_audit(tmp_path, monkeypatch, blocked):
    path = tmp_path / "enterprise.db"
    initialize = database.initialize_database
    initialize(path)
    monkeypatch.setattr(database, "DB_PATH", path)
    monkeypatch.setattr(database, "initialize_database", lambda: initialize(path))
    monkeypatch.setattr(workflow, "get_schema", lambda: database.get_schema(path))
    monkeypatch.setattr(workflow, "read_only_connection", lambda: database.read_only_connection(path))
    monkeypatch.setattr(workflow, "execute_query", lambda sql, limit: database.execute_query(sql, limit, path))
    monkeypatch.setattr(workflow, "check_intent", lambda *args: SafetyDecision(allowed=True, reason="Read only"))
    monkeypatch.setattr("tools.read_table_metadata", lambda table: database.get_table_metadata(table, path))
    monkeypatch.setattr("plan_server.start_plan_server", lambda: Mock())
    notify = Mock(return_value={"status": "Sent"})
    monkeypatch.setattr(workflow, "send_dba_alert", notify)
    sql = "SELECT * FROM employees" if blocked else "SELECT name, salary FROM employees WHERE id=1"
    responses = iter([
        AIMessage(content="", tool_calls=[{"name": "get_table_metadata", "args": {"table_name": "employees"}, "id": "meta"}]),
        AIMessage(content="", tool_calls=[{"name": "validate_sql_syntax", "args": {"sql_query": sql}, "id": "lint"}]),
        AIMessage(content=json.dumps({"sql": sql})),
    ])
    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = lambda messages: next(responses)
    monkeypatch.setattr("agents.get_model", lambda: model)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    app.text_area[0].set_value("Inspect employees and show employee 1" if not blocked else "List all employees")
    next(button for button in app.button if button.label == "Submit request").click().run(timeout=20)
    assert not app.exception
    if not blocked:
        assert app.session_state["result"]["status"] == "awaiting_approval"
        assert not audit_logger.read_recent_audit_logs()
        next(button for button in app.button if button.label == "Approve and execute").click().run(timeout=20)
        assert not app.exception
    result = app.session_state["result"]
    assert result["status"] == ("blocked" if blocked else "completed")
    assert result["audit_written"]
    assert "Workflow failed" not in result["message"]
    records = audit_logger.read_recent_audit_logs()
    assert len(records) == 1
    assert records[0]["request_id"] == result["request_id"]
    if blocked:
        notify.assert_called_once()
        assert records[0]["execution_outcome"] == "Blocked-by-Guardrail"
    else:
        notify.assert_not_called()
        assert records[0]["execution_outcome"] == "Success"
        assert result["rows"] == [["Alex Morgan", 82000.0]]

@pytest.mark.parametrize("source,cache_status,execution", [
    ("result cache", "Hit (query results)", "Skipped"),
    ("agents / LLM cache", "Hit (LLM responses)", "Executed"),
    ("live agents", "Miss", "Executed"),
])
def test_results_hide_technical_details(monkeypatch, source, cache_status, execution):
    monkeypatch.setattr(database, "initialize_database", lambda: None)
    monkeypatch.setattr("plan_server.start_plan_server", lambda: Mock())
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state["result"] = {
        "status": "completed", "message": "Returned 1 row(s).",
        "cache_source": source, "execution_status": "Success", "elapsed_seconds": 0.01,
        "columns": ["name"], "rows": [["Alex Morgan"]],
    }
    app.run(timeout=20)
    assert not app.exception
    assert all(expander.label != "Technical details" for expander in app.expander)
    assert any("Request completed in" in caption.value for caption in app.caption)
    assert len(app.dataframe) >= 1
    assert not app.code
    assert all("Cache status" not in info.value and "Database execution" not in info.value
               for info in app.info)


def test_ui_retries_saved_failure_then_requires_explicit_approval(tmp_path, monkeypatch):
    path = tmp_path / "recovery_ui.db"
    initialize = database.initialize_database
    initialize(path)
    monkeypatch.setattr(database, "initialize_database", lambda: initialize(path))
    monkeypatch.setattr(workflow, "get_schema", lambda: database.get_schema(path))
    monkeypatch.setattr(workflow, "read_only_connection", lambda: database.read_only_connection(path))
    safety = Mock(return_value=SafetyDecision(allowed=True, reason="Safe"))
    generation = Mock(side_effect=[TimeoutError("model timeout"), "SELECT name FROM employees WHERE id=1"])
    execution = Mock(return_value={"rows": [["Alex Morgan"]], "columns": ["name"], "truncated": False})
    monkeypatch.setattr(workflow, "check_intent", safety)
    monkeypatch.setattr(workflow, "generate_sql", generation)
    monkeypatch.setattr(workflow, "execute_query", execution)
    monkeypatch.setattr(workflow, "graph", workflow.build_graph())
    monkeypatch.setattr("plan_server.start_plan_server", lambda: Mock())
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    app.text_area[0].set_value("Find employee 1")
    next(b for b in app.button if b.label == "Submit request").click().run(timeout=20)
    assert not app.exception
    assert app.session_state["result"]["run_state"] == "failed_recoverable"
    tid = app.session_state["active_request_id"]
    app.run(timeout=20)
    assert generation.call_count == 1  # Browser rerun must never retry on its own.
    next(b for b in app.button if b.label == "Retry saved request").click().run(timeout=20)
    assert not app.exception
    assert app.session_state["active_request_id"] == tid
    assert app.session_state["result"]["run_state"] == "awaiting_approval"
    execution.assert_not_called()
    safety.assert_called_once()
    next(b for b in app.button if b.label == "Approve and execute").click().run(timeout=20)
    assert not app.exception
    assert app.session_state["result"]["run_state"] == "finished"
    execution.assert_called_once()
    assert len(audit_logger.read_recent_audit_logs()) == 1
