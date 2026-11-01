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
    run_cached = cache.run_cached
    monkeypatch.setattr(cache, "run_cached", lambda request, limit, live, db:
                        run_cached(request, limit, live, db, result_path=tmp_path / "cache.db"))
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
        assert result["rows"] == [("Alex Morgan", 82000.0)]
