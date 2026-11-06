"""Approval must survive rebuilds, bind exact SQL, and audit once."""
import sqlite3
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from agents import SafetyDecision
import audit_logger
import graph as workflow

SQL = "SELECT name FROM employees WHERE id=1"


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(workflow, "get_schema", lambda: "CREATE TABLE employees(id INTEGER PRIMARY KEY, name TEXT)")
    monkeypatch.setattr(workflow, "check_intent", lambda *args: SafetyDecision(allowed=True, reason="Read only"))
    monkeypatch.setattr(workflow, "generate_sql", lambda *args, **kwargs: SQL)
    monkeypatch.setattr(workflow, "inspect_query_plan", lambda *args, **kwargs: (True, "Plan approved"))
    from contextlib import nullcontext
    monkeypatch.setattr(workflow, "read_only_connection", lambda: nullcontext(None))
    execute = Mock(return_value={"rows": [["Alex"]], "columns": ["name"], "truncated": False})
    monkeypatch.setattr(workflow, "execute_query", execute)
    monkeypatch.setattr(workflow, "send_dba_alert", Mock(return_value={"status": "Disabled"}))
    return execute


def test_pauses_then_approves_exactly_once(setup):
    engine = workflow.build_graph(MemorySaver())
    tid, state = workflow.start_request("Find employee 1", engine=engine)
    assert state["status"] == "awaiting_approval"
    assert state["request_id"] == tid
    setup.assert_not_called()
    assert not audit_logger.read_recent_audit_logs()
    result = workflow.resume_request(tid, "approve", state["approval_token"], engine=engine)
    assert result["status"] == "completed"
    workflow.resume_request(tid, "approve", state["approval_token"], engine=engine)
    workflow.start_request("Find employee 1", thread_id=tid, engine=engine)
    setup.assert_called_once()
    assert len(audit_logger.read_recent_audit_logs()) == 1


def test_reject_never_executes(setup):
    engine = workflow.build_graph()
    tid, state = workflow.start_request("Find employee 1", engine=engine)
    result = workflow.resume_request(tid, "reject", state["approval_token"], engine=engine)
    assert result["status"] == "rejected"
    setup.assert_not_called()
    assert audit_logger.read_recent_audit_logs()[0]["execution_outcome"] == "Rejected-by-Human"


def test_edit_revalidates_and_requires_new_approval(setup):
    engine = workflow.build_graph()
    tid, state = workflow.start_request("Find employee 1", engine=engine)
    revised = "SELECT name FROM employees WHERE id=2"
    state2 = workflow.resume_request(tid, "edit", state["approval_token"], sql=revised, engine=engine)
    assert state2["status"] == "awaiting_approval"
    assert state2["sql"] == revised
    assert state2["approval_token"] != state["approval_token"]
    setup.assert_not_called()
    with pytest.raises(ValueError, match="out of date"):
        workflow.resume_request(tid, "approve", state["approval_token"], engine=engine)
    workflow.resume_request(tid, "approve", state2["approval_token"], engine=engine)
    assert setup.call_args.args[0] == revised


def test_destructive_edit_is_blocked(setup):
    engine = workflow.build_graph()
    tid, state = workflow.start_request("Find employee 1", engine=engine)
    result = workflow.resume_request(tid, "edit", state["approval_token"], sql="DELETE FROM employees", engine=engine)
    assert result["status"] == "blocked"
    setup.assert_not_called()


def test_sqlite_checkpoint_survives_connection_and_graph_rebuild(setup, tmp_path):
    path = tmp_path / "checkpoints.db"
    with sqlite3.connect(path, check_same_thread=False) as conn:
        engine = workflow.build_graph(SqliteSaver(conn))
        tid, state = workflow.start_request("Find employee 1", engine=engine)
    with sqlite3.connect(path, check_same_thread=False) as conn:
        restored = workflow.build_graph(SqliteSaver(conn))
        result = workflow.resume_request(tid, "approve", state["approval_token"], engine=restored)
    assert result["status"] == "completed"
    setup.assert_called_once()


def test_terminal_audit_is_idempotent_after_checkpoint_replay(setup):
    state = {"request_id": "idempotent", "status": "rejected", "approval_status": "rejected"}
    audit_logger.append_audit_event(state)
    audit_logger.append_audit_event(state)
    assert len(audit_logger.read_recent_audit_logs()) == 1


def test_timeout_retries_only_failed_stage_and_does_not_approve(setup, monkeypatch):
    generation = Mock(side_effect=[TimeoutError("slow model"), SQL])
    monkeypatch.setattr(workflow, "generate_sql", generation)
    safety = Mock(return_value=SafetyDecision(allowed=True, reason="Read only"))
    monkeypatch.setattr(workflow, "check_intent", safety)
    engine = workflow.build_graph()
    tid, state = workflow.start_request("Find employee 1", engine=engine)
    assert state["run_state"] == "failed_recoverable"
    assert state["error_kind"] == "timeout"
    assert not audit_logger.read_recent_audit_logs()
    with pytest.raises(ValueError, match="not awaiting SQL approval"):
        workflow.resume_request(tid, "approve", "invalid", engine=engine)
    state = workflow.recover_request(tid, recovery_token=state["recovery_token"], engine=engine)
    assert state["run_state"] == "awaiting_approval"
    assert state["recovery_attempts"] == 1
    safety.assert_called_once()
    setup.assert_not_called()
    with pytest.raises(ValueError, match="SQL approval is required"):
        workflow.recover_request(tid, engine=engine)
    state = workflow.resume_request(tid, "approve", state["approval_token"], engine=engine)
    assert state["run_state"] == "finished"
    assert len(audit_logger.read_recent_audit_logs()) == 1


def test_recovery_checkpoint_survives_restart(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(workflow, "generate_sql", Mock(side_effect=[TimeoutError("slow"), SQL]))
    path = tmp_path / "recovery.db"
    with sqlite3.connect(path, check_same_thread=False) as conn:
        tid, failed = workflow.start_request("Find employee 1", engine=workflow.build_graph(SqliteSaver(conn)))
    with sqlite3.connect(path, check_same_thread=False) as conn:
        engine = workflow.build_graph(SqliteSaver(conn))
        assert workflow.get_request_state(tid, engine=engine)["run_state"] == "failed_recoverable"
        state = workflow.recover_request(tid, recovery_token=failed["recovery_token"], engine=engine)
        assert state["run_state"] == "awaiting_approval"
        state = workflow.resume_request(tid, "reject", state["approval_token"], engine=engine)
        assert state["run_state"] == "finished"
    setup.assert_not_called()


def test_recovery_exhaustion_is_terminal_and_audited_once(setup, monkeypatch):
    generation = Mock(side_effect=TimeoutError("model timeout"))
    monkeypatch.setattr(workflow, "generate_sql", generation)
    engine = workflow.build_graph()
    tid, state = workflow.start_request("Find employee 1", engine=engine)
    for _ in range(workflow.MAX_RECOVERY_ATTEMPTS):
        state = workflow.recover_request(tid, recovery_token=state["recovery_token"], engine=engine)
    assert state["run_state"] == "finished"
    assert state["status"] == "error"
    assert state["recovery_attempts"] == 3
    assert "limit reached" in state["message"]
    assert generation.call_count == 4
    workflow.recover_request(tid, engine=engine)
    assert generation.call_count == 4
    setup.assert_not_called()
    assert len(audit_logger.read_recent_audit_logs()) == 1


def test_audit_retry_does_not_execute_sql_again(setup, monkeypatch):
    real_append = workflow.append_audit_event
    append = Mock(side_effect=[OSError("storage offline"), None])
    def persist(state):
        if append.call_count == 0:
            return append(state)
        append(state)
        return real_append(state)
    monkeypatch.setattr(workflow, "append_audit_event", persist)
    engine = workflow.build_graph()
    tid, state = workflow.start_request("Find employee 1", engine=engine)
    state = workflow.resume_request(tid, "approve", state["approval_token"], engine=engine)
    assert state["run_state"] == "failed_recoverable"
    assert state["failed_node"] == "audit"
    setup.assert_called_once()
    state = workflow.recover_request(tid, recovery_token=state["recovery_token"], engine=engine)
    assert state["run_state"] == "finished" and state["status"] == "completed"
    assert state["message"] == "Returned 1 row(s)."
    setup.assert_called_once()
    assert len(audit_logger.read_recent_audit_logs()) == 1


def test_stale_recovery_and_cancellation_are_safe(setup, monkeypatch):
    monkeypatch.setattr(workflow, "generate_sql", Mock(side_effect=TimeoutError("slow")))
    engine = workflow.build_graph()
    tid, state = workflow.start_request("Find employee 1", engine=engine)
    with pytest.raises(ValueError, match="out of date"):
        workflow.recover_request(tid, recovery_token="old", engine=engine)
    state = workflow.recover_request(tid, "cancel", state["recovery_token"], engine=engine)
    assert state["run_state"] == "finished"
    assert state["status"] == "error"
    setup.assert_not_called()


def test_native_failed_task_resumes_after_restart_without_replaying_input(setup, monkeypatch, tmp_path):
    schema = Mock(return_value="CREATE TABLE employees(id INTEGER PRIMARY KEY, name TEXT)")
    monkeypatch.setattr(workflow, "get_schema", schema)
    original = workflow.safety_node
    monkeypatch.setattr(workflow, "safety_node", Mock(side_effect=TimeoutError("process interrupted")))
    path = tmp_path / "native.db"
    cfg = workflow.request_config("native-failure")
    with sqlite3.connect(path, check_same_thread=False) as conn:
        engine = workflow.build_graph(SqliteSaver(conn))
        with pytest.raises(TimeoutError):
            engine.invoke({"request": "Find employee 1"}, cfg)
    monkeypatch.setattr(workflow, "safety_node", original)
    with sqlite3.connect(path, check_same_thread=False) as conn:
        engine = workflow.build_graph(SqliteSaver(conn))
        state = workflow.get_request_state("native-failure", engine=engine)
        assert state["run_state"] == "failed_recoverable"
        state = workflow.recover_request("native-failure", engine=engine)
        assert state["run_state"] == "awaiting_approval"
    schema.assert_called_once()
    setup.assert_not_called()


def test_runner_records_escaped_task_failure_as_recovery(setup, monkeypatch):
    monkeypatch.setattr(workflow, "safety_node", Mock(side_effect=TimeoutError("unexpected timeout")))
    engine = workflow.build_graph()
    tid, state = workflow.start_request("Find employee 1", engine=engine)
    assert state["run_state"] == "failed_recoverable"
    assert state["failed_node"] == "safety"
    assert state["error_kind"] == "timeout"
    setup.assert_not_called()
