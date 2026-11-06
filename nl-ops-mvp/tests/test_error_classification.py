"""Failures identify their actual source and a bounded, explicit retry policy."""
import errno
import sqlite3
from unittest.mock import Mock
import httpx
import pytest
from agents import ModelOutputError
from workflow_errors import classify_error, OperationError
from guardrails import inspect_query_plan
import audit_logger


@pytest.mark.parametrize("exc,kind,retryable", [
    (TimeoutError(), "timeout", True),
    (httpx.ReadTimeout("slow"), "timeout", True),
    (httpx.ConnectError("offline"), "infrastructure", True),
    (ModelOutputError("invalid JSON"), "model_format", False),
    (sqlite3.OperationalError("database is locked"), "database", True),
    (sqlite3.OperationalError("no such column: missing"), "database", False),
    (PermissionError(errno.EACCES, "denied"), "infrastructure", False),
    (OperationError("tool budget exhausted", kind="tool"), "tool", False),
])
def test_error_categories(exc, kind, retryable):
    failure = classify_error(exc)
    assert (failure.kind, failure.retryable) == (kind, retryable)


def test_plan_database_failure_is_not_reported_as_sql_syntax():
    conn = Mock()
    conn.execute.side_effect = sqlite3.OperationalError("database is locked")
    allowed, message = inspect_query_plan(conn, "SELECT name FROM employees WHERE id=1")
    assert not allowed and "Database error" in message and "busy" in message
    with pytest.raises(sqlite3.OperationalError):
        inspect_query_plan(conn, "SELECT name FROM employees WHERE id=1", raise_sqlite_errors=True)


def test_audit_serialization_failure_is_infrastructure_not_model_format():
    with pytest.raises(OperationError) as failure:
        audit_logger.append_audit_event({"request_id": "bad", "review_history": [object()]})
    assert failure.value.kind == "infrastructure"
    assert not failure.value.retryable


def test_audit_retry_recovers_after_torn_trailing_record():
    audit_logger.AUDIT_PATH.write_text('{"request_id":"torn', encoding="utf-8")
    audit_logger.append_audit_event({"request_id": "restored", "status": "completed"})
    records = audit_logger.read_recent_audit_logs()
    assert len(records) == 1 and records[0]["request_id"] == "restored"
