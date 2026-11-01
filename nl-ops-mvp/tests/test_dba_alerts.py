"""SMTP formatting and graceful failure; no real messages are sent."""
from contextlib import ExitStack
from unittest.mock import patch
import sqlite3

import pytest

import audit_logger
from agents import SafetyDecision
from database import get_schema, initialize_database, read_only_connection
from graph import graph
from notifications import send_dba_alert
from query_plan import plan_to_xml
from xml.etree import ElementTree as ET


@pytest.fixture(autouse=True)
def smtp_settings(monkeypatch):
    for name, value in {"SMTP_SERVER": "localhost", "SMTP_PORT": "1025", "SMTP_USER": "",
                        "SMTP_PASSWORD": "", "SMTP_FROM": "nl-ops@enterprise-mock.com",
                        "DBA_EMAIL_RECIPIENT": "dba-alerts@enterprise-mock.com", "SMTP_STARTTLS": "false"}.items():
        monkeypatch.setenv(name, value)


def test_email_fields_and_full_context_are_formatted():
    with patch("notifications.smtplib.SMTP") as smtp:
        server = smtp.return_value.__enter__.return_value
        plan = plan_to_xml("SELECT * FROM employees", [(2, 0, 0, "SCAN employees")])
        result = send_dba_alert("SELECT * FROM employees", plan,
                                "Unindexed full table scan detected", user_prompt="Show all employees",
                                request_id="request-123", timestamp="2026-11-01T02:00:00+00:00")
    assert result["status"] == "Sent"
    smtp.assert_called_once_with("localhost", 1025, timeout=5)
    message = server.send_message.call_args.args[0]
    assert message["To"] == "dba-alerts@enterprise-mock.com"
    assert message["From"] == "nl-ops@enterprise-mock.com"
    assert "failed or blocked" in message["Subject"]
    body = message.get_body(preferencelist=("plain",)).get_content()
    for text in ["SELECT * FROM employees", "Download SQL Plan", "Unindexed full table scan detected",
                 "Show all employees", "request-123", "2026-11-01T02:00:00+00:00"]:
        assert text in body
    server.login.assert_not_called()
    assert not list(message.iter_attachments())
    assert "<SQLiteQueryPlan" not in body
    assert '/plans/' in body
    html = message.get_body(preferencelist=("html",)).get_content()
    assert '<a href="http' in html
    assert "<SQLiteQueryPlan" not in html


def test_blank_credentials_and_connection_failure_are_safe(caplog):
    with patch("notifications.smtplib.SMTP", side_effect=ConnectionRefusedError("offline")):
        result = send_dba_alert("SELECT * FROM employees", "SCAN employees", "Unindexed scan")
    assert result["status"] == "Failed"
    assert "query remains blocked" in caplog.text


def test_remote_missing_credentials_skip_without_connecting(monkeypatch):
    monkeypatch.setenv("SMTP_SERVER", "smtp.example.com")
    with patch("notifications.smtplib.SMTP") as smtp:
        result = send_dba_alert("SELECT 1", "SCAN", "Blocked")
    assert result["status"] == "Skipped"
    smtp.assert_not_called()


def test_remote_authenticated_delivery_uses_starttls(monkeypatch):
    monkeypatch.setenv("SMTP_SERVER", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "test-user")
    monkeypatch.setenv("SMTP_PASSWORD", "test-password")
    monkeypatch.delenv("SMTP_STARTTLS")
    with patch("notifications.smtplib.SMTP") as smtp:
        server = smtp.return_value.__enter__.return_value
        assert send_dba_alert("SELECT 1", "SCAN", "Blocked")["status"] == "Sent"
    server.starttls.assert_called_once()
    server.login.assert_called_once_with("test-user", "test-password")


def test_invalid_port_is_handled(monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "not-a-port")
    assert send_dba_alert("SELECT 1", "SCAN", "Blocked")["status"] == "Failed"


@pytest.mark.parametrize("sql,should_alert", [
    ("SELECT * FROM employees", True),
    ("SELECT name FROM employees WHERE salary>0 ORDER BY name", True),
    ("SELECT name FROM employees WHERE id=1", False),
    ("SELECT missing_column FROM employees", True),
    ("DROP TABLE employees", True),
])
def test_graph_notifies_for_all_rejections(tmp_path, sql, should_alert):
    path = tmp_path / "alerts.db"
    initialize_database(path)
    with patch("graph.get_schema", side_effect=lambda: get_schema(path)), \
         patch("graph.read_only_connection", side_effect=lambda: read_only_connection(path)), \
         patch("graph.check_intent", return_value=SafetyDecision(allowed=True, reason="Read only")), \
         patch("graph.generate_sql", return_value=sql), \
         patch("graph.execute_query", return_value={"rows": [], "columns": [], "truncated": False}) as execute, \
         patch("graph.send_dba_alert", return_value={"status": "Failed", "message": "SMTP offline"}) as notify:
        result = graph.invoke({"request": "Prompt context for DBA"})
    if should_alert:
        assert result["status"] == "blocked"
        notify.assert_called_once()
        execute.assert_not_called()
        args = notify.call_args
        assert args.args[0] == sql
        if result["execution_plan"]:
            assert "SCAN employees" in args.args[1]
            assert ET.fromstring(args.args[1]).find(".//Operator").get("parentId") is not None
        assert args.kwargs["user_prompt"] == "Prompt context for DBA"
        assert args.kwargs["timestamp"]
        record = audit_logger.read_recent_audit_logs()[0]
        assert record["dba_alert"]["status"] == "Failed"
        assert record["execution_outcome"] == "Blocked-by-Guardrail"
        assert "execution_plan" not in record
        if result["execution_plan"]:
            assert record["plan_links"]["plan_url"].endswith(".xml")
    else:
        notify.assert_not_called()


@pytest.mark.parametrize("failure", ["input", "safety_block", "safety_error", "sql_error", "ast_error",
                                     "plan_error", "database_error", "unexpected_database_error", "audit_error", "notification_error"])
def test_each_failure_alerts_once(tmp_path, failure):
    path = tmp_path / "failures.db"
    initialize_database(path)
    with ExitStack() as stack:
        stack.enter_context(patch("graph.get_schema", side_effect=lambda: get_schema(path)))
        stack.enter_context(patch("graph.read_only_connection", side_effect=lambda: read_only_connection(path)))
        stack.enter_context(patch("graph.check_intent", return_value=SafetyDecision(allowed=True, reason="Read only")))
        stack.enter_context(patch("graph.generate_sql", return_value="SELECT name FROM employees WHERE id=1"))
        stack.enter_context(patch("graph.execute_query", return_value={"rows": [["Alex Morgan"]], "columns": ["name"], "truncated": False}))
        notify = stack.enter_context(patch("graph.send_dba_alert", return_value={"status": "Sent"}))
        request = "Show employee 1"
        if failure == "input":
            request = ""
        elif failure == "safety_block":
            stack.enter_context(patch("graph.check_intent", return_value=SafetyDecision(allowed=False, reason="Unsafe")))
        elif failure in {"safety_error", "sql_error", "ast_error", "plan_error", "audit_error"}:
            target = {"safety_error": "check_intent", "sql_error": "generate_sql", "ast_error": "validate_sql",
                      "plan_error": "inspect_query_plan", "audit_error": "append_audit_event"}[failure]
            error = OSError("Disk unavailable") if failure == "audit_error" else RuntimeError("Test failure")
            stack.enter_context(patch("graph." + target, side_effect=error))
        elif failure in {"database_error", "unexpected_database_error"}:
            error = sqlite3.OperationalError("Test failure") if failure == "database_error" else RuntimeError("Test failure")
            stack.enter_context(patch("graph.execute_query", side_effect=error))
        elif failure == "notification_error":
            request = ""
            notify.side_effect = RuntimeError("SMTP unavailable")
        result = graph.invoke({"request": request})
    assert result["status"] in {"blocked", "error"}
    notify.assert_called_once()
    assert result["dba_alert"]["status"] == ("Failed" if failure == "notification_error" else "Sent")


def test_missing_plan_and_sql_are_explained_in_email():
    with patch("notifications.smtplib.SMTP") as smtp:
        result = send_dba_alert("", "", "Intent safety check failed")
    assert result["status"] == "Sent"
    message = smtp.return_value.__enter__.return_value.send_message.call_args.args[0]
    body = message.get_body(preferencelist=("plain",)).get_content()
    assert "(not generated)" in body
    assert "SQL Plan: unavailable" in body
    assert "Download SQL Plan:" not in body
    assert '<a href=""' not in message.get_body(preferencelist=("html",)).get_content()


def test_artifact_failure_does_not_stop_email():
    with patch("notifications.save_plan_artifacts", side_effect=OSError("Disk full")), patch("notifications.smtplib.SMTP") as smtp:
        assert send_dba_alert("SELECT 1", "plan", "Audit disk error")["status"] == "Sent"
    smtp.return_value.__enter__.return_value.send_message.assert_called_once()
