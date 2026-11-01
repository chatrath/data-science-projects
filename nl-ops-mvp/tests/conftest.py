"""Tests must not upload synthetic test traces with the user's credentials."""
import os
import pytest

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


@pytest.fixture(autouse=True)
def isolated_audit_log(tmp_path, monkeypatch):
    import audit_logger
    monkeypatch.setattr(audit_logger, "AUDIT_PATH", tmp_path / "audit.jsonl")
    import plan_artifacts
    monkeypatch.setattr(plan_artifacts, "ARTIFACT_PATH", tmp_path / "plans")


@pytest.fixture(autouse=True)
def prevent_real_smtp(monkeypatch):
    import smtplib
    from unittest.mock import Mock
    monkeypatch.setattr(smtplib, "SMTP", Mock(side_effect=ConnectionRefusedError("Test SMTP is offline")))
