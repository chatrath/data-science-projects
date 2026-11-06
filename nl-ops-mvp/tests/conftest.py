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


def invoke_reviewed(engine, payload, config=None):
    """Test-only reviewer: explicitly approve successful requests at each gate."""
    from uuid import uuid4
    from graph import request_config
    from langgraph.types import Command
    cfg = request_config(str(uuid4()))
    if config:
        cfg.update({key: value for key, value in config.items() if key != "configurable"})
    result = engine.invoke(payload, cfg)
    for _ in range(4):
        snapshot = engine.get_state(cfg)
        if not any(task.interrupts for task in snapshot.tasks):
            return dict(snapshot.values)
        if snapshot.values.get("run_state") == "failed_recoverable":
            decision = {"action": "cancel", "recovery_token": snapshot.values["recovery_token"]}
        else:
            decision = {"action": "approve", "approval_token": snapshot.values["approval_token"]}
        result = engine.invoke(Command(resume=decision), cfg)
    raise AssertionError("Unexpected repeated approval gates")
