"""Durable JSON-lines audit records for this local, single-process MVP."""
from collections import deque
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from uuid import uuid4
from plan_artifacts import save_plan_artifacts

AUDIT_PATH = Path(__file__).resolve().parent / "enterprise_audit.log"
_lock = threading.RLock()


def request_identity(user_query: str) -> dict:
    return {"request_id": str(uuid4()), "request_timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_request": user_query}


def append_audit_event(state: dict, *, cached: bool = False, path=None) -> dict:
    """Append exactly one JSON record; propagate write errors to the caller."""
    status = state.get("status", "error")
    safety = state.get("safety_status", "NotRun")
    ast = state.get("ast_status", "NotRun")
    plan = state.get("plan_status", "NotRun")
    if cached:
        outcome = "Cached"
    elif status == "completed":
        outcome = "Success"
    elif safety == "Blocked":
        outcome = "Blocked-by-Safety"
    elif "Blocked" in (ast, plan) or state.get("execution_status") == "Blocked":
        outcome = "Blocked-by-Guardrail"
    elif status == "blocked":
        outcome = "Invalid-Request"
    else:
        outcome = "Error"
    record = {
        "timestamp": state.get("request_timestamp", datetime.now(timezone.utc).isoformat()),
        "request_id": state.get("request_id", str(uuid4())),
        "user_query": state.get("raw_request", state.get("request", "")),
        "safety_status": safety,
        "sql_generated": state.get("sql", ""),
        "guardrail_status": {"ast": ast, "explain": plan},
        "execution_outcome": outcome,
        "row_count": len(state.get("rows", [])) if status == "completed" else 0,
        "cache_source": "result cache" if cached else state.get("cache_source", "workflow"),
        "message": state.get("message", ""),
        "plan_links": state.get("plan_links") or save_plan_artifacts(state.get("sql", ""), state.get("execution_plan", "")),
        "dba_alert": state.get("dba_alert", {"status": "NotNeeded"}),
    }
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    with _lock, Path(path or AUDIT_PATH).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return record


def read_recent_audit_logs(limit: int = 100, path=None) -> list[dict]:
    if not 1 <= limit <= 1000:
        raise ValueError("Audit display limit must be between 1 and 1000.")
    target = Path(path or AUDIT_PATH)
    with _lock:
        if not target.exists():
            return []
        with target.open(encoding="utf-8") as handle:
            lines = deque(handle, maxlen=limit)
    records = []
    for line in reversed(lines):
        try:
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
        except json.JSONDecodeError:
            logging.getLogger(__name__).warning("Skipped a malformed audit log line.")
    return records
