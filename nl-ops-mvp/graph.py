"""Input -> safety -> SQL -> dry run/execution -> deterministic response."""
import config  # Load .env before LangGraph, including direct graph.invoke usage.

import logging
import sqlite3
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from agents import ModelConfigurationError, check_intent, generate_sql
from database import MAX_ROWS, TABLES, read_only_connection, execute_query, get_schema
from guardrails import inspect_query_plan, validate_sql
from schema_context import get_relevant_schema
from audit_logger import append_audit_event, request_identity
from notifications import send_dba_alert
from plan_artifacts import save_plan_artifacts

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict, total=False):
    request: str
    schema: str
    schema_context: str
    row_limit: int
    status: str
    message: str
    sql: str
    columns: list[str]
    rows: list
    truncated: bool
    plan_message: str
    raw_request: str
    request_id: str
    request_timestamp: str
    safety_status: str
    ast_status: str
    plan_status: str
    execution_status: str
    audit_written: bool
    audit_error: bool
    execution_plan: str
    dba_alert: dict
    plan_links: dict


def input_node(state: WorkflowState) -> dict:
    request = state.get("request", "").strip()
    limit = state.get("row_limit", MAX_ROWS)
    fresh = {"sql": "", "columns": [], "rows": [], "truncated": False, "message": "", "plan_message": "", "schema_context": ""}
    fresh.update(request_identity(state.get("request", "")), safety_status="NotRun", ast_status="NotRun",
                 plan_status="NotRun", execution_status="NotRun", audit_written=False, audit_error=False,
                 execution_plan="", dba_alert={"status": "NotNeeded"}, plan_links={})
    if not request or len(request) > 4000:
        return {**fresh, "status": "blocked", "message": "Enter a request between 1 and 4,000 characters."}
    if type(limit) is not int or not 1 <= limit <= MAX_ROWS:
        return {**fresh, "status": "blocked", "message": f"Row limit must be 1–{MAX_ROWS}."}
    try:
        return {**fresh, "request": request, "row_limit": limit, "schema": get_schema(), "status": "ready"}
    except Exception:
        logger.exception("Database initialization check failed")
        return {**fresh, "status": "error", "message": "Database unavailable. Run python database.py."}


def safety_node(state: WorkflowState) -> dict:
    try:
        decision = check_intent(state["request"], state["schema"])
        return {"status": "approved" if decision.allowed else "blocked", "message": decision.reason,
                "safety_status": "Passed" if decision.allowed else "Blocked"}
    except ModelConfigurationError as exc:
        return {"status": "error", "message": str(exc), "safety_status": "Error"}
    except Exception:
        logger.exception("Safety agent failed")
        return {"status": "error", "safety_status": "Error", "message": "Safety check failed. Check that Ollama is running and OLLAMA_MODEL is installed (ollama list)."}


def sql_node(state: WorkflowState) -> dict:
    try:
        context = get_relevant_schema(state["request"])
        return {"sql": generate_sql(state["request"], context, state["row_limit"]),
                "schema_context": context, "status": "generated"}
    except ModelConfigurationError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception:
        logger.exception("SQL generation failed")
        return {"status": "error", "message": "SQL generation failed. Check Ollama connectivity and OLLAMA_MODEL, then retry."}


def validate_sql_node(state: WorkflowState) -> dict:
    try:
        valid, message = validate_sql(state["sql"], sorted(TABLES))
    except Exception:
        logger.exception("SQL guardrail failed")
        return {"status": "error", "ast_status": "Error", "message": "SQL security validation could not complete. Execution was stopped."}
    return {"status": "validated" if valid else "blocked", "message": message, "ast_status": "Passed" if valid else "Blocked"}


def inspect_query_plan_node(state: WorkflowState) -> dict:
    try:
        details = []
        with read_only_connection() as conn:
            allowed, message = inspect_query_plan(conn, state["sql"], plan_details=details)
        plan = "\n".join(details)
        links = save_plan_artifacts(state["sql"], plan)
        return {"status": "plan_approved" if allowed else "blocked", "message": message, "plan_message": message,
                "plan_status": "Passed" if allowed else "Blocked", "execution_plan": plan, "plan_links": links}
    except Exception:
        logger.exception("Query plan inspection failed")
        return {"status": "error", "plan_status": "Error", "message": "Query plan inspection could not complete. Execution was stopped."}


def execution_node(state: WorkflowState) -> dict:
    try:
        return {**execute_query(state["sql"], state["row_limit"]), "status": "completed", "execution_status": "Success"}
    except (ValueError, sqlite3.Error):
        logger.exception("Query rejected or execution failed")
        return {"status": "blocked", "execution_status": "Blocked", "message": "Query failed SQLite validation, read-only policy, or execution limits. Try rephrasing."}
    except Exception:
        logger.exception("Unexpected database execution error")
        return {"status": "error", "execution_status": "Error", "message": "Database execution failed unexpectedly. Check server logs."}


def response_node(state: WorkflowState) -> dict:
    if state["status"] != "completed":
        return {"message": state.get("message", "Request could not be completed.")}
    suffix = " Results were capped at the row limit." if state["truncated"] else ""
    return {"message": f"Returned {len(state['rows'])} row(s).{suffix}"}


def notify_failure_node(state: WorkflowState) -> dict:
    if state.get("status") not in {"blocked", "error"} or state.get("dba_alert", {}).get("status", "NotNeeded") != "NotNeeded":
        return {}
    try:
        alert = send_dba_alert(state.get("sql", ""), state.get("execution_plan", ""),
                               state.get("message", "Workflow failed."),
                               user_prompt=state.get("raw_request", state.get("request", "")),
                               request_id=state.get("request_id", ""), timestamp=state.get("request_timestamp", ""))
    except Exception:
        logger.warning("DBA notification helper failed; workflow outcome is preserved.")
        alert = {"status": "Failed", "message": "Notification helper failed."}
    return {"dba_alert": alert}


def audit_node(state: WorkflowState) -> dict:
    try:
        append_audit_event(state)
        return {"audit_written": True}
    except (OSError, ValueError, TypeError):
        logger.exception("Enterprise audit persistence failed")
        return {"audit_error": True, "status": "error",
                "message": "Enterprise audit record could not be saved. Check log file permissions and disk space. The query may already have executed."}


def build_graph():
    builder = StateGraph(WorkflowState)
    for name, node in [("input", input_node), ("safety", safety_node), ("sql_generation", sql_node),
                       ("validate_sql_node", validate_sql_node),
                       ("inspect_query_plan_node", inspect_query_plan_node),
                       ("execution", execution_node), ("response", response_node),
                       ("notify_failure", notify_failure_node), ("audit", audit_node),
                       ("notify_audit_failure", notify_failure_node)]:
        builder.add_node(name, node)
    builder.add_edge(START, "input")
    builder.add_conditional_edges("input", lambda s: "safety" if s["status"] == "ready" else "response")
    builder.add_conditional_edges("safety", lambda s: "sql_generation" if s["status"] == "approved" else "response")
    builder.add_conditional_edges("sql_generation", lambda s: "validate_sql_node" if s["status"] == "generated" else "response")
    builder.add_conditional_edges("validate_sql_node", lambda s: "inspect_query_plan_node" if s["status"] == "validated" else "response")
    builder.add_conditional_edges("inspect_query_plan_node", lambda s: "execution" if s["status"] == "plan_approved" else "response")
    builder.add_edge("execution", "response")
    builder.add_edge("response", "notify_failure")
    builder.add_edge("notify_failure", "audit")
    builder.add_edge("audit", "notify_audit_failure")
    builder.add_edge("notify_audit_failure", END)
    return builder.compile()


graph = build_graph()
