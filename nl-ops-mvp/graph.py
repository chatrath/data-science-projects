"""Input -> safety -> SQL -> dry run/execution -> deterministic response."""
import config  # Load .env before LangGraph, including direct graph.invoke usage.

import logging
import sqlite3
import json
from pathlib import Path
from functools import lru_cache
from threading import RLock
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt
from typing import Annotated, TypedDict, Literal

from sqlglot.errors import ParseError

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from agents import ModelConfigurationError, ModelOutputError, check_intent, generate_sql, get_tool_model
from database import MAX_ROWS, TABLES, read_only_connection, execute_query, get_schema
from guardrails import inspect_query_plan, validate_sql
from schema_context import get_relevant_schema
from audit_logger import append_audit_event, request_identity
from notifications import send_dba_alert
from plan_artifacts import save_plan_artifacts
from tools import AGENT_TOOLS, tool_error_payload
from workflow_errors import OperationError, classify_error, is_sql_correction
from langchain_core.messages import messages_to_dict

logger = logging.getLogger(__name__)

MAX_AGENT_TURNS = 6
MAX_TOOL_CALLS = 8


class ToolAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turns: int
    tool_calls: int
    required_tools: list[str]
    retry: bool
    error: str
    error_kind: str
    error_retryable: bool


def _successful_tools(messages):
    completed = set()
    for message in messages:
        if not isinstance(message, ToolMessage) or message.status == "error":
            continue
        try:
            payload = json.loads(message.content)
            if message.name == "get_table_metadata" and payload.get("ok"):
                completed.add(message.name)
            if message.name == "validate_sql_syntax" and payload.get("valid") and payload.get("policy_allowed"):
                completed.add(message.name)
        except (TypeError, ValueError, AttributeError):
            continue
    return completed


def tool_agent_node(state: ToolAgentState) -> dict:
    if state["turns"] >= MAX_AGENT_TURNS:
        return {"error": "Agent exceeded the bounded tool-calling turn limit."}
    try:
        response = get_tool_model().invoke(state["messages"])
    except Exception as exc:
        failure = classify_error(exc, source="model")
        return {"error": failure.message, "error_kind": failure.kind, "error_retryable": failure.retryable}
    if not isinstance(response, AIMessage) or response.invalid_tool_calls:
        return {"error": "Agent returned malformed tool calls or an invalid response."}
    calls = state["tool_calls"] + len(response.tool_calls)
    if calls > MAX_TOOL_CALLS or len(response.tool_calls) > 4:
        return {"error": "Agent exceeded the tool-call budget."}
    messages = [response]
    missing = set(state["required_tools"]) - _successful_tools(state["messages"])
    retry = not response.tool_calls and bool(missing)
    if retry:
        messages.append(HumanMessage(content="Before a final answer, successfully invoke these required tools: "
                                             + ", ".join(sorted(missing)) + ". Use their results; do not guess."))
    return {"messages": messages, "turns": state["turns"] + 1, "tool_calls": calls, "retry": retry}


def _tool_route(state: ToolAgentState):
    if state.get("error"):
        return END
    if state.get("retry"):
        return "agent"
    return tools_condition(state)


def build_tool_agent_graph():
    builder = StateGraph(ToolAgentState)
    builder.add_node("agent", tool_agent_node)
    builder.add_node("tools", ToolNode(AGENT_TOOLS, handle_tool_errors=tool_error_payload))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _tool_route)
    builder.add_edge("tools", "agent")
    return builder.compile()


tool_agent_graph = build_tool_agent_graph()


def run_agent_tools(messages, run_name, required_tools=(), *, tool_state=None, include_history=False):
    """Keep verified tool transcripts and counters across recovery attempts."""
    initial = {"messages": messages, "turns": 0, "tool_calls": 0,
               "required_tools": list(required_tools), "retry": False, "error": ""}
    if tool_state:
        from langchain_core.messages import messages_from_dict
        initial.update(tool_state)
        initial["messages"] = messages_from_dict(tool_state["messages"])
        initial.update(error="", error_kind="", error_retryable=False)
    result = tool_agent_graph.invoke(initial, config={"run_name": run_name, "recursion_limit": 25})
    serialized = {**result, "messages": messages_to_dict(result["messages"])}
    if result.get("error"):
        raise OperationError(result["error"], kind=result.get("error_kind") or "tool",
                             retryable=result.get("error_retryable", False), tool_state=serialized)
    final = result["messages"][-1]
    if not isinstance(final, AIMessage) or final.tool_calls:
        raise OperationError("Agent did not finish with a final response.", kind="model_format", tool_state=serialized)
    return (final, result["messages"], serialized) if include_history else final


class WorkflowState(TypedDict, total=False):
    run_state: Literal["running", "awaiting_approval", "failed_recoverable", "finished"]
    error_kind: str
    error_message: str
    error_history: list[dict]
    failed_node: str
    recovery_attempts: int
    recovery_token: str
    resume_status: str
    resume_message: str
    agent_tool_state: dict
    approval_status: str
    approved_sql: str
    approval_token: str
    review_count: int
    review_history: list[dict]
    request: str
    schema: str
    schema_context: str
    row_limit: int
    status: str
    message: str
    sql: str
    sql_syntax_error: str
    correction_attempts: int
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
    fresh = {"run_state": "running", "error_kind": "", "error_message": "", "error_history": [], "failed_node": "", "recovery_attempts": 0, "recovery_token": "", "agent_tool_state": {}, "approval_status": "pending", "approved_sql": "", "approval_token": "", "review_count": 0, "review_history": [], "sql_syntax_error": "", "correction_attempts": 0, "sql": "", "columns": [], "rows": [], "truncated": False, "message": "", "plan_message": "", "schema_context": ""}
    if state.get("recovery_attempts", 0):
        # Retrying initial DB setup must retain the global recovery budget.
        for key in ("recovery_attempts", "error_history"):
            fresh[key] = state.get(key, fresh[key])
    identity = request_identity(state.get("request", ""))
    for key in ("request_id", "request_timestamp"):
        if state.get(key):
            identity[key] = state[key]
    fresh.update(identity, safety_status="NotRun", ast_status="NotRun",
                 plan_status="NotRun", execution_status="NotRun", audit_written=False, audit_error=False,
                 execution_plan="", dba_alert={"status": "NotNeeded"}, plan_links={})
    if not request or len(request) > 4000:
        return {**fresh, "status": "blocked", "message": "Enter a request between 1 and 4,000 characters."}
    if type(limit) is not int or not 1 <= limit <= MAX_ROWS:
        return {**fresh, "status": "blocked", "message": f"Row limit must be 1–{MAX_ROWS}."}
    try:
        return {**fresh, "request": request, "row_limit": limit, "schema": get_schema(), "status": "ready"}
    except Exception as exc:
        logger.exception("Database initialization check failed")
        return {**fresh, **node_failure({**state, **fresh}, "input", exc)}


def safety_node(state: WorkflowState) -> dict:
    try:
        evidence = {"tool_state": state["agent_tool_state"]} if state.get("agent_tool_state") else {}
        decision = check_intent(state["request"], state["schema"], **evidence)
        return {"status": "approved" if decision.allowed else "blocked", "message": decision.reason,
                "safety_status": "Passed" if decision.allowed else "Blocked", "agent_tool_state": {}}
    except (ModelConfigurationError, ModelOutputError) as exc:
        return {**node_failure(state, "safety", exc), "safety_status": "Error"}
    except Exception as exc:
        logger.exception("Safety agent failed")
        return {**node_failure(state, "safety", exc), "safety_status": "Error"}


def sql_node(state: WorkflowState) -> dict:
    try:
        context = state.get("schema_context") or get_relevant_schema(state["request"])
        correction = {"tool_state": state["agent_tool_state"]} if state.get("agent_tool_state") else {}
        if state.get("sql_syntax_error"):
            correction.update(previous_sql=state["sql"], sql_syntax_error=state["sql_syntax_error"])
        return {"sql": generate_sql(state["request"], context, state["row_limit"], **correction),
                "schema_context": context, "status": "generated", "approved_sql": "", "approval_status": "pending", "agent_tool_state": {}, "error_kind": "", "error_message": ""}
    except (ModelConfigurationError, ModelOutputError) as exc:
        return node_failure(state, "sql_generation", exc)
    except Exception as exc:
        logger.exception("SQL generation failed")
        return node_failure(state, "sql_generation", exc)


def validate_sql_node(state: WorkflowState) -> dict:
    try:
        valid, message = validate_sql(state["sql"], sorted(TABLES), raise_parse_errors=True)
    except ParseError as exc:
        attempts = state.get("correction_attempts", 0) + 1
        return {"sql_syntax_error": str(exc), "correction_attempts": attempts,
                "error_kind": "sql_syntax", "error_message": str(exc),
                "error_history": [*state.get("error_history", []), {"node": "validate_sql_node", "kind": "sql_syntax", "message": str(exc)}],
                "status": "error" if attempts >= 3 else "syntax_retry", "ast_status": "Error",
                "message": ("SQL syntax validation failed after 3 attempts. " if attempts >= 3 else "") + str(exc)}
    except Exception as exc:
        logger.exception("SQL guardrail failed")
        return {**node_failure(state, "validate_sql_node", exc), "ast_status": "Error"}
    return {"status": "validated" if valid else "blocked", "message": message, "ast_status": "Passed" if valid else "Blocked",
            "sql_syntax_error": "", "error_kind": "" if valid else "policy"}


def validation_route(state: WorkflowState) -> str:
    if state["status"] == "failed_recoverable":
        return "recovery"
    if state["status"] == "validated":
        return "inspect_query_plan_node"
    if state["status"] == "syntax_retry" and state.get("correction_attempts", 0) < 3:
        return "sql_generation"
    return "response"


def sql_error_update(state: WorkflowState, diagnostic: str, stage: str) -> dict:
    """Bound semantic/database corrections and invalidate any earlier approval."""
    attempts = state.get("correction_attempts", 0) + 1
    return {"status": "syntax_retry" if attempts < 3 else "error", stage: "Error",
            "sql_syntax_error": diagnostic, "correction_attempts": attempts,
            "error_kind": "database", "error_message": diagnostic,
            "error_history": [*state.get("error_history", []), {"node": stage, "kind": "database", "message": diagnostic}],
            "approved_sql": "", "approval_status": "pending",
            "message": diagnostic if attempts < 3 else "SQL correction limit reached. " + diagnostic}

def inspect_query_plan_node(state: WorkflowState) -> dict:
    try:
        details = []
        with read_only_connection() as conn:
            allowed, message = inspect_query_plan(conn, state["sql"], plan_details=details, raise_sqlite_errors=True)
        plan = "\n".join(details)
        links = save_plan_artifacts(state["sql"], plan)
        return {"status": "plan_approved" if allowed else "blocked", "message": message, "plan_message": message,
                "plan_status": "Passed" if allowed else "Blocked", "execution_plan": plan, "plan_links": links,
                "error_kind": "" if allowed else "performance"}
    except sqlite3.OperationalError as exc:
        return sql_error_update(state, str(exc), "plan_status") if is_sql_correction(exc) else node_failure(state, "inspect_query_plan_node", exc)
    except Exception as exc:
        logger.exception("Query plan inspection failed")
        return {**node_failure(state, "inspect_query_plan_node", exc), "plan_status": "Error"}


def execution_node(state: WorkflowState) -> dict:
    if state.get("approval_status") != "approved" or state.get("approved_sql") != state.get("sql"):
        return {"status": "blocked", "execution_status": "Blocked", "message": "Explicit approval of this SQL is required."}
    try:
        # Recheck after the human pause: the database or policy may have changed.
        valid, reason = validate_sql(state["sql"], sorted(TABLES))
        if not valid:
            return {"status": "blocked", "execution_status": "Blocked", "message": reason}
        with read_only_connection() as conn:
            valid, reason = inspect_query_plan(conn, state["sql"], raise_sqlite_errors=True)
        if not valid:
            return {"status": "blocked", "execution_status": "Blocked", "message": reason}
        return {**execute_query(state["sql"], state["row_limit"]), "status": "completed", "execution_status": "Success"}
    except sqlite3.OperationalError as exc:
        return sql_error_update(state, str(exc), "execution_status") if is_sql_correction(exc) else node_failure(state, "execution", exc)
    except (ValueError, sqlite3.Error) as exc:
        logger.exception("Query rejected or execution failed")
        return {**node_failure(state, "execution", exc), "execution_status": "Error"}
    except Exception as exc:
        logger.exception("Unexpected database execution error")
        return {**node_failure(state, "execution", exc), "execution_status": "Error"}


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
    if state.get("audit_written"):
        return {}
    try:
        append_audit_event(state)
        return {"audit_written": True, "audit_error": False}
    except (OSError, ValueError, TypeError) as exc:
        logger.exception("Enterprise audit persistence failed")
        return {**node_failure(state, "audit", exc), "audit_error": True}


def schema_retrieval_node(state: WorkflowState) -> dict:
    """Scope trusted schema metadata before generation; failures stop safely."""
    try:
        return {"schema_context": get_relevant_schema(state["request"]), "status": "schema_ready"}
    except Exception as exc:
        logger.exception("Schema retrieval failed")
        return node_failure(state, "schema_retrieval", exc)


def prepare_approval_node(state: WorkflowState) -> dict:
    """Persist a new review revision before interrupting, with no external effects."""
    return {"run_state": "awaiting_approval", "status": "awaiting_approval", "approval_status": "pending", "approved_sql": "",
            "approval_token": str(uuid4()), "message": "Review the SQL before execution."}


def approval_node(state: WorkflowState) -> dict:
    """Pause for an explicit decision. This node restarts on resume.

    Never catch GraphInterrupt or perform IO before interrupt(). Untrusted edits
    go back through AST and plan validation and require a separate new approval.
    """
    decision = interrupt({"request_id": state["request_id"], "sql": state["sql"],
                          "plan": state.get("execution_plan", ""),
                          "approval_token": state["approval_token"],
                          "actions": ["approve", "reject", "edit"]})
    if not isinstance(decision, dict) or decision.get("approval_token") != state["approval_token"]:
        return {"status": "rejected", "approval_status": "rejected", "message": "Invalid or stale approval. Submit a new request."}
    action = decision.get("action")
    history = [*state.get("review_history", []),
               {"action": action if action in {"approve", "reject", "edit"} else "invalid",
                "sql": state["sql"], "approval_token": state["approval_token"]}]
    if action == "approve":
        return {"run_state": "running", "status": "human_approved", "approval_status": "approved",
                "approved_sql": state["sql"], "review_history": history}
    if action == "edit":
        sql = decision.get("sql")
        if not isinstance(sql, str) or not sql.strip() or len(sql) > 10000 or state.get("review_count", 0) >= 3:
            return {"status": "rejected", "approval_status": "rejected", "review_history": history,
                    "message": "SQL revision is invalid or the three-edit limit was reached."}
        return {"run_state": "running", "sql": sql.strip(), "status": "generated", "approval_status": "pending",
                "approved_sql": "", "approval_token": "", "review_history": history,
                "review_count": state.get("review_count", 0) + 1,
                "ast_status": "NotRun", "plan_status": "NotRun", "execution_plan": "", "plan_links": {}}
    return {"status": "rejected", "approval_status": "rejected", "review_history": history,
            "message": "Request rejected. No query was executed."}


MAX_RECOVERY_ATTEMPTS = 3


def node_failure(state: WorkflowState, node: str, exc: Exception) -> dict:
    """Checkpoint a categorized failure without losing previous successful work."""
    failure = classify_error(exc, source=node)
    exhausted = state.get("recovery_attempts", 0) >= MAX_RECOVERY_ATTEMPTS
    recoverable = failure.retryable and not exhausted
    detail = {"node": node, "kind": failure.kind, "message": failure.message}
    return {"status": "failed_recoverable" if recoverable else "error",
            "run_state": "failed_recoverable" if recoverable else "running",
            "error_kind": failure.kind, "error_message": failure.message,
            "message": failure.message + (" Recovery limit reached." if exhausted else ""),
            "error_history": [*state.get("error_history", []), detail],
            "failed_node": node, "resume_status": state.get("status", "ready"),
            "resume_message": state.get("message", ""),
            "recovery_token": str(uuid4()) if recoverable else "",
            "agent_tool_state": getattr(exc, "tool_state", None) or state.get("agent_tool_state", {})}


def recovery_node(state: WorkflowState) -> dict:
    """Await retry/cancel separately from SQL approval; no side effects precede pause."""
    decision = interrupt({"kind": "recovery", "request_id": state["request_id"],
                          "node": state["failed_node"], "error_kind": state["error_kind"],
                          "message": state["error_message"], "recovery_token": state["recovery_token"],
                          "attempts": state.get("recovery_attempts", 0), "actions": ["retry", "cancel"]})
    valid = isinstance(decision, dict) and decision.get("recovery_token") == state["recovery_token"]
    if valid and decision.get("action") == "retry" and state.get("recovery_attempts", 0) < MAX_RECOVERY_ATTEMPTS:
        return {"run_state": "running", "status": state["resume_status"],
                "message": state.get("resume_message", ""),
                "recovery_attempts": state.get("recovery_attempts", 0) + 1,
                "error_kind": "", "error_message": "", "recovery_token": ""}
    return {"run_state": "running", "status": "error", "recovery_token": "",
            "message": "Recovery cancelled. " + state["error_message"]}


def recovery_route(state: WorkflowState) -> str:
    if state["status"] == "error":
        return "notify_audit_failure" if state["failed_node"] == "audit" else "response"
    return state["failed_node"]


def finalize_node(state: WorkflowState) -> dict:
    """Terminal outcome is distinct from the lifecycle of the request."""
    return {"run_state": "finished"}


def build_graph(checkpointer=None):
    """Build the single canonical flow. Tests can supply MemorySaver.

    Always pass configurable.thread_id. Each new submission gets its own thread;
    a resume uses Command(resume=...) with that same ID, never fresh input state.
    """
    builder = StateGraph(WorkflowState)
    for name, node in [("input", input_node), ("safety", safety_node),
                       ("schema_retrieval", schema_retrieval_node), ("sql_generation", sql_node),
                       ("validate_sql_node", validate_sql_node), ("inspect_query_plan_node", inspect_query_plan_node),
                       ("prepare_approval", prepare_approval_node), ("approval", approval_node),
                       ("execution", execution_node), ("response", response_node),
                       ("notify_failure", notify_failure_node), ("audit", audit_node),
                       ("notify_audit_failure", notify_failure_node), ("recovery", recovery_node), ("finalize", finalize_node)]:
        builder.add_node(name, node)
    builder.add_edge(START, "input")
    builder.add_conditional_edges("input", lambda s: "recovery" if s["status"] == "failed_recoverable" else "safety" if s["status"] == "ready" else "response")
    builder.add_conditional_edges("safety", lambda s: "recovery" if s["status"] == "failed_recoverable" else "schema_retrieval" if s["status"] == "approved" else "response")
    builder.add_conditional_edges("schema_retrieval", lambda s: "recovery" if s["status"] == "failed_recoverable" else "sql_generation" if s["status"] == "schema_ready" else "response")
    builder.add_conditional_edges("sql_generation", lambda s: "recovery" if s["status"] == "failed_recoverable" else "validate_sql_node" if s["status"] == "generated" else "response")
    builder.add_conditional_edges("validate_sql_node", validation_route)
    builder.add_conditional_edges("inspect_query_plan_node", lambda s: "recovery" if s["status"] == "failed_recoverable" else "prepare_approval" if s["status"] == "plan_approved" else "sql_generation" if s["status"] == "syntax_retry" else "response")
    builder.add_edge("prepare_approval", "approval")
    builder.add_conditional_edges("approval", lambda s: "execution" if s["status"] == "human_approved" else
                                  "validate_sql_node" if s["status"] == "generated" else "response")
    builder.add_conditional_edges("execution", lambda s: "recovery" if s["status"] == "failed_recoverable" else "sql_generation" if s["status"] == "syntax_retry" else "response")
    builder.add_edge("response", "notify_failure")
    builder.add_edge("notify_failure", "audit")
    builder.add_conditional_edges("audit", lambda s: "recovery" if s["status"] == "failed_recoverable" else "notify_audit_failure")
    builder.add_edge("notify_audit_failure", "finalize")
    builder.add_conditional_edges("recovery", recovery_route)
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


@lru_cache(maxsize=1)
def persistent_checkpointer() -> SqliteSaver:
    """Single-process durable SQLite store shared across Streamlit reruns."""
    connection = sqlite3.connect(Path(__file__).with_name(".workflow_checkpoints.db"),
                                 check_same_thread=False, timeout=30)
    return SqliteSaver(connection)


graph = build_graph(persistent_checkpointer())
# Compatibility name only: there is no second workflow or second state schema.
enterprise_graph = graph
_run_lock = RLock()


def request_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 64,
            "run_name": "NL-Ops Helpdesk Request"}


def start_request(request: str, row_limit: int = MAX_ROWS, *, thread_id: str | None = None, engine=None) -> tuple[str, dict]:
    """Start once; a repeated thread ID returns its checkpoint without replaying IO."""
    engine = graph if engine is None else engine
    thread_id = thread_id or str(uuid4())
    cfg = request_config(thread_id)
    with _run_lock:
        snapshot = engine.get_state(cfg)
        if snapshot.values:
            return thread_id, dict(snapshot.values)
        identity = request_identity(request)
        identity["request_id"] = thread_id
        _drive(engine, {**identity, "request": request, "row_limit": row_limit}, cfg)
        return thread_id, dict(engine.get_state(cfg).values)


def resume_request(thread_id: str, action: str, approval_token: str, *, sql: str | None = None, engine=None) -> dict:
    """Resume only a currently interrupted approval; stale/double submits do no IO.

    This local runner serializes decisions in one server process. Deployments with
    multiple workers need request ownership/authentication and a distributed lock.
    """
    engine = graph if engine is None else engine
    cfg = request_config(thread_id)
    with _run_lock:
        snapshot = engine.get_state(cfg)
        if not snapshot.values:
            raise ValueError("Request was not found.")
        pending = any(task.interrupts for task in snapshot.tasks)
        if not pending:
            return dict(snapshot.values)
        if snapshot.values.get("run_state", snapshot.values.get("status")) != "awaiting_approval":
            raise ValueError("This request is not awaiting SQL approval. Use recovery instead.")
        if snapshot.values.get("approval_token") != approval_token:
            raise ValueError("This approval is out of date. Review the current SQL.")
        if action not in {"approve", "reject", "edit"}:
            raise ValueError("Choose approve, reject, or edit.")
        _drive(engine, Command(resume={"action": action, "approval_token": approval_token, "sql": sql}), cfg)
        return dict(engine.get_state(cfg).values)


_RECOVERABLE_NODES = {"input", "safety", "schema_retrieval", "sql_generation", "validate_sql_node",
                      "inspect_query_plan_node", "execution", "audit"}


def _drive(engine, payload, cfg: dict) -> None:
    """Convert an escaped task failure into the same durable recovery protocol.

    GraphInterrupt is not an Exception and must propagate to LangGraph. If the
    checkpoint store itself is unavailable, do not claim that a failure was saved.
    """
    try:
        engine.invoke(payload, cfg)
    except Exception as exc:
        snapshot = engine.get_state(cfg)
        failed = next((task for task in snapshot.tasks if task.error and task.name in _RECOVERABLE_NODES), None)
        if failed is None:
            raise
        engine.update_state(cfg, node_failure(dict(snapshot.values), failed.name, exc), as_node=failed.name)
        engine.invoke(None, cfg)  # Route the recorded failure; do not replay the failed IO.


def get_request_state(thread_id: str, *, engine=None) -> dict:
    """Read progress without executing nodes or mistaking a pause for completion."""
    engine = graph if engine is None else engine
    snapshot = engine.get_state(request_config(thread_id))
    if not snapshot.values:
        raise ValueError("Request was not found.")
    state = dict(snapshot.values)
    state.setdefault("run_state", "awaiting_approval" if state.get("status") == "awaiting_approval" else
                     "finished" if not snapshot.next else "running")
    if snapshot.next and not any(task.interrupts for task in snapshot.tasks):
        state["run_state"] = "failed_recoverable"
        state["recovery_mode"] = "checkpoint"
        state["message"] = "Processing stopped before completion. Resume the saved checkpoint."
    return state


def recover_request(thread_id: str, action: str = "retry", recovery_token: str = "", *, engine=None) -> dict:
    """Retry/cancel a saved failure. This cannot approve SQL or restart a terminal run."""
    engine = graph if engine is None else engine
    cfg = request_config(thread_id)
    if action not in {"retry", "cancel"}:
        raise ValueError("Choose retry or cancel.")
    with _run_lock:
        snapshot = engine.get_state(cfg)
        if not snapshot.values:
            raise ValueError("Request was not found.")
        if not snapshot.next:
            return dict(snapshot.values)
        if any(task.interrupts for task in snapshot.tasks):
            if snapshot.values.get("run_state") != "failed_recoverable":
                raise ValueError("SQL approval is required. Recovery cannot approve this query.")
            if not recovery_token or recovery_token != snapshot.values.get("recovery_token"):
                raise ValueError("This recovery decision is out of date.")
            _drive(engine, Command(resume={"action": action, "recovery_token": recovery_token}), cfg)
        elif action == "retry":
            # A process stopped between nodes (or an uncaught task failed). Invoke
            # None, never initial input, to keep successful nodes and writes intact.
            _drive(engine, None, cfg)
        else:
            raise ValueError("Resume the interrupted checkpoint before cancelling its task.")
        return dict(engine.get_state(cfg).values)
