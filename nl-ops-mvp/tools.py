"""Read-only tools exposed to native model function calling."""
import config  # Preserve .env-first initialization for direct tool imports too.

import sqlite3
import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from database import TABLES, get_table_metadata as read_table_metadata
from guardrails import validate_sql
from workflow_errors import classify_error


class TableArguments(BaseModel):
    table_name: str = Field(min_length=1, max_length=64, description="Allowed SQLite table name: employees, customers, or tickets.")


class SQLArguments(BaseModel):
    sql_query: str = Field(min_length=1, max_length=10000, description="One SQLite SELECT query to check; the tool never executes it.")


@tool(args_schema=TableArguments)
def get_table_metadata(table_name: str) -> dict:
    """Inspect real columns, SQLite types, primary keys, and row count for one allowed table. Never returns data rows."""
    try:
        return {"ok": True, **read_table_metadata(table_name)}
    except ValueError:
        return {"ok": False, "error": "Table is not allowed or does not exist.", "error_kind": "policy", "retryable": False}
    except (sqlite3.Error, OSError) as exc:
        failure = classify_error(exc, source="metadata tool")
        return {"ok": False, "error": failure.message, "error_kind": failure.kind, "retryable": failure.retryable}


@tool(args_schema=SQLArguments)
def validate_sql_syntax(sql_query: str) -> dict:
    """Lint SQLite syntax with SQLGlot and report SELECT/table policy errors without executing SQL. Call before submitting final SQL."""
    try:
        tree = sqlglot.parse_one(sql_query, dialect="sqlite")
        statements = sqlglot.parse(sql_query, dialect="sqlite")
        if len(statements) != 1 or any(not select.expressions for select in tree.find_all(exp.Select)):
            return {"valid": False, "errors": ["Expected one complete SQL statement."], "policy_allowed": False}
    except (SqlglotError, ValueError, RecursionError) as exc:
        return {"valid": False, "errors": [str(exc)], "policy_allowed": False,
                "error_kind": "sql_syntax", "retryable": False, "sql_query": sql_query}
    allowed, reason = validate_sql(sql_query, sorted(TABLES))
    return {"valid": True, "errors": [], "policy_allowed": allowed, "policy_message": reason,
            "sql_query": sql_query, "error_kind": None if allowed else "policy", "retryable": False}


AGENT_TOOLS = [get_table_metadata, validate_sql_syntax]



def tool_error_payload(exc: Exception) -> str:
    """Make ToolNode invocation failures actionable without inventing results."""
    failure = classify_error(exc, source="tool")
    kind = "tool" if failure.kind in {"model_format", "infrastructure"} and not failure.retryable else failure.kind
    return json.dumps({"ok": False, "error": failure.message,
                       "error_kind": kind, "retryable": failure.retryable})
