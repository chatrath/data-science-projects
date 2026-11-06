"""Fail-closed AST policy; SQLite authorization remains the execution boundary."""
import sqlglot
import re
import sqlite3
from sqlglot import exp
from sqlglot.errors import ParseError, SqlglotError
from query_plan import plan_to_xml
from workflow_errors import classify_error

SCAN_REJECTION = "Query rejected: Estimated plan requires an unindexed full-table scan."
SORT_REJECTION = "Query rejected: Estimated plan requires a temporary b-tree sort. Add an appropriate index or simplify sorting/grouping."
PERFORMANCE_REJECTIONS = {SCAN_REJECTION, SORT_REJECTION}


def validate_sql(sql_query: str, allowed_tables: list, *, raise_parse_errors: bool = False) -> tuple[bool, str]:
    """Allow one SQLite SELECT over explicitly permitted physical tables.

    Inspect all tables, including joins and nested subqueries. CTEs and set
    operations are deliberately unsupported, consistent with the MVP policy.
    This parser is not a substitute for SQLite's own syntax and access checks.
    """
    if not isinstance(sql_query, str) or not sql_query.strip():
        return False, "Invalid SQL syntax."
    if len(sql_query) > 10000:
        return False, "SQL exceeds the 10,000-character limit."
    try:
        tree = sqlglot.parse_one(sql_query, dialect="sqlite")
        statements = sqlglot.parse(sql_query, dialect="sqlite")
    except ParseError:
        if raise_parse_errors:
            raise
        return False, "Invalid SQL syntax."
    except SqlglotError as exc:
        return False, "SQL parser error: " + str(exc)
    except (RecursionError, ValueError):
        return False, "SQL parser resource or input limit exceeded."
    if len(statements) != 1:
        return False, "Only one SQL statement is allowed."
    if not isinstance(tree, exp.Select):
        return False, "Only read-only SELECT statements are allowed."
    nodes = list(tree.walk())
    if len(nodes) > 1000:
        return False, "SQL exceeds the query complexity limit."
    if any(isinstance(node, exp.Select) and not node.expressions for node in nodes):
        return False, "Invalid SQL syntax."
    prohibited = {"Insert", "Update", "Delete", "Drop", "Alter", "Create",
                  "Command", "Into", "Lock", "With", "Union", "Intersect", "Except"}
    if any(type(node).__name__ in prohibited for node in nodes):
        return False, "SQL contains an unsupported or unsafe operation."
    allowed = {name.casefold() for name in allowed_tables if isinstance(name, str)}
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            return False, "Table-valued functions and dynamic table references are not allowed."
        if table.catalog or (table.db and table.db.casefold() != "main"):
            return False, "Access to external or temporary database schemas is not allowed."
        if table.name.casefold() not in allowed:
            return False, "Query references a table outside the allowed table list."
    return True, "SQL validation passed."


def inspect_query_plan(db_connection, sql_query: str, *, plan_details: list[str] | None = None, raise_sqlite_errors: bool = False) -> tuple[bool, str]:
    """Compile only; reject unfiltered unindexed core scans and temp sorts.

    Call with database.read_only_connection() to retain SQLite authorization.
    This conservative MVP policy uses plan text, not row-count/cost estimates.
    """
    from database import TABLES
    from sqlglot.optimizer.scope import traverse_scope

    valid, message = validate_sql(sql_query, sorted(TABLES))
    if not valid:
        return False, message
    try:
        cursor = db_connection.execute("EXPLAIN QUERY PLAN " + sql_query)
        try:
            rows = cursor.fetchall()
            details = [str(row[3]) for row in rows]
            if plan_details is not None:
                plan_details.append(plan_to_xml(sql_query, rows))
        finally:
            cursor.close()
        tree = sqlglot.parse_one(sql_query, dialect="sqlite")
        # Track each table/alias within its own SELECT, so an unrelated nested
        # WHERE cannot exempt an unfiltered scan in another scope.
        unfiltered = set()
        for scope in traverse_scope(tree):
            where = scope.expression.args.get("where")
            tables = [(alias, source) for alias, source in scope.sources.items()
                      if isinstance(source, exp.Table)]
            columns = list(where.find_all(exp.Column)) if where is not None else []
            for alias, table in tables:
                filtered = any(
                    column.table.casefold() == alias.casefold()
                    or (not column.table and len(tables) == 1)
                    for column in columns
                    if column.find_ancestor(exp.Select) is scope.expression
                )
                if not filtered:
                    unfiltered.update((alias.casefold(), table.name.casefold()))
        for detail in details:
            if re.search(r"USE TEMP B-TREE FOR\b", detail, re.I):
                return False, SORT_REJECTION
            match = re.match(r"SCAN\s+(?:TABLE\s+)?([\w.]+)", detail, re.I)
            if match and not re.search(r"\bUSING\b.*\bINDEX\b", detail, re.I):
                target = match.group(1).rsplit(".", 1)[-1].casefold()
                if target in unfiltered:
                    return False, SCAN_REJECTION
        return True, "Plan looks optimal."
    except sqlite3.Error as exc:
        if raise_sqlite_errors:
            raise
        failure = classify_error(exc, source="query plan")
        return False, f"Database error during plan inspection: {failure.message}"
    except SqlglotError as exc:
        return False, "SQL parser error during plan inspection: " + str(exc)
    except (RecursionError, ValueError, IndexError):
        return False, "Plan inspection failed due to invalid plan data or resource limits."


def validate_sql_security(sql_query: str) -> tuple[bool, str]:
    """Fail closed on destructive, stacked, unsupported, or out-of-scope SQL.

    ParseError is intentionally propagated so orchestration can pass the exact
    diagnostic to the generation agent. SQLite authorization is still required.
    """
    from database import TABLES

    if not isinstance(sql_query, str) or not sql_query.strip():
        return False, "Invalid SQL syntax."
    if len(sql_query) > 10000:
        return False, "SQL exceeds the 10,000-character limit."
    tree = sqlglot.parse_one(sql_query, read="sqlite")
    # Inspect every statement: parse_one alone does not reject stacked SQL.
    statements = sqlglot.parse(sql_query, read="sqlite")
    destructive = (exp.Drop, exp.Delete, exp.Update, exp.Insert, exp.Alter)
    if any(isinstance(node, destructive) for statement in statements if statement is not None
           for node in statement.walk()):
        return False, "Security Violation: Destructive statement detected"
    if not isinstance(tree, exp.Select):
        return False, "Security Violation: Only read-only SELECT statements are allowed"
    allowed, reason = validate_sql(sql_query, sorted(TABLES), raise_parse_errors=True)
    return (True, "Safe") if allowed else (False, reason)


def inspect_execution_plan(db_conn: sqlite3.Connection, sql_query: str) -> tuple[bool, str]:
    """Compile a secured SELECT and reject unindexed scans and temporary sorts.

    Diagnostics are returned in the message for correction and escalation.
    SQLite errors propagate to the caller, preserving missing-column diagnostics.
    No result query is executed here. Use an authorized read-only connection.
    """
    allowed, reason = validate_sql_security(sql_query)
    if not allowed:
        return False, reason
    cursor = db_conn.execute("EXPLAIN QUERY PLAN " + sql_query)
    try:
        details = [str(row[3]) for row in cursor.fetchall()]
    finally:
        cursor.close()
    diagnostic = "\n".join(details)
    for detail in details:
        # SQLite versions emit either SCAN TABLE name or SCAN name.
        scan = re.match(r"\s*SCAN\s+(?:TABLE\s+)?", detail, re.I)
        indexed = re.search(r"\bUSING\b.*\b(?:INDEX|PRIMARY KEY)\b", detail, re.I)
        constant = re.search(r"\bSCAN\s+CONSTANT\s+ROW", detail, re.I)
        if scan and not indexed and not constant:
            return False, "Performance Warning: Unindexed full-table scan detected\n" + diagnostic
        if re.search(r"\bUSE TEMP B-TREE\b", detail, re.I):
            return False, "Performance Warning: Temporary B-tree sorting detected\n" + diagnostic
    return True, "Plan approved\n" + diagnostic
