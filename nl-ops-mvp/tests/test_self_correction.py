"""Syntax correction uses the real parser and compiled graph, without a model."""
from conftest import invoke_reviewed
from unittest.mock import patch

from sqlglot import parse_one
from sqlglot.errors import ParseError

from agents import SafetyDecision
from graph import graph, validate_sql_node, validation_route

BAD_SQL = "SELECT ( FROM employees"
GOOD_SQL = "SELECT id FROM employees WHERE id = 1"


def test_parse_error_is_preserved_and_routes_to_generation():
    try:
        parse_one(BAD_SQL, dialect="sqlite")
    except ParseError as exc:
        expected = str(exc)
    state = {"sql": BAD_SQL, "correction_attempts": 0}
    state.update(validate_sql_node(state))
    assert state["sql_syntax_error"] == expected
    assert state["correction_attempts"] == 1
    assert validation_route(state) == "sql_generation"


def run_workflow(outputs):
    with patch("graph.get_schema", return_value="schema"), \
         patch("graph.get_relevant_schema", return_value="scoped schema"), \
         patch("graph.check_intent", return_value=SafetyDecision(allowed=True, reason="Read only")), \
         patch("graph.generate_sql", side_effect=outputs) as generate, \
         patch("graph.inspect_query_plan_node", return_value={"status": "plan_approved"}), \
         patch("graph.execute_query", return_value={"columns": ["id"], "rows": [[1]], "truncated": False}) as execute, \
         patch("graph.send_dba_alert", return_value={"status": "Disabled"}), \
         patch("graph.append_audit_event"):
        # Build after patching the node to avoid database/plan side effects.
        from graph import build_graph
        result = invoke_reviewed(build_graph(), {"request": "Find employee 1"})
    return result, generate, execute


def test_graph_corrects_before_execution():
    result, generate, execute = run_workflow([BAD_SQL, GOOD_SQL])
    assert result["status"] == "completed"
    assert result["correction_attempts"] == 1
    assert result["sql_syntax_error"] == ""
    assert generate.call_count == 2
    correction = generate.call_args.kwargs
    assert correction["previous_sql"] == BAD_SQL
    assert correction["sql_syntax_error"]
    execute.assert_called_once_with(GOOD_SQL, result["row_limit"])


def test_graph_stops_after_three_syntax_failures():
    result, generate, execute = run_workflow([BAD_SQL] * 3)
    assert result["status"] == "error"
    assert result["correction_attempts"] == 3
    assert "after 3 attempts" in result["message"]
    assert generate.call_count == 3
    execute.assert_not_called()


def test_policy_violation_does_not_retry():
    result, generate, execute = run_workflow(["DELETE FROM employees"])
    assert result["status"] == "blocked"
    assert result["correction_attempts"] == 0
    assert generate.call_count == 1
    execute.assert_not_called()
