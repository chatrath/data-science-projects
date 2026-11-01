"""Exercise native AIMessage calls through the real ToolNode without Ollama."""
import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agents import SafetyDecision, generate_sql
import database
import graph as workflow
from schema_context import get_relevant_schema
from tools import get_table_metadata, validate_sql_syntax


SQL = "SELECT name FROM employees WHERE id=1"


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return next(self.responses)


def metadata_call(table="employees"):
    return AIMessage(content="", tool_calls=[{"name": "get_table_metadata", "args": {"table_name": table}, "id": "metadata-1"}])


def lint_call(sql=SQL):
    return AIMessage(content="", tool_calls=[{"name": "validate_sql_syntax", "args": {"sql_query": sql}, "id": "lint-1"}])


def test_agent_inspects_real_metadata_and_consumes_tool_results(tmp_path):
    path = tmp_path / "tools.db"
    database.initialize_database(path)
    model = ScriptedModel([metadata_call(), lint_call(), AIMessage(content=json.dumps({"sql": SQL}))])
    with patch("agents.get_model", return_value=model), \
         patch("tools.read_table_metadata", side_effect=lambda table: database.get_table_metadata(table, path)) as metadata:
        result = generate_sql("Inspect the employees schema, then show employee 1's name", get_relevant_schema("employees"), 50)
    assert result == SQL
    assert {tool.name for tool in model.bound_tools} == {"get_table_metadata", "validate_sql_syntax"}
    metadata.assert_called_once_with("employees")
    returned = [message for message in model.calls[1] if isinstance(message, ToolMessage)]
    assert returned[0].tool_call_id == "metadata-1"
    payload = json.loads(returned[0].content)
    assert payload["ok"]
    assert payload["row_count"] == 3
    assert {column["name"] for column in payload["columns"]} == {"id", "name", "department", "salary"}
    assert {column["type"] for column in payload["columns"]} >= {"INTEGER", "TEXT", "REAL"}
    assert any(isinstance(message, ToolMessage) and message.tool_call_id == "lint-1" for message in model.calls[2])


def test_tool_loop_runs_inside_full_guarded_workflow(tmp_path):
    path = tmp_path / "tools.db"
    database.initialize_database(path)
    model = ScriptedModel([metadata_call(), lint_call(), AIMessage(content=json.dumps({"sql": SQL}))])
    with patch("agents.get_model", return_value=model), \
         patch("graph.check_intent", return_value=SafetyDecision(allowed=True, reason="Read only")), \
         patch("graph.get_schema", side_effect=lambda: database.get_schema(path)), \
         patch("graph.read_only_connection", side_effect=lambda: database.read_only_connection(path)), \
         patch("graph.execute_query", side_effect=lambda sql, limit: database.execute_query(sql, limit, path)), \
         patch("tools.read_table_metadata", side_effect=lambda table: database.get_table_metadata(table, path)):
        result = workflow.graph.invoke({"request": "Inspect employees and show the name for employee id 1"})
    assert result["status"] == "completed"
    assert result["rows"] == [("Alex Morgan",)]
    assert result["ast_status"] == result["plan_status"] == "Passed"
    assert result["audit_written"]


def test_metadata_tool_blocks_unauthorized_identifier(tmp_path):
    path = tmp_path / "tools.db"
    database.initialize_database(path)
    with patch("tools.read_table_metadata", side_effect=lambda table: database.get_table_metadata(table, path)):
        result = get_table_metadata.invoke({"table_name": "employees; DROP TABLE customers"})
    assert not result["ok"]
    assert database.execute_query("SELECT COUNT(*) FROM customers", path=path)["rows"] == [(4,)]


def test_syntax_tool_reports_syntax_and_policy_separately():
    assert validate_sql_syntax.invoke({"sql_query": SQL})["valid"]
    assert not validate_sql_syntax.invoke({"sql_query": "SELECT ( FROM employees"})["valid"]
    destructive = validate_sql_syntax.invoke({"sql_query": "DELETE FROM employees"})
    assert destructive["valid"]
    assert not destructive["policy_allowed"]


def test_repeated_tool_calls_stop_at_budget():
    class LoopingModel(ScriptedModel):
        def invoke(self, messages):
            self.calls.append(list(messages))
            call = metadata_call()
            call.tool_calls[0]["id"] = f"call-{len(self.calls)}"
            return call
    model = LoopingModel([])
    with patch("agents.get_model", return_value=model), patch("tools.read_table_metadata", return_value={"table": "employees", "row_count": 3, "columns": []}):
        with pytest.raises(ValueError, match="turn limit"):
            generate_sql("employees", get_relevant_schema("employees"), 50)
    assert len(model.calls) == workflow.MAX_AGENT_TURNS


def test_final_answer_cannot_skip_required_tools():
    model = ScriptedModel([AIMessage(content=json.dumps({"sql": SQL})) for _ in range(workflow.MAX_AGENT_TURNS)])
    with patch("agents.get_model", return_value=model):
        with pytest.raises(ValueError, match="turn limit"):
            generate_sql("employees", get_relevant_schema("employees"), 50)


def test_unknown_tool_is_handled_without_execution():
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": "execute_sql", "args": {"sql": "DROP TABLE employees"}, "id": "unknown"}]),
        AIMessage(content='{"allowed":false,"reason":"Unsupported tool"}'),
    ])
    from agents import check_intent
    with patch("agents.get_model", return_value=model), patch("graph.execute_query") as execute:
        assert not check_intent("Check the schema", "employees(id INTEGER)").allowed
    execute.assert_not_called()
    assert any(isinstance(message, ToolMessage) and message.status == "error" for message in model.calls[1])


def test_excessive_parallel_calls_do_not_reach_tools():
    calls = [{"name": "get_table_metadata", "args": {"table_name": "employees"}, "id": f"call-{i}"} for i in range(5)]
    model = ScriptedModel([AIMessage(content="", tool_calls=calls)])
    with patch("agents.get_model", return_value=model), patch("tools.read_table_metadata") as metadata:
        with pytest.raises(ValueError, match="tool-call budget"):
            generate_sql("employees", get_relevant_schema("employees"), 50)
    metadata.assert_not_called()


def test_tool_loop_failure_reaches_audit_and_failure_alert(tmp_path):
    path = tmp_path / "tools.db"
    database.initialize_database(path)
    model = ScriptedModel([AIMessage(content=json.dumps({"sql": SQL})) for _ in range(workflow.MAX_AGENT_TURNS)])
    with patch("agents.get_model", return_value=model), \
         patch("graph.get_schema", side_effect=lambda: database.get_schema(path)), \
         patch("graph.check_intent", return_value=SafetyDecision(allowed=True, reason="Read only")), \
         patch("graph.send_dba_alert", return_value={"status": "Sent"}) as notify, \
         patch("graph.execute_query") as execute:
        result = workflow.graph.invoke({"request": "Show employee 1"})
    assert result["status"] == "error"
    assert result["execution_status"] == "NotRun"
    assert result["audit_written"]
    notify.assert_called_once()
    execute.assert_not_called()
