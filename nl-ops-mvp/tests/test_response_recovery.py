"""Malformed final JSON gets one schema-constrained recovery attempt."""
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from agents import _run_tool_agent, SQLQuery, ModelOutputError
from graph import sql_node


def test_truncated_response_is_regenerated_with_json_schema():
    model = Mock()
    expected = SQLQuery(sql="SELECT country, COUNT(*) FROM customers GROUP BY country ORDER BY country LIMIT 50")
    model.with_structured_output.return_value.invoke.return_value = expected
    prompt = ChatPromptTemplate.from_messages([("human", "{request}")])
    with patch("graph.run_agent_tools", return_value=AIMessage(content='{"sql":"SELECT country, 000000')), \
         patch("agents.get_model", return_value=model):
        assert _run_tool_agent(prompt, {"request": "Count customers by country"}, SQLQuery, "test") == expected
    model.with_structured_output.assert_called_once_with(SQLQuery, method="json_schema")
    model.with_structured_output.return_value.invoke.assert_called_once()


def test_invalid_recovery_stops_with_accurate_error():
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = OutputParserException("Invalid JSON")
    prompt = ChatPromptTemplate.from_messages([("human", "{request}")])
    with patch("graph.run_agent_tools", return_value=AIMessage(content='{"sql":')), patch("agents.get_model", return_value=model):
        with pytest.raises(ModelOutputError, match="incomplete or invalid"):
            _run_tool_agent(prompt, {"request": "Count customers"}, SQLQuery, "test")


def test_output_error_is_not_reported_as_connectivity_failure():
    with patch("graph.get_relevant_schema", return_value="schema"), \
         patch("graph.generate_sql", side_effect=ModelOutputError("Invalid response after retrying")):
        result = sql_node({"request": "Count customers", "row_limit": 50})
    assert result["status"] == "error"
    assert result["message"] == "Invalid response after retrying"
