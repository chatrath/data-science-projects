"""Check relevance, relationship closure, database consistency, and injection."""
import json
from contextlib import closing
import sqlite3
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

from agents import SQLQuery, generate_sql
from database import initialize_database
from graph import sql_node
from schema_context import SCHEMA_CATALOG, get_relevant_schema


def tables(query):
    return json.loads(get_relevant_schema(query))["tables"]


def test_employee_query_excludes_unrelated_tables():
    selected = tables("Show employees with salary over 60000")
    assert set(selected) == {"employees"}
    assert selected["employees"]["description"]
    assert selected["employees"]["columns"]["salary"]["type"] == "REAL"


def test_customer_synonyms_are_scoped():
    assert set(tables("Show users who signed up last week")) == {"customers"}


def test_ticket_context_includes_join_dependency():
    selected = tables("List open tickets")
    assert set(selected) == {"tickets", "employees"}
    assert selected["tickets"]["relationships"][0]["mapping"] == "tickets.employee_id references employees.id"


def test_multiple_entities_and_unknown_fallback():
    assert set(tables("Compare employees and customers")) == {"employees", "customers"}
    assert set(tables("Summarize the data")) == set(SCHEMA_CATALOG)
    assert set(tables("")) == set(SCHEMA_CATALOG)


def test_word_boundaries_and_case():
    assert set(tables("EMPLOYEES")) == {"employees"}
    assert set(tables("customers with employeeish emails")) == {"customers"}


def test_catalog_matches_actual_columns_and_relationships(tmp_path):
    path = tmp_path / "catalog.db"
    initialize_database(path)
    with closing(sqlite3.connect(path)) as conn:
        for name, metadata in SCHEMA_CATALOG.items():
            actual = {row[1]: row[2] for row in conn.execute(f'PRAGMA table_info("{name}")')}
            expected = {column: details["type"] for column, details in metadata["columns"].items()}
            assert actual == expected
            foreign_keys = {(row[3], row[2], row[4]) for row in conn.execute(f'PRAGMA foreign_key_list("{name}")')}
            assert foreign_keys == {(r["column"], r["references_table"], r["references_column"])
                                    for r in metadata["relationships"]}


def test_graph_injects_scoped_context_not_raw_schema():
    with patch("graph.generate_sql", return_value="SELECT name FROM employees WHERE id=1") as generate:
        result = sql_node({"request": "Show employee 1", "schema": "RAW DDL", "row_limit": 50})
    context = generate.call_args.args[1]
    assert set(json.loads(context)["tables"]) == {"employees"}
    assert result["schema_context"] == context
    assert "RAW DDL" not in context


def test_sql_agent_receives_structured_prompt():
    captured = []
    def fake_model(messages):
        captured.append(messages[0].content)
        if len(captured) == 1:
            return AIMessage(content="", tool_calls=[{"name": "get_table_metadata", "args": {"table_name": "employees"}, "id": "metadata"}])
        if len(captured) == 2:
            return AIMessage(content="", tool_calls=[{"name": "validate_sql_syntax", "args": {"sql_query": "SELECT name FROM employees WHERE id=1"}, "id": "lint"}])
        return AIMessage(content='{"sql":"SELECT name FROM employees WHERE id=1"}')
    class Model:
        def bind_tools(self, *args, **kwargs):
            return RunnableLambda(fake_model)
    context = get_relevant_schema("Show employee 1")
    with patch("agents.get_model", return_value=Model()), patch("tools.read_table_metadata", return_value={"table": "employees", "columns": [], "row_count": 3}):
        assert generate_sql("Show employee 1", context, 50).startswith("SELECT")
    assert context in captured[0]
    assert '"customers"' not in captured[0]
