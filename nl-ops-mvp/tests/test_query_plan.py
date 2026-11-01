from xml.etree import ElementTree as ET

import pytest

from database import initialize_database, read_only_connection
from guardrails import inspect_query_plan
from query_plan import plan_to_xml


def test_plan_xml_has_hierarchy_and_exact_details():
    rows = [(2, 0, 0, "SCALAR SUBQUERY"), (4, 2, 0, "SEARCH employees USING INTEGER PRIMARY KEY (rowid=?)")]
    root = ET.fromstring(plan_to_xml("SELECT name FROM employees WHERE id=1", rows))
    assert root.get("planType") == "Estimated"
    assert root.find("Statistics").get("actualExecution") == "false"
    parent = root.find("Operators/Operator")
    assert parent.get("id") == "2"
    child = parent.find("Operator")
    assert child.get("id") == "4"
    assert child.get("parentId") == "2"
    assert child.findtext("Detail") == rows[1][3]
    assert child.get("access") == "IndexedSearch"


def test_sql_and_details_are_escaped_and_unicode_preserved():
    sql = "SELECT 'A&B <tag> café' FROM employees"
    detail = "SCAN employees & <index>"
    xml = plan_to_xml(sql, [(2, 0, 0, detail)])
    root = ET.fromstring(xml)
    assert root.findtext("Statement") == sql
    assert root.findtext(".//Detail") == detail
    assert "&amp;" in xml
    ET.fromstring(plan_to_xml("SELECT '\x01'", []))


def test_inspection_exports_full_xml_for_rejected_plan(tmp_path):
    path = tmp_path / "plan.db"
    initialize_database(path)
    plans = []
    with read_only_connection(path) as conn:
        allowed, _ = inspect_query_plan(conn, "SELECT * FROM employees", plan_details=plans)
    assert not allowed
    assert len(plans) == 1
    assert ET.fromstring(plans[0]).findtext(".//Detail") == "SCAN employees"


def test_cyclic_plan_is_rejected():
    with pytest.raises(ValueError):
        plan_to_xml("SELECT 1", [(1, 2, 0, "A"), (2, 1, 0, "B")])
