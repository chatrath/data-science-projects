"""XML representation of SQLite estimated plan rows, without invented costs."""
import re
import sqlite3
from xml.etree import ElementTree as ET


def _xml_text(value) -> str:
    # XML 1.0 excludes most control characters even when SQLite accepts them
    # inside string literals. Keep the exported document well formed.
    return "".join(c if c in "\t\n\r" or 0x20 <= ord(c) <= 0xD7FF
                   or 0xE000 <= ord(c) <= 0xFFFD or 0x10000 <= ord(c) <= 0x10FFFF
                   else "\uFFFD" for c in str(value))


def plan_to_xml(sql_query: str, rows: list) -> str:
    """Retain every EQP row and nest operators using SQLite node/parent IDs.

    This is an application-owned XML format, not SQL Server ShowPlan XML and
    not an actual execution plan. SQLite supplies no actual rows or costs.
    """
    root = ET.Element("SQLiteQueryPlan", {"formatVersion": "1", "planType": "Estimated",
                                         "sqliteVersion": sqlite3.sqlite_version,
                                         "source": "EXPLAIN QUERY PLAN"})
    ET.SubElement(root, "Statement").text = _xml_text(sql_query)
    ET.SubElement(root, "Statistics", {"actualExecution": "false", "costEstimatesAvailable": "false"})
    operators = ET.SubElement(root, "Operators")
    nodes = {}
    parents = {}
    for node_id, parent_id, auxiliary, detail in rows:
        node_id, parent_id = str(node_id), str(parent_id)
        text = str(detail)
        if re.match(r"SEARCH\b", text, re.I):
            access = "IndexedSearch"
        elif re.match(r"SCAN\b", text, re.I):
            access = "IndexScan" if re.search(r"\bUSING\b.*\bINDEX\b", text, re.I) else "Scan"
        elif re.search(r"TEMP B-TREE", text, re.I):
            access = "TemporaryBTree"
        else:
            access = "Other"
        node = ET.Element("Operator", {"id": node_id, "parentId": parent_id,
                                       "auxiliary": str(auxiliary), "access": access})
        ET.SubElement(node, "Detail").text = _xml_text(detail)
        nodes[node_id] = node
        parents[node_id] = parent_id
    for node_id, node in nodes.items():
        parent_id = parents[node_id]
        # SQLite produces an acyclic plan. Detect bad input rather than build
        # a cyclic ElementTree that cannot be serialized.
        seen = {node_id}
        ancestor = parent_id
        while ancestor in parents:
            if ancestor in seen:
                raise ValueError("Cyclic execution plan hierarchy")
            seen.add(ancestor)
            ancestor = parents[ancestor]
        (nodes[parent_id] if parent_id in nodes else operators).append(node)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
