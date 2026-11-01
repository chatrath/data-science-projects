"""Immutable plan files and download URLs shared by UI, email, and audit."""
import hashlib
import os
import re
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET
from query_plan import plan_to_xml

ARTIFACT_PATH = Path(__file__).resolve().parent / "plan_artifacts"


def save_plan_artifacts(sql_query: str, plan_xml: str) -> dict:
    if not plan_xml:
        return {}
    if not plan_xml.lstrip().startswith("<"):
        rows = []
        for line in plan_xml.splitlines():
            match = re.fullmatch(r"id=(\d+) parent=(\d+) auxiliary=(\d+) \| (.*)", line)
            if not match:
                return {}
            rows.append((int(match[1]), int(match[2]), int(match[3]), match[4]))
        plan_xml = plan_to_xml(sql_query, rows)
    try:
        if ET.fromstring(plan_xml).tag != "SQLiteQueryPlan":
            return {}
    except ET.ParseError:
        return {}
    identifier = hashlib.sha256((sql_query + "\0" + plan_xml).encode("utf-8")).hexdigest()
    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
    for extension, content in [("xml", plan_xml)]:
        target = ARTIFACT_PATH / f"{identifier}.{extension}"
        # Identical content reuses its immutable file, including cached requests.
        if not target.exists():
            temporary = ARTIFACT_PATH / f"{uuid4().hex}.tmp"
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    base = os.getenv("PLAN_BASE_URL", "http://127.0.0.1:8502").rstrip("/")
    return {"plan_url": f"{base}/plans/{identifier}.xml"}
