from io import BytesIO
from unittest.mock import Mock
from urllib.parse import urlsplit

import plan_artifacts
from plan_artifacts import save_plan_artifacts
from plan_server import PlanDownloadHandler
from query_plan import plan_to_xml
from xml.etree import ElementTree as ET


def request(path):
    handler = PlanDownloadHandler.__new__(PlanDownloadHandler)
    handler.path = path
    handler.wfile = BytesIO()
    handler.send_response = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock()
    handler.send_error = Mock()
    handler.do_GET()
    return handler


def test_stable_files_and_download_links():
    sql = "SELECT name FROM employees WHERE id=1"
    xml = plan_to_xml(sql, [(2, 0, 0, "SEARCH employees USING INTEGER PRIMARY KEY (rowid=?)")])
    links = save_plan_artifacts(sql, xml)
    assert links == save_plan_artifacts(sql, xml)
    files = list(plan_artifacts.ARTIFACT_PATH.glob("*"))
    assert len(files) == 1
    assert [p.suffix for p in files] == [".xml"]
    assert set(links) == {"plan_url"}
    for key in ("plan_url",):
        handler = request(urlsplit(links[key]).path)
        handler.send_response.assert_called_once_with(200)
        assert handler.wfile.getvalue()
        assert any(call.args[0] == "Content-Disposition" and "attachment" in call.args[1]
                   for call in handler.send_header.call_args_list)


def test_paths_cannot_escape_artifact_directory():
    for path in ["/plans/../../.env", "/plans/", "/.env", "/plans/%2e%2e/.env", "/plans/"+'a'*64+".html", "/plans/"+'a'*64+".sql"]:
        request(path).send_error.assert_called_once_with(404)


def test_missing_artifact_returns_404():
    request("/plans/" + "0"*64 + ".xml").send_error.assert_called_once_with(404)


def test_legacy_text_plan_converts_to_valid_xml():
    links = save_plan_artifacts("SELECT * FROM employees", "id=2 parent=0 auxiliary=0 | SCAN employees")
    handler = request(urlsplit(links["plan_url"]).path)
    assert ET.fromstring(handler.wfile.getvalue()).findtext(".//Detail") == "SCAN employees"
