from email.message import EmailMessage

from smtp_inbox import read_local_inbox, store_message


def test_captured_email_is_persistent_and_readable(tmp_path):
    message = EmailMessage()
    message["From"] = "nl-ops@enterprise-mock.com"
    message["To"] = "dba-alerts@enterprise-mock.com"
    message["Subject"] = "Performance guardrail test"
    message.set_content("SQL: SELECT * FROM employees\nPlan: SCAN employees")
    message.add_attachment(b'<SQLiteQueryPlan planType="Estimated" />', maintype="application",
                           subtype="xml", filename="query_plan.xml")
    path = store_message(message.as_bytes(), tmp_path)
    assert path.suffix == ".eml"
    result = read_local_inbox(inbox=tmp_path)
    assert len(result) == 1
    assert result[0]["to"] == message["To"]
    assert result[0]["subject"] == message["Subject"]
    assert "SCAN employees" in result[0]["body"]
    assert result[0]["attachments"][0]["filename"] == "query_plan.xml"
    assert "SQLiteQueryPlan" in result[0]["attachments"][0]["content"]


def test_missing_inbox_is_empty(tmp_path):
    assert read_local_inbox(inbox=tmp_path / "absent") == []
