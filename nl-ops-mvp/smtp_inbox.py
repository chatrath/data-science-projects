"""Persistent mailbox for local development SMTP; never forwards email."""
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
import os
import re
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET
from plan_artifacts import save_plan_artifacts

INBOX_PATH = Path(__file__).resolve().parent / "local_mailbox"


def store_message(content: bytes, inbox=None) -> Path:
    folder = Path(inbox or INBOX_PATH)
    folder.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex
    temporary = folder / (name + ".tmp")
    target = folder / (name + ".eml")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return target


def read_local_inbox(limit=20, inbox=None) -> list[dict]:
    folder = Path(inbox or INBOX_PATH)
    messages = []
    for path in sorted(folder.glob("*.eml"), reverse=True)[:limit]:
        message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        body = message.get_body(preferencelist=("plain",))
        attachments = [{"filename": part.get_filename() or "query_plan.xml",
                        "content": (part.get_payload(decode=True) or b"").decode("utf-8", errors="replace")}
                       for part in message.iter_attachments() if part.get_content_type() == "application/xml"]
        text = body.get_content() if body else ""
        links = {}
        for extension, key in [("xml", "plan_url")]:
            match = re.search(r"https?://[^\s<>]+/plans/[a-f0-9]{64}\." + extension + r"\b", text)
            if match:
                links[key] = match.group(0)
        if not links:
            for attachment in attachments:
                try:
                    root = ET.fromstring(attachment["content"])
                    if root.tag == "SQLiteQueryPlan":
                        links = save_plan_artifacts(root.findtext("Statement", ""), attachment["content"])
                        break
                except ET.ParseError:
                    continue
        messages.append({"file": path.name, "from": str(message.get("From", "")),
                         "to": str(message.get("To", "")), "subject": str(message.get("Subject", "")),
                         "body": text, "attachments": attachments, "plan_links": links})
    return messages
