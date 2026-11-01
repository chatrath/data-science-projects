"""Best-effort SMTP alerts; failures never change the query rejection decision."""
import config  # Load .env before reading SMTP settings.

from datetime import datetime, timezone
from email.message import EmailMessage
import logging
import os
import smtplib
import ssl
from html import escape
from plan_artifacts import save_plan_artifacts

logger = logging.getLogger(__name__)


def send_dba_alert(sql_query: str, execution_plan: str, reason: str, *,
                   user_prompt: str = "", request_id: str = "", timestamp: str = "") -> dict:
    """Send inline with bounded socket timeouts; return a safe audit status.

    Local test SMTP servers may be unauthenticated. Remote servers require
    credentials; TLS defaults on except for a localhost development server.
    """
    host = os.getenv("SMTP_SERVER", "localhost").strip()
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "")
    local = host.casefold() in {"localhost", "127.0.0.1", "::1"}
    if (bool(user) != bool(password)) or (not local and not (user and password)):
        logger.warning("DBA alert skipped: SMTP credentials are incomplete or absent for a remote server.")
        return {"status": "Skipped", "message": "Configure SMTP_USER and SMTP_PASSWORD for remote email delivery."}
    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        message = EmailMessage()
        message["Subject"] = "[NL-Ops] Query failed or blocked"
        message["From"] = os.getenv("SMTP_FROM", user or "nl-ops@enterprise-mock.com")
        message["To"] = os.getenv("DBA_EMAIL_RECIPIENT", "dba-alerts@enterprise-mock.com")
        try:
            links = save_plan_artifacts(sql_query, execution_plan)
        except OSError:
            logger.warning("Plan file could not be saved; sending DBA alert without a download link.")
            links = {}
        plan_text = f"Download SQL Plan: {links['plan_url']}" if links else "SQL Plan: unavailable for this failure."
        plan_html = (f'<a href="{escape(links["plan_url"], quote=True)}">Download SQL Plan</a>'
                     if links else "SQL Plan: unavailable for this failure.")
        message.set_content(
            f"A workflow request failed or was blocked.\n\n"
            f"Timestamp (UTC): {timestamp or datetime.now(timezone.utc).isoformat()}\n"
            f"Request ID: {request_id or 'not supplied'}\n"
            f"Reason: {reason}\n\nUser prompt:\n{user_prompt or '(not supplied)'}\n\n"
            f"Generated SQL:\n{sql_query or '(not generated)'}\n\n"
            f"{plan_text}\n\n"
            "The XML is a SQLite estimated plan, not an SSMS .sqlplan file.\n"
        )
        message.add_alternative(
            "<h3>Query failed or blocked</h3>"
            f"<p><strong>Reason:</strong> {escape(reason)}</p>"
            f"<p><strong>Timestamp:</strong> {escape(timestamp or datetime.now(timezone.utc).isoformat())}</p>"
            f"<p><strong>Request:</strong> {escape(request_id)}</p>"
            f"<p><strong>User prompt:</strong> {escape(user_prompt)}</p>"
            f"<pre>{escape(sql_query or '(not generated)')}</pre>"
            f"<p>{plan_html}</p>",
            subtype="html",
        )
        with smtplib.SMTP(host, port, timeout=5) as server:
            server.ehlo()
            if os.getenv("SMTP_STARTTLS", "false" if local else "true").lower() == "true":
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
            if user and password:
                server.login(user, password)
            server.send_message(message)
        logger.info("DBA failure alert accepted by SMTP server.")
        return {"status": "Sent", "message": "Alert accepted by SMTP server; inbox delivery is not verified."}
    except Exception as exc:
        # Never log SMTP errors verbatim: authentication failures may contain
        # server diagnostics. The exception class is enough for local triage.
        logger.warning("DBA alert failed (%s); query remains blocked.", type(exc).__name__)
        return {"status": "Failed", "message": "SMTP delivery failed. Check SMTP settings, credentials, and connectivity."}
