"""Streamlit helpdesk interface for the NL-Ops workflow."""
from config import log_tracing_status
from cache import initialize_cache, run_cached

initialize_cache()  # Must run before graph/agents instantiate LangChain components.

import logging

import pandas as pd
import streamlit as st

from database import DB_PATH, initialize_database
from graph import graph, notify_failure_node
from audit_logger import append_audit_event, read_recent_audit_logs, request_identity
from smtp_inbox import read_local_inbox
from plan_artifacts import save_plan_artifacts
from plan_server import start_plan_server


@st.cache_resource
def plan_download_service():
    return start_plan_server()


def show_plan_links(links):
    if links.get("plan_url"):
        st.markdown(f"[Download SQL Plan]({links['plan_url']})")

logging.basicConfig(level=logging.INFO)
log_tracing_status()
st.set_page_config(page_title="NL-Ops Helpdesk", page_icon="🎫", layout="wide", initial_sidebar_state="collapsed")
st.title("🎫 NL-Ops Helpdesk")
st.caption("Submit a read-only data request to your mock enterprise helpdesk.")
try:
    plan_download_service()
except (OSError, ValueError):
    logging.exception("Audit/DBA plan downloads unavailable; check local port 8502")

try:
    initialize_database()
except Exception:
    logging.exception("Database initialization failed")
    st.error("Could not initialize the mock database. Check folder write permissions.")
    st.stop()

# Keep requests bounded without exposing operational settings in the UI.
row_limit = 50

with st.form("ticket"):
    request = st.text_area("How can we help?", placeholder="Show open IT tickets with employee names", max_chars=4000)
    submitted = st.form_submit_button("Submit request", type="primary")

if submitted:
    inflight = {**request_identity(request), "request": request, "row_limit": row_limit}
    def run_live():
        result = inflight
        for update in graph.stream(result, config={"run_name": "NL-Ops Helpdesk Request"}, stream_mode="updates"):
            for node, values in update.items():
                # Nodes that return no state changes can stream a None delta.
                # This is normal (e.g. no failure notification is necessary).
                if values is None:
                    continue
                result.update(values)
                st.write(f"{node.replace('_', ' ').title()}: {values.get('status', result.get('status', 'ready'))}")
        return result

    try:
        with st.status("Processing helpdesk request…", expanded=True) as progress:
            result = run_cached(request, row_limit, run_live, DB_PATH)
            progress.update(label="Request processed", state="complete" if result.get("status") == "completed" else "error", expanded=False)
        st.session_state["result"] = result
    except Exception:
        logging.exception("Workflow failed")
        failed = {**inflight, "status": "error", "message": "Workflow failed. Check server logs and retry."}
        failed.update(notify_failure_node(failed))
        st.session_state["result"] = failed
        try:
            append_audit_event(failed)
        except (OSError, ValueError, TypeError):
            st.error("Enterprise audit persistence also failed. Check disk space and file permissions.")

if "result" in st.session_state:
    result = st.session_state["result"]
    st.subheader("Execution result")
    st.write(f"Status: **{result['status']}**")
    if "cache_source" in result:
        cached_result = result["cache_source"] == "result cache"
        source = {"result cache": "Saved result", "agents / LLM cache": "Cached LLM responses",
                  "live agents": "Live agents"}.get(result["cache_source"], result["cache_source"])
        execution = "Database Hit" if not cached_result and result.get("execution_status") in {"Success", "Blocked", "Error"} else "Skipped"
        st.info(f"**Source -** {source}  \n"
                f"**Agents and Database execution -** {execution}  \n"
                f"**Request duration -** {result['elapsed_seconds']:.3f} seconds")
    if result.get("sql"):
        st.code(result["sql"], language="sql")
    if result["status"] == "completed":
        st.success(result["message"])
        st.dataframe(pd.DataFrame(result["rows"], columns=result["columns"]), use_container_width=True, hide_index=True)
    else:
        st.warning(result["message"])

with st.expander("Enterprise Audit Logs"):
    st.caption("Recent requests, newest first. Cached requests retain the original validation outcomes; checks are not rerun.")
    st.button("Refresh audit logs")
    try:
        entries = read_recent_audit_logs()
        if entries:
            rows = []
            for entry in entries:
                links = entry.get("plan_links") or save_plan_artifacts(entry.get("sql_generated", ""), entry.get("execution_plan", ""))
                rows.append({"Time": entry.get("timestamp"), "Request": entry.get("user_query"),
                             "Safety": entry.get("safety_status"),
                             "SQL validation": entry.get("guardrail_status", {}).get("ast"),
                             "Plan check": entry.get("guardrail_status", {}).get("explain"),
                             "Outcome": entry.get("execution_outcome"), "Rows": entry.get("row_count"),
                             "SQL Plan": links.get("plan_url"),
                             "DBA alert": entry.get("dba_alert", {}).get("status", "NotNeeded")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                         column_config={"SQL Plan": st.column_config.LinkColumn(display_text="Download SQL Plan")})
        else:
            st.info("No enterprise audit records yet.")
    except (OSError, UnicodeError):
        st.warning("Could not read enterprise audit logs. Check file permissions.")

with st.expander("DBA Email Inbox (Local Test)"):
    st.caption("Captured test emails stay on this machine. They are not delivered to external inboxes.")
    st.button("Refresh local inbox")
    try:
        messages = read_local_inbox()
        if not messages:
            st.info("No captured emails yet. Try: Show all employees.")
        for message in messages:
            st.markdown(f"**{message['subject']}**")
            st.caption(f"From: {message['from']} · To: {message['to']}")
            for line in message["body"].splitlines():
                if line.startswith(("Reason:", "Timestamp (UTC):")):
                    st.write(line)
            show_plan_links(message.get("plan_links", {}))
            if not message.get("plan_links"):
                st.caption("Legacy email. Submit a new request to generate plan links.")
    except (OSError, ValueError, UnicodeError):
        st.warning("Could not read the local email inbox. Check file permissions.")
