"""Streamlit helpdesk interface for the NL-Ops workflow."""
from config import log_tracing_status
from cache import initialize_cache

initialize_cache()  # Must run before graph/agents instantiate LangChain components.

import logging

import pandas as pd
import streamlit as st

from database import initialize_database
from graph import graph, start_request, resume_request, request_config, get_request_state, recover_request
from audit_logger import read_recent_audit_logs
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
st.set_page_config(page_title="NL-Ops Helpdesk", page_icon="\U0001f3ab", layout="wide", initial_sidebar_state="collapsed")
st.title("\U0001f3ab NL-Ops Helpdesk")
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

# Persist the request/thread ID before starting. A rerun reads the checkpoint;
# it never starts another run or audits an interrupt as a terminal failure.
if submitted:
    if st.session_state.get("result", {}).get("run_state", st.session_state.get("result", {}).get("status")) in {"awaiting_approval", "failed_recoverable"}:
        st.warning("Resolve the pending request before starting another.")
    else:
        from uuid import uuid4
        st.session_state["active_request_id"] = str(uuid4())
        st.session_state["pending_request"] = request

active_id = st.session_state.get("active_request_id")
if active_id:
    try:
        snapshot = graph.get_state(request_config(active_id))
        if not snapshot.values and "pending_request" in st.session_state:
            with st.status("Preparing your query for review…", expanded=True) as progress:
                _, result = start_request(st.session_state["pending_request"], row_limit, thread_id=active_id)
                progress.update(label="Ready for review" if result.get("status") == "awaiting_approval" else "Request processed",
                                state="complete" if result.get("status") == "awaiting_approval" else "error", expanded=False)
            st.session_state["result"] = result
        elif snapshot.values:
            st.session_state["result"] = get_request_state(active_id)
    except Exception:
        logging.exception("Checkpointed workflow failed")
        st.warning("The request could not complete. Its checkpoint is retained; please contact support before retrying.")

if st.session_state.get("result", {}).get("run_state") == "failed_recoverable":
    failure = st.session_state["result"]
    st.warning(failure.get("message", "The request paused after a temporary failure."))
    retry = st.button("Retry saved request", type="primary")
    cancel = st.button("Cancel recovery", disabled=failure.get("recovery_mode") == "checkpoint")
    if retry or cancel:
        try:
            st.session_state["result"] = recover_request(
                active_id, "retry" if retry else "cancel", failure.get("recovery_token", ""))
            st.rerun()
        except Exception:
            logging.exception("Checkpoint recovery failed")
            st.warning("Recovery could not complete. The saved request has been retained.")

if st.session_state.get("result", {}).get("status") == "awaiting_approval":
    pending = st.session_state["result"]
    st.subheader("Review query")
    st.info("Security and performance checks passed. Review the query before allowing data retrieval.")
    st.code(pending["sql"], language="sql")
    with st.form(f"approval_{pending['approval_token']}"):
        revised_sql = st.text_area("SQL to review", value=pending["sql"], max_chars=10000)
        approve = st.form_submit_button("Approve and execute", type="primary")
        reject = st.form_submit_button("Reject request")
        revise = st.form_submit_button("Validate revised SQL")
    if approve or reject or revise:
        if approve and revised_sql != pending["sql"]:
            st.warning("SQL was edited. Select Validate revised SQL and review it again before approval.")
        else:
            try:
                action = "approve" if approve else "reject" if reject else "edit"
                with st.spinner("Processing your decision…"):
                    st.session_state["result"] = resume_request(
                        active_id, action, pending["approval_token"], sql=revised_sql)
                st.rerun()
            except Exception:
                logging.exception("Approval resume failed")
                st.warning("The decision could not be completed. Reload to check the saved request state.")

if "result" in st.session_state and st.session_state["result"].get("run_state") not in {"awaiting_approval", "failed_recoverable"}:
    result = st.session_state["result"]
    st.subheader("Results")
    st.write(f"Status: **{result['status']}**")
    if "elapsed_seconds" in result:
        st.caption(f"Request completed in {result['elapsed_seconds']:.3f} seconds")
    if result["status"] == "completed":
        st.success(result["message"])
        st.dataframe(pd.DataFrame(result["rows"], columns=result["columns"]), use_container_width=True, hide_index=True)
    else:
        st.warning(result["message"])


with st.expander("Enterprise Audit Logs"):
    st.caption("Recent requests, newest first. Human approval decisions are recorded with each request.")
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
            st.caption(f"From: {message['from']} Â· To: {message['to']}")
            for line in message["body"].splitlines():
                if line.startswith(("Reason:", "Timestamp (UTC):")):
                    st.write(line)
            show_plan_links(message.get("plan_links", {}))
            if not message.get("plan_links"):
                st.caption("Legacy email. Submit a new request to generate plan links.")
    except (OSError, ValueError, UnicodeError):
        st.warning("Could not read the local email inbox. Check file permissions.")
