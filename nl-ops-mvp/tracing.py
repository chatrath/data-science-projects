"""Explicit, small end-to-end trace upload/readback check; no LLM required."""
from config import TRACING_ENABLED, TRACING_ENDPOINT, TRACING_PROJECT

from datetime import datetime, timezone
import os
import re
import time
from uuid import uuid4

from langsmith import Client


def verify_tracing() -> str:
    if not TRACING_ENABLED:
        return "Tracing inactive. Set LANGCHAIN_TRACING_V2=true and a real LANGCHAIN_API_KEY in .env, then restart the app."
    run_id = uuid4()
    client = Client(api_url=TRACING_ENDPOINT, api_key=os.environ["LANGCHAIN_API_KEY"],
                    auto_batch_tracing=False, timeout_ms=10000)
    try:
        client.create_run(name="LangSmith tracing verification", run_type="chain",
                          inputs={"check": "configuration"}, id=run_id,
                          start_time=datetime.now(timezone.utc), project_name=TRACING_PROJECT)
        client.update_run(run_id, outputs={"verified": True}, end_time=datetime.now(timezone.utc))
        # Ingestion/readback may be briefly eventually consistent.
        for attempt in range(3):
            try:
                client.read_run(run_id)
                return f"Verified: trace uploaded and read back from {TRACING_PROJECT}. Run ID: {run_id}"
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
    except Exception as exc:
        # SDK errors can contain headers or server details; do not expose them.
        codes = set(re.findall(r"\b(?:401|403|404|429)\b", str(exc)))
        if codes & {"401", "403"}:
            return ("Verification failed: LangSmith rejected access (HTTP "
                    + ("403" if "403" in codes else "401")
                    + "). Create a LangSmith API key in Settings > API Keys for the target "
                    "workspace, ensure it can create/write traces and read runs, replace "
                    "LANGCHAIN_API_KEY in .env, and fully restart the Streamlit server. "
                    "A browser refresh does not reload server credentials. Check that "
                    "LANGCHAIN_ENDPOINT matches your account's US or EU region.")
        if "429" in codes:
            return "Verification failed: LangSmith rate limit reached (HTTP 429). Wait and retry."
        return "Verification failed. Check your LangSmith API key, workspace permissions, endpoint/region, and network access."
    finally:
        client.close()


if __name__ == "__main__":
    print(verify_tracing())
