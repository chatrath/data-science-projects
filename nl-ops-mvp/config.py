"""Load configuration before importing any LangChain/LangGraph/LangSmith code."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)  # Existing process environment takes precedence.

# Accept current SDK names as well as the requested LANGCHAIN_* names.
for legacy, current, default in [
    ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "true"),
    ("LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", ""),
    ("LANGCHAIN_PROJECT", "LANGSMITH_PROJECT", "nl-ops-db-mvp"),
]:
    value = os.environ.get(current, os.environ.get(legacy, default))
    os.environ[legacy] = os.environ[current] = value

TRACING_REQUESTED = os.environ["LANGCHAIN_TRACING_V2"].lower() == "true"
TRACING_PROJECT = os.environ["LANGCHAIN_PROJECT"]
TRACING_ENDPOINT = os.environ["LANGCHAIN_ENDPOINT"]
_key = os.environ["LANGCHAIN_API_KEY"].strip()
TRACING_ENABLED = TRACING_REQUESTED and bool(_key) and _key not in {
    "your-langsmith-api-key-here", "your-langsmith-key"
}
# Keep the app usable with a placeholder; never attempt unauthenticated uploads.
os.environ["LANGCHAIN_TRACING_V2"] = os.environ["LANGSMITH_TRACING"] = str(TRACING_ENABLED).lower()


def log_tracing_status():
    logging.getLogger(__name__).info(
        "LangSmith tracing %s; project=%s; endpoint=%s. Remote delivery requires verification.",
        "configured" if TRACING_ENABLED else "inactive (disabled or API key missing)",
        TRACING_PROJECT, TRACING_ENDPOINT,
    )
