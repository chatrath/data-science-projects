"""Persistent exact LLM cache plus short-lived, data-aware workflow results."""
import config  # .env must precede LangChain imports.

from contextlib import closing
from contextvars import ContextVar
from datetime import date
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from audit_logger import append_audit_event, request_identity

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / ".langchain_cache.db"
RESULT_PATH = ROOT / ".query_cache.db"
RESULT_TTL = 300
_lock = threading.RLock()
_hits = ContextVar("llm_cache_hits", default=0)


class ObservedSQLiteCache(SQLiteCache):
    def lookup(self, prompt, llm_string):
        result = super().lookup(prompt, llm_string)
        if result is not None:
            _hits.set(_hits.get() + 1)
        return result


@lru_cache(maxsize=1)
def initialize_cache():
    cache = ObservedSQLiteCache(database_path=str(CACHE_PATH))
    set_llm_cache(cache)
    return cache


def _connection(path=RESULT_PATH):
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, created REAL, payload TEXT)")
    return conn


def result_key(request, row_limit, database_path):
    """Conservative matching: trim outer whitespace; preserve values and case."""
    digest = hashlib.sha256()
    for path in [Path(database_path), Path(str(database_path) + "-wal"),
                 ROOT / "agents.py", ROOT / "graph.py", ROOT / "database.py", ROOT / "guardrails.py",
                 ROOT / "schema_context.py", ROOT / "query_plan.py", ROOT / "plan_artifacts.py", ROOT / "tools.py"]:
        digest.update(str(path.resolve()).encode())
        if path.exists():
            digest.update(path.read_bytes())
    context = [request.strip(), row_limit, date.today().isoformat(),
               os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct"),
               os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
               os.getenv("PLAN_BASE_URL", "http://127.0.0.1:8502")]
    digest.update(json.dumps(context, ensure_ascii=False).encode())
    return digest.hexdigest()


def run_cached(request, row_limit, run_live, database_path, result_path=RESULT_PATH):
    """Return successful cached results or execute once; never cache failures."""
    started = time.perf_counter()
    identity = request_identity(request)
    with _lock:
        key = result_key(request, row_limit, database_path)
        with closing(_connection(result_path)) as conn, conn:
            conn.execute("DELETE FROM results WHERE created < ?", (time.time() - RESULT_TTL,))
            row = conn.execute("SELECT payload FROM results WHERE key=?", (key,)).fetchone()
        if row:
            result = json.loads(row[0])
            result.update(request=request, cache_source="result cache", llm_cache_hits=0)
            result.update(identity, audit_written=False)
            append_audit_event(result, cached=True)
            result["audit_written"] = True
        else:
            token = _hits.set(0)
            try:
                result = run_live()
                result.update(cache_source="agents / LLM cache" if _hits.get() else "live agents",
                              llm_cache_hits=_hits.get())
            finally:
                _hits.reset(token)
            if not result.get("audit_written") and not result.get("audit_error"):
                result.update(identity)
                append_audit_event(result)
                result["audit_written"] = True
            if result.get("status") == "completed" and key == result_key(request, row_limit, database_path):
                with closing(_connection(result_path)) as conn, conn:
                    conn.execute("INSERT OR REPLACE INTO results VALUES (?, ?, ?)",
                                 (key, time.time(), json.dumps(result)))
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return result


def clear_caches():
    with _lock:
        initialize_cache().clear()
        with closing(_connection()) as conn, conn:
            conn.execute("DELETE FROM results")


if __name__ == "__main__":
    clear_caches()
    print("Cleared local LLM and query result caches.")
