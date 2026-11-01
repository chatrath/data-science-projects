"""Verify persistent reuse, invalidation, and model-call suppression."""
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from cache import ObservedSQLiteCache, run_cached
from database import initialize_database


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "enterprise.db"
        self.results = self.root / "results.db"
        initialize_database(self.db)

    def run_request(self, request, live, limit=50):
        return run_cached(request, limit, live, self.db, self.results)

    def test_repeat_skips_live_work_and_survives_new_callback(self):
        live = Mock(return_value={"status": "completed", "rows": [[1]], "sql": "SELECT 1"})
        first = self.run_request("Show employees", live)
        second = self.run_request(" Show employees ", Mock(side_effect=AssertionError("Must not execute")))
        self.assertEqual(first["cache_source"], "live agents")
        self.assertEqual(second["cache_source"], "result cache")
        self.assertEqual(second["rows"], [[1]])
        live.assert_called_once()

    def test_data_change_and_row_limit_invalidate(self):
        live = Mock(side_effect=lambda: {"status": "completed", "rows": []})
        self.run_request("List customers", live)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("UPDATE customers SET country='Canada' WHERE id=1")
        self.run_request("List customers", live)
        self.run_request("List customers", live, limit=2)
        self.assertEqual(live.call_count, 3)

    def test_failures_not_cached_and_different_values_not_merged(self):
        live = Mock(side_effect=lambda: {"status": "blocked"})
        self.run_request("Delete customers", live)
        self.run_request("Delete customers", live)
        self.assertEqual(live.call_count, 2)
        live = Mock(side_effect=lambda: {"status": "completed", "rows": []})
        self.run_request("salary above 60000", live)
        self.run_request("salary above 70000", live)
        self.assertEqual(live.call_count, 2)

    def test_expired_results_recompute(self):
        live = Mock(side_effect=lambda: {"status": "completed", "rows": []})
        self.run_request("List customers", live)
        with patch("cache.time.time", return_value=9999999999):
            self.run_request("List customers", live)
        self.assertEqual(live.call_count, 2)

    def test_idempotent_initialization_preserves_cached_result(self):
        live = Mock(side_effect=lambda: {"status": "completed", "rows": []})
        self.run_request("List customers", live)
        initialize_database(self.db)
        result = self.run_request("List customers", live)
        self.assertEqual(result["cache_source"], "result cache")
        live.assert_called_once()

    def test_sqlite_llm_cache_skips_second_generation(self):
        previous = get_llm_cache()
        llm_cache = ObservedSQLiteCache(database_path=str(self.root / "llm.db"))
        try:
            set_llm_cache(llm_cache)
            model = FakeListChatModel(responses=["first generation", "second generation"], cache=True)
            self.assertEqual(model.invoke("same prompt").content, "first generation")
            self.assertEqual(model.invoke("same prompt").content, "first generation")
            self.assertEqual(model.i, 1)
            llm_cache.clear()
            self.assertEqual(model.invoke("same prompt").content, "second generation")
        finally:
            set_llm_cache(previous)
            llm_cache.engine.dispose()
