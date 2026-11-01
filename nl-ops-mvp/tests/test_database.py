"""Exercise the SQL security boundary against a disposable database."""
from pathlib import Path
import sqlite3
import tempfile
import unittest

from database import dry_run, execute_query, initialize_database


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.db"
        initialize_database(self.path)

    def test_seed_is_idempotent(self):
        initialize_database(self.path)
        self.assertEqual(execute_query("SELECT count(*) FROM customers", path=self.path)["rows"], [(4,)])

    def test_limit_and_columns(self):
        result = execute_query("SELECT id, email FROM customers ORDER BY id", 2, self.path)
        self.assertEqual(result["columns"], ["id", "email"])
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(result["truncated"])

    def test_dry_run_and_join(self):
        sql = "SELECT e.name, t.subject FROM tickets t JOIN employees e ON e.id=t.employee_id WHERE t.status='open'"
        dry_run(sql, self.path)
        self.assertEqual(len(execute_query(sql, path=self.path)["rows"]), 1)

    def test_writes_stacking_system_access_and_extensions_are_blocked(self):
        for sql in ["DELETE FROM customers", "DROP TABLE employees", "PRAGMA table_info(customers)",
                    "ATTACH DATABASE ':memory:' AS other", "SELECT * FROM sqlite_master",
                    "SELECT load_extension('anything')", "SELECT 1; DELETE FROM customers",
                    "SELECT * FROM missing_table"]:
            with self.subTest(sql=sql), self.assertRaises((ValueError, sqlite3.Error)):
                execute_query(sql, path=self.path)
        self.assertEqual(execute_query("SELECT count(*) FROM customers", path=self.path)["rows"], [(4,)])

    def test_invalid_limits(self):
        for limit in [0, -1, 201, True, 1.5]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                execute_query("SELECT id FROM customers", limit, self.path)


if __name__ == "__main__":
    unittest.main()
