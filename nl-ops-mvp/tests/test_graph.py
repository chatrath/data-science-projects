"""Test actual compiled routing without external model requests."""
from conftest import invoke_reviewed
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents import ModelConfigurationError, SafetyDecision
import database
from graph import graph


class GraphTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "graph.db"
        database.initialize_database(self.path)
        for name, replacement in [
            ("get_schema", lambda: database.get_schema(self.path)),
            ("read_only_connection", lambda: database.read_only_connection(self.path)),
            ("execute_query", lambda sql, limit: database.execute_query(sql, limit, self.path)),
        ]:
            patcher = patch(f"graph.{name}", replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_success(self):
        with patch("graph.check_intent", return_value=SafetyDecision(allowed=True, reason="Read only")), \
             patch("graph.generate_sql", return_value="SELECT id FROM customers WHERE id > 0 ORDER BY id"):
            result = invoke_reviewed(graph, {"request": "List customers", "row_limit": 2})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(result["truncated"])

    def test_safety_blocks_generation(self):
        with patch("graph.check_intent", return_value=SafetyDecision(allowed=False, reason="Writes prohibited")), \
             patch("graph.generate_sql") as generate:
            result = invoke_reviewed(graph, {"request": "Delete customers"})
        self.assertEqual(result["status"], "blocked")
        generate.assert_not_called()

    def test_guardrail_rejects_unsafe_model_output_without_execution(self):
        with patch("graph.check_intent", return_value=SafetyDecision(allowed=True, reason="Read only")), \
             patch("graph.generate_sql", return_value="DELETE FROM customers"), \
             patch("graph.inspect_query_plan") as plan, patch("graph.execute_query") as execute:
            result = invoke_reviewed(graph, {"request": "List customers"})
        self.assertEqual(result["status"], "blocked")
        plan.assert_not_called()
        execute.assert_not_called()

    def test_full_scan_is_intercepted_before_execution(self):
        with patch("graph.check_intent", return_value=SafetyDecision(allowed=True, reason="Read only")), \
             patch("graph.generate_sql", return_value="SELECT * FROM employees"), \
             patch("graph.execute_query") as execute:
            result = invoke_reviewed(graph, {"request": "List all employees"})
        self.assertEqual(result["status"], "blocked")
        self.assertIn("unindexed full-table scan", result["message"])
        execute.assert_not_called()

    def test_plan_failure_stops_execution(self):
        with patch("graph.check_intent", return_value=SafetyDecision(allowed=True, reason="Read only")), \
             patch("graph.generate_sql", return_value="SELECT * FROM employees WHERE id=1"), \
             patch("graph.inspect_query_plan", side_effect=RuntimeError("Inspection failed")), \
             patch("graph.execute_query") as execute, self.assertLogs("graph", level="ERROR"):
            result = invoke_reviewed(graph, {"request": "Find employee 1"})
        self.assertEqual(result["status"], "error")
        execute.assert_not_called()

    def test_guardrail_failure_stops_execution(self):
        with patch("graph.check_intent", return_value=SafetyDecision(allowed=True, reason="Read only")), \
             patch("graph.generate_sql", return_value="SELECT id FROM customers"), \
             patch("graph.validate_sql", side_effect=RuntimeError("Unexpected validator failure")), \
             patch("graph.execute_query") as execute, self.assertLogs("graph", level="ERROR"):
            result = invoke_reviewed(graph, {"request": "List customers"})
        self.assertEqual(result["status"], "error")
        execute.assert_not_called()

    def test_empty_input_skips_model(self):
        with patch("graph.check_intent") as check:
            result = invoke_reviewed(graph, {"request": " "})
        self.assertEqual(result["status"], "blocked")
        check.assert_not_called()

    def test_model_failure_stops_before_sql(self):
        with patch("graph.check_intent", side_effect=RuntimeError("Unavailable")), \
             patch("graph.generate_sql") as generate, \
             self.assertLogs("graph", level="ERROR"):
            result = invoke_reviewed(graph, {"request": "List customers"})
        self.assertEqual(result["status"], "error")
        generate.assert_not_called()

    def test_configuration_error_is_actionable(self):
        message = "Configure OLLAMA_MODEL in .env."
        with patch("graph.check_intent", side_effect=ModelConfigurationError(message)), \
             patch("graph.generate_sql") as generate:
            result = invoke_reviewed(graph, {"request": "List customers"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], message)
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
