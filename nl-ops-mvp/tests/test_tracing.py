"""Verify status reporting and upload/readback without real credentials."""
import unittest
from unittest.mock import patch

import tracing


class TracingTests(unittest.TestCase):
    def test_inactive_never_connects(self):
        with patch.object(tracing, "TRACING_ENABLED", False), patch.object(tracing, "Client") as client:
            self.assertIn("Tracing inactive", tracing.verify_tracing())
            client.assert_not_called()

    def test_verified_requires_upload_and_readback(self):
        with patch.object(tracing, "TRACING_ENABLED", True), patch.object(tracing, "Client") as factory:
            result = tracing.verify_tracing()
            client = factory.return_value
            client.create_run.assert_called_once()
            client.update_run.assert_called_once()
            client.read_run.assert_called_once()
            client.close.assert_called_once()
            self.assertTrue(result.startswith("Verified:"))

    def test_remote_failure_does_not_expose_error_details(self):
        with patch.object(tracing, "TRACING_ENABLED", True), patch.object(tracing, "Client") as factory:
            factory.return_value.create_run.side_effect = RuntimeError("private credential detail")
            result = tracing.verify_tracing()
            self.assertIn("Verification failed", result)
            self.assertNotIn("private credential", result)
            factory.return_value.read_run.assert_not_called()

    def test_forbidden_has_actionable_message_without_secret(self):
        with patch.object(tracing, "TRACING_ENABLED", True), patch.object(tracing, "Client") as factory:
            factory.return_value.create_run.side_effect = RuntimeError("403 Forbidden private-key")
            result = tracing.verify_tracing()
            self.assertIn("HTTP 403", result)
            self.assertIn("fully restart", result)
            self.assertNotIn("private-key", result)
