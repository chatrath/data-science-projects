"""Loopback-only artifact downloads. No directory listing or arbitrary paths."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import re
import threading
from urllib.parse import urlsplit

import plan_artifacts


class PlanDownloadHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        match = re.fullmatch(r"/plans/([a-f0-9]{64})\.(xml)", path)
        if not match:
            self.send_error(404)
            return
        name = f"{match[1]}.{match[2]}"
        try:
            payload = (plan_artifacts.ARTIFACT_PATH / name).read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/xml; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="query_plan.{match[2]}"')
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


def start_plan_server():
    server = ThreadingHTTPServer(("127.0.0.1", int(os.getenv("PLAN_SERVER_PORT", "8502"))), PlanDownloadHandler)
    worker = threading.Thread(target=server.serve_forever, name="plan-downloads", daemon=True)
    worker.start()
    return server
