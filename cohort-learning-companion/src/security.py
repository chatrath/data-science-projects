"""Loopback-origin enforcement and bounded, non-queuing request admission."""
from __future__ import annotations
import os
import re
from pathlib import Path
from dotenv import dotenv_values
from threading import Lock
from urllib.parse import urlsplit

LOCAL_ORIGIN_REGEX = r"http://(?:localhost|127\.0\.0\.1)(?::[0-9]{1,5})?"


def trusted_origin(origin: str) -> bool:
    if not re.fullmatch(LOCAL_ORIGIN_REGEX, origin):
        return False
    try:
        parsed = urlsplit(origin)
        return parsed.port is None or 1 <= parsed.port <= 65535
    except ValueError:
        return False


class LocalOriginMiddleware:
    """Reject untrusted browser origins for HTTP AND WebSocket before handlers.

    Requests without Origin remain available to local CLI/health clients.
    Origin is a browser defense, not authentication for other local programs.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] in {'http', 'websocket'}:
            origins = [v.decode('latin-1') for k, v in scope.get('headers', []) if k.lower() == b'origin']
            if origins and (len(origins) != 1 or not trusted_origin(origins[0])):
                if scope['type'] == 'websocket':
                    await send({'type': 'websocket.close', 'code': 1008})
                else:
                    from starlette.responses import JSONResponse
                    await JSONResponse({'detail': 'Origin is not allowed'}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def install_origin_security():
    """Install before Chainlit starts; also supports direct chainlit run app.py."""
    from chainlit.server import app
    from starlette.middleware.cors import CORSMiddleware
    if getattr(app.state, 'local_origin_security_installed', False):
        return
    # Chainlit HTTP CORS and its mounted Socket.IO server are separate layers.
    # Our outer ASGI middleware guards both; configure HTTP response headers too.
    for middleware in app.user_middleware:
        if middleware.cls is CORSMiddleware:
            middleware.kwargs['allow_origins'] = []
            middleware.kwargs['allow_origin_regex'] = LOCAL_ORIGIN_REGEX
    app.add_middleware(LocalOriginMiddleware)
    app.state.local_origin_security_installed = True


class RequestLimiter:
    """Process-wide cap with immediate rejection, no unbounded waiting queue."""
    def __init__(self, limit: int = 2):
        if not 1 <= limit <= 8:
            raise ValueError('MAX_CONCURRENT_REQUESTS must be between 1 and 8')
        self.limit = limit
        self.active = 0
        self._lock = Lock()

    def acquire(self) -> bool:
        with self._lock:
            if self.active >= self.limit:
                return False
            self.active += 1
            return True

    def release(self):
        with self._lock:
            self.active = max(0, self.active - 1)


def configured_request_limit() -> int:
    # Invalid configuration fails safely to the conservative default.
    try:
        config = dotenv_values(Path(__file__).resolve().parents[1] / '.env')
        value = int(config.get('MAX_CONCURRENT_REQUESTS') or os.getenv('MAX_CONCURRENT_REQUESTS', '2'))
        return value if 1 <= value <= 8 else 2
    except ValueError:
        return 2
