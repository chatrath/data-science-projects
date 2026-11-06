"""Shared, serializable failure taxonomy for models, tools, SQL, and storage."""
from dataclasses import dataclass
import sqlite3
import errno
import httpx
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError
from sqlglot.errors import SqlglotError


class OperationError(ValueError):
    """An operation failed with an explicit category and retry policy."""
    def __init__(self, message: str, *, kind: str, retryable: bool = False, tool_state: dict | None = None):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.tool_state = tool_state


@dataclass(frozen=True)
class Failure:
    kind: str
    retryable: bool
    message: str


def classify_error(exc: Exception, *, source: str = "workflow") -> Failure:
    """Classify by exception type/codes; never infer model outages from any error."""
    if isinstance(exc, OperationError):
        return Failure(exc.kind, exc.retryable, str(exc))
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return Failure("timeout", True, "The operation timed out. You can retry from the saved checkpoint.")
    if isinstance(exc, httpx.TransportError):
        return Failure("infrastructure", True, "The model service could not be reached. Check the service, then retry.")
    if isinstance(exc, (ValidationError, OutputParserException)):
        return Failure("model_format", False, "The model returned an invalid structured response.")
    if isinstance(exc, SqlglotError):
        return Failure("sql_syntax", False, str(exc))
    if isinstance(exc, sqlite3.Error):
        code = getattr(exc, "sqlite_errorcode", 0) or 0
        busy = (code & 255) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
        # Tests and drivers may omit extended SQLite error codes.
        busy = busy or str(exc).casefold() in {"database is locked", "database table is locked"}
        return Failure("database", busy, "Database is busy. Retry from the checkpoint." if busy else str(exc))
    if isinstance(exc, OSError):
        retryable = exc.errno not in {errno.EACCES, errno.EPERM, errno.EROFS}
        return Failure("infrastructure", retryable, f"{source.title()} storage or service is unavailable.")
    return Failure("infrastructure", False, f"{source.title()} failed unexpectedly. Check server logs.")


def is_sql_correction(exc: sqlite3.OperationalError) -> bool:
    """Retry SQL generation only for query defects, never database locks/outages."""
    message = str(exc).casefold()
    return any(marker in message for marker in ("syntax error", "incomplete input", "no such column", "no such table", "ambiguous column"))
