"""Tool-call retry/lease columns.

Version 8 adds an attempt counter to the idempotency ledger so a terminal
failure can be retried a bounded number of times, and a stale pending
reservation (crashed worker) can be distinguished from an active one.
"""

from typing import Final

TOOL_CALL_RETRY_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    "ALTER TABLE tool_calls ADD COLUMN attempts INTEGER NOT NULL DEFAULT 1",
)
