from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

from novel_agent.sqlite_transactions import write_transaction


@dataclass(frozen=True, slots=True)
class ToolCallLedgerWrite:
    call_id: str
    run_id: str
    task_id: str
    tool_id: str
    idempotency_key: str
    request_sha256: str
    response_sha256: str | None
    state: str
    result_payload_json: str | None
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class ToolCallReservation:
    status: Literal["claimed", "existing"]
    tool_id: str
    request_sha256: str
    state: str
    result_payload_json: str | None
    attempts: int = 1


def persist_tool_call(connection: sqlite3.Connection, write: ToolCallLedgerWrite) -> None:
    with write_transaction(connection):
        connection.execute(
            "INSERT INTO tool_calls(call_id, run_id, task_id, tool_id, idempotency_key, "
            "request_sha256, response_sha256, state, result_payload_json, attempts, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (
                write.call_id,
                write.run_id,
                write.task_id,
                write.tool_id,
                write.idempotency_key,
                write.request_sha256,
                write.response_sha256,
                write.state,
                write.result_payload_json,
                write.attempts,
            ),
        )


def reserve_tool_call(
    connection: sqlite3.Connection,
    write: ToolCallLedgerWrite,
    *,
    max_attempts: int = 1,
    retry_after_seconds: float = 30.0,
    stale_after_seconds: float = 300.0,
) -> ToolCallReservation:
    """Claim a tool-call key, reclaiming stale pending or retryable failures.

    A fresh pending row means another worker is executing, so the caller gets
    IDEMPOTENCY_IN_PROGRESS.  A pending row older than stale_after_seconds is
    treated as a crashed worker and can be reclaimed.  A terminal failed row
    younger than retry_after_seconds is replayed, but afterwards it may be
    retried while attempts < max_attempts.
    """
    with write_transaction(connection):
        row = connection.execute(
            "SELECT tool_id, request_sha256, state, result_payload_json, attempts, "
            "(julianday('now') - julianday(created_at)) * 86400.0 AS age_seconds "
            "FROM tool_calls WHERE run_id = ? AND idempotency_key = ?",
            (write.run_id, write.idempotency_key),
        ).fetchone()
        if row is not None:
            existing_attempts = int(row[4] or 1)
            age_seconds = float(row[5] or 0.0)
            state = str(row[2])
            reclaim = (
                state == "pending" and age_seconds >= stale_after_seconds
            ) or (
                state == "failed"
                and existing_attempts < max_attempts
                and age_seconds >= retry_after_seconds
            )
            if not reclaim:
                return ToolCallReservation(
                    status="existing",
                    tool_id=str(row[0]),
                    request_sha256=str(row[1]),
                    state=state,
                    result_payload_json=None if row[3] is None else str(row[3]),
                    attempts=existing_attempts,
                )
            connection.execute(
                "DELETE FROM tool_calls WHERE run_id = ? AND idempotency_key = ?",
                (write.run_id, write.idempotency_key),
            )
            attempts = existing_attempts + 1
        else:
            attempts = max(1, write.attempts)
        connection.execute(
            "INSERT INTO tool_calls(call_id, run_id, task_id, tool_id, idempotency_key, "
            "request_sha256, response_sha256, state, result_payload_json, attempts, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?, CURRENT_TIMESTAMP)",
            (
                write.call_id,
                write.run_id,
                write.task_id,
                write.tool_id,
                write.idempotency_key,
                write.request_sha256,
                None,
                attempts,
            ),
        )
        return ToolCallReservation(
            status="claimed",
            tool_id=write.tool_id,
            request_sha256=write.request_sha256,
            state="pending",
            result_payload_json=None,
            attempts=attempts,
        )


def complete_tool_call(connection: sqlite3.Connection, write: ToolCallLedgerWrite) -> None:
    with write_transaction(connection):
        updated = connection.execute(
            "UPDATE tool_calls SET response_sha256 = ?, state = ?, result_payload_json = ? "
            "WHERE run_id = ? AND idempotency_key = ? AND state = 'pending'",
            (
                write.response_sha256,
                write.state,
                write.result_payload_json,
                write.run_id,
                write.idempotency_key,
            ),
        ).rowcount
        if updated != 1:
            raise sqlite3.IntegrityError("tool call reservation is no longer pending")


__all__ = (
    "ToolCallLedgerWrite",
    "ToolCallReservation",
    "complete_tool_call",
    "persist_tool_call",
    "reserve_tool_call",
)
