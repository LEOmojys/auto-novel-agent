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


@dataclass(frozen=True, slots=True)
class ToolCallReservation:
    status: Literal["claimed", "existing"]
    tool_id: str
    request_sha256: str
    state: str
    result_payload_json: str | None


def persist_tool_call(connection: sqlite3.Connection, write: ToolCallLedgerWrite) -> None:
    with write_transaction(connection):
        connection.execute(
            "INSERT INTO tool_calls(call_id, run_id, task_id, tool_id, idempotency_key, "
            "request_sha256, response_sha256, state, result_payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
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
            ),
        )


def reserve_tool_call(connection: sqlite3.Connection, write: ToolCallLedgerWrite) -> ToolCallReservation:
    with write_transaction(connection):
        row = connection.execute(
            "SELECT tool_id, request_sha256, state, result_payload_json "
            "FROM tool_calls WHERE run_id = ? AND idempotency_key = ?",
            (write.run_id, write.idempotency_key),
        ).fetchone()
        if row is not None:
            return ToolCallReservation(
                status="existing",
                tool_id=str(row[0]),
                request_sha256=str(row[1]),
                state=str(row[2]),
                result_payload_json=None if row[3] is None else str(row[3]),
            )
        connection.execute(
            "INSERT INTO tool_calls(call_id, run_id, task_id, tool_id, idempotency_key, "
            "request_sha256, response_sha256, state, result_payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, CURRENT_TIMESTAMP)",
            (
                write.call_id,
                write.run_id,
                write.task_id,
                write.tool_id,
                write.idempotency_key,
                write.request_sha256,
                None,
            ),
        )
        return ToolCallReservation(
            status="claimed",
            tool_id=write.tool_id,
            request_sha256=write.request_sha256,
            state="pending",
            result_payload_json=None,
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
