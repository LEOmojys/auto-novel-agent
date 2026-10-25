from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from novel_agent.tool_call_ledger import (
    ToolCallLedgerWrite,
    ToolCallReservation,
    complete_tool_call,
    persist_tool_call,
    reserve_tool_call,
)

if TYPE_CHECKING:
    from novel_agent.tool_registry import ToolCallResult, ToolInvocationRequest, ToolSpec


@dataclass(frozen=True, slots=True)
class PersistedResultDecodeError(ValueError):
    detail: str

    def __str__(self) -> str:
        return f"persisted result decode failed: {self.detail}"


def decode_persisted_result(payload: str | None, spec: ToolSpec) -> ToolCallResult:
    from novel_agent.contracts import ToolOutcome
    from novel_agent.tool_registry import InvocationTrace, ToolCallResult

    if payload is None:
        raise PersistedResultDecodeError(detail="missing persisted result")
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as error:
        raise PersistedResultDecodeError(detail="invalid JSON") from error
    if not isinstance(raw, dict):
        raise PersistedResultDecodeError(detail="persisted result must be an object")
    try:
        data_raw = raw.get("data")
        data = None if data_raw is None else spec.output_model.model_validate(data_raw)
        return ToolCallResult(
            outcome=ToolOutcome(raw["outcome"]),
            data=data,
            reason=raw["reason"],
            trace=InvocationTrace.model_validate(raw["trace"]),
        )
    except KeyError as error:
        raise PersistedResultDecodeError(detail="missing required field") from error
    except (ValidationError, ValueError) as error:
        raise PersistedResultDecodeError(detail="invalid persisted result value") from error


def persist_result(
    connection: sqlite3.Connection,
    request: ToolInvocationRequest,
    spec: ToolSpec,
    result: ToolCallResult,
    input_hash: str,
) -> None:
    complete_tool_call(connection, _ledger_write(request, spec, result, input_hash))


def reserve_result(
    connection: sqlite3.Connection,
    request: ToolInvocationRequest,
    spec: ToolSpec,
    input_hash: str,
    *,
    max_attempts: int = 1,
    retry_after_seconds: float = 30.0,
) -> ToolCallReservation:
    return reserve_tool_call(
        connection,
        _ledger_write(request, spec, None, input_hash),
        max_attempts=max_attempts,
        retry_after_seconds=retry_after_seconds,
    )


def record_result(
    connection: sqlite3.Connection,
    request: ToolInvocationRequest,
    spec: ToolSpec,
    result: ToolCallResult,
    input_hash: str,
) -> None:
    """Insert a completed audit row for a non-idempotent tool call."""
    persist_tool_call(connection, _ledger_write(request, spec, result, input_hash))


def _ledger_write(
    request: ToolInvocationRequest,
    spec: ToolSpec,
    result: ToolCallResult | None,
    input_hash: str,
) -> ToolCallLedgerWrite:
    payload = None if result is None else json.dumps(
        {
            "outcome": result.outcome.value,
            "reason": result.reason,
            "data": None if result.data is None else json.loads(result.data.model_dump_json()),
            "trace": json.loads(result.trace.model_dump_json()),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    call_id = hashlib.sha256(f"{request.run_id}\0{request.idempotency_key}".encode()).hexdigest()
    return ToolCallLedgerWrite(
        call_id=call_id,
        run_id=request.run_id,
        task_id=request.task_id,
        tool_id=spec.tool_ref,
        idempotency_key=request.idempotency_key,
        request_sha256=input_hash,
        response_sha256=None if result is None else result.trace.output_sha256,
        state="pending" if result is None else result.outcome.value,
        result_payload_json=payload,
    )


__all__ = (
    "PersistedResultDecodeError",
    "decode_persisted_result",
    "persist_result",
    "reserve_result",
)
