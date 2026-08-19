from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, RootModel


_SENSITIVE_KEY_PARTS: Final[tuple[str, ...]] = (
    "api_key",
    "credential",
    "holder",
    "password",
    "prompt",
    "secret",
    "token",
)


class RunEventView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequence: int = Field(ge=0)
    event_type: str = Field(min_length=1)
    created_at: datetime
    payload_json: str


class _EventPayload(RootModel[dict[str, str | int | float | bool | None]]):
    model_config = ConfigDict(frozen=True, strict=True)


def read_run_events(connection: sqlite3.Connection, run_id: str) -> tuple[RunEventView, ...]:
    rows = connection.execute(
        "SELECT sequence, event_type, payload_json, created_at FROM run_events "
        "WHERE run_id = ? ORDER BY sequence, event_id",
        (run_id,),
    ).fetchall()
    return tuple(
        RunEventView(
            sequence=int(row[0]),
            event_type=str(row[1]),
            payload_json=_safe_payload(str(row[2])),
            created_at=datetime.fromisoformat(str(row[3])),
        )
        for row in rows
    )


def review_context(connection: sqlite3.Connection, run_id: str) -> tuple[str | None, str | None]:
    row = connection.execute(
        "SELECT event_id, event_type, payload_json FROM run_events WHERE run_id = ? "
        "AND event_type IN ('plan_blocked', 'chapter_waiting_review', 'chapter_rebase_proposed', "
        "'artifact_approval_required', 'plan_approval_required') "
        "ORDER BY sequence DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if row is None:
        return None, None
    event_id = str(row[0])
    if str(row[1]) == "chapter_rebase_proposed":
        return "chapter_rebase_confirmation_required", event_id
    if str(row[1]) == "artifact_approval_required":
        return "artifact_approval_required", event_id
    if str(row[1]) == "plan_approval_required":
        return "plan_approval_required", event_id
    value = _EventPayload.model_validate_json(str(row[2])).root.get("reason")
    return (value if isinstance(value, str) else None), event_id


def _safe_payload(payload_json: str) -> str:
    payload = _EventPayload.model_validate_json(payload_json)
    safe_payload = {
        key: "[redacted]" if any(part in key.casefold() for part in _SENSITIVE_KEY_PARTS) else value
        for key, value in payload.root.items()
    }
    return json.dumps(safe_payload, ensure_ascii=False, sort_keys=True)


__all__ = ("RunEventView", "read_run_events", "review_context")
