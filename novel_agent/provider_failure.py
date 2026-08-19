"""Provider failure traceability.

Every LLM call that fails records a structured row in ``provider_failures``
so operators can trace why a run degraded, retried, or stopped.  Recording is
best-effort: a locked database must never turn a provider failure into a
persistence failure.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from typing import Final

FAILURE_ID_PREFIX: Final = "pf"

_FAILURE_ID_SEED_PREFIX: Final = "provider-failure"


def record_provider_failure(
    connection: sqlite3.Connection | None,
    *,
    project_id: str,
    run_id: str,
    caller: str,
    model_name: str,
    failure_kind: str,
    reason: str,
    detail: str = "",
    prompt_sha256: str = "",
    request_excerpt: str = "",
    created_at: datetime | None = None,
) -> None:
    """Persist one provider failure for later forensics.

    ``connection`` may be None (callers without a live project database
    simply skip recording) or a project connection; the write is wrapped so
    any persistence error is swallowed.
    """
    if connection is None:
        return
    timestamp = created_at or datetime.now()
    failure_id = _failure_id(run_id, caller, prompt_sha256, timestamp)
    try:
        connection.execute(
            "INSERT INTO provider_failures(failure_id, project_id, run_id, caller, model_name, "
            "failure_kind, reason, detail, prompt_sha256, request_excerpt, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(failure_id) DO NOTHING",
            (
                failure_id,
                project_id,
                run_id,
                caller[:120],
                model_name[:120],
                failure_kind[:64],
                reason[:500],
                detail[:2_000],
                prompt_sha256,
                request_excerpt[:500],
                timestamp.isoformat(),
            ),
        )
    except sqlite3.Error:
        return


def read_provider_failures(
    connection: sqlite3.Connection,
    run_id: str,
    limit: int = 100,
) -> tuple[dict[str, str], ...]:
    """Return recent provider failures for a run, newest first."""
    rows = connection.execute(
        "SELECT failure_id, caller, model_name, failure_kind, reason, detail, "
        "prompt_sha256, request_excerpt, created_at FROM provider_failures "
        "WHERE run_id = ? ORDER BY created_at DESC, failure_id LIMIT ?",
        (run_id, limit),
    ).fetchall()
    return tuple(
        {
            "failure_id": str(row[0]),
            "caller": str(row[1]),
            "model_name": str(row[2]),
            "failure_kind": str(row[3]),
            "reason": str(row[4]),
            "detail": str(row[5]),
            "prompt_sha256": str(row[6]),
            "request_excerpt": str(row[7]),
            "created_at": str(row[8]),
        }
        for row in rows
    )


def _failure_id(run_id: str, caller: str, prompt_sha256: str, created_at: datetime) -> str:
    seed = ":".join((_FAILURE_ID_SEED_PREFIX, run_id, caller, prompt_sha256, created_at.isoformat()))
    return f"{FAILURE_ID_PREFIX}:{hashlib.sha256(seed.encode()).hexdigest()[:40]}"


__all__ = ("read_provider_failures", "record_provider_failure")
