"""Shared quality-gate queries.

The quality report is the authority for chapter acceptance: a staged chapter
may only become accepted when the latest report for that exact artifact
revision is ACCEPT.
"""

from __future__ import annotations

import sqlite3


def latest_quality_decision(
    connection: sqlite3.Connection,
    artifact_id: str,
    artifact_revision: int,
) -> str | None:
    row = connection.execute(
        "SELECT decision FROM quality_reports WHERE artifact_id = ? AND artifact_revision = ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (artifact_id, artifact_revision),
    ).fetchone()
    return None if row is None else str(row[0])


def quality_accepted(
    connection: sqlite3.Connection,
    artifact_id: str,
    artifact_revision: int,
) -> bool:
    return latest_quality_decision(connection, artifact_id, artifact_revision) == "ACCEPT"


__all__ = ("latest_quality_decision", "quality_accepted")
