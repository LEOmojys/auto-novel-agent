from __future__ import annotations

import hashlib
import json
import sqlite3

from novel_agent.run_state import RunEvent


def append_run_event(connection: sqlite3.Connection, event: RunEvent) -> None:
    row = connection.execute(
        "SELECT COALESCE(MAX(sequence) + 1, 0) FROM run_events WHERE run_id = ?", (event.run_id,)
    ).fetchone()
    sequence = int(row[0])
    event_id = hashlib.sha256(f"{event.run_id}:{sequence}:{event.event_type}".encode()).hexdigest()
    connection.execute(
        "INSERT INTO run_events(event_id, run_id, sequence, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (event_id, event.run_id, sequence, event.event_type, json.dumps(event.payload, sort_keys=True), event.created_at.isoformat()),
    )


__all__ = ("append_run_event",)
