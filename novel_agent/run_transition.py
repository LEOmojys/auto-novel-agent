from __future__ import annotations

import sqlite3

from novel_agent.contracts import RunCommandType, RunStatus
from novel_agent.run_state import next_state


def resolve_control_transition(
    connection: sqlite3.Connection,
    run_id: str,
    current: RunStatus,
    command_type: RunCommandType,
) -> RunStatus | None:
    resolved = next_state(current, command_type)
    if resolved is not None or current is not RunStatus.WAITING_REVIEW or command_type is not RunCommandType.RESUME:
        return resolved
    row = connection.execute(
        "SELECT event_type FROM run_events WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return RunStatus.QUEUED if row is not None and str(row[0]) == "chapter_rebase_proposed" else None


__all__ = ("resolve_control_transition",)
