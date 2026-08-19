from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum, unique

from novel_agent.contracts import RunStatus
from novel_agent.run_event_store import append_run_event
from novel_agent.run_state import RunEvent
from novel_agent.storage import write_transaction


@unique
class QueuedRunFailureReason(StrEnum):
    SCHEDULER_DRAIN_FAILED = "scheduler_drain_failed"
    STAGE_EXECUTOR_FACTORY_FAILED = "stage_executor_factory_failed"


def fail_queued_run(
    connection: sqlite3.Connection,
    now: Callable[[], datetime],
    run_id: str,
    reason: QueuedRunFailureReason,
) -> bool:
    with write_transaction(connection):
        failed_at = now()
        updated = connection.execute(
            "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ? AND state = ?",
            (RunStatus.FAILED.value, failed_at.isoformat(), run_id, RunStatus.QUEUED.value),
        )
        if updated.rowcount != 1:
            return False
        append_run_event(connection, RunEvent(run_id, "run_failed", {"reason": reason}, failed_at))
        return True


__all__ = ("QueuedRunFailureReason", "fail_queued_run")
