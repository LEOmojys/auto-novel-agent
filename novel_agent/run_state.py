from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import assert_never

from novel_agent.contracts import RunCommandType, RunStatus


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    event_type: str
    payload: dict[str, str | int]
    created_at: datetime


def command_id(run_id: str, idempotency_key: str) -> str:
    return hashlib.sha256(f"{run_id}:{idempotency_key}".encode()).hexdigest()


def next_state(current: RunStatus, command: RunCommandType) -> RunStatus | None:
    match command:
        case RunCommandType.PAUSE:
            return RunStatus.PAUSED if current is RunStatus.RUNNING else None
        case RunCommandType.REPAIR:
            return RunStatus.QUEUED if current is RunStatus.WAITING_REVIEW else None
        case RunCommandType.REVISE:
            return RunStatus.QUEUED if current is RunStatus.SUCCEEDED else None
        case RunCommandType.RESUME:
            return RunStatus.QUEUED if current is RunStatus.PAUSED else None
        case RunCommandType.CANCEL:
            match current:
                case RunStatus.QUEUED | RunStatus.PAUSED | RunStatus.WAITING_REVIEW:
                    # Waiting for a worker / paused / queued: nothing is
                    # executing, so the cancel completes immediately.
                    return RunStatus.CANCELLED
                case RunStatus.RUNNING:
                    # A worker is executing; mark it so the worker aborts and
                    # the stage finalizes the run as cancelled.
                    return RunStatus.CANCEL_REQUESTED
                case RunStatus.CANCELLED | RunStatus.CANCEL_REQUESTED | RunStatus.FAILED | RunStatus.SUCCEEDED:
                    return None
                case unreachable:
                    assert_never(unreachable)
        case RunCommandType.START:
            return None
        case unreachable:
            assert_never(unreachable)


__all__ = ("RunEvent", "command_id", "next_state")
