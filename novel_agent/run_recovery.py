from __future__ import annotations

import sqlite3

from novel_agent.run_models import RecoverableTask, RunCheckpoint, RunLease, RunRecoveryState


def load_recovery_state(connection: sqlite3.Connection, lease: RunLease) -> RunRecoveryState:
    task_row = connection.execute(
        "SELECT task_id, task_type, idempotency_key FROM tasks WHERE run_id = ? AND state = ? "
        "ORDER BY created_at, task_id LIMIT 1",
        (lease.run_id, "running"),
    ).fetchone()
    checkpoint_row = connection.execute(
        "SELECT task_id, stage, result_sha256, fencing_token, created_at FROM run_checkpoints "
        "WHERE run_id = ? ORDER BY created_at DESC, checkpoint_id DESC LIMIT 1",
        (lease.run_id,),
    ).fetchone()
    active_task = (
        None
        if task_row is None
        else RecoverableTask(task_id=str(task_row[0]), task_type=str(task_row[1]), idempotency_key=str(task_row[2]))
    )
    last_checkpoint = (
        None
        if checkpoint_row is None
        else RunCheckpoint(
            run_id=lease.run_id,
            task_id=str(checkpoint_row[0]),
            stage=str(checkpoint_row[1]),
            result_sha256=str(checkpoint_row[2]),
            fencing_token=int(checkpoint_row[3]),
            created_at=str(checkpoint_row[4]),
        )
    )
    return RunRecoveryState(lease=lease, active_task=active_task, last_checkpoint=last_checkpoint)


__all__ = ("load_recovery_state",)
