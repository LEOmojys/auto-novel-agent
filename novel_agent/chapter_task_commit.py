from __future__ import annotations

import hashlib

from novel_agent.chapter_commit_models import ChapterCommitContext
from novel_agent.chapter_errors import ChapterTaskStateError
from novel_agent.chapter_models import ChapterAcceptanceRequest
from novel_agent.run_event_store import append_run_event
from novel_agent.run_state import RunEvent


def complete_chapter_task(
    context: ChapterCommitContext,
    request: ChapterAcceptanceRequest,
    result_sha256: str,
) -> str | None:
    updated = context.connection.execute(
        "UPDATE tasks SET state = ?, updated_at = ? WHERE task_id = ? AND run_id = ? AND state = ?",
        ("succeeded", context.committed_at.isoformat(), request.task_id, request.run_id, "running"),
    )
    if updated.rowcount != 1:
        raise ChapterTaskStateError(request.task_id)
    context.connection.execute(
        "INSERT INTO task_results(task_id, run_id, result_sha256, committed_fence, created_at) VALUES (?, ?, ?, ?, ?)",
        (request.task_id, request.run_id, result_sha256, context.fencing_token, context.committed_at.isoformat()),
    )
    checkpoint_id = hashlib.sha256(f"{request.task_id}:{result_sha256}".encode()).hexdigest()
    context.connection.execute(
        "INSERT INTO run_checkpoints(checkpoint_id, run_id, task_id, stage, result_sha256, fencing_token, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            checkpoint_id,
            request.run_id,
            request.task_id,
            "chapter_accepted",
            result_sha256,
            context.fencing_token,
            context.committed_at.isoformat(),
        ),
    )
    append_run_event(
        context.connection,
        RunEvent(request.run_id, "chapter_accepted", {"chapter": request.chapter_plan.chapter_number, "task": request.task_id}, context.committed_at),
    )
    if request.next_chapter is None:
        context.connection.execute(
            "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
            ("succeeded", context.committed_at.isoformat(), request.run_id),
        )
        append_run_event(context.connection, RunEvent(request.run_id, "run_succeeded", {}, context.committed_at))
        return None
    next_task_id = f"{request.run_id}:chapter:{request.next_chapter.chapter_plan_id}"
    context.connection.execute(
        "INSERT INTO tasks(task_id, run_id, task_type, state, idempotency_key, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(task_id) DO NOTHING",
        (
            next_task_id,
            request.run_id,
            "chapter",
            "pending",
            next_task_id,
            context.committed_at.isoformat(),
            context.committed_at.isoformat(),
        ),
    )
    append_run_event(context.connection, RunEvent(request.run_id, "chapter_queued", {"task": next_task_id}, context.committed_at))
    return next_task_id


__all__ = ("complete_chapter_task",)
