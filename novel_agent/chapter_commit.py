from __future__ import annotations

from novel_agent.artifacts import enqueue_materialization
from novel_agent.chapter_audit import insert_chapter_audit_manifest
from novel_agent.chapter_commit_models import ChapterCommitContext
from novel_agent.chapter_errors import ChapterCommitBlocked, ChapterPersistenceError
from novel_agent.chapter_models import ChapterAcceptanceReceipt, ChapterAcceptanceRequest
from novel_agent.chapter_task_commit import complete_chapter_task
from novel_agent.quality_gate import quality_accepted
from novel_agent.memory_promotion import promote_candidates


def commit_chapter_acceptance(
    context: ChapterCommitContext,
    request: ChapterAcceptanceRequest,
) -> ChapterAcceptanceReceipt:
    connection = context.connection
    staged = request.staged
    _require_quality_accept(connection, staged.artifact_id.root, staged.revision)
    updated = connection.execute(
        "UPDATE artifacts SET status = 'accepted', is_current = ? WHERE artifact_id = ? AND revision = ? "
        "AND project_id = ? AND run_id = ? AND status = 'staged' AND sha256 = ?",
        (
            1 if request.activate else 0,
            staged.artifact_id.root,
            staged.revision,
            request.project_id.root,
            request.run_id,
            staged.raw_sha256,
        ),
    )
    if updated.rowcount != 1:
        raise ChapterPersistenceError("staged chapter revision is no longer eligible for acceptance")
    if request.activate:
        connection.execute(
            "UPDATE artifacts SET is_current = 0 WHERE artifact_id = ? AND revision <> ?",
            (staged.artifact_id.root, staged.revision),
        )
    promoted = _promote_candidates(context, request)
    enqueue_materialization(
        connection,
        project_id=request.project_id.root,
        artifact_id=staged.artifact_id.root,
        revision=staged.revision,
        raw_bytes=staged.raw_bytes,
        sha256=staged.raw_sha256,
        created_at=context.committed_at,
    )
    next_task_id = complete_chapter_task(context, request, staged.raw_sha256)
    insert_chapter_audit_manifest(context, request, staged.raw_sha256)
    return ChapterAcceptanceReceipt(staged.artifact_id, staged.revision, next_task_id, promoted)


def _require_quality_accept(
    connection,
    artifact_id: str,
    revision: int,
) -> None:
    """The quality gate is authoritative: no ACCEPT report, no commit."""
    if not quality_accepted(connection, artifact_id, revision):
        raise ChapterCommitBlocked("quality_not_accepted")


def _promote_candidates(
    context: ChapterCommitContext,
    request: ChapterAcceptanceRequest,
) -> tuple[str, ...]:
    """Promote the chapter's M0 candidates through the shared policy path.

    A conflict that the policy says must pause stops the acceptance with
    ``ChapterCommitBlocked`` so the run enters review instead of silently
    dropping the new fact.
    """
    batch = promote_candidates(
        context.connection,
        run_id=request.run_id,
        artifact_id=request.staged.artifact_id.root,
        artifact_revision=request.staged.revision,
        chapter_number=request.chapter_plan.chapter_number,
        committed_at=context.committed_at,
        created_by="chapter-coordinator:v1",
    )
    if batch.blocked_reason is not None:
        raise ChapterCommitBlocked(batch.blocked_reason)
    return batch.promoted


__all__ = ("commit_chapter_acceptance",)
