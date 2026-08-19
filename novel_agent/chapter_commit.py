from __future__ import annotations

import hashlib
from typing import assert_never

from novel_agent.chapter_audit import insert_chapter_audit_manifest
from novel_agent.chapter_commit_models import ChapterCommitContext
from novel_agent.chapter_errors import ChapterCommitBlocked, ChapterPersistenceError
from novel_agent.chapter_models import ChapterAcceptanceReceipt, ChapterAcceptanceRequest
from novel_agent.chapter_task_commit import complete_chapter_task
from novel_agent.memory_models import CandidateEvidence, CandidateStatus, MemoryCandidateRecord, PromotionOutcome
from novel_agent.memory_policy import decide_promotion
from novel_agent.memory_store import build_evidence, load_source, memory_id, record_status
from novel_agent.pathing import ArtifactExtension, ArtifactName


def commit_chapter_acceptance(
    context: ChapterCommitContext,
    request: ChapterAcceptanceRequest,
) -> ChapterAcceptanceReceipt:
    connection = context.connection
    staged = request.staged
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
    target = context.path_resolver.resolve_export(
        request.project_id,
        ArtifactName(
            artifact_id=staged.artifact_id,
            content_hash=staged.raw_sha256,
            extension=ArtifactExtension.JSON,
        ),
    )
    outbox_id = hashlib.sha256(
        f"{staged.artifact_id.root}:{staged.revision}:{staged.raw_sha256}".encode()
    ).hexdigest()
    timestamp = context.committed_at.isoformat()
    connection.execute(
        "INSERT INTO artifact_outbox(outbox_id, artifact_id, artifact_revision, target_path, state, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
        (outbox_id, staged.artifact_id.root, staged.revision, str(target), timestamp, timestamp),
    )
    next_task_id = complete_chapter_task(context, request, staged.raw_sha256)
    insert_chapter_audit_manifest(context, request, staged.raw_sha256)
    return ChapterAcceptanceReceipt(staged.artifact_id, staged.revision, next_task_id, promoted)


def _promote_candidates(
    context: ChapterCommitContext,
    request: ChapterAcceptanceRequest,
) -> tuple[str, ...]:
    rows = context.connection.execute(
        "SELECT candidate_id, candidate_json FROM memory_candidates WHERE run_id = ? "
        "AND source_artifact_id = ? AND source_artifact_revision = ? ORDER BY candidate_id",
        (request.run_id, request.staged.artifact_id.root, request.staged.revision),
    ).fetchall()
    promoted: list[str] = []
    for row in rows:
        candidate = MemoryCandidateRecord.model_validate_json(str(row[1]))
        evidence = build_evidence(
            candidate.evidence.proposal,
            load_source(context.connection, candidate.evidence.proposal),
        )
        decision = decide_promotion(evidence, _has_conflict(context, evidence))
        match decision.outcome:
            case PromotionOutcome.PROMOTED:
                identifier = _insert_memory(context, request, str(row[0]), evidence)
                promoted.append(identifier)
            case PromotionOutcome.REVIEW_REQUIRED | PromotionOutcome.CONFLICTED:
                context.connection.execute(
                    "UPDATE memory_candidates SET status = ? WHERE candidate_id = ?",
                    (decision.candidate_status.value, str(row[0])),
                )
                _record_conflict(context, request, str(row[0]), decision)
                if decision.requires_pause:
                    raise ChapterCommitBlocked("memory_conflict")
            case unreachable:
                assert_never(unreachable)
    return tuple(promoted)


def _record_conflict(
    context: ChapterCommitContext,
    request: ChapterAcceptanceRequest,
    candidate_id: str,
    decision,
) -> None:
    """Persist a conflicted/review-required promotion decision for auditability."""
    import hashlib

    conflict_id = hashlib.sha256(
        f"{request.run_id}:{candidate_id}:{decision.review_code}".encode()
    ).hexdigest()[:32]
    context.connection.execute(
        "INSERT INTO memory_conflicts(conflict_id, project_id, run_id, candidate_id, existing_memory_id, "
        "reason_code, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            conflict_id,
            request.project_id.root,
            request.run_id,
            candidate_id,
            None,
            decision.review_code.value if decision.review_code is not None else decision.outcome.value,
            context.committed_at.isoformat(),
        ),
    )


def _has_conflict(context: ChapterCommitContext, evidence: CandidateEvidence) -> bool:
    proposal = evidence.proposal
    if proposal.subject_id is None:
        return False
    rows = context.connection.execute(
        "SELECT value_json FROM memory_records current WHERE project_id = ? AND layer = ? "
        "AND subject_id = ? AND predicate = ? AND NOT EXISTS (SELECT 1 FROM memory_records "
        "successor WHERE successor.supersedes_id = current.memory_id)",
        (proposal.project_id.root, proposal.target_layer.value, proposal.subject_id, proposal.predicate),
    ).fetchall()
    return any(str(row[0]) != proposal.value_json for row in rows)


def _insert_memory(
    context: ChapterCommitContext,
    request: ChapterAcceptanceRequest,
    candidate_identifier: str,
    evidence: CandidateEvidence,
) -> str:
    proposal = evidence.proposal
    if proposal.subject_id is None:
        raise ChapterPersistenceError("promotable chapter memory has no resolved subject")
    identifier = memory_id(candidate_identifier)
    context.connection.execute(
        "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, "
        "subject_id, predicate, value_json, chapter_start, chapter_end, source_artifact_id, "
        "source_artifact_revision, source_span, source_hash, confidence, created_by, created_at, supersedes_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            identifier,
            proposal.project_id.root,
            proposal.run_id,
            proposal.plan_revision,
            proposal.target_layer.value,
            proposal.kind.value,
            record_status(proposal.target_layer),
            proposal.subject_id,
            proposal.predicate,
            proposal.value_json,
            request.chapter_plan.chapter_number,
            request.chapter_plan.chapter_number,
            proposal.source_artifact_id.root,
            proposal.source_artifact_revision,
            f"{evidence.source_byte_start}:{evidence.source_byte_end}",
            evidence.source_span_sha256,
            proposal.confidence,
            "chapter-coordinator:v1",
            context.committed_at.isoformat(),
            proposal.supersedes_memory_id,
        ),
    )
    context.connection.execute(
        "INSERT INTO memory_fts(memory_id, content) VALUES (?, ?)",
        (identifier, proposal.value_json),
    )
    context.connection.execute(
        "UPDATE memory_candidates SET status = ? WHERE candidate_id = ?",
        (CandidateStatus.PROMOTED.value, candidate_identifier),
    )
    return identifier


__all__ = ("commit_chapter_acceptance",)
