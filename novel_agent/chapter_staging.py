from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime

from novel_agent.chapter_errors import ChapterPersistenceError
from novel_agent.chapter_models import (
    ChapterCandidateStageRequest,
    ChapterQualityReportRequest,
    ChapterStageRequest,
    StagedChapter,
)
from novel_agent.memory_models import CandidateStatus, MemoryCandidateRecord, MemoryProposal
from novel_agent.memory_store import build_evidence, candidate_id, load_source
from novel_agent.pathing import ArtifactId
from novel_agent.run_event_store import append_run_event
from novel_agent.run_state import RunEvent


def stage_chapter(
    connection: sqlite3.Connection,
    request: ChapterStageRequest,
    staged_at: datetime,
) -> StagedChapter:
    _require_accepted_plan(connection, request)
    seed = f"{request.run_id}:{request.chapter_plan.chapter_plan_id}"
    artifact_id = ArtifactId(f"chapter-{hashlib.sha256(seed.encode()).hexdigest()[:24]}")
    row = connection.execute(
        "SELECT COALESCE(MAX(revision), 0) FROM artifacts WHERE artifact_id = ?",
        (artifact_id.root,),
    ).fetchone()
    if row is None:
        raise ChapterPersistenceError("chapter staging revision is unavailable")
    revision = int(row[0]) + 1
    raw = request.adaptation.raw
    connection.execute(
        "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
        "schema_version, is_current, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
        (
            artifact_id.root,
            revision,
            request.project_id.root,
            request.run_id,
            "chapter",
            "staged",
            raw.raw_bytes,
            raw.content_sha256,
            request.adaptation.value.schema_version,
            staged_at.isoformat(),
        ),
    )
    connection.execute(
        "INSERT INTO artifact_dependencies(artifact_id, artifact_revision, depends_on_artifact_id, "
        "depends_on_revision, dependency_kind) VALUES (?, ?, ?, 1, ?)",
        (artifact_id.root, revision, request.plan_id, "chapter_plan"),
    )
    append_run_event(
        connection,
        RunEvent(
            request.run_id,
            "chapter_staged",
            {"attempt": request.attempt, "chapter": request.chapter_plan.chapter_number, "task": request.task_id},
            staged_at,
        ),
    )
    return StagedChapter(
        artifact_id=artifact_id,
        revision=revision,
        task_id=request.task_id,
        chapter_plan_id=request.chapter_plan.chapter_plan_id,
        chapter_number=request.chapter_plan.chapter_number,
        raw_bytes=raw.raw_bytes,
        raw_sha256=raw.content_sha256,
        body=request.adaptation.value.body,
        adapter_version=raw.adapter_version,
        parse_route=request.adaptation.parse_route.value,
        generation_attempt=request.attempt,
    )


def stage_candidates(connection: sqlite3.Connection, request: ChapterCandidateStageRequest) -> tuple[str, ...]:
    identifiers: list[str] = []
    for proposal in request.proposals:
        _require_candidate_source(request, proposal)
        evidence = build_evidence(proposal, load_source(connection, proposal))
        row = connection.execute(
            "SELECT candidate_id FROM memory_candidates WHERE source_artifact_id = ? "
            "AND source_artifact_revision = ? AND extraction_schema_version = ? AND normalized_fact_hash = ?",
            (
                proposal.source_artifact_id.root,
                proposal.source_artifact_revision,
                proposal.extraction_schema_version,
                evidence.normalized_fact_hash,
            ),
        ).fetchone()
        identifier = candidate_id(evidence) if row is None else str(row[0])
        if row is None:
            candidate = MemoryCandidateRecord(
                candidate_id=identifier,
                evidence=evidence,
                status=CandidateStatus.QUARANTINED,
            )
            connection.execute(
                "INSERT INTO memory_candidates(candidate_id, run_id, source_artifact_id, source_artifact_revision, "
                "extraction_schema_version, normalized_fact_hash, status, candidate_json, chapter_number, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    identifier,
                    request.run_id,
                    proposal.source_artifact_id.root,
                    proposal.source_artifact_revision,
                    proposal.extraction_schema_version,
                    evidence.normalized_fact_hash,
                    CandidateStatus.QUARANTINED.value,
                    candidate.model_dump_json(),
                    request.staged.chapter_number,
                    proposal.created_at.isoformat(),
                ),
            )
        identifiers.append(identifier)
    return tuple(identifiers)


def record_quality(
    connection: sqlite3.Connection,
    request: ChapterQualityReportRequest,
    recorded_at: datetime,
) -> None:
    report_json = request.decision.model_dump_json()
    seed = f"{request.task_id}:{request.staged.artifact_id.root}:{request.staged.revision}:{report_json}"
    report_id = hashlib.sha256(seed.encode()).hexdigest()
    connection.execute(
        "INSERT INTO quality_reports(report_id, run_id, task_id, artifact_id, artifact_revision, decision, report_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(report_id) DO NOTHING",
        (
            report_id,
            request.run_id,
            request.task_id,
            request.staged.artifact_id.root,
            request.staged.revision,
            request.decision.outcome.value,
            report_json,
            recorded_at.isoformat(),
        ),
    )
    append_run_event(
        connection,
        RunEvent(
            request.run_id,
            "chapter_quality_recorded",
            {"decision": request.decision.outcome.value, "task": request.task_id},
            recorded_at,
        ),
    )


def _require_accepted_plan(connection: sqlite3.Connection, request: ChapterStageRequest) -> None:
    row = connection.execute(
        "SELECT project_id, run_id, status FROM artifacts "
        "WHERE artifact_id = ? AND status = 'accepted' AND is_current = 1",
        (request.plan_id,),
    ).fetchone()
    expected = (request.project_id.root, request.run_id, "accepted")
    if row is None or (str(row[0]), str(row[1]), str(row[2])) != expected:
        raise ChapterPersistenceError("chapter plan is not an accepted artifact of this run")


def _require_candidate_source(request: ChapterCandidateStageRequest, proposal: MemoryProposal) -> None:
    actual = (
        proposal.run_id,
        proposal.project_id,
        proposal.source_artifact_id,
        proposal.source_artifact_revision,
    )
    expected = (request.run_id, request.project_id, request.staged.artifact_id, request.staged.revision)
    if actual != expected:
        raise ChapterPersistenceError("memory proposal does not belong to the staged chapter")


__all__ = ("record_quality", "stage_candidates", "stage_chapter")
