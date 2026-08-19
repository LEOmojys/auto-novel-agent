from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from novel_agent.adapters import CharacterAdaptation, OutlineAdaptation, WorldAdaptation
from novel_agent.contracts import StoryBrief
from novel_agent.pathing import ProjectId
from novel_agent.planning_audit import PlanningAuditRequest, insert_planning_audit_manifest
from novel_agent.planning_memory_commit import insert_memory, memory_writes
from novel_agent.run_event_store import append_run_event
from novel_agent.run_state import RunEvent
from novel_agent.skill_manifests.models import ActiveSkillPin


@dataclass(frozen=True, slots=True)
class PlanningCommitRequest:
    run_id: str
    project_id: ProjectId
    brief: StoryBrief
    world: WorldAdaptation
    characters: CharacterAdaptation
    outline: OutlineAdaptation
    skills: tuple[ActiveSkillPin, ...]


@dataclass(frozen=True, slots=True)
class PlanningCommitReceipt:
    plan_id: str
    plan_revision: int
    first_task_id: str


@dataclass(frozen=True, slots=True)
class _ArtifactWrite:
    artifact_id: str
    kind: str
    raw_bytes: bytes
    sha256: str
    schema_version: int


def commit_validated_plan(
    connection: sqlite3.Connection,
    request: PlanningCommitRequest,
    committed_at: datetime,
) -> PlanningCommitReceipt:
    plan = request.outline.value
    prior_revision = int(
        connection.execute("SELECT plan_revision FROM runs WHERE run_id = ?", (request.run_id,)).fetchone()[0]
    )
    existing = _existing_receipt(connection, request, prior_revision)
    if existing is not None:
        return existing
    plan_revision = prior_revision + 1
    first_task_id = f"{request.run_id}:chapter:{plan.chapters[0].chapter_plan_id}"
    artifacts = _artifact_writes(request)
    for artifact in artifacts:
        _insert_artifact(connection, request, artifact, committed_at)
    _insert_dependencies(connection, request)
    for memory in memory_writes(request, plan_revision):
        insert_memory(connection, request, memory, plan_revision, committed_at)
    connection.execute(
        "INSERT INTO tasks(task_id, run_id, task_type, state, idempotency_key, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (first_task_id, request.run_id, "chapter", "pending", first_task_id, committed_at.isoformat(), committed_at.isoformat()),
    )
    connection.execute(
        "UPDATE runs SET plan_revision = ?, updated_at = ? WHERE run_id = ?",
        (plan_revision, committed_at.isoformat(), request.run_id),
    )
    insert_planning_audit_manifest(
        connection,
        PlanningAuditRequest(
            run_id=request.run_id,
            brief_id=request.brief.brief_id,
            plan_id=plan.plan_id,
            plan_revision=plan_revision,
            artifacts=tuple((artifact.artifact_id, artifact.sha256) for artifact in artifacts),
            skills=request.skills,
            created_at=committed_at,
            parameters=(
                ("characters_parse_route", request.characters.parse_route.value),
                ("outline_parse_route", request.outline.parse_route.value),
                ("world_parse_route", request.world.parse_route.value),
            ),
            prompt_sha256=hashlib.sha256(request.brief.user_requirements.encode()).hexdigest(),
            response_sha256=request.outline.raw.content_sha256,
            adapter_version=", ".join(
                sorted(
                    {
                        request.characters.raw.adapter_version,
                        request.outline.raw.adapter_version,
                        request.world.raw.adapter_version,
                    }
                )
            ),
            artifact_revision=plan_revision,
        ),
    )
    append_run_event(
        connection,
        RunEvent(
            request.run_id,
            "plan_validated",
            {"first_task_id": first_task_id, "plan_id": plan.plan_id, "plan_revision": plan_revision},
            committed_at,
        ),
    )
    return PlanningCommitReceipt(plan.plan_id, plan_revision, first_task_id)


def _existing_receipt(
    connection: sqlite3.Connection,
    request: PlanningCommitRequest,
    prior_revision: int,
) -> PlanningCommitReceipt | None:
    if prior_revision == 0:
        return None
    row = connection.execute(
        "SELECT t.task_id FROM artifacts a JOIN tasks t ON t.run_id = a.run_id "
        "WHERE a.run_id = ? AND a.artifact_id = ? AND a.revision = 1 AND a.sha256 = ? "
        "AND t.task_type = ? ORDER BY t.created_at, t.task_id LIMIT 1",
        (request.run_id, request.outline.value.plan_id, request.outline.raw.content_sha256, "chapter"),
    ).fetchone()
    if row is None:
        return None
    return PlanningCommitReceipt(request.outline.value.plan_id, prior_revision, str(row[0]))


def _artifact_writes(request: PlanningCommitRequest) -> tuple[_ArtifactWrite, ...]:
    return (
        _ArtifactWrite(
            request.world.value.world_id,
            "world_bible",
            request.world.raw.raw_bytes,
            request.world.raw.content_sha256,
            request.world.value.schema_version,
        ),
        _ArtifactWrite(
            request.characters.value.character_set_id,
            "character_set",
            request.characters.raw.raw_bytes,
            request.characters.raw.content_sha256,
            request.characters.value.schema_version,
        ),
        _ArtifactWrite(
            request.outline.value.plan_id,
            "plan_contract",
            request.outline.raw.raw_bytes,
            request.outline.raw.content_sha256,
            request.outline.value.schema_version,
        ),
    )


def _insert_artifact(
    connection: sqlite3.Connection,
    request: PlanningCommitRequest,
    artifact: _ArtifactWrite,
    committed_at: datetime,
) -> None:
    connection.execute(
        "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
        "schema_version, is_current, created_at) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
        (
            artifact.artifact_id,
            request.project_id.root,
            request.run_id,
            artifact.kind,
            "accepted",
            artifact.raw_bytes,
            artifact.sha256,
            artifact.schema_version,
            committed_at.isoformat(),
        ),
    )


def _insert_dependencies(connection: sqlite3.Connection, request: PlanningCommitRequest) -> None:
    plan_id = request.outline.value.plan_id
    dependencies = (
        (request.world.value.world_id, "world_bible"),
        (request.characters.value.character_set_id, "character_set"),
    )
    for dependency_id, dependency_kind in dependencies:
        connection.execute(
            "INSERT INTO artifact_dependencies(artifact_id, artifact_revision, depends_on_artifact_id, "
            "depends_on_revision, dependency_kind) VALUES (?, 1, ?, 1, ?)",
            (plan_id, dependency_id, dependency_kind),
        )


__all__ = ("PlanningCommitReceipt", "PlanningCommitRequest", "commit_validated_plan")
