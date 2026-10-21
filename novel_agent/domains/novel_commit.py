"""Partial planning commits for sub-task intents (world-only, cast-only).

When the brief targets a single artifact (e.g. "design three characters"),
the graph has no outline node and therefore no validated plan.  These commit
helpers persist the produced artifact and its M1 canonical memory in one
transaction and complete the run, mirroring the artifact/memory SQL of the
full planning commit without creating a plan or chapter tasks.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

from novel_agent.contracts import MemoryLayer
from novel_agent.contracts import MemoryLayer
from novel_agent.memory_models import MemoryFactKind
from novel_agent.artifacts import enqueue_materialization
from novel_agent.memory_store import index_memory, record_status
from novel_agent.run_event_store import append_run_event
from novel_agent.run_state import RunEvent
from novel_agent.sqlite_transactions import write_transaction

if TYPE_CHECKING:
    from novel_agent.adapters import CharacterAdaptation, OutlineAdaptation, WorldAdaptation


def commit_world_only(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    project_id: str,
    brief_id: str,
    world_result: WorldAdaptation,
    committed_at: datetime,
    complete: bool = True,
) -> None:
    world = world_result.value
    with write_transaction(connection):
        _insert_artifact(
            connection,
            run_id=run_id,
            project_id=project_id,
            artifact_id=world.world_id,
            kind="world_bible",
            raw_bytes=world_result.raw.raw_bytes,
            committed_at=committed_at,
        )
        memory_id = _stable_id(f"{run_id}:1:world:{world.world_id}")
        _insert_m1_memory(
            connection,
            run_id=run_id,
            project_id=project_id,
            plan_revision=0,
            memory_id=memory_id,
            kind=MemoryFactKind.WORLD_RULE,
            subject_id=world.world_id,
            predicate="world_bible",
            value_json=world.model_dump_json(),
            source_artifact_id=world.world_id,
            source_sha256=world_result.raw.content_sha256,
            source_size=len(world_result.raw.raw_bytes),
            committed_at=committed_at,
        )
        if complete:
            _complete_run(connection, run_id, committed_at)


def commit_cast_only(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    project_id: str,
    brief_id: str,
    cast_result: CharacterAdaptation,
    committed_at: datetime,
    complete: bool = True,
) -> None:
    cast = cast_result.value
    with write_transaction(connection):
        _insert_artifact(
            connection,
            run_id=run_id,
            project_id=project_id,
            artifact_id=cast.character_set_id,
            kind="character_set",
            raw_bytes=cast_result.raw.raw_bytes,
            committed_at=committed_at,
        )
        for index, character in enumerate(cast.characters, start=1):
            memory_id = _stable_id(f"{run_id}:1:character:{character.character_id}")
            _insert_m1_memory(
                connection,
                run_id=run_id,
                project_id=project_id,
                plan_revision=0,
                memory_id=memory_id,
                kind=MemoryFactKind.CHARACTER_IDENTITY,
                subject_id=character.character_id,
                predicate="character_identity",
                value_json=character.model_dump_json(),
                source_artifact_id=cast.character_set_id,
                source_sha256=cast_result.raw.content_sha256,
                source_size=len(cast_result.raw.raw_bytes),
                committed_at=committed_at,
            )
        if complete:
            _complete_run(connection, run_id, committed_at)


def commit_outline_only(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    project_id: str,
    brief_id: str,
    world_result: WorldAdaptation,
    cast_result: CharacterAdaptation,
    outline_result: OutlineAdaptation,
    committed_at: datetime,
    complete: bool = True,
) -> None:
    """Persist a complete plan package (world/cast/outline) without chapter tasks.

    Used when the plan has no chapter steps: the run finishes after the
    outline is committed, and no phantom ``chapter`` task rows are created.
    """
    from novel_agent.adapters import OutlineAdaptation

    world = world_result.value
    cast = cast_result.value
    plan = outline_result.value
    with write_transaction(connection):
        _insert_artifact(
            connection,
            run_id=run_id,
            project_id=project_id,
            artifact_id=world.world_id,
            kind="world_bible",
            raw_bytes=world_result.raw.raw_bytes,
            committed_at=committed_at,
        )
        _insert_artifact(
            connection,
            run_id=run_id,
            project_id=project_id,
            artifact_id=cast.character_set_id,
            kind="character_set",
            raw_bytes=cast_result.raw.raw_bytes,
            committed_at=committed_at,
        )
        _insert_artifact(
            connection,
            run_id=run_id,
            project_id=project_id,
            artifact_id=plan.plan_id,
            kind="plan_contract",
            raw_bytes=outline_result.raw.raw_bytes,
            committed_at=committed_at,
        )
        _insert_m1_memory(
            connection,
            run_id=run_id,
            project_id=project_id,
            plan_revision=1,
            memory_id=_stable_id(f"{run_id}:1:world:{world.world_id}"),
            kind=MemoryFactKind.WORLD_RULE,
            subject_id=world.world_id,
            predicate="world_bible",
            value_json=world.model_dump_json(),
            source_artifact_id=world.world_id,
            source_sha256=world_result.raw.content_sha256,
            source_size=len(world_result.raw.raw_bytes),
            committed_at=committed_at,
        )
        for character in cast.characters:
            _insert_m1_memory(
                connection,
                run_id=run_id,
                project_id=project_id,
                plan_revision=1,
                memory_id=_stable_id(f"{run_id}:1:character:{character.character_id}"),
                kind=MemoryFactKind.CHARACTER_IDENTITY,
                subject_id=character.character_id,
                predicate="character_identity",
                value_json=character.model_dump_json(),
                source_artifact_id=cast.character_set_id,
                source_sha256=cast_result.raw.content_sha256,
                source_size=len(cast_result.raw.raw_bytes),
                committed_at=committed_at,
            )
        for chapter in plan.chapters:
            _insert_m2_memory(
                connection,
                run_id=run_id,
                project_id=project_id,
                plan_revision=1,
                memory_id=_stable_id(f"{run_id}:1:chapter-plan:{chapter.chapter_plan_id}"),
                kind=MemoryFactKind.PLAN_GOAL,
                subject_id=chapter.chapter_plan_id,
                predicate="chapter_goal",
                value_json=chapter.model_dump_json(),
                source_artifact_id=plan.plan_id,
                source_sha256=outline_result.raw.content_sha256,
                source_size=len(outline_result.raw.raw_bytes),
                committed_at=committed_at,
            )
        if complete:
            _complete_run(connection, run_id, committed_at)


def _insert_m2_memory(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    project_id: str,
    plan_revision: int,
    memory_id: str,
    kind: MemoryFactKind,
    subject_id: str,
    predicate: str,
    value_json: str,
    source_artifact_id: str,
    source_sha256: str,
    source_size: int,
    committed_at: datetime,
) -> None:
    connection.execute(
        "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, subject_id, "
        "predicate, value_json, source_artifact_id, source_artifact_revision, source_span, source_hash, confidence, "
        "created_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 1.0, "
        "'partial-commit:v1', ?) ON CONFLICT(memory_id) DO NOTHING",
        (
            memory_id,
            project_id,
            run_id,
            plan_revision,
            MemoryLayer.M2_PLAN.value,
            kind.value,
            record_status(MemoryLayer.M2_PLAN),
            subject_id,
            predicate,
            value_json,
            source_artifact_id,
            f"0:{source_size}",
            source_sha256,
            committed_at.isoformat(),
        ),
    )
    index_memory(connection, memory_id, value_json)


def _insert_artifact(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    project_id: str,
    artifact_id: str,
    kind: str,
    raw_bytes: bytes,
    committed_at: datetime,
) -> None:
    connection.execute(
        "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
        "schema_version, is_current, created_at) VALUES (?, 1, ?, ?, ?, 'accepted', ?, ?, 1, 1, ?) "
        "ON CONFLICT(artifact_id, revision) DO NOTHING",
        (
            artifact_id,
            project_id,
            run_id,
            kind,
            raw_bytes,
            hashlib.sha256(raw_bytes).hexdigest(),
            committed_at.isoformat(),
        ),
    )
    enqueue_materialization(
        connection,
        project_id=project_id,
        artifact_id=artifact_id,
        revision=1,
        raw_bytes=raw_bytes,
        created_at=committed_at,
    )


def _insert_m1_memory(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    project_id: str,
    plan_revision: int,
    memory_id: str,
    kind: MemoryFactKind,
    subject_id: str,
    predicate: str,
    value_json: str,
    source_artifact_id: str,
    source_sha256: str,
    source_size: int,
    committed_at: datetime,
) -> None:
    connection.execute(
        "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, subject_id, "
        "predicate, value_json, source_artifact_id, source_artifact_revision, source_span, source_hash, confidence, "
        "created_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 1.0, "
        "'partial-commit:v1', ?) ON CONFLICT(memory_id) DO NOTHING",
        (
            memory_id,
            project_id,
            run_id,
            plan_revision,
            MemoryLayer.M1_CANONICAL.value,
            kind.value,
            record_status(MemoryLayer.M1_CANONICAL),
            subject_id,
            predicate,
            value_json,
            source_artifact_id,
            f"0:{source_size}",
            source_sha256,
            committed_at.isoformat(),
        ),
    )
    index_memory(connection, memory_id, value_json)


def _complete_run(connection: sqlite3.Connection, run_id: str, committed_at: datetime) -> None:
    connection.execute(
        "UPDATE runs SET state = 'succeeded', updated_at = ? WHERE run_id = ?",
        (committed_at.isoformat(), run_id),
    )
    append_run_event(connection, RunEvent(run_id, "run_succeeded", {}, committed_at))


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


__all__ = ("commit_cast_only", "commit_outline_only", "commit_world_only")
