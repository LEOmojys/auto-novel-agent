from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from novel_agent.contracts import MemoryLayer
from novel_agent.memory_models import MemoryFactKind
from novel_agent.memory_store import record_status

if TYPE_CHECKING:
    from novel_agent.planning_commit import PlanningCommitRequest


@dataclass(frozen=True, slots=True)
class MemoryWrite:
    memory_id: str
    layer: MemoryLayer
    kind: MemoryFactKind
    subject_id: str
    predicate: str
    value_json: str
    source_artifact_id: str
    source_sha256: str
    source_size: int


def memory_writes(request: PlanningCommitRequest, plan_revision: int) -> tuple[MemoryWrite, ...]:
    world = request.world.value
    writes: list[MemoryWrite] = [
        MemoryWrite(
            _stable_id(f"{request.run_id}:{plan_revision}:world:{world.world_id}"),
            MemoryLayer.M1_CANONICAL,
            MemoryFactKind.WORLD_RULE,
            world.world_id,
            "world_bible",
            world.model_dump_json(),
            world.world_id,
            request.world.raw.content_sha256,
            len(request.world.raw.raw_bytes),
        )
    ]
    writes.extend(
        MemoryWrite(
            _stable_id(f"{request.run_id}:{plan_revision}:character:{character.character_id}"),
            MemoryLayer.M1_CANONICAL,
            MemoryFactKind.CHARACTER_IDENTITY,
            character.character_id,
            "character_identity",
            character.model_dump_json(),
            request.characters.value.character_set_id,
            request.characters.raw.content_sha256,
            len(request.characters.raw.raw_bytes),
        )
        for character in request.characters.value.characters
    )
    writes.extend(
        MemoryWrite(
            _stable_id(f"{request.run_id}:{plan_revision}:chapter-plan:{chapter.chapter_plan_id}"),
            MemoryLayer.M2_PLAN,
            MemoryFactKind.PLAN_GOAL,
            chapter.chapter_plan_id,
            "chapter_goal",
            chapter.model_dump_json(),
            request.outline.value.plan_id,
            request.outline.raw.content_sha256,
            len(request.outline.raw.raw_bytes),
        )
        for chapter in request.outline.value.chapters
    )
    return tuple(writes)


def insert_memory(
    connection: sqlite3.Connection,
    request: PlanningCommitRequest,
    memory: MemoryWrite,
    plan_revision: int,
    committed_at: datetime,
) -> None:
    connection.execute(
        "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, subject_id, "
        "predicate, value_json, source_artifact_id, source_artifact_revision, source_span, source_hash, confidence, "
        "created_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 1.0, ?, ?)",
        (
            memory.memory_id,
            request.project_id.root,
            request.run_id,
            plan_revision,
            memory.layer.value,
            memory.kind.value,
            record_status(memory.layer),
            memory.subject_id,
            memory.predicate,
            memory.value_json,
            memory.source_artifact_id,
            f"0:{memory.source_size}",
            memory.source_sha256,
            "planning-coordinator:v1",
            committed_at.isoformat(),
        ),
    )
    connection.execute(
        "INSERT INTO memory_fts(memory_id, content) VALUES (?, ?)",
        (memory.memory_id, memory.value_json),
    )


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


__all__ = ("MemoryWrite", "insert_memory", "memory_writes")
