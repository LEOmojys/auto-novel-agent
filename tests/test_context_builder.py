from __future__ import annotations

import hashlib
import sqlite3
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path

from novel_agent.context import (
    ContextBudget,
    ContextBuilder,
    ContextRequest,
)
from novel_agent.pathing import ProjectId
from novel_agent.storage import connect_project_database, migrate


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


def _database(tmp_path: Path) -> tuple[sqlite3.Connection, AbstractContextManager[sqlite3.Connection]]:
    context = connect_project_database(tmp_path / "projects" / "novel-one")
    connection = context.__enter__()
    migrate(connection)
    connection.execute(
        "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("run-001", "novel-one", "running", FIXED_TIME.isoformat(), FIXED_TIME.isoformat()),
    )
    raw = b"Lin enters the harbor at dawn."
    digest = hashlib.sha256(raw).hexdigest()
    connection.execute(
        "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, schema_version, is_current, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("chapter-001", 1, "novel-one", "run-001", "chapter", "accepted", raw, digest, 1, 1, FIXED_TIME.isoformat()),
    )
    return connection, context


def _record(
    connection: sqlite3.Connection,
    memory_id: str,
    layer: str,
    status: str,
    kind: str,
    subject_id: str,
    value: str,
    chapter_start: int | None,
    plan_revision: int = 1,
    supersedes_id: str | None = None,
) -> None:
    connection.execute(
        "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, subject_id, predicate, value_json, "
        "chapter_start, chapter_end, source_artifact_id, source_artifact_revision, source_span, source_hash, confidence, created_by, created_at, supersedes_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            memory_id,
            "novel-one",
            "run-001",
            plan_revision,
            layer,
            kind,
            status,
            subject_id,
            "fact",
            value,
            chapter_start,
            chapter_start,
            "chapter-001",
            1,
            "0:3",
            hashlib.sha256(b"Lin").hexdigest(),
            1.0,
            "fixture",
            FIXED_TIME.isoformat(),
            supersedes_id,
        ),
    )


def test_rebuild_is_deterministic(tmp_path: Path) -> None:
    # Given: eligible layered facts, a superseded revision, and quarantined/rejected records.
    connection, context = _database(tmp_path)
    _record(connection, "m1-world", "m1_canonical", "canonical", "world_rule", "world-1", '{"text":"harbor"}', None)
    _record(connection, "m2-goal", "m2_plan", "validated", "plan_goal", "chapter-1", '{"text":"find harbor"}', 1)
    _record(connection, "m3-recent", "m3_confirmed", "confirmed", "direct_event", "char-1", '{"text":"harbor"}', 1)
    _record(connection, "m3-long", "m3_confirmed", "confirmed", "state_change", "char-2", '{"text":"compass"}', 0)
    _record(connection, "m3-old", "m3_confirmed", "confirmed", "state_change", "char-3", '{"text":"old"}', 0)
    _record(connection, "m3-new", "m3_confirmed", "confirmed", "state_change", "char-3", '{"text":"new"}', 0, supersedes_id="m3-old")
    _record(connection, "m0-only", "m0_candidate", "proposed", "direct_event", "char-4", '{"text":"hidden"}', 1)
    _record(connection, "rejected", "m3_confirmed", "rejected", "direct_event", "char-5", '{"text":"rejected"}', 1)
    request = ContextRequest(
        project_id=ProjectId("novel-one"),
        run_id="run-001",
        plan_revision=1,
        chapter_plan_id="chapter-plan-1",
        chapter_number=1,
        direct_memory_ids=("m3-recent",),
        entity_ids=("char-1",),
        query_terms=("harbor",),
        style_guidance=("克制、具体。",),
    )
    try:
        # When: the same context is rebuilt twice from the authoritative store.
        first = ContextBuilder(connection).build(request)
        second = ContextBuilder(connection).build(request)
    finally:
        context.__exit__(None, None, None)

    # Then: ordered IDs, prompt digest, provenance, and exclusions are identical and safe.
    assert first.memory_ids == second.memory_ids
    assert first.digest == second.digest
    assert first.prompt == second.prompt
    assert first.memory_ids[0] == "m3-recent"
    assert {"m0-only", "rejected", "m3-old"}.isdisjoint(first.memory_ids)
    assert all(item.source_hash for item in first.items)


def test_direct_references_outrank_fts_candidates(tmp_path: Path) -> None:
    # Given: two confirmed facts match the same search term, but only one is directly referenced.
    connection, context = _database(tmp_path)
    _record(connection, "m-direct", "m3_confirmed", "confirmed", "direct_event", "char-direct", '{"text":"harbor"}', 1)
    _record(connection, "m-search", "m3_confirmed", "confirmed", "direct_event", "char-other", '{"text":"harbor"}', 1)
    request = ContextRequest(
        project_id=ProjectId("novel-one"),
        run_id="run-001",
        plan_revision=1,
        chapter_plan_id="chapter-plan-1",
        chapter_number=1,
        entity_ids=("char-direct",),
        query_terms=("harbor",),
    )
    try:
        # When: deterministic retrieval ranks direct references and lexical candidates.
        result = ContextBuilder(connection).build(request)
    finally:
        context.__exit__(None, None, None)

    # Then: the direct reference is first despite equal lexical relevance.
    assert result.memory_ids[:2] == ("m-direct", "m-search")
    assert result.items[0].tier < result.items[1].tier


def test_invalid_source_span_or_hash_is_excluded(tmp_path: Path) -> None:
    # Given: one supported memory and records with forged or malformed source evidence.
    connection, context = _database(tmp_path)
    for memory_id in ("valid", "forged", "negative", "empty", "outside", "malformed"):
        _record(
            connection,
            memory_id,
            "m3_confirmed",
            "confirmed",
            "direct_event",
            f"char-{memory_id}",
            f'{{"text":"{memory_id}"}}',
            1,
        )
    connection.execute(
        "UPDATE memory_records SET source_hash = ? WHERE memory_id = ?",
        ("f" * 64, "forged"),
    )
    for memory_id, source_span in (
        ("negative", "-1:3"),
        ("empty", "3:3"),
        ("outside", "0:300"),
        ("malformed", "first:third"),
    ):
        connection.execute(
            "UPDATE memory_records SET source_span = ? WHERE memory_id = ?",
            (source_span, memory_id),
        )
    request = ContextRequest(
        project_id=ProjectId("novel-one"),
        run_id="run-001",
        plan_revision=1,
        chapter_plan_id="chapter-plan-1",
        chapter_number=1,
    )
    try:
        # When: authoritative artifact bytes are checked before context assembly.
        result = ContextBuilder(connection).build(request)
    finally:
        context.__exit__(None, None, None)

    # Then: only the record backed by the exact cited bytes is eligible.
    exclusion_reasons = {item.memory_id: item.reason for item in result.exclusions}
    assert "valid" in result.memory_ids
    assert exclusion_reasons == {
        "empty": "source_span_invalid",
        "forged": "source_span_hash_mismatch",
        "malformed": "source_span_invalid",
        "negative": "source_span_invalid",
        "outside": "source_span_invalid",
    }


def test_budget_and_disallowed_layers_are_explicitly_reported(tmp_path: Path) -> None:
    # Given: a small budget and facts that exceed it or are not eligible for a prompt.
    connection, context = _database(tmp_path)
    _record(connection, "m1-long", "m1_canonical", "canonical", "world_rule", "world-1", '{"text":"a" * 30}', None)
    _record(connection, "m0-hidden", "m0_candidate", "proposed", "direct_event", "char-1", '{"text":"secret"}', 1)
    request = ContextRequest(
        project_id=ProjectId("novel-one"),
        run_id="run-001",
        plan_revision=1,
        chapter_plan_id="chapter-plan-1",
        chapter_number=1,
        budget=ContextBudget(total_tokens=12, canonical_tokens=6, plan_tokens=2, continuity_tokens=2, long_term_tokens=2, style_tokens=2),
    )
    try:
        # When: the builder assembles the bounded working context.
        result = ContextBuilder(connection).build(request)
    finally:
        context.__exit__(None, None, None)

    # Then: selected content stays within budget and every omission has a deterministic reason.
    assert sum(item.token_count for item in result.items) <= 12
    assert "m0-hidden" not in result.memory_ids
    assert result.exclusions
    assert all(item.reason for item in result.exclusions)
