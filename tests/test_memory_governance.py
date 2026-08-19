from __future__ import annotations

import hashlib
import sqlite3
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from novel_agent.memory import (
    CandidateStatus,
    EntityResolution,
    InferenceMode,
    MemoryFactKind,
    MemoryGovernance,
    MemoryGovernanceError,
    MemoryProposal,
    PromotionAuthority,
    PromotionOutcome,
    TemporalCertainty,
)
from novel_agent.pathing import ArtifactId, ProjectId
from novel_agent.storage import (
    ArtifactRevisionWrite,
    commit_artifact_revision,
    connect_project_database,
    migrate,
)


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")
RUN_ID = "run-001"


def _service(
    tmp_path: Path,
) -> tuple[MemoryGovernance, AbstractContextManager[sqlite3.Connection]]:
    connection_context = connect_project_database(tmp_path / "projects" / PROJECT_ID.root)
    connection = connection_context.__enter__()
    migrate(connection)
    connection.execute(
        "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (RUN_ID, PROJECT_ID.root, "running", FIXED_TIME.isoformat(), FIXED_TIME.isoformat()),
    )
    return MemoryGovernance(connection), connection_context


def _close(connection_context: AbstractContextManager[sqlite3.Connection]) -> None:
    connection_context.__exit__(None, None, None)


def _accept_artifact(
    service: MemoryGovernance,
    raw_bytes: bytes,
    artifact_id: ArtifactId = ArtifactId("chapter-001"),
    revision: int = 1,
    kind: str = "chapter",
    status: str = "accepted",
) -> None:
    commit_artifact_revision(
        service.connection,
        ArtifactRevisionWrite(
            artifact_id=artifact_id.root,
            revision=revision,
            project_id=PROJECT_ID.root,
            run_id=RUN_ID,
            kind=kind,
            status=status,
            raw_bytes=raw_bytes,
            schema_version=1,
            created_at=FIXED_TIME.isoformat(),
        ),
    )


def _proposal(
    raw_text: str,
    *,
    artifact_id: ArtifactId = ArtifactId("chapter-001"),
    revision: int = 1,
    target_layer: str = "m3_confirmed",
    kind: MemoryFactKind = MemoryFactKind.DIRECT_EVENT,
    subject_id: str | None = "character-001",
    entity_resolution: EntityResolution = EntityResolution.EXACT,
    inference_mode: InferenceMode = InferenceMode.DIRECT,
    temporal_certainty: TemporalCertainty = TemporalCertainty.EXACT,
    authority: PromotionAuthority = PromotionAuthority.ACCEPTED_CHAPTER,
    value_json: str = '{"event":"opens compass"}',
    supersedes_memory_id: str | None = None,
) -> MemoryProposal:
    return MemoryProposal(
        run_id=RUN_ID,
        project_id=PROJECT_ID,
        source_artifact_id=artifact_id,
        source_artifact_revision=revision,
        extraction_schema_version=1,
        target_layer=target_layer,
        kind=kind,
        subject_id=subject_id,
        predicate="state",
        value_json=value_json,
        source_start=0,
        source_end=len(raw_text),
        source_excerpt=raw_text,
        entity_resolution=entity_resolution,
        inference_mode=inference_mode,
        temporal_certainty=temporal_certainty,
        confidence=1.0,
        authority=authority,
        plan_revision=1,
        supersedes_memory_id=supersedes_memory_id,
        created_at=FIXED_TIME,
    )


def test_evidence_backed_direct_event_auto_promotes(tmp_path: Path) -> None:
    # Given: an accepted chapter with a direct event at an exact Unicode character span.
    service, connection_context = _service(tmp_path)
    raw_text = "林在黎明打开黄铜罗盘。"
    _accept_artifact(service, raw_text.encode())
    try:
        # When: the candidate is staged and ingested through the governed promotion gate.
        result = service.ingest(_proposal(raw_text))
        row = service.connection.execute(
            "SELECT source_span, source_hash, source_artifact_id, source_artifact_revision "
            "FROM memory_records"
        ).fetchone()
    finally:
        _close(connection_context)

    # Then: M3 retains byte-exact evidence and a committed confirmed record.
    assert result.outcome is PromotionOutcome.PROMOTED
    assert result.candidate.status is CandidateStatus.PROMOTED
    assert result.memory is not None
    assert row == (
        f"0:{len(raw_text.encode())}",
        hashlib.sha256(raw_text.encode()).hexdigest(),
        "chapter-001",
        1,
    )


def test_fuzzy_entity_requires_review_without_canonical_write(tmp_path: Path) -> None:
    # Given: an accepted chapter candidate with an ambiguous character resolution.
    service, connection_context = _service(tmp_path)
    raw_text = "有人在黎明打开黄铜罗盘。"
    _accept_artifact(service, raw_text.encode())
    try:
        # When: extraction proposes an M3 event without an exact entity identity.
        result = service.ingest(
            _proposal(
                raw_text,
                subject_id=None,
                entity_resolution=EntityResolution.AMBIGUOUS,
            )
        )
        record_count = service.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
    finally:
        _close(connection_context)

    # Then: the candidate stays quarantined for review and no canonical fact is created.
    assert result.outcome is PromotionOutcome.REVIEW_REQUIRED
    assert result.candidate.status is CandidateStatus.NEEDS_REVIEW
    assert result.requires_pause is False
    assert record_count == 0


def test_unsupported_inference_requires_review(tmp_path: Path) -> None:
    # Given: an accepted chapter candidate that infers a causal relationship.
    service, connection_context = _service(tmp_path)
    raw_text = "林打开黄铜罗盘。"
    _accept_artifact(service, raw_text.encode())
    try:
        # When: the extraction asks to promote that inferred fact.
        result = service.ingest(_proposal(raw_text, inference_mode=InferenceMode.CAUSAL))
    finally:
        _close(connection_context)

    # Then: causal inference is never auto-promoted to a story fact.
    assert result.outcome is PromotionOutcome.REVIEW_REQUIRED
    assert result.candidate.status is CandidateStatus.NEEDS_REVIEW


def test_conflict_requires_pause_and_preserves_existing_fact(tmp_path: Path) -> None:
    # Given: a canonical character identity already verified from the world bible.
    service, connection_context = _service(tmp_path)
    world_text = "林是一名制图师。"
    _accept_artifact(service, world_text.encode(), ArtifactId("world-001"), kind="world")
    initial = _proposal(
        world_text,
        artifact_id=ArtifactId("world-001"),
        target_layer="m1_canonical",
        kind=MemoryFactKind.CHARACTER_IDENTITY,
        authority=PromotionAuthority.VALIDATED_WORLD,
        value_json='{"identity":"cartographer"}',
    )
    try:
        service.ingest(initial)

        # When: another validated candidate contradicts that identity without a retcon.
        result = service.ingest(initial.model_copy(update={"value_json": '{"identity":"spy"}'}))
        rows = tuple(
            service.connection.execute("SELECT memory_id, value_json FROM memory_records ORDER BY memory_id")
        )
    finally:
        _close(connection_context)

    # Then: the conflict asks the coordinator to pause and preserves the prior record byte-for-byte.
    assert result.outcome is PromotionOutcome.CONFLICTED
    assert result.candidate.status is CandidateStatus.CONFLICTED
    assert result.requires_pause is True
    assert len(rows) == 1
    assert rows[0][1] == '{"identity":"cartographer"}'


def test_duplicate_candidate_reuses_existing_m3_record(tmp_path: Path) -> None:
    # Given: a direct event whose artifact, schema version, and normalized fact are stable.
    service, connection_context = _service(tmp_path)
    raw_text = "林在黎明打开黄铜罗盘。"
    _accept_artifact(service, raw_text.encode())
    proposal = _proposal(raw_text)
    try:
        # When: the same extraction is delivered twice after a retry.
        first = service.ingest(proposal)
        second = service.ingest(proposal)
        record_count = service.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
    finally:
        _close(connection_context)

    # Then: the idempotency key produces one candidate and one confirmed record.
    assert first.candidate.candidate_id == second.candidate.candidate_id
    assert first.memory is not None
    assert second.memory is not None
    assert first.memory.memory_id == second.memory.memory_id
    assert record_count == 1


def test_user_retcon_creates_superseding_version_without_overwrite(tmp_path: Path) -> None:
    # Given: an accepted world rule that is already canonical.
    service, connection_context = _service(tmp_path)
    first_text = "地图只会在黎明改变。"
    second_text = "地图只会在月食时改变。"
    _accept_artifact(service, first_text.encode(), ArtifactId("world-001"), kind="world")
    initial = _proposal(
        first_text,
        artifact_id=ArtifactId("world-001"),
        target_layer="m1_canonical",
        kind=MemoryFactKind.WORLD_RULE,
        authority=PromotionAuthority.VALIDATED_WORLD,
        value_json='{"rule":"dawn"}',
    )
    try:
        first = service.ingest(initial)
        _accept_artifact(
            service,
            second_text.encode(),
            ArtifactId("world-001"),
            revision=2,
            kind="world",
        )

        # When: an explicit user retcon replaces the rule with a new revision.
        second = service.ingest(
            _proposal(
                second_text,
                artifact_id=ArtifactId("world-001"),
                revision=2,
                target_layer="m1_canonical",
                kind=MemoryFactKind.WORLD_RULE,
                authority=PromotionAuthority.USER_RETCON,
                value_json='{"rule":"eclipse"}',
                supersedes_memory_id=first.memory.memory_id if first.memory is not None else None,
            )
        )
        rows = tuple(
            service.connection.execute(
                "SELECT memory_id, value_json, supersedes_id FROM memory_records ORDER BY created_at, memory_id"
            )
        )
    finally:
        _close(connection_context)

    # Then: the old fact remains immutable and the replacement points to it explicitly.
    assert second.outcome is PromotionOutcome.PROMOTED
    assert first.memory is not None
    assert second.memory is not None
    assert rows[0][1] == '{"rule":"dawn"}'
    assert rows[1] == (second.memory.memory_id, '{"rule":"eclipse"}', first.memory.memory_id)


def test_draft_source_cannot_enter_candidate_quarantine(tmp_path: Path) -> None:
    # Given: a draft artifact that has never been accepted or staged by the coordinator.
    service, connection_context = _service(tmp_path)
    raw_text = "林打开黄铜罗盘。"
    _accept_artifact(service, raw_text.encode(), status="draft")
    try:
        # When: an extractor attempts to stage an M0 candidate from that draft.
        with pytest.raises(MemoryGovernanceError, match="source_not_eligible"):
            service.stage(_proposal(raw_text))
    finally:
        _close(connection_context)

    # Then: no unaccepted model output becomes governed memory evidence.
