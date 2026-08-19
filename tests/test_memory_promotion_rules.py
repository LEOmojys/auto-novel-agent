from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from novel_agent.contracts import MemoryLayer
from novel_agent.memory_models import (
    CandidateEvidence,
    CandidateStatus,
    EntityResolution,
    InferenceMode,
    MemoryFactKind,
    MemoryProposal,
    PromotionAuthority,
    PromotionOutcome,
    ReviewCode,
    SourceArtifactStatus,
    TemporalCertainty,
)
from novel_agent.memory import MemoryGovernance
from novel_agent.memory_policy import decide_promotion
from novel_agent.pathing import ArtifactId, ProjectId
from novel_agent.storage import (
    ArtifactRevisionWrite,
    commit_artifact_revision,
    connect_project_database,
    migrate,
)


def _evidence(
    target_layer: MemoryLayer,
    kind: MemoryFactKind,
    authority: PromotionAuthority,
    source_status: SourceArtifactStatus,
) -> CandidateEvidence:
    proposal = MemoryProposal(
        run_id="run-001",
        project_id=ProjectId("novel-one"),
        source_artifact_id=ArtifactId("source-001"),
        source_artifact_revision=1,
        extraction_schema_version=1,
        target_layer=target_layer,
        kind=kind,
        subject_id="subject-001",
        predicate="fact",
        value_json='{"value":"exact"}',
        source_start=0,
        source_end=5,
        source_excerpt="exact",
        entity_resolution=EntityResolution.EXACT,
        inference_mode=InferenceMode.DIRECT,
        temporal_certainty=TemporalCertainty.EXACT,
        confidence=1.0,
        authority=authority,
        plan_revision=1,
        supersedes_memory_id=None,
        created_at=datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    return CandidateEvidence(
        proposal=proposal,
        source_artifact_sha256="a" * 64,
        source_byte_start=0,
        source_byte_end=5,
        source_span_sha256="b" * 64,
        source_status=source_status,
        normalized_fact_hash="c" * 64,
    )


def test_validated_plan_goal_auto_promotes_to_m2() -> None:
    # Given: an accepted, exact goal from a validated plan revision.
    evidence = _evidence(
        MemoryLayer.M2_PLAN,
        MemoryFactKind.PLAN_GOAL,
        PromotionAuthority.VALIDATED_PLAN,
        SourceArtifactStatus.ACCEPTED,
    )

    # When: the deterministic promotion policy evaluates it.
    decision = decide_promotion(evidence, has_conflict=False)

    # Then: it is eligible for automatic M2 promotion.
    assert decision.outcome is PromotionOutcome.PROMOTED
    assert decision.candidate_status is CandidateStatus.PROMOTED


def test_staged_artifact_remains_in_review_until_accepted() -> None:
    # Given: a direct chapter event extracted from a staged revision.
    evidence = _evidence(
        MemoryLayer.M3_CONFIRMED,
        MemoryFactKind.DIRECT_EVENT,
        PromotionAuthority.ACCEPTED_CHAPTER,
        SourceArtifactStatus.STAGED,
    )

    # When: the policy evaluates whether it may become a confirmed fact.
    decision = decide_promotion(evidence, has_conflict=False)

    # Then: M0 persists the evidence, but automatic M3 promotion remains blocked.
    assert decision.outcome is PromotionOutcome.REVIEW_REQUIRED
    assert decision.review_code is ReviewCode.SOURCE_NOT_ACCEPTED


def test_staged_candidate_promotes_after_source_acceptance(tmp_path: Path) -> None:
    # Given: an M0 candidate extracted once from a staged artifact revision.
    evidence = _evidence(
        MemoryLayer.M3_CONFIRMED,
        MemoryFactKind.DIRECT_EVENT,
        PromotionAuthority.ACCEPTED_CHAPTER,
        SourceArtifactStatus.STAGED,
    )
    proposal = evidence.proposal
    connection_context = connect_project_database(tmp_path / "projects" / proposal.project_id.root)
    connection = connection_context.__enter__()
    migrate(connection)
    connection.execute(
        "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (proposal.run_id, proposal.project_id.root, "running", proposal.created_at.isoformat(), proposal.created_at.isoformat()),
    )
    commit_artifact_revision(
        connection,
        ArtifactRevisionWrite(
            artifact_id=proposal.source_artifact_id.root,
            revision=proposal.source_artifact_revision,
            project_id=proposal.project_id.root,
            run_id=proposal.run_id,
            kind="chapter",
            status="staged",
            raw_bytes=proposal.source_excerpt.encode(),
            schema_version=1,
            created_at=proposal.created_at.isoformat(),
        ),
    )
    service = MemoryGovernance(connection)
    try:
        first = service.ingest(proposal)
        connection.execute("UPDATE artifacts SET status = ?", ("accepted",))

        # When: the coordinator accepts that exact artifact without re-extracting it.
        second = service.ingest(proposal)
        record_count = connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
    finally:
        connection_context.__exit__(None, None, None)

    # Then: the cached candidate refreshes its lifecycle state and promotes exactly once.
    assert first.outcome is PromotionOutcome.REVIEW_REQUIRED
    assert second.outcome is PromotionOutcome.PROMOTED
    assert second.candidate.status is CandidateStatus.PROMOTED
    assert record_count == 1
