from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import assert_never

from novel_agent.memory_models import (
    CandidateEvidence,
    CandidateStatus,
    EntityResolution,
    InferenceMode,
    MemoryCandidateRecord,
    MemoryFactKind,
    MemoryPromotionResult,
    MemoryProposal,
    PromotionAuthority,
    PromotionOutcome,
    RetrievalView,
    ReviewCode,
    TemporalCertainty,
    WorkingContext,
)
from novel_agent.memory_policy import decide_promotion
from novel_agent.memory_promotion import insert_promoted_memory
from novel_agent.memory_store import (
    MemoryErrorCode,
    MemoryGovernanceError,
    build_evidence,
    candidate_id,
    load_promoted_memory,
    load_source,
    memory_id,
    refresh_candidate_evidence,
)
from novel_agent.storage import write_transaction


@dataclass(frozen=True, slots=True)
class MemoryGovernance:
    connection: sqlite3.Connection

    def stage(self, proposal: MemoryProposal) -> MemoryCandidateRecord:
        evidence = build_evidence(proposal, load_source(self.connection, proposal))
        existing = self._candidate(evidence)
        if existing is not None:
            return refresh_candidate_evidence(self.connection, existing, evidence)
        candidate = MemoryCandidateRecord(
            candidate_id=candidate_id(evidence),
            evidence=evidence,
            status=CandidateStatus.QUARANTINED,
        )
        with write_transaction(self.connection):
            cursor = self.connection.execute(
                "INSERT INTO memory_candidates(candidate_id, run_id, source_artifact_id, "
                "source_artifact_revision, extraction_schema_version, normalized_fact_hash, status, "
                "candidate_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                (
                    candidate.candidate_id,
                    proposal.run_id,
                    proposal.source_artifact_id.root,
                    proposal.source_artifact_revision,
                    proposal.extraction_schema_version,
                    evidence.normalized_fact_hash,
                    candidate.status.value,
                    evidence.model_dump_json(),
                    proposal.created_at.isoformat(),
                ),
            )
        if cursor.rowcount == 1:
            return candidate
        winner = self._candidate(evidence)
        if winner is None:
            raise MemoryGovernanceError(MemoryErrorCode.PROMOTION_MISSING)
        return winner

    def ingest(self, proposal: MemoryProposal) -> MemoryPromotionResult:
        candidate = self.stage(proposal)
        if candidate.status is CandidateStatus.PROMOTED:
            return self._existing_promotion(candidate)
        decision = decide_promotion(candidate.evidence, self._has_conflict(candidate.evidence))
        candidate = self._set_status(candidate, decision.candidate_status, decision.review_code)
        match decision.outcome:
            case PromotionOutcome.PROMOTED:
                return self._promote(candidate)
            case PromotionOutcome.REVIEW_REQUIRED | PromotionOutcome.CONFLICTED:
                return MemoryPromotionResult(
                    candidate=candidate,
                    outcome=decision.outcome,
                    memory=None,
                    requires_pause=decision.requires_pause,
                )
            case unreachable:
                assert_never(unreachable)

    def _candidate(self, evidence: CandidateEvidence) -> MemoryCandidateRecord | None:
        proposal = evidence.proposal
        row = self.connection.execute(
            "SELECT candidate_id, status, candidate_json FROM memory_candidates WHERE "
            "source_artifact_id = ? AND source_artifact_revision = ? AND "
            "extraction_schema_version = ? AND normalized_fact_hash = ?",
            (
                proposal.source_artifact_id.root,
                proposal.source_artifact_revision,
                proposal.extraction_schema_version,
                evidence.normalized_fact_hash,
            ),
        ).fetchone()
        if row is None:
            return None
        return MemoryCandidateRecord(
            candidate_id=str(row[0]),
            evidence=CandidateEvidence.model_validate_json(str(row[2])),
            status=CandidateStatus(str(row[1])),
        )

    def _set_status(
        self,
        candidate: MemoryCandidateRecord,
        status: CandidateStatus,
        review_code: ReviewCode | None,
    ) -> MemoryCandidateRecord:
        if candidate.status is not status:
            with write_transaction(self.connection):
                self.connection.execute(
                    "UPDATE memory_candidates SET status = ? WHERE candidate_id = ?",
                    (status.value, candidate.candidate_id),
                )
        return candidate.model_copy(update={"status": status, "review_code": review_code})

    def _has_conflict(self, evidence: CandidateEvidence) -> bool:
        proposal = evidence.proposal
        if proposal.supersedes_memory_id is not None:
            self._validate_supersession(proposal)
        if proposal.subject_id is None:
            return False
        rows = self.connection.execute(
            "SELECT value_json FROM memory_records current WHERE project_id = ? AND layer = ? "
            "AND subject_id = ? AND predicate = ? AND NOT EXISTS (SELECT 1 FROM memory_records "
            "successor WHERE successor.supersedes_id = current.memory_id)",
            (proposal.project_id.root, proposal.target_layer.value, proposal.subject_id, proposal.predicate),
        ).fetchall()
        return any(str(row[0]) != proposal.value_json for row in rows)

    def _validate_supersession(self, proposal: MemoryProposal) -> None:
        prior_id = proposal.supersedes_memory_id
        if prior_id is None:
            return
        row = self.connection.execute(
            "SELECT project_id, layer, subject_id, predicate FROM memory_records WHERE memory_id = ?",
            (prior_id,),
        ).fetchone()
        if row is None or proposal.subject_id is None:
            raise MemoryGovernanceError(MemoryErrorCode.INVALID_SUPERSESSION)
        prior_identity = (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
        candidate_identity = (
            proposal.project_id.root,
            proposal.target_layer.value,
            proposal.subject_id,
            proposal.predicate,
        )
        if prior_identity != candidate_identity:
            raise MemoryGovernanceError(MemoryErrorCode.INVALID_SUPERSESSION)

    def _promote(self, candidate: MemoryCandidateRecord) -> MemoryPromotionResult:
        evidence = candidate.evidence
        proposal = evidence.proposal
        subject_id = proposal.subject_id
        if subject_id is None:
            raise MemoryGovernanceError(MemoryErrorCode.INVALID_PROMOTION)
        with write_transaction(self.connection):
            identifier = insert_promoted_memory(
                self.connection,
                candidate_identifier=candidate.candidate_id,
                proposal=proposal,
                evidence=evidence,
                chapter_number=None,
                created_at=proposal.created_at,
                created_by="memory-governance:v1",
            )
            self.connection.execute(
                "UPDATE memory_candidates SET status = ? WHERE candidate_id = ?",
                (CandidateStatus.PROMOTED.value, candidate.candidate_id),
            )
        promoted = load_promoted_memory(self.connection, identifier)
        if promoted is None:
            raise MemoryGovernanceError(MemoryErrorCode.PROMOTION_MISSING)
        final_candidate = candidate.model_copy(update={"status": CandidateStatus.PROMOTED, "review_code": None})
        return MemoryPromotionResult(
            candidate=final_candidate,
            outcome=PromotionOutcome.PROMOTED,
            memory=promoted,
            requires_pause=False,
        )

    def _existing_promotion(self, candidate: MemoryCandidateRecord) -> MemoryPromotionResult:
        promoted = load_promoted_memory(self.connection, memory_id(candidate.candidate_id))
        if promoted is None:
            raise MemoryGovernanceError(MemoryErrorCode.PROMOTION_MISSING)
        return MemoryPromotionResult(
            candidate=candidate,
            outcome=PromotionOutcome.PROMOTED,
            memory=promoted,
            requires_pause=False,
        )

__all__ = (
    "CandidateStatus",
    "EntityResolution",
    "InferenceMode",
    "MemoryFactKind",
    "MemoryGovernance",
    "MemoryGovernanceError",
    "MemoryProposal",
    "PromotionAuthority",
    "PromotionOutcome",
    "RetrievalView",
    "TemporalCertainty",
    "WorkingContext",
)
