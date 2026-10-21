"""Human review loop for memory candidates that automatic policy refused.

The promotion policy intentionally stops at `needs_review` / `conflicted` for
inferred facts, ambiguous entities, low confidence and value conflicts.  Until
now those candidates had no exit: they sat in `memory_candidates` forever.

This service is the exit:

* `list` shows reviewable candidates for a project;
* `resolve(..., decision="accept")` promotes the candidate, superseding the
  current value when it conflicts (authority `user_retcon`) or asserting a new
  fact the policy could not auto-promote (authority `user_review`);
* `resolve(..., decision="reject")` marks it rejected.

Every decision is written to `memory_reviews` so the audit trail explains who
changed story memory and why.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from novel_agent.memory_models import (
    CandidateStatus,
    MemoryCandidateRecord,
    PromotionAuthority,
)
from novel_agent.memory_policy import memory_kind_allowed
from novel_agent.memory_promotion import current_memory_id, insert_promoted_memory
from novel_agent.memory_store import build_evidence, load_source
from novel_agent.sqlite_transactions import write_transaction

_REVIEWABLE = (
    CandidateStatus.QUARANTINED,
    CandidateStatus.NEEDS_REVIEW,
    CandidateStatus.CONFLICTED,
)


class MemoryReviewError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class MemoryCandidateView:
    candidate_id: str
    run_id: str
    status: str
    layer: str
    kind: str
    subject_id: str
    predicate: str
    value_json: str
    confidence: float
    chapter_number: int | None
    source_artifact_id: str
    source_artifact_revision: int
    review_code: str | None

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "status": self.status,
            "layer": self.layer,
            "kind": self.kind,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "value_json": self.value_json,
            "confidence": self.confidence,
            "chapter_number": self.chapter_number,
            "source_artifact_id": self.source_artifact_id,
            "source_artifact_revision": self.source_artifact_revision,
            "review_code": self.review_code,
        }


@dataclass(frozen=True, slots=True)
class MemoryResolution:
    candidate_id: str
    decision: str
    memory_id: str | None
    status: str

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "decision": self.decision,
            "memory_id": self.memory_id,
            "status": self.status,
        }


class MemoryReviewService:
    def __init__(self, connection: sqlite3.Connection, now=None) -> None:
        self._connection = connection
        self._now = now or (lambda: datetime.now(UTC))

    # -- listing -----------------------------------------------------------

    def list(
        self,
        project_id: str,
        *,
        statuses: tuple[str, ...] | None = None,
        run_id: str | None = None,
        limit: int = 200,
    ) -> tuple[MemoryCandidateView, ...]:
        wanted = tuple(status.value for status in _REVIEWABLE) if statuses is None else tuple(statuses)
        if not wanted:
            return ()
        placeholders = ", ".join("?" for _ in wanted)
        query = (
            "SELECT c.candidate_id, c.run_id, c.status, c.candidate_json, c.chapter_number "
            "FROM memory_candidates c JOIN runs r ON r.run_id = c.run_id "
            f"WHERE r.project_id = ? AND c.status IN ({placeholders})"
        )
        parameters: list[object] = [project_id, *wanted]
        if run_id is not None:
            query += " AND c.run_id = ?"
            parameters.append(run_id)
        query += " ORDER BY c.created_at, c.candidate_id LIMIT ?"
        parameters.append(int(limit))
        rows = self._connection.execute(query, tuple(parameters)).fetchall()
        views: list[MemoryCandidateView] = []
        for row in rows:
            try:
                candidate = MemoryCandidateRecord.model_validate_json(str(row[3]))
            except Exception:  # noqa: BROAD_EXCEPT_OK - a corrupt row must not hide the rest
                continue
            proposal = candidate.evidence.proposal
            views.append(
                MemoryCandidateView(
                    candidate_id=str(row[0]),
                    run_id=str(row[1]),
                    status=str(row[2]),
                    layer=proposal.target_layer.value,
                    kind=proposal.kind.value,
                    subject_id=proposal.subject_id or "",
                    predicate=proposal.predicate,
                    value_json=proposal.value_json,
                    confidence=proposal.confidence,
                    chapter_number=None if row[4] is None else int(row[4]),
                    source_artifact_id=proposal.source_artifact_id.root,
                    source_artifact_revision=proposal.source_artifact_revision,
                    review_code=(
                        candidate.review_code.value if candidate.review_code is not None else None
                    ),
                )
            )
        return tuple(views)

    # -- resolution --------------------------------------------------------

    def resolve(
        self,
        candidate_id: str,
        *,
        project_id: str,
        decision: str,
        actor: str = "console",
        reason: str = "",
    ) -> MemoryResolution:
        if decision not in {"accept", "reject"}:
            raise MemoryReviewError("invalid_decision")
        row = self._connection.execute(
            "SELECT c.run_id, c.status, c.candidate_json, c.chapter_number, r.project_id "
            "FROM memory_candidates c JOIN runs r ON r.run_id = c.run_id WHERE c.candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise MemoryReviewError("candidate_not_found")
        if str(row[4]) != project_id:
            raise MemoryReviewError("candidate_not_found")
        current_status = CandidateStatus(str(row[1]))
        if current_status not in _REVIEWABLE:
            raise MemoryReviewError("candidate_not_reviewable")
        candidate = MemoryCandidateRecord.model_validate_json(str(row[2]))
        proposal = candidate.evidence.proposal
        now = self._now()

        if decision == "reject":
            with write_transaction(self._connection):
                self._set_status(candidate_id, CandidateStatus.REJECTED)
                self._record_review(candidate_id, project_id, str(row[0]), "rejected", actor, reason, None, now)
            return MemoryResolution(candidate_id, "rejected", None, CandidateStatus.REJECTED.value)

        if proposal.subject_id is None:
            raise MemoryReviewError("candidate_without_subject")
        if not memory_kind_allowed(proposal.target_layer, proposal.kind):
            raise MemoryReviewError("kind_not_allowed_in_layer")
        evidence = build_evidence(proposal, load_source(self._connection, proposal))
        existing_id = current_memory_id(self._connection, proposal)
        existing_value = None
        if existing_id is not None:
            found = self._connection.execute(
                "SELECT value_json FROM memory_records WHERE memory_id = ?", (existing_id,)
            ).fetchone()
            existing_value = None if found is None else str(found[0])
        if existing_id is not None and existing_value == proposal.value_json:
            # Same fact is already canon: promote the candidate without adding
            # a duplicate row.
            with write_transaction(self._connection):
                self._set_status(candidate_id, CandidateStatus.PROMOTED)
                self._record_review(
                    candidate_id, project_id, str(row[0]), "accepted", actor, reason, existing_id, now
                )
            return MemoryResolution(candidate_id, "accepted", existing_id, CandidateStatus.PROMOTED.value)

        authority = PromotionAuthority.USER_RETCON if existing_id else PromotionAuthority.USER_REVIEW
        reviewed = proposal.model_copy(
            update={"authority": authority, "supersedes_memory_id": existing_id}
        )
        reviewed_evidence = build_evidence(reviewed, load_source(self._connection, reviewed))
        chapter_number = None if row[3] is None else int(row[3])
        with write_transaction(self._connection):
            identifier = insert_promoted_memory(
                self._connection,
                candidate_identifier=candidate_id,
                proposal=reviewed,
                evidence=reviewed_evidence,
                chapter_number=chapter_number,
                created_at=now,
                created_by=f"memory-review:{actor}",
            )
            self._set_status(candidate_id, CandidateStatus.PROMOTED)
            self._record_review(
                candidate_id, project_id, str(row[0]), "accepted", actor, reason, identifier, now
            )
        return MemoryResolution(candidate_id, "accepted", identifier, CandidateStatus.PROMOTED.value)

    # -- helpers -----------------------------------------------------------

    def _set_status(self, candidate_id: str, status: CandidateStatus) -> None:
        self._connection.execute(
            "UPDATE memory_candidates SET status = ? WHERE candidate_id = ?",
            (status.value, candidate_id),
        )

    def _record_review(
        self,
        candidate_id: str,
        project_id: str,
        run_id: str,
        decision: str,
        actor: str,
        reason: str,
        memory_identifier: str | None,
        decided_at: datetime,
    ) -> None:
        review_id = hashlib.sha256(
            f"{candidate_id}:{decision}:{decided_at.isoformat()}".encode()
        ).hexdigest()[:32]
        self._connection.execute(
            "INSERT INTO memory_reviews(review_id, candidate_id, project_id, run_id, decision, "
            "actor, reason, memory_id, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(review_id) DO NOTHING",
            (
                review_id,
                candidate_id,
                project_id,
                run_id,
                decision,
                actor,
                reason,
                memory_identifier,
                decided_at.isoformat(),
            ),
        )


__all__ = (
    "MemoryCandidateView",
    "MemoryResolution",
    "MemoryReviewError",
    "MemoryReviewService",
)
