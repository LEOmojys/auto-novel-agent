"""Single promotion path shared by every writer of long-term memory.

Before this module the promotion SQL existed three times (chapter acceptance,
console approval, and `MemoryGovernance.ingest`), which let the status
vocabulary and evidence checks drift apart.  Everything that turns a stored M0
candidate into a `memory_records` row now goes through
`insert_promoted_memory` / `promote_candidates`.

Callers own the transaction; `promote_candidates` joins an active one and opens
its own otherwise (the console approval path runs in autocommit mode).
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from novel_agent.memory_models import (
    CandidateEvidence,
    CandidateStatus,
    MemoryCandidateRecord,
    MemoryProposal,
    PromotionOutcome,
)
from novel_agent.memory_policy import decide_promotion
from novel_agent.memory_store import (
    build_evidence,
    index_memory,
    load_source,
    memory_id,
    record_status,
)
from novel_agent.sqlite_transactions import write_transaction


@dataclass(frozen=True, slots=True)
class PromotionBatch:
    """Outcome of promoting every candidate attached to one artifact revision."""

    promoted: tuple[str, ...]
    blocked_reason: str | None = None


@contextmanager
def _transaction(connection: sqlite3.Connection) -> Iterator[None]:
    if connection.in_transaction:
        yield
        return
    with write_transaction(connection):
        yield


def current_memory_id(connection: sqlite3.Connection, proposal: MemoryProposal) -> str | None:
    """The current (non-superseded) memory row for the proposal's identity."""
    if proposal.subject_id is None:
        return None
    row = connection.execute(
        "SELECT current.memory_id FROM memory_records current "
        "WHERE current.project_id = ? AND current.layer = ? AND current.subject_id = ? "
        "AND current.predicate = ? AND NOT EXISTS ("
        "  SELECT 1 FROM memory_records successor WHERE successor.supersedes_id = current.memory_id"
        ") ORDER BY current.created_at DESC, current.memory_id DESC LIMIT 1",
        (
            proposal.project_id.root,
            proposal.target_layer.value,
            proposal.subject_id,
            proposal.predicate,
        ),
    ).fetchone()
    return None if row is None else str(row[0])


def _current_memory_value(connection: sqlite3.Connection, proposal: MemoryProposal) -> str | None:
    identifier = current_memory_id(connection, proposal)
    if identifier is None:
        return None
    row = connection.execute(
        "SELECT value_json FROM memory_records WHERE memory_id = ?", (identifier,)
    ).fetchone()
    return None if row is None else str(row[0])


def has_conflict(connection: sqlite3.Connection, evidence: CandidateEvidence) -> bool:
    """Whether a different value already holds the same (subject, predicate)."""
    proposal = evidence.proposal
    if proposal.subject_id is None:
        return False
    current_value = _current_memory_value(connection, proposal)
    return current_value is not None and current_value != proposal.value_json


def record_conflict(
    connection: sqlite3.Connection,
    proposal: MemoryProposal,
    candidate_identifier: str,
    decision: object,
    recorded_at: datetime,
) -> None:
    """Persist a conflicted/review-required promotion decision for auditability."""
    reason_code = getattr(decision, "review_code", None)
    outcome = getattr(decision, "outcome", None)
    reason = (
        reason_code.value
        if reason_code is not None
        else (outcome.value if outcome is not None else "conflict")
    )
    conflict_id = hashlib.sha256(
        f"{proposal.run_id}:{candidate_identifier}:{reason}".encode()
    ).hexdigest()[:32]
    connection.execute(
        "INSERT INTO memory_conflicts(conflict_id, project_id, run_id, candidate_id, "
        "existing_memory_id, reason_code, created_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(conflict_id) DO NOTHING",
        (
            conflict_id,
            proposal.project_id.root,
            proposal.run_id,
            candidate_identifier,
            current_memory_id(connection, proposal),
            reason,
            recorded_at.isoformat(),
        ),
    )


def _insert_supersedes_edge(
    connection: sqlite3.Connection,
    new_memory_id: str,
    superseded_memory_id: str,
    created_at: datetime,
) -> None:
    """Record the replacement in `memory_edges` (append-only audit graph)."""
    edge_id = hashlib.sha256(
        f"{new_memory_id}:{superseded_memory_id}:supersedes".encode()
    ).hexdigest()[:32]
    connection.execute(
        "INSERT INTO memory_edges(edge_id, source_memory_id, target_memory_id, relation, created_at) "
        "VALUES (?, ?, ?, 'supersedes', ?) ON CONFLICT(edge_id) DO NOTHING",
        (edge_id, new_memory_id, superseded_memory_id, created_at.isoformat()),
    )


def insert_promoted_memory(
    connection: sqlite3.Connection,
    *,
    candidate_identifier: str,
    proposal: MemoryProposal,
    evidence: CandidateEvidence,
    chapter_number: int | None,
    created_at: datetime,
    created_by: str,
) -> str:
    """Insert the promoted row, its FTS entry and its supersedes edge (idempotent)."""
    if proposal.subject_id is None:
        raise ValueError("promotable memory has no resolved subject")
    identifier = memory_id(candidate_identifier)
    connection.execute(
        "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, "
        "subject_id, predicate, value_json, chapter_start, chapter_end, source_artifact_id, "
        "source_artifact_revision, source_span, source_hash, confidence, created_by, created_at, supersedes_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(memory_id) DO NOTHING",
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
            chapter_number,
            chapter_number,
            proposal.source_artifact_id.root,
            proposal.source_artifact_revision,
            f"{evidence.source_byte_start}:{evidence.source_byte_end}",
            evidence.source_span_sha256,
            proposal.confidence,
            created_by,
            created_at.isoformat(),
            proposal.supersedes_memory_id,
        ),
    )
    # Idempotent: the FTS row can be missing even when the record exists
    # (derived index rebuilds, older databases), so always ensure it is there.
    index_memory(connection, identifier, proposal.value_json)
    if proposal.supersedes_memory_id is not None:
        _insert_supersedes_edge(connection, identifier, proposal.supersedes_memory_id, created_at)
    return identifier


def _set_candidate_status(
    connection: sqlite3.Connection,
    candidate_identifier: str,
    status: CandidateStatus,
) -> None:
    connection.execute(
        "UPDATE memory_candidates SET status = ? WHERE candidate_id = ?",
        (status.value, candidate_identifier),
    )


def promote_candidates(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    artifact_id: str,
    artifact_revision: int,
    chapter_number: int | None,
    committed_at: datetime,
    created_by: str,
) -> PromotionBatch:
    """Promote every stored candidate of one artifact revision.

    Automatic policy still applies: a candidate that needs review is marked and
    a conflict is recorded (and reported as blocked when the policy requires a
    pause).  Returns the promoted memory ids and an optional block reason.
    """
    promoted: list[str] = []
    blocked_reason: str | None = None
    with _transaction(connection):
        rows = connection.execute(
            "SELECT candidate_id, candidate_json, chapter_number FROM memory_candidates "
            "WHERE run_id = ? AND source_artifact_id = ? AND source_artifact_revision = ? "
            "ORDER BY candidate_id",
            (run_id, artifact_id, artifact_revision),
        ).fetchall()
        for row in rows:
            candidate_identifier = str(row[0])
            candidate = MemoryCandidateRecord.model_validate_json(str(row[1]))
            if candidate.status is CandidateStatus.PROMOTED:
                continue
            evidence = build_evidence(
                candidate.evidence.proposal,
                load_source(connection, candidate.evidence.proposal),
            )
            decision = decide_promotion(evidence, has_conflict(connection, evidence))
            if decision.outcome is PromotionOutcome.PROMOTED:
                identifier = insert_promoted_memory(
                    connection,
                    candidate_identifier=candidate_identifier,
                    proposal=evidence.proposal,
                    evidence=evidence,
                    chapter_number=int(row[2]) if row[2] is not None else chapter_number,
                    created_at=committed_at,
                    created_by=created_by,
                )
                _set_candidate_status(connection, candidate_identifier, CandidateStatus.PROMOTED)
                promoted.append(identifier)
                continue
            _set_candidate_status(connection, candidate_identifier, decision.candidate_status)
            _record_conflict_decision(connection, evidence.proposal, candidate_identifier, decision, committed_at)
            if decision.requires_pause and blocked_reason is None:
                blocked_reason = "memory_conflict"
    return PromotionBatch(tuple(promoted), blocked_reason)


def _record_conflict_decision(connection, proposal, candidate_identifier, decision, committed_at) -> None:
    record_conflict(connection, proposal, candidate_identifier, decision, committed_at)


__all__ = (
    "PromotionBatch",
    "current_memory_id",
    "has_conflict",
    "insert_promoted_memory",
    "promote_candidates",
    "record_conflict",
)
