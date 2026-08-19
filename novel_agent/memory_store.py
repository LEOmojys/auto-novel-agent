from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import assert_never

from novel_agent.contracts.common import MemoryLayer
from novel_agent.memory_models import (
    CandidateEvidence,
    MemoryCandidateRecord,
    MemoryProposal,
    PromotedMemory,
    SourceArtifactStatus,
)
from novel_agent.pathing import ArtifactId, ProjectId
from novel_agent.storage import write_transaction


@unique
class MemoryErrorCode(StrEnum):
    INVALID_PROMOTION = "invalid_promotion"
    INVALID_SUPERSESSION = "invalid_supersession"
    PROMOTION_MISSING = "promotion_missing"
    SOURCE_INTEGRITY = "source_integrity"
    SOURCE_NOT_ELIGIBLE = "source_not_eligible"
    SOURCE_NOT_FOUND = "source_not_found"
    SOURCE_NOT_TEXT = "source_not_text"
    SOURCE_SPAN_INVALID = "source_span_invalid"


@dataclass(frozen=True, slots=True)
class MemoryGovernanceError(Exception):
    code: MemoryErrorCode

    def __str__(self) -> str:
        return self.code.value


@dataclass(frozen=True, slots=True)
class ArtifactSource:
    artifact_id: ArtifactId
    revision: int
    project_id: ProjectId
    status: SourceArtifactStatus
    raw_bytes: bytes
    sha256: str


def load_source(connection: sqlite3.Connection, proposal: MemoryProposal) -> ArtifactSource:
    row = connection.execute(
        "SELECT artifact_id, revision, project_id, status, raw_bytes, sha256 FROM artifacts "
        "WHERE artifact_id = ? AND revision = ?",
        (proposal.source_artifact_id.root, proposal.source_artifact_revision),
    ).fetchone()
    if row is None:
        raise MemoryGovernanceError(MemoryErrorCode.SOURCE_NOT_FOUND)
    project_id = ProjectId(str(row[2]))
    if project_id.root != proposal.project_id.root:
        raise MemoryGovernanceError(MemoryErrorCode.SOURCE_NOT_ELIGIBLE)
    source = ArtifactSource(
        artifact_id=ArtifactId(str(row[0])),
        revision=int(row[1]),
        project_id=project_id,
        status=_source_status(str(row[3])),
        raw_bytes=bytes(row[4]),
        sha256=str(row[5]),
    )
    if hashlib.sha256(source.raw_bytes).hexdigest() != source.sha256:
        raise MemoryGovernanceError(MemoryErrorCode.SOURCE_INTEGRITY)
    return source


def build_evidence(proposal: MemoryProposal, source: ArtifactSource) -> CandidateEvidence:
    try:
        source_text = source.raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MemoryGovernanceError(MemoryErrorCode.SOURCE_NOT_TEXT) from error
    if proposal.source_end > len(source_text):
        raise MemoryGovernanceError(MemoryErrorCode.SOURCE_SPAN_INVALID)
    excerpt = source_text[proposal.source_start : proposal.source_end]
    if excerpt != proposal.source_excerpt:
        raise MemoryGovernanceError(MemoryErrorCode.SOURCE_SPAN_INVALID)
    byte_start = len(source_text[: proposal.source_start].encode())
    byte_end = len(source_text[: proposal.source_end].encode())
    span = source.raw_bytes[byte_start:byte_end]
    if span != proposal.source_excerpt.encode():
        raise MemoryGovernanceError(MemoryErrorCode.SOURCE_SPAN_INVALID)
    return CandidateEvidence(
        proposal=proposal,
        source_artifact_sha256=source.sha256,
        source_byte_start=byte_start,
        source_byte_end=byte_end,
        source_span_sha256=hashlib.sha256(span).hexdigest(),
        source_status=source.status,
        normalized_fact_hash=proposal.normalized_fact_hash(),
    )


def candidate_id(evidence: CandidateEvidence) -> str:
    proposal = evidence.proposal
    identity = "\x00".join(
        (
            proposal.source_artifact_id.root,
            str(proposal.source_artifact_revision),
            str(proposal.extraction_schema_version),
            evidence.normalized_fact_hash,
        )
    )
    return f"candidate:{hashlib.sha256(identity.encode()).hexdigest()}"


def memory_id(candidate_identifier: str) -> str:
    return f"memory:{hashlib.sha256(candidate_identifier.encode()).hexdigest()}"


def record_status(layer: MemoryLayer) -> str:
    match layer:
        case MemoryLayer.M1_CANONICAL:
            return "canonical"
        case MemoryLayer.M2_PLAN:
            return "validated"
        case MemoryLayer.M3_CONFIRMED:
            return "confirmed"
        case MemoryLayer.M0_CANDIDATE | MemoryLayer.M4_RETRIEVAL | MemoryLayer.M5_WORKING:
            raise MemoryGovernanceError(MemoryErrorCode.INVALID_PROMOTION)
        case unreachable:
            assert_never(unreachable)


def load_promoted_memory(
    connection: sqlite3.Connection,
    memory_identifier: str,
) -> PromotedMemory | None:
    row = connection.execute(
        "SELECT layer, source_artifact_id, source_artifact_revision, source_span, source_hash, "
        "value_json, supersedes_id FROM memory_records WHERE memory_id = ?",
        (memory_identifier,),
    ).fetchone()
    if row is None:
        return None
    return PromotedMemory(
        memory_id=memory_identifier,
        memory_layer=MemoryLayer(str(row[0])),
        source_artifact_id=ArtifactId(str(row[1])),
        source_artifact_revision=int(row[2]),
        source_span=str(row[3]),
        source_span_sha256=str(row[4]),
        value_json=str(row[5]),
        supersedes_memory_id=None if row[6] is None else str(row[6]),
    )


def refresh_candidate_evidence(
    connection: sqlite3.Connection,
    candidate: MemoryCandidateRecord,
    evidence: CandidateEvidence,
) -> MemoryCandidateRecord:
    existing = candidate.evidence
    immutable_evidence = (
        existing.source_artifact_sha256,
        existing.source_byte_start,
        existing.source_byte_end,
        existing.source_span_sha256,
    )
    refreshed_evidence = (
        evidence.source_artifact_sha256,
        evidence.source_byte_start,
        evidence.source_byte_end,
        evidence.source_span_sha256,
    )
    if immutable_evidence != refreshed_evidence:
        raise MemoryGovernanceError(MemoryErrorCode.SOURCE_INTEGRITY)
    if existing.source_status is evidence.source_status:
        return candidate
    with write_transaction(connection):
        connection.execute(
            "UPDATE memory_candidates SET candidate_json = ? WHERE candidate_id = ?",
            (evidence.model_dump_json(), candidate.candidate_id),
        )
    return candidate.model_copy(update={"evidence": evidence})


def _source_status(value: str) -> SourceArtifactStatus:
    try:
        return SourceArtifactStatus(value)
    except ValueError:
        raise MemoryGovernanceError(MemoryErrorCode.SOURCE_NOT_ELIGIBLE) from None


def rebuild_retrieval_index(connection: sqlite3.Connection, project_id: str) -> int:
    """Rebuild the derived M4 FTS index from current memory records.

    The FTS view is fully derived and rebuildable: dropping and repopulating
    it never changes story truth, which lives in ``memory_records``.
    """
    connection.execute("DELETE FROM memory_fts")
    rows = connection.execute(
        "SELECT memory_id, value_json FROM memory_records WHERE project_id = ?",
        (project_id,),
    ).fetchall()
    connection.executemany(
        "INSERT INTO memory_fts(memory_id, content) VALUES (?, ?)",
        ((str(row[0]), str(row[1])) for row in rows),
    )
    return len(rows)


__all__ = (
    "ArtifactSource",
    "MemoryErrorCode",
    "MemoryGovernanceError",
    "build_evidence",
    "candidate_id",
    "load_promoted_memory",
    "load_source",
    "memory_id",
    "rebuild_retrieval_index",
    "record_status",
    "refresh_candidate_evidence",
)
