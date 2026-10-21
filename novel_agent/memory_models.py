from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum, unique
from typing import Literal, Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from novel_agent.contracts.common import MemoryLayer, validate_sha256
from novel_agent.pathing import ArtifactId, ProjectId


@unique
class CandidateStatus(StrEnum):
    QUARANTINED = "quarantined"
    NEEDS_REVIEW = "needs_review"
    CONFLICTED = "conflicted"
    PROMOTED = "promoted"
    REJECTED = "rejected"


@unique
class EntityResolution(StrEnum):
    EXACT = "exact"
    AMBIGUOUS = "ambiguous"
    NEW = "new"
    UNRESOLVED = "unresolved"


@unique
class InferenceMode(StrEnum):
    DIRECT = "direct"
    RELATIONSHIP = "relationship"
    CAUSAL = "causal"
    MOTIVATION = "motivation"


@unique
class MemoryFactKind(StrEnum):
    WORLD_RULE = "world_rule"
    CHARACTER_IDENTITY = "character_identity"
    PLAN_GOAL = "plan_goal"
    PLAN_HOOK = "plan_hook"
    CHAPTER_SUMMARY = "chapter_summary"
    DIRECT_EVENT = "direct_event"
    STATE_CHANGE = "state_change"
    HOOK_STATUS = "hook_status"
    RELATIONSHIP = "relationship"
    CAUSALITY = "causality"
    MOTIVATION = "motivation"


@unique
class PromotionAuthority(StrEnum):
    VALIDATED_WORLD = "validated_world"
    USER_RETCON = "user_retcon"
    VALIDATED_PLAN = "validated_plan"
    ACCEPTED_CHAPTER = "accepted_chapter"
    #: A human explicitly accepted a candidate the automatic policy could not
    #: promote (inference, low confidence, entity ambiguity) without changing
    #: an existing fact.
    USER_REVIEW = "user_review"


@unique
class PromotionOutcome(StrEnum):
    PROMOTED = "promoted"
    REVIEW_REQUIRED = "review_required"
    CONFLICTED = "conflicted"


@unique
class ReviewCode(StrEnum):
    AMBIGUOUS_ENTITY = "ambiguous_entity"
    NEW_ENTITY = "new_entity"
    UNRESOLVED_ENTITY = "unresolved_entity"
    INFERRED_RELATIONSHIP = "inferred_relationship"
    INFERRED_CAUSALITY = "inferred_causality"
    INFERRED_MOTIVATION = "inferred_motivation"
    UNCERTAIN_TIME = "uncertain_time"
    LOW_CONFIDENCE = "low_confidence"
    SOURCE_NOT_ACCEPTED = "source_not_accepted"
    INVALID_PROMOTION = "invalid_promotion"
    CONFLICT = "conflict"


@unique
class SourceArtifactStatus(StrEnum):
    ACCEPTED = "accepted"
    STAGED = "staged"


@unique
class TemporalCertainty(StrEnum):
    EXACT = "exact"
    UNCERTAIN = "uncertain"


def _candidate_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


def _canonical_json(value: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise _candidate_error("memory_invalid_json", "value_json must be valid JSON") from error
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class MemoryProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    project_id: ProjectId
    source_artifact_id: ArtifactId
    source_artifact_revision: int = Field(ge=1)
    extraction_schema_version: int = Field(ge=1)
    target_layer: MemoryLayer
    kind: MemoryFactKind
    subject_id: str | None
    predicate: str = Field(min_length=1)
    value_json: str
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=1)
    source_excerpt: str = Field(min_length=1)
    entity_resolution: EntityResolution
    inference_mode: InferenceMode
    temporal_certainty: TemporalCertainty
    confidence: float = Field(ge=0.0, le=1.0)
    authority: PromotionAuthority
    plan_revision: int = Field(ge=0)
    supersedes_memory_id: str | None
    created_at: datetime

    @field_validator("value_json")
    @classmethod
    def normalize_value_json(cls, value: str) -> str:
        return _canonical_json(value)

    @model_validator(mode="after")
    def validate_target_layer(self) -> Self:
        match self.target_layer:
            case MemoryLayer.M1_CANONICAL | MemoryLayer.M2_PLAN | MemoryLayer.M3_CONFIRMED:
                return self
            case MemoryLayer.M0_CANDIDATE | MemoryLayer.M4_RETRIEVAL | MemoryLayer.M5_WORKING:
                raise _candidate_error("memory_invalid_target_layer", "candidates may only target M1, M2, or M3")
            case unreachable:
                assert_never(unreachable)

    @model_validator(mode="after")
    def validate_entity_resolution(self) -> Self:
        match self.entity_resolution:
            case EntityResolution.EXACT:
                if self.subject_id is not None:
                    return self
                raise _candidate_error("memory_missing_entity_id", "exact resolution requires a stable entity ID")
            case EntityResolution.AMBIGUOUS | EntityResolution.NEW | EntityResolution.UNRESOLVED:
                if self.subject_id is None:
                    return self
                raise _candidate_error("memory_untrusted_entity_id", "non-exact resolutions cannot carry an entity ID")
            case unreachable:
                assert_never(unreachable)

    @model_validator(mode="after")
    def validate_source_span(self) -> Self:
        if self.source_end > self.source_start:
            return self
        raise _candidate_error("memory_invalid_span", "source span must be non-empty")

    def normalized_fact_hash(self) -> str:
        payload = {
            "kind": self.kind.value,
            "predicate": self.predicate,
            "subject_id": self.subject_id,
            "target_layer": self.target_layer.value,
            "value_json": self.value_json,
        }
        normalized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()


class CandidateEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal: MemoryProposal
    source_artifact_sha256: str
    source_byte_start: int = Field(ge=0)
    source_byte_end: int = Field(ge=1)
    source_span_sha256: str
    source_status: SourceArtifactStatus
    normalized_fact_hash: str

    @field_validator("source_artifact_sha256", "source_span_sha256", "normalized_fact_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return validate_sha256(value)

    @model_validator(mode="after")
    def validate_byte_span(self) -> Self:
        if self.source_byte_end > self.source_byte_start:
            return self
        raise _candidate_error("memory_invalid_byte_span", "source byte span must be non-empty")


class MemoryCandidateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    memory_layer: Literal[MemoryLayer.M0_CANDIDATE] = MemoryLayer.M0_CANDIDATE
    evidence: CandidateEvidence
    status: CandidateStatus
    review_code: ReviewCode | None = None


class PromotedMemory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str
    memory_layer: MemoryLayer
    source_artifact_id: ArtifactId
    source_artifact_revision: int
    source_span: str
    source_span_sha256: str
    value_json: str
    supersedes_memory_id: str | None

    @field_validator("source_span_sha256")
    @classmethod
    def validate_span_hash(cls, value: str) -> str:
        return validate_sha256(value)


class MemoryPromotionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: MemoryCandidateRecord
    outcome: PromotionOutcome
    memory: PromotedMemory | None
    requires_pause: bool


class RetrievalView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieval_id: str
    memory_layer: Literal[MemoryLayer.M4_RETRIEVAL] = MemoryLayer.M4_RETRIEVAL
    source_memory_ids: tuple[str, ...]
    assembled_at: datetime


class WorkingContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_id: str
    memory_layer: Literal[MemoryLayer.M5_WORKING] = MemoryLayer.M5_WORKING
    retrieval_id: str
    token_budget: int = Field(ge=1)
    assembled_at: datetime


__all__ = (
    "CandidateEvidence",
    "CandidateStatus",
    "EntityResolution",
    "InferenceMode",
    "MemoryCandidateRecord",
    "MemoryFactKind",
    "MemoryPromotionResult",
    "MemoryProposal",
    "PromotedMemory",
    "PromotionAuthority",
    "PromotionOutcome",
    "RetrievalView",
    "ReviewCode",
    "SourceArtifactStatus",
    "TemporalCertainty",
    "WorkingContext",
)
