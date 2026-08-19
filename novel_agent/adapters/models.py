from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum, unique

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from novel_agent.contracts import CharacterSet, ContractModel, MemoryLayer, PlanContract, Provenance, WorldBible
from novel_agent.memory_models import MemoryProposal
from novel_agent.pathing import ArtifactId, ProjectId


@unique
class CreativeArtifactKind(StrEnum):
    WORLD = "world"
    CHARACTERS = "characters"
    OUTLINE = "outline"
    CHAPTER = "chapter"
    EVALUATION = "evaluation"


@unique
class CreativeParseRoute(StrEnum):
    STRICT_JSON = "strict_json"
    JSON_REPAIR = "json_repair"
    SCHEMA_REPAIR = "schema_repair"
    LEGACY_NORMALIZATION = "legacy_normalization"


class RawCreativeArtifact(ContractModel):
    kind: CreativeArtifactKind
    raw_bytes: bytes = Field(repr=False)
    content_sha256: str
    adapter_version: str

    @model_validator(mode="after")
    def validate_content_digest(self) -> RawCreativeArtifact:
        actual_digest = hashlib.sha256(self.raw_bytes).hexdigest()
        if actual_digest == self.content_sha256:
            return self
        raise PydanticCustomError("creative_raw_digest", "raw artifact digest does not match its content")


class SchemaRepairRequest(ContractModel):
    raw_artifact: RawCreativeArtifact
    expected_contract: str
    contract_schema_json: str = Field(repr=False)


class SchemaFailure(ContractModel):
    raw_artifact: RawCreativeArtifact
    expected_contract: str
    parse_routes: tuple[CreativeParseRoute, ...]
    reason_code: str = "creative_schema_failure"


class ChapterDraft(ContractModel):
    chapter_draft_id: str
    chapter_plan_id: str
    body: str
    created_at: datetime
    provenance: Provenance


class WorldAdaptation(ContractModel):
    raw: RawCreativeArtifact
    value: WorldBible
    parse_route: CreativeParseRoute


class CharacterAdaptation(ContractModel):
    raw: RawCreativeArtifact
    value: CharacterSet
    parse_route: CreativeParseRoute


class OutlineAdaptation(ContractModel):
    raw: RawCreativeArtifact
    value: PlanContract
    parse_route: CreativeParseRoute


class ChapterAdaptation(ContractModel):
    raw: RawCreativeArtifact
    value: ChapterDraft
    parse_route: CreativeParseRoute


class M0CandidateExtractionRequest(ContractModel):
    run_id: str
    project_id: ProjectId
    source_artifact_id: ArtifactId
    source_artifact_revision: int
    chapter_body: str
    #: The verbatim artifact text the memory span is measured against.  The
    #: staged chapter raw is the JSON envelope (canonical since the JSON
    #: unification), so memory spans must reference that text, not the bare
    #: prose body, or the span validation rejects every candidate.
    source_text: str | None = None
    subject_id: str
    predicate: str
    value_json: str
    target_layer: MemoryLayer
    created_at: datetime


class M0CandidateExtraction(ContractModel):
    request: M0CandidateExtractionRequest
    proposal: MemoryProposal


class LegacyEvaluationRequest(ContractModel):
    chapter_body: str
    characters_json: str
    chapter_label: str
    chapter_requirement: str


class LegacyEvaluationResult(ContractModel):
    raw_bytes: bytes = Field(repr=False)
    raw_sha256: str
    scores_json: str
    reasons_json: str
    average_score: float

    @model_validator(mode="after")
    def validate_raw_digest(self) -> LegacyEvaluationResult:
        actual_digest = hashlib.sha256(self.raw_bytes).hexdigest()
        if actual_digest == self.raw_sha256:
            return self
        raise PydanticCustomError("creative_evaluation_digest", "evaluation digest does not match its content")


class LegacyEvaluationFailure(ContractModel):
    raw_bytes: bytes = Field(repr=False)
    raw_sha256: str
    reason_code: str

    @model_validator(mode="after")
    def validate_raw_digest(self) -> LegacyEvaluationFailure:
        actual_digest = hashlib.sha256(self.raw_bytes).hexdigest()
        if actual_digest == self.raw_sha256:
            return self
        raise PydanticCustomError("creative_evaluation_digest", "evaluation digest does not match its content")


__all__ = (
    "ChapterAdaptation",
    "ChapterDraft",
    "CharacterAdaptation",
    "CreativeArtifactKind",
    "CreativeParseRoute",
    "LegacyEvaluationRequest",
    "LegacyEvaluationFailure",
    "LegacyEvaluationResult",
    "M0CandidateExtraction",
    "M0CandidateExtractionRequest",
    "OutlineAdaptation",
    "RawCreativeArtifact",
    "SchemaFailure",
    "SchemaRepairRequest",
    "WorldAdaptation",
)
