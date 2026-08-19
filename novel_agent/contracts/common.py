from __future__ import annotations

from datetime import datetime
from enum import StrEnum, unique
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_core import PydanticCustomError


CONTRACT_SCHEMA_VERSION: Final = 1
SHA256_LENGTH: Final = 64


@unique
class ContractErrorCode(StrEnum):
    DUPLICATE_CHAPTER = "contract_duplicate_chapter"
    INVALID_CONTEXT_BUDGET = "contract_invalid_context_budget"
    INVALID_REQUEST_DIGEST = "contract_invalid_request_digest"
    INVALID_SHA256 = "contract_invalid_sha256"
    INVALID_STATE_TRANSITION = "contract_invalid_state_transition"
    INVALID_MEMORY_LAYER = "contract_invalid_memory_layer"
    MISSING_EVIDENCE = "contract_missing_evidence"


@unique
class ProvenanceKind(StrEnum):
    ARTIFACT_REVISION = "artifact_revision"
    LEGACY_IMPORT = "legacy_import"
    PROVIDER_RESPONSE = "provider_response"
    SYSTEM = "system"
    TOOL_INVOCATION = "tool_invocation"
    USER_REQUEST = "user_request"


@unique
class PlanApprovalStatus(StrEnum):
    DRAFT = "draft"
    NEEDS_REVIEW = "needs_review"
    VALIDATED = "validated"


@unique
class RunStatus(StrEnum):
    CANCELLED = "cancelled"
    CANCEL_REQUESTED = "cancel_requested"
    FAILED = "failed"
    PAUSED = "paused"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    WAITING_REVIEW = "waiting_review"


@unique
class TaskStatus(StrEnum):
    CANCELLED = "cancelled"
    FAILED = "failed"
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    WAITING_REVIEW = "waiting_review"


@unique
class RunCommandType(StrEnum):
    CANCEL = "cancel"
    PAUSE = "pause"
    REPAIR = "repair"
    RESUME = "resume"
    REVISE = "revise"
    START = "start"


@unique
class ProviderOutcome(StrEnum):
    DISCARDED = "discarded"
    FAILED = "failed"
    SUCCEEDED = "succeeded"


@unique
class ToolOutcome(StrEnum):
    DENIED = "denied"
    FAILED = "failed"
    SUCCEEDED = "succeeded"


@unique
class ArtifactStatus(StrEnum):
    ACCEPTED = "accepted"
    DRAFT = "draft"
    SUPERSEDED = "superseded"


@unique
class QualityOutcome(StrEnum):
    ACCEPT = "accept"
    PAUSE = "pause"
    REPAIR = "repair"


@unique
class MemoryLayer(StrEnum):
    M0_CANDIDATE = "m0_candidate"
    M1_CANONICAL = "m1_canonical"
    M2_PLAN = "m2_plan"
    M3_CONFIRMED = "m3_confirmed"
    M4_RETRIEVAL = "m4_retrieval"
    M5_WORKING = "m5_working"


def contract_error(code: ContractErrorCode, message: str) -> PydanticCustomError:
    return PydanticCustomError(code.value, message)


def validate_sha256(value: str) -> str:
    if len(value) != SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        raise contract_error(ContractErrorCode.INVALID_SHA256, "value must be a lowercase SHA-256 digest")
    return value


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = CONTRACT_SCHEMA_VERSION


class Provenance(ContractModel):
    source_kind: ProvenanceKind
    source_id: str
    source_sha256: str
    observed_at: datetime

    @field_validator("source_sha256")
    @classmethod
    def validate_source_sha256(cls, value: str) -> str:
        return validate_sha256(value)


class EvidenceRef(ContractModel):
    evidence_id: str
    artifact_id: str
    artifact_revision: int
    source_sha256: str
    recorded_at: datetime

    @field_validator("artifact_revision")
    @classmethod
    def validate_artifact_revision(cls, value: int) -> int:
        if value < 1:
            raise contract_error(ContractErrorCode.INVALID_CONTEXT_BUDGET, "artifact revision must be positive")
        return value

    @field_validator("source_sha256")
    @classmethod
    def validate_source_sha256(cls, value: str) -> str:
        return validate_sha256(value)


class EvidenceBackedContract(ContractModel):
    evidence: tuple[EvidenceRef, ...]

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, value: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
        if not value:
            raise contract_error(ContractErrorCode.MISSING_EVIDENCE, "evidence cannot be empty")
        return value
