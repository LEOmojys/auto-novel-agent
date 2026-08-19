from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic_core import PydanticCustomError

from novel_agent.contracts.common import validate_sha256
from novel_agent.pathing import ArtifactExtension, ArtifactId, ProjectId


@unique
class ArtifactFaultPoint(StrEnum):
    BEFORE_DB_COMMIT = "before_db_commit"
    AFTER_DB_COMMIT = "after_db_commit"
    AFTER_TEMP_FSYNC = "after_temp_fsync"
    AFTER_REPLACE = "after_replace"


class ConfirmedMemoryWrite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    memory_id: str
    plan_revision: int
    kind: str
    subject_id: str
    predicate: str
    value_json: str
    source_start: int
    source_end: int
    source_sha256: str
    confidence: float
    created_by: str

    @field_validator("plan_revision", "source_start")
    @classmethod
    def validate_nonnegative(cls, value: int) -> int:
        if value < 0:
            raise PydanticCustomError("artifact_negative_value", "value must be nonnegative")
        return value

    @field_validator("source_sha256")
    @classmethod
    def validate_source_sha256(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("value_json")
    @classmethod
    def validate_value_json(cls, value: str) -> str:
        try:
            json.loads(value)
        except json.JSONDecodeError as error:
            raise PydanticCustomError("artifact_invalid_json", "value_json must be valid JSON") from error
        return value

    @model_validator(mode="after")
    def validate_span_and_confidence(self) -> ConfirmedMemoryWrite:
        if self.source_end <= self.source_start:
            raise PydanticCustomError("artifact_invalid_span", "source span must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise PydanticCustomError("artifact_invalid_confidence", "confidence must be between zero and one")
        return self


class ArtifactCommitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    artifact_id: ArtifactId
    revision: int
    project_id: ProjectId
    run_id: str | None
    kind: str
    raw_bytes: bytes
    extension: ArtifactExtension
    schema_version: int
    created_at: str
    memories: tuple[ConfirmedMemoryWrite, ...]

    @field_validator("revision", "schema_version")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value < 1:
            raise PydanticCustomError("artifact_nonpositive_value", "value must be positive")
        return value


@dataclass(frozen=True, slots=True)
class ArtifactCommitReceipt:
    artifact_id: ArtifactId
    revision: int
    sha256: str
    target_path: Path
    outbox_id: str


@dataclass(frozen=True, slots=True)
class MaterializationReport:
    processed: int
    materialized: int
    already_present: int
    failed: int


@dataclass(frozen=True, slots=True)
class EvidenceMismatchError(Exception):
    memory_id: str
    expected_sha256: str
    actual_sha256: str

    def __str__(self) -> str:
        return f"evidence_checksum_mismatch: {self.memory_id}"


@dataclass(slots=True)
class ArtifactRunAuthorityError(Exception):
    run_id: str
    request_project_id: str
    persisted_project_id: str | None

    def __str__(self) -> str:
        if self.persisted_project_id is None:
            return f"artifact_commit_run_not_found: {self.run_id}"
        return (
            f"artifact_commit_project_mismatch: run {self.run_id} belongs to "
            f"{self.persisted_project_id}, not {self.request_project_id}"
        )


@dataclass(frozen=True, slots=True)
class OutboxTargetMismatchError(Exception):
    outbox_id: str

    def __str__(self) -> str:
        return f"outbox_target_mismatch: {self.outbox_id}"


@dataclass(frozen=True, slots=True)
class MaterializationChecksumError(Exception):
    outbox_id: str

    def __str__(self) -> str:
        return f"materialized_checksum_mismatch: {self.outbox_id}"


@dataclass(frozen=True, slots=True)
class ArtifactAuthorityMismatchError(Exception):
    outbox_id: str

    def __str__(self) -> str:
        return f"artifact_authority_checksum_mismatch: {self.outbox_id}"


FaultHook = Callable[[ArtifactFaultPoint], None]


def no_fault(_point: ArtifactFaultPoint) -> None:
    return


__all__ = (
    "ArtifactAuthorityMismatchError",
    "ArtifactCommitReceipt",
    "ArtifactCommitRequest",
    "ArtifactFaultPoint",
    "ArtifactRunAuthorityError",
    "ConfirmedMemoryWrite",
    "EvidenceMismatchError",
    "FaultHook",
    "MaterializationChecksumError",
    "MaterializationReport",
    "OutboxTargetMismatchError",
    "no_fault",
)
