from __future__ import annotations

import hashlib
import re
from enum import StrEnum, unique
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from novel_agent.contracts import RunStatus
from novel_agent.tool_registry import AgentRole


_ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_VERSION_PATTERN: Final = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
BUILTIN_SKILL_IDS: Final = frozenset(
    {
        "brief-normalization",
        "plan-execution",
        "world-building",
        "cast-network",
        "story-architecture",
        "chapter-scene",
        "continuity-audit",
        "quality-review",
        "targeted-revision",
        "memory-extraction",
    }
)


@unique
class SkillStatus(StrEnum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    ENABLED = "enabled"
    DEPRECATED = "deprecated"


@unique
class SkillKind(StrEnum):
    PRIMARY = "primary"
    AUDIT = "audit"


@unique
class SelectionCode(StrEnum):
    SELECTED = "SKILL_SELECTED"
    SKILL_AMBIGUOUS = "SKILL_AMBIGUOUS"
    SKILL_NOT_FOUND = "SKILL_NOT_FOUND"


class SchemaCompatibility(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_input: int = Field(ge=1)
    maximum_input: int = Field(ge=1)
    minimum_output: int = Field(ge=1)
    maximum_output: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_ranges(self) -> SchemaCompatibility:
        if self.minimum_input > self.maximum_input or self.minimum_output > self.maximum_output:
            raise PydanticCustomError("skill_schema_range", "schema compatibility ranges must be ordered")
        return self


class SkillManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_id: str
    version: str
    status: SkillStatus = SkillStatus.DISABLED
    kind: SkillKind
    summary: str
    owner: str
    changelog: str
    task_types: frozenset[str]
    run_states: frozenset[RunStatus]
    allowed_roles: frozenset[AgentRole]
    genre_tags: frozenset[str] = frozenset()
    input_model: str
    output_model: str
    required_artifact_kinds: frozenset[str]
    required_context_slices: frozenset[str]
    context_order: tuple[str, ...]
    context_budget: int = Field(gt=0)
    allowed_tool_ids: frozenset[str]
    max_tool_calls: int = Field(ge=0)
    timeout_budget: float = Field(gt=0.0)
    acceptance_checks: tuple[str, ...]
    failure_strategy: str
    max_repairs: int = Field(ge=0)
    prompt_template_path: str
    prompt_template_hash: str
    examples_hash: str
    compatibility: SchemaCompatibility
    audit_for: frozenset[str] = frozenset()

    @computed_field
    @property
    def skill_ref(self) -> str:
        return f"{self.skill_id}@{self.version}"

    @computed_field
    @property
    def manifest_hash(self) -> str:
        payload = self.model_dump_json(exclude={"manifest_hash", "skill_ref"})
        return hashlib.sha256(payload.encode()).hexdigest()

    @field_validator("skill_id")
    @classmethod
    def validate_skill_id(cls, value: str) -> str:
        if _ID_PATTERN.fullmatch(value) is None:
            raise PydanticCustomError("skill_id_invalid", "skill_id must be lowercase kebab-case")
        return value

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if _VERSION_PATTERN.fullmatch(value) is None:
            raise PydanticCustomError("skill_version_invalid", "version must be stable semantic versioning")
        return value

    @field_validator("prompt_template_hash", "examples_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise PydanticCustomError("skill_hash_invalid", "hash fields must contain a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_declarations(self) -> SkillManifest:
        if frozenset(self.context_order) != self.required_context_slices:
            raise PydanticCustomError(
                "skill_context_order", "context_order must name each required context slice exactly once"
            )
        if any("@" not in tool_ref for tool_ref in self.allowed_tool_ids):
            raise PydanticCustomError("skill_tool_version", "allowed tools must be exact versioned references")
        return self


class TaskDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    plan_revision: int = Field(ge=1)
    task_id: str
    task_type: str
    run_state: RunStatus
    actor_role: AgentRole
    input_schema_version: int = Field(ge=1)
    expected_output_schema_version: int = Field(ge=1)
    available_context_slices: frozenset[str]
    artifact_kinds: frozenset[str]
    genre_tags: frozenset[str]
    authorized_tool_ids: frozenset[str]
    remaining_context_tokens: int = Field(ge=0)
    remaining_tool_calls: int = Field(ge=0)
    remaining_repairs: int = Field(ge=0)
    remaining_timeout_seconds: float = Field(ge=0.0)
    policy_flags: frozenset[str] = frozenset()
    opaque_data_slots: tuple[str, ...] = Field(default=(), repr=False)

    @computed_field
    @property
    def descriptor_hash(self) -> str:
        payload = self.model_dump_json(exclude={"descriptor_hash"})
        return hashlib.sha256(payload.encode()).hexdigest()


class CandidateEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_ref: str
    score: int | None
    score_evidence: tuple[str, ...]
    reasons: tuple[str, ...]


class SelectedSkill(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: SkillManifest
    score: int
    score_evidence: tuple[str, ...]

    @property
    def skill_ref(self) -> str:
        return self.manifest.skill_ref

    @property
    def manifest_hash(self) -> str:
        return self.manifest.manifest_hash

    @property
    def template_hash(self) -> str:
        return self.manifest.prompt_template_hash


class SkillSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: SelectionCode = SelectionCode.SELECTED
    descriptor_hash: str
    primary: SelectedSkill
    audit: SelectedSkill | None
    candidates: tuple[CandidateEvidence, ...]
    shadow: bool = False


class SkillPause(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: SelectionCode
    descriptor_hash: str
    reason: str
    candidates: tuple[CandidateEvidence, ...]


class ActiveSkillPin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    manifest: SkillManifest
    manifest_hash: str
    template_hash: str
