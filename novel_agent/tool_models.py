from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from novel_agent.contracts import RunStatus, ToolOutcome


_ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$")
_VERSION_PATTERN: Final = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


@unique
class AgentRole(StrEnum):
    BRIEF_PLANNER = "BriefPlanner"
    STORY_ARCHITECT = "StoryArchitect"
    CAST_DESIGNER = "CastDesigner"
    OUTLINE_PLANNER = "OutlinePlanner"
    CHAPTER_EXECUTOR = "ChapterExecutor"
    CONTINUITY_AUDITOR = "ContinuityAuditor"
    QUALITY_REVIEWER = "QualityReviewer"
    MEMORY_EXTRACTOR = "MemoryExtractor"
    RUN_COORDINATOR = "RunCoordinator"
    USER = "User"


@unique
class AccessLevel(StrEnum):
    READ = "read"
    PROVIDER = "provider"
    STAGE_WRITE = "stage_write"
    CANDIDATE_WRITE = "candidate_write"
    COORDINATOR_COMMIT = "coordinator_commit"
    REVIEW_ACTION = "review_action"


@unique
class ApprovalPolicy(StrEnum):
    NONE = "none"
    COORDINATOR = "coordinator"
    EXPLICIT_USER = "explicit_user"


@unique
class ArtifactPolicy(StrEnum):
    NONE = "none"
    STAGE_ONLY = "stage_only"
    CANDIDATE_ONLY = "candidate_only"
    COMMIT_ACCEPTED = "commit_accepted"


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = Field(ge=1, le=5)
    retryable_error_codes: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ToolRegistryError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    tool_id: str
    version: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    allowed_roles: frozenset[AgentRole]
    allowed_skill_ids: frozenset[str]
    allowed_run_states: frozenset[RunStatus]
    access_level: AccessLevel
    timeout_seconds: float = Field(gt=0.0, le=600.0)
    retry_policy: RetryPolicy
    idempotent: bool
    approval_policy: ApprovalPolicy
    artifact_policy: ArtifactPolicy

    @property
    def tool_ref(self) -> str:
        return f"{self.tool_id}@{self.version}"

    @field_validator("tool_id")
    @classmethod
    def validate_tool_id(cls, value: str) -> str:
        if _ID_PATTERN.fullmatch(value) is None:
            raise PydanticCustomError("tool_id_invalid", "tool_id must be a lowercase dotted identifier")
        return value

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if _VERSION_PATTERN.fullmatch(value) is None:
            raise PydanticCustomError("tool_version_invalid", "version must be stable semantic versioning")
        return value

    @model_validator(mode="after")
    def validate_write_policy(self) -> ToolSpec:
        write_levels = {
            AccessLevel.STAGE_WRITE,
            AccessLevel.CANDIDATE_WRITE,
            AccessLevel.COORDINATOR_COMMIT,
            AccessLevel.REVIEW_ACTION,
        }
        if self.access_level in write_levels and self.approval_policy is ApprovalPolicy.NONE:
            raise PydanticCustomError("tool_approval_required", "write-capable tools require an approval policy")
        return self


class ToolInvocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation_id: str
    run_id: str
    task_id: str | None = None
    tool_ref: str
    idempotency_key: str
    actor_role: AgentRole
    skill_ref: str
    run_status: RunStatus
    input_data: BaseModel
    approval_granted: bool = False
    shadow: bool = False


class InvocationTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation_id: str
    tool_ref: str
    skill_ref: str
    idempotency_key: str
    input_sha256: str
    output_sha256: str | None
    outcome: ToolOutcome
    reason: str
    reused: bool
    shadow: bool


class ToolCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: ToolOutcome
    data: BaseModel | None
    reason: str
    trace: InvocationTrace


class InvocationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: ToolOutcome
    reason: str
    data: BaseModel | None = None
    output_hash: str | None = None
    shadow: bool = False


ToolHandler = Callable[[BaseModel], BaseModel]


__all__ = (
    "AccessLevel",
    "AgentRole",
    "ApprovalPolicy",
    "ArtifactPolicy",
    "InvocationDecision",
    "InvocationTrace",
    "RetryPolicy",
    "ToolCallResult",
    "ToolHandler",
    "ToolInvocationRequest",
    "ToolRegistryError",
    "ToolSpec",
)
