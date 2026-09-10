from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, unique
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from novel_agent.contracts import RunCommandType
from novel_agent.pathing import ProjectId


class RunModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


@unique
class CommitDisposition(StrEnum):
    COMMITTED = "committed"
    DISCARDED = "discarded"


class StartRunRequest(RunModel):
    project_id: ProjectId
    run_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    brief_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    execution_payload: "StartRunExecutionPayload | None" = None


class StartRunExecutionPayload(RunModel):
    project_id: ProjectId
    run_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    user_requirements: str = Field(min_length=1, max_length=100_000)
    provider: Literal["offline_fake", "legacy_real", "agent_loop"] | None = None
    target_chapter_count: int | None = Field(default=None, ge=1, le=100)
    auto_approve: bool = True


class ControlRunRequest(RunModel):
    run_id: str = Field(min_length=1)
    command_type: RunCommandType
    idempotency_key: str = Field(min_length=1)
    payload: dict[str, object] = Field(default_factory=dict)


class TaskStartRequest(RunModel):
    task_id: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)


class ProviderAttemptRequest(RunModel):
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: str = Field(min_length=1)


class TaskResultRequest(RunModel):
    task_id: str = Field(min_length=1)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CommandReceipt:
    command_id: str
    run_id: str
    state: str
    reused: bool


@dataclass(frozen=True, slots=True)
class RunLease:
    run_id: str
    holder_id: str
    fencing_token: int
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class RecoverableTask:
    task_id: str
    task_type: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    run_id: str
    task_id: str
    stage: str
    result_sha256: str
    fencing_token: int
    created_at: str


@dataclass(frozen=True, slots=True)
class RunRecoveryState:
    lease: RunLease
    active_task: RecoverableTask | None
    last_checkpoint: RunCheckpoint | None


@dataclass(frozen=True, slots=True)
class TaskResultReceipt:
    disposition: CommitDisposition
    reused: bool


@dataclass(slots=True)  # noqa: MUTABLE_OK
class RunTransitionError(Exception):
    """Keeps Python-managed traceback state mutable during propagation."""

    run_id: str
    source_state: str
    command_type: str

    def __str__(self) -> str:
        return f"run transition rejected: {self.run_id} {self.source_state} -> {self.command_type}"


@dataclass(slots=True)  # noqa: MUTABLE_OK
class RunLeaseError(Exception):
    """Keeps Python-managed traceback state mutable during propagation."""

    run_id: str

    def __str__(self) -> str:
        return f"run lease is no longer current: {self.run_id}"


@dataclass(slots=True)  # noqa: MUTABLE_OK
class ActiveTaskError(Exception):
    """Keeps Python-managed traceback state mutable during propagation."""

    run_id: str

    def __str__(self) -> str:
        return f"run already has an active task: {self.run_id}"


__all__ = (
    "ActiveTaskError",
    "CommitDisposition",
    "CommandReceipt",
    "ControlRunRequest",
    "ProviderAttemptRequest",
    "RecoverableTask",
    "RunCheckpoint",
    "RunLease",
    "RunLeaseError",
    "RunRecoveryState",
    "RunTransitionError",
    "StartRunRequest",
    "StartRunExecutionPayload",
    "TaskResultReceipt",
    "TaskResultRequest",
    "TaskStartRequest",
)
