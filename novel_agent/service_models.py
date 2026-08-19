from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from novel_agent.contracts import RunCommandType, RunStatus
from novel_agent.pathing import ProjectId
from novel_agent.run_coordinator import RunCoordinator
from novel_agent.run_worker import RunStageExecutor


_RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"


class ServiceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class StartNovelRunRequest(ServiceRequest):
    project_id: ProjectId
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    idempotency_key: str = Field(min_length=1, max_length=256)
    #: Blank requirements (empty or whitespace-only) are rejected up front so
    #: a start without a creative request fails fast with a clear validation
    #: error instead of degrading through the analysis pipeline.
    user_requirements: str = Field(min_length=1, max_length=100_000)
    provider: Literal["offline_fake", "legacy_real"] | None = None
    target_chapter_count: int | None = Field(default=None, ge=1, le=100)
    auto_approve: bool = True

    @field_validator("user_requirements")
    @classmethod
    def _requirements_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("user_requirements must not be blank")
        return stripped


class ApproveArtifactRequest(ServiceRequest):
    project_id: ProjectId
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    artifact_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1, max_length=256)


class RegenerateArtifactRequest(ServiceRequest):
    project_id: ProjectId
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    artifact_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1, max_length=256)


class ControlNovelRunRequest(ServiceRequest):
    project_id: ProjectId
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    command_type: RunCommandType
    idempotency_key: str = Field(min_length=1, max_length=256)


class RepairNovelRunRequest(ServiceRequest):
    project_id: ProjectId
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    idempotency_key: str = Field(min_length=1, max_length=256)


class ReviseChapterRequest(ServiceRequest):
    project_id: ProjectId
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    chapter_number: int = Field(ge=1, le=100)
    idempotency_key: str = Field(min_length=1, max_length=256)


class RunUiSnapshot(ServiceRequest):
    project_id: ProjectId
    run_id: str
    status: RunStatus
    active_task_id: str | None
    review_reason: str | None
    review_event_id: str | None = None
    event_count: int
    command_id: str | None
    reused: bool


class RunStageExecutorFactory(Protocol):
    def __call__(self, coordinator: RunCoordinator, request: StartNovelRunRequest) -> RunStageExecutor: ...


__all__ = (
    "ControlNovelRunRequest",
    "RepairNovelRunRequest",
    "RunStageExecutorFactory",
    "RunUiSnapshot",
    "StartNovelRunRequest",
)
