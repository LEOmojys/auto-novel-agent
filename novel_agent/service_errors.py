from __future__ import annotations

from dataclasses import dataclass

from novel_agent.pathing import ProjectId


@dataclass(slots=True)  # noqa: MUTABLE_OK
class StageTimeoutConfigurationError(ValueError):
    stage_timeout_seconds: float

    def __str__(self) -> str:
        return "stage timeout must be positive"


@dataclass(slots=True)  # noqa: MUTABLE_OK
class RunNotFoundError(Exception):
    project_id: ProjectId
    run_id: str

    def __str__(self) -> str:
        return f"run does not belong to project: {self.project_id.root}/{self.run_id}"


@dataclass(slots=True)  # noqa: MUTABLE_OK
class RebaseWorkerUnavailableError(Exception):
    run_id: str

    def __str__(self) -> str:
        return f"rebase confirmation cannot claim run: {self.run_id}"


@dataclass(slots=True)  # noqa: MUTABLE_OK
class RunRepairUnavailableError(Exception):
    run_id: str
    review_reason: str | None

    def __str__(self) -> str:
        return f"run review requires a different action: {self.run_id} ({self.review_reason})"


__all__ = (
    "RebaseWorkerUnavailableError",
    "RunNotFoundError",
    "RunRepairUnavailableError",
    "StageTimeoutConfigurationError",
)
