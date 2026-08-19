from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Final, Self

from pydantic import Field, field_validator, model_validator

from novel_agent.contracts.common import (
    ArtifactStatus,
    ContractErrorCode,
    ContractModel,
    ProviderOutcome,
    Provenance,
    QualityOutcome,
    RunCommandType,
    RunStatus,
    TaskStatus,
    ToolOutcome,
    contract_error,
    validate_sha256,
)
from novel_agent.pathing import ProjectId


_ALLOWED_COMMAND_STATES: Final[frozenset[tuple[RunCommandType, RunStatus]]] = frozenset(
    {
        (RunCommandType.START, RunStatus.QUEUED),
        (RunCommandType.PAUSE, RunStatus.RUNNING),
        (RunCommandType.REPAIR, RunStatus.WAITING_REVIEW),
        (RunCommandType.RESUME, RunStatus.PAUSED),
        (RunCommandType.CANCEL, RunStatus.QUEUED),
        (RunCommandType.CANCEL, RunStatus.RUNNING),
        (RunCommandType.CANCEL, RunStatus.PAUSED),
        (RunCommandType.CANCEL, RunStatus.WAITING_REVIEW),
    }
)


def is_run_command_allowed(command_type: RunCommandType, expected_status: RunStatus) -> bool:
    return (command_type, expected_status) in _ALLOWED_COMMAND_STATES


class RunCommand(ContractModel):
    command_id: str
    run_id: str
    command_type: RunCommandType
    expected_status: RunStatus
    issued_at: datetime
    provenance: Provenance

    @model_validator(mode="after")
    def validate_transition(self) -> RunCommand:
        if not is_run_command_allowed(self.command_type, self.expected_status):
            raise contract_error(
                ContractErrorCode.INVALID_STATE_TRANSITION,
                "command type cannot be issued from the expected run status",
            )
        return self


class RunSnapshot(ContractModel):
    run_id: str
    project_id: ProjectId
    status: RunStatus
    plan_id: str | None
    active_task_id: str | None
    updated_at: datetime
    provenance: Provenance


class TaskRecord(ContractModel):
    task_id: str
    run_id: str
    stage: str
    status: TaskStatus
    attempt: int
    created_at: datetime
    updated_at: datetime
    provenance: Provenance

    @field_validator("attempt")
    @classmethod
    def validate_attempt(cls, value: int) -> int:
        if value < 1:
            raise contract_error(ContractErrorCode.INVALID_CONTEXT_BUDGET, "attempt must be positive")
        return value


class ProviderRequest(ContractModel):
    provider_call_id: str
    run_id: str
    task_id: str
    model_name: str
    prompt: str = Field(repr=False)
    prompt_sha256: str
    temperature: float
    created_at: datetime
    provenance: Provenance

    @field_validator("prompt_sha256")
    @classmethod
    def validate_prompt_sha256(cls, value: str) -> str:
        return validate_sha256(value)

    @model_validator(mode="after")
    def validate_prompt_digest(self) -> Self:
        actual_digest = hashlib.sha256(self.prompt.encode()).hexdigest()
        if actual_digest != self.prompt_sha256:
            raise contract_error(
                ContractErrorCode.INVALID_REQUEST_DIGEST,
                "prompt must match its declared SHA-256 digest",
            )
        return self


class ProviderResponse(ContractModel):
    provider_call_id: str
    outcome: ProviderOutcome
    response_sha256: str | None
    prompt_tokens: int
    completion_tokens: int
    received_at: datetime
    provenance: Provenance

    @field_validator("response_sha256")
    @classmethod
    def validate_response_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return validate_sha256(value)


class ToolInvocation(ContractModel):
    invocation_id: str
    run_id: str
    task_id: str
    tool_name: str
    idempotency_key: str
    input_sha256: str
    requested_at: datetime
    provenance: Provenance

    @field_validator("input_sha256")
    @classmethod
    def validate_input_sha256(cls, value: str) -> str:
        return validate_sha256(value)


class ToolResult(ContractModel):
    invocation_id: str
    outcome: ToolOutcome
    output_sha256: str | None
    completed_at: datetime
    provenance: Provenance

    @field_validator("output_sha256")
    @classmethod
    def validate_output_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return validate_sha256(value)


class ArtifactRevision(ContractModel):
    artifact_id: str
    revision: int
    project_id: ProjectId
    run_id: str | None
    kind: str
    status: ArtifactStatus
    content_sha256: str
    created_at: datetime
    provenance: Provenance

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: int) -> int:
        if value < 1:
            raise contract_error(ContractErrorCode.INVALID_CONTEXT_BUDGET, "revision must be positive")
        return value

    @field_validator("content_sha256")
    @classmethod
    def validate_content_sha256(cls, value: str) -> str:
        return validate_sha256(value)


class QualityPolicy(ContractModel):
    policy_id: str
    minimum_score: float
    maximum_repairs: int
    hard_rules: tuple[str, ...]
    created_at: datetime
    provenance: Provenance

    @field_validator("minimum_score")
    @classmethod
    def validate_minimum_score(cls, value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise contract_error(ContractErrorCode.INVALID_CONTEXT_BUDGET, "minimum score must be between zero and one hundred")
        return value

    @field_validator("maximum_repairs")
    @classmethod
    def validate_maximum_repairs(cls, value: int) -> int:
        if value < 0:
            raise contract_error(ContractErrorCode.INVALID_CONTEXT_BUDGET, "maximum repairs cannot be negative")
        return value


class QualityDecision(ContractModel):
    decision_id: str
    artifact_id: str
    policy_id: str
    outcome: QualityOutcome
    score: float | None
    reasons: tuple[str, ...]
    created_at: datetime
    provenance: Provenance
