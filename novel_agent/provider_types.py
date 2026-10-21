from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, unique
from threading import Event
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from novel_agent.contracts import ProviderOutcome, ProviderRequest, ProviderResponse


@unique
class ProviderFailureKind(StrEnum):
    CONFIGURATION = "configuration"
    PERMANENT = "permanent"
    TIMEOUT = "timeout"
    TRANSIENT = "transient"


@dataclass(frozen=True, slots=True)
class ProviderTransportError(Exception):
    kind: ProviderFailureKind
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class ProviderPolicyError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


class ProviderTransportResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    content: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int

    @field_validator("prompt_tokens", "completion_tokens")
    @classmethod
    def validate_token_count(cls, value: int) -> int:
        if value < 0:
            raise ProviderTransportError(
                kind=ProviderFailureKind.PERMANENT,
                detail="provider token counts must be non-negative",
            )
        return value


class ProviderTransport(Protocol):
    def generate(
        self,
        request: ProviderRequest,
        cancellation: object | None = None,
    ) -> ProviderTransportResult:
        """Invoke the model.

        ``cancellation`` is an optional token exposing ``is_cancelled()``; a
        transport that can abort an in-flight request (e.g. streaming HTTP)
        should honor it so a user cancel takes effect promptly.
        """
        ...


def _call_accepts_cancellation(call) -> bool:
    """Whether a legacy callable accepts an optional cancellation keyword."""
    try:
        parameters = tuple(inspect.signature(call).parameters.values())
    except (TypeError, ValueError):
        return False
    for parameter in parameters:
        if parameter.name == "cancellation":
            return True
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


class LegacyDeepSeekCallable(Protocol):
    def __call__(
        self,
        text: str,
        *,
        max_retry: int,
        model_name: str,
        return_tokens: Literal[True],
        caller: str,
        cancellation: object | None = None,
    ) -> tuple[str, Mapping[str, int | str]]:
        ...


class _LegacyUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model: str
    prompt_tokens: int
    completion_tokens: int
    # ``access_deepseek`` appends total_tokens to its usage payload; the
    # official DeepSeek usage object also carries it.  Optional so both
    # shapes validate.
    total_tokens: int | None = None

    @field_validator("prompt_tokens", "completion_tokens")
    @classmethod
    def validate_token_count(cls, value: int) -> int:
        if value < 0:
            raise ProviderTransportError(
                kind=ProviderFailureKind.PERMANENT,
                detail="legacy provider token counts must be non-negative",
            )
        return value


@dataclass(frozen=True, slots=True)
class LegacyDeepSeekTransport:
    call: LegacyDeepSeekCallable

    def generate(
        self,
        request: ProviderRequest,
        cancellation: object | None = None,
    ) -> ProviderTransportResult:
        try:
            if _call_accepts_cancellation(self.call):
                content, usage = self.call(
                    request.prompt,
                    max_retry=1,
                    model_name=request.model_name,
                    return_tokens=True,
                    caller="novel_agent",
                    cancellation=cancellation,
                )
            else:
                content, usage = self.call(
                    request.prompt,
                    max_retry=1,
                    model_name=request.model_name,
                    return_tokens=True,
                    caller="novel_agent",
                )
        except ProviderTransportError:
            raise
        except Exception as error:  # noqa: BROAD_EXCEPT_OK - legacy callback has no typed error contract.
            # Surface the underlying provider message (status codes like 401/402/429,
            # key errors, network failures) so the console can explain the failure.
            raise ProviderTransportError(
                kind=ProviderFailureKind.TRANSIENT,
                detail=f"legacy provider invocation failed: {error}",
            ) from None
        try:
            parsed_usage = _LegacyUsage.model_validate(usage)
        except ValidationError as error:
            raise ProviderTransportError(
                kind=ProviderFailureKind.PERMANENT,
                detail="legacy provider returned an invalid usage payload",
            ) from error
        return ProviderTransportResult(
            content=content,
            model_name=parsed_usage.model,
            prompt_tokens=parsed_usage.prompt_tokens,
            completion_tokens=parsed_usage.completion_tokens,
        )


@dataclass(frozen=True, slots=True)
class ProviderPolicy:
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 180.0
    logical_deadline_seconds: float = 600.0
    maximum_attempts: int = 3
    retry_delays_seconds: tuple[float, ...] = (1.0, 2.0)
    adapter_version: str = "legacy-deepseek-v1"

    def __post_init__(self) -> None:
        if self.connect_timeout_seconds <= 0.0 or self.read_timeout_seconds <= 0.0:
            raise ProviderPolicyError("provider connect and read timeouts must be positive")
        if self.logical_deadline_seconds <= 0.0:
            raise ProviderPolicyError("provider logical deadline must be positive")
        if self.maximum_attempts < 1:
            raise ProviderPolicyError("provider must permit at least one attempt")
        if len(self.retry_delays_seconds) < self.maximum_attempts - 1:
            raise ProviderPolicyError("provider retry policy lacks required retry delays")

    def retry_delay(self, attempt_number: int) -> float:
        return self.retry_delays_seconds[attempt_number - 1]


@dataclass(frozen=True, slots=True)
class ProviderDependencies:
    transport: ProviderTransport
    now: Callable[[], datetime]
    sleep: Callable[[float], None]
    monotonic: Callable[[], float] = field(default=time.monotonic)


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


class ProviderAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempt_number: int = Field(ge=1)
    provider_call_id: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_version: str
    outcome: ProviderOutcome
    failure_kind: ProviderFailureKind | None


class ProviderAuditContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_name: str
    temperature: float
    adapter_version: str
    connect_timeout_seconds: float
    read_timeout_seconds: float
    logical_deadline_seconds: float
    maximum_attempts: int


class ProviderGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    response: ProviderResponse
    content: str | None
    attempts: tuple[ProviderAttempt, ...]
    audit: ProviderAuditContext
    failure_kind: ProviderFailureKind | None
    failure_summary: str


__all__ = (
    "CancellationToken",
    "LegacyDeepSeekTransport",
    "ProviderAttempt",
    "ProviderAuditContext",
    "ProviderDependencies",
    "ProviderFailureKind",
    "ProviderGenerationResult",
    "ProviderPolicy",
    "ProviderPolicyError",
    "ProviderTransport",
    "ProviderTransportError",
    "ProviderTransportResult",
)
