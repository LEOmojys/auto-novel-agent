from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import assert_never

from novel_agent.contracts import ProviderOutcome, ProviderRequest, ProviderResponse, Provenance, ProvenanceKind
from novel_agent.tool_runtime import combine_tokens, current_tool_token
from novel_agent.provider_types import (
    CancellationToken,
    LegacyDeepSeekTransport,
    ProviderAttempt,
    ProviderAuditContext,
    ProviderDependencies,
    ProviderFailureKind,
    ProviderGenerationResult,
    ProviderPolicy,
    ProviderPolicyError,
    ProviderTransport,
    ProviderTransportError,
    ProviderTransportResult,
)


def _generate_with_cancellation(transport: ProviderTransport, request: ProviderRequest, cancellation: object) -> ProviderTransportResult:
    try:
        parameters = inspect.signature(transport.generate).parameters
    except (TypeError, ValueError):
        return transport.generate(request)
    if "cancellation" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return transport.generate(request, cancellation=cancellation)
    return transport.generate(request)


@dataclass(frozen=True, slots=True)
class ProviderClient:
    dependencies: ProviderDependencies
    policy: ProviderPolicy = ProviderPolicy()

    def generate(self, request: ProviderRequest, cancellation: CancellationToken) -> ProviderGenerationResult:
        # The tool scope may carry a run cancellation token and a deadline; the
        # provider transport checks it between stream chunks.
        cancellation = combine_tokens(cancellation, current_tool_token())
        attempts: list[ProviderAttempt] = []
        started_at = self.dependencies.monotonic()
        if cancellation.is_cancelled():
            return self._discarded(request, tuple(attempts))

        for attempt_number in range(1, self.policy.maximum_attempts + 1):
            if cancellation.is_cancelled():
                return self._discarded(request, tuple(attempts))
            if self._deadline_expired(started_at):
                return self._failed(request, tuple(attempts), ProviderFailureKind.TIMEOUT)
            try:
                payload = _generate_with_cancellation(
                    self.dependencies.transport, request, cancellation
                )
            except ProviderTransportError as failure:
                attempts.append(
                    ProviderAttempt(
                        attempt_number=attempt_number,
                        provider_call_id=request.provider_call_id,
                        request_sha256=request.prompt_sha256,
                        adapter_version=self.policy.adapter_version,
                        outcome=ProviderOutcome.FAILED,
                        failure_kind=failure.kind,
                    )
                )
                match failure.kind:
                    case ProviderFailureKind.TRANSIENT:
                        if attempt_number == self.policy.maximum_attempts:
                            return self._failed(request, tuple(attempts), failure.kind)
                        if cancellation.is_cancelled():
                            return self._discarded(request, tuple(attempts))
                        self.dependencies.sleep(self.policy.retry_delay(attempt_number))
                        if cancellation.is_cancelled():
                            return self._discarded(request, tuple(attempts))
                    case ProviderFailureKind.CONFIGURATION | ProviderFailureKind.PERMANENT | ProviderFailureKind.TIMEOUT:
                        return self._failed(request, tuple(attempts), failure.kind)
                    case unreachable:
                        assert_never(unreachable)
                continue

            if cancellation.is_cancelled():
                attempts.append(
                    ProviderAttempt(
                        attempt_number=attempt_number,
                        provider_call_id=request.provider_call_id,
                        request_sha256=request.prompt_sha256,
                        adapter_version=self.policy.adapter_version,
                        outcome=ProviderOutcome.DISCARDED,
                        failure_kind=None,
                    )
                )
                return self._discarded(request, tuple(attempts))
            if self._deadline_expired(started_at):
                attempts.append(
                    ProviderAttempt(
                        attempt_number=attempt_number,
                        provider_call_id=request.provider_call_id,
                        request_sha256=request.prompt_sha256,
                        adapter_version=self.policy.adapter_version,
                        outcome=ProviderOutcome.FAILED,
                        failure_kind=ProviderFailureKind.TIMEOUT,
                    )
                )
                return self._failed(request, tuple(attempts), ProviderFailureKind.TIMEOUT)
            attempts.append(
                ProviderAttempt(
                    attempt_number=attempt_number,
                    provider_call_id=request.provider_call_id,
                    request_sha256=request.prompt_sha256,
                    adapter_version=self.policy.adapter_version,
                    outcome=ProviderOutcome.SUCCEEDED,
                    failure_kind=None,
                )
            )
            return self._succeeded(request, payload, tuple(attempts))

        return self._failed(request, tuple(attempts), ProviderFailureKind.TRANSIENT)

    def _deadline_expired(self, started_at: float) -> bool:
        elapsed_seconds = self.dependencies.monotonic() - started_at
        return elapsed_seconds >= self.policy.logical_deadline_seconds

    def _discarded(
        self,
        request: ProviderRequest,
        attempts: tuple[ProviderAttempt, ...],
    ) -> ProviderGenerationResult:
        return ProviderGenerationResult(
            response=self._empty_response(request, ProviderOutcome.DISCARDED),
            content=None,
            attempts=attempts,
            audit=self._audit_context(request),
            failure_kind=None,
            failure_summary="provider output discarded after cancellation or fence loss",
        )

    def _failed(
        self,
        request: ProviderRequest,
        attempts: tuple[ProviderAttempt, ...],
        failure_kind: ProviderFailureKind,
    ) -> ProviderGenerationResult:
        return ProviderGenerationResult(
            response=self._empty_response(request, ProviderOutcome.FAILED),
            content=None,
            attempts=attempts,
            audit=self._audit_context(request),
            failure_kind=failure_kind,
            failure_summary=_failure_summary(failure_kind),
        )

    def _succeeded(
        self,
        request: ProviderRequest,
        payload: ProviderTransportResult,
        attempts: tuple[ProviderAttempt, ...],
    ) -> ProviderGenerationResult:
        response_hash = hashlib.sha256(payload.content.encode()).hexdigest()
        response = ProviderResponse(
            provider_call_id=request.provider_call_id,
            outcome=ProviderOutcome.SUCCEEDED,
            response_sha256=response_hash,
            prompt_tokens=payload.prompt_tokens,
            completion_tokens=payload.completion_tokens,
            received_at=self.dependencies.now(),
            provenance=Provenance(
                source_kind=ProvenanceKind.PROVIDER_RESPONSE,
                source_id=request.provider_call_id,
                source_sha256=response_hash,
                observed_at=self.dependencies.now(),
            ),
        )
        return ProviderGenerationResult(
            response=response,
            content=payload.content,
            attempts=attempts,
            audit=self._audit_context(request),
            failure_kind=None,
            failure_summary="",
        )

    def _audit_context(self, request: ProviderRequest) -> ProviderAuditContext:
        return ProviderAuditContext(
            request_sha256=request.prompt_sha256,
            model_name=request.model_name,
            temperature=request.temperature,
            adapter_version=self.policy.adapter_version,
            connect_timeout_seconds=self.policy.connect_timeout_seconds,
            read_timeout_seconds=self.policy.read_timeout_seconds,
            logical_deadline_seconds=self.policy.logical_deadline_seconds,
            maximum_attempts=self.policy.maximum_attempts,
        )

    def _empty_response(self, request: ProviderRequest, outcome: ProviderOutcome) -> ProviderResponse:
        return ProviderResponse(
            provider_call_id=request.provider_call_id,
            outcome=outcome,
            response_sha256=None,
            prompt_tokens=0,
            completion_tokens=0,
            received_at=self.dependencies.now(),
            provenance=Provenance(
                source_kind=ProvenanceKind.PROVIDER_RESPONSE,
                source_id=request.provider_call_id,
                source_sha256=request.prompt_sha256,
                observed_at=self.dependencies.now(),
            ),
        )


def _failure_summary(kind: ProviderFailureKind) -> str:
    match kind:
        case ProviderFailureKind.CONFIGURATION:
            return "provider configuration failed"
        case ProviderFailureKind.PERMANENT:
            return "provider rejected a non-retryable request"
        case ProviderFailureKind.TIMEOUT:
            return "provider operation exceeded its deadline"
        case ProviderFailureKind.TRANSIENT:
            return "provider retries were exhausted"
        case unreachable:
            assert_never(unreachable)


__all__ = (
    "CancellationToken",
    "LegacyDeepSeekTransport",
    "ProviderAttempt",
    "ProviderAuditContext",
    "ProviderClient",
    "ProviderDependencies",
    "ProviderFailureKind",
    "ProviderGenerationResult",
    "ProviderPolicy",
    "ProviderPolicyError",
    "ProviderTransport",
    "ProviderTransportError",
    "ProviderTransportResult",
)
