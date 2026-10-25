from __future__ import annotations

import hashlib
import sqlite3
from threading import RLock

from pydantic import ValidationError

from novel_agent.contracts import ToolOutcome
from novel_agent.tool_call_persistence import (
    decode_persisted_result,
    persist_result,
    record_result,
    reserve_result,
)
from novel_agent.tool_runtime import ToolCancelledError
from novel_agent.tool_models import (
    AccessLevel,
    AgentRole,
    ApprovalPolicy,
    ArtifactPolicy,
    InvocationDecision,
    InvocationTrace,
    RetryPolicy,
    ToolCallResult,
    ToolHandler,
    ToolInvocationRequest,
    ToolRegistryError,
    ToolSpec,
)


class ToolRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[ToolSpec, ToolHandler]] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.tool_ref in self._entries:
            raise ToolRegistryError(detail=f"duplicate tool declaration: {spec.tool_ref}")
        self._entries[spec.tool_ref] = (spec, handler)

    def resolve(self, tool_ref: str) -> tuple[ToolSpec, ToolHandler] | None:
        return self._entries.get(tool_ref)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(entry[0] for entry in self._entries.values())


class ToolGateway:
    def __init__(self, registry: ToolRegistry, connection: sqlite3.Connection | None = None) -> None:
        self._registry = registry
        self._connection = connection
        self._completed: dict[tuple[str, str], tuple[str, ToolCallResult]] = {}
        self._traces: tuple[InvocationTrace, ...] = ()
        self._local_lock = RLock()

    @property
    def traces(self) -> tuple[InvocationTrace, ...]:
        return self._traces

    def invoke(self, request: ToolInvocationRequest) -> ToolCallResult:
        if self._connection is None:
            with self._local_lock:
                return self._invoke(request)
        return self._invoke(request)

    def _invoke(self, request: ToolInvocationRequest) -> ToolCallResult:
        input_hash = hashlib.sha256(request.input_data.model_dump_json().encode()).hexdigest()
        resolved = self._registry.resolve(request.tool_ref)
        if resolved is None:
            return self._denied(request, input_hash, "UNKNOWN_TOOL")
        spec, handler = resolved
        if request.shadow:
            return self._denied(request, input_hash, "SHADOW_NO_SIDE_EFFECT")
        reason = self._authorization_reason(spec, request)
        if reason:
            return self._denied(request, input_hash, reason)
        ledger_key = (request.run_id, request.idempotency_key)
        prior = self._completed.get(ledger_key) if self._connection is None else None
        if spec.idempotent and self._connection is not None:
            reservation = reserve_result(
                self._connection,
                request,
                spec,
                input_hash,
                max_attempts=spec.retry_policy.max_attempts,
                retry_after_seconds=spec.retry_policy.retry_after_seconds,
            )
            if reservation.tool_id != spec.tool_ref or reservation.request_sha256 != input_hash:
                return self._denied(request, input_hash, "IDEMPOTENCY_CONFLICT")
            if reservation.status == "existing":
                if reservation.state == "pending":
                    return self._failed(request, input_hash, "IDEMPOTENCY_IN_PROGRESS")
                try:
                    persisted = decode_persisted_result(reservation.result_payload_json, spec)
                except (KeyError, TypeError, ValueError, ValidationError):
                    return self._failed(request, input_hash, "PERSISTED_RESULT_INVALID")
                prior = (input_hash, persisted)
        if prior is not None:
            prior_hash, prior_result = prior
            if prior_hash != input_hash or prior_result.trace.tool_ref != request.tool_ref:
                return self._denied(request, input_hash, "IDEMPOTENCY_CONFLICT")
            trace = prior_result.trace.model_copy(update={"invocation_id": request.invocation_id, "reused": True})
            return self._record(prior_result.model_copy(update={"trace": trace}))
        try:
            parsed_input = spec.input_model.model_validate(request.input_data.model_dump())
            raw_output = handler(parsed_input)
            output = spec.output_model.model_validate(raw_output.model_dump())
        except ToolCancelledError:
            # Cancellation/timeout is control flow, not a domain failure: let the
            # loop mark the tool as aborted instead of surfacing HANDLER_FAILED.
            raise
        except ValidationError:
            return self._finish(request, spec, input_hash, ledger_key, self._result(
                request,
                input_hash,
                InvocationDecision(outcome=ToolOutcome.FAILED, reason="SCHEMA_INVALID"),
            ))
        except Exception:  # noqa: BROAD_EXCEPT_OK - handler is an untyped external boundary.
            return self._finish(request, spec, input_hash, ledger_key, self._result(
                request,
                input_hash,
                InvocationDecision(outcome=ToolOutcome.FAILED, reason="HANDLER_FAILED"),
            ))
        output_hash = hashlib.sha256(output.model_dump_json().encode()).hexdigest()
        decision = InvocationDecision(outcome=ToolOutcome.SUCCEEDED, reason="", data=output, output_hash=output_hash)
        result = self._result(request, input_hash, decision)
        return self._finish(request, spec, input_hash, ledger_key, result)

    def _finish(
        self,
        request: ToolInvocationRequest,
        spec: ToolSpec,
        input_hash: str,
        ledger_key: tuple[str, str],
        result: ToolCallResult,
    ) -> ToolCallResult:
        if spec.idempotent:
            self._completed[ledger_key] = (input_hash, result)
            if self._connection is not None:
                persist_result(self._connection, request, spec, result, input_hash)
        elif spec.audit_calls and self._connection is not None:
            record_result(self._connection, request, spec, result, input_hash)
        return self._record(result)

    @staticmethod
    def _authorization_reason(spec: ToolSpec, request: ToolInvocationRequest) -> str:
        if request.actor_role not in spec.allowed_roles:
            return "ROLE_DENIED"
        if request.run_status not in spec.allowed_run_states:
            return "STATE_DENIED"
        if request.skill_ref not in spec.allowed_skill_ids:
            return "SKILL_TOOL_DENIED"
        if spec.approval_policy is not ApprovalPolicy.NONE and not request.approval_granted:
            return "APPROVAL_REQUIRED"
        return ""

    def _denied(self, request: ToolInvocationRequest, input_hash: str, reason: str) -> ToolCallResult:
        decision = InvocationDecision(
            outcome=ToolOutcome.DENIED,
            reason=reason,
            shadow=reason == "SHADOW_NO_SIDE_EFFECT",
        )
        return self._record(self._result(request, input_hash, decision))

    def _failed(self, request: ToolInvocationRequest, input_hash: str, reason: str) -> ToolCallResult:
        decision = InvocationDecision(outcome=ToolOutcome.FAILED, reason=reason)
        return self._record(self._result(request, input_hash, decision))

    @staticmethod
    def _result(
        request: ToolInvocationRequest,
        input_hash: str,
        decision: InvocationDecision,
    ) -> ToolCallResult:
        trace = InvocationTrace(
            invocation_id=request.invocation_id,
            tool_ref=request.tool_ref,
            skill_ref=request.skill_ref,
            idempotency_key=request.idempotency_key,
            input_sha256=input_hash,
            output_sha256=decision.output_hash,
            outcome=decision.outcome,
            reason=decision.reason,
            reused=False,
            shadow=decision.shadow,
        )
        return ToolCallResult(outcome=decision.outcome, data=decision.data, reason=decision.reason, trace=trace)

    def _record(self, result: ToolCallResult) -> ToolCallResult:
        with self._local_lock:
            self._traces = (*self._traces, result.trace)
        return result


__all__ = (
    "AccessLevel",
    "AgentRole",
    "ApprovalPolicy",
    "ArtifactPolicy",
    "InvocationTrace",
    "InvocationDecision",
    "RetryPolicy",
    "ToolCallResult",
    "ToolGateway",
    "ToolInvocationRequest",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolSpec",
)
