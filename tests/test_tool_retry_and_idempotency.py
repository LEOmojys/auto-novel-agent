"""Retry/idempotency semantics of the governed tool gateway."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ConfigDict

from novel_agent.agent_loop import AgentLoop, AgentTool, LoopConfig, ToolResult
from novel_agent.contracts import RunStatus, ToolOutcome
from novel_agent.provider_types import ProviderTransportResult
from novel_agent.tool_runtime import ToolTimedOut, check_tool_active, tool_scope
from novel_agent.storage import connect_project_database, migrate
from novel_agent.tool_registry import (
    AccessLevel,
    AgentRole,
    ApprovalPolicy,
    ArtifactPolicy,
    RetryPolicy,
    ToolGateway,
    ToolInvocationRequest,
    ToolRegistry,
    ToolSpec,
)


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: str


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ok: bool = True


def _spec(*, retry: RetryPolicy | None = None, audit_calls: bool = False) -> ToolSpec:
    return ToolSpec(
        tool_id="test.echo",
        version="1.0.0",
        input_model=_In,
        output_model=_Out,
        allowed_roles=frozenset({AgentRole.CHAPTER_EXECUTOR}),
        allowed_skill_ids=frozenset({"loop@1.0.0"}),
        allowed_run_states=frozenset({RunStatus.RUNNING}),
        access_level=AccessLevel.READ,
        timeout_seconds=5.0,
        retry_policy=retry or RetryPolicy(max_attempts=1),
        idempotent=not audit_calls,
        audit_calls=audit_calls,
        approval_policy=ApprovalPolicy.NONE,
        artifact_policy=ArtifactPolicy.NONE,
    )


def _request(key: str = "key-1") -> ToolInvocationRequest:
    return ToolInvocationRequest(
        invocation_id=f"invoke-{key}",
        run_id="run-1",
        tool_ref="test.echo@1.0.0",
        idempotency_key=key,
        actor_role=AgentRole.CHAPTER_EXECUTOR,
        skill_ref="loop@1.0.0",
        run_status=RunStatus.RUNNING,
        input_data=_In(value="x"),
    )


def _project(tmp_path):
    return connect_project_database(tmp_path / "project")


def test_failed_result_can_retry_after_ttl(tmp_path) -> None:
    calls: list[int] = []

    def handler(payload: _In) -> _Out:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return _Out()

    with _project(tmp_path) as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) "
            "VALUES ('run-1', 'project-1', 'running', '2030-01-01', '2030-01-01')"
        )
        registry = ToolRegistry()
        registry.register(
            _spec(retry=RetryPolicy(max_attempts=2, retry_after_seconds=0.0)),
            handler,
        )
        gateway = ToolGateway(registry, connection)
        first = gateway.invoke(_request())
        second = gateway.invoke(_request())

        assert first.outcome is ToolOutcome.FAILED
        assert second.outcome is ToolOutcome.SUCCEEDED
        assert len(calls) == 2
        row = connection.execute(
            "SELECT state, attempts FROM tool_calls WHERE run_id='run-1' AND idempotency_key='key-1'"
        ).fetchone()
        assert row == ("succeeded", 2)


def test_stale_pending_reservation_is_reclaimed(tmp_path) -> None:
    calls: list[int] = []

    def handler(payload: _In) -> _Out:
        calls.append(1)
        return _Out()

    with _project(tmp_path) as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) "
            "VALUES ('run-1', 'project-1', 'running', '2030-01-01', '2030-01-01')"
        )
        connection.execute(
            "INSERT INTO tool_calls(call_id, run_id, tool_id, idempotency_key, request_sha256, "
            "state, attempts, created_at) VALUES ('old', 'run-1', 'test.echo@1.0.0', 'key-1', "
            "'hash', 'pending', 1, '2000-01-01 00:00:00')"
        )
        registry = ToolRegistry()
        registry.register(_spec(), handler)
        result = ToolGateway(registry, connection).invoke(_request())

        assert result.outcome is ToolOutcome.SUCCEEDED
        assert calls == [1]


def test_audit_only_tool_records_every_call(tmp_path) -> None:
    calls: list[int] = []

    def handler(payload: _In) -> _Out:
        calls.append(1)
        return _Out()

    with _project(tmp_path) as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) "
            "VALUES ('run-1', 'project-1', 'running', '2030-01-01', '2030-01-01')"
        )
        registry = ToolRegistry()
        registry.register(_spec(audit_calls=True), handler)
        gateway = ToolGateway(registry, connection)
        gateway.invoke(_request("key-a"))
        gateway.invoke(_request("key-b"))

        assert len(calls) == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE run_id='run-1'"
        ).fetchone() == (2,)


class _SeqTransport:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, request, cancellation=None) -> ProviderTransportResult:
        content = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return ProviderTransportResult(content=content, model_name="x", prompt_tokens=1, completion_tokens=1)


def test_agent_loop_retries_transient_tool_failure() -> None:
    calls: list[int] = []

    def flaky(args: dict) -> ToolResult:
        calls.append(1)
        if len(calls) == 1:
            return ToolResult(ok=False, error="transient", retryable=True)
        return ToolResult(ok=True, summary="ok")

    transport = _SeqTransport(['{"plan": [{"tool": "a", "args": {}}]}', '{"done": true}'])
    loop = AgentLoop(
        tools={
            "a": AgentTool(
                name="a",
                description="a",
                input_schema={},
                execute=flaky,
                max_attempts=2,
                retry_delay_seconds=0.0,
            )
        },
        transport=transport,
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=4),
    )
    outcome, _history = loop.run(system_prompt="sys", task="任务")

    assert outcome == "done"
    assert len(calls) == 2


# -- deadline / cancellation ---------------------------------------------------


def test_tool_scope_enforces_deadline_at_boundaries() -> None:
    import time

    with tool_scope(None, 0.01):
        time.sleep(0.02)
        with pytest.raises(ToolTimedOut):
            check_tool_active()


def test_agent_loop_times_out_a_read_only_tool() -> None:
    import time

    def slow(args: dict) -> ToolResult:
        time.sleep(0.05)
        return ToolResult(ok=True, summary="done")

    transport = _SeqTransport(['{"plan": [{"tool": "slow", "args": {}}]}', '{"done": true}'])
    loop = AgentLoop(
        tools={
            "slow": AgentTool(
                name="slow",
                description="slow",
                input_schema={},
                execute=slow,
                supports_parallel=True,
                timeout_seconds=0.01,
            )
        },
        transport=transport,
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=4, doom_loop_threshold=1_000),
    )
    outcome, history = loop.run(system_prompt="sys", task="任务")

    assert outcome == "done"
    assert any("超时" in message.content for message in history if message.role == "tool")
