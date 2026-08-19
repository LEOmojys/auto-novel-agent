from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from pydantic import BaseModel, ConfigDict

from novel_agent.contracts import RunStatus, ToolOutcome
from novel_agent.storage import connect_project_database, migrate
from novel_agent.tool_call_persistence import PersistedResultDecodeError, decode_persisted_result
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


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str


class EchoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    length: int


def _spec() -> ToolSpec:
    return ToolSpec(
        tool_id="text.echo",
        version="1.0.0",
        input_model=EchoInput,
        output_model=EchoOutput,
        allowed_roles=frozenset({AgentRole.CHAPTER_EXECUTOR}),
        allowed_skill_ids=frozenset({"chapter-scene@1.0.0"}),
        allowed_run_states=frozenset({RunStatus.RUNNING}),
        access_level=AccessLevel.READ,
        timeout_seconds=5.0,
        retry_policy=RetryPolicy(max_attempts=1),
        idempotent=True,
        approval_policy=ApprovalPolicy.NONE,
        artifact_policy=ArtifactPolicy.NONE,
    )


def _request(*, role: AgentRole = AgentRole.CHAPTER_EXECUTOR, key: str = "echo-1") -> ToolInvocationRequest:
    return ToolInvocationRequest(
        invocation_id="invoke-1",
        run_id="run-1",
        task_id="task-1",
        tool_ref="text.echo@1.0.0",
        idempotency_key=key,
        actor_role=role,
        skill_ref="chapter-scene@1.0.0",
        run_status=RunStatus.RUNNING,
        input_data=EchoInput(text="opaque"),
    )


def test_gateway_reuses_idempotent_result() -> None:
    # Given: an authorized tool and a handler whose calls are observable.
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(_spec(), lambda payload: calls.append(payload.text) or EchoOutput(length=len(payload.text)))
    gateway = ToolGateway(registry)

    # When: the same idempotency key and input are invoked twice.
    first = gateway.invoke(_request())
    second = gateway.invoke(_request())

    # Then: one handler effect produces one immutable trace and a reused result.
    assert first.outcome is ToolOutcome.SUCCEEDED
    assert second.trace.reused is True
    assert calls == ["opaque"]
    assert gateway.traces == (first.trace, second.trace)


def test_gateway_reuses_idempotent_result_after_gateway_recreation(tmp_path: Path) -> None:
    # Given: a migrated SQLite ledger and a side-effectful idempotent tool.
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(_spec(), lambda payload: calls.append(payload.text) or EchoOutput(length=len(payload.text)))
    with connect_project_database(tmp_path / "project") as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("run-1", "project-1", "running", "2030-01-01", "2030-01-01"),
        )
        connection.execute(
            "INSERT INTO tasks(task_id, run_id, task_type, state, created_at) VALUES (?, ?, ?, ?, ?)",
            ("task-1", "run-1", "chapter", "running", "2030-01-01"),
        )
        first = ToolGateway(registry, connection=connection).invoke(_request())

        # When: a fresh gateway instance handles the same request after recreation.
        second = ToolGateway(registry, connection=connection).invoke(
            _request().model_copy(update={"invocation_id": "invoke-2"}),
        )

    # Then: the durable result is reused and the handler effect occurs once.
    assert first.outcome is ToolOutcome.SUCCEEDED
    assert second.trace.reused is True
    assert calls == ["opaque"]


def test_persisted_idempotency_conflict_is_denied_before_handler(tmp_path: Path) -> None:
    # Given: a persisted successful call for one input.
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(_spec(), lambda payload: calls.append(payload.text) or EchoOutput(length=len(payload.text)))
    with connect_project_database(tmp_path / "project") as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("run-1", "project-1", "running", "2030-01-01", "2030-01-01"),
        )
        connection.execute(
            "INSERT INTO tasks(task_id, run_id, task_type, state, created_at) VALUES (?, ?, ?, ?, ?)",
            ("task-1", "run-1", "chapter", "running", "2030-01-01"),
        )
        ToolGateway(registry, connection=connection).invoke(_request())

        # When: a recreated gateway changes the request under the same key.
        result = ToolGateway(registry, connection=connection).invoke(
            _request().model_copy(update={"input_data": EchoInput(text="changed")}),
        )

    # Then: the conflicting request is denied and cannot invoke the handler.
    assert result.reason == "IDEMPOTENCY_CONFLICT"
    assert calls == ["opaque"]


def test_malformed_persisted_result_fails_without_handler_execution(tmp_path: Path) -> None:
    # Given: a durable row whose result payload is malformed.
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(_spec(), lambda payload: calls.append(payload.text) or EchoOutput(length=6))
    with connect_project_database(tmp_path / "project") as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("run-1", "project-1", "running", "2030-01-01", "2030-01-01"),
        )
        connection.execute(
            "INSERT INTO tool_calls(call_id, run_id, tool_id, idempotency_key, request_sha256, "
            "state, result_payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "call-1",
                "run-1",
                "text.echo@1.0.0",
                "echo-1",
                hashlib.sha256(_request().input_data.model_dump_json().encode()).hexdigest(),
                "succeeded",
                "{",
                "2030-01-01",
            ),
        )

        # When: a recreated gateway reads the malformed row.
        result = ToolGateway(registry, connection=connection).invoke(_request())

    # Then: corruption is a typed failure and never reaches the handler.
    assert result.reason == "PERSISTED_RESULT_INVALID"
    assert calls == []


@pytest.mark.parametrize(
    ("payload", "detail"),
    (
        (None, "missing persisted result"),
        ("{", "invalid JSON"),
        ("[]", "persisted result must be an object"),
    ),
)
def test_decode_persisted_result_raises_typed_error_for_malformed_payload(
    payload: str | None,
    detail: str,
) -> None:
    # Given: a persisted tool result that cannot be decoded into the ledger contract.
    # When: the persistence boundary parses it.
    with pytest.raises(PersistedResultDecodeError, match=detail):
        decode_persisted_result(payload, _spec())

    # Then: corruption is reported through the typed decode error.


def test_gateway_returns_explicit_role_and_state_denials() -> None:
    # Given: a tool restricted to ChapterExecutor while a reviewer requests it.
    registry = ToolRegistry()
    registry.register(_spec(), lambda payload: EchoOutput(length=len(payload.text)))
    gateway = ToolGateway(registry)

    # When: the role is outside the declaration.
    result = gateway.invoke(_request(role=AgentRole.QUALITY_REVIEWER))

    # Then: no handler result is fabricated and the denial is typed.
    assert result.outcome is ToolOutcome.DENIED
    assert result.reason == "ROLE_DENIED"


def test_gateway_returns_explicit_state_denial() -> None:
    # Given: an authorized role invoking a tool outside its declared run state.
    registry = ToolRegistry()
    registry.register(_spec(), lambda payload: EchoOutput(length=len(payload.text)))
    request = _request().model_copy(update={"run_status": RunStatus.PAUSED})

    # When: the gateway evaluates the invocation state.
    result = ToolGateway(registry).invoke(request)

    # Then: execution is denied with a stable state reason.
    assert result.reason == "STATE_DENIED"


def test_gateway_denies_undeclared_tool_for_skill() -> None:
    # Given: an invocation naming a Skill that did not declare the tool.
    registry = ToolRegistry()
    registry.register(_spec(), lambda payload: EchoOutput(length=len(payload.text)))
    request = _request().model_copy(update={"skill_ref": "quality-review@1.0.0"})

    # When: the gateway checks the server-side declaration.
    result = ToolGateway(registry).invoke(request)

    # Then: capability escalation is denied before handler execution.
    assert result.reason == "SKILL_TOOL_DENIED"


def test_gateway_shadow_invocation_has_zero_handler_effect() -> None:
    # Given: an authorized request marked for shadow evaluation.
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(_spec(), lambda payload: calls.append(payload.text) or EchoOutput(length=len(payload.text)))
    request = _request().model_copy(update={"shadow": True})

    # When: the gateway evaluates the shadow invocation.
    result = ToolGateway(registry).invoke(request)

    # Then: a trace is emitted without calling the effectful handler.
    assert result.reason == "SHADOW_NO_SIDE_EFFECT"
    assert result.trace.shadow is True
    assert calls == []


def test_gateway_redacts_untyped_handler_error() -> None:
    # Given: an authorized handler whose untyped exception contains a sensitive sentinel.
    secret = "tool-handler-secret-sentinel"

    def failing_handler(_payload: BaseModel) -> BaseModel:
        raise RuntimeError(secret)

    registry = ToolRegistry()
    registry.register(_spec(), failing_handler)

    # When: the gateway invokes the external handler boundary.
    result = ToolGateway(registry).invoke(_request())

    # Then: the failure is typed and none of the external error text reaches the trace.
    assert result.outcome is ToolOutcome.FAILED
    assert result.reason == "HANDLER_FAILED"
    assert secret not in repr(result)


def test_idempotency_reuse_rechecks_authorization() -> None:
    # Given: an authorized result already exists for an idempotency key.
    registry = ToolRegistry()
    registry.register(_spec(), lambda payload: EchoOutput(length=len(payload.text)))
    gateway = ToolGateway(registry)
    gateway.invoke(_request())

    # When: an unauthorized role attempts to reuse that key.
    result = gateway.invoke(_request(role=AgentRole.QUALITY_REVIEWER))

    # Then: authorization denial takes precedence over cached output.
    assert result.reason == "ROLE_DENIED"
    assert result.data is None


def test_concurrent_durable_idempotency_reserves_before_handler(tmp_path: Path) -> None:
    # Given: two gateways with independent SQLite connections for the same idempotency key.
    effects: list[str] = []
    handler_started = Event()
    release_handler = Event()
    registry = ToolRegistry()

    def handler(payload: EchoInput) -> EchoOutput:
        handler_started.set()
        assert release_handler.wait(timeout=5.0)
        effects.append(payload.text)
        return EchoOutput(length=len(payload.text))

    registry.register(_spec(), handler)
    database_path = tmp_path / "project"
    with connect_project_database(database_path) as setup:
        migrate(setup)
        setup.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("run-1", "project-1", "running", "2030-01-01", "2030-01-01"),
        )
        setup.execute(
            "INSERT INTO tasks(task_id, run_id, task_type, state, created_at) VALUES (?, ?, ?, ?, ?)",
            ("task-1", "run-1", "chapter", "running", "2030-01-01"),
        )

    connections: list[sqlite3.Connection] = []
    try:
        for _ in range(2):
            connection = sqlite3.connect(database_path / "agent.sqlite", timeout=5.0, check_same_thread=False)
            connection.execute("PRAGMA busy_timeout = 5000")
            connections.append(connection)
        requests = (
            _request().model_copy(update={"invocation_id": "invoke-a"}),
            _request().model_copy(update={"invocation_id": "invoke-b"}),
        )

        with ThreadPoolExecutor(max_workers=1) as pool:
            first_future = pool.submit(ToolGateway(registry, connection=connections[0]).invoke, requests[0])
            try:
                assert handler_started.wait(timeout=5.0)

                # When: the second connection invokes the same key while the first handler is blocked.
                second = ToolGateway(registry, connection=connections[1]).invoke(requests[1])
            finally:
                release_handler.set()
            first = first_future.result(timeout=5.0)
    finally:
        for connection in connections:
            connection.close()

    # Then: one durable reservation executes the handler and the duplicate sees in-progress state.
    assert (first.outcome, second.outcome) == (ToolOutcome.SUCCEEDED, ToolOutcome.FAILED)
    assert second.reason == "IDEMPOTENCY_IN_PROGRESS"
    assert effects == ["opaque"]
