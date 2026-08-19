from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import assert_never

from novel_agent.contracts import ProviderOutcome, ProviderRequest, Provenance, ProvenanceKind
from novel_agent.pathing import ProjectId
from novel_agent.provider import (
    CancellationToken,
    ProviderClient,
    ProviderDependencies,
    ProviderTransport,
    ProviderTransportError,
    ProviderTransportResult,
)
from novel_agent.run_coordinator import CommitDisposition, RunCoordinator
from novel_agent.run_models import (
    ProviderAttemptRequest,
    StartRunRequest,
    TaskResultRequest,
    TaskStartRequest,
)
from novel_agent.storage import connect_project_database, migrate


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
FIXED_HASH = "a" * 64
PROMPT = "Write a contained mystery scene."
PROMPT_HASH = hashlib.sha256(PROMPT.encode()).hexdigest()


def provider_request(provider_call_id: str = "provider-call-001") -> ProviderRequest:
    return ProviderRequest(
        provider_call_id=provider_call_id,
        run_id="run-001",
        task_id="task-001",
        model_name="fake-model",
        prompt=PROMPT,
        prompt_sha256=PROMPT_HASH,
        temperature=0.2,
        created_at=FIXED_TIME,
        provenance=Provenance(
            source_kind=ProvenanceKind.USER_REQUEST,
            source_id="request-001",
            source_sha256=FIXED_HASH,
            observed_at=FIXED_TIME,
        ),
    )


def completion() -> ProviderTransportResult:
    return ProviderTransportResult(
        content="A safe, deterministic response.",
        model_name="fake-model",
        prompt_tokens=11,
        completion_tokens=13,
    )


class SequenceTransport:
    def __init__(self, outcomes: tuple[ProviderTransportResult | ProviderTransportError, ...]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def generate(self, request: ProviderRequest) -> ProviderTransportResult:
        assert request.task_id == "task-001"
        outcome = self.outcomes[self.calls]
        self.calls += 1
        match outcome:
            case ProviderTransportResult():
                return outcome
            case ProviderTransportError():
                raise outcome
            case unreachable:
                assert_never(unreachable)


class RecordingSleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay_seconds: float) -> None:
        self.delays.append(delay_seconds)


class CancellationAfterResponseTransport:
    def __init__(self, cancellation: CancellationToken) -> None:
        self.cancellation = cancellation

    def generate(self, request: ProviderRequest) -> ProviderTransportResult:
        assert request.task_id == "task-001"
        self.cancellation.cancel()
        return completion()


class CancellationSleeper:
    def __init__(self, cancellation: CancellationToken) -> None:
        self.cancellation = cancellation
        self.delays: list[float] = []

    def __call__(self, delay_seconds: float) -> None:
        self.delays.append(delay_seconds)
        self.cancellation.cancel()


def provider_client(
    transport: ProviderTransport,
    sleep: Callable[[float], None] | None = None,
) -> ProviderClient:
    active_sleep = RecordingSleeper() if sleep is None else sleep
    return ProviderClient(
        dependencies=ProviderDependencies(
            transport=transport,
            now=lambda: FIXED_TIME,
            sleep=active_sleep,
        )
    )


class SequenceMonotonic:
    def __init__(self, readings: tuple[float, ...]) -> None:
        self.readings = readings
        self.index = 0

    def __call__(self) -> float:
        reading = self.readings[self.index]
        self.index += 1
        return reading


@dataclass(slots=True)  # noqa: MUTABLE_OK
class MutableClock:
    instant: datetime

    def now(self) -> datetime:
        return self.instant

    def advance(self, seconds: int) -> None:
        self.instant += timedelta(seconds=seconds)


@dataclass(frozen=True, slots=True)
class CrashScenarioEvidence:
    provider_calls: int
    task_results: int
    run_checkpoints: int
    old_fencing_token: int
    new_fencing_token: int
    request_sha256: str
    artifact_rows: int
    memory_rows: int
    transport_calls: int


def run_crash_after_response_scenario(tmp_path: Path) -> CrashScenarioEvidence:
    # Given: a running task whose provider response is received before its process crashes.
    project_dir = tmp_path / "projects" / "novel-one"
    clock = MutableClock(FIXED_TIME)
    first_transport = SequenceTransport((completion(),))
    with connect_project_database(project_dir) as first_connection:
        migrate(first_connection)
        first_coordinator = RunCoordinator(first_connection, clock.now, lease_seconds=60)
        first_coordinator.submit_start(
            StartRunRequest(
                project_id=ProjectId("novel-one"),
                run_id="run-001",
                idempotency_key="start-001",
            )
        )
        old_lease = first_coordinator.claim_next("worker-before-crash")
        assert old_lease is not None
        first_coordinator.start_task(
            old_lease,
            TaskStartRequest(
                task_id="task-001",
                task_type="chapter",
                idempotency_key="task-001",
            ),
        )
        first_result = provider_client(first_transport).generate(
            provider_request("provider-call-before-crash"),
            CancellationToken(),
        )
        assert first_result.response.outcome is ProviderOutcome.SUCCEEDED
        assert first_result.content is not None
        first_attempt = first_result.attempts[0]
        first_coordinator.record_provider_attempt(
            old_lease,
            ProviderAttemptRequest(
                run_id="run-001",
                task_id="task-001",
                attempt=first_attempt.attempt_number,
                request_sha256=first_attempt.request_sha256,
                state=first_attempt.outcome.value,
            )
        )

    # When: a later process reclaims the expired lease, retries, and commits the result.
    clock.advance(61)
    second_transport = SequenceTransport((completion(),))
    with connect_project_database(project_dir) as recovered_connection:
        recovered_coordinator = RunCoordinator(recovered_connection, clock.now, lease_seconds=60)
        new_lease = recovered_coordinator.reclaim_expired("worker-after-crash")
        assert new_lease is not None
        second_result = provider_client(second_transport).generate(
            provider_request("provider-call-after-crash"),
            CancellationToken(),
        )
        second_attempt = second_result.attempts[0]
        recovered_coordinator.record_provider_attempt(
            new_lease,
            ProviderAttemptRequest(
                run_id="run-001",
                task_id="task-001",
                attempt=first_attempt.attempt_number + second_attempt.attempt_number,
                request_sha256=second_attempt.request_sha256,
                state=second_attempt.outcome.value,
            )
        )
        result_sha256 = second_result.response.response_sha256
        assert result_sha256 is not None
        stale = recovered_coordinator.commit_task_result(
            old_lease,
            TaskResultRequest(task_id="task-001", result_sha256=result_sha256),
        )
        committed = recovered_coordinator.commit_task_result(
            new_lease,
            TaskResultRequest(task_id="task-001", result_sha256=result_sha256),
        )
        repeated = recovered_coordinator.commit_task_result(
            new_lease,
            TaskResultRequest(task_id="task-001", result_sha256=result_sha256),
        )

        # Then: delivery is audited twice but only the current fence creates durable state.
        assert (old_lease.fencing_token, new_lease.fencing_token) == (1, 2)
        assert stale.disposition is CommitDisposition.DISCARDED
        assert (committed.disposition, committed.reused) == (CommitDisposition.COMMITTED, False)
        assert (repeated.disposition, repeated.reused) == (CommitDisposition.COMMITTED, True)
        provider_rows = recovered_connection.execute(
            "SELECT attempt, request_sha256, state FROM provider_calls ORDER BY attempt"
        ).fetchall()
        task_results = int(recovered_connection.execute("SELECT COUNT(*) FROM task_results").fetchone()[0])
        run_checkpoints = int(
            recovered_connection.execute("SELECT COUNT(*) FROM run_checkpoints").fetchone()[0]
        )
        artifact_rows = sum(
            int(recovered_connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("artifacts", "artifact_dependencies", "artifact_outbox")
        )
        memory_rows = sum(
            int(recovered_connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("memory_candidates", "memory_records", "memory_edges")
        )
        assert provider_rows == [(1, PROMPT_HASH, "succeeded"), (2, PROMPT_HASH, "succeeded")]
        assert (task_results, run_checkpoints, artifact_rows, memory_rows) == (1, 1, 0, 0)
        assert (first_transport.calls, second_transport.calls) == (1, 1)
        return CrashScenarioEvidence(
            provider_calls=len(provider_rows),
            task_results=task_results,
            run_checkpoints=run_checkpoints,
            old_fencing_token=old_lease.fencing_token,
            new_fencing_token=new_lease.fencing_token,
            request_sha256=PROMPT_HASH,
            artifact_rows=artifact_rows,
            memory_rows=memory_rows,
            transport_calls=first_transport.calls + second_transport.calls,
        )
