from __future__ import annotations

import sqlite3
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from novel_agent.contracts import RunCommandType, RunStatus
from novel_agent.pathing import ProjectId
from novel_agent.run_coordinator import (
    CommitDisposition,
    RunCoordinator,
    RunTransitionError,
)
from novel_agent.run_models import (
    ControlRunRequest,
    ProviderAttemptRequest,
    StartRunRequest,
    TaskResultRequest,
    TaskStartRequest,
)
from novel_agent.run_service import RunService
from novel_agent.run_worker import StageCancellationToken
from novel_agent.storage import connect_project_database, migrate


@dataclass(slots=True)  # noqa: MUTABLE_OK
class MutableClock:
    instant: datetime

    def now(self) -> datetime:
        return self.instant

    def advance(self, seconds: int) -> None:
        self.instant += timedelta(seconds=seconds)


def _start(run_id: str, key: str) -> StartRunRequest:
    return StartRunRequest(project_id=ProjectId("novel-one"), run_id=run_id, idempotency_key=key)


def _open(
    tmp_path: Path,
    lease_seconds: int = 900,
) -> tuple[sqlite3.Connection, AbstractContextManager[sqlite3.Connection], MutableClock, RunCoordinator]:
    database = connect_project_database(tmp_path / "projects" / "novel-one")
    connection = database.__enter__()
    migrate(connection)
    clock = MutableClock(datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC))
    return connection, database, clock, RunCoordinator(connection, clock.now, lease_seconds=lease_seconds)


def test_duplicate_start_reuses_the_same_command_and_run(tmp_path: Path) -> None:
    # Given: an empty authoritative project database.
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        # When: the UI submits the same start command twice.
        first = coordinator.submit_start(_start("run-001", "start-001"))
        repeated = coordinator.submit_start(_start("run-001", "start-001"))

        # Then: only one queued run and command exist, and the second receipt is reused.
        assert (first.command_id, repeated.command_id, repeated.reused) == (first.command_id, first.command_id, True)
        assert connection.execute("SELECT state FROM runs WHERE run_id = ?", ("run-001",)).fetchone() == ("queued",)
        assert connection.execute("SELECT COUNT(*) FROM commands").fetchone() == (1,)
    finally:
        database.__exit__(None, None, None)


def test_duplicate_start_receipt_reports_current_run_state(tmp_path: Path) -> None:
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-state", "start-state"))
        lease = coordinator.claim_run("run-state", "worker-a")
        assert lease is not None
        coordinator.fail_run(lease, "stage_failed")

        # A failed run + the SAME idempotency key is an explicit retry: the
        # run requeues instead of erroring or being a silent no-op.
        repeated = coordinator.submit_start(_start("run-state", "start-state"))

        assert repeated.reused is True
        assert repeated.state == RunStatus.QUEUED.value
        assert connection.execute(
            "SELECT state FROM runs WHERE run_id = ?", ("run-state",)
        ).fetchone() == ("queued",)
        requeued = connection.execute(
            "SELECT payload_json FROM run_events WHERE run_id = ? AND event_type = 'run_requeued'",
            ("run-state",),
        ).fetchall()
        assert len(requeued) == 1
    finally:
        database.__exit__(None, None, None)


def test_requeue_terminal_run_with_new_key_updates_payload_and_keeps_artifacts(tmp_path: Path) -> None:
    """A new idempotency key on a terminal run requeues with the new payload."""
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-requeue", "key-a"))
        lease = coordinator.claim_run("run-requeue", "worker-a")
        assert lease is not None
        coordinator.fail_run(lease, "stage_failed")
        # Plant an accepted artifact + a staged artifact to prove reuse/purge.
        connection.execute(
            "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, "
            "sha256, schema_version, is_current, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run-requeue:brief:plan", 1, "novel-one", "run-requeue", "plan_contract", "accepted",
             b"{}", "x" * 64, 1, 1, _clock.now().isoformat()),
        )
        connection.execute(
            "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, "
            "sha256, schema_version, is_current, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("staged-plan-x", 1, "novel-one", "run-requeue", "plan_contract", "staged",
             b"{}", "y" * 64, 1, 0, _clock.now().isoformat()),
        )

        receipt = coordinator.submit_start(_start("run-requeue", "key-b"))

        assert receipt.reused is False
        assert receipt.state == RunStatus.QUEUED.value
        assert connection.execute(
            "SELECT COUNT(*) FROM commands WHERE run_id = ?", ("run-requeue",)
        ).fetchone() == (2,)
        # Accepted artifact survives (asset reuse), staged artifact purged.
        assert connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND status = 'accepted'", ("run-requeue",)
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND status = 'staged'", ("run-requeue",)
        ).fetchone() == (0,)
    finally:
        database.__exit__(None, None, None)


def test_requeue_purges_execution_traces_but_keeps_canonical_memory(tmp_path: Path) -> None:
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-purge", "key-a"))
        lease = coordinator.claim_run("run-purge", "worker-a")
        assert lease is not None
        coordinator.fail_run(lease, "stage_failed")

        # Plant run-scoped execution traces and canonical memory.
        now = _clock.now().isoformat()
        connection.execute(
            "INSERT INTO tasks(task_id, run_id, task_type, state, idempotency_key, created_at, updated_at) "
            "VALUES (?, ?, 'chapter', 'succeeded', ?, ?, ?)",
            ("run-purge:chapter:c1", "run-purge", "run-purge:chapter:c1", now, now),
        )
        connection.execute(
            "INSERT INTO tool_calls(call_id, run_id, task_id, tool_id, idempotency_key, request_sha256, state, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("tc-1", "run-purge", "run-purge:chapter:c1", "file.read", "run-purge:tool:1", "r1", "succeeded", now),
        )
        connection.execute(
            "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, "
            "subject_id, predicate, value_json, created_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("mem-plan", "novel-one", "run-purge", 1, "m2_plan", "chapter_plan", "validated",
             "s1", "chapter", '"{}"', "t", now),
        )
        connection.execute(
            "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, "
            "subject_id, predicate, value_json, created_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("mem-canon", "novel-one", "run-purge", 0, "m1_canonical", "world_rule", "validated",
             "w1", "premise", '"{}"', "t", now),
        )

        coordinator.submit_start(_start("run-purge", "key-b"))

        assert connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE run_id = ?", ("run-purge",)
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE run_id = ?", ("run-purge",)
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE memory_id = 'mem-plan'"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE memory_id = 'mem-canon'"
        ).fetchone() == (1,)
    finally:
        database.__exit__(None, None, None)


def test_start_rejected_while_run_busy(tmp_path: Path) -> None:
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-busy", "key-a"))
        lease = coordinator.claim_run("run-busy", "worker-a")
        assert lease is not None

        # A new start while RUNNING collides with the active execution.
        with pytest.raises(RunTransitionError):
            coordinator.submit_start(_start("run-busy", "key-b"))

        # The SAME key while busy is a no-op receipt (double-click guard).
        repeated = coordinator.submit_start(_start("run-busy", "key-a"))
        assert repeated.reused is True
        assert repeated.state == RunStatus.RUNNING.value
        assert connection.execute(
            "SELECT state FROM runs WHERE run_id = ?", ("run-busy",)
        ).fetchone() == ("running",)
    finally:
        database.__exit__(None, None, None)


def test_service_callback_only_submits_command(tmp_path: Path) -> None:
    _connection, database, _clock, coordinator = _open(tmp_path)
    service = RunService(coordinator)
    try:
        # When: the UI-facing service starts a governed run.
        receipt = service.start(_start("run-001", "start-001"))

        # Then: it only queues the command and leaves external generation to a worker.
        assert receipt.run_id == "run-001"
        assert receipt.state == "queued"
    finally:
        database.__exit__(None, None, None)


def test_one_worker_slot_keeps_a_second_run_queued(tmp_path: Path) -> None:
    # Given: two queued runs and the default single worker capacity.
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-001", "start-001"))
        coordinator.submit_start(_start("run-002", "start-002"))

        # When: two workers attempt a claim.
        first = coordinator.claim_next("worker-a")
        second = coordinator.claim_next("worker-b")

        # Then: only the first lease is active and the other run remains queued.
        assert first is not None
        assert second is None
        assert connection.execute("SELECT state FROM runs WHERE run_id = ?", ("run-002",)).fetchone() == ("queued",)
    finally:
        database.__exit__(None, None, None)


def test_claim_run_claims_only_requested_queued_run(tmp_path: Path) -> None:
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-a", "start-a"))
        coordinator.submit_start(_start("run-b", "start-b"))

        lease = coordinator.claim_run("run-b", "worker-a")

        assert lease is not None
        assert lease.run_id == "run-b"
        assert connection.execute("SELECT state FROM runs WHERE run_id = 'run-a'").fetchone() == ("queued",)
    finally:
        database.__exit__(None, None, None)


def test_claim_next_preserves_monotonic_fence_after_existing_lease(tmp_path: Path) -> None:
    connection, database, clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-001", "start-001"))
        connection.execute(
            "INSERT INTO run_leases(run_id, holder_id, fencing_token, expires_at, acquired_at) VALUES (?, ?, ?, ?, ?)",
            ("run-001", "old-worker", 7, (clock.instant - timedelta(seconds=1)).isoformat(), clock.instant.isoformat()),
        )

        lease = coordinator.claim_next("worker-a")

        assert lease is not None
        assert lease.fencing_token == 8
    finally:
        database.__exit__(None, None, None)


def test_repeated_start_task_is_idempotent(tmp_path: Path) -> None:
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-001", "start-001"))
        lease = coordinator.claim_next("worker-a")
        assert lease is not None
        request = TaskStartRequest(task_id="task-001", task_type="chapter", idempotency_key="task-001")

        coordinator.start_task(lease, request)
        coordinator.start_task(lease, request)

        assert connection.execute("SELECT COUNT(*) FROM tasks WHERE task_id = 'task-001'").fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id = 'run-001' AND event_type = 'task_started'"
        ).fetchone() == (1,)
    finally:
        database.__exit__(None, None, None)


def test_worker_executor_failure_marks_run_failed(tmp_path: Path) -> None:
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-001", "start-001"))

        class FailingStage:
            def execute(self, _recovery: RunRecoveryState) -> None:
                raise RuntimeError("stage failed")

        result = RunService(coordinator, stage_executor=FailingStage()).run_once("worker-a")

        assert result is None
        assert connection.execute("SELECT state FROM runs WHERE run_id = 'run-001'").fetchone() == ("failed",)
        assert connection.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id = 'run-001' AND event_type = 'run_failed'"
        ).fetchone() == (1,)
    finally:
        database.__exit__(None, None, None)


def test_worker_hung_executor_is_bounded_and_marks_run_failed(tmp_path: Path) -> None:
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-001", "start-001"))

        class HungStage:
            def execute(self, _recovery: RunRecoveryState) -> None:
                time.sleep(0.2)

        started = time.monotonic()
        result = RunService(coordinator, stage_executor=HungStage(), stage_timeout_seconds=0.02).run_once("worker-a")

        assert result is None
        assert time.monotonic() - started < 0.15
        assert connection.execute("SELECT state FROM runs WHERE run_id = 'run-001'").fetchone() == ("failed",)
        assert connection.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id = 'run-001' AND event_type = 'run_failed'"
        ).fetchone() == (1,)
    finally:
        database.__exit__(None, None, None)


def test_worker_timeout_does_not_overlap_a_late_stage_with_the_next_run(tmp_path: Path) -> None:
    connection, database, _clock, coordinator = _open(tmp_path)
    release = Event()
    started = Event()
    executions: list[str] = []
    try:
        coordinator.submit_start(_start("run-001", "start-001"))
        coordinator.submit_start(_start("run-002", "start-002"))

        class HungStage:
            def execute(self, recovery: RunRecoveryState) -> None:
                executions.append(recovery.lease.run_id)
                started.set()
                release.wait(timeout=1)

        service = RunService(coordinator, stage_executor=HungStage(), stage_timeout_seconds=0.02)
        assert service.run_once("worker-a") is None
        assert started.wait(timeout=1)

        second = service.run_once("worker-b")

        assert second is None
        assert executions == ["run-001"]
    finally:
        release.set()
        time.sleep(0.05)
        database.__exit__(None, None, None)


def test_worker_timeout_signals_cooperative_stage_cancellation(tmp_path: Path) -> None:
    connection, database, _clock, coordinator = _open(tmp_path)
    started = Event()
    cancelled = Event()
    try:
        coordinator.submit_start(_start("run-001", "start-001"))

        class CooperativeStage:
            def execute(self, _recovery: RunRecoveryState, cancellation: StageCancellationToken) -> None:
                started.set()
                while not cancellation.is_cancelled():
                    time.sleep(0.001)
                cancelled.set()

        result = RunService(
            coordinator,
            stage_executor=CooperativeStage(),
            stage_timeout_seconds=0.02,
        ).run_once("worker-a")

        assert result is None
        assert started.is_set()
        assert cancelled.is_set()
        assert connection.execute("SELECT state FROM runs WHERE run_id = 'run-001'").fetchone() == ("failed",)
    finally:
        database.__exit__(None, None, None)


def test_expired_lease_is_reclaimed_with_a_higher_fence(tmp_path: Path) -> None:
    # Given: a claimed run whose lease later expires (short lease on purpose).
    _connection, database, clock, coordinator = _open(tmp_path, lease_seconds=60)
    try:
        coordinator.submit_start(_start("run-001", "start-001"))
        first = coordinator.claim_next("worker-a")
        assert first is not None
        clock.advance(61)

        # When: a different worker performs startup recovery.
        reclaimed = coordinator.reclaim_expired("worker-b")

        # Then: the live lease belongs to the new worker with a monotonic fence.
        assert reclaimed is not None
        assert (reclaimed.holder_id, reclaimed.fencing_token) == ("worker-b", first.fencing_token + 1)
    finally:
        database.__exit__(None, None, None)


def test_illegal_transition_is_rejected_and_audited(tmp_path: Path) -> None:
    # Given: a queued run that cannot be paused.
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-001", "start-001"))

        # When: a caller requests an illegal pause transition.
        with pytest.raises(RunTransitionError):
            coordinator.control(ControlRunRequest(run_id="run-001", command_type=RunCommandType.PAUSE, idempotency_key="pause-001"))

        # Then: state remains queued and the rejected command has a readable event.
        assert connection.execute("SELECT state FROM runs WHERE run_id = ?", ("run-001",)).fetchone() == ("queued",)
        assert connection.execute("SELECT event_type FROM run_events WHERE run_id = ? ORDER BY sequence DESC", ("run-001",)).fetchone() == ("transition_rejected",)
    finally:
        database.__exit__(None, None, None)


def test_cancel_or_fence_loss_discards_late_task_result(tmp_path: Path) -> None:
    # Given: a running task held by a worker lease.
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-001", "start-001"))
        lease = coordinator.claim_next("worker-a")
        assert lease is not None
        coordinator.start_task(lease, TaskStartRequest(task_id="task-001", task_type="chapter", idempotency_key="task-key-001"))
        coordinator.control(ControlRunRequest(run_id="run-001", command_type=RunCommandType.CANCEL, idempotency_key="cancel-001"))

        # When: the cancelled worker returns after the provider boundary.
        outcome = coordinator.commit_task_result(lease, TaskResultRequest(task_id="task-001", result_sha256="a" * 64))

        # Then: no late result becomes a durable task checkpoint.
        assert outcome.disposition is CommitDisposition.DISCARDED
        assert connection.execute("SELECT COUNT(*) FROM task_results").fetchone() == (0,)
    finally:
        database.__exit__(None, None, None)


def test_crash_after_provider_send_records_two_attempts_but_one_commit(tmp_path: Path) -> None:
    # Given: a task whose first provider request was sent before a simulated process crash.
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-001", "start-001"))
        lease = coordinator.claim_next("worker-a")
        assert lease is not None
        coordinator.start_task(lease, TaskStartRequest(task_id="task-001", task_type="chapter", idempotency_key="task-key-001"))
        coordinator.record_provider_attempt(
            lease,
            ProviderAttemptRequest(run_id="run-001", task_id="task-001", attempt=1, request_sha256="b" * 64, state="sent"),
        )

        # When: recovery retries the same logical task and a late duplicate commit arrives.
        coordinator.record_provider_attempt(
            lease,
            ProviderAttemptRequest(run_id="run-001", task_id="task-001", attempt=2, request_sha256="b" * 64, state="succeeded"),
        )
        first = coordinator.commit_task_result(lease, TaskResultRequest(task_id="task-001", result_sha256="c" * 64))
        repeated = coordinator.commit_task_result(lease, TaskResultRequest(task_id="task-001", result_sha256="c" * 64))

        # Then: audit preserves at-least-once delivery while authoritative state commits exactly once.
        assert (first.disposition, repeated.disposition, repeated.reused) == (
            CommitDisposition.COMMITTED,
            CommitDisposition.COMMITTED,
            True,
        )
        assert connection.execute("SELECT COUNT(*) FROM provider_calls WHERE task_id = ?", ("task-001",)).fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM task_results WHERE task_id = ?", ("task-001",)).fetchone() == (1,)
    finally:
        database.__exit__(None, None, None)


def test_current_lease_cannot_commit_a_task_from_a_different_run(tmp_path: Path) -> None:
    # Given: an active task from a paused run and a current lease for another run.
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-a", "start-a"))
        coordinator.submit_start(_start("run-b", "start-b"))
        lease_a = coordinator.claim_next("worker-a")
        assert lease_a is not None
        coordinator.start_task(lease_a, TaskStartRequest(task_id="task-a", task_type="chapter", idempotency_key="task-a"))
        coordinator.control(ControlRunRequest(run_id="run-a", command_type=RunCommandType.PAUSE, idempotency_key="pause-a"))
        lease_b = coordinator.claim_next("worker-b")
        assert lease_b is not None

        # When: the second worker tries to promote the first run's task.
        outcome = coordinator.commit_task_result(lease_b, TaskResultRequest(task_id="task-a", result_sha256="d" * 64))

        # Then: no cross-run result, checkpoint, or task-state mutation is committed.
        assert outcome.disposition is CommitDisposition.DISCARDED
        assert connection.execute("SELECT state FROM tasks WHERE task_id = ?", ("task-a",)).fetchone() == ("running",)
        assert connection.execute("SELECT COUNT(*) FROM task_results WHERE task_id = ?", ("task-a",)).fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM run_checkpoints WHERE task_id = ?", ("task-a",)).fetchone() == (0,)
    finally:
        database.__exit__(None, None, None)


def test_provider_attempt_requires_current_lease_and_owned_running_task(tmp_path: Path) -> None:
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-a", "start-a"))
        coordinator.submit_start(_start("run-b", "start-b"))
        stale_lease = coordinator.claim_next("worker-a")
        assert stale_lease is not None
        coordinator.start_task(
            stale_lease,
            TaskStartRequest(task_id="task-a", task_type="chapter", idempotency_key="task-a"),
        )
        coordinator.control(
            ControlRunRequest(run_id="run-a", command_type=RunCommandType.CANCEL, idempotency_key="cancel-a")
        )
        current_lease = coordinator.claim_next("worker-b")
        assert current_lease is not None
        request = ProviderAttemptRequest(
            run_id="run-a",
            task_id="task-a",
            attempt=1,
            request_sha256="d" * 64,
            state="succeeded",
        )

        stale = coordinator.record_provider_attempt(stale_lease, request)
        wrong_owner = coordinator.record_provider_attempt(current_lease, request)

        assert stale is CommitDisposition.DISCARDED
        assert wrong_owner is CommitDisposition.DISCARDED
        assert connection.execute("SELECT COUNT(*) FROM provider_calls").fetchone() == (0,)
    finally:
        database.__exit__(None, None, None)


def test_provider_attempt_discards_a_pending_task(tmp_path: Path) -> None:
    # Given: a current run lease whose next task has not been activated.
    connection, database, clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-001", "start-001"))
        lease = coordinator.claim_next("worker-a")
        assert lease is not None
        timestamp = clock.instant.isoformat()
        connection.execute(
            "INSERT INTO tasks(task_id, run_id, task_type, state, idempotency_key, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("task-pending", "run-001", "chapter", "pending", "task-pending", timestamp, timestamp),
        )

        # When: the current lease tries to audit a provider attempt before task activation.
        outcome = coordinator.record_provider_attempt(
            lease,
            ProviderAttemptRequest(
                run_id="run-001",
                task_id="task-pending",
                attempt=1,
                request_sha256="f" * 64,
                state="sent",
            ),
        )

        # Then: pending work has no provider authority and leaves no provider-call row.
        assert outcome is CommitDisposition.DISCARDED
        assert connection.execute("SELECT COUNT(*) FROM provider_calls").fetchone() == (0,)
        assert connection.execute(
            "SELECT event_type FROM run_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone() == ("provider_attempt_discarded",)
    finally:
        database.__exit__(None, None, None)


def test_conflicting_duplicate_provider_attempt_is_discarded_and_audited(tmp_path: Path) -> None:
    connection, database, _clock, coordinator = _open(tmp_path)
    try:
        coordinator.submit_start(_start("run-001", "start-001"))
        lease = coordinator.claim_next("worker-a")
        assert lease is not None
        coordinator.start_task(lease, TaskStartRequest(task_id="task-001", task_type="chapter", idempotency_key="task-001"))
        first = ProviderAttemptRequest(run_id="run-001", task_id="task-001", attempt=1, request_sha256="a" * 64, state="sent")
        conflict = first.model_copy(update={"request_sha256": "b" * 64, "state": "succeeded"})

        assert coordinator.record_provider_attempt(lease, first) is CommitDisposition.COMMITTED
        assert coordinator.record_provider_attempt(lease, conflict) is CommitDisposition.DISCARDED
        assert connection.execute("SELECT COUNT(*) FROM provider_calls WHERE task_id = 'task-001'").fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id = 'run-001' AND event_type = 'provider_attempt_discarded'"
        ).fetchone() == (1,)
    finally:
        database.__exit__(None, None, None)


def test_resume_from_committed_checkpoint(tmp_path: Path) -> None:
    # Given: a process that stops after a provider send and a durable project database.
    project_dir = tmp_path / "projects" / "novel-one"
    initial_clock = MutableClock(datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC))
    with connect_project_database(project_dir) as first_connection:
        migrate(first_connection)
        initial = RunCoordinator(first_connection, initial_clock.now, lease_seconds=60)
        initial.submit_start(_start("run-001", "start-001"))
        first_lease = initial.claim_next("worker-a")
        assert first_lease is not None
        initial.start_task(first_lease, TaskStartRequest(task_id="task-001", task_type="chapter", idempotency_key="task-001"))
        initial.record_provider_attempt(
            first_lease,
            ProviderAttemptRequest(run_id="run-001", task_id="task-001", attempt=1, request_sha256="e" * 64, state="sent")
        )

    resumed_clock = MutableClock(initial_clock.instant + timedelta(seconds=61))
    completed_calls: list[tuple[str | None, str | None]] = []
    with connect_project_database(project_dir) as resumed_connection:
        recovered = RunCoordinator(resumed_connection, resumed_clock.now, lease_seconds=60)

        class CompletingStage:
            def execute(self, recovery) -> None:
                task_id = None if recovery.active_task is None else recovery.active_task.task_id
                checkpoint_id = None if recovery.last_checkpoint is None else recovery.last_checkpoint.task_id
                completed_calls.append((task_id, checkpoint_id))
                assert recovery.active_task is not None
                recovered.record_provider_attempt(
                    recovery.lease,
                    ProviderAttemptRequest(
                        run_id=recovery.lease.run_id,
                        task_id=recovery.active_task.task_id,
                        attempt=2,
                        request_sha256="e" * 64,
                        state="succeeded",
                    )
                )
                recovered.commit_task_result(
                    recovery.lease,
                    TaskResultRequest(task_id=recovery.active_task.task_id, result_sha256="f" * 64),
                )

        # When: the service reclaims the expired lease and runs the unfinished task stage once.
        first_resume = RunService(recovered, stage_executor=CompletingStage()).run_once("worker-b")

        # Then: recovery resumes only the uncommitted task and commits one result/checkpoint.
        assert first_resume is not None
        assert completed_calls == [("task-001", None)]
        assert resumed_connection.execute("SELECT COUNT(*) FROM provider_calls WHERE task_id = ?", ("task-001",)).fetchone() == (2,)
        assert resumed_connection.execute("SELECT COUNT(*) FROM task_results WHERE task_id = ?", ("task-001",)).fetchone() == (1,)
        assert resumed_connection.execute("SELECT COUNT(*) FROM run_checkpoints WHERE task_id = ?", ("task-001",)).fetchone() == (1,)

    checkpoint_clock = MutableClock(resumed_clock.instant + timedelta(seconds=61))
    checkpoint_calls: list[tuple[str | None, str | None]] = []
    with connect_project_database(project_dir) as checkpoint_connection:
        checkpoint_coordinator = RunCoordinator(checkpoint_connection, checkpoint_clock.now)

        class CheckpointStage:
            def execute(self, recovery) -> None:
                task_id = None if recovery.active_task is None else recovery.active_task.task_id
                checkpoint_id = None if recovery.last_checkpoint is None else recovery.last_checkpoint.task_id
                checkpoint_calls.append((task_id, checkpoint_id))

        # When: another restart occurs after the first task checkpoint committed.
        second_resume = RunService(
            checkpoint_coordinator,
            stage_executor=CheckpointStage(),
        ).run_once("worker-c")

        # Then: only the committed checkpoint, not the completed task, is handed to the next stage.
        assert second_resume is not None
        assert checkpoint_calls == [(None, "task-001")]
        assert checkpoint_connection.execute("SELECT COUNT(*) FROM task_results WHERE task_id = ?", ("task-001",)).fetchone() == (1,)
        assert checkpoint_connection.execute("SELECT COUNT(*) FROM run_checkpoints WHERE task_id = ?", ("task-001",)).fetchone() == (1,)
