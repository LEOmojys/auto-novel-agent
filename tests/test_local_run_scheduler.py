from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import pytest

from novel_agent.contracts import RunCommandType, RunStatus
from novel_agent.pathing import ProjectId, ProjectPathResolver
from novel_agent.queued_run_failure import QueuedRunFailureReason, fail_queued_run
from novel_agent.run_coordinator import RunCoordinator
from novel_agent.run_models import RunRecoveryState
from novel_agent.service import NovelAgentService
from novel_agent.service_models import ControlNovelRunRequest, StartNovelRunRequest
from novel_agent.storage import connect_project_database


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("scheduler-factory-project")


class FactorySetupError(Exception):
    pass


def test_factory_failure_fails_run_and_continues_draining(tmp_path: Path) -> None:
    # Given: one factory invocation is held until a later request is durably queued.
    factory_started = Event()
    release_failure = Event()
    second_completed = Event()

    class SuccessfulStage:
        def __init__(self, coordinator: RunCoordinator) -> None:
            self.coordinator = coordinator

        def execute(self, recovery: RunRecoveryState) -> None:
            self.coordinator.block_planning(recovery.lease, "test_stage_complete")
            second_completed.set()

    def factory(coordinator: RunCoordinator, request: StartNovelRunRequest) -> SuccessfulStage:
        if request.run_id == "factory-failure-first":
            factory_started.set()
            assert release_failure.wait(2.0)
            raise FactorySetupError("secret-token-must-not-be-persisted")
        return SuccessfulStage(coordinator)

    service = NovelAgentService(
        ProjectPathResolver(tmp_path / "projects"),
        now=lambda: FIXED_TIME,
        stage_executor_factory=factory,
    )
    service.start(StartNovelRunRequest(
        project_id=PROJECT_ID,
        run_id="factory-failure-first",
        idempotency_key="start-factory-failure-first",
        user_requirements="first",
    ))
    assert factory_started.wait(2.0)

    # When: the later run is queued before the first factory raises.
    service.start(StartNovelRunRequest(
        project_id=PROJECT_ID,
        run_id="factory-failure-second",
        idempotency_key="start-factory-failure-second",
        user_requirements="second",
    ))
    release_failure.set()

    # Then: the scheduler audits the first failure and drains the second request.
    assert second_completed.wait(2.0)
    project_dir = tmp_path / "projects" / PROJECT_ID.root
    with connect_project_database(project_dir) as connection:
        states = connection.execute(
            "SELECT run_id, state FROM runs WHERE run_id LIKE 'factory-failure-%' ORDER BY run_id"
        ).fetchall()
        failed_event = connection.execute(
            "SELECT event_type, payload_json FROM run_events "
            "WHERE run_id = ? AND event_type = 'run_failed'",
            ("factory-failure-first",),
        ).fetchone()
    assert states == [
        ("factory-failure-first", RunStatus.FAILED.value),
        ("factory-failure-second", RunStatus.WAITING_REVIEW.value),
    ]
    assert failed_event is not None
    assert failed_event[0] == "run_failed"
    assert json.loads(str(failed_event[1])) == {"reason": "stage_executor_factory_failed"}


def test_drain_run_once_failure_fails_run_and_keeps_draining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a RunService whose construction explodes inside the drain loop.
    second_completed = Event()

    class SuccessfulStage:
        def __init__(self, coordinator: RunCoordinator) -> None:
            self.coordinator = coordinator

        def execute(self, recovery: RunRecoveryState) -> None:
            self.coordinator.block_planning(recovery.lease, "test_stage_complete")
            second_completed.set()

    def factory(coordinator: RunCoordinator, _request: StartNovelRunRequest) -> SuccessfulStage:
        return SuccessfulStage(coordinator)

    def exploding_run_service(*_args, **_kwargs):
        raise RuntimeError("injected drain failure")

    monkeypatch.setattr("novel_agent.local_run_scheduler.RunService", exploding_run_service)

    service = NovelAgentService(
        ProjectPathResolver(tmp_path / "projects"),
        now=lambda: FIXED_TIME,
        stage_executor_factory=factory,
    )
    service.start(StartNovelRunRequest(
        project_id=PROJECT_ID,
        run_id="drain-failure-first",
        idempotency_key="start-drain-failure-first",
        user_requirements="first",
    ))

    # When: the run service keeps exploding for the first run only.
    monkeypatch.setattr(
        "novel_agent.local_run_scheduler.RunService",
        lambda coordinator, stage_executor, stage_timeout_seconds: (
            _FirstRunRunService(coordinator, stage_executor, stage_timeout_seconds, second_completed)
        ),
    )
    service.start(StartNovelRunRequest(
        project_id=PROJECT_ID,
        run_id="drain-failure-second",
        idempotency_key="start-drain-failure-second",
        user_requirements="second",
    ))

    # Then: the first run is audited as failed and the queue keeps draining.
    assert second_completed.wait(3.0)
    project_dir = tmp_path / "projects" / PROJECT_ID.root
    with connect_project_database(project_dir) as connection:
        states = connection.execute(
            "SELECT run_id, state FROM runs WHERE run_id LIKE 'drain-failure-%' ORDER BY run_id"
        ).fetchall()
        failed_event = connection.execute(
            "SELECT payload_json FROM run_events "
            "WHERE run_id = ? AND event_type = 'run_failed'",
            ("drain-failure-first",),
        ).fetchone()
    assert states == [
        ("drain-failure-first", RunStatus.FAILED.value),
        ("drain-failure-second", RunStatus.WAITING_REVIEW.value),
    ]
    assert json.loads(str(failed_event[0])) == {"reason": "scheduler_drain_failed"}


class _FirstRunRunService:
    """Real RunService, except the first run still explodes during run_once."""

    def __init__(self, coordinator, stage_executor, stage_timeout_seconds, completed: Event) -> None:
        self._delegate = __import__("novel_agent.run_service", fromlist=["RunService"]).RunService(
            coordinator,
            stage_executor=stage_executor,
            stage_timeout_seconds=stage_timeout_seconds,
        )
        self._completed = completed

    def run_once(self, holder_id: str, run_id: str | None = None):
        if run_id == "drain-failure-first":
            raise RuntimeError("injected drain failure")
        return self._delegate.run_once(holder_id, run_id)


def test_queued_failure_reason_rejects_malformed_value() -> None:
    # Given: an unrecognized failure value from outside the typed scheduler path.
    malformed_reason = "factory-secret-or-exception-text"

    # When/Then: parsing rejects it instead of allowing arbitrary audit content.
    with pytest.raises(ValueError):
        QueuedRunFailureReason(malformed_reason)


def test_queued_failure_does_not_overwrite_cancelled_run(tmp_path: Path) -> None:
    # Given: a request has left QUEUED through the governed cancel transition.
    service = NovelAgentService(ProjectPathResolver(tmp_path / "projects"), now=lambda: FIXED_TIME)
    service.start(StartNovelRunRequest(
        project_id=PROJECT_ID,
        run_id="cancelled-before-factory-failure",
        idempotency_key="start-cancel-race",
        user_requirements="cancel race",
    ))
    service.control(ControlNovelRunRequest(
        project_id=PROJECT_ID,
        run_id="cancelled-before-factory-failure",
        command_type=RunCommandType.CANCEL,
        idempotency_key="cancel-before-factory-failure",
    ))

    # When: a late factory failure attempts its guarded repository transition.
    project_dir = tmp_path / "projects" / PROJECT_ID.root
    with connect_project_database(project_dir) as connection:
        changed = fail_queued_run(
            connection,
            lambda: FIXED_TIME,
            "cancelled-before-factory-failure",
            QueuedRunFailureReason.STAGE_EXECUTOR_FACTORY_FAILED,
        )
        state = connection.execute(
            "SELECT state FROM runs WHERE run_id = ?",
            ("cancelled-before-factory-failure",),
        ).fetchone()
        failure_events = connection.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id = ? AND event_type = 'run_failed'",
            ("cancelled-before-factory-failure",),
        ).fetchone()

    # Then: the stale failure loses the race and emits no misleading audit event.
    assert changed is False
    assert state == (RunStatus.CANCELLED.value,)
    assert failure_events == (0,)


def test_recovery_picks_latest_start_command(tmp_path: Path) -> None:
    """After a requeue the scheduler must recover the newest start payload."""
    project_dir = tmp_path / "projects" / PROJECT_ID.root
    project_dir.mkdir(parents=True)
    captured: list[str] = []

    class Stage:
        def __init__(self, coordinator: RunCoordinator) -> None:
            self.coordinator = coordinator

        def execute(self, recovery: RunRecoveryState) -> None:
            self.coordinator.block_planning(recovery.lease, "test_stage_complete")

    def factory(coordinator: RunCoordinator, request: StartNovelRunRequest) -> Stage:
        captured.append(request.user_requirements)
        return Stage(coordinator)

    # Seed the database directly: one QUEUED run with TWO start commands — an
    # older payload and a newer one (as a requeue with a fresh key produces).
    from novel_agent.run_models import StartRunExecutionPayload
    from novel_agent.storage import migrate

    def payload(key: str, requirements: str) -> str:
        return StartRunExecutionPayload(
            project_id=PROJECT_ID,
            run_id="recovery-latest",
            idempotency_key=key,
            user_requirements=requirements,
            provider="offline_fake",
        ).model_dump_json()

    with connect_project_database(project_dir) as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("recovery-latest", PROJECT_ID.root, RunStatus.QUEUED.value, FIXED_TIME.isoformat(), FIXED_TIME.isoformat()),
        )
        connection.execute(
            "INSERT INTO commands(command_id, run_id, command_type, idempotency_key, payload_json, state, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'accepted', ?)",
            ("cmd-old", "recovery-latest", RunCommandType.START.value, "key-old", payload("key-old", "旧需求"), FIXED_TIME.isoformat()),
        )
        connection.execute(
            "INSERT INTO commands(command_id, run_id, command_type, idempotency_key, payload_json, state, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'accepted', ?)",
            ("cmd-new", "recovery-latest", RunCommandType.START.value, "key-new", payload("key-new", "新需求"), FIXED_TIME.isoformat()),
        )

    # When: the scheduler boots, it recovers queued requests from the DB.
    NovelAgentService(
        ProjectPathResolver(tmp_path / "projects"),
        now=lambda: FIXED_TIME,
        stage_executor_factory=factory,
    )

    # Then: the drain executes with the NEWEST start payload, not the stale one.
    deadline = time.monotonic() + 2.0
    while not captured and time.monotonic() < deadline:
        time.sleep(0.01)
    assert captured == ["新需求"]
