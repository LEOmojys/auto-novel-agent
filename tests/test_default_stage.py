from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

import pytest

import config
from novel_agent.chapter_executor import ChapterExecutionCancelled, ChapterExecutionWorkflow
from novel_agent.contracts import RunStatus, StoryBrief
from novel_agent.default_stage import default_stage_executor_factory
from novel_agent.pathing import ProjectId, ProjectPathResolver
from novel_agent.offline_fake import OfflinePlanningGenerator, offline_provenance
from novel_agent.planning import PlanningSucceeded, PlanningWorkflow
from novel_agent.run_coordinator import RunCoordinator
from novel_agent.run_models import StartRunRequest
from novel_agent.service import NovelAgentService, StartNovelRunRequest
from novel_agent.storage import connect_project_database, migrate
from novel_agent.skills import load_builtin_catalog


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_default_stage_turns_missing_provider_into_review_state(tmp_path: Path) -> None:
    # Given: a queued run using the launcher's safe default stage.
    project_id = ProjectId("novel-one")
    with connect_project_database(tmp_path / "projects" / project_id.root) as connection:
        migrate(connection)
        coordinator = RunCoordinator(connection, lambda: FIXED_TIME)
        coordinator.submit_start(
            StartRunRequest(
                project_id=project_id,
                run_id="run-default-stage",
                idempotency_key="start-default-stage",
            )
        )
        lease = coordinator.claim_next("default-worker")
        assert lease is not None
        recovery = coordinator.resume_state(lease)

        # When: the default stage executes without a configured Provider adapter.
        stage = default_stage_executor_factory(
            coordinator,
            StartNovelRunRequest(
                project_id=project_id,
                run_id="run-default-stage",
                idempotency_key="start-default-stage",
                user_requirements="offline review",
            ),
        )
        stage.execute(recovery)

        # Then: the run pauses with a durable, user-visible reason instead of staying queued.
        row = connection.execute(
            "SELECT state FROM runs WHERE run_id = ?", ("run-default-stage",)
        ).fetchone()
        event = connection.execute(
            "SELECT event_type, payload_json FROM run_events WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
            ("run-default-stage",),
        ).fetchone()
        assert row == (RunStatus.WAITING_REVIEW.value,)
        assert event == ("plan_blocked", '{"reason": "provider_executor_not_configured"}')


def test_configured_offline_stage_completes_plan_chapter_and_memory(tmp_path: Path) -> None:
    # Given: an explicitly configured offline provider and a fresh governed project.
    project_id = ProjectId("novel-one")
    project_dir = tmp_path / "projects" / project_id.root
    with connect_project_database(project_dir) as connection:
        migrate(connection)
        coordinator = RunCoordinator(connection, lambda: FIXED_TIME)
        coordinator.submit_start(
            StartRunRequest(
                project_id=project_id,
                run_id="run-offline-configured",
                idempotency_key="start-offline-configured",
            )
        )
        lease = coordinator.claim_next("offline-worker")
        assert lease is not None
        recovery = coordinator.resume_state(lease)

        # When: the explicitly selected default executor runs the governed pipeline.
        stage = default_stage_executor_factory(
            coordinator,
            StartNovelRunRequest(
                project_id=project_id,
                run_id="run-offline-configured",
                idempotency_key="start-offline-configured",
                user_requirements="A deterministic offline mystery.",
                provider="offline_fake",
            ),
            path_resolver=ProjectPathResolver(tmp_path / "projects"),
            now=lambda: FIXED_TIME,
        )
        stage.execute(recovery)

        # Then: the coordinator owns the successful plan/chapter/M3 writes.
        assert connection.execute(
            "SELECT state FROM runs WHERE run_id = ?",
            ("run-offline-configured",),
        ).fetchone() == (RunStatus.SUCCEEDED.value,)
        assert connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND kind IN ('world_bible', 'character_set', 'plan_contract', 'chapter')",
            ("run-offline-configured",),
        ).fetchone() == (4,)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE run_id = ? AND layer = 'm3_confirmed'",
            ("run-offline-configured",),
        ).fetchone() == (1,)


def test_accepted_plan_load_does_not_treat_plan_revision_as_artifact_revision(tmp_path: Path) -> None:
    # Given: planning committed artifact revision one while the run's plan revision later advanced.
    project_id = ProjectId("novel-one")
    with connect_project_database(tmp_path / "projects" / project_id.root) as connection:
        migrate(connection)
        coordinator = RunCoordinator(connection, lambda: FIXED_TIME)
        coordinator.submit_start(
            StartRunRequest(project_id=project_id, run_id="run-plan-revision", idempotency_key="start")
        )
        lease = coordinator.claim_next("planning-worker")
        assert lease is not None
        brief = StoryBrief(
            brief_id="run-plan-revision:brief",
            project_id=project_id,
            user_requirements="A revision-domain regression.",
            genre="fiction",
            target_chapter_count=1,
            created_at=FIXED_TIME,
            provenance=offline_provenance("run-plan-revision", FIXED_TIME),
        )
        planning = PlanningWorkflow(
            coordinator,
            OfflinePlanningGenerator(FIXED_TIME),
            load_builtin_catalog(),
        ).execute(lease, brief)
        assert isinstance(planning, PlanningSucceeded)
        connection.execute("UPDATE runs SET plan_revision = 7 WHERE run_id = ?", (lease.run_id,))

        # When: the coordinator loads the current accepted plan by artifact authority.
        accepted_plan = coordinator.load_accepted_plan(lease, planning.plan_id)

        # Then: artifact revision one loads without pausing the run at plan revision seven.
        assert accepted_plan is not None
        assert accepted_plan.artifact_revision == 1
        assert accepted_plan.plan.plan_id == planning.plan_id
        assert connection.execute(
            "SELECT state FROM runs WHERE run_id = ?",
            (lease.run_id,),
        ).fetchone() == (RunStatus.RUNNING.value,)


def test_from_config_runs_explicit_offline_provider_to_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the canonical configured service pointed at a fresh project root.
    projects_root = tmp_path / "projects"
    project_id = ProjectId("novel-one")
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME)

    # When: an explicitly configured offline run enters through the service boundary.
    service.start(
        StartNovelRunRequest(
            project_id=project_id,
            run_id="run-from-config",
            idempotency_key="start-from-config",
            user_requirements="A canonical offline mystery.",
            provider="offline_fake",
        )
    )
    deadline = monotonic() + 2.0
    terminal_states = frozenset(
        (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED)
    )
    while (snapshot := service.snapshot(project_id, "run-from-config")).status not in terminal_states:
        assert monotonic() < deadline
        sleep(0.01)

    # Then: the real worker committed the accepted chapter and M3 memory.
    assert snapshot.status is RunStatus.SUCCEEDED
    with connect_project_database(projects_root / project_id.root) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND status = 'accepted'",
            ("run-from-config",),
        ).fetchone() == (4,)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE run_id = ? AND layer = 'm3_confirmed'",
            ("run-from-config",),
        ).fetchone() == (1,)


def test_configured_offline_stage_handles_cancelled_chapter_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a configured stage whose governed chapter workflow reports cancellation.
    project_id = ProjectId("novel-one")
    project_root = tmp_path / "projects"
    with connect_project_database(project_root / project_id.root) as connection:
        migrate(connection)
        coordinator = RunCoordinator(connection, lambda: FIXED_TIME)
        coordinator.submit_start(
            StartRunRequest(
                project_id=project_id,
                run_id="run-offline-cancelled",
                idempotency_key="start-offline-cancelled",
            )
        )
        lease = coordinator.claim_next("offline-worker")
        assert lease is not None

        def cancelled_execute(
            _workflow,
            _lease,
            _request,
            _chapter,
            _next_chapter,
            _task_id,
        ) -> ChapterExecutionCancelled:
            return ChapterExecutionCancelled("cancelled-task")

        monkeypatch.setattr("novel_agent.domains.novel.execute_single_chapter", cancelled_execute)
        stage = default_stage_executor_factory(
            coordinator,
            StartNovelRunRequest(
                project_id=project_id,
                run_id="run-offline-cancelled",
                idempotency_key="start-offline-cancelled",
                user_requirements="Cancel after planning.",
                provider="offline_fake",
            ),
            path_resolver=ProjectPathResolver(project_root),
            now=lambda: FIXED_TIME,
        )

        # When: the cancellation variant crosses the default-stage boundary.
        stage.execute(coordinator.resume_state(lease))

        # Then: cancellation is handled as a valid outcome instead of asserting.
        assert connection.execute(
            "SELECT state FROM runs WHERE run_id = ?",
            ("run-offline-cancelled",),
        ).fetchone() == (RunStatus.RUNNING.value,)
