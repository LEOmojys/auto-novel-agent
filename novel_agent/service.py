from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import config
from pydantic import ValidationError

from novel_agent.audit import (
    ArtifactView,
    AuditManifestView,
    QualityReportView,
    RunProjectAuditError,
    read_artifact,
    read_artifacts,
    read_audit_manifests,
    read_quality_reports,
)
from novel_agent.contracts import RunCommandType, RunStatus
from novel_agent.default_stage import default_stage_executor_factory
from novel_agent.local_run_scheduler import LocalRunScheduler
from novel_agent.pathing import ArtifactId, ProjectId, ProjectPathResolver
from novel_agent.revision_models import RebaseConfirmation
from novel_agent.run_coordinator import RunCoordinator
from novel_agent.run_event_store import append_run_event
from novel_agent.run_event_view import RunEventView, read_run_events, review_context
from novel_agent.run_models import ControlRunRequest, RunLease, RunTransitionError, StartRunExecutionPayload, StartRunRequest
from novel_agent.run_state import RunEvent
from novel_agent.service_errors import (
    RebaseWorkerUnavailableError,
    RunNotFoundError,
    RunRepairUnavailableError,
    StageTimeoutConfigurationError,
)
from novel_agent.service_models import (
    ApproveArtifactRequest,
    ControlNovelRunRequest,
    RegenerateArtifactRequest,
    RepairNovelRunRequest,
    ReviseChapterRequest,
    RunStageExecutorFactory,
    RunUiSnapshot,
    StartNovelRunRequest,
)
from novel_agent.storage import connect_project_database, migrate
from novel_agent.workflow_view import WorkflowView, build_workflow_view

if TYPE_CHECKING:
    from novel_agent.provider_types import ProviderTransport


class NovelAgentService:
    def __init__(
        self,
        path_resolver: ProjectPathResolver,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        stage_executor_factory: RunStageExecutorFactory | None = None,
        stage_timeout_seconds: float = 30.0,
    ) -> None:
        if stage_timeout_seconds <= 0.0:
            raise StageTimeoutConfigurationError(stage_timeout_seconds)
        self._path_resolver = path_resolver
        self._now = now
        self._scheduler = (
            None
            if stage_executor_factory is None
            else LocalRunScheduler(
                project_ids=self.project_ids,
                open_connection=self._connection,
                now=now,
                stage_executor_factory=stage_executor_factory,
                stage_timeout_seconds=stage_timeout_seconds,
            )
        )

    @classmethod
    def from_config(
        cls,
        *,
        now: Callable[[], datetime] | None = None,
        stage_executor_factory: RunStageExecutorFactory | None = None,
        provider_transport: ProviderTransport | None = None,
    ) -> NovelAgentService:
        resolver = ProjectPathResolver.from_config()
        clock = now or (lambda: datetime.now(UTC))
        factory = stage_executor_factory
        if factory is None:
            factory = lambda coordinator, request: default_stage_executor_factory(
                coordinator,
                request,
                path_resolver=resolver,
                now=clock,
                provider_transport=provider_transport,
            )
        timeout = float(os.environ.get("NOVEL_AGENT_STAGE_TIMEOUT_SECONDS", "30.0"))
        return cls(resolver, now=clock, stage_executor_factory=factory, stage_timeout_seconds=timeout)

    def project_ids(self) -> tuple[ProjectId, ...]:
        root = self._path_resolver.projects_root
        candidates = (entry.name for entry in root.iterdir() if entry.is_dir()) if root.is_dir() else ()
        project_ids: list[ProjectId] = []
        for candidate in candidates:
            try:
                project_ids.append(ProjectId(candidate))
            except ValidationError:
                continue
        default = ProjectId(config.DEFAULT_PROJECT)
        return tuple(sorted(set(project_ids + [default]), key=lambda item: item.root))

    def start(self, request: StartNovelRunRequest) -> RunUiSnapshot:
        requirement_sha256 = hashlib.sha256(request.user_requirements.encode()).hexdigest()
        with self._connection(request.project_id) as connection:
            coordinator = RunCoordinator(connection, self._now)
            receipt = coordinator.submit_start(
                StartRunRequest(
                    project_id=request.project_id,
                    run_id=request.run_id,
                    idempotency_key=request.idempotency_key,
                    brief_sha256=requirement_sha256,
                    execution_payload=StartRunExecutionPayload(
                        project_id=request.project_id,
                        run_id=request.run_id,
                        idempotency_key=request.idempotency_key,
                        user_requirements=request.user_requirements,
                        provider=request.provider,
                        target_chapter_count=request.target_chapter_count,
                        auto_approve=request.auto_approve,
                    ),
                )
            )
            snapshot = self._snapshot(connection, request.project_id, request.run_id, receipt.command_id, receipt.reused)
        if self._scheduler is not None:
            self._scheduler.enqueue(request)
        return snapshot

    def start_auto(
        self,
        project_id: ProjectId,
        user_requirements: str,
        *,
        provider: str = "legacy_real",
        target_chapter_count: int | None = None,
        auto_approve: bool = True,
    ) -> RunUiSnapshot:
        """Start a run without an explicit run_id (multi-turn console mode).

        The run id is assigned automatically:
        - if the project's latest run is terminal AND its stored requirements
          match the submitted ones, that run_id is reused so a retry after a
          failure/cancellation (or a re-run after success) requeues the same
          execution cycle instead of spawning a look-alike;
        - if the project's latest run is still ACTIVE (queued/running/...)
          with the same requirements, the run is already being executed — a
          duplicate submission (double-click) returns its snapshot instead of
          allocating a second run;
        - otherwise the next ``{project}-run-{N}`` id is allocated, so each
          new requirement gets its own auditable cycle while project-level
          asset reuse keeps carrying the story forward.
        """
        latest_row: tuple | None
        suffix = 0
        with self._connection(project_id) as connection:
            latest_row = connection.execute(
                "SELECT r.run_id, r.state, c.payload_json FROM runs r "
                "LEFT JOIN commands c ON c.command_id = ("
                "  SELECT command_id FROM commands "
                "  WHERE run_id = r.run_id AND command_type = 'start' "
                "  ORDER BY created_at DESC, rowid DESC LIMIT 1"
                ") WHERE r.project_id = ? ORDER BY r.created_at DESC, r.rowid DESC LIMIT 1",
                (project_id.root,),
            ).fetchone()
            for (existing_id,) in connection.execute(
                "SELECT run_id FROM runs WHERE project_id = ?", (project_id.root,)
            ).fetchall():
                match = re.match(rf"^{re.escape(project_id.root)}-run-(\d+)$", str(existing_id))
                if match is not None:
                    suffix = max(suffix, int(match.group(1)))
        if latest_row is not None:
            latest_run_id = str(latest_row[0])
            latest_state = RunStatus(str(latest_row[1]))
            payload_json = latest_row[2]
            previous: StartRunExecutionPayload | None = None
            if payload_json is not None:
                try:
                    previous = StartRunExecutionPayload.model_validate_json(str(payload_json))
                except (ValidationError, ValueError, TypeError, json.JSONDecodeError):
                    previous = None
            same_requirements = previous is not None and previous.user_requirements == user_requirements
            if latest_state in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED):
                if same_requirements:
                    # Explicit retry of the finished cycle: requeue the same run.
                    return self.start(
                        StartNovelRunRequest(
                            project_id=project_id,
                            run_id=latest_run_id,
                            idempotency_key=f"auto-retry:{latest_run_id}:{self._now().isoformat()}",
                            user_requirements=user_requirements,
                            provider=provider,
                            target_chapter_count=target_chapter_count,
                            auto_approve=auto_approve,
                        )
                    )
            elif same_requirements:
                # The latest run is still active with the same requirements:
                # a duplicate submission (double-click) must NOT spawn a second
                # run — return the existing run's snapshot unchanged.
                return self.snapshot(project_id, latest_run_id)
        run_id = f"{project_id.root}-run-{suffix + 1}"
        return self.start(
            StartNovelRunRequest(
                project_id=project_id,
                run_id=run_id,
                idempotency_key=f"auto-start:{run_id}:{self._now().isoformat()}",
                user_requirements=user_requirements,
                provider=provider,
                target_chapter_count=target_chapter_count,
                auto_approve=auto_approve,
            )
        )

    def control(self, request: ControlNovelRunRequest) -> RunUiSnapshot:
        with self._connection(request.project_id) as connection:
            self._require_run_project(connection, request.project_id, request.run_id)
            receipt = RunCoordinator(connection, self._now).control(
                ControlRunRequest(
                    run_id=request.run_id,
                    command_type=request.command_type,
                    idempotency_key=request.idempotency_key,
                )
            )
            return self._snapshot(connection, request.project_id, request.run_id, receipt.command_id, receipt.reused)

    def repair(self, request: RepairNovelRunRequest) -> RunUiSnapshot:
        """Requeue a WAITING_REVIEW run so the stage re-executes it.

        Unlike approval flows (which have their own enqueue path), the plain
        REPAIR transition only flips the state back to QUEUED — the scheduler
        must be told so the run is actually picked up again, otherwise it
        would sit queued forever.
        """
        start_row: tuple | None = None
        with self._connection(request.project_id) as connection:
            self._require_run_project(connection, request.project_id, request.run_id)
            review_reason, review_event_id = review_context(connection, request.run_id)
            if review_event_id is None or review_reason == "chapter_rebase_confirmation_required":
                raise RunRepairUnavailableError(request.run_id, review_reason)
            receipt = RunCoordinator(connection, self._now).control(
                ControlRunRequest(
                    run_id=request.run_id,
                    command_type=RunCommandType.REPAIR,
                    idempotency_key=request.idempotency_key,
                )
            )
            snapshot = self._snapshot(connection, request.project_id, request.run_id, receipt.command_id, receipt.reused)
        if not receipt.reused:
            self._enqueue_from_start_command(request.project_id, request.run_id)
        return snapshot

    def _enqueue_from_start_command(self, project_id: ProjectId, run_id: str) -> None:
        """Requeue a run through the scheduler using its latest start payload.

        Shared by repair/revise/approval flows that return a run to the queue.
        A corrupt or missing start payload must never leave the run silently
        stuck in QUEUED: a ``run_failed`` event is recorded so the operator
        can see why it never executed.
        """
        if self._scheduler is None:
            return
        with self._connection(project_id) as connection:
            start_row = connection.execute(
                "SELECT payload_json FROM commands WHERE run_id = ? AND command_type = 'start' "
                "ORDER BY created_at DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if start_row is None:
            return
        try:
            start_payload = StartRunExecutionPayload.model_validate_json(str(start_row[0]))
            self._scheduler.enqueue(StartNovelRunRequest.model_validate(start_payload.model_dump()))
        except (ValidationError, ValueError, TypeError):
            with self._connection(project_id) as connection:
                append_run_event(
                    connection,
                    RunEvent(run_id, "run_failed", {"reason": "requeue_enqueue_failed"}, self._now()),
                )

    def revise_chapter(
        self,
        request: ReviseChapterRequest,
    ) -> RunUiSnapshot:
        """Request a rewrite of one chapter of a finished run.

        The run returns to the queue; the stage rebuilds a single-chapter
        graph, redrafts the target chapter as a non-activating revision, and
        proposes a rebase so the user can confirm (or reject) the swap.
        """
        receipt: CommandReceipt
        with self._connection(request.project_id) as connection:
            self._require_run_project(connection, request.project_id, request.run_id)
            row = connection.execute(
                "SELECT state FROM runs WHERE run_id = ?", (request.run_id,)
            ).fetchone()
            if row is None or str(row[0]) != RunStatus.SUCCEEDED.value:
                raise RunTransitionError(request.run_id, str(row[0]) if row else "missing", RunCommandType.REVISE.value)
            receipt = RunCoordinator(connection, self._now).control(
                ControlRunRequest(
                    run_id=request.run_id,
                    command_type=RunCommandType.REVISE,
                    idempotency_key=request.idempotency_key,
                    payload={"chapter_number": request.chapter_number},
                )
            )
            snapshot = self._snapshot(connection, request.project_id, request.run_id, receipt.command_id, receipt.reused)
            if receipt.reused:
                return snapshot
        # Requeue through the scheduler, restoring the original start payload
        # so the stage can run the rewrite graph for this run.
        self._enqueue_from_start_command(request.project_id, request.run_id)
        return snapshot

    def approve_artifact(self, request: ApproveArtifactRequest) -> RunUiSnapshot:
        """Approve a staged artifact: persist it to the canonical stores.

        When the approved artifact was the graph's final output the run
        finishes; otherwise it returns to the queue so the remaining graph
        nodes execute.
        """
        from novel_agent.domains.novel_commit import (
            commit_cast_only,
            commit_outline_only,
            commit_world_only,
        )

        receipt: CommandReceipt | None = None
        start_row: tuple | None = None
        snapshot: RunUiSnapshot
        with self._connection(request.project_id) as connection:
            self._require_run_project(connection, request.project_id, request.run_id)
            row = connection.execute(
                "SELECT raw_bytes, kind, project_id FROM artifacts "
                "WHERE artifact_id = ? AND revision = ? AND run_id = ? AND status = 'staged'",
                (request.artifact_id, request.revision, request.run_id),
            ).fetchone()
            if row is None:
                raise RunTransitionError(request.run_id, "no staged artifact", "approve")
            raw_bytes = bytes(row[0])
            kind = str(row[1])
            committed_at = self._now()
            brief_id = f"{request.run_id}:brief"
            is_final = _is_final_graph_output(connection, request.run_id, kind)
            if kind == "world_bible":
                from novel_agent.contracts import WorldBible

                world = WorldBible.model_validate_json(raw_bytes)
                from novel_agent.adapters import CreativeArtifactKind, CreativeParseRoute, RawCreativeArtifact, WorldAdaptation

                commit_world_only(
                    connection,
                    run_id=request.run_id,
                    project_id=request.project_id.root,
                    brief_id=brief_id,
                    world_result=WorldAdaptation(
                        raw=RawCreativeArtifact(
                            kind=CreativeArtifactKind.WORLD,
                            raw_bytes=raw_bytes,
                            content_sha256=_sha256(raw_bytes),
                            adapter_version="approval:v1",
                        ),
                        value=world,
                        parse_route=CreativeParseRoute.STRICT_JSON,
                    ),
                    committed_at=committed_at,
                    complete=False,
                )
            elif kind == "character_set":
                from novel_agent.contracts import CharacterSet
                from novel_agent.adapters import CharacterAdaptation, CreativeArtifactKind, CreativeParseRoute, RawCreativeArtifact

                commit_cast_only(
                    connection,
                    run_id=request.run_id,
                    project_id=request.project_id.root,
                    brief_id=brief_id,
                    cast_result=CharacterAdaptation(
                        raw=RawCreativeArtifact(
                            kind=CreativeArtifactKind.CHARACTERS,
                            raw_bytes=raw_bytes,
                            content_sha256=_sha256(raw_bytes),
                            adapter_version="approval:v1",
                        ),
                        value=CharacterSet.model_validate_json(raw_bytes),
                        parse_route=CreativeParseRoute.STRICT_JSON,
                    ),
                    committed_at=committed_at,
                    complete=False,
                )
            elif kind == "plan_contract":
                from novel_agent.contracts import PlanContract
                from novel_agent.adapters import CreativeArtifactKind, CreativeParseRoute, OutlineAdaptation, RawCreativeArtifact

                commit_outline_only(
                    connection,
                    run_id=request.run_id,
                    project_id=request.project_id.root,
                    brief_id=brief_id,
                    world_result=_existing_world_adaptation(connection, request.run_id, request.project_id),
                    cast_result=_existing_cast_adaptation(connection, request.run_id, request.project_id),
                    outline_result=OutlineAdaptation(
                        raw=RawCreativeArtifact(
                            kind=CreativeArtifactKind.OUTLINE,
                            raw_bytes=raw_bytes,
                            content_sha256=_sha256(raw_bytes),
                            adapter_version="approval:v1",
                        ),
                        value=PlanContract.model_validate_json(raw_bytes),
                        parse_route=CreativeParseRoute.STRICT_JSON,
                    ),
                    committed_at=committed_at,
                    complete=False,
                )
            elif kind == "chapter":
                _approve_chapter(
                    connection,
                    request.project_id,
                    request.run_id,
                    request.artifact_id,
                    request.revision,
                    committed_at,
                )
            else:
                raise RunTransitionError(request.run_id, f"unsupported staged kind: {kind}", "approve")
            connection.execute(
                "DELETE FROM artifacts WHERE artifact_id = ? AND revision = ? AND run_id = ? AND status = 'staged'",
                (request.artifact_id, request.revision, request.run_id),
            )
            if is_final:
                # Approved the graph's final output: finish the run.
                connection.execute(
                    "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
                    (RunStatus.SUCCEEDED.value, committed_at.isoformat(), request.run_id),
                )
                from novel_agent.run_event_store import append_run_event as _append_event
                from novel_agent.run_state import RunEvent as _RunEvent

                _append_event(
                    connection,
                    _RunEvent(request.run_id, "run_succeeded", {}, committed_at),
                )
                snapshot = self._snapshot(connection, request.project_id, request.run_id, None, False)
                return snapshot
            # Requeue the run so the remaining graph nodes execute; the
            # approved artifact is now visible to the reuse detection.
            receipt = RunCoordinator(connection, self._now).control(
                ControlRunRequest(
                    run_id=request.run_id,
                    command_type=RunCommandType.REPAIR,
                    idempotency_key=request.idempotency_key,
                )
            )
            snapshot = self._snapshot(connection, request.project_id, request.run_id, receipt.command_id, receipt.reused)
        self._enqueue_from_start_command(request.project_id, request.run_id)
        return snapshot

    def approve_plan(self, project_id: ProjectId, run_id: str, idempotency_key: str) -> RunUiSnapshot:
        """Confirm the generated plan and requeue the run to execute it.

        The pending plan proposal is marked approved; on the requeued run the
        stage loads the approved plan directly (no second model call) and
        builds the task graph from it.
        """
        receipt: CommandReceipt
        start_row: tuple | None = None
        with self._connection(project_id) as connection:
            self._require_run_project(connection, project_id, run_id)
            updated = connection.execute(
                "UPDATE commands SET state = 'approved' WHERE run_id = ? AND command_type = 'plan_proposed' "
                "AND state = 'proposed'",
                (run_id,),
            )
            if updated.rowcount != 1:
                raise RunTransitionError(run_id, "no pending plan proposal", "approve plan")
            receipt = RunCoordinator(connection, self._now).control(
                ControlRunRequest(
                    run_id=run_id,
                    command_type=RunCommandType.REPAIR,
                    idempotency_key=idempotency_key,
                )
            )
            snapshot = self._snapshot(connection, project_id, run_id, receipt.command_id, receipt.reused)
        self._enqueue_from_start_command(project_id, run_id)
        return snapshot

    def regenerate_plan(self, project_id: ProjectId, run_id: str, idempotency_key: str) -> RunUiSnapshot:
        """Discard the pending plan and requeue the run to replan it.

        The proposed-plan command and the planning tool ledger are cleared so
        the model is called again on the requeued run.
        """
        receipt: CommandReceipt
        start_row: tuple | None = None
        with self._connection(project_id) as connection:
            self._require_run_project(connection, project_id, run_id)
            connection.execute(
                "DELETE FROM commands WHERE run_id = ? AND command_type = 'plan_proposed'",
                (run_id,),
            )
            connection.execute(
                "DELETE FROM tool_calls WHERE run_id = ? AND tool_id IN "
                "('world.generate@1.0.0','character.generate@1.0.0','outline.generate@1.0.0','character.revise@1.0.0')",
                (run_id,),
            )
            receipt = RunCoordinator(connection, self._now).control(
                ControlRunRequest(
                    run_id=run_id,
                    command_type=RunCommandType.REPAIR,
                    idempotency_key=idempotency_key,
                )
            )
            snapshot = self._snapshot(connection, project_id, run_id, receipt.command_id, receipt.reused)
        self._enqueue_from_start_command(project_id, run_id)
        return snapshot

    def regenerate_artifact(self, request: RegenerateArtifactRequest) -> RunUiSnapshot:
        """Discard a staged artifact and requeue the run to regenerate it.

        The tool-idempotency ledger entries for the planning generators are
        cleared too, so the redraft actually calls the provider again instead
        of replaying the recorded result.
        """
        receipt: CommandReceipt
        start_row: tuple | None = None
        with self._connection(request.project_id) as connection:
            self._require_run_project(connection, request.project_id, request.run_id)
            kind_row = connection.execute(
                "SELECT kind FROM artifacts WHERE artifact_id = ? AND revision = ? AND run_id = ? AND status = 'staged'",
                (request.artifact_id, request.revision, request.run_id),
            ).fetchone()
            if kind_row is None:
                raise RunTransitionError(request.run_id, "no staged artifact", "regenerate")
            kind = str(kind_row[0])
            deleted = connection.execute(
                "DELETE FROM artifacts WHERE artifact_id = ? AND revision = ? AND run_id = ? AND status = 'staged'",
                (request.artifact_id, request.revision, request.run_id),
            )
            if deleted.rowcount != 1:
                raise RunTransitionError(request.run_id, "no staged artifact", "regenerate")
            _clear_tool_ledger_for_regenerate(connection, request.run_id, kind)
            receipt = RunCoordinator(connection, self._now).control(
                ControlRunRequest(
                    run_id=request.run_id,
                    command_type=RunCommandType.REPAIR,
                    idempotency_key=request.idempotency_key,
                )
            )
            snapshot = self._snapshot(connection, request.project_id, request.run_id, receipt.command_id, receipt.reused)
        self._enqueue_from_start_command(request.project_id, request.run_id)
        return snapshot

    def claim_next(self, project_id: ProjectId, holder_id: str) -> RunLease | None:
        with self._connection(project_id) as connection:
            return RunCoordinator(connection, self._now).claim_next(holder_id)

    def snapshot(self, project_id: ProjectId, run_id: str) -> RunUiSnapshot:
        with self._connection(project_id) as connection:
            return self._snapshot(connection, project_id, run_id, None, False)

    def events(self, project_id: ProjectId, run_id: str) -> tuple[RunEventView, ...]:
        with self._connection(project_id) as connection:
            self._require_run_project(connection, project_id, run_id)
            return read_run_events(connection, run_id)

    def artifact(
        self,
        project_id: ProjectId,
        run_id: str,
        artifact_id: str,
        revision: int,
    ) -> ArtifactView:
        with self._connection(project_id) as connection:
            return read_artifact(connection, project_id, run_id, ArtifactId(artifact_id), revision)

    def artifacts(self, project_id: ProjectId, run_id: str) -> tuple[ArtifactView, ...]:
        with self._connection(project_id) as connection:
            return read_artifacts(connection, project_id, run_id)

    def audit_manifests(self, project_id: ProjectId, run_id: str) -> tuple[AuditManifestView, ...]:
        with self._connection(project_id) as connection:
            return read_audit_manifests(connection, project_id, run_id)

    def quality_reports(self, project_id: ProjectId, run_id: str) -> tuple[QualityReportView, ...]:
        with self._connection(project_id) as connection:
            return read_quality_reports(connection, project_id, run_id)

    def provider_failures(self, project_id: ProjectId, run_id: str) -> tuple[dict[str, str], ...]:
        with self._connection(project_id) as connection:
            from novel_agent.provider_failure import read_provider_failures

            return read_provider_failures(connection, run_id)

    def workflow_view(self, project_id: ProjectId, run_id: str) -> WorkflowView:
        with self._connection(project_id) as connection:
            self._require_run_project(connection, project_id, run_id)
            return build_workflow_view(connection, run_id)

    def confirm_rebase(self, request: RebaseConfirmation, holder_id: str) -> RunUiSnapshot:
        with self._connection(request.project_id) as connection:
            self._require_run_project(connection, request.project_id, request.run_id)
            current = self._snapshot(connection, request.project_id, request.run_id, None, False)
            if (
                current.status is not RunStatus.WAITING_REVIEW
                or current.review_reason != "chapter_rebase_confirmation_required"
                or current.review_event_id is None
            ):
                raise RunRepairUnavailableError(request.run_id, current.review_reason)
            coordinator = RunCoordinator(connection, self._now)
            coordinator.control(
                ControlRunRequest(
                    run_id=request.run_id,
                    command_type=RunCommandType.REPAIR,
                    idempotency_key=f"rebase-repair:{request.proposal_id}",
                )
            )
            lease = coordinator.claim_run(request.run_id, holder_id)
            if lease is None:
                raise RebaseWorkerUnavailableError(request.run_id)
            coordinator.confirm_chapter_rebase(lease, request)
            return self._snapshot(connection, request.project_id, request.run_id, None, False)

    @contextmanager
    def _connection(self, project_id: ProjectId) -> Iterator[sqlite3.Connection]:
        project_dir = self._path_resolver.resolve_project(project_id)
        with connect_project_database(project_dir) as connection:
            migrate(connection)
            yield connection

    def _snapshot(
        self,
        connection: sqlite3.Connection,
        project_id: ProjectId,
        run_id: str,
        command_id: str | None,
        reused: bool,
    ) -> RunUiSnapshot:
        self._require_run_project(connection, project_id, run_id)
        run = connection.execute("SELECT state FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        active = connection.execute(
            "SELECT task_id FROM tasks WHERE run_id = ? AND state = 'running' ORDER BY updated_at DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        count_row = connection.execute("SELECT COUNT(*) FROM run_events WHERE run_id = ?", (run_id,)).fetchone()
        reason, review_event_id = review_context(connection, run_id)
        return RunUiSnapshot(
            project_id=project_id,
            run_id=run_id,
            status=RunStatus(str(run[0])),
            active_task_id=None if active is None else str(active[0]),
            review_reason=reason,
            review_event_id=review_event_id,
            event_count=int(count_row[0]),
            command_id=command_id,
            reused=reused,
        )

    @staticmethod
    def _require_run_project(connection: sqlite3.Connection, project_id: ProjectId, run_id: str) -> None:
        row = connection.execute("SELECT project_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None or str(row[0]) != project_id.root:
            raise RunNotFoundError(project_id, run_id)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _existing_world_adaptation(connection: sqlite3.Connection, run_id: str, project_id: ProjectId):
    """Reconstruct the accepted world adaptation to back a plan commit.

    The world may come from any run of the project (an outline-only run
    reuses assets approved by an earlier run) or, failing that, from the
    legacy world files — mirroring the node executor's resolve fallback.
    """
    from novel_agent.adapters import CreativeArtifactKind, CreativeParseRoute, RawCreativeArtifact, WorldAdaptation
    from novel_agent.contracts import Provenance, ProvenanceKind, WorldBible

    row = connection.execute(
        "SELECT raw_bytes FROM artifacts WHERE project_id = ? AND kind = 'world_bible' "
        "AND status = 'accepted' AND is_current = 1 ORDER BY created_at DESC, revision DESC LIMIT 1",
        (project_id.root,),
    ).fetchone()
    if row is not None:
        raw_bytes = bytes(row[0])
        return WorldAdaptation(
            raw=RawCreativeArtifact(
                kind=CreativeArtifactKind.WORLD,
                raw_bytes=raw_bytes,
                content_sha256=_sha256(raw_bytes),
                adapter_version="approval:v1",
            ),
            value=WorldBible.model_validate_json(raw_bytes),
            parse_route=CreativeParseRoute.STRICT_JSON,
        )
    # Fall back to legacy world files (outline-only runs often have no
    # accepted world artifact of their own yet).
    from novel_agent.domains.novel import detect_existing_assets
    from novel_agent.pathing import ProjectPathResolver

    project_root = ProjectPathResolver.from_config().resolve_project(project_id)
    assets = detect_existing_assets(project_root, connection=connection)
    world_text = assets.get("world_text")
    if not world_text:
        raise RunTransitionError(run_id, "world not approved yet", "approve plan_contract")
    world = WorldBible(
        world_id=f"{run_id}:existing-world",
        brief_id=f"{run_id}:brief",
        title=str(assets.get("world_title") or "world"),
        premise=str(world_text),
        rules=(),
        constraints=(),
        created_at=datetime.now(UTC),
        provenance=Provenance(
            source_kind=ProvenanceKind.LEGACY_IMPORT,
            source_id=run_id,
            source_sha256=_sha256(str(world_text).encode()),
            observed_at=datetime.now(UTC),
        ),
    )
    payload = world.model_dump_json().encode()
    return WorldAdaptation(
        raw=RawCreativeArtifact(
            kind=CreativeArtifactKind.WORLD,
            raw_bytes=payload,
            content_sha256=_sha256(payload),
            adapter_version="existing-files-v1",
        ),
        value=world,
        parse_route=CreativeParseRoute.LEGACY_NORMALIZATION,
    )


def _existing_cast_adaptation(connection: sqlite3.Connection, run_id: str, project_id: ProjectId):
    """Reconstruct the accepted cast adaptation to back a plan commit.

    Same project-level lookup as the world: an outline-only run reuses
    characters approved by any run of the project, or from the legacy files.
    """
    from novel_agent.adapters import CharacterAdaptation, CreativeArtifactKind, CreativeParseRoute, RawCreativeArtifact
    from novel_agent.contracts import CharacterCard, CharacterSet, Provenance, ProvenanceKind

    row = connection.execute(
        "SELECT raw_bytes FROM artifacts WHERE project_id = ? AND kind = 'character_set' "
        "AND status = 'accepted' AND is_current = 1 ORDER BY created_at DESC, revision DESC LIMIT 1",
        (project_id.root,),
    ).fetchone()
    if row is not None:
        raw_bytes = bytes(row[0])
        return CharacterAdaptation(
            raw=RawCreativeArtifact(
                kind=CreativeArtifactKind.CHARACTERS,
                raw_bytes=raw_bytes,
                content_sha256=_sha256(raw_bytes),
                adapter_version="approval:v1",
            ),
            value=CharacterSet.model_validate_json(raw_bytes),
            parse_route=CreativeParseRoute.STRICT_JSON,
        )
    # Fall back to legacy character files (outline-only runs often have no
    # accepted cast artifact of their own yet).
    from novel_agent.domains.novel import detect_existing_assets
    from novel_agent.pathing import ProjectPathResolver

    project_root = ProjectPathResolver.from_config().resolve_project(project_id)
    assets = detect_existing_assets(project_root, connection=connection)
    cards_raw = assets.get("characters")
    if not cards_raw:
        raise RunTransitionError(run_id, "cast not approved yet", "approve plan_contract")
    timestamp = datetime.now(UTC)
    cards = tuple(
        CharacterCard(
            character_id=f"{run_id}:existing-character:{index}",
            name=str(item.get("name", "")),
            role=str(item.get("role_introduction") or ""),
            goal=str(item.get("role_experience") or ""),
            traits=(),
            created_at=timestamp,
            provenance=Provenance(
                source_kind=ProvenanceKind.LEGACY_IMPORT,
                source_id=run_id,
                source_sha256=_sha256(str(item.get("name", "")).encode()),
                observed_at=timestamp,
            ),
        )
        for index, item in enumerate(cards_raw, start=1)
        if item.get("name")
    )
    cast = CharacterSet(
        character_set_id=f"{run_id}:existing-cast",
        world_id="",
        characters=cards,
        created_at=timestamp,
        provenance=Provenance(
            source_kind=ProvenanceKind.LEGACY_IMPORT,
            source_id=run_id,
            source_sha256=_sha256("".join(card.name for card in cards).encode()),
            observed_at=timestamp,
        ),
    )
    payload = cast.model_dump_json().encode()
    return CharacterAdaptation(
        raw=RawCreativeArtifact(
            kind=CreativeArtifactKind.CHARACTERS,
            raw_bytes=payload,
            content_sha256=_sha256(payload),
            adapter_version="existing-files-v1",
        ),
        value=cast,
        parse_route=CreativeParseRoute.LEGACY_NORMALIZATION,
    )


def _approve_chapter(
    connection: sqlite3.Connection,
    project_id: ProjectId,
    run_id: str,
    artifact_id: str,
    revision: int,
    committed_at: datetime,
) -> None:
    """Approve a staged chapter: persist it as the current accepted revision.

    The staged chapter already passed the quality gate and its memory
    candidates were staged; approval flips the artifact to accepted/current
    and promotes the candidates to the M3 confirmed layer.
    """
    import hashlib as _hashlib

    from novel_agent.chapter_commit_models import ChapterCommitContext
    from novel_agent.chapter_models import StagedChapter
    from novel_agent.contracts import PlanContract
    from novel_agent.memory_models import MemoryCandidateRecord
    from novel_agent.memory_policy import decide_promotion
    from novel_agent.memory_store import build_evidence, load_source, memory_id, record_status
    from novel_agent.pathing import ArtifactId

    row = connection.execute(
        "SELECT raw_bytes, run_id FROM artifacts WHERE artifact_id = ? AND revision = ? "
        "AND kind = 'chapter' AND status = 'staged'",
        (artifact_id, revision),
    ).fetchone()
    if row is None:
        raise RunTransitionError(run_id, "staged chapter not found", "approve chapter")
    raw_bytes = bytes(row[0])
    plan_row = connection.execute(
        "SELECT raw_bytes FROM artifacts WHERE project_id = ? AND run_id = ? AND kind = 'plan_contract' "
        "AND status = 'accepted' AND is_current = 1 ORDER BY revision DESC LIMIT 1",
        (project_id.root, run_id),
    ).fetchone()
    if plan_row is None:
        raise RunTransitionError(run_id, "no accepted plan for staged chapter", "approve chapter")
    plan = PlanContract.model_validate_json(bytes(plan_row[0]))
    # The staged chapter is either a JSON draft envelope (offline/older data)
    # or the verbatim prose text (governed adapter keeps the model output
    # as-is).  Resolve the chapter plan from the artifact id when the raw is
    # not JSON: stage_chapter derives it as sha256(f"{run_id}:{plan_id}")[:24].
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    draft: dict[str, object]
    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict) or not parsed.get("chapter_plan_id"):
            raise ValueError("not a draft envelope")
        draft = parsed
    except (json.JSONDecodeError, ValueError):
        draft = {}
    chapter_plan_id: str
    if draft.get("chapter_plan_id"):
        chapter_plan_id = str(draft["chapter_plan_id"])
    else:
        suffix = artifact_id.removeprefix("chapter-")
        chapter_plan_id = next(
            (
                candidate.chapter_plan_id
                for candidate in plan.chapters
                if _hashlib.sha256(f"{run_id}:{candidate.chapter_plan_id}".encode()).hexdigest()[:24] == suffix
            ),
            "",
        )
    chapter = next((c for c in plan.chapters if c.chapter_plan_id == chapter_plan_id), None)
    if chapter is None:
        raise RunTransitionError(
            run_id,
            f"chapter plan missing for staged artifact {artifact_id}",
            "approve chapter",
        )
    task_id = f"{run_id}:chapter:{chapter_plan_id}"
    staged = StagedChapter(
        artifact_id=ArtifactId(artifact_id),
        revision=revision,
        task_id=task_id,
        chapter_plan_id=chapter_plan_id,
        chapter_number=chapter.chapter_number,
        raw_bytes=raw_bytes,
        raw_sha256=_hashlib.sha256(raw_bytes).hexdigest(),
        # Plain-text raw: the whole prose is the body.  JSON envelope: use its
        # body field (an empty body stays empty — never fall back to the
        # envelope text itself).
        body=str(draft.get("body") or (raw_text if not draft else "")),
        adapter_version=str(draft.get("adapter_version") or "approval:v1"),
        parse_route=str(draft.get("parse_route") or "approval"),
        generation_attempt=int(draft.get("generation_attempt") or 1),
    )
    # Persist the staged revision as the current accepted chapter.
    updated = connection.execute(
        "UPDATE artifacts SET status = 'accepted', is_current = 1 WHERE artifact_id = ? AND revision = ? "
        "AND status = 'staged'",
        (artifact_id, revision),
    )
    if updated.rowcount != 1:
        raise RunTransitionError(run_id, "staged chapter no longer eligible", "approve chapter")
    connection.execute(
        "UPDATE artifacts SET is_current = 0 WHERE artifact_id = ? AND revision <> ?",
        (artifact_id, revision),
    )
    # Promote staged memory candidates to the confirmed M3 layer.
    context = ChapterCommitContext(
        connection,
        ProjectPathResolver.from_config(),
        committed_at,
        fencing_token=_current_fence(connection, run_id),
    )
    candidates = connection.execute(
        "SELECT candidate_id, candidate_json FROM memory_candidates WHERE run_id = ? "
        "AND source_artifact_id = ? AND source_artifact_revision = ? ORDER BY candidate_id",
        (run_id, artifact_id, revision),
    ).fetchall()
    for candidate_id, candidate_json in candidates:
        candidate = MemoryCandidateRecord.model_validate_json(str(candidate_json))
        evidence = build_evidence(candidate.evidence.proposal, load_source(connection, candidate.evidence.proposal))
        outcome = decide_promotion(evidence, _has_memory_conflict(connection, evidence))
        if outcome.outcome.value != "promoted":
            connection.execute(
                "UPDATE memory_candidates SET status = ? WHERE candidate_id = ?",
                (outcome.candidate_status.value, candidate_id),
            )
            continue
        proposal = evidence.proposal
        identifier = memory_id(candidate_id)
        connection.execute(
            "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, "
            "subject_id, predicate, value_json, chapter_start, chapter_end, source_artifact_id, "
            "source_artifact_revision, source_span, source_hash, confidence, created_by, created_at, supersedes_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                identifier,
                proposal.project_id.root,
                proposal.run_id,
                proposal.plan_revision,
                proposal.target_layer.value,
                proposal.kind.value,
                record_status(proposal.target_layer),
                proposal.subject_id,
                proposal.predicate,
                proposal.value_json,
                chapter.chapter_number,
                chapter.chapter_number,
                proposal.source_artifact_id.root,
                proposal.source_artifact_revision,
                f"{evidence.source_byte_start}:{evidence.source_byte_end}",
                evidence.source_span_sha256,
                proposal.confidence,
                "chapter-approval:v1",
                committed_at.isoformat(),
                proposal.supersedes_memory_id,
            ),
        )
        connection.execute(
            "INSERT INTO memory_fts(memory_id, content) VALUES (?, ?)",
            (identifier, proposal.value_json),
        )
        connection.execute(
            "UPDATE memory_candidates SET status = 'promoted' WHERE candidate_id = ?",
            (candidate_id,),
        )
    connection.execute(
        "UPDATE tasks SET state = 'succeeded', updated_at = ? WHERE task_id = ? AND run_id = ?",
        (committed_at.isoformat(), task_id, run_id),
    )


def _has_memory_conflict(connection: sqlite3.Connection, evidence) -> bool:
    proposal = evidence.proposal
    if proposal.subject_id is None:
        return False
    rows = connection.execute(
        "SELECT value_json FROM memory_records current WHERE project_id = ? AND layer = ? "
        "AND subject_id = ? AND predicate = ? AND NOT EXISTS (SELECT 1 FROM memory_records "
        "successor WHERE successor.supersedes_id = current.memory_id)",
        (proposal.project_id.root, proposal.target_layer.value, proposal.subject_id, proposal.predicate),
    ).fetchall()
    return any(str(row[0]) != proposal.value_json for row in rows)


def _current_fence(connection: sqlite3.Connection, run_id: str) -> int:
    row = connection.execute(
        "SELECT fencing_token FROM run_leases WHERE run_id = ? ORDER BY fencing_token DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return int(row[0]) if row is not None else 1


def _clear_tool_ledger_for_regenerate(
    connection: sqlite3.Connection,
    run_id: str,
    kind: str,
) -> None:
    """Drop the idempotency ledger rows so a regenerated artifact is redrafted.

    The planning tool calls are keyed by (run_id, idempotency_key); without
    clearing them the gateway would replay the recorded provider result and
    the regeneration would never hit the live model.
    """
    kind_to_tool = {
        "world_bible": "world.generate@1.0.0",
        "character_set": "character.generate@1.0.0",
        "plan_contract": "outline.generate@1.0.0",
        "chapter": "chapter.draft@1.0.0",
    }
    tool = kind_to_tool.get(kind)
    if tool is None:
        return
    connection.execute(
        "DELETE FROM tool_calls WHERE run_id = ? AND tool_id = ?",
        (run_id, tool),
    )


def _is_final_graph_output(
    connection: sqlite3.Connection,
    run_id: str,
    kind: str,
) -> bool:
    """True when the staged kind corresponds to the graph's final output node.

    Approving the final output finishes the run; approving an intermediate
    output requeues it to continue the remaining nodes.
    """
    kind_to_node = {
        "world_bible": "world",
        "character_set": "cast",
        "plan_contract": "outline",
    }
    node_name = kind_to_node.get(kind)
    if kind == "chapter":
        node_name = None  # chapters are always the tail when present
        return True
    if node_name is None:
        return False
    row = connection.execute(
        "SELECT payload_json FROM run_events WHERE run_id = ? AND event_type = 'task_graph_built' "
        "ORDER BY sequence DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if row is None:
        return False
    try:
        payload = json.loads(str(row[0]))
        nodes = [node for node in str(payload.get("nodes", "")).split(",") if node]
    except (json.JSONDecodeError, AttributeError):
        return False
    return bool(nodes) and nodes[-1] == node_name


__all__ = (
    "ApproveArtifactRequest",
    "ControlNovelRunRequest",
    "NovelAgentService",
    "RegenerateArtifactRequest",
    "RepairNovelRunRequest",
    "RebaseWorkerUnavailableError",
    "RunNotFoundError",
    "RunEventView",
    "RunRepairUnavailableError",
    "RunUiSnapshot",
    "RunStageExecutorFactory",
    "StageTimeoutConfigurationError",
    "StartNovelRunRequest",
)