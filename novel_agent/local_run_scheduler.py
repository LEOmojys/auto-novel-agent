from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from threading import Lock, Thread

from pydantic import ValidationError

from novel_agent.contracts import RunCommandType, RunStatus
from novel_agent.queued_run_failure import QueuedRunFailureReason, fail_queued_run
from novel_agent.pathing import ProjectId
from novel_agent.run_coordinator import RunCoordinator
from novel_agent.run_models import StartRunExecutionPayload
from novel_agent.run_service import RunService
from novel_agent.service_models import RunStageExecutorFactory, StartNovelRunRequest


class LocalRunScheduler:
    def __init__(
        self,
        *,
        project_ids: Callable[[], tuple[ProjectId, ...]],
        open_connection: Callable[[ProjectId], AbstractContextManager[sqlite3.Connection]],
        now: Callable[[], datetime],
        stage_executor_factory: RunStageExecutorFactory,
        stage_timeout_seconds: float,
    ) -> None:
        self._project_ids = project_ids
        self._open_connection = open_connection
        self._now = now
        self._stage_executor_factory = stage_executor_factory
        self._stage_timeout_seconds = stage_timeout_seconds
        self._drain_lock = Lock()
        self._pending_lock = Lock()
        self._pending_requests: dict[tuple[str, str], StartNovelRunRequest] = {}
        self._recover_queued_requests()

    def enqueue(self, request: StartNovelRunRequest) -> None:
        with self._pending_lock:
            self._pending_requests[(request.project_id.root, request.run_id)] = request
        self._schedule_drain()

    def _recover_queued_requests(self) -> None:
        for project_id in self._project_ids():
            with self._open_connection(project_id) as connection:
                rows = connection.execute(
                    "SELECT r.run_id, c.idempotency_key, c.payload_json "
                    "FROM runs r JOIN commands c ON c.command_id = ("
                    "  SELECT command_id FROM commands "
                    "  WHERE run_id = r.run_id AND command_type = ? "
                    "  ORDER BY created_at DESC, rowid DESC LIMIT 1"
                    ") LEFT JOIN run_leases l ON l.run_id = r.run_id "
                    "WHERE r.project_id = ? AND "
                    "(r.state = ? OR (r.state = ? AND l.expires_at <= ?))",
                    (
                        RunCommandType.START.value,
                        project_id.root,
                        RunStatus.QUEUED.value,
                        RunStatus.RUNNING.value,
                        self._now().isoformat(),
                    ),
                ).fetchall()
            for run_id, idempotency_key, payload_json in rows:
                try:
                    payload = StartRunExecutionPayload.model_validate_json(str(payload_json))
                    request = StartNovelRunRequest.model_validate(payload.model_dump())
                    if (
                        request.project_id != project_id
                        or request.run_id != str(run_id)
                        or request.idempotency_key != str(idempotency_key)
                    ):
                        continue
                except (ValueError, TypeError, ValidationError, json.JSONDecodeError):
                    continue
                with self._pending_lock:
                    self._pending_requests[(project_id.root, request.run_id)] = request
        self._schedule_drain()

    def _schedule_drain(self) -> None:
        if not self._drain_lock.acquire(blocking=False):
            return

        def drain() -> None:
            try:
                while True:
                    with self._pending_lock:
                        if not self._pending_requests:
                            break
                        request_key = next(iter(self._pending_requests))
                        request = self._pending_requests.pop(request_key)
                    try:
                        with self._open_connection(request.project_id) as connection:
                            coordinator = RunCoordinator(connection, self._now)
                            try:
                                executor = self._stage_executor_factory(coordinator, request)
                            except Exception:  # noqa: BROAD_EXCEPT_OK
                                fail_queued_run(
                                    connection,
                                    self._now,
                                    request.run_id,
                                    QueuedRunFailureReason.STAGE_EXECUTOR_FACTORY_FAILED,
                                )
                                continue
                            try:
                                outcome = RunService(
                                    coordinator,
                                    stage_executor=executor,
                                    stage_timeout_seconds=self._stage_timeout_seconds,
                                ).run_once("novel-agent-local-worker", request.run_id)
                            except Exception:  # noqa: BROAD_EXCEPT_OK - a drain failure must never strand a queued run.
                                fail_queued_run(
                                    connection,
                                    self._now,
                                    request.run_id,
                                    QueuedRunFailureReason.SCHEDULER_DRAIN_FAILED,
                                )
                                continue
                            if outcome is None and self._is_still_queued(connection, request.run_id):
                                with self._pending_lock:
                                    self._pending_requests[request_key] = request
                                time.sleep(0.05)
                            elif outcome is not None:
                                # The run reached a terminal state: keep the
                                # legacy-format files in sync with SQLite.
                                self._sync_exports(connection, request.project_id)
                    except Exception:  # noqa: BROAD_EXCEPT_OK - connection failures must not strand the queue.
                        self._retry_connection_failure(request)
            finally:
                self._drain_lock.release()
                with self._pending_lock:
                    has_pending = bool(self._pending_requests)
                if has_pending:
                    self._schedule_drain()

        Thread(target=drain, name="novel-agent-local-worker", daemon=True).start()

    def _retry_connection_failure(self, request: StartNovelRunRequest) -> None:
        """Requeue a run whose project database could not be opened.

        A locked or momentarily unavailable SQLite file must not strand the
        run in QUEUED forever; keep the request pending and retry with a short
        backoff so a transient lock clears.  Permanent database corruption
        requires operator intervention (repair or restart).
        """
        with self._pending_lock:
            self._pending_requests[(request.project_id.root, request.run_id)] = request
        time.sleep(0.2)

    @staticmethod
    def _sync_exports(connection: sqlite3.Connection, project_id: ProjectId) -> None:
        """Drain the artifact outbox, then materialize legacy-format files.

        Both are best-effort derived views over authoritative SQLite state;
        failures never break the drain loop (the console exposes manual
        export/materialization entry points).
        """
        try:
            from novel_agent.artifacts import ArtifactService
            from novel_agent.pathing import ProjectPathResolver

            ArtifactService(connection, ProjectPathResolver.from_config()).materialize_pending()
        except Exception:  # noqa: BROAD_EXCEPT_OK - outbox drain is best-effort
            pass
        try:
            from novel_agent.export import export_project

            export_project(connection, project_id)
        except Exception:  # noqa: BROAD_EXCEPT_OK - export is a best-effort compatibility view
            return

    @staticmethod
    def _is_still_queued(connection: sqlite3.Connection, run_id: str) -> bool:
        state_row = connection.execute("SELECT state FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return state_row is not None and str(state_row[0]) == RunStatus.QUEUED.value


__all__ = ("LocalRunScheduler",)
