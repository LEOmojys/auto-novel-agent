from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Callable, Protocol

from novel_agent.chapter_commit import commit_chapter_acceptance
from novel_agent.chapter_commit_models import ChapterCommitContext
from novel_agent.chapter_errors import ChapterCommitBlocked, ChapterPersistenceError, ChapterTaskStateError
from novel_agent.chapter_models import (
    ChapterAcceptanceBlocked,
    ChapterAcceptanceReceipt,
    ChapterAcceptanceRequest,
    ChapterCandidateStageRequest,
    ChapterPauseRequest,
    ChapterQualityReportRequest,
    ChapterStageRequest,
    StagedChapter,
)
from novel_agent.chapter_staging import record_quality, stage_candidates, stage_chapter
from novel_agent.contracts import RunStatus
from novel_agent.pathing import ProjectId, ProjectPathResolver
from novel_agent.run_event_store import append_run_event
from novel_agent.run_models import ActiveTaskError, RunLease, RunLeaseError
from novel_agent.run_state import RunEvent
from novel_agent.storage import write_transaction


class _ChapterLeaseAuthority(Protocol):
    _connection: sqlite3.Connection
    _now: Callable[[], datetime]

    def _require_current_lease(self, lease: RunLease) -> None: ...


class ChapterCoordinatorMixin:
    def activate_chapter_task(self: _ChapterLeaseAuthority, lease: RunLease, task_id: str) -> None:
        with write_transaction(self._connection):
            self._require_current_lease(lease)
            active = self._connection.execute(
                "SELECT task_id FROM tasks WHERE run_id = ? AND state = 'running' AND task_id <> ? LIMIT 1",
                (lease.run_id, task_id),
            ).fetchone()
            if active is not None:
                raise ActiveTaskError(lease.run_id)
            updated = self._connection.execute(
                "UPDATE tasks SET state = 'running', updated_at = ? WHERE task_id = ? AND run_id = ? "
                "AND state = 'pending'",
                (self._now().isoformat(), task_id, lease.run_id),
            )
            if updated.rowcount != 1:
                raise ChapterTaskStateError(task_id)
            append_run_event(self._connection, RunEvent(lease.run_id, "task_started", {"task": task_id}, self._now()))

    def stage_chapter(self: _ChapterLeaseAuthority, lease: RunLease, request: ChapterStageRequest) -> StagedChapter:
        with write_transaction(self._connection):
            self._require_current_lease(lease)
            _require_request_authority(self._connection, lease, request.project_id, request.run_id, request.task_id)
            return stage_chapter(self._connection, request, self._now())

    def stage_memory_candidates(
        self: _ChapterLeaseAuthority,
        lease: RunLease,
        request: ChapterCandidateStageRequest,
    ) -> tuple[str, ...]:
        with write_transaction(self._connection):
            self._require_current_lease(lease)
            _require_request_authority(
                self._connection, lease, request.project_id, request.run_id, request.staged.task_id
            )
            return stage_candidates(self._connection, request)

    def record_chapter_quality(
        self: _ChapterLeaseAuthority,
        lease: RunLease,
        request: ChapterQualityReportRequest,
    ) -> None:
        with write_transaction(self._connection):
            self._require_current_lease(lease)
            _require_task(self._connection, lease.run_id, request.task_id)
            record_quality(self._connection, request, self._now())

    def accept_chapter(
        self: _ChapterLeaseAuthority,
        lease: RunLease,
        request: ChapterAcceptanceRequest,
        path_resolver: ProjectPathResolver,
    ) -> ChapterAcceptanceReceipt | ChapterAcceptanceBlocked:
        try:
            with write_transaction(self._connection):
                self._require_current_lease(lease)
                _require_request_authority(
                    self._connection, lease, request.project_id, request.run_id, request.task_id
                )
                context = ChapterCommitContext(self._connection, path_resolver, self._now(), lease.fencing_token)
                return commit_chapter_acceptance(context, request)
        except ChapterCommitBlocked as blocked:
            return ChapterAcceptanceBlocked(blocked.reason_code)

    def pause_chapter(
        self: _ChapterLeaseAuthority,
        lease: RunLease,
        request: ChapterPauseRequest,
    ) -> None:
        with write_transaction(self._connection):
            self._require_current_lease(lease)
            _require_task(self._connection, lease.run_id, request.task_id)
            now = self._now()
            self._connection.execute(
                "UPDATE tasks SET state = 'waiting_review', updated_at = ? WHERE task_id = ? AND run_id = ? "
                "AND state IN ('pending', 'running')",
                (now.isoformat(), request.task_id, lease.run_id),
            )
            self._connection.execute(
                "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
                (RunStatus.WAITING_REVIEW.value, now.isoformat(), lease.run_id),
            )
            append_run_event(
                self._connection,
                RunEvent(lease.run_id, "chapter_waiting_review", {"reason": request.reason_code}, now),
            )

    def cancellation_requested(self: _ChapterLeaseAuthority, lease: RunLease) -> bool:
        row = self._connection.execute(
            "SELECT l.holder_id, l.fencing_token, l.expires_at, r.state FROM run_leases l "
            "JOIN runs r ON r.run_id = l.run_id WHERE l.run_id = ?",
            (lease.run_id,),
        ).fetchone()
        if row is None:
            raise RunLeaseError(lease.run_id)
        current = (
            str(row[0]) == lease.holder_id
            and int(row[1]) == lease.fencing_token
            and datetime.fromisoformat(str(row[2])) > self._now()
        )
        if not current:
            raise RunLeaseError(lease.run_id)
        return str(row[3]) == RunStatus.CANCEL_REQUESTED.value

    def cancel_chapter(self: _ChapterLeaseAuthority, lease: RunLease, task_id: str) -> None:
        with write_transaction(self._connection):
            if not self.cancellation_requested(lease):
                raise ChapterPersistenceError("chapter cancellation was not requested")
            now = self._now()
            self._connection.execute(
                "UPDATE tasks SET state = 'cancelled', updated_at = ? WHERE task_id = ? AND run_id = ? AND state = 'running'",
                (now.isoformat(), task_id, lease.run_id),
            )
            self._connection.execute(
                "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
                (RunStatus.CANCELLED.value, now.isoformat(), lease.run_id),
            )
            append_run_event(self._connection, RunEvent(lease.run_id, "chapter_cancelled", {"task": task_id}, now))


def _require_request_authority(
    connection: sqlite3.Connection,
    lease: RunLease,
    project_id: ProjectId,
    request_run_id: str,
    task_id: str,
) -> None:
    if lease.run_id != request_run_id:
        raise ChapterPersistenceError("chapter request does not target the leased run")
    row = connection.execute("SELECT project_id FROM runs WHERE run_id = ?", (request_run_id,)).fetchone()
    if row is None or str(row[0]) != project_id.root:
        raise ChapterPersistenceError("chapter request does not target the run project")
    _require_task(connection, request_run_id, task_id)


def _require_task(connection: sqlite3.Connection, run_id: str, task_id: str) -> None:
    row = connection.execute(
        "SELECT state FROM tasks WHERE task_id = ? AND run_id = ? AND task_type = 'chapter'",
        (task_id, run_id),
    ).fetchone()
    if row is None or str(row[0]) not in {"pending", "running"}:
        raise ChapterTaskStateError(task_id)


__all__ = ("ChapterCoordinatorMixin",)
