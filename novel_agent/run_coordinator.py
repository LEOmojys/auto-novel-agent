from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable, Final

from novel_agent.chapter_authority import ChapterCoordinatorMixin
from novel_agent.contracts import RunCommandType, RunStatus
from novel_agent.planning_authority import PlanningCoordinatorMixin
from novel_agent.revision_authority import RevisionCoordinatorMixin
from novel_agent.run_models import (
    ActiveTaskError,
    CommitDisposition,
    CommandReceipt,
    ControlRunRequest,
    ProviderAttemptRequest,
    RunLease,
    RunLeaseError,
    RunRecoveryState,
    RunTransitionError,
    StartRunRequest,
    TaskResultReceipt,
    TaskResultRequest,
    TaskStartRequest,
)
from novel_agent.run_event_store import append_run_event
from novel_agent.run_recovery import load_recovery_state
from novel_agent.run_state import RunEvent, command_id as build_command_id
from novel_agent.run_transition import resolve_control_transition
from novel_agent.sqlite_transactions import TransactionStateError
from novel_agent.storage import write_transaction

if TYPE_CHECKING:
    from novel_agent.context import ContextBuilder


# A stage can run several sequential LLM calls (each up to 5 minutes with
# the legacy transport timeout), so the lease must outlive the whole stage.
_LEASE_SECONDS: Final = 900
_RUNNING_TASK_STATE: Final = "running"
#: Terminal states from which a re-start requeues the same run_id.
_TERMINAL_STATES: Final = frozenset(
    (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED)
)


class RunCoordinator(RevisionCoordinatorMixin, ChapterCoordinatorMixin, PlanningCoordinatorMixin):
    def __init__(
        self,
        connection: sqlite3.Connection,
        now: Callable[[], datetime],
        lease_seconds: int = _LEASE_SECONDS,
    ) -> None:
        self._connection = connection
        self._now = now
        self._lease_seconds = lease_seconds

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def context_builder(self) -> ContextBuilder:
        from novel_agent.context import ContextBuilder

        return ContextBuilder(self._connection)

    def scope_key(self) -> str:
        row = self._connection.execute("PRAGMA database_list").fetchone()
        if row is None or not row[2]:
            return f"connection:{id(self._connection)}"
        return str(row[2])

    def submit_start(self, request: StartRunRequest) -> CommandReceipt:
        command_id = build_command_id(request.run_id, request.idempotency_key)
        with write_transaction(self._connection):
            now = self._now()
            existing = self._command(request.run_id, request.idempotency_key)
            if existing is not None:
                # Same idempotency key: the command already exists.  It never
                # raises — a terminal run is requeued (explicit retry), any
                # other state is a no-op receipt (double-click protection).
                state = self._run_state(request.run_id)
                if state in _TERMINAL_STATES:
                    self._requeue(request, existing, now, previous_state=state)
                    state = RunStatus.QUEUED
                return CommandReceipt(existing, request.run_id, state.value, True)
            row = self._connection.execute(
                "SELECT state FROM runs WHERE run_id = ?", (request.run_id,)
            ).fetchone()
            if row is None:
                # Fresh run.
                self._connection.execute(
                    "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (request.run_id, request.project_id.root, RunStatus.QUEUED.value, now.isoformat(), now.isoformat()),
                )
                self._insert_command(command_id, request.run_id, RunCommandType.START, request.idempotency_key, now)
                append_run_event(self._connection, RunEvent(request.run_id, "run_queued", {}, now))
                if request.brief_sha256 is not None:
                    append_run_event(
                        self._connection,
                        RunEvent(request.run_id, "run_brief_submitted", {"sha256": request.brief_sha256}, now),
                    )
                if request.execution_payload is not None:
                    self._connection.execute(
                        "UPDATE commands SET payload_json = ? WHERE command_id = ?",
                        (request.execution_payload.model_dump_json(), command_id),
                    )
                return CommandReceipt(command_id, request.run_id, RunStatus.QUEUED.value, False)
            current = RunStatus(str(row[0]))
            if current in _TERMINAL_STATES:
                # Terminal run with a new idempotency key: requeue with the
                # new start payload (different requirement / retry).
                self._requeue(request, command_id, now, previous_state=current)
                return CommandReceipt(command_id, request.run_id, RunStatus.QUEUED.value, False)
            # An execution, approval or pause flow is in progress: starting
            # again would collide with the active state.
            raise RunTransitionError(request.run_id, current.value, RunCommandType.START.value)

    def _requeue(
        self,
        request: StartRunRequest,
        command_id: str,
        now: datetime,
        *,
        previous_state: RunStatus,
    ) -> None:
        """Requeue a terminal run for a fresh execution cycle (retry/continue).

        Execution traces of the previous cycle are purged so the re-run starts
        clean (fresh task rows, no stale idempotent tool results), while
        accepted artifacts and canonical memory are kept so project-scoped
        asset reuse carries the story forward.  The start command is updated
        in place when the idempotency key is unchanged (the commands table has
        ``UNIQUE (run_id, idempotency_key)``); a new key inserts a new row.
        """
        self._connection.execute(
            "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
            (RunStatus.QUEUED.value, now.isoformat(), request.run_id),
        )
        existing = self._command(request.run_id, request.idempotency_key)
        if existing is not None:
            # Same key: refresh the row in place (``UNIQUE (run_id,
            # idempotency_key)`` forbids a second row) and bump created_at so
            # the latest-start-command recovery keeps seeing this cycle.  The
            # payload column is NOT NULL, so it is only overwritten when a new
            # execution payload is supplied.
            if request.execution_payload is not None:
                self._connection.execute(
                    "UPDATE commands SET payload_json = ?, created_at = ? WHERE command_id = ?",
                    (request.execution_payload.model_dump_json(), now.isoformat(), existing),
                )
            else:
                self._connection.execute(
                    "UPDATE commands SET created_at = ? WHERE command_id = ?",
                    (now.isoformat(), existing),
                )
        else:
            self._insert_command(command_id, request.run_id, RunCommandType.START, request.idempotency_key, now)
            if request.execution_payload is not None:
                self._connection.execute(
                    "UPDATE commands SET payload_json = ? WHERE command_id = ?",
                    (request.execution_payload.model_dump_json(), command_id),
                )
        if request.brief_sha256 is not None:
            append_run_event(
                self._connection,
                RunEvent(request.run_id, "run_brief_submitted", {"sha256": request.brief_sha256}, now),
            )
        append_run_event(
            self._connection,
            RunEvent(
                request.run_id,
                "run_requeued",
                {"reason": "restart", "previous_state": previous_state.value},
                now,
            ),
        )
        self._purge_execution_traces(request.run_id)
        # A re-run (same requirement, same run id) restarts the story from
        # chapter 1: drop the previous cycle's accepted chapters (and the
        # memory rows that reference them) so the written-chapter detection
        # does not make the rebuilt graph skip chapter 1 as "already
        # written".  REPAIR/REVISE go through control() and never touch the
        # accepted chapters.
        self._connection.execute(
            "DELETE FROM memory_records WHERE source_artifact_id IN ("
            "  SELECT artifact_id FROM artifacts WHERE run_id = ? AND kind = 'chapter' AND status = 'accepted'"
            ")",
            (request.run_id,),
        )
        self._connection.execute(
            "DELETE FROM artifacts WHERE run_id = ? AND kind = 'chapter' AND status = 'accepted'",
            (request.run_id,),
        )

    def _purge_execution_traces(self, run_id: str) -> None:
        """Delete the previous cycle's execution traces for a requeued run.

        Accepted artifacts and canonical/confirmed memory survive (they are
        the reusable assets); staged artifacts, task rows, tool calls, branch
        state, provider calls and other run-scoped execution state are wiped
        so the re-execution starts from a clean ledger.
        """
        # Rebase proposals reference branches RESTRICT; branches cascade to
        # their members.  Memory rows referencing staged artifacts must go
        # before the staged artifacts are removed (FK RESTRICT).
        self._connection.execute(
            "DELETE FROM chapter_rebase_proposals WHERE run_id = ?", (run_id,)
        )
        self._connection.execute(
            "DELETE FROM chapter_branches WHERE run_id = ?", (run_id,)
        )
        purged_memory_ids = [
            str(row[0])
            for row in self._connection.execute(
                "SELECT memory_id FROM memory_records WHERE run_id = ? AND layer IN "
                "('m0_candidate', 'm2_plan', 'm4_retrieval', 'm5_working')",
                (run_id,),
            ).fetchall()
        ]
        if purged_memory_ids:
            placeholders = ", ".join("?" for _ in purged_memory_ids)
            self._connection.execute(
                f"UPDATE memory_records SET supersedes_id = NULL "
                f"WHERE supersedes_id IN ({placeholders})",
                tuple(purged_memory_ids),
            )
        for identifier in purged_memory_ids:
            self._connection.execute(
                "DELETE FROM memory_edges WHERE source_memory_id = ? OR target_memory_id = ?",
                (identifier, identifier),
            )
            self._connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (identifier,))
        self._connection.execute("DELETE FROM context_packs WHERE run_id = ?", (run_id,))
        self._connection.execute(
            "DELETE FROM memory_records WHERE run_id = ? AND layer IN "
            "('m0_candidate', 'm2_plan', 'm4_retrieval', 'm5_working')",
            (run_id,),
        )
        for table in (
            "memory_candidates",
            "quality_reports",
            "audit_manifests",
            "provider_failures",
            "memory_conflicts",
            "tool_calls",
            "provider_calls",
            "task_results",
            "tasks",
            "run_checkpoints",
            "run_leases",
        ):
            self._connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        self._connection.execute(
            "DELETE FROM artifacts WHERE run_id = ? AND status = 'staged'", (run_id,)
        )

    def control(self, request: ControlRunRequest) -> CommandReceipt:
        command_id = build_command_id(request.run_id, request.idempotency_key)
        with write_transaction(self._connection):
            existing = self._command(request.run_id, request.idempotency_key)
            if existing is not None:
                state = self._run_state(request.run_id)
                return CommandReceipt(existing, request.run_id, state.value, True)
            now = self._now()
            current = self._run_state(request.run_id)
            next_state = resolve_control_transition(self._connection, request.run_id, current, request.command_type)
            if next_state is not None:
                self._connection.execute(
                    "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
                    (next_state.value, now.isoformat(), request.run_id),
                )
                self._insert_command(
                    command_id,
                    request.run_id,
                    request.command_type,
                    request.idempotency_key,
                    now,
                    request.payload,
                )
                if request.command_type in (RunCommandType.REPAIR, RunCommandType.REVISE):
                    # A repair/revise requeues the run for re-execution; purge
                    # the previous cycle's traces (tasks, tool calls, staged
                    # artifacts, ...) so the re-run starts clean — otherwise
                    # stale idempotent tool-call rows make the gateway replay
                    # the old result without calling the provider again.
                    self._purge_execution_traces(request.run_id)
                event_type = (
                    "run_repaired"
                    if request.command_type is RunCommandType.REPAIR
                    else "run_revised"
                    if request.command_type is RunCommandType.REVISE
                    else "run_transitioned"
                )
                append_run_event(
                    self._connection,
                    RunEvent(
                        request.run_id,
                        event_type,
                        {"command": request.command_type.value, "state": next_state.value},
                        now,
                    ),
                )
                return CommandReceipt(command_id, request.run_id, next_state.value, False)
            append_run_event(
                self._connection,
                RunEvent(
                    request.run_id,
                    "transition_rejected",
                    {"command": request.command_type.value, "state": current.value},
                    now,
                )
            )
        raise RunTransitionError(request.run_id, current.value, request.command_type.value)

    def claim_next(self, holder_id: str) -> RunLease | None:
        with write_transaction(self._connection):
            if self._has_running_run():
                return None
            row = self._connection.execute(
                "SELECT run_id FROM runs WHERE state = ? ORDER BY created_at, run_id LIMIT 1",
                (RunStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            return self._claim_queued_run(str(row[0]), holder_id)

    def claim_run(self, run_id: str, holder_id: str) -> RunLease | None:
        with write_transaction(self._connection):
            row = self._connection.execute(
                "SELECT r.state, l.expires_at, l.fencing_token FROM runs r "
                "LEFT JOIN run_leases l ON l.run_id = r.run_id WHERE r.run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            state = RunStatus(str(row[0]))
            if state is RunStatus.QUEUED:
                if self._has_running_run():
                    return None
                return self._claim_queued_run(run_id, holder_id)
            if (
                state is RunStatus.RUNNING
                and row[1] is not None
                and datetime.fromisoformat(str(row[1])) <= self._now()
                and self._has_other_running_run(run_id) is False
            ):
                lease = self._write_lease(run_id, holder_id, int(row[2]) + 1)
                append_run_event(
                    self._connection,
                    RunEvent(
                        run_id,
                        "lease_reclaimed",
                        {"fence": lease.fencing_token, "holder": holder_id},
                        self._now(),
                    ),
                )
                return lease
            return None

    def _has_other_running_run(self, run_id: str) -> bool:
        return self._connection.execute(
            "SELECT 1 FROM runs WHERE state = ? AND run_id <> ? LIMIT 1",
            (RunStatus.RUNNING.value, run_id),
        ).fetchone() is not None

    def _claim_queued_run(self, run_id: str, holder_id: str) -> RunLease:
            existing_lease = self._connection.execute(
                "SELECT fencing_token FROM run_leases WHERE run_id = ?", (run_id,)
            ).fetchone()
            next_fence = 1 if existing_lease is None else int(existing_lease[0]) + 1
            lease = self._write_lease(run_id, holder_id, next_fence)
            self._connection.execute(
                "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
                (RunStatus.RUNNING.value, self._now().isoformat(), run_id),
            )
            append_run_event(
                self._connection,
                RunEvent(run_id, "run_claimed", {"fence": lease.fencing_token, "holder": holder_id}, self._now())
            )
            return lease

    def reclaim_expired(self, holder_id: str) -> RunLease | None:
        with write_transaction(self._connection):
            row = self._connection.execute(
                "SELECT l.run_id, l.fencing_token FROM run_leases l JOIN runs r ON r.run_id = l.run_id "
                "WHERE r.state = ? AND l.expires_at <= ? ORDER BY l.expires_at, l.run_id LIMIT 1",
                (RunStatus.RUNNING.value, self._now().isoformat()),
            ).fetchone()
            if row is None:
                return None
            lease = self._write_lease(str(row[0]), holder_id, int(row[1]) + 1)
            append_run_event(
                self._connection,
                RunEvent(lease.run_id, "lease_reclaimed", {"fence": lease.fencing_token, "holder": holder_id}, self._now())
            )
            return lease

    def start_task(self, lease: RunLease, request: TaskStartRequest) -> None:
        with write_transaction(self._connection):
            self._require_current_lease(lease)
            active = self._connection.execute(
                "SELECT task_id FROM tasks WHERE run_id = ? AND state = ? AND task_id <> ? LIMIT 1",
                (lease.run_id, _RUNNING_TASK_STATE, request.task_id),
            ).fetchone()
            if active is not None:
                raise ActiveTaskError(lease.run_id)
            now = self._now().isoformat()
            existing = self._connection.execute(
                "SELECT run_id, task_type, idempotency_key, state FROM tasks WHERE task_id = ?",
                (request.task_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing[0]) != lease.run_id
                    or str(existing[1]) != request.task_type
                    or str(existing[2]) != request.idempotency_key
                ):
                    raise ActiveTaskError(lease.run_id)
                return
            self._connection.execute(
                "INSERT INTO tasks(task_id, run_id, task_type, state, idempotency_key, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(task_id) DO NOTHING",
                (request.task_id, lease.run_id, request.task_type, _RUNNING_TASK_STATE, request.idempotency_key, now, now),
            )
            append_run_event(self._connection, RunEvent(lease.run_id, "task_started", {"task": request.task_id}, self._now()))

    def fail_run(self, lease: RunLease, reason: str) -> bool:
        try:
            return self._fail_run_on(self._connection, lease, reason)
        except TransactionStateError:
            # The stage thread currently owns a transaction on the main
            # connection (timeout path).  Finalize the run on an independent
            # connection instead of waiting for the stage to finish.
            connection = self._write_connection()
            if connection is None:
                return False
            try:
                return self._fail_run_on(connection, lease, reason)
            finally:
                connection.close()

    def _fail_run_on(self, connection: sqlite3.Connection, lease: RunLease, reason: str) -> bool:
        with write_transaction(connection):
            if not self._is_current_lease(lease, connection=connection):
                # The lease may have expired during a long LLM call; the run
                # must still leave RUNNING or it appears stuck forever.
                row = connection.execute(
                    "SELECT state FROM runs WHERE run_id = ?", (lease.run_id,)
                ).fetchone()
                if row is None or str(row[0]) != RunStatus.RUNNING.value:
                    return False
            now = self._now()
            connection.execute(
                "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ? AND state = ?",
                (RunStatus.FAILED.value, now.isoformat(), lease.run_id, RunStatus.RUNNING.value),
            )
            append_run_event(connection, RunEvent(lease.run_id, "run_failed", {"reason": reason}, now))
            return True

    def record_provider_attempt(self, lease: RunLease, request: ProviderAttemptRequest) -> CommitDisposition:
        call_id = hashlib.sha256(f"{request.task_id}:{request.attempt}".encode()).hexdigest()
        with write_transaction(self._connection):
            task = self._connection.execute(
                "SELECT run_id, state FROM tasks WHERE task_id = ?", (request.task_id,)
            ).fetchone()
            if (
                request.run_id != lease.run_id
                or task is None
                or str(task[0]) != lease.run_id
                or str(task[1]) != _RUNNING_TASK_STATE
                or not self._is_current_lease(lease)
            ):
                append_run_event(
                    self._connection,
                    RunEvent(lease.run_id, "provider_attempt_discarded", {"task": request.task_id}, self._now()),
                )
                return CommitDisposition.DISCARDED
            existing = self._connection.execute(
                "SELECT request_sha256, state FROM provider_calls WHERE task_id = ? AND attempt = ?",
                (request.task_id, request.attempt),
            ).fetchone()
            if existing is not None:
                if str(existing[0]) != request.request_sha256 or str(existing[1]) != request.state:
                    append_run_event(
                        self._connection,
                        RunEvent(lease.run_id, "provider_attempt_discarded", {"task": request.task_id, "attempt": request.attempt}, self._now()),
                    )
                    return CommitDisposition.DISCARDED
                return CommitDisposition.COMMITTED
            self._connection.execute(
                "INSERT INTO provider_calls(call_id, run_id, task_id, attempt, request_sha256, state, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(task_id, attempt) DO NOTHING",
                (call_id, request.run_id, request.task_id, request.attempt, request.request_sha256, request.state, self._now().isoformat()),
            )
            return CommitDisposition.COMMITTED

    def commit_task_result(self, lease: RunLease, request: TaskResultRequest) -> TaskResultReceipt:
        with write_transaction(self._connection):
            task = self._connection.execute(
                "SELECT state FROM tasks WHERE task_id = ? AND run_id = ?", (request.task_id, lease.run_id)
            ).fetchone()
            if task is None or not self._is_current_lease(lease):
                append_run_event(self._connection, RunEvent(lease.run_id, "late_result_discarded", {"task": request.task_id}, self._now()))
                return TaskResultReceipt(CommitDisposition.DISCARDED, False)
            existing = self._connection.execute(
                "SELECT result_sha256 FROM task_results WHERE task_id = ?", (request.task_id,)
            ).fetchone()
            if existing is not None:
                return TaskResultReceipt(CommitDisposition.COMMITTED, True)
            if str(task[0]) != _RUNNING_TASK_STATE:
                append_run_event(self._connection, RunEvent(lease.run_id, "late_result_discarded", {"task": request.task_id}, self._now()))
                return TaskResultReceipt(CommitDisposition.DISCARDED, False)
            now = self._now()
            self._connection.execute(
                "INSERT INTO task_results(task_id, run_id, result_sha256, committed_fence, created_at) VALUES (?, ?, ?, ?, ?)",
                (request.task_id, lease.run_id, request.result_sha256, lease.fencing_token, now.isoformat()),
            )
            self._connection.execute(
                "UPDATE tasks SET state = ?, updated_at = ? WHERE task_id = ? AND run_id = ? AND state = ?",
                ("succeeded", now.isoformat(), request.task_id, lease.run_id, _RUNNING_TASK_STATE),
            )
            checkpoint_id = hashlib.sha256(f"{request.task_id}:{request.result_sha256}".encode()).hexdigest()
            self._connection.execute(
                "INSERT INTO run_checkpoints(checkpoint_id, run_id, task_id, stage, result_sha256, fencing_token, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (checkpoint_id, lease.run_id, request.task_id, "result_committed", request.result_sha256, lease.fencing_token, now.isoformat()),
            )
            append_run_event(self._connection, RunEvent(lease.run_id, "task_result_committed", {"task": request.task_id}, now))
            return TaskResultReceipt(CommitDisposition.COMMITTED, False)

    def resume_state(self, lease: RunLease) -> RunRecoveryState:
        self._require_current_lease(lease)
        return load_recovery_state(self._connection, lease)

    def _command(self, run_id: str, idempotency_key: str) -> str | None:
        row = self._connection.execute(
            "SELECT command_id FROM commands WHERE run_id = ? AND idempotency_key = ?",
            (run_id, idempotency_key),
        ).fetchone()
        return None if row is None else str(row[0])

    def _insert_command(
        self,
        command_id: str,
        run_id: str,
        command_type: RunCommandType,
        key: str,
        now: datetime,
        payload: dict[str, object] | None = None,
    ) -> None:
        import json as json_module

        self._connection.execute(
            "INSERT INTO commands(command_id, run_id, command_type, idempotency_key, payload_json, state, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                command_id,
                run_id,
                command_type.value,
                key,
                json_module.dumps(payload or {}, ensure_ascii=False),
                "accepted",
                now.isoformat(),
            ),
        )

    def _run_state(self, run_id: str) -> RunStatus:
        row = self._connection.execute("SELECT state FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise RunTransitionError(run_id, "missing", "unknown")
        return RunStatus(str(row[0]))

    def is_cancel_requested(self, run_id: str) -> bool:
        """Whether the run has been asked to cancel (for worker polling).

        The stage thread owns the main connection; polling it from the worker
        thread would risk reading inside the stage's transaction.  A short
        read-only connection keeps the poll separate from stage writes.
        """
        connection = self._read_only_connection()
        try:
            row = (connection or self._connection).execute(
                "SELECT state FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        finally:
            if connection is not None:
                connection.close()
        return row is not None and str(row[0]) in (
            RunStatus.CANCEL_REQUESTED.value,
            RunStatus.CANCELLED.value,
        )

    def _write_connection(self) -> sqlite3.Connection | None:
        row = self._connection.execute("PRAGMA database_list").fetchone()
        path = "" if row is None else str(row[2])
        if not path:
            return None
        try:
            connection = sqlite3.connect(f"file:{path}", uri=True, timeout=5.0)
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except sqlite3.Error:
            return None

    def _read_only_connection(self) -> sqlite3.Connection | None:
        row = self._connection.execute("PRAGMA database_list").fetchone()
        path = "" if row is None else str(row[2])
        if not path:
            return None
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
            connection.execute("PRAGMA busy_timeout = 2000")
            return connection
        except sqlite3.Error:
            return None

    def _has_running_run(self) -> bool:
        return self._connection.execute(
            "SELECT 1 FROM runs WHERE state = ? LIMIT 1", (RunStatus.RUNNING.value,)
        ).fetchone() is not None

    def _write_lease(self, run_id: str, holder_id: str, token: int) -> RunLease:
        now = self._now()
        expires_at = now + timedelta(seconds=self._lease_seconds)
        self._connection.execute(
            "INSERT INTO run_leases(run_id, holder_id, fencing_token, expires_at, acquired_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET holder_id = excluded.holder_id, fencing_token = excluded.fencing_token, "
            "expires_at = excluded.expires_at, acquired_at = excluded.acquired_at",
            (run_id, holder_id, token, expires_at.isoformat(), now.isoformat()),
        )
        return RunLease(run_id, holder_id, token, expires_at)

    def _require_current_lease(self, lease: RunLease) -> None:
        if not self._is_current_lease(lease):
            raise RunLeaseError(lease.run_id)

    def _is_current_lease(
        self,
        lease: RunLease,
        connection: sqlite3.Connection | None = None,
    ) -> bool:
        row = (connection or self._connection).execute(
            "SELECT l.holder_id, l.fencing_token, l.expires_at, r.state FROM run_leases l "
            "JOIN runs r ON r.run_id = l.run_id WHERE l.run_id = ?",
            (lease.run_id,),
        ).fetchone()
        if row is None:
            return False
        expires_at = datetime.fromisoformat(str(row[2]))
        return (
            str(row[0]) == lease.holder_id
            and int(row[1]) == lease.fencing_token
            and expires_at > self._now()
            and str(row[3]) == RunStatus.RUNNING.value
        )

__all__ = (
    "CommitDisposition",
    "RunCoordinator",
    "RunLease",
    "RunLeaseError",
    "RunTransitionError",
)
