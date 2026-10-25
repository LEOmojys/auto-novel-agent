from __future__ import annotations

import inspect
from collections.abc import Callable
from threading import Event, Lock, Thread
from time import monotonic
from typing import Protocol

from novel_agent.run_coordinator import RunCoordinator
from novel_agent.run_models import RunLease, RunRecoveryState


class RunStageExecutor(Protocol):
    def execute(self, recovery: RunRecoveryState, cancellation: StageCancellationToken | None = None) -> None: ...


class StageCancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


class RunWorker:
    _quarantine_guard = Lock()
    _quarantined_scopes: set[str] = set()

    def __init__(
        self,
        coordinator: RunCoordinator,
        stage_executor: RunStageExecutor,
        stage_timeout_seconds: float = 30.0,
        cancellation_grace_seconds: float = 0.05,
    ) -> None:
        self._coordinator = coordinator
        self._stage_executor = stage_executor
        self._stage_timeout_seconds = stage_timeout_seconds
        self._cancellation_grace_seconds = cancellation_grace_seconds
        self._lock = Lock()
        self._scope_key = coordinator.scope_key()

    @classmethod
    def is_scope_quarantined(cls, scope_key: str) -> bool:
        with cls._quarantine_guard:
            return scope_key in cls._quarantined_scopes

    @property
    def scope_key(self) -> str:
        return self._scope_key

    def can_start(self) -> bool:
        return not self.is_scope_quarantined(self._scope_key) and not self._lock.locked()

    def run(self, lease: RunLease) -> RunRecoveryState | None:
        if self.is_scope_quarantined(self._scope_key) or not self._lock.acquire(blocking=False):
            return None
        release_lock = True
        try:
            recovery = self._coordinator.resume_state(lease)
            errors: list[Exception] = []
            cancellation = StageCancellationToken()

            def execute() -> None:
                try:
                    self._execute_stage(recovery, cancellation)
                except Exception as error:  # noqa: BROAD_EXCEPT_OK
                    errors.append(error)

            stage_thread = Thread(target=execute, name="novel-agent-stage", daemon=True)
            stage_thread.start()
            # Poll for a user CANCEL while the stage runs, so the cancellation
            # token reaches the stage promptly instead of only at timeout.
            deadline = monotonic() + self._stage_timeout_seconds
            user_cancelled = False
            while stage_thread.is_alive():
                if self._coordinator.is_cancel_requested(lease.run_id):
                    user_cancelled = True
                    cancellation.cancel()
                    break
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                stage_thread.join(min(self._cancellation_grace_seconds, remaining) or 0.01)
            if stage_thread.is_alive() and not cancellation.is_cancelled():
                cancellation.cancel()
                stage_thread.join(self._cancellation_grace_seconds)
                if stage_thread.is_alive():
                    release_lock = False
                    self._quarantine_until_finished(stage_thread)
                # `fail_run` falls back to an independent connection when the
                # stage still holds a transaction, so this call is safe even
                # while the quarantined thread finishes.
                self._coordinator.fail_run(lease, "stage_timeout")
                return None
            if stage_thread.is_alive():
                # The user asked to cancel; give the cooperative stage a moment
                # to finalize, then quarantine so this thread never touches the
                # connection while the stage still owns it.
                stage_thread.join(self._cancellation_grace_seconds)
            if stage_thread.is_alive():
                release_lock = False
                self._quarantine_until_finished(
                    stage_thread,
                    on_finished=lambda: self._coordinator.finalize_cancelled(
                        lease, "user_cancelled_after_grace"
                    )
                    if user_cancelled
                    else None,
                )
                return None
            if user_cancelled:
                self._coordinator.finalize_cancelled(lease, "user_cancelled")
                return recovery
            if errors:
                # Keep the exception class and a bounded message so failures are
                # diagnosable from run_events (previously only the class name).
                reason = f"{type(errors[0]).__name__}: {str(errors[0])[:160]}"
                self._coordinator.fail_run(lease, reason)
                return None
            return recovery
        finally:
            if release_lock:
                self._lock.release()

    def _execute_stage(self, recovery: RunRecoveryState, cancellation: StageCancellationToken) -> None:
        execute = self._stage_executor.execute
        try:
            parameters = tuple(inspect.signature(execute).parameters.values())
        except (TypeError, ValueError):
            parameters = ()
        positional = tuple(
            parameter
            for parameter in parameters
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        accepts_token = any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters) or len(positional) >= 2
        if accepts_token:
            execute(recovery, cancellation)
        else:
            execute(recovery)

    def _quarantine_until_finished(
        self,
        stage_thread: Thread,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        with self._quarantine_guard:
            self._quarantined_scopes.add(self._scope_key)

        def clear() -> None:
            stage_thread.join()
            if on_finished is not None:
                try:
                    on_finished()
                except Exception:  # noqa: BROAD_EXCEPT_OK - quarantine cleanup is best effort
                    pass
            with self._quarantine_guard:
                self._quarantined_scopes.discard(self._scope_key)
            self._lock.release()

        Thread(target=clear, name="novel-agent-stage-quarantine", daemon=True).start()


__all__ = ("RunStageExecutor", "RunWorker", "StageCancellationToken")
