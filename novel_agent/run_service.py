from __future__ import annotations

from threading import Lock
from novel_agent.run_coordinator import RunCoordinator
from novel_agent.run_models import CommandReceipt, RunLease, RunRecoveryState, StartRunRequest
from novel_agent.run_worker import RunStageExecutor, RunWorker


class RunService:
    def __init__(
        self,
        coordinator: RunCoordinator,
        stage_executor: RunStageExecutor | None = None,
        stage_timeout_seconds: float = 30.0,
    ) -> None:
        self._coordinator = coordinator
        self._worker = None if stage_executor is None else RunWorker(coordinator, stage_executor, stage_timeout_seconds)
        self._worker_lock = Lock()

    def start(self, request: StartRunRequest) -> CommandReceipt:
        return self._coordinator.submit_start(request)

    def claim_next(self, holder_id: str) -> RunLease | None:
        return self._coordinator.claim_next(holder_id)

    def run_once(self, holder_id: str, run_id: str | None = None) -> RunRecoveryState | None:
        if self._worker is None or not self._worker_lock.acquire(blocking=False):
            return None
        try:
            if not self._worker.can_start():
                return None
            if run_id is None:
                lease = self._coordinator.reclaim_expired(holder_id)
                if lease is None:
                    lease = self._coordinator.claim_next(holder_id)
            else:
                lease = self._coordinator.claim_run(run_id, holder_id)
            return None if lease is None else self._worker.run(lease)
        finally:
            self._worker_lock.release()


__all__ = ("RunService",)
