from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Protocol

from novel_agent.contracts import PlanContract, RunStatus
from novel_agent.planning_commit import PlanningCommitReceipt, PlanningCommitRequest, commit_validated_plan
from novel_agent.run_event_store import append_run_event
from novel_agent.run_models import RunLease
from novel_agent.run_state import RunEvent
from novel_agent.storage import write_transaction


class _PlanningLeaseAuthority(Protocol):
    _connection: sqlite3.Connection
    _now: Callable[[], datetime]

    def _require_current_lease(self, lease: RunLease) -> None:
        ...


class PlanningCommitAuthorityError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class AcceptedPlanContract:
    plan: PlanContract
    artifact_revision: int


@dataclass(slots=True)  # noqa: MUTABLE_OK
class PlanningRunMismatchError(PlanningCommitAuthorityError):
    lease_run_id: str
    request_run_id: str

    def __str__(self) -> str:
        return f"planning request targets {self.request_run_id}, not leased run {self.lease_run_id}"


@dataclass(slots=True)  # noqa: MUTABLE_OK
class PlanningProjectMismatchError(PlanningCommitAuthorityError):
    run_id: str
    request_project_id: str
    persisted_project_id: str

    def __str__(self) -> str:
        return f"planning request project {self.request_project_id} does not own run {self.run_id}"


class PlanningCoordinatorMixin:
    def commit_validated_plan(
        self: _PlanningLeaseAuthority,
        lease: RunLease,
        request: PlanningCommitRequest,
    ) -> PlanningCommitReceipt:
        with write_transaction(self._connection):
            self._require_current_lease(lease)
            _require_planning_commit_authority(self._connection, lease, request)
            return commit_validated_plan(self._connection, request, self._now())

    def block_planning(self: _PlanningLeaseAuthority, lease: RunLease, reason_code: str) -> None:
        with write_transaction(self._connection):
            self._require_current_lease(lease)
            now = self._now()
            self._connection.execute(
                "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
                (RunStatus.WAITING_REVIEW.value, now.isoformat(), lease.run_id),
            )
            append_run_event(self._connection, RunEvent(lease.run_id, "plan_blocked", {"reason": reason_code}, now))

    def load_accepted_plan(
        self: _PlanningLeaseAuthority,
        lease: RunLease,
        plan_id: str,
    ) -> AcceptedPlanContract | None:
        self._require_current_lease(lease)
        row = self._connection.execute(
            "SELECT raw_bytes, revision FROM artifacts WHERE artifact_id = ? AND run_id = ? "
            "AND kind = 'plan_contract' AND status = 'accepted' AND is_current = 1 "
            "ORDER BY revision DESC LIMIT 1",
            (plan_id, lease.run_id),
        ).fetchone()
        if row is None:
            return None
        return AcceptedPlanContract(
            plan=PlanContract.model_validate_json(bytes(row[0])),
            artifact_revision=int(row[1]),
        )


def _require_planning_commit_authority(
    connection: sqlite3.Connection,
    lease: RunLease,
    request: PlanningCommitRequest,
) -> None:
    if lease.run_id != request.run_id:
        raise PlanningRunMismatchError(lease.run_id, request.run_id)
    row = connection.execute("SELECT project_id FROM runs WHERE run_id = ?", (request.run_id,)).fetchone()
    assert row is not None
    persisted_project_id = str(row[0])
    if request.project_id.root != persisted_project_id:
        raise PlanningProjectMismatchError(request.run_id, request.project_id.root, persisted_project_id)


__all__ = (
    "AcceptedPlanContract",
    "PlanningCommitAuthorityError",
    "PlanningCoordinatorMixin",
    "PlanningProjectMismatchError",
    "PlanningRunMismatchError",
)
