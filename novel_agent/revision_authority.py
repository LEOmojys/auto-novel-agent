from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Callable, Protocol

from novel_agent.contracts import RunStatus
from novel_agent.pathing import ProjectId
from novel_agent.revision_models import (
    ChapterReplacementRequest,
    ChapterRevisionError,
    RebaseApplied,
    RebaseConfirmation,
    RebaseProposal,
)
from novel_agent.revisions import confirm_chapter_rebase, propose_chapter_rebase
from novel_agent.run_event_store import append_run_event
from novel_agent.run_models import RunLease
from novel_agent.run_state import RunEvent
from novel_agent.storage import write_transaction


class _RevisionLeaseAuthority(Protocol):
    _connection: sqlite3.Connection
    _now: Callable[[], datetime]

    def _require_current_lease(self, lease: RunLease) -> None: ...


class RevisionCoordinatorMixin:
    def propose_chapter_rebase(
        self: _RevisionLeaseAuthority,
        lease: RunLease,
        request: ChapterReplacementRequest,
    ) -> RebaseProposal:
        with write_transaction(self._connection):
            self._require_current_lease(lease)
            _require_revision_authority(self._connection, lease, request.project_id, request.run_id)
            proposal = propose_chapter_rebase(self._connection, request, self._now())
            now = self._now()
            self._connection.execute(
                "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
                (RunStatus.WAITING_REVIEW.value, now.isoformat(), lease.run_id),
            )
            append_run_event(
                self._connection,
                RunEvent(
                    lease.run_id,
                    "chapter_rebase_proposed",
                    {"proposal": proposal.proposal_id, "branch": proposal.branch_id},
                    now,
                ),
            )
            return proposal

    def confirm_chapter_rebase(
        self: _RevisionLeaseAuthority,
        lease: RunLease,
        request: RebaseConfirmation,
    ) -> RebaseApplied:
        with write_transaction(self._connection):
            self._require_current_lease(lease)
            _require_revision_authority(self._connection, lease, request.project_id, request.run_id)
            result = confirm_chapter_rebase(self._connection, request, self._now())
            append_run_event(
                self._connection,
                RunEvent(
                    lease.run_id,
                    "chapter_rebase_confirmed",
                    {"proposal": result.proposal_id, "branch": result.branch_id},
                    self._now(),
                ),
            )
            return result


def _require_revision_authority(
    connection: sqlite3.Connection,
    lease: RunLease,
    project_id: ProjectId,
    run_id: str,
) -> None:
    if lease.run_id != run_id:
        raise ChapterRevisionError("revision request does not target the leased run")
    row = connection.execute("SELECT project_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None or str(row[0]) != project_id.root:
        raise ChapterRevisionError("revision request does not target the run project")


__all__ = ("RevisionCoordinatorMixin",)
