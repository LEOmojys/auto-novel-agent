from __future__ import annotations

import hashlib
import sqlite3

import pytest
from pydantic import ValidationError

from novel_agent.context import ContextBuilder
from novel_agent.context_models import ContextRequest
from novel_agent.contracts import PlanContract, RunCommandType
from novel_agent.pathing import ArtifactId
from novel_agent.revisions import (
    ChapterReplacementRequest,
    ChapterRevisionError,
    ChapterRevisionRef,
    RebaseChapter,
    RebaseConfirmation,
    RebaseProposal,
)
from novel_agent.run_models import ControlRunRequest

from tests.chapter_execution_support import FIXED_TIME, PROJECT_ID, open_run, plan


def test_unconfirmed_downstream_replacement_creates_review_proposal(tmp_path) -> None:
    # Given: an accepted chapter has two later current chapters, accepted replacement revisions, and governed memory.
    plan_contract = plan(chapter_count=3)
    connection, context, coordinator, lease = open_run(tmp_path, plan_contract)
    try:
        _seed_current_chain(connection, plan_contract)
        request = _replacement_request()

        # When: the automatic flow detects that replacing chapter one would invalidate downstream story state.
        proposal = coordinator.propose_chapter_rebase(lease, request)

        # Then: it pauses for review with the full impact while leaving every old current pointer untouched.
        assert isinstance(proposal, RebaseProposal)
        assert proposal.requires_downstream_rebase is True
        assert tuple(item.chapter_number for item in proposal.impact.affected_chapters) == (1, 2, 3)
        assert proposal.impact.affected_m3_memory_ids == ("m3-1", "m3-2", "m3-3")
        assert proposal.impact.affected_plan_memory_ids == ("m2-2", "m2-3")
        assert proposal.impact.invalidated_context_layers == ("m4_retrieval", "m5_working")
        assert _current_revisions(connection) == (("chapter-1", 1), ("chapter-2", 1), ("chapter-3", 1))
        assert connection.execute("SELECT state FROM runs WHERE run_id = 'run-001'").fetchone() == ("waiting_review",)
        assert connection.execute("SELECT status FROM chapter_branches WHERE branch_id = ?", (proposal.branch_id,)).fetchone() == (
            "proposed",
        )
    finally:
        context.__exit__(None, None, None)


def test_confirmed_rebase_creates_new_current_chain_and_keeps_legacy_bytes(tmp_path) -> None:
    # Given: a review proposal and accepted replacement revisions for every chapter in its affected chain.
    plan_contract = plan(chapter_count=3)
    connection, context, coordinator, lease = open_run(tmp_path, plan_contract)
    try:
        _seed_current_chain(connection, plan_contract)
        legacy_path = tmp_path / "projects" / PROJECT_ID.root / "novel" / "legacy-chapter.txt"
        legacy_path.parent.mkdir(parents=True)
        legacy_bytes = b"legacy chapter must remain unchanged"
        legacy_path.write_bytes(legacy_bytes)
        proposal = coordinator.propose_chapter_rebase(lease, _replacement_request())
        coordinator.control(ControlRunRequest(run_id="run-001", command_type=RunCommandType.RESUME, idempotency_key="resume-rebase"))
        confirmation_lease = coordinator.claim_next("rebase-worker")
        assert confirmation_lease is not None

        # When: an explicitly confirmed worker accepts the complete, already-validated replacement chain.
        result = coordinator.confirm_chapter_rebase(confirmation_lease, _confirmation(proposal))

        # Then: only replacement revisions are current, obsolete M3/M2 are excluded, and the branch becomes active.
        assert result.branch_id == proposal.branch_id
        assert _current_revisions(connection) == (("chapter-1", 2), ("chapter-2", 2), ("chapter-3", 2))
        assert connection.execute("SELECT status FROM memory_records WHERE memory_id LIKE 'm3-%' ORDER BY memory_id").fetchall() == [
            ("superseded",),
            ("superseded",),
            ("superseded",),
        ]
        assert connection.execute("SELECT status FROM memory_records WHERE memory_id LIKE 'm2-%' ORDER BY memory_id").fetchall() == [
            ("validated",),
            ("invalidated",),
            ("invalidated",),
        ]
        assert connection.execute("SELECT status FROM chapter_branches WHERE branch_id = ?", (proposal.branch_id,)).fetchone() == (
            "active",
        )
        context_result = ContextBuilder(connection).build(
            ContextRequest(
                project_id=PROJECT_ID,
                run_id="run-001",
                plan_revision=1,
                chapter_plan_id="chapter-plan-004",
                chapter_number=4,
            )
        )
        assert set(context_result.memory_ids).isdisjoint({"m3-1", "m3-2", "m3-3", "m2-2", "m2-3"})
        assert legacy_path.read_bytes() == legacy_bytes
    finally:
        context.__exit__(None, None, None)


def test_automatic_rebase_confirmation_requires_explicit_user_confirmation() -> None:
    # Given: a rebase proposal identifier and one valid replacement chain entry.
    chapter = _chapter_ref(1, 1)

    # When: an automatic caller attempts to construct a confirmation without user approval.
    with pytest.raises(ValidationError):
        RebaseConfirmation(
            project_id=PROJECT_ID,
            run_id="run-001",
            proposal_id="proposal-001",
            user_confirmed=False,
            replacements=(RebaseChapter(source=chapter, replacement=_chapter_ref(1, 2)),),
        )

    # Then: no unconfirmed request can reach the destructive current-pointer operation.


def test_stale_rebase_candidate_cannot_switch_current_chain(tmp_path) -> None:
    # Given: a downstream rebase proposal whose first replacement revision is invalidated before confirmation.
    plan_contract = plan(chapter_count=3)
    connection, context, coordinator, lease = open_run(tmp_path, plan_contract)
    try:
        _seed_current_chain(connection, plan_contract)
        proposal = coordinator.propose_chapter_rebase(lease, _replacement_request())
        coordinator.control(ControlRunRequest(run_id="run-001", command_type=RunCommandType.RESUME, idempotency_key="resume-stale"))
        confirmation_lease = coordinator.claim_next("rebase-worker")
        assert confirmation_lease is not None
        connection.execute("UPDATE artifacts SET status = 'staged' WHERE artifact_id = 'chapter-1' AND revision = 2")

        # When: the worker attempts to confirm a chain containing the stale replacement revision.
        with pytest.raises(ChapterRevisionError, match="replacement"):
            coordinator.confirm_chapter_rebase(confirmation_lease, _confirmation(proposal))

        # Then: the transaction rolls back and no original current pointer is replaced.
        assert _current_revisions(connection) == (("chapter-1", 1), ("chapter-2", 1), ("chapter-3", 1))
    finally:
        context.__exit__(None, None, None)


def _seed_current_chain(connection: sqlite3.Connection, plan_contract: PlanContract) -> None:
    plan_raw = plan_contract.model_dump_json().encode()
    plan_sha256 = hashlib.sha256(plan_raw).hexdigest()
    for number in range(1, 4):
        _insert_chapter(connection, number, 1, is_current=True)
        _insert_chapter(connection, number, 2, is_current=False)
        _insert_memory(connection, f"m3-{number}", "m3_confirmed", "confirmed", f"chapter-{number}", number, number)
        _insert_memory(connection, f"m2-{number}", "m2_plan", "validated", plan_contract.chapters[number - 1].chapter_plan_id, number, None, plan_sha256)


def _insert_chapter(connection: sqlite3.Connection, number: int, revision: int, is_current: bool) -> None:
    raw_bytes = f"chapter-{number}-revision-{revision}".encode()
    connection.execute(
        "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
        "schema_version, is_current, created_at) VALUES (?, ?, ?, ?, 'chapter', 'accepted', ?, ?, 1, ?, ?)",
        (
            f"chapter-{number}",
            revision,
            PROJECT_ID.root,
            "run-001",
            raw_bytes,
            hashlib.sha256(raw_bytes).hexdigest(),
            int(is_current),
            FIXED_TIME.isoformat(),
        ),
    )


def _insert_memory(
    connection: sqlite3.Connection,
    memory_id: str,
    layer: str,
    status: str,
    subject_id: str,
    chapter_number: int,
    source_chapter: int | None,
    plan_sha256: str = "",
) -> None:
    source_id = f"chapter-{source_chapter}" if source_chapter is not None else "plan-001"
    source_revision = 1
    source_hash = plan_sha256
    if source_chapter is not None:
        source_hash = connection.execute(
            "SELECT sha256 FROM artifacts WHERE artifact_id = ? AND revision = 1", (source_id,)
        ).fetchone()[0]
    source_size = connection.execute(
        "SELECT length(raw_bytes) FROM artifacts WHERE artifact_id = ? AND revision = ?", (source_id, source_revision)
    ).fetchone()[0]
    connection.execute(
        "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, subject_id, "
        "predicate, value_json, chapter_start, chapter_end, source_artifact_id, source_artifact_revision, source_span, "
        "source_hash, confidence, created_by, created_at, supersedes_id) "
        "VALUES (?, ?, ?, 1, ?, 'direct_event', ?, ?, 'fact', '{}', ?, ?, ?, ?, ?, ?, 1.0, 'test', ?, NULL)",
        (
            memory_id,
            PROJECT_ID.root,
            "run-001",
            layer,
            status,
            subject_id,
            chapter_number,
            chapter_number,
            source_id,
            source_revision,
            f"0:{source_size}",
            source_hash,
            FIXED_TIME.isoformat(),
        ),
    )


def _replacement_request() -> ChapterReplacementRequest:
    return ChapterReplacementRequest(
        project_id=PROJECT_ID,
        run_id="run-001",
        source=_chapter_ref(1, 1),
        replacement=_chapter_ref(1, 2),
    )


def _confirmation(proposal: RebaseProposal) -> RebaseConfirmation:
    replacements = tuple(
        RebaseChapter(source=chapter, replacement=_chapter_ref(chapter.chapter_number, 2))
        for chapter in proposal.impact.affected_chapters
    )
    return RebaseConfirmation(
        project_id=PROJECT_ID,
        run_id="run-001",
        proposal_id=proposal.proposal_id,
        user_confirmed=True,
        replacements=replacements,
    )


def _chapter_ref(number: int, revision: int) -> ChapterRevisionRef:
    return ChapterRevisionRef(artifact_id=ArtifactId(f"chapter-{number}"), revision=revision, chapter_number=number)


def _current_revisions(connection: sqlite3.Connection) -> tuple[tuple[str, int], ...]:
    return tuple(
        (str(row[0]), int(row[1]))
        for row in connection.execute(
            "SELECT artifact_id, revision FROM artifacts WHERE kind = 'chapter' AND is_current = 1 ORDER BY artifact_id"
        ).fetchall()
    )
