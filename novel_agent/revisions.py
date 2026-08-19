from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime

from pydantic import ValidationError

from novel_agent.contracts import PlanContract
from novel_agent.revision_models import (
    ChapterReplacementRequest,
    ChapterRevisionError,
    ChapterRevisionRef,
    RebaseApplied,
    RebaseChapter,
    RebaseConfirmation,
    RebaseImpact,
    RebaseProposal,
)


def propose_chapter_rebase(
    connection: sqlite3.Connection,
    request: ChapterReplacementRequest,
    created_at: datetime,
) -> RebaseProposal:
    _require_current_source(connection, request)
    _require_accepted_replacement(connection, request)
    current_chain = _current_chapter_chain(connection, request)
    affected_chapters = tuple(
        chapter for chapter in current_chain if chapter.chapter_number >= request.source.chapter_number
    )
    impact = RebaseImpact(
        affected_chapters=affected_chapters,
        affected_m3_memory_ids=_affected_m3_memory_ids(connection, request),
        affected_plan_memory_ids=_affected_plan_memory_ids(connection, request),
        invalidated_context_layers=("m4_retrieval", "m5_working"),
    )
    branch_id = _stable_identifier("branch", request.run_id, request.source, request.replacement)
    proposal = RebaseProposal(
        proposal_id=_stable_identifier("proposal", request.run_id, request.source, request.replacement),
        branch_id=branch_id,
        project_id=request.project_id,
        run_id=request.run_id,
        source=request.source,
        replacement=request.replacement,
        impact=impact,
        requires_downstream_rebase=len(affected_chapters) > 1,
        created_at=created_at,
    )
    connection.execute(
        "INSERT INTO chapter_branches(branch_id, run_id, parent_branch_id, status, created_at, activated_at) "
        "VALUES (?, ?, (SELECT branch_id FROM chapter_branches WHERE run_id = ? AND status = 'active' "
        "ORDER BY activated_at DESC, branch_id LIMIT 1), 'proposed', ?, NULL) ON CONFLICT(branch_id) DO NOTHING",
        (branch_id, request.run_id, request.run_id, created_at.isoformat()),
    )
    inserted = connection.execute(
        "INSERT INTO chapter_rebase_proposals(proposal_id, branch_id, run_id, project_id, status, proposal_json, "
        "created_at, confirmed_at) VALUES (?, ?, ?, ?, 'proposed', ?, ?, NULL) ON CONFLICT(proposal_id) DO NOTHING",
        (
            proposal.proposal_id,
            proposal.branch_id,
            proposal.run_id,
            proposal.project_id.root,
            proposal.model_dump_json(),
            created_at.isoformat(),
        ),
    )
    if inserted.rowcount == 1:
        return proposal
    row = connection.execute(
        "SELECT proposal_json FROM chapter_rebase_proposals WHERE proposal_id = ?", (proposal.proposal_id,)
    ).fetchone()
    if row is None:
        raise ChapterRevisionError("rebase proposal was not persisted")
    return RebaseProposal.model_validate_json(str(row[0]))


def confirm_chapter_rebase(
    connection: sqlite3.Connection,
    request: RebaseConfirmation,
    confirmed_at: datetime,
) -> RebaseApplied:
    row = connection.execute(
        "SELECT status, proposal_json FROM chapter_rebase_proposals WHERE proposal_id = ? AND run_id = ? AND project_id = ?",
        (request.proposal_id, request.run_id, request.project_id.root),
    ).fetchone()
    if row is None:
        raise ChapterRevisionError("rebase proposal does not belong to this run project")
    proposal = RebaseProposal.model_validate_json(str(row[1]))
    if str(row[0]) == "confirmed":
        return proposal.applied()
    if str(row[0]) != "proposed":
        raise ChapterRevisionError("rebase proposal is not awaiting confirmation")
    _require_complete_replacement_chain(connection, proposal, request)
    for replacement in request.replacements:
        connection.execute(
            "UPDATE artifacts SET is_current = 0 WHERE artifact_id = ? AND revision = ? AND is_current = 1",
            (replacement.source.artifact_id.root, replacement.source.revision),
        )
        updated = connection.execute(
            "UPDATE artifacts SET is_current = 1 WHERE artifact_id = ? AND revision = ? AND status = 'accepted'",
            (replacement.replacement.artifact_id.root, replacement.replacement.revision),
        )
        if updated.rowcount != 1:
            raise ChapterRevisionError("accepted replacement revision is no longer available")
        connection.execute(
            "UPDATE memory_records SET status = 'superseded' WHERE layer = 'm3_confirmed' "
            "AND source_artifact_id = ? AND source_artifact_revision = ? AND status = 'confirmed'",
            (replacement.source.artifact_id.root, replacement.source.revision),
        )
        connection.execute(
            "INSERT INTO chapter_branch_members(branch_id, chapter_number, artifact_id, artifact_revision) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(branch_id, chapter_number) DO UPDATE SET artifact_id = excluded.artifact_id, "
            "artifact_revision = excluded.artifact_revision",
            (
                proposal.branch_id,
                replacement.replacement.chapter_number,
                replacement.replacement.artifact_id.root,
                replacement.replacement.revision,
            ),
        )
    for memory_id in proposal.impact.affected_plan_memory_ids:
        connection.execute(
            "UPDATE memory_records SET status = 'invalidated' WHERE memory_id = ? AND layer = 'm2_plan' "
            "AND status = 'validated'",
            (memory_id,),
        )
    connection.execute(
        "UPDATE chapter_branches SET status = 'superseded' WHERE run_id = ? AND status = 'active' AND branch_id <> ?",
        (request.run_id, proposal.branch_id),
    )
    connection.execute(
        "UPDATE chapter_branches SET status = 'active', activated_at = ? WHERE branch_id = ?",
        (confirmed_at.isoformat(), proposal.branch_id),
    )
    connection.execute(
        "UPDATE chapter_rebase_proposals SET status = 'confirmed', confirmed_at = ? WHERE proposal_id = ?",
        (confirmed_at.isoformat(), proposal.proposal_id),
    )
    return proposal.applied()


def _require_current_source(connection: sqlite3.Connection, request: ChapterReplacementRequest) -> None:
    row = connection.execute(
        "SELECT project_id, run_id, kind, status, is_current FROM artifacts WHERE artifact_id = ? AND revision = ?",
        (request.source.artifact_id.root, request.source.revision),
    ).fetchone()
    expected = (request.project_id.root, request.run_id, "chapter", "accepted", 1)
    if row is None or tuple(row) != expected:
        raise ChapterRevisionError("replacement source is not this run's accepted current chapter")


def _require_accepted_replacement(connection: sqlite3.Connection, request: ChapterReplacementRequest) -> None:
    row = connection.execute(
        "SELECT project_id, run_id, kind, status FROM artifacts WHERE artifact_id = ? AND revision = ?",
        (request.replacement.artifact_id.root, request.replacement.revision),
    ).fetchone()
    expected = (request.project_id.root, request.run_id, "chapter", "accepted")
    if row is None or tuple(row) != expected:
        raise ChapterRevisionError("replacement is not an accepted revision of this run chapter")


def _current_chapter_chain(
    connection: sqlite3.Connection,
    request: ChapterReplacementRequest,
) -> tuple[ChapterRevisionRef, ...]:
    rows = connection.execute(
        "SELECT a.artifact_id, a.revision, MIN(m.chapter_start) FROM artifacts a LEFT JOIN memory_records m "
        "ON m.source_artifact_id = a.artifact_id AND m.source_artifact_revision = a.revision "
        "AND m.layer = 'm3_confirmed' WHERE a.project_id = ? AND a.run_id = ? AND a.kind = 'chapter' "
        "AND a.status = 'accepted' AND a.is_current = 1 GROUP BY a.artifact_id, a.revision",
        (request.project_id.root, request.run_id),
    ).fetchall()
    chain: list[ChapterRevisionRef] = []
    for row in rows:
        artifact_id = str(row[0])
        revision = int(row[1])
        chapter_number = request.source.chapter_number if (
            artifact_id == request.source.artifact_id.root and revision == request.source.revision
        ) else None if row[2] is None else int(row[2])
        if chapter_number is None:
            raise ChapterRevisionError("current chapter lacks confirmed sequence evidence")
        chain.append(ChapterRevisionRef(artifact_id=artifact_id, revision=revision, chapter_number=chapter_number))
    ordered = tuple(sorted(chain, key=lambda chapter: chapter.chapter_number))
    if not ordered or ordered[0].chapter_number < 1:
        raise ChapterRevisionError("run has no ordered current chapter chain")
    if len({chapter.chapter_number for chapter in ordered}) != len(ordered):
        raise ChapterRevisionError("current chapter chain has duplicate sequence evidence")
    return ordered


def _affected_m3_memory_ids(connection: sqlite3.Connection, request: ChapterReplacementRequest) -> tuple[str, ...]:
    rows = connection.execute(
        "SELECT memory_id FROM memory_records WHERE project_id = ? AND run_id = ? AND layer = 'm3_confirmed' "
        "AND status = 'confirmed' AND chapter_start >= ? ORDER BY chapter_start, memory_id",
        (request.project_id.root, request.run_id, request.source.chapter_number),
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _affected_plan_memory_ids(connection: sqlite3.Connection, request: ChapterReplacementRequest) -> tuple[str, ...]:
    row = connection.execute(
        "SELECT raw_bytes FROM artifacts WHERE project_id = ? AND run_id = ? AND kind = 'plan_contract' "
        "AND status = 'accepted' AND is_current = 1 ORDER BY revision DESC LIMIT 1",
        (request.project_id.root, request.run_id),
    ).fetchone()
    if row is None:
        raise ChapterRevisionError("run has no current accepted plan contract")
    try:
        plan = PlanContract.model_validate_json(bytes(row[0]))
    except ValidationError as error:
        raise ChapterRevisionError("current plan contract cannot establish future chapter impact") from error
    chapter_numbers = {chapter.chapter_plan_id: chapter.chapter_number for chapter in plan.chapters}
    rows = connection.execute(
        "SELECT memory_id, subject_id FROM memory_records WHERE project_id = ? AND run_id = ? "
        "AND layer = 'm2_plan' AND status = 'validated' ORDER BY memory_id",
        (request.project_id.root, request.run_id),
    ).fetchall()
    return tuple(
        str(row[0])
        for row in rows
        if chapter_numbers.get(str(row[1]), 0) > request.source.chapter_number
    )


def _require_complete_replacement_chain(
    connection: sqlite3.Connection,
    proposal: RebaseProposal,
    confirmation: RebaseConfirmation,
) -> None:
    expected = {chapter.chapter_number: chapter for chapter in proposal.impact.affected_chapters}
    supplied = {chapter.source.chapter_number: chapter for chapter in confirmation.replacements}
    if set(expected) != set(supplied):
        raise ChapterRevisionError("confirmation does not replace every affected current chapter")
    for chapter_number, source in expected.items():
        replacement = supplied[chapter_number]
        if replacement.source != source:
            raise ChapterRevisionError("confirmation source does not match the proposed current chain")
        request = ChapterReplacementRequest(
            project_id=proposal.project_id,
            run_id=proposal.run_id,
            source=replacement.source,
            replacement=replacement.replacement,
        )
        _require_current_source(connection, request)
        _require_accepted_replacement(connection, request)


def _stable_identifier(
    prefix: str,
    run_id: str,
    source: ChapterRevisionRef,
    replacement: ChapterRevisionRef,
) -> str:
    seed = ":".join((prefix, run_id, source.artifact_id.root, str(source.revision), str(replacement.revision)))
    return f"{prefix}:{hashlib.sha256(seed.encode()).hexdigest()}"


__all__ = (
    "ChapterReplacementRequest",
    "ChapterRevisionError",
    "ChapterRevisionRef",
    "RebaseApplied",
    "RebaseChapter",
    "RebaseConfirmation",
    "RebaseProposal",
    "confirm_chapter_rebase",
    "propose_chapter_rebase",
)
