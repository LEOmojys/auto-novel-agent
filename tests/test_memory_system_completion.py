"""End-to-end tests for the completed memory system (v7 behaviour).

Covers the gaps the earlier design left open:

* partial planning commits write M1 with the canonical status and reach context;
* legacy `validated` M1 rows are normalized by the v7 migration;
* oversized facts are truncated into their budget instead of being dropped;
* memories render as readable text (not raw JSON);
* explicit lookup (`mem_query`) sees later chapters;
* the FTS/BM25 index is populated and produces ranking scores;
* needs_review/conflicted candidates can be accepted (with supersession) or
  rejected, with an audit row;
* purging a run also removes its derived FTS rows and supersede edges;
* assembled context packs are persisted for audit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from novel_agent.adapters import (
    CreativeArtifactKind,
    CreativeParseRoute,
    RawCreativeArtifact,
    WorldAdaptation,
)
from novel_agent.context import ContextBuilder, ContextBudget, ContextRequest, memory_query
from novel_agent.contracts import Provenance, ProvenanceKind, WorldBible
from novel_agent.contracts.common import MemoryLayer
from novel_agent.domains.novel_commit import commit_world_only
from novel_agent.memory_models import (
    CandidateStatus,
    EntityResolution,
    InferenceMode,
    MemoryCandidateRecord,
    MemoryFactKind,
    MemoryProposal,
    PromotionAuthority,
    TemporalCertainty,
)
from novel_agent.memory_promotion import current_memory_id
from novel_agent.memory_review import MemoryReviewService
from novel_agent.memory_store import build_evidence, candidate_id, index_memory, load_source
from novel_agent.pathing import ArtifactId, ProjectId
from novel_agent.run_coordinator import RunCoordinator
from novel_agent.storage import MIGRATIONS, connect_project_database, migrate


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("memory-one")
RUN_ID = "run-1"


def _provenance() -> Provenance:
    return Provenance(
        source_kind=ProvenanceKind.SYSTEM,
        source_id="memory-tests",
        source_sha256=hashlib.sha256(b"memory-tests").hexdigest(),
        observed_at=FIXED_TIME,
    )


def _database(tmp_path: Path):
    project_path = tmp_path / "projects" / PROJECT_ID.root
    context = connect_project_database(project_path)
    connection = context.__enter__()
    migrate(connection)
    connection.execute(
        "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, 'running', ?, ?)",
        (RUN_ID, PROJECT_ID.root, FIXED_TIME.isoformat(), FIXED_TIME.isoformat()),
    )
    return connection, context


def _artifact(
    connection,
    artifact_id: str,
    text: str,
    *,
    kind: str = "chapter",
    run_id: str = RUN_ID,
    status: str = "accepted",
) -> None:
    raw = text.encode()
    connection.execute(
        "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, "
        "sha256, schema_version, is_current, created_at) VALUES (?, 1, ?, ?, ?, ?, ?, ?, 1, 1, ?)",
        (artifact_id, PROJECT_ID.root, run_id, kind, status, raw, hashlib.sha256(raw).hexdigest(), FIXED_TIME.isoformat()),
    )


def _memory(
    connection,
    memory_id: str,
    *,
    layer: str,
    kind: str,
    status: str,
    subject_id: str,
    predicate: str,
    value_json: str,
    artifact_id: str,
    text: str,
    chapter_start: int | None = None,
    plan_revision: int = 1,
    supersedes_id: str | None = None,
    run_id: str = RUN_ID,
) -> None:
    span_end = len(text.encode())
    connection.execute(
        "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, "
        "subject_id, predicate, value_json, chapter_start, chapter_end, source_artifact_id, "
        "source_artifact_revision, source_span, source_hash, confidence, created_by, created_at, supersedes_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 1.0, 'test', ?, ?)",
        (
            memory_id,
            PROJECT_ID.root,
            run_id,
            plan_revision,
            layer,
            kind,
            status,
            subject_id,
            predicate,
            value_json,
            chapter_start,
            chapter_start,
            artifact_id,
            f"0:{span_end}",
            hashlib.sha256(text.encode()).hexdigest(),
            FIXED_TIME.isoformat(),
            supersedes_id,
        ),
    )


def _request(
    *,
    chapter_number: int = 1,
    plan_revision: int = 1,
    budget: ContextBudget | None = None,
) -> ContextRequest:
    return ContextRequest(
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        plan_revision=plan_revision,
        chapter_plan_id="plan-1:chapter:1",
        chapter_number=chapter_number,
        query_terms=("潮汐", "魔法"),
        budget=budget or ContextBudget(),
    )


def _proposal(
    text: str,
    *,
    artifact_id: str = "chapter-art",
    value_json: str = '{"event":"发现潮汐之门","object":"潮汐之门","chapter":3}',
    subject_id: str = "char-1",
    predicate: str = "direct_event:chapter_3:1",
    chapter_number: int = 3,
) -> MemoryProposal:
    return MemoryProposal(
        run_id=RUN_ID,
        project_id=PROJECT_ID,
        source_artifact_id=ArtifactId(artifact_id),
        source_artifact_revision=1,
        extraction_schema_version=1,
        target_layer=MemoryLayer.M3_CONFIRMED,
        kind=MemoryFactKind.DIRECT_EVENT,
        subject_id=subject_id,
        predicate=predicate,
        value_json=value_json,
        source_start=0,
        source_end=len(text),
        source_excerpt=text,
        entity_resolution=EntityResolution.EXACT,
        inference_mode=InferenceMode.DIRECT,
        temporal_certainty=TemporalCertainty.EXACT,
        confidence=0.6,
        authority=PromotionAuthority.ACCEPTED_CHAPTER,
        plan_revision=1,
        supersedes_memory_id=None,
        created_at=FIXED_TIME,
    )


def _insert_candidate(
    connection,
    proposal,
    *,
    chapter_number: int = 3,
    status: CandidateStatus = CandidateStatus.CONFLICTED,
) -> str:
    evidence = build_evidence(proposal, load_source(connection, proposal))
    identifier = candidate_id(evidence)
    record = MemoryCandidateRecord(candidate_id=identifier, evidence=evidence, status=status)
    connection.execute(
        "INSERT INTO memory_candidates(candidate_id, run_id, source_artifact_id, source_artifact_revision, "
        "extraction_schema_version, normalized_fact_hash, status, candidate_json, chapter_number, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            identifier,
            RUN_ID,
            proposal.source_artifact_id.root,
            proposal.source_artifact_revision,
            proposal.extraction_schema_version,
            evidence.normalized_fact_hash,
            status.value,
            record.model_dump_json(),
            chapter_number,
            FIXED_TIME.isoformat(),
        ),
    )
    return identifier


# -- M1 status and context reach ------------------------------------------------


def test_partial_world_commit_is_canonical_and_reaches_chapter_context(tmp_path: Path) -> None:
    connection, context = _database(tmp_path)
    try:
        world = WorldBible(
            world_id="world-1",
            brief_id="brief-1",
            title="雾港",
            premise="潮汐魔法支配着群岛，海水会记住名字。",
            rules=("潮水退去时不能说谎",),
            constraints=(),
            created_at=FIXED_TIME,
            provenance=_provenance(),
        )
        payload = world.model_dump_json().encode()
        adaptation = WorldAdaptation(
            raw=RawCreativeArtifact(
                kind=CreativeArtifactKind.WORLD,
                raw_bytes=payload,
                content_sha256=hashlib.sha256(payload).hexdigest(),
                adapter_version="test:v1",
            ),
            value=world,
            parse_route=CreativeParseRoute.STRICT_JSON,
        )
        commit_world_only(
            connection,
            run_id=RUN_ID,
            project_id=PROJECT_ID.root,
            brief_id="brief-1",
            world_result=adaptation,
            committed_at=FIXED_TIME,
            complete=False,
        )

        status = connection.execute(
            "SELECT status FROM memory_records WHERE layer = 'm1_canonical'"
        ).fetchone()
        assert status == ("canonical",)

        result = ContextBuilder(connection).build(_request())
        assert result.budget_ledger.canonical_tokens > 0
        assert any("潮汐魔法" in item.content for item in result.items)
        assert "潮汐魔法" in result.prompt
    finally:
        context.__exit__(None, None, None)


def test_v7_migration_normalizes_legacy_validated_m1(tmp_path: Path) -> None:
    project_path = tmp_path / "projects" / PROJECT_ID.root
    project_path.mkdir(parents=True)
    with connect_project_database(project_path) as connection:
        migrate(connection, MIGRATIONS[:6])
        connection.execute(
            "INSERT INTO memory_records(memory_id, project_id, plan_revision, layer, kind, status, "
            "subject_id, predicate, value_json, created_by, created_at) "
            "VALUES ('legacy-m1', ?, 1, 'm1_canonical', 'world_rule', 'validated', 'w1', 'world_bible', "
            "'{}', 'legacy', ?)",
            (PROJECT_ID.root, FIXED_TIME.isoformat()),
        )
        migrate(connection)
        status = connection.execute(
            "SELECT status FROM memory_records WHERE memory_id = 'legacy-m1'"
        ).fetchone()
        assert status == ("canonical",)
        assert connection.execute("SELECT COUNT(*) FROM memory_fts").fetchone() == (1,)


# -- rendering and budget -------------------------------------------------------


def test_memory_renders_readable_text_instead_of_raw_json(tmp_path: Path) -> None:
    connection, context = _database(tmp_path)
    try:
        text = "潮汐魔法让海水记住名字。"
        _artifact(connection, "world-art", text, kind="world_bible")
        value = json.dumps(
            {"title": "雾港", "premise": "潮汐魔法让海水记住名字。", "rules": ["退潮时不能说谎"]},
            ensure_ascii=False,
        )
        _memory(
            connection,
            "m1-world",
            layer="m1_canonical",
            kind="world_rule",
            status="canonical",
            subject_id="w1",
            predicate="world_bible",
            value_json=value,
            artifact_id="world-art",
            text=text,
        )
        result = ContextBuilder(connection).build(_request())
        item = next(item for item in result.items if item.memory_id == "m1-world")
        assert "世界观《雾港》" in item.content
        assert "潮汐魔法让海水记住名字" in item.content
        assert "{" not in item.content
        assert '"premise"' not in result.prompt
    finally:
        context.__exit__(None, None, None)


def test_oversized_fact_is_truncated_into_budget_not_dropped(tmp_path: Path) -> None:
    connection, context = _database(tmp_path)
    try:
        text = "世界设定" * 1200
        _artifact(connection, "world-art", text, kind="world_bible")
        _memory(
            connection,
            "m1-huge",
            layer="m1_canonical",
            kind="world_rule",
            status="canonical",
            subject_id="w1",
            predicate="world_bible",
            value_json=json.dumps({"title": "雾港", "premise": text}, ensure_ascii=False),
            artifact_id="world-art",
            text=text,
        )
        budget = ContextBudget(
            total_tokens=400,
            canonical_tokens=200,
            plan_tokens=50,
            continuity_tokens=50,
            long_term_tokens=50,
            style_tokens=20,
        )
        result = ContextBuilder(connection).build(_request(budget=budget))
        selected = [item for item in result.items if item.memory_id == "m1-huge"]
        assert selected, "an oversized canonical fact must still be represented"
        assert selected[0].truncated is True
        assert selected[0].token_count <= 200
        assert result.budget_ledger.canonical_tokens <= 200
    finally:
        context.__exit__(None, None, None)


# -- retrieval scope and FTS ----------------------------------------------------


def test_memory_query_sees_later_chapters_but_chapter_context_does_not(tmp_path: Path) -> None:
    connection, context = _database(tmp_path)
    try:
        text = "第三章：林知夏在潮汐之门发现了旧誓言。"
        _artifact(connection, "chapter-3", text)
        _memory(
            connection,
            "m3-ch3",
            layer="m3_confirmed",
            kind="direct_event",
            status="confirmed",
            subject_id="char-1",
            predicate="direct_event:chapter_3:1",
            value_json=json.dumps({"event": "发现潮汐之门", "object": "旧誓言", "chapter": 3}, ensure_ascii=False),
            artifact_id="chapter-3",
            text=text,
            chapter_start=3,
        )

        chapter_one = ContextBuilder(connection).build(_request(chapter_number=1))
        assert "m3-ch3" not in chapter_one.memory_ids
        assert any(
            exclusion.memory_id == "m3-ch3" and exclusion.reason == "future_memory"
            for exclusion in chapter_one.exclusions
        )

        hits = memory_query(connection, project_id=PROJECT_ID.root, query_terms=("潮汐",), chapter_number=1)
        assert any(hit.memory_id == "m3-ch3" for hit in hits)
    finally:
        context.__exit__(None, None, None)


def test_memory_fts_produces_bm25_scores(tmp_path: Path) -> None:
    connection, context = _database(tmp_path)
    try:
        index_memory(connection, "mem-fts", "潮汐魔法与雾港群岛")
        builder = ContextBuilder(connection)
        scores = builder._fts_scores(("潮汐魔法",))
        assert "mem-fts" in scores
        assert scores["mem-fts"] < 0  # SQLite bm25(): lower is better
        assert builder._fts_scores(("潮",)) == {}  # trigram needs 3+ characters
    finally:
        context.__exit__(None, None, None)


# -- human review loop ----------------------------------------------------------


def test_review_accepts_conflicted_candidate_and_supersedes_current_value(tmp_path: Path) -> None:
    connection, context = _database(tmp_path)
    try:
        text = "第三章：旧誓言被改写，潮汐之门给出了新的答案。"
        _artifact(connection, "chapter-art", text)
        _memory(
            connection,
            "m3-current",
            layer="m3_confirmed",
            kind="direct_event",
            status="confirmed",
            subject_id="char-1",
            predicate="direct_event:chapter_3:1",
            value_json=json.dumps({"event": "旧誓言未被改写", "chapter": 3}, ensure_ascii=False),
            artifact_id="chapter-art",
            text=text,
            chapter_start=3,
        )
        proposal = _proposal(text)
        identifier = _insert_candidate(connection, proposal)

        service = MemoryReviewService(connection, now=lambda: FIXED_TIME)
        listed = service.list(PROJECT_ID.root)
        assert [view.candidate_id for view in listed] == [identifier]

        resolution = service.resolve(
            identifier,
            project_id=PROJECT_ID.root,
            decision="accept",
            actor="tester",
            reason="确认新事件",
        )
        assert resolution.status == "promoted"
        assert resolution.memory_id

        promoted = connection.execute(
            "SELECT supersedes_id, chapter_start, status, created_by FROM memory_records WHERE memory_id = ?",
            (resolution.memory_id,),
        ).fetchone()
        assert promoted is not None
        assert promoted[0] == "m3-current"
        assert promoted[1] == 3
        assert promoted[2] == "confirmed"
        assert promoted[3] == "memory-review:tester"

        # The superseded value is replaced, not deleted.
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE memory_id = 'm3-current'"
        ).fetchone() == (1,)
        assert current_memory_id(connection, proposal) == resolution.memory_id

        edge = connection.execute(
            "SELECT relation FROM memory_edges WHERE source_memory_id = ? AND target_memory_id = 'm3-current'",
            (resolution.memory_id,),
        ).fetchone()
        assert edge == ("supersedes",)
        assert connection.execute(
            "SELECT decision, actor FROM memory_reviews WHERE candidate_id = ?", (identifier,)
        ).fetchone() == ("accepted", "tester")

        # After resolution the candidate leaves the review queue.
        assert service.list(PROJECT_ID.root) == ()
    finally:
        context.__exit__(None, None, None)


def test_review_rejects_candidate_and_audits_it(tmp_path: Path) -> None:
    connection, context = _database(tmp_path)
    try:
        text = "第三章：一段不可靠的推断。"
        _artifact(connection, "chapter-art", text)
        identifier = _insert_candidate(
            connection, _proposal(text), status=CandidateStatus.NEEDS_REVIEW
        )
        service = MemoryReviewService(connection, now=lambda: FIXED_TIME)
        resolution = service.resolve(
            identifier,
            project_id=PROJECT_ID.root,
            decision="reject",
            actor="tester",
            reason="推断无据",
        )
        assert resolution.status == "rejected"
        assert connection.execute(
            "SELECT status FROM memory_candidates WHERE candidate_id = ?", (identifier,)
        ).fetchone() == ("rejected",)
        assert connection.execute(
            "SELECT decision, reason FROM memory_reviews WHERE candidate_id = ?", (identifier,)
        ).fetchone() == ("rejected", "推断无据")
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE source_artifact_id = 'chapter-art'"
        ).fetchone() == (0,)
    finally:
        context.__exit__(None, None, None)


# -- retention/audit ------------------------------------------------------------


def test_purge_removes_derived_fts_rows_and_edges(tmp_path: Path) -> None:
    connection, context = _database(tmp_path)
    try:
        text = "计划记忆"
        _artifact(connection, "plan-art", text, kind="plan_contract")
        _memory(
            connection,
            "m2-plan",
            layer="m2_plan",
            kind="plan_goal",
            status="validated",
            subject_id="chapter-1",
            predicate="chapter_goal",
            value_json='{"goal":"测试"}',
            artifact_id="plan-art",
            text=text,
        )
        _memory(
            connection,
            "m2-other",
            layer="m2_plan",
            kind="plan_goal",
            status="validated",
            subject_id="chapter-2",
            predicate="chapter_goal",
            value_json='{"goal":"另一个计划"}',
            artifact_id="plan-art",
            text=text,
        )
        index_memory(connection, "m2-plan", '{"goal":"测试"}')
        connection.execute(
            "INSERT INTO memory_edges(edge_id, source_memory_id, target_memory_id, relation, created_at) "
            "VALUES ('edge-1', 'm2-plan', 'm2-other', 'supersedes', ?)",
            (FIXED_TIME.isoformat(),),
        )

        RunCoordinator(connection, lambda: FIXED_TIME)._purge_execution_traces(RUN_ID)

        assert connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE memory_id = 'm2-plan'"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_fts WHERE memory_id = 'm2-plan'"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_edges WHERE edge_id = 'edge-1'"
        ).fetchone() == (0,)
    finally:
        context.__exit__(None, None, None)


def test_context_pack_is_persisted_for_audit(tmp_path: Path) -> None:
    connection, context = _database(tmp_path)
    try:
        text = "潮汐之门在雾港开启。"
        _artifact(connection, "chapter-art", text)
        _memory(
            connection,
            "m3-pack",
            layer="m3_confirmed",
            kind="direct_event",
            status="confirmed",
            subject_id="char-1",
            predicate="direct_event:chapter_1:1",
            value_json=json.dumps({"event": "潮汐之门开启", "chapter": 1}, ensure_ascii=False),
            artifact_id="chapter-art",
            text=text,
            chapter_start=1,
        )
        result = ContextBuilder(connection).build(_request(), persist=True)
        assert result.pack_id is not None
        row = connection.execute(
            "SELECT digest, payload_json FROM context_packs WHERE pack_id = ?", (result.pack_id,)
        ).fetchone()
        assert row is not None
        assert row[0] == result.digest
        payload = json.loads(row[1])
        assert payload["items"][0]["memory_id"] in result.memory_ids
    finally:
        context.__exit__(None, None, None)


def test_purge_clears_cross_run_supersede_pointer_and_context_packs(tmp_path: Path) -> None:
    connection, context = _database(tmp_path)
    try:
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) "
            "VALUES ('run-2', ?, 'succeeded', ?, ?)",
            (PROJECT_ID.root, FIXED_TIME.isoformat(), FIXED_TIME.isoformat()),
        )
        text = "计划记忆"
        _artifact(connection, "plan-art", text, kind="plan_contract")
        _memory(
            connection,
            "m2-old",
            layer="m2_plan",
            kind="plan_goal",
            status="validated",
            subject_id="chapter-1",
            predicate="chapter_goal",
            value_json='{"goal":"旧计划"}',
            artifact_id="plan-art",
            text=text,
            run_id="run-1",
        )
        _memory(
            connection,
            "m2-new",
            layer="m2_plan",
            kind="plan_goal",
            status="validated",
            subject_id="chapter-1",
            predicate="chapter_goal",
            value_json='{"goal":"新计划"}',
            artifact_id="plan-art",
            text=text,
            run_id="run-2",
            supersedes_id="m2-old",
        )
        connection.execute(
            "INSERT INTO context_packs(pack_id, project_id, run_id, chapter_plan_id, digest, "
            "tokenizer_version, payload_json, created_at) VALUES ('pack-1', ?, 'run-1', 'cp', "
            "'digest', 'cjk-1', '{}', ?)",
            (PROJECT_ID.root, FIXED_TIME.isoformat()),
        )

        RunCoordinator(connection, lambda: FIXED_TIME)._purge_execution_traces("run-1")

        assert connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE memory_id = 'm2-old'"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT supersedes_id FROM memory_records WHERE memory_id = 'm2-new'"
        ).fetchone() == (None,)
        assert connection.execute(
            "SELECT COUNT(*) FROM context_packs WHERE run_id = 'run-1'"
        ).fetchone() == (0,)
    finally:
        context.__exit__(None, None, None)
