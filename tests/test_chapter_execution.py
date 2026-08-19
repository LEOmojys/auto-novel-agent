from __future__ import annotations

import hashlib
import json

from novel_agent.adapters import CreativeArtifactKind, RawCreativeArtifact, SchemaFailure
from novel_agent.chapter_executor import (
    ChapterDraftRequest,
    ChapterExecutionCancelled,
    ChapterExecutionPolicy,
    ChapterExecutionSucceeded,
    ChapterExecutionWaitingReview,
)
from novel_agent.contracts import RunCommandType
from novel_agent.run_models import ControlRunRequest

from tests.chapter_execution_support import (
    FakeChapterGenerator,
    FakeContinuityAuditor,
    FakeQualityReviewer,
    execution_request,
    open_run,
    plan,
    workflow,
)


class SchemaFailingChapterGenerator:
    def generate(self, _request: ChapterDraftRequest) -> SchemaFailure:
        raw_bytes = b"not-a-valid-chapter"
        return SchemaFailure(
            raw_artifact=RawCreativeArtifact(
                kind=CreativeArtifactKind.CHAPTER,
                raw_bytes=raw_bytes,
                content_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                adapter_version="schema-failure-fake-v1",
            ),
            expected_contract="ChapterDraft",
            parse_routes=(),
        )


def test_two_chapter_run_commits_in_order_and_repairs_once(tmp_path) -> None:
    # Given: a planned two-chapter run whose first quality review asks for one repair.
    plan_contract = plan()
    connection, context, coordinator, lease = open_run(tmp_path, plan_contract)
    generator = FakeChapterGenerator()
    reviewer = FakeQualityReviewer(repairs_before_accept=1)
    try:
        chapter_workflow = workflow(connection, coordinator, generator, reviewer, FakeContinuityAuditor(), tmp_path)

        # When: the governed executor processes the full plan sequentially.
        result = chapter_workflow.execute(lease, execution_request(plan_contract))

        # Then: chapter two sees only accepted chapter-one memory and both accepted commits advance in order.
        assert isinstance(result, ChapterExecutionSucceeded)
        assert generator.chapter_numbers == [1, 1, 2]
        assert generator.context_memory_ids[:2] == [(), ()]
        assert generator.context_memory_ids[2]
        assert connection.execute(
            "SELECT chapter_start FROM memory_records WHERE layer = 'm3_confirmed' ORDER BY chapter_start"
        ).fetchall() == [(1,), (2,)]
        assert connection.execute(
            "SELECT state FROM tasks WHERE run_id = 'run-001' ORDER BY created_at, task_id"
        ).fetchall() == [("succeeded",), ("succeeded",)]
        assert connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE kind = 'chapter' AND status = 'accepted'"
        ).fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM quality_reports").fetchone() == (3,)
        assert connection.execute("SELECT state FROM runs WHERE run_id = 'run-001'").fetchone() == ("succeeded",)
        manifests = [
            json.loads(str(row[0]))
            for row in connection.execute(
                "SELECT manifest_json FROM audit_manifests WHERE run_id = 'run-001' ORDER BY created_at, manifest_id"
            ).fetchall()
        ]
        assert len(manifests) == 2
        for manifest in manifests:
            assert manifest["model_name"] == "not-recorded"
            assert manifest["parameters"]["parse_route"] == "strict_json"
            assert manifest["prompt_sha256"] == manifest["context_digest"]
            assert manifest["response_sha256"] == manifest["raw_sha256"]
            assert manifest["adapter_version"] == "fake-chapter-v1"
            assert manifest["artifact_revision"] >= 1
    finally:
        context.__exit__(None, None, None)


def test_second_failed_repair_pauses_without_next_chapter_commit(tmp_path) -> None:
    # Given: a two-chapter plan whose first chapter keeps failing every repair attempt.
    plan_contract = plan()
    connection, context, coordinator, lease = open_run(tmp_path, plan_contract)
    generator = FakeChapterGenerator()
    try:
        chapter_workflow = workflow(
            connection,
            coordinator,
            generator,
            FakeQualityReviewer(repairs_before_accept=99),
            FakeContinuityAuditor(),
            tmp_path,
        )

        # When: the repair budget is exhausted.
        result = chapter_workflow.execute(lease, execution_request(plan_contract))

        # Then: staged attempts remain, while chapter two receives no accepted context or task.
        assert isinstance(result, ChapterExecutionWaitingReview)
        assert generator.chapter_numbers == [1, 1, 1]
        assert connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE kind = 'chapter' AND status = 'accepted'"
        ).fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM memory_records WHERE layer = 'm3_confirmed'").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM tasks WHERE run_id = 'run-001'").fetchone() == (1,)
        assert connection.execute("SELECT state FROM tasks WHERE run_id = 'run-001'").fetchone() == ("waiting_review",)
        assert connection.execute("SELECT state FROM runs WHERE run_id = 'run-001'").fetchone() == ("waiting_review",)
    finally:
        context.__exit__(None, None, None)


def test_continuity_conflict_preserves_staging_without_acceptance(tmp_path) -> None:
    # Given: one planned chapter that reaches a hard continuity conflict.
    plan_contract = plan(chapter_count=1)
    connection, context, coordinator, lease = open_run(tmp_path, plan_contract)
    try:
        chapter_workflow = workflow(
            connection,
            coordinator,
            FakeChapterGenerator(),
            FakeQualityReviewer(repairs_before_accept=0),
            FakeContinuityAuditor(hard_conflict=True),
            tmp_path,
        )

        # When: continuity rejects the staged draft before quality acceptance.
        result = chapter_workflow.execute(lease, execution_request(plan_contract))

        # Then: the draft remains staged and the run pauses with no M3 record.
        assert isinstance(result, ChapterExecutionWaitingReview)
        assert connection.execute("SELECT status FROM artifacts WHERE kind = 'chapter'").fetchone() == ("staged",)
        assert connection.execute("SELECT COUNT(*) FROM memory_records WHERE layer = 'm3_confirmed'").fetchone() == (0,)
    finally:
        context.__exit__(None, None, None)


def test_cancellation_preserves_pending_task_without_provider_draft(tmp_path) -> None:
    # Given: a running chapter run that received a cancel command before its checkpoint.
    plan_contract = plan(chapter_count=1)
    connection, context, coordinator, lease = open_run(tmp_path, plan_contract)
    generator = FakeChapterGenerator()
    try:
        coordinator.control(ControlRunRequest(run_id="run-001", command_type=RunCommandType.CANCEL, idempotency_key="cancel"))
        chapter_workflow = workflow(
            connection,
            coordinator,
            generator,
            FakeQualityReviewer(repairs_before_accept=0),
            FakeContinuityAuditor(),
            tmp_path,
        )

        # When: the worker observes cooperative cancellation.
        result = chapter_workflow.execute(lease, execution_request(plan_contract))

        # Then: no provider draft or accepted chapter is written after cancellation.
        assert isinstance(result, ChapterExecutionCancelled)
        assert generator.chapter_numbers == []
        assert connection.execute("SELECT COUNT(*) FROM artifacts WHERE kind = 'chapter'").fetchone() == (0,)
        assert connection.execute("SELECT state FROM runs WHERE run_id = 'run-001'").fetchone() == ("cancelled",)
    finally:
        context.__exit__(None, None, None)


def test_chapter_budget_blocks_before_draft(tmp_path) -> None:
    # Given: a plan that exceeds the execution policy's automatic chapter budget.
    plan_contract = plan()
    connection, context, coordinator, lease = open_run(tmp_path, plan_contract)
    generator = FakeChapterGenerator()
    try:
        chapter_workflow = workflow(
            connection,
            coordinator,
            generator,
            FakeQualityReviewer(repairs_before_accept=0),
            FakeContinuityAuditor(),
            tmp_path,
            policy=ChapterExecutionPolicy(maximum_automatic_chapters=1),
        )

        # When: the executor validates its bounded automatic workload.
        result = chapter_workflow.execute(lease, execution_request(plan_contract))

        # Then: it pauses before any generation or governed chapter write.
        assert isinstance(result, ChapterExecutionWaitingReview)
        assert generator.chapter_numbers == []
        assert connection.execute("SELECT COUNT(*) FROM artifacts WHERE kind = 'chapter'").fetchone() == (0,)
        assert connection.execute("SELECT state FROM runs WHERE run_id = 'run-001'").fetchone() == ("waiting_review",)
    finally:
        context.__exit__(None, None, None)


def test_chapter_schema_failure_pauses_before_staging(tmp_path) -> None:
    # Given: the creative adapter rejects the provider response before producing a typed chapter draft.
    plan_contract = plan(chapter_count=1)
    connection, context, coordinator, lease = open_run(tmp_path, plan_contract)
    try:
        chapter_workflow = workflow(
            connection,
            coordinator,
            SchemaFailingChapterGenerator(),
            FakeQualityReviewer(repairs_before_accept=0),
            FakeContinuityAuditor(),
            tmp_path,
        )

        # When: chapter execution receives the typed schema failure.
        result = chapter_workflow.execute(lease, execution_request(plan_contract))

        # Then: the run pauses without a staged or accepted chapter, memory candidate, or provider-derived next task.
        assert isinstance(result, ChapterExecutionWaitingReview)
        assert result.reason_code == "chapter_schema_failure"
        assert connection.execute("SELECT COUNT(*) FROM artifacts WHERE kind = 'chapter'").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM memory_candidates").fetchone() == (0,)
        assert connection.execute("SELECT state FROM tasks WHERE run_id = 'run-001'").fetchone() == ("waiting_review",)
    finally:
        context.__exit__(None, None, None)
