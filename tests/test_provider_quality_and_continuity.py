"""Tests for the provider-backed quality reviewer and continuity auditor.

These are the two gates that previously ran on offline fakes in production:
the agent loop always ACCEPTed, and the continuity auditor claimed every
planned beat was present.  The tests script a provider response and assert the
real parsing / degradation behaviour.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from novel_agent.chapter_models import ContinuityAuditRequest, StagedChapter
from novel_agent.contracts import (
    ChapterPlan,
    CharacterCard,
    CharacterSet,
    EvidenceRef,
    Provenance,
    ProvenanceKind,
)
from novel_agent.context_models import BudgetLedger, ContextBuildResult
from novel_agent.domains.continuity_evaluators import ProviderContinuityAuditor
from novel_agent.domains.quality_runtime import LazyProviderQualityReviewer
from novel_agent.pathing import ArtifactId, ProjectId
from novel_agent.provider_types import (
    ProviderFailureKind,
    ProviderTransportError,
    ProviderTransportResult,
)
from novel_agent.quality import ChapterQualityRequest, QualityOutcome
from novel_agent.storage import connect_project_database, migrate


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("quality-one")
_SIX_DIMENSIONS = {
    "style_consistency": {"score": 4, "reason": "一致"},
    "character_consistency": {"score": 4, "reason": "一致"},
    "prose_fluency": {"score": 4, "reason": "流畅"},
    "scene_completeness": {"score": 4, "reason": "完整"},
    "dialogue_naturalness": {"score": 4, "reason": "自然"},
    "content_alignment": {"score": 4, "reason": "贴合"},
}


class _ScriptedTransport:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.prompts: list[str] = []

    def generate(self, request, cancellation=None) -> ProviderTransportResult:
        self.prompts.append(request.prompt)
        if self.error is not None:
            raise self.error
        return ProviderTransportResult(
            content=self.content or "{}",
            model_name="scripted",
            prompt_tokens=1,
            completion_tokens=1,
        )


def _provenance() -> Provenance:
    return Provenance(
        source_kind=ProvenanceKind.SYSTEM,
        source_id="quality-tests",
        source_sha256=hashlib.sha256(b"quality").hexdigest(),
        observed_at=FIXED_TIME,
    )


def _chapter_plan() -> ChapterPlan:
    return ChapterPlan(
        chapter_plan_id="plan:chapter:1",
        chapter_number=1,
        title="初入学院",
        goal="主角报到",
        planned_events=("抵达学院", "结识室友"),
        evidence=(
            EvidenceRef(
                evidence_id="evidence-1",
                artifact_id="plan-art",
                artifact_revision=1,
                source_sha256=hashlib.sha256(b"plan").hexdigest(),
                recorded_at=FIXED_TIME,
            ),
        ),
        created_at=FIXED_TIME,
        provenance=_provenance(),
    )


def _staged(body: str) -> StagedChapter:
    raw = body.encode()
    return StagedChapter(
        artifact_id=ArtifactId("chapter-art"),
        revision=1,
        task_id="task-1",
        chapter_plan_id="plan:chapter:1",
        chapter_number=1,
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        body=body,
    )


def _context() -> ContextBuildResult:
    return ContextBuildResult(
        items=(),
        exclusions=(),
        prompt="学院悬浮于云海之上。",
        digest=hashlib.sha256(b"context").hexdigest(),
        tokenizer_version="cjk-1",
        budget_ledger=BudgetLedger(0, 0, 0, 0, 0, 0),
    )


# -- continuity auditor ---------------------------------------------------------


def test_continuity_auditor_parses_report_and_drops_invented_beats() -> None:
    transport = _ScriptedTransport(
        json.dumps(
            {"located_beats": ["抵达学院", "编造的事件"], "hard_conflicts": ["与既有设定矛盾"]},
            ensure_ascii=False,
        )
    )
    auditor = ProviderContinuityAuditor(transport=transport, now=lambda: FIXED_TIME, run_id="run-1")
    report = auditor.audit(ContinuityAuditRequest(_chapter_plan(), _staged("正文"), _context()))
    assert report.located_beats == ("抵达学院",)
    assert report.hard_conflicts == ("与既有设定矛盾",)
    assert "抵达学院" in transport.prompts[0]


def test_continuity_auditor_failure_degrades_to_hard_conflict() -> None:
    transport = _ScriptedTransport(error=ProviderTransportError(ProviderFailureKind.TRANSIENT, "boom"))
    failures: list[tuple[str, str, str, str, str]] = []
    auditor = ProviderContinuityAuditor(
        transport=transport,
        now=lambda: FIXED_TIME,
        run_id="run-1",
        failure_sink=lambda *args: failures.append(args),
    )
    report = auditor.audit(ContinuityAuditRequest(_chapter_plan(), _staged("正文"), _context()))
    assert report.hard_conflicts == ("continuity_auditor_error",)
    assert failures and failures[0][0] == "continuity"


# -- lazy quality reviewer ------------------------------------------------------


_OPEN_CONTEXTS: list[object] = []


def _database(tmp_path: Path, *, with_cast: bool):
    context = connect_project_database(tmp_path / "projects" / PROJECT_ID.root)
    connection = context.__enter__()
    _OPEN_CONTEXTS.append(context)
    migrate(connection)
    if with_cast:
        cast = CharacterSet(
            character_set_id="cast-1",
            world_id="world-1",
            characters=(
                CharacterCard(
                    character_id="char-1",
                    name="林晚",
                    role="主角",
                    goal="查明真相",
                    traits=("冷静",),
                    created_at=FIXED_TIME,
                    provenance=_provenance(),
                ),
            ),
            created_at=FIXED_TIME,
            provenance=_provenance(),
        )
        raw = cast.model_dump_json().encode()
        connection.execute(
            "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, "
            "sha256, schema_version, is_current, created_at) VALUES ('cast-1', 1, ?, NULL, "
            "'character_set', 'accepted', ?, ?, 1, 1, ?)",
            (PROJECT_ID.root, raw, hashlib.sha256(raw).hexdigest(), FIXED_TIME.isoformat()),
        )
    return connection


def _quality_request(body: str) -> ChapterQualityRequest:
    return ChapterQualityRequest(
        project_id=PROJECT_ID,
        artifact_id="chapter-art",
        plan_id="plan-1",
        expected_plan_id="plan-1",
        chapter_plan_id="plan:chapter:1",
        expected_chapter_plan_id="plan:chapter:1",
        chapter_number=1,
        expected_chapter_number=1,
        body=body,
        target_chinese_characters=len(body),
    )


def test_lazy_reviewer_uses_provider_scores(tmp_path: Path) -> None:
    connection = _database(tmp_path, with_cast=True)
    transport = _ScriptedTransport(json.dumps(_SIX_DIMENSIONS, ensure_ascii=False))
    reviewer = LazyProviderQualityReviewer(
        connection=connection,
        project_id=PROJECT_ID,
        run_id="run-1",
        transport=transport,
        now=lambda: FIXED_TIME,
    )
    decision = reviewer.review(_quality_request("字" * 3000))
    assert decision.outcome is QualityOutcome.ACCEPT
    assert decision.score == 4.0
    assert decision.policy_version
    assert transport.prompts, "the provider evaluator must actually be called"


def test_lazy_reviewer_pauses_without_cast(tmp_path: Path) -> None:
    connection = _database(tmp_path, with_cast=False)
    reviewer = LazyProviderQualityReviewer(
        connection=connection,
        project_id=PROJECT_ID,
        run_id="run-1",
        transport=_ScriptedTransport("{}"),
        now=lambda: FIXED_TIME,
    )
    decision = reviewer.review(_quality_request("字" * 100))
    assert decision.outcome is QualityOutcome.PAUSE
    assert decision.reason_codes == ("reviewer_unavailable",)


def test_continuity_auditor_mode_switch(monkeypatch) -> None:
    from novel_agent.default_stage import _continuity_auditor
    from novel_agent.offline_fake import OfflineContinuityAuditor

    def build() -> object:
        return _continuity_auditor(
            transport=_ScriptedTransport("{}"),
            now=lambda: FIXED_TIME,
            run_id="run-1",
            prompt=None,
            failure_sink=None,
            call_event_sink=None,
        )

    monkeypatch.setenv("NOVEL_AGENT_CONTINUITY_AUDIT", "offline")
    assert isinstance(build(), OfflineContinuityAuditor)
    monkeypatch.setenv("NOVEL_AGENT_CONTINUITY_AUDIT", "llm")
    assert isinstance(build(), ProviderContinuityAuditor)
