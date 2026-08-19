"""审批纯文本章节 + 续写装配真实评审器的回归测试。"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

import pytest

import config
from novel_agent.contracts import RunStatus
from novel_agent.pathing import ProjectId
from novel_agent.provider_types import ProviderTransportResult
from novel_agent.service import (
    ApproveArtifactRequest,
    NovelAgentService,
    StartNovelRunRequest,
)
from novel_agent.storage import connect_project_database


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")
TERMINAL = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
LONG_TEXT = ("我站在学院门口，晨风拂过衣角，图书馆的钟声在远处响起。" * 120)  # ~3200字


class _TextChapterTransport:
    """Fakes the governed pipeline but returns a PLAIN-TEXT chapter body.

    Mirrors GovernedCreativeAdapter: world/cast/outline are JSON envelopes,
    the chapter raw is the verbatim prose (NOT a JSON draft envelope).
    """

    def __init__(self) -> None:
        self.reviewer_calls = 0

    def generate(self, request) -> ProviderTransportResult:
        prompt = request.prompt
        if "需求分析器" in prompt:
            payload = {"domain": "novel", "intent": "full_novel", "goal": "写小说", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 1}
        elif "规划器" in prompt:
            payload = {"steps": [{"step_id": 1, "description": "设计世界观", "tool": "world.generate@1.0.0", "goal": "世界观", "parameters": None}, {"step_id": 2, "description": "写章节", "tool": "chapter.draft@1.0.0", "goal": "章节", "parameters": None}]}
        elif "资深世界观架构师" in prompt:
            payload = {"title": "现代校园", "premise": "一所普通中学。", "rules": ["现实世界规则"], "constraints": []}
        elif "资深角色设计师" in prompt:
            payload = {"characters": [{"name": "林一", "role": "男主角", "goal": "解开误会", "traits": ["温和"]}]}
        elif "剧情架构师" in prompt:
            payload = {"chapters": [{"title": "第一章", "goal": "目标" * 10, "planned_events": ["事件"]}]}
        elif "华语小说家" in prompt:
            # Plain-text prose: exactly what GovernedCreativeAdapter stores.
            return ProviderTransportResult(content=LONG_TEXT, model_name="t", prompt_tokens=1, completion_tokens=1)
        elif "苛刻的小说评审编辑" in prompt:
            self.reviewer_calls += 1
            payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
        elif "提取章节摘要" in prompt:
            payload = {"summary": "s", "events": []}
        else:
            return ProviderTransportResult(content=LONG_TEXT, model_name="t", prompt_tokens=1, completion_tokens=1)
        return ProviderTransportResult(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            model_name="t",
            prompt_tokens=1,
            completion_tokens=1,
        )


def _wait_terminal(service: NovelAgentService, run_id: str, seconds: float = 10.0) -> RunStatus:
    deadline = monotonic() + seconds
    while (snapshot := service.snapshot(PROJECT_ID, run_id)).status not in TERMINAL:
        assert monotonic() < deadline, f"run did not terminate; status={snapshot.status}"
        sleep(0.01)
    return snapshot.status


@pytest.fixture
def projects_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "projects"
    (root / PROJECT_ID.root).mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(root))
    return root


def _approve_pending(service: NovelAgentService, projects_root: Path, run_id: str, cycle: int) -> RunStatus:
    """Approve whatever gate the run is waiting on (plan or staged artifact)."""
    snapshot = service.snapshot(PROJECT_ID, run_id)
    assert snapshot.status is RunStatus.WAITING_REVIEW, snapshot.status
    if snapshot.review_reason == "plan_approval_required":
        service.approve_plan(PROJECT_ID, run_id, idempotency_key=f"plan-approve-{cycle}")
    else:
        with connect_project_database(projects_root / PROJECT_ID.root) as connection:
            staged = connection.execute(
                "SELECT artifact_id, revision FROM artifacts WHERE run_id = ? AND status = 'staged' "
                "ORDER BY created_at DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        assert staged is not None, f"no staged artifact for review={snapshot.review_reason}"
        service.approve_artifact(
            ApproveArtifactRequest(
                project_id=PROJECT_ID,
                run_id=run_id,
                artifact_id=str(staged[0]),
                revision=int(staged[1]),
                idempotency_key=f"approve-{cycle}",
            )
        )
    return _wait_terminal(service, run_id)


def test_plain_text_chapter_approval_persists(projects_root: Path) -> None:
    """A staged chapter stored as verbatim prose must be approvable (no JSONDecodeError)."""
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=_TextChapterTransport())
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="run-text-chapter",
            idempotency_key="start-text-chapter",
            user_requirements="写一本一章的小说",
            provider="legacy_real",
            auto_approve=False,
        )
    )
    status = _wait_terminal(service, "run-text-chapter")
    assert status is RunStatus.WAITING_REVIEW

    # Walk the approval chain: world -> cast -> outline -> chapter (text).
    for cycle in range(1, 6):
        status = _approve_pending(service, projects_root, "run-text-chapter", cycle)
        if status is RunStatus.SUCCEEDED:
            break
        assert status is RunStatus.WAITING_REVIEW, f"unexpected status {status}"
    assert status is RunStatus.SUCCEEDED

    with connect_project_database(projects_root / PROJECT_ID.root) as connection:
        chapter = connection.execute(
            "SELECT raw_bytes, status, is_current FROM artifacts "
            "WHERE run_id = ? AND kind = 'chapter' AND status = 'accepted'",
            ("run-text-chapter",),
        ).fetchone()
        assert chapter is not None
        # The accepted chapter keeps the verbatim prose (no JSON wrapper).
        assert LONG_TEXT[:40] in bytes(chapter[0]).decode("utf-8", errors="replace")


def test_continuation_run_uses_real_reviewer(projects_root: Path) -> None:
    """A chapter-only (reuse) run must swap in the provider-backed reviewer."""
    transport = _TextChapterTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="run-reuse-review",
            idempotency_key="start-reuse-review",
            user_requirements="写一本一章的小说",
            provider="legacy_real",
            auto_approve=False,
        )
    )
    status = _wait_terminal(service, "run-reuse-review")
    assert status is RunStatus.WAITING_REVIEW
    for cycle in range(1, 6):
        status = _approve_pending(service, projects_root, "run-reuse-review", cycle)
        if status is RunStatus.SUCCEEDED:
            break
        assert status is RunStatus.WAITING_REVIEW
    assert status is RunStatus.SUCCEEDED

    # The quality gate must have used the provider evaluator (LLM review),
    # not the offline fake — the fake never calls the transport.
    assert transport.reviewer_calls > 0
    with connect_project_database(projects_root / PROJECT_ID.root) as connection:
        policy = connection.execute(
            "SELECT report_json FROM quality_reports WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
            ("run-reuse-review",),
        ).fetchone()
        assert policy is not None
        assert "offline-fake-v1" not in str(policy[0])
        assert '"metrics"' in str(policy[0]) or '"outcome"' in str(policy[0])
