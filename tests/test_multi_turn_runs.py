"""Multi-turn console mode: start_auto assigns run ids and reuses terminal runs.

These tests cover the run_id-internalization contract:
- a fresh project gets ``{project}-run-1``,
- re-submitting the SAME requirement on a terminal run reuses its run_id and
  requeues it (retry/continue),
- a NEW requirement on a terminal run allocates the next run_id while
  project-scoped asset reuse still carries the story forward.
"""
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
from novel_agent.service import NovelAgentService, StartNovelRunRequest
from novel_agent.storage import connect_project_database


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")


class MultiTurnTransport:
    """Answers brief analysis with a fixed intent; fakes the rest."""

    def __init__(self, intent: str) -> None:
        self.intent = intent
        self.brief_calls = 0

    def generate(self, request) -> ProviderTransportResult:
        prompt = request.prompt
        if "需求分析器" in prompt:
            self.brief_calls += 1
            payload = {
                "domain": "novel",
                "intent": self.intent,
                "goal": "写小说",
                "constraints": [],
                "style_guidance": [],
                "acceptance_criteria": [],
                "chapter_count": 2,
            }
        elif "资深世界观架构师" in prompt:
            payload = {"title": "现代校园", "premise": "一所普通中学。", "rules": ["现实世界规则"], "constraints": []}
        elif "资深角色设计师" in prompt:
            payload = {
                "characters": [
                    {"name": "林一", "role": "男主角", "goal": "解开误会", "traits": ["温和"]},
                    {"name": "苏晚", "role": "女主角", "goal": "守护友情", "traits": ["倔强"]},
                ]
            }
        elif "剧情架构师" in prompt:
            payload = {
                "chapters": [
                    {"title": f"第{n}章", "goal": f"目标{n}", "planned_events": [f"事件{n}"]}
                    for n in range(1, 3)
                ]
            }
        elif "规划器" in prompt:
            payload = {
                "steps": [
                    {"step_id": 1, "description": "生成大纲", "tool": "outline.generate@1.0.0", "goal": "大纲", "parameters": None},
                    {"step_id": 2, "description": "写章节", "tool": "chapter.draft@1.0.0", "goal": "章节", "parameters": None},
                ]
            }
        elif "苛刻的小说评审编辑" in prompt:
            payload = {
                "style_consistency": {"score": 4, "reason": "r"},
                "character_consistency": {"score": 4, "reason": "r"},
                "prose_fluency": {"score": 4, "reason": "r"},
                "scene_completeness": {"score": 4, "reason": "r"},
                "dialogue_naturalness": {"score": 4, "reason": "r"},
                "content_alignment": {"score": 4, "reason": "r"},
            }
        elif "提取章节摘要" in prompt:
            payload = {"summary": "s", "events": []}
        else:
            return ProviderTransportResult(content="x" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
        return ProviderTransportResult(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            model_name="t",
            prompt_tokens=1,
            completion_tokens=1,
        )


def _wait_terminal(service: NovelAgentService, run_id: str, seconds: float = 20.0) -> RunStatus:
    deadline = monotonic() + seconds
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, run_id)).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate; status={snapshot.status}"
        sleep(0.01)
    return snapshot.status


@pytest.fixture
def projects_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "projects"
    (root / PROJECT_ID.root).mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(root))
    return root


def test_start_auto_assigns_next_run_id(projects_root: Path) -> None:
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=MultiTurnTransport("outline_only"))

    first = service.start_auto(PROJECT_ID, "创作10章大纲")
    second = service.start_auto(PROJECT_ID, "写第一章正文")

    assert first.run_id == "novel-one-run-1"
    assert second.run_id == "novel-one-run-2"
    assert first.status is RunStatus.QUEUED
    assert second.status is RunStatus.QUEUED


def test_start_auto_reuses_terminal_run_id_for_same_requirement(projects_root: Path) -> None:
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=MultiTurnTransport("outline_only"))
    service.start_auto(PROJECT_ID, "创作10章大纲")
    assert _wait_terminal(service, "novel-one-run-1") is RunStatus.SUCCEEDED

    # Same requirement on a finished run: retry on the SAME run id.
    retry = service.start_auto(PROJECT_ID, "创作10章大纲")

    assert retry.run_id == "novel-one-run-1"
    assert retry.status is RunStatus.QUEUED
    with connect_project_database(projects_root / PROJECT_ID.root) as connection:
        requeued = connection.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id = ? AND event_type = 'run_requeued'",
            ("novel-one-run-1",),
        ).fetchone()
        assert requeued == (1,)


def test_start_auto_allocates_new_run_id_for_new_requirement(projects_root: Path) -> None:
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=MultiTurnTransport("outline_only"))
    service.start_auto(PROJECT_ID, "创作10章大纲")
    assert _wait_terminal(service, "novel-one-run-1") is RunStatus.SUCCEEDED

    # New requirement: a fresh auditable cycle, not a requeue.
    next_run = service.start_auto(PROJECT_ID, "基于已有大纲写第一章正文")

    assert next_run.run_id == "novel-one-run-2"
    assert next_run.status is RunStatus.QUEUED
    with connect_project_database(projects_root / PROJECT_ID.root) as connection:
        requeued = connection.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id = ? AND event_type = 'run_requeued'",
            ("novel-one-run-1",),
        ).fetchone()
        assert requeued == (0,)


def test_multi_turn_retry_after_failure_reuses_same_run(projects_root: Path) -> None:
    """A failed run retried via start_auto requeues under the same run id."""
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=MultiTurnTransport("outline_only"))
    service.start_auto(PROJECT_ID, "创作10章大纲")
    assert _wait_terminal(service, "novel-one-run-1") is RunStatus.SUCCEEDED

    # Simulate a failed execution (e.g. provider outage mid-run).
    with connect_project_database(projects_root / PROJECT_ID.root) as connection:
        connection.execute(
            "UPDATE runs SET state = ? WHERE run_id = ?",
            (RunStatus.FAILED.value, "novel-one-run-1"),
        )

    # Retry with the same requirement: same run id, requeued, and it succeeds.
    retry = service.start_auto(PROJECT_ID, "创作10章大纲")
    assert retry.run_id == "novel-one-run-1"
    assert retry.status is RunStatus.QUEUED
    with connect_project_database(projects_root / PROJECT_ID.root) as connection:
        requeued = connection.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id = ? AND event_type = 'run_requeued'",
            ("novel-one-run-1",),
        ).fetchone()
        assert requeued == (1,)
    assert _wait_terminal(service, "novel-one-run-1") is RunStatus.SUCCEEDED


# -- repair (修复重排) ----------------------------------------------------------


class _FlakyWorldTransport:
    """First world generation returns garbage; everything else is valid.

    Used to prove that REPAIR genuinely re-executes the run (the world node
    must call the provider again instead of replaying the stale failed tool
    call) and finishes with all artifacts accepted.
    """

    LONG_TEXT = ("我站在学院门口，晨风拂过衣角，图书馆的钟声在远处响起。" * 120)  # ~3200字

    def __init__(self) -> None:
        self.world_calls = 0

    def generate(self, request) -> ProviderTransportResult:
        prompt = request.prompt
        if "需求分析器" in prompt:
            payload = {"domain": "novel", "intent": "full_novel", "goal": "写小说", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 1}
        elif "规划器" in prompt:
            payload = {
                "steps": [
                    {"step_id": 1, "description": "设计世界观", "tool": "world.generate@1.0.0", "goal": "世界观", "parameters": None},
                    {"step_id": 2, "description": "写章节", "tool": "chapter.draft@1.0.0", "goal": "章节", "parameters": None},
                ]
            }
        elif "资深世界观架构师" in prompt:
            self.world_calls += 1
            if self.world_calls == 1:
                return ProviderTransportResult(content="完全不是JSON", model_name="t", prompt_tokens=1, completion_tokens=1)
            payload = {"title": "现代校园", "premise": "一所普通中学。", "rules": ["现实世界规则"], "constraints": []}
        elif "资深角色设计师" in prompt:
            payload = {"characters": [{"name": "林一", "role": "男主角", "goal": "解开误会", "traits": ["温和"]}]}
        elif "剧情架构师" in prompt:
            payload = {"chapters": [{"title": "第一章", "goal": "目标" * 10, "planned_events": ["事件"]}]}
        elif "华语小说家" in prompt:
            return ProviderTransportResult(content=self.LONG_TEXT, model_name="t", prompt_tokens=1, completion_tokens=1)
        elif "苛刻的小说评审编辑" in prompt:
            payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
        elif "提取章节摘要" in prompt:
            payload = {"summary": "s", "events": []}
        else:
            return ProviderTransportResult(content=self.LONG_TEXT, model_name="t", prompt_tokens=1, completion_tokens=1)
        return ProviderTransportResult(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            model_name="t",
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_repair_requeues_and_recovers_from_schema_failure(projects_root: Path) -> None:
    """REPAIR must genuinely re-execute: provider called again, run succeeds."""
    from novel_agent.service import RepairNovelRunRequest

    transport = _FlakyWorldTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="run-repair",
            idempotency_key="start-repair",
            user_requirements="写一本一章的小说",
            provider="legacy_real",
            auto_approve=True,
        )
    )
    # First attempt: world schema failure pauses the run for repair.
    assert _wait_terminal(service, "run-repair") is RunStatus.WAITING_REVIEW
    assert transport.world_calls == 1

    # Repair requeues AND the re-run must call the provider again (the stale
    # tool-call row must have been purged, otherwise the gateway replays the
    # old failure without a provider call).
    snapshot = service.repair(
        RepairNovelRunRequest(project_id=PROJECT_ID, run_id="run-repair", idempotency_key="repair-1")
    )
    assert snapshot.status is RunStatus.QUEUED
    assert _wait_terminal(service, "run-repair") is RunStatus.SUCCEEDED
    assert transport.world_calls == 2

    with connect_project_database(projects_root / PROJECT_ID.root) as connection:
        kinds = [str(row[0]) for row in connection.execute(
            "SELECT kind FROM artifacts WHERE run_id = ? AND status = 'accepted' ORDER BY kind",
            ("run-repair",),
        ).fetchall()]
        assert kinds == ["chapter", "character_set", "plan_contract", "world_bible"]
        tool_rows = connection.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE run_id = ?", ("run-repair",)
        ).fetchone()
        # The re-execution produced fresh tool-call audit rows (the stale
        # failed row was purged, then the re-run recorded its own calls).
        assert tool_rows[0] > 0


# -- continuation planning (续写规划) -----------------------------------------


class _OverplanningTransport:
    """Planner ALWAYS returns the full world/cast/outline/chapter pipeline.

    Mirrors the real model behaviour seen in production: a "续写第二章"
    request still produces a full regeneration plan.  The system must trim
    the existing-asset steps deterministically so the continuation run only
    drafts chapters.
    """

    def __init__(self) -> None:
        self.world_calls = 0
        self.cast_calls = 0
        self.outline_calls = 0

    def generate(self, request) -> ProviderTransportResult:
        prompt = request.prompt
        if "需求分析器" in prompt:
            payload = {"domain": "novel", "intent": "full_novel", "goal": "写小说", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 2}
        elif "规划器" in prompt:
            payload = {
                "steps": [
                    {"step_id": 1, "description": "设计世界观", "tool": "world.generate@1.0.0", "goal": "世界观", "parameters": None},
                    {"step_id": 2, "description": "设计角色", "tool": "character.generate@1.0.0", "goal": "角色", "parameters": None},
                    {"step_id": 3, "description": "生成大纲", "tool": "outline.generate@1.0.0", "goal": "大纲", "parameters": None},
                    {"step_id": 4, "description": "撰写章节", "tool": "chapter.draft@1.0.0", "goal": "章节", "parameters": None},
                ]
            }
        elif "资深世界观架构师" in prompt:
            self.world_calls += 1
            payload = {"title": "现代校园", "premise": "一所普通中学。", "rules": ["现实世界规则"], "constraints": []}
        elif "资深角色设计师" in prompt:
            self.cast_calls += 1
            payload = {"characters": [{"name": "林一", "role": "男主角", "goal": "解开误会", "traits": ["温和"]}]}
        elif "剧情架构师" in prompt:
            self.outline_calls += 1
            payload = {"chapters": [{"title": f"第{n}章", "goal": "目标" * 10, "planned_events": ["事件"]} for n in range(1, 3)]}
        elif "华语小说家" in prompt:
            return ProviderTransportResult(content=("我站在学院门口，晨风拂过衣角，图书馆的钟声在远处响起。" * 120), model_name="t", prompt_tokens=1, completion_tokens=1)
        elif "苛刻的小说评审编辑" in prompt:
            payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
        elif "提取章节摘要" in prompt:
            payload = {"summary": "s", "events": []}
        else:
            return ProviderTransportResult(content="x" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
        return ProviderTransportResult(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            model_name="t",
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_continuation_run_skips_existing_asset_regeneration(projects_root: Path) -> None:
    """续写请求不得重新生成已有世界观/角色/大纲，只写章节。"""
    transport = _OverplanningTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)

    # First run: full pipeline, everything approved automatically.
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID, run_id="run-first",
            idempotency_key="start-first",
            user_requirements="写一本两章的小说", provider="legacy_real", auto_approve=True,
        )
    )
    assert _wait_terminal(service, "run-first") is RunStatus.SUCCEEDED
    assert transport.world_calls == 1 and transport.cast_calls == 1 and transport.outline_calls == 1

    # Continuation run: the planner still over-plans the full pipeline, but
    # the system must trim world/cast/outline and only draft the chapter.
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID, run_id="run-cont",
            idempotency_key="start-cont",
            user_requirements="续写第二章", provider="legacy_real", auto_approve=True,
        )
    )
    assert _wait_terminal(service, "run-cont") is RunStatus.SUCCEEDED

    # No regeneration of existing assets on the continuation run.
    assert transport.world_calls == 1
    assert transport.cast_calls == 1
    assert transport.outline_calls == 1

    with connect_project_database(projects_root / PROJECT_ID.root) as connection:
        # The continuation run mirrored the plan and wrote chapter 2.
        chapters = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND kind = 'chapter' AND status = 'accepted'",
            ("run-cont",),
        ).fetchone()
        assert chapters[0] >= 1
        # It produced no world/cast artifacts of its own.  (A plan_contract
        # row is expected: the reused outline is mirrored as an accepted
        # artifact of this run so the chapter staging authority accepts it.)
        regenerated = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND kind IN ('world_bible', 'character_set') "
            "AND status = 'accepted'",
            ("run-cont",),
        ).fetchone()
        assert regenerated == (0,)


def test_start_auto_double_submit_does_not_spawn_second_run(projects_root: Path) -> None:
    """Rapid duplicate submission with the same requirement must reuse the
    active run's snapshot instead of allocating a second run."""
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=MultiTurnTransport("outline_only"))
    first = service.start_auto(PROJECT_ID, "创作10章大纲")
    # Second click while the run is still active (QUEUED/RUNNING) with the
    # same requirement: return the existing run, never create run-2.
    second = service.start_auto(PROJECT_ID, "创作10章大纲")

    assert first.run_id == "novel-one-run-1"
    assert second.run_id == "novel-one-run-1"
    with connect_project_database(projects_root / PROJECT_ID.root) as connection:
        runs = connection.execute("SELECT run_id FROM runs ORDER BY run_id").fetchall()
        assert [str(r[0]) for r in runs] == ["novel-one-run-1"]


def test_requeue_rerun_restarts_chapters_from_one(projects_root: Path) -> None:
    """Re-running the same full-novel requirement must restart from chapter 1.

    Regression: after a successful full_novel run, submitting the SAME
    requirement again (requeue on the reused run id) used to rebuild the
    graph starting at the NEXT chapter (the old accepted chapter counted as
    "already written"), which failed with chapter_plan_missing when the new
    outline only covered chapter 1.  The requeue now clears the previous
    cycle's accepted chapters so the story restarts from the beginning.
    """
    from tests.test_chapter_approval import _TextChapterTransport

    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=_TextChapterTransport())
    first = service.start_auto(PROJECT_ID, "写一本一章的小说")
    assert _wait_terminal(service, first.run_id) is RunStatus.SUCCEEDED

    second = service.start_auto(PROJECT_ID, "写一本一章的小说")
    assert second.run_id == first.run_id  # same requirement -> reused run id
    assert _wait_terminal(service, second.run_id) is RunStatus.SUCCEEDED

    with connect_project_database(projects_root / PROJECT_ID.root) as connection:
        chapters = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND kind = 'chapter' AND status = 'accepted'",
            (first.run_id,),
        ).fetchone()
        assert chapters[0] >= 1
