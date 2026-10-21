from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

import pytest

import config
from novel_agent.agent.brief import BriefAnalyzer, BriefContract
from novel_agent.contracts import RunStatus
from novel_agent.pathing import ProjectId
from novel_agent.provider_types import ProviderFailureKind, ProviderTransportError, ProviderTransportResult
from novel_agent.service import NovelAgentService, StartNovelRunRequest
from novel_agent.storage import connect_project_database


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


class IntentTransport:
    """Answers brief analysis with the given intent, then creative prompts with fakes."""

    def __init__(self, intent: str) -> None:
        self.intent = intent

    def generate(self, request) -> ProviderTransportResult:
        prompt = request.prompt
        if "需求分析器" in prompt:
            payload = {
                "domain": "novel",
                "intent": self.intent,
                "goal": "设计三个中学生角色",
                "constraints": ["两女一男", "三角关系", "17岁"],
                "style_guidance": [],
                "acceptance_criteria": ["角色关系清晰"],
            }
        elif "资深世界观架构师" in prompt:
            payload = {"title": "现代校园", "premise": "一所普通中学。", "rules": ["现实世界规则"], "constraints": []}
        elif "资深角色设计师" in prompt:
            payload = {
                "characters": [
                    {"name": "林一", "role": "男主角", "goal": "解开误会", "traits": ["温和"]},
                    {"name": "苏晚", "role": "女主角", "goal": "守护友情", "traits": ["倔强"]},
                    {"name": "夏雨", "role": "女主角", "goal": "追求梦想", "traits": ["开朗"]},
                ]
            }
        elif "剧情架构师" in prompt:
            payload = {
                "chapters": [
                    {"title": f"第{n}章", "goal": f"目标{n}", "planned_events": [f"事件{n}"]}
                    for n in range(1, 3)
                ]
            }
        else:
            return ProviderTransportResult(
                content="我站在学院门口，晨风拂过衣角。" * 200,
                model_name="t",
                prompt_tokens=1,
                completion_tokens=1,
            )
        return ProviderTransportResult(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            model_name="t",
            prompt_tokens=1,
            completion_tokens=1,
        )


def _analyze(intent: str) -> BriefContract:
    return BriefAnalyzer(transport=IntentTransport(intent), now=lambda: FIXED_TIME).analyze(
        "为我设计3个角色，两女一男，三角关系，17岁中学生",
        project_id=ProjectId("novel-one"),
        run_id="run-001",
        brief_id="run-001:brief",
    )


def test_brief_analyzer_recognizes_characters_only_intent() -> None:
    brief = _analyze("characters_only")

    assert brief.intent == "characters_only"
    assert brief.degraded is False
    assert "两女一男" in brief.constraints


def test_brief_analyzer_recognizes_world_only_intent() -> None:
    brief = _analyze("world_only")

    assert brief.intent == "world_only"


def test_brief_analyzer_defaults_to_full_novel_on_failure() -> None:
    from novel_agent.provider_types import ProviderFailureKind, ProviderTransportError

    class Broken:
        def generate(self, _request) -> ProviderTransportResult:
            raise ProviderTransportError(kind=ProviderFailureKind.TRANSIENT, detail="wire down")

    brief = BriefAnalyzer(transport=Broken(), now=lambda: FIXED_TIME).analyze(
        "写一部奇幻小说",
        project_id=ProjectId("novel-one"),
        run_id="run-002",
        brief_id="run-002:brief",
    )

    assert brief.intent == "full_novel"
    assert brief.degraded is True


@pytest.mark.parametrize(
    ("requirements", "expected_intent"),
    [
        ("为我设计3个角色，一个男主角，两个女主角，三角关系", "characters_only"),
        ("帮我创建几个角色卡", "characters_only"),
        ("新增一个角色到故事里", "add_character"),
        ("设定一个现代校园世界观", "world_only"),
        ("写一份故事大纲", "outline_only"),
        ("写一本20章的奇幻小说", "full_novel"),
        # 续写场景：正文写作不能被"大纲"字样误导为大纲任务
        ("基于已有大纲写第一章正文", "full_novel"),
        ("继续写小说的第二章", "full_novel"),
        ("写第一章正文", "full_novel"),
        ("续写第二章", "full_novel"),
        ("接着写下一章", "full_novel"),
        ("写第三章正文", "full_novel"),
        # 大纲任务：创作性动词 + 大纲，而非仅含"大纲"字样
        ("创作10章大纲", "outline_only"),
        ("基于已有角色和世界观设定，创作一部共10个章节的小说故事大纲", "outline_only"),
        ("写十章节小说大纲", "outline_only"),
    ],
)
def test_degraded_analysis_uses_heuristic_intent(requirements: str, expected_intent: str) -> None:
    from novel_agent.provider_types import ProviderFailureKind, ProviderTransportError

    class Broken:
        def generate(self, _request) -> ProviderTransportResult:
            raise ProviderTransportError(kind=ProviderFailureKind.TRANSIENT, detail="wire down")

    brief = BriefAnalyzer(transport=Broken(), now=lambda: FIXED_TIME).analyze(
        requirements,
        project_id=ProjectId("novel-one"),
        run_id="run-003",
        brief_id="run-003:brief",
    )

    assert brief.degraded is True
    assert brief.intent == expected_intent


def _wait_terminal(service: NovelAgentService, project_id: ProjectId, run_id: str) -> RunStatus:
    deadline = monotonic() + 5.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(project_id, run_id)).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate; status={snapshot.status}"
        sleep(0.01)
    return snapshot.status


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, intent: str, run_id: str) -> None:
    projects_root = tmp_path / "projects"
    project_id = ProjectId("novel-one")
    (projects_root / project_id.root).mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=IntentTransport(intent))
    service.start(
        StartNovelRunRequest(
            project_id=project_id,
            run_id=run_id,
            idempotency_key=f"start-{run_id}",
            user_requirements="为我设计3个角色，两女一男，17岁中学生",
            provider="legacy_real",
        )
    )
    status = _wait_terminal(service, project_id, run_id)
    assert status is RunStatus.SUCCEEDED


def test_characters_only_run_produces_cast_artifact_without_chapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects_root = tmp_path / "projects"
    project_id = ProjectId("novel-one")
    (projects_root / project_id.root).mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=IntentTransport("characters_only"))
    service.start(
        StartNovelRunRequest(
            project_id=project_id,
            run_id="run-cast-only",
            idempotency_key="start-cast-only",
            user_requirements="为我设计3个角色，两女一男，三角关系，17岁中学生",
            provider="legacy_real",
        )
    )
    assert _wait_terminal(service, project_id, "run-cast-only") is RunStatus.SUCCEEDED

    with connect_project_database(projects_root / project_id.root) as connection:
        artifacts = connection.execute(
            "SELECT kind FROM artifacts WHERE run_id = ? AND status = 'accepted' ORDER BY kind",
            ("run-cast-only",),
        ).fetchall()
        assert [str(row[0]) for row in artifacts] == ["character_set"]
        m1 = connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE run_id = ? AND layer = 'm1_canonical'",
            ("run-cast-only",),
        ).fetchone()
        assert m1 == (3,)  # three character identities
        chapter_tasks = connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE run_id = ? AND task_type = 'chapter'",
            ("run-cast-only",),
        ).fetchone()
        assert chapter_tasks == (0,)


def test_world_only_run_produces_world_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    projects_root = tmp_path / "projects"
    project_id = ProjectId("novel-one")
    (projects_root / project_id.root).mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=IntentTransport("world_only"))
    service.start(
        StartNovelRunRequest(
            project_id=project_id,
            run_id="run-world-only",
            idempotency_key="start-world-only",
            user_requirements="设定一个现代校园世界观",
            provider="legacy_real",
        )
    )
    assert _wait_terminal(service, project_id, "run-world-only") is RunStatus.SUCCEEDED

    with connect_project_database(projects_root / project_id.root) as connection:
        artifacts = connection.execute(
            "SELECT kind FROM artifacts WHERE run_id = ? AND status = 'accepted' ORDER BY kind",
            ("run-world-only",),
        ).fetchall()
        assert [str(row[0]) for row in artifacts] == ["world_bible"]
        m1 = connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE run_id = ? AND layer = 'm1_canonical'",
            ("run-world-only",),
        ).fetchone()
        assert m1 == (1,)


# -- out-of-distribution defense ----------------------------------------------


class _OODTransport:
    """Brief analysis always fails; creative prompts are answered with fakes.

    Mirrors ``IntentTransport`` so a recognizable request can still complete
    the full run after the degraded (heuristic) brief path.
    """

    def generate(self, request) -> ProviderTransportResult:
        from novel_agent.provider_types import ProviderFailureKind, ProviderTransportError

        prompt = request.prompt
        if "需求分析器" in prompt:
            raise ProviderTransportError(kind=ProviderFailureKind.TRANSIENT, detail="wire down")
        if "资深世界观架构师" in prompt:
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
                    for n in range(1, 21)
                ]
            }
        elif "苛刻的小说评审编辑" in prompt:
            payload = {
                "style_consistency": {"score": 4, "reason": "风格一致"},
                "character_consistency": {"score": 4, "reason": "人物一致"},
                "prose_fluency": {"score": 4, "reason": "行文流畅"},
                "scene_completeness": {"score": 4, "reason": "场景完整"},
                "dialogue_naturalness": {"score": 4, "reason": "对话自然"},
                "content_alignment": {"score": 4, "reason": "内容契合"},
            }
        elif "提取章节摘要" in prompt:
            payload = {"summary": "本章概要", "events": []}
        elif "规划器" in prompt:
            payload = {
                "steps": [
                    {"step_id": 1, "description": "生成大纲", "tool": "outline.generate@1.0.0", "goal": "大纲", "parameters": None},
                    {"step_id": 2, "description": "写章节", "tool": "chapter.draft@1.0.0", "goal": "章节", "parameters": None},
                ]
            }
        else:
            return ProviderTransportResult(
                content="我站在学院门口，晨风拂过衣角。" * 200,
                model_name="t",
                prompt_tokens=1,
                completion_tokens=1,
            )
        return ProviderTransportResult(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            model_name="t",
            prompt_tokens=1,
            completion_tokens=1,
        )


@pytest.mark.parametrize(
    "requirements",
    [
        "今天天气真好，中午吃什么？",
        "asdfghjklqwertyuiop",
        "哈哈哈哈哈哈",
    ],
)
def test_ood_request_fails_run_with_intent_recognition_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requirements: str,
) -> None:
    """Chatter/gibberish must fail the run, not start a full-novel plan."""
    projects_root = tmp_path / "projects"
    project_id = ProjectId("novel-one")
    (projects_root / project_id.root).mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=_OODTransport())
    service.start(
        StartNovelRunRequest(
            project_id=project_id,
            run_id="run-ood",
            idempotency_key="start-ood",
            user_requirements=requirements,
            provider="legacy_real",
        )
    )
    assert _wait_terminal(service, project_id, "run-ood") is RunStatus.FAILED

    with connect_project_database(projects_root / project_id.root) as connection:
        failure = connection.execute(
            "SELECT payload_json FROM run_events WHERE run_id = ? AND event_type = 'run_failed'",
            ("run-ood",),
        ).fetchone()
        assert failure is not None
        assert json.loads(failure[0])["reason"] == "intent_recognition_failed"
        unrecognized = connection.execute(
            "SELECT payload_json FROM run_events WHERE run_id = ? AND event_type = 'brief_unrecognized'",
            ("run-ood",),
        ).fetchone()
        assert unrecognized is not None
        assert json.loads(unrecognized[0])["requirements"] == requirements
        assert json.loads(unrecognized[0])["reason"] == "provider_failure"
        plans = connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE run_id = ? AND task_type = 'chapter'",
            ("run-ood",),
        ).fetchone()
        assert plans == (0,)


def test_creative_request_still_succeeds_when_provider_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recognizable creative request must NOT be rejected as OOD."""
    projects_root = tmp_path / "projects"
    project_id = ProjectId("novel-one")
    (projects_root / project_id.root).mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=_OODTransport())
    service.start(
        StartNovelRunRequest(
            project_id=project_id,
            run_id="run-valid",
            idempotency_key="start-valid",
            user_requirements="写一本20章的奇幻小说",
            provider="legacy_real",
        )
    )
    assert _wait_terminal(service, project_id, "run-valid") is RunStatus.SUCCEEDED


class _FreshTransport:
    """Brief analysis that flags a brand-new story."""

    def generate(self, request) -> ProviderTransportResult:
        prompt = request.prompt
        if "需求分析器" in prompt:
            payload = {
                "domain": "novel",
                "intent": "full_novel",
                "goal": "换一个全新的太空歌剧故事",
                "constraints": [],
                "style_guidance": [],
                "acceptance_criteria": ["与旧故事完全不同"],
                "fresh": True,
            }
        else:
            payload = {"title": "星际", "premise": "深空。", "rules": [], "constraints": []}
        return ProviderTransportResult(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            model_name="t",
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_brief_analyzer_flags_fresh_story_from_provider() -> None:
    brief = BriefAnalyzer(transport=_FreshTransport(), now=lambda: FIXED_TIME).analyze(
        "换个故事，写一部太空歌剧",
        project_id=ProjectId("novel-one"),
        run_id="run-fresh",
        brief_id="run-fresh:brief",
    )
    assert brief.fresh is True
    assert brief.intent == "full_novel"


class _FailingTransport:
    def generate(self, _request) -> ProviderTransportResult:
        raise ProviderTransportError(kind=ProviderFailureKind.TRANSIENT, detail="wire down")


def test_brief_analyzer_fresh_fallback_on_degraded() -> None:
    brief = BriefAnalyzer(transport=_FailingTransport(), now=lambda: FIXED_TIME).analyze(
        "新写一本完全不同的奇幻小说，不要之前的设定",
        project_id=ProjectId("novel-one"),
        run_id="run-fresh2",
        brief_id="run-fresh2:brief",
    )
    assert brief.fresh is True
    assert brief.degraded is True


def test_brief_analyzer_not_fresh_on_continuation() -> None:
    brief = BriefAnalyzer(transport=_FailingTransport(), now=lambda: FIXED_TIME).analyze(
        "续写第二章",
        project_id=ProjectId("novel-one"),
        run_id="run-cont",
        brief_id="run-cont:brief",
    )
    assert brief.fresh is False
