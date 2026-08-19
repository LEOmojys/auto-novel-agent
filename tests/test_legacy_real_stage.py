from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

import pytest

import config
from novel_agent.contracts import ProviderRequest, RunStatus
from novel_agent.pathing import ProjectId
from novel_agent.provider_types import ProviderTransportResult
from novel_agent.service import NovelAgentService, StartNovelRunRequest
from novel_agent.storage import connect_project_database


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


class DispatchTransport:
    """Returns contract-shaped content for every governed creative prompt."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, request: ProviderRequest) -> ProviderTransportResult:
        self.calls.append(request.prompt)
        content = self._dispatch(request.prompt)
        return ProviderTransportResult(
            content=content,
            model_name="dispatch-test",
            prompt_tokens=4,
            completion_tokens=4,
        )

    @staticmethod
    def _dispatch(prompt: str) -> str:
        if "资深世界观架构师" in prompt:
            payload = {
                "title": "晨光学院",
                "premise": "一座漂浮在云海之上的魔法学院，魔力来自星辰。",
                "rules": ["魔力来自星辰", "禁忌魔法不可触碰"],
                "constraints": [],
            }
        elif "资深角色设计师" in prompt:
            payload = {
                "characters": [
                    {"name": "林晚", "role": "主角", "goal": "查明学院禁地之谜", "traits": ["冷静", "执着"]},
                    {"name": "阿泽", "role": "挚友", "goal": "守护林晚", "traits": ["热情"]},
                ]
            }
        elif "资深小说编辑和剧情架构师" in prompt:
            match = re.search(r"必须且只能包含 (\d+) 章", prompt)
            chapter_count = int(match.group(1)) if match else 2
            payload = {
                "chapters": [
                    {"title": f"第{number}章", "goal": f"第{number}章的目标", "planned_events": [f"事件{number}"]}
                    for number in range(1, chapter_count + 1)
                ]
            }
        elif "100-200 字的摘要" in prompt:
            payload = {"summary": "主角抵达学院并完成报到，认识了新室友。"}
        elif "苛刻的小说评审编辑" in prompt:
            payload = {
                "style_consistency": {"score": 4, "reason": "风格一致"},
                "character_consistency": {"score": 4, "reason": "角色言行符合设定"},
                "prose_fluency": {"score": 4, "reason": "文笔流畅"},
                "scene_completeness": {"score": 4, "reason": "场景完整"},
                "dialogue_naturalness": {"score": 4, "reason": "对话自然"},
                "content_alignment": {"score": 4, "reason": "符合章节目标"},
            }
        else:
            return "我站在学院门口，晨风拂过衣角，第一次抬头看见了漂浮的魔法学院。" * 100  # ~3000 chars, within the length gate
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def test_legacy_real_run_generates_real_plan_and_chapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a canonical service whose wire transport answers every creative prompt.
    projects_root = tmp_path / "projects"
    project_id = ProjectId("novel-one")
    (projects_root / project_id.root).mkdir(parents=True)
    transport = DispatchTransport()
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(
        now=lambda: FIXED_TIME,
        provider_transport=transport,
    )

    # When: a legacy_real run with a two-chapter target completes through the worker.
    service.start(
        StartNovelRunRequest(
            project_id=project_id,
            run_id="run-legacy-real",
            idempotency_key="start-legacy-real",
            user_requirements="一个关于魔法学院的成长故事",
            provider="legacy_real",
            target_chapter_count=2,
        )
    )
    deadline = monotonic() + 5.0
    terminal_states = frozenset(
        (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED)
    )
    while (snapshot := service.snapshot(project_id, "run-legacy-real")).status not in terminal_states:
        assert monotonic() < deadline, f"run did not terminate; status={snapshot.status}"
        sleep(0.01)

    # Then: every creative stage used a live provider call and artifacts were accepted.
    assert snapshot.status is RunStatus.SUCCEEDED
    assert len(transport.calls) == 11  # plan + brief + world + cast + outline + 2×(chapter+summary+quality review)
    with connect_project_database(projects_root / project_id.root) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND status = 'accepted'",
            ("run-legacy-real",),
        ).fetchone() == (5,)  # world_bible + character_set + plan_contract + 2 chapters
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE run_id = ? AND layer = 'm3_confirmed'",
            ("run-legacy-real",),
        ).fetchone() == (2,)
        plan = connection.execute(
            "SELECT raw_bytes FROM artifacts WHERE run_id = ? AND kind = 'plan_contract' AND status = 'accepted'",
            ("run-legacy-real",),
        ).fetchone()
        assert plan is not None
        chapter_kinds = connection.execute(
            "SELECT kind FROM artifacts WHERE run_id = ? AND kind = 'chapter' AND status = 'accepted' ORDER BY artifact_id",
            ("run-legacy-real",),
        ).fetchall()
        assert len(chapter_kinds) == 2


def test_legacy_real_run_defaults_chapter_count_when_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the same governed service but without an explicit chapter target.
    projects_root = tmp_path / "projects"
    project_id = ProjectId("novel-two")
    (projects_root / project_id.root).mkdir(parents=True)
    transport = DispatchTransport()
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(
        now=lambda: FIXED_TIME,
        provider_transport=transport,
    )

    # When: the run starts with the default chapter count (10).
    service.start(
        StartNovelRunRequest(
            project_id=project_id,
            run_id="run-legacy-default",
            idempotency_key="start-legacy-default",
            user_requirements="一段奇幻旅程",
            provider="legacy_real",
        )
    )
    deadline = monotonic() + 5.0
    terminal_states = frozenset(
        (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED)
    )
    while (snapshot := service.snapshot(project_id, "run-legacy-default")).status not in terminal_states:
        assert monotonic() < deadline, f"run did not terminate; status={snapshot.status}"
        sleep(0.01)

    # Then: the run completes and every chapter plan is exported as an accepted artifact.
    assert snapshot.status is RunStatus.SUCCEEDED
    assert len(transport.calls) == 35  # plan + brief + world + cast + outline + 10×(chapter+summary+quality review)
    with connect_project_database(projects_root / project_id.root) as connection:
        accepted = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND status = 'accepted'",
            ("run-legacy-default",),
        ).fetchone()
        assert accepted == (13,)  # 3 planning artifacts + 10 chapters
