from __future__ import annotations

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


class DispatchTransport:
    """Returns contract-shaped content for every governed creative prompt."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, _request) -> ProviderTransportResult:
        self.calls += 1
        if "世界观架构师" in _request.prompt:
            payload = {"title": "晨光学院", "premise": "漂浮的魔法学院。", "rules": ["魔力来自星辰"], "constraints": []}
        elif "角色设计师" in _request.prompt:
            payload = {
                "characters": [
                    {"name": "林晚", "role": "主角", "goal": "查明真相", "traits": ["冷静"]},
                    {"name": "阿泽", "role": "挚友", "goal": "守护", "traits": ["热情"]},
                ]
            }
        elif "剧情架构师" in _request.prompt:
            payload = {
                "chapters": [
                    {"title": f"第{n}章", "goal": f"目标{n}", "planned_events": [f"事件{n}"]}
                    for n in range(1, 3)
                ]
            }
        elif "摘要" in _request.prompt:
            payload = {"summary": "主角抵达学院。"}
        elif "苛刻的小说评审编辑" in _request.prompt:
            payload = {
                "style_consistency": {"score": 4, "reason": "风格一致"},
                "character_consistency": {"score": 4, "reason": "角色符合设定"},
                "prose_fluency": {"score": 4, "reason": "文笔流畅"},
                "scene_completeness": {"score": 4, "reason": "场景完整"},
                "dialogue_naturalness": {"score": 4, "reason": "对话自然"},
                "content_alignment": {"score": 4, "reason": "符合目标"},
            }
        else:
            content = "我站在学院门口，晨风拂过衣角。" * 200  # ~2800 chars, within the length gate
            return ProviderTransportResult(content=content, model_name="dispatch", prompt_tokens=1, completion_tokens=1)
        import json

        return ProviderTransportResult(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            model_name="dispatch",
            prompt_tokens=1,
            completion_tokens=1,
        )


def _wait_terminal(service: NovelAgentService, project_id: ProjectId, run_id: str) -> RunStatus:
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(project_id, run_id)).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate; status={snapshot.status}"
        sleep(0.01)
    return snapshot.status


def test_legacy_real_run_records_tool_call_audit_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a canonical service over a fresh project with a dispatch transport.
    projects_root = tmp_path / "projects"
    project_id = ProjectId("novel-one")
    (projects_root / project_id.root).mkdir(parents=True)
    transport = DispatchTransport()
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)

    # When: a legacy_real run completes through the worker.
    service.start(
        StartNovelRunRequest(
            project_id=project_id,
            run_id="run-audited",
            idempotency_key="start-audited",
            user_requirements="一个魔法学院的故事",
            provider="legacy_real",
            target_chapter_count=2,
        )
    )
    status = _wait_terminal(service, project_id, "run-audited")

    # Then: every creative call is recorded in the audited tool ledger.
    assert status is RunStatus.SUCCEEDED
    with connect_project_database(projects_root / project_id.root) as connection:
        rows = connection.execute(
            "SELECT tool_id, state FROM tool_calls WHERE run_id = ? ORDER BY tool_id",
            ("run-audited",),
        ).fetchall()
        tools = [str(row[0]) for row in rows]
        assert tools == [
            "chapter.draft@1.0.0",
            "chapter.draft@1.0.0",
            "character.generate@1.0.0",
            "continuity.audit@1.0.0",
            "continuity.audit@1.0.0",
            "memory.extract@1.0.0",
            "memory.extract@1.0.0",
            "outline.generate@1.0.0",
            "quality.evaluate@1.0.0",
            "quality.evaluate@1.0.0",
            "world.generate@1.0.0",
        ]
        assert all(str(row[1]) == "succeeded" for row in rows)
        # Planning-stage rows are not bound to a durable task; chapter rows are.
        planning_rows = connection.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE run_id = ? AND tool_id LIKE '%.generate%' AND task_id IS NULL",
            ("run-audited",),
        ).fetchone()
        assert planning_rows == (3,)
        # Every committed memory record (M1 world/cast + M2 plan + M3 summaries) is indexed into the M4 FTS view.
        fts_rows = connection.execute("SELECT COUNT(*) FROM memory_fts").fetchone()
        assert fts_rows == (7,)
        # The run records auditable stage progression events for planning and each chapter.
        stages = [
            str(row[0])
            for row in connection.execute(
                "SELECT payload_json FROM run_events WHERE run_id = ? AND event_type = 'stage_entered' "
                "ORDER BY sequence",
                ("run-audited",),
            )
        ]
        import json

        parsed = [json.loads(payload)["stage"] for payload in stages]
        assert "node_complete:world" in parsed
        assert "node_complete:cast" in parsed
        assert "node_complete:outline" in parsed
        assert parsed.count("chapter_committed") == 2
        assert parsed.count("chapter_quality_reviewed") == 2


def test_offline_fake_run_also_records_tool_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the default offline provider service.
    projects_root = tmp_path / "projects"
    project_id = ProjectId("novel-two")
    (projects_root / project_id.root).mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME)

    # When: an offline_fake run completes.
    service.start(
        StartNovelRunRequest(
            project_id=project_id,
            run_id="run-offline-audit",
            idempotency_key="start-offline-audit",
            user_requirements="任意需求",
            provider="offline_fake",
            target_chapter_count=1,
        )
    )
    status = _wait_terminal(service, project_id, "run-offline-audit")

    # Then: the audited tool ledger contains every fake call.
    assert status is RunStatus.SUCCEEDED
    with connect_project_database(projects_root / project_id.root) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE run_id = ?",
            ("run-offline-audit",),
        ).fetchone()
        assert count == (7,)  # world + cast + outline + draft + extract + audit + review
