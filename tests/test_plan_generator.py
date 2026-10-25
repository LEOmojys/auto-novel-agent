from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

import pytest

import config
from novel_agent.agent.brief import BriefContract
from novel_agent.agent.plan_generator import (
    AVAILABLE_PLAN_TOOLS,
    ExecutionPlan,
    PlanGenerator,
    validate_execution_plan,
)
from novel_agent.contracts import Provenance, RunStatus
from novel_agent.pathing import ProjectId
from novel_agent.provider_types import ProviderFailureKind, ProviderTransportError, ProviderTransportResult
from novel_agent.service import NovelAgentService, StartNovelRunRequest
from novel_agent.storage import connect_project_database


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")


def _approve_pending_plan(service: NovelAgentService, run_id: str, key: str) -> None:
    """Confirm the plan-approval gate when auto_approve=False."""
    snapshot = service.snapshot(PROJECT_ID, run_id)
    if snapshot.review_reason == "plan_approval_required":
        service.approve_plan(PROJECT_ID, run_id, key)


def _brief() -> BriefContract:
    return BriefContract(
        brief_id="brief-001",
        project_id=PROJECT_ID,
        raw_requirements="写一本两章的魔法学院小说",
        goal="写小说",
        target_chapter_count=2,
        created_at=FIXED_TIME,
        provenance=Provenance(
            source_kind="user_request",
            source_id="run-001",
            source_sha256="a" * 64,
            observed_at=FIXED_TIME,
        ),
    )


class PlanTransport:
    def __init__(self, steps: list[dict] | None) -> None:
        self.steps = steps

    def generate(self, _request) -> ProviderTransportResult:
        payload = {} if self.steps is None else {"steps": self.steps}
        return ProviderTransportResult(
            content=json.dumps(payload, ensure_ascii=False),
            model_name="t",
            prompt_tokens=1,
            completion_tokens=1,
        )


class BrokenTransport:
    def generate(self, _request) -> ProviderTransportResult:
        raise ProviderTransportError(kind=ProviderFailureKind.TRANSIENT, detail="wire down")


def _generator(transport) -> PlanGenerator:
    return PlanGenerator(transport=transport, now=lambda: FIXED_TIME, run_id="run-001")


def _plan(steps: list[dict]) -> ExecutionPlan:
    raw = json.dumps({"steps": steps}, ensure_ascii=False)
    generator = _generator(PlanTransport(steps))
    plan = generator.generate(_brief())
    assert plan is not None, f"plan rejected: {raw}"
    return plan


def test_generator_builds_validated_execution_plan() -> None:
    plan = _plan(
        [
            {"step_id": 1, "description": "设计世界观", "tool": "world.generate@1.0.0", "goal": "生成世界观"},
            {"step_id": 2, "description": "设计角色", "tool": "character.generate@1.0.0", "goal": "生成角色"},
            {"step_id": 3, "description": "生成大纲", "tool": "outline.generate@1.0.0", "goal": "生成两章大纲"},
            {"step_id": 4, "description": "写第一章", "tool": "chapter.draft@1.0.0", "goal": "第一章正文"},
            {"step_id": 5, "description": "写第二章", "tool": "chapter.draft@1.0.0", "goal": "第二章正文"},
        ]
    )

    assert validate_execution_plan(plan) == ()
    assert len(plan.chapter_steps) == 2
    assert plan.steps[0].tool == "world.generate@1.0.0"


def _raw_plan(steps: list[dict]) -> ExecutionPlan:
    """Build an ExecutionPlan directly (bypassing generator rejection) for validation tests."""
    from novel_agent.agent.plan_generator import PlanStep

    return ExecutionPlan(
        plan_id="plan-x",
        brief_id="brief-001",
        steps=tuple(PlanStep(**step) for step in steps),
        created_at=FIXED_TIME,
        provenance=Provenance(
            source_kind="provider_response",
            source_id="plan-x",
            source_sha256="b" * 64,
            observed_at=FIXED_TIME,
        ),
    )


def test_plan_rejects_unknown_tool() -> None:
    plan = _raw_plan(
        [
            {"step_id": 1, "description": "做奇怪的事", "tool": "hack.system@9.9.9", "goal": "入侵"},
        ]
    )
    assert "unknown_tool:hack.system@9.9.9" in validate_execution_plan(plan)


def test_plan_rejects_chapters_without_outline() -> None:
    plan = _raw_plan(
        [
            {"step_id": 1, "description": "设计世界观", "tool": "world.generate@1.0.0", "goal": "世界观"},
            {"step_id": 2, "description": "写第一章", "tool": "chapter.draft@1.0.0", "goal": "第一章"},
        ]
    )
    assert "chapter_steps_require_outline" in validate_execution_plan(plan)


def test_plan_rejects_cast_without_world() -> None:
    plan = _raw_plan(
        [
            {"step_id": 1, "description": "设计角色", "tool": "character.generate@1.0.0", "goal": "角色"},
        ]
    )
    assert "cast_steps_require_world" in validate_execution_plan(plan)


def test_plan_rejects_non_sequential_step_ids() -> None:
    plan = _raw_plan(
        [
            {"step_id": 5, "description": "设计世界观", "tool": "world.generate@1.0.0", "goal": "世界观"},
        ]
    )
    assert "step_ids_not_sequential" in validate_execution_plan(plan)


def test_generator_degrades_on_provider_failure() -> None:
    plan = _generator(BrokenTransport()).generate(_brief())
    assert plan is None


def test_generator_degrades_on_malformed_output() -> None:
    plan = _generator(PlanTransport(None)).generate(_brief())
    assert plan is None


class FullTransport:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request) -> ProviderTransportResult:
        self.calls += 1
        prompt = request.prompt
        if "需求分析器" in prompt:
            payload = {"domain": "novel", "intent": "full_novel", "goal": "写一本两章小说", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 2}
        elif "规划器" in prompt:
            payload = {
                "steps": [
                    {"step_id": 1, "description": "设计世界观", "tool": "world.generate@1.0.0", "goal": "世界观"},
                    {"step_id": 2, "description": "设计角色", "tool": "character.generate@1.0.0", "goal": "角色"},
                    {"step_id": 3, "description": "生成大纲", "tool": "outline.generate@1.0.0", "goal": "两章大纲"},
                    {"step_id": 4, "description": "写第一章", "tool": "chapter.draft@1.0.0", "goal": "第一章"},
                    {"step_id": 5, "description": "写第二章", "tool": "chapter.draft@1.0.0", "goal": "第二章"},
                ]
            }
        elif "资深世界观架构师" in prompt:
            payload = {"title": "晨光学院", "premise": "魔法学院。", "rules": ["魔力来自星辰"], "constraints": []}
        elif "资深角色设计师" in prompt:
            payload = {"characters": [{"name": "林晚", "role": "主角", "goal": "查明真相", "traits": ["冷静"]}]}
        elif "剧情架构师" in prompt:
            payload = {"chapters": [{"title": f"第{n}章", "goal": f"目标{n}", "planned_events": [f"事件{n}"]} for n in range(1, 3)]}
        elif "华语小说家" in prompt:
            return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
        elif "苛刻的小说评审编辑" in prompt:
            payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
        elif "提取章节摘要" in prompt:
            payload = {"summary": "主角抵达学院。", "events": [{"subject": "林晚", "kind": "direct_event", "predicate": "抵达学院", "object": "学院"}]}
        else:
            return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
        return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)



def test_plan_generated_event_and_llm_driven_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: the model's step list drives the executed task graph."""

    projects_root = tmp_path / "projects"
    (projects_root / PROJECT_ID.root).mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = FullTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="plan-driven",
            idempotency_key="start-plan-driven",
            user_requirements="写一本两章的魔法学院小说",
            provider="legacy_real",
            target_chapter_count=2,
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "plan-driven")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED

    with connect_project_database(projects_root / PROJECT_ID.root) as connection:
        plan_event = connection.execute(
            "SELECT payload_json FROM run_events WHERE run_id = ? AND event_type = 'plan_generated'",
            ("plan-driven",),
        ).fetchone()
        assert plan_event is not None
        payload = json.loads(str(plan_event[0]))
        assert payload["degraded"] is False
        assert "chapter.draft@1.0.0" in payload["steps"]
        chapters = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND kind = 'chapter' AND status = 'accepted'",
            ("plan-driven",),
        ).fetchone()
        assert chapters == (2,)


def test_existing_assets_reused_without_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With existing world/character files, the plan reuses them and never regenerates."""
    from novel_agent.contracts import RunStatus
    from novel_agent.storage import connect_project_database

    class ReuseTransport:
        def __init__(self) -> None:
            self.hits: list[str] = []

        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                self.hits.append("brief")
                payload = {"domain": "novel", "intent": "outline_only", "goal": "写大纲", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 10}
            elif "规划器" in prompt:
                self.hits.append("plan")
                payload = {"steps": [{"step_id": 1, "description": "生成大纲", "tool": "outline.generate@1.0.0", "goal": "基于已有设定生成大纲"}]}
            elif "资深世界观架构师" in prompt:
                self.hits.append("world")  # must never happen
                payload = {"title": "新世界", "premise": "不应被生成。", "rules": [], "constraints": []}
            elif "资深角色设计师" in prompt:
                self.hits.append("cast")  # must never happen
                payload = {"characters": [{"name": "不应存在", "role": "x", "goal": "y", "traits": []}]}
            elif "剧情架构师" in prompt:
                self.hits.append("outline")
                payload = {
                    "chapters": [
                        {"title": f"第{number}章", "goal": "目标", "planned_events": ["事件"]}
                        for number in range(1, 11)
                    ]
                }
            elif "华语小说家" in prompt:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "主角抵达学院。", "events": []}
            else:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    (project_root / "worlds").mkdir(parents=True)
    (project_root / "characters").mkdir(parents=True)
    (project_root / "worlds" / "晨光学院.txt").write_text("一座漂浮的魔法学院，魔力来自星辰。", encoding="utf-8")
    (project_root / "characters" / "林晚.json").write_text(
        json.dumps({"name": "林晚", "role_introduction": "主角", "role_experience": "查明真相"}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = ReuseTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="reuse-assets",
            idempotency_key="start-reuse-assets",
            user_requirements="基于已有设定写一份故事大纲",
            provider="legacy_real",
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "reuse-assets")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED

    # The planner reused the existing files: world/cast generators never ran.
    assert "world" not in transport.hits
    assert "cast" not in transport.hits
    assert "outline" in transport.hits
    with connect_project_database(project_root) as connection:
        kinds = [
            str(row[0])
            for row in connection.execute(
                "SELECT kind FROM artifacts WHERE run_id = ? AND status = 'accepted' ORDER BY kind",
                ("reuse-assets",),
            )
        ]
        assert kinds == ["character_set", "plan_contract", "world_bible"]
        character_memory = connection.execute(
            "SELECT value_json FROM memory_records WHERE run_id = ? AND layer = 'm1_canonical' "
            "AND kind = 'character_identity' LIMIT 1",
            ("reuse-assets",),
        ).fetchone()
        assert character_memory is not None
        assert "林晚" in character_memory[0]


def test_sqlite_memory_assets_reused_without_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validated M1 canonical memory (no files) is reused: world/cast never regenerate."""
    import hashlib

    from novel_agent.storage import migrate

    class MemoryReuseTransport:
        def __init__(self) -> None:
            self.hits: list[str] = []

        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                self.hits.append("brief")
                payload = {"domain": "novel", "intent": "outline_only", "goal": "写大纲", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 10}
            elif "规划器" in prompt:
                self.hits.append("plan")
                payload = {"steps": [{"step_id": 1, "description": "生成大纲", "tool": "outline.generate@1.0.0", "goal": "基于已有设定生成大纲"}]}
            elif "资深世界观架构师" in prompt:
                self.hits.append("world")  # must never happen
                payload = {"title": "新世界", "premise": "不应被生成。", "rules": [], "constraints": []}
            elif "资深角色设计师" in prompt:
                self.hits.append("cast")  # must never happen
                payload = {"characters": [{"name": "不应存在", "role": "x", "goal": "y", "traits": []}]}
            elif "剧情架构师" in prompt:
                self.hits.append("outline")
                payload = {
                    "chapters": [
                        {"title": f"第{number}章", "goal": "目标", "planned_events": ["事件"]}
                        for number in range(1, 11)
                    ]
                }
            elif "华语小说家" in prompt:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "主角抵达学院。", "events": []}
            else:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    earlier = datetime(2029, 12, 31, 12, 0, 0, tzinfo=UTC)
    # Simulate a previous governed run whose canonical memory exists in SQLite
    # (no legacy files on disk): one world rule + one character identity.
    with connect_project_database(project_root) as connection:
        migrate(connection)
        world_value = json.dumps(
            {
                "world_id": "prior:world",
                "brief_id": "prior:brief",
                "title": "晨光学院",
                "premise": "一座漂浮的魔法学院，魔力来自星辰。",
                "rules": ["魔力来自星辰"],
                "constraints": [],
                "created_at": earlier.isoformat(),
                "provenance": {"source_kind": "provider", "source_id": "prior"},
            },
            ensure_ascii=False,
        )
        cast_value = json.dumps(
            {
                "character_id": "prior:char:1",
                "name": "林晚",
                "role": "主角",
                "goal": "查明真相",
                "traits": [],
                "created_at": earlier.isoformat(),
                "provenance": {"source_kind": "provider", "source_id": "prior"},
            },
            ensure_ascii=False,
        )
        # Foreign keys: seed the prior run and its source artifacts before memory.
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) "
            "VALUES ('prior-run', ?, 'succeeded', ?, ?)",
            (PROJECT_ID.root, earlier.isoformat(), earlier.isoformat()),
        )
        for artifact_id, kind, value in (
            ("prior:world", "world_bible", world_value),
            ("prior:char:1", "character_set", cast_value),
        ):
            connection.execute(
                "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
                "schema_version, is_current, created_at) VALUES (?, 1, ?, 'prior-run', ?, 'accepted', ?, ?, 1, 1, ?)",
                (
                    artifact_id,
                    PROJECT_ID.root,
                    kind,
                    value.encode(),
                    hashlib.sha256(value.encode()).hexdigest(),
                    earlier.isoformat(),
                ),
            )
        for memory_id, kind, subject, predicate, value in (
            ("prior:m1:world", "world_rule", "prior:world", "world_bible", world_value),
            ("prior:m1:char:1", "character_identity", "prior:char:1", "character_identity", cast_value),
        ):
            connection.execute(
                "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, "
                "subject_id, predicate, value_json, source_artifact_id, source_artifact_revision, source_span, "
                "source_hash, confidence, created_by, created_at) "
                "VALUES (?, ?, 'prior-run', 1, 'm1_canonical', ?, 'validated', ?, ?, ?, ?, 1, '0:1', ?, 1.0, "
                "'test-seed:v1', ?)",
                (
                    memory_id,
                    PROJECT_ID.root,
                    kind,
                    subject,
                    predicate,
                    value,
                    subject,
                    hashlib.sha256(value.encode()).hexdigest(),
                    earlier.isoformat(),
                ),
            )
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = MemoryReuseTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="reuse-memory",
            idempotency_key="start-reuse-memory",
            user_requirements="基于已有设定写一份故事大纲",
            provider="legacy_real",
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "reuse-memory")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED

    # The planner reused the canonical memory: world/cast generators never ran.
    assert "world" not in transport.hits
    assert "cast" not in transport.hits
    assert "outline" in transport.hits
    with connect_project_database(project_root) as connection:
        kinds = [
            str(row[0])
            for row in connection.execute(
                "SELECT kind FROM artifacts WHERE run_id = ? AND status = 'accepted' ORDER BY kind",
                ("reuse-memory",),
            )
        ]
        assert kinds == ["character_set", "plan_contract", "world_bible"]
        world_artifact = connection.execute(
            "SELECT raw_bytes FROM artifacts WHERE run_id = ? AND kind = 'world_bible' AND status = 'accepted'",
            ("reuse-memory",),
        ).fetchone()
        assert world_artifact is not None
        assert "晨光学院" in bytes(world_artifact[0]).decode()
        character_memory = connection.execute(
            "SELECT value_json FROM memory_records WHERE run_id = ? AND layer = 'm1_canonical' "
            "AND kind = 'character_identity' LIMIT 1",
            ("reuse-memory",),
        ).fetchone()
        assert character_memory is not None
        assert "林晚" in character_memory[0]


def test_partial_cast_revise_keeps_untouched_characters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """character.revise re-designs only the named characters; others are kept."""
    from novel_agent.contracts import RunStatus
    from novel_agent.storage import connect_project_database

    class ReviseTransport:
        def __init__(self) -> None:
            self.hits: list[str] = []

        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                self.hits.append("brief")
                payload = {"domain": "novel", "intent": "characters_only", "goal": "重做部分角色", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 10}
            elif "规划器" in prompt:
                self.hits.append("plan")
                payload = {
                    "steps": [
                        {
                            "step_id": 1,
                            "description": "重新设计林晚",
                            "tool": "character.revise@1.0.0",
                            "goal": "重新设计角色林晚",
                            "parameters": {"target_names": ["林晚"]},
                        }
                    ]
                }
            elif "资深世界观架构师" in prompt:
                self.hits.append("world")  # must never happen
                payload = {"title": "新世界", "premise": "不应被生成。", "rules": [], "constraints": []}
            elif "资深角色设计师" in prompt:
                if "保持不变的角色" in prompt:
                    self.hits.append("revise")
                    payload = {"characters": [{"name": "林晚", "role": "女主", "goal": "新目标", "traits": ["新特质"]}]}
                else:
                    self.hits.append("cast")  # must never happen (full regenerate)
                    payload = {"characters": [{"name": "不应存在", "role": "x", "goal": "y", "traits": []}]}
            elif "剧情架构师" in prompt:
                self.hits.append("outline")
                payload = {"chapters": []}
            elif "华语小说家" in prompt:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "主角抵达学院。", "events": []}
            else:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    (project_root / "worlds").mkdir(parents=True)
    (project_root / "characters").mkdir(parents=True)
    (project_root / "worlds" / "晨光学院.txt").write_text("一座漂浮的魔法学院，魔力来自星辰。", encoding="utf-8")
    (project_root / "characters" / "林晚.json").write_text(
        json.dumps({"name": "林晚", "role_introduction": "旧主角", "role_experience": "旧目标"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (project_root / "characters" / "陈默.json").write_text(
        json.dumps({"name": "陈默", "role_introduction": "导师", "role_experience": "守护学院"}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = ReviseTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="partial-revise",
            idempotency_key="start-partial-revise",
            user_requirements="我对林晚这个角色不满意，请重新设计她，其他角色保持不变",
            provider="legacy_real",
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "partial-revise")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED

    # The planner used the revise tool: no world generation, no full cast regeneration.
    assert "world" not in transport.hits
    assert "cast" not in transport.hits
    assert "revise" in transport.hits
    with connect_project_database(project_root) as connection:
        cast_artifact = connection.execute(
            "SELECT raw_bytes FROM artifacts WHERE run_id = ? AND kind = 'character_set' "
            "AND status = 'accepted' ORDER BY revision DESC LIMIT 1",
            ("partial-revise",),
        ).fetchone()
        assert cast_artifact is not None
        cast_json = bytes(cast_artifact[0]).decode()
        assert "林晚" in cast_json
        assert "陈默" in cast_json
        assert "新目标" in cast_json  # revised Lin Wan
        assert "旧主角" not in cast_json  # old Lin Wan card is gone
        assert "守护学院" in cast_json  # Chen Mo untouched


@pytest.mark.parametrize(
    ("steps", "expected_kinds"),
    [
        (
            [
                {"step_id": 1, "description": "设定世界观", "tool": "world.generate@1.0.0", "goal": "世界观"},
            ],
            ["world_bible"],
        ),
        (
            [
                {"step_id": 1, "description": "设定世界观", "tool": "world.generate@1.0.0", "goal": "世界观"},
                {"step_id": 2, "description": "设计角色", "tool": "character.generate@1.0.0", "goal": "角色"},
            ],
            ["character_set"],  # world stays a background artifact, like characters_only
        ),
    ],
)
def test_llm_plan_without_chapters_completes_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: list[dict],
    expected_kinds: list[str],
) -> None:
    """Any valid LLM step combination terminates and commits its artifacts."""
    from novel_agent.contracts import RunStatus
    from novel_agent.storage import connect_project_database

    class SparsePlanTransport:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request) -> ProviderTransportResult:
            self.calls += 1
            prompt = request.prompt
            if "需求分析器" in prompt:
                payload = {"domain": "novel", "intent": "full_novel", "goal": "任务", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 10}
            elif "规划器" in prompt:
                payload = {"steps": steps}
            elif "资深世界观架构师" in prompt:
                payload = {"title": "晨光学院", "premise": "魔法学院。", "rules": ["魔力来自星辰"], "constraints": []}
            elif "资深角色设计师" in prompt:
                payload = {"characters": [{"name": "林晚", "role": "主角", "goal": "查明真相", "traits": ["冷静"]}]}
            elif "剧情架构师" in prompt:
                payload = {"chapters": [{"title": "第1章", "goal": "目标", "planned_events": ["事件"]}]}
            elif "华语小说家" in prompt:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "主角抵达学院。", "events": []}
            else:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    (projects_root / PROJECT_ID.root).mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=SparsePlanTransport())
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="sparse-plan",
            idempotency_key="start-sparse-plan",
            user_requirements="任务",
            provider="legacy_real",
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "sparse-plan")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED

    with connect_project_database(projects_root / PROJECT_ID.root) as connection:
        kinds = [
            str(row[0])
            for row in connection.execute(
                "SELECT kind FROM artifacts WHERE run_id = ? AND status = 'accepted' ORDER BY kind",
                ("sparse-plan",),
            )
        ]
        assert kinds == expected_kinds
        chapter_tasks = connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE run_id = ? AND task_type = 'chapter'",
            ("sparse-plan",),
        ).fetchone()
        assert chapter_tasks == (0,)


def test_existing_outline_reused_for_single_chapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With an accepted plan contract, chapter steps reuse it: no outline regeneration."""
    import hashlib

    from novel_agent.contracts import RunStatus
    from novel_agent.storage import migrate

    class OutlineReuseTransport:
        def __init__(self) -> None:
            self.hits: list[str] = []

        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                self.hits.append("brief")
                payload = {"domain": "novel", "intent": "full_novel", "goal": "写第一章", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 10}
            elif "规划器" in prompt:
                self.hits.append("plan")
                payload = {
                    "steps": [
                        {
                            "step_id": 1,
                            "description": "写第一章",
                            "tool": "chapter.draft@1.0.0",
                            "goal": "按已有大纲写第一章",
                            "parameters": {"chapter_number": 1},
                        }
                    ]
                }
            elif "资深世界观架构师" in prompt:
                self.hits.append("world")  # must never happen
                payload = {"title": "新世界", "premise": "不应被生成。", "rules": [], "constraints": []}
            elif "资深角色设计师" in prompt:
                self.hits.append("cast")  # must never happen
                payload = {"characters": [{"name": "不应存在", "role": "x", "goal": "y", "traits": []}]}
            elif "剧情架构师" in prompt:
                self.hits.append("outline")  # must never happen
                payload = {"chapters": []}
            elif "华语小说家" in prompt:
                self.hits.append("chapter")
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "主角抵达学院。", "events": []}
            else:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    earlier = datetime(2029, 12, 31, 12, 0, 0, tzinfo=UTC)
    # Seed a prior governed run: world + cast + accepted plan contract (10 chapters).
    from novel_agent.contracts import (
        ChapterPlan,
        EvidenceRef,
        PlanApprovalStatus,
        PlanContract,
        Provenance,
        ProvenanceKind,
    )

    prior_provenance = Provenance(
        source_kind=ProvenanceKind.PROVIDER_RESPONSE,
        source_id="prior",
        source_sha256="1" * 64,
        observed_at=earlier,
    )
    prior_evidence = EvidenceRef(
        evidence_id="prior:evidence",
        artifact_id="prior:world",
        artifact_revision=1,
        source_sha256="2" * 64,
        recorded_at=earlier,
    )
    plan = PlanContract(
        plan_id="prior:plan",
        brief_id="prior:brief",
        world_id="prior:world",
        character_set_id="prior:cast",
        approval_status=PlanApprovalStatus.VALIDATED,
        chapters=tuple(
            ChapterPlan(
                chapter_plan_id=f"prior:chapter:{number}",
                chapter_number=number,
                title=f"第{number}章",
                goal="目标",
                planned_events=("事件",),
                created_at=earlier,
                provenance=prior_provenance,
                evidence=(prior_evidence,),
            )
            for number in range(1, 11)
        ),
        created_at=earlier,
        provenance=prior_provenance,
    )
    plan_value = plan.model_dump_json()
    with connect_project_database(project_root) as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) "
            "VALUES ('prior-run', ?, 'succeeded', ?, ?)",
            (PROJECT_ID.root, earlier.isoformat(), earlier.isoformat()),
        )
        for artifact_id, kind, value in (
            ("prior:world", "world_bible", json.dumps({"title": "晨光学院", "premise": "魔法学院。"}, ensure_ascii=False)),
            ("prior:cast", "character_set", json.dumps({"characters": [{"name": "林晚"}]}, ensure_ascii=False)),
            ("prior:plan", "plan_contract", plan_value),
        ):
            connection.execute(
                "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
                "schema_version, is_current, created_at) VALUES (?, 1, ?, 'prior-run', ?, 'accepted', ?, ?, 1, 1, ?)",
                (
                    artifact_id,
                    PROJECT_ID.root,
                    kind,
                    value.encode(),
                    hashlib.sha256(value.encode()).hexdigest(),
                    earlier.isoformat(),
                ),
            )
        # M1 canonical memory mirrors the artifacts (as a governed run would leave).
        for memory_id, kind, subject, predicate, value in (
            ("prior:m1:world", "world_rule", "prior:world", "world_bible", json.dumps({"title": "晨光学院", "premise": "魔法学院。"}, ensure_ascii=False)),
            ("prior:m1:cast", "character_identity", "prior:cast", "character_identity", json.dumps({"name": "林晚", "role": "主角", "goal": "查明真相"}, ensure_ascii=False)),
        ):
            connection.execute(
                "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, "
                "subject_id, predicate, value_json, source_artifact_id, source_artifact_revision, source_span, "
                "source_hash, confidence, created_by, created_at) "
                "VALUES (?, ?, 'prior-run', 1, 'm1_canonical', ?, 'validated', ?, ?, ?, ?, 1, '0:1', ?, 1.0, "
                "'test-seed:v1', ?)",
                (
                    memory_id,
                    PROJECT_ID.root,
                    kind,
                    subject,
                    predicate,
                    value,
                    subject,
                    hashlib.sha256(value.encode()).hexdigest(),
                    earlier.isoformat(),
                ),
            )
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = OutlineReuseTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="reuse-outline",
            idempotency_key="start-reuse-outline",
            user_requirements="设定和大纲都准备好了，请创作第一章",
            provider="legacy_real",
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "reuse-outline")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED

    # The plan reused the outline: world/cast/outline generators never ran.
    assert "world" not in transport.hits
    assert "cast" not in transport.hits
    assert "outline" not in transport.hits
    assert "chapter" in transport.hits
    with connect_project_database(project_root) as connection:
        chapter = connection.execute(
            "SELECT raw_bytes FROM artifacts WHERE run_id = ? AND kind = 'chapter' AND status = 'accepted'",
            ("reuse-outline",),
        ).fetchone()
        assert chapter is not None
        assert "我站在学院门口" in bytes(chapter[0]).decode()


def test_existing_outline_skips_already_written_chapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When chapter 1 was already produced under the reused outline, chapter-1 steps skip."""
    import hashlib

    from novel_agent.contracts import RunStatus
    from novel_agent.storage import migrate

    class SkipChapterTransport:
        def __init__(self) -> None:
            self.hits: list[str] = []

        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                self.hits.append("brief")
                payload = {"domain": "novel", "intent": "full_novel", "goal": "继续创作", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 10}
            elif "规划器" in prompt:
                self.hits.append("plan")
                payload = {
                    "steps": [
                        {
                            "step_id": 1,
                            "description": "写第一章",
                            "tool": "chapter.draft@1.0.0",
                            "goal": "写第一章",
                            "parameters": {"chapter_number": 1},
                        }
                    ]
                }
            elif "资深世界观架构师" in prompt:
                self.hits.append("world")
                payload = {"title": "新世界", "premise": "不应被生成。", "rules": [], "constraints": []}
            elif "资深角色设计师" in prompt:
                self.hits.append("cast")
                payload = {"characters": [{"name": "不应存在", "role": "x", "goal": "y", "traits": []}]}
            elif "剧情架构师" in prompt:
                self.hits.append("outline")
                payload = {"chapters": []}
            elif "华语小说家" in prompt:
                self.hits.append("chapter")  # must never happen
                return ProviderTransportResult(content="不应被生成。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "主角抵达学院。", "events": []}
            else:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    earlier = datetime(2029, 12, 31, 12, 0, 0, tzinfo=UTC)
    from novel_agent.contracts import (
        ChapterPlan,
        EvidenceRef,
        PlanApprovalStatus,
        PlanContract,
        Provenance,
        ProvenanceKind,
    )

    prior_provenance = Provenance(
        source_kind=ProvenanceKind.PROVIDER_RESPONSE,
        source_id="prior",
        source_sha256="1" * 64,
        observed_at=earlier,
    )
    prior_evidence = EvidenceRef(
        evidence_id="prior:evidence",
        artifact_id="prior:world",
        artifact_revision=1,
        source_sha256="2" * 64,
        recorded_at=earlier,
    )
    plan = PlanContract(
        plan_id="prior:plan",
        brief_id="prior:brief",
        world_id="prior:world",
        character_set_id="prior:cast",
        approval_status=PlanApprovalStatus.VALIDATED,
        chapters=tuple(
            ChapterPlan(
                chapter_plan_id=f"prior:chapter:{number}",
                chapter_number=number,
                title=f"第{number}章",
                goal="目标",
                planned_events=("事件",),
                created_at=earlier,
                provenance=prior_provenance,
                evidence=(prior_evidence,),
            )
            for number in range(1, 11)
        ),
        created_at=earlier,
        provenance=prior_provenance,
    )
    plan_value = plan.model_dump_json()
    with connect_project_database(project_root) as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) "
            "VALUES ('prior-run', ?, 'succeeded', ?, ?)",
            (PROJECT_ID.root, earlier.isoformat(), earlier.isoformat()),
        )
        for artifact_id, kind, value in (
            ("prior:world", "world_bible", json.dumps({"title": "晨光学院", "premise": "魔法学院。"}, ensure_ascii=False)),
            ("prior:cast", "character_set", json.dumps({"characters": [{"name": "林晚"}]}, ensure_ascii=False)),
            ("prior:plan", "plan_contract", plan_value),
            ("prior:chapter:1", "chapter", json.dumps({"chapter_plan_id": "prior:chapter:1", "body": "已写好的第一章。"}, ensure_ascii=False)),
        ):
            connection.execute(
                "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
                "schema_version, is_current, created_at) VALUES (?, 1, ?, 'prior-run', ?, 'accepted', ?, ?, 1, 1, ?)",
                (
                    artifact_id,
                    PROJECT_ID.root,
                    kind,
                    value.encode(),
                    hashlib.sha256(value.encode()).hexdigest(),
                    earlier.isoformat(),
                ),
            )
        # M1 canonical memory mirrors the artifacts (as a governed run would leave).
        for memory_id, kind, subject, predicate, value in (
            ("prior:m1:world", "world_rule", "prior:world", "world_bible", json.dumps({"title": "晨光学院", "premise": "魔法学院。"}, ensure_ascii=False)),
            ("prior:m1:cast", "character_identity", "prior:cast", "character_identity", json.dumps({"name": "林晚", "role": "主角", "goal": "查明真相"}, ensure_ascii=False)),
        ):
            connection.execute(
                "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, "
                "subject_id, predicate, value_json, source_artifact_id, source_artifact_revision, source_span, "
                "source_hash, confidence, created_by, created_at) "
                "VALUES (?, ?, 'prior-run', 1, 'm1_canonical', ?, 'validated', ?, ?, ?, ?, 1, '0:1', ?, 1.0, "
                "'test-seed:v1', ?)",
                (
                    memory_id,
                    PROJECT_ID.root,
                    kind,
                    subject,
                    predicate,
                    value,
                    subject,
                    hashlib.sha256(value.encode()).hexdigest(),
                    earlier.isoformat(),
                ),
            )
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = SkipChapterTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="skip-written",
            idempotency_key="start-skip-written",
            user_requirements="第一章已经写好了，请继续",
            provider="legacy_real",
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "skip-written")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED

    # Chapter 1 was already written under the reused outline: no redraft.
    assert "chapter" not in transport.hits
    with connect_project_database(project_root) as connection:
        written = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND kind = 'chapter'",
            ("skip-written",),
        ).fetchone()
        assert written == (0,)


def test_continuation_after_written_chapter_starts_at_next_chapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare chapter step after chapter 1 is written continues at chapter 2."""
    import hashlib

    from novel_agent.contracts import RunStatus
    from novel_agent.storage import migrate

    class ContinueTransport:
        def __init__(self) -> None:
            self.hits: list[str] = []

        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                self.hits.append("brief")
                payload = {"domain": "novel", "intent": "full_novel", "goal": "继续创作", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 10}
            elif "规划器" in prompt:
                self.hits.append("plan")
                payload = {
                    "steps": [
                        {
                            "step_id": 1,
                            "description": "继续写正文",
                            "tool": "chapter.draft@1.0.0",
                            "goal": "继续写后续章节",
                            "parameters": None,
                        }
                    ]
                }
            elif "资深世界观架构师" in prompt:
                self.hits.append("world")
                payload = {"title": "新世界", "premise": "不应被生成。", "rules": [], "constraints": []}
            elif "资深角色设计师" in prompt:
                self.hits.append("cast")
                payload = {"characters": [{"name": "不应存在", "role": "x", "goal": "y", "traits": []}]}
            elif "剧情架构师" in prompt:
                self.hits.append("outline")
                payload = {"chapters": []}
            elif "华语小说家" in prompt:
                self.hits.append("chapter")
                return ProviderTransportResult(content="续写的第二章正文，剧情继续推进。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "第二章剧情推进。", "events": []}
            else:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    earlier = datetime(2029, 12, 31, 12, 0, 0, tzinfo=UTC)
    from novel_agent.contracts import (
        ChapterPlan,
        EvidenceRef,
        PlanApprovalStatus,
        PlanContract,
        Provenance,
        ProvenanceKind,
    )

    prior_provenance = Provenance(
        source_kind=ProvenanceKind.PROVIDER_RESPONSE,
        source_id="prior",
        source_sha256="1" * 64,
        observed_at=earlier,
    )
    prior_evidence = EvidenceRef(
        evidence_id="prior:evidence",
        artifact_id="prior:world",
        artifact_revision=1,
        source_sha256="2" * 64,
        recorded_at=earlier,
    )
    plan = PlanContract(
        plan_id="prior:plan",
        brief_id="prior:brief",
        world_id="prior:world",
        character_set_id="prior:cast",
        approval_status=PlanApprovalStatus.VALIDATED,
        chapters=tuple(
            ChapterPlan(
                chapter_plan_id=f"prior:chapter:{number}",
                chapter_number=number,
                title=f"第{number}章",
                goal="目标",
                planned_events=("事件",),
                created_at=earlier,
                provenance=prior_provenance,
                evidence=(prior_evidence,),
            )
            for number in range(1, 11)
        ),
        created_at=earlier,
        provenance=prior_provenance,
    )
    plan_value = plan.model_dump_json()
    with connect_project_database(project_root) as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) "
            "VALUES ('prior-run', ?, 'succeeded', ?, ?)",
            (PROJECT_ID.root, earlier.isoformat(), earlier.isoformat()),
        )
        for artifact_id, kind, value in (
            ("prior:world", "world_bible", json.dumps({"title": "晨光学院", "premise": "魔法学院。"}, ensure_ascii=False)),
            ("prior:cast", "character_set", json.dumps({"characters": [{"name": "林晚"}]}, ensure_ascii=False)),
            ("prior:plan", "plan_contract", plan_value),
            ("prior:chapter:1", "chapter", json.dumps({"chapter_plan_id": "prior:chapter:1", "body": "已写好的第一章。"}, ensure_ascii=False)),
        ):
            connection.execute(
                "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
                "schema_version, is_current, created_at) VALUES (?, 1, ?, 'prior-run', ?, 'accepted', ?, ?, 1, 1, ?)",
                (
                    artifact_id,
                    PROJECT_ID.root,
                    kind,
                    value.encode(),
                    hashlib.sha256(value.encode()).hexdigest(),
                    earlier.isoformat(),
                ),
            )
        for memory_id, kind, subject, predicate, value in (
            ("prior:m1:world", "world_rule", "prior:world", "world_bible", json.dumps({"title": "晨光学院", "premise": "魔法学院。"}, ensure_ascii=False)),
            ("prior:m1:cast", "character_identity", "prior:cast", "character_identity", json.dumps({"name": "林晚", "role": "主角", "goal": "查明真相"}, ensure_ascii=False)),
        ):
            connection.execute(
                "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, "
                "subject_id, predicate, value_json, source_artifact_id, source_artifact_revision, source_span, "
                "source_hash, confidence, created_by, created_at) "
                "VALUES (?, ?, 'prior-run', 1, 'm1_canonical', ?, 'validated', ?, ?, ?, ?, 1, '0:1', ?, 1.0, "
                "'test-seed:v1', ?)",
                (
                    memory_id,
                    PROJECT_ID.root,
                    kind,
                    subject,
                    predicate,
                    value,
                    subject,
                    hashlib.sha256(value.encode()).hexdigest(),
                    earlier.isoformat(),
                ),
            )
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = ContinueTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="continue-next",
            idempotency_key="start-continue-next",
            user_requirements="第一章已经写好了，请继续写后面的章节",
            provider="legacy_real",
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "continue-next")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED

    # The single bare chapter step continued at chapter 2: chapter 1 was not
    # rewritten, chapter 2 was drafted once.
    assert "chapter" in transport.hits
    assert transport.hits.count("chapter") == 1
    with connect_project_database(project_root) as connection:
        chapter = connection.execute(
            "SELECT raw_bytes FROM artifacts WHERE run_id = ? AND kind = 'chapter' AND status = 'accepted'",
            ("continue-next",),
        ).fetchone()
        assert chapter is not None
        body = bytes(chapter[0]).decode()
        assert "续写的第二章正文" in body
        assert "已写好的第一章" not in body


def test_revise_chapter_rewrites_and_waits_for_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REVISE redrafts a chapter as a non-activating revision, then proposes a
    rebase so the user can confirm (or reject) the swap."""
    import hashlib

    from novel_agent.contracts import RunStatus
    from novel_agent.storage import migrate

    class ReviseTransport:
        def __init__(self) -> None:
            self.hits: list[str] = []

        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                self.hits.append("brief")
                payload = {"domain": "novel", "intent": "full_novel", "goal": "重写章节", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 10}
            elif "规划器" in prompt:
                self.hits.append("plan")
                payload = {"steps": []}
            elif "资深世界观架构师" in prompt:
                self.hits.append("world")
                payload = {"title": "新世界", "premise": "x", "rules": [], "constraints": []}
            elif "资深角色设计师" in prompt:
                self.hits.append("cast")
                payload = {"characters": [{"name": "x", "role": "x", "goal": "y", "traits": []}]}
            elif "剧情架构师" in prompt:
                self.hits.append("outline")
                payload = {"chapters": []}
            elif "华语小说家" in prompt:
                self.hits.append("chapter")
                return ProviderTransportResult(content="重写后的第二章正文，焕然一新。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "重写后的第二章。", "events": []}
            else:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    earlier = datetime(2029, 12, 31, 12, 0, 0, tzinfo=UTC)
    from novel_agent.contracts import (
        ChapterPlan,
        EvidenceRef,
        PlanApprovalStatus,
        PlanContract,
        Provenance,
        ProvenanceKind,
    )

    prior_provenance = Provenance(
        source_kind=ProvenanceKind.PROVIDER_RESPONSE,
        source_id="prior",
        source_sha256="1" * 64,
        observed_at=earlier,
    )
    prior_evidence = EvidenceRef(
        evidence_id="prior:evidence",
        artifact_id="prior:world",
        artifact_revision=1,
        source_sha256="2" * 64,
        recorded_at=earlier,
    )
    plan = PlanContract(
        plan_id="prior:plan",
        brief_id="prior:brief",
        world_id="prior:world",
        character_set_id="prior:cast",
        approval_status=PlanApprovalStatus.VALIDATED,
        chapters=tuple(
            ChapterPlan(
                chapter_plan_id=f"prior:chapter:{number}",
                chapter_number=number,
                title=f"第{number}章",
                goal="目标",
                planned_events=("事件",),
                created_at=earlier,
                provenance=prior_provenance,
                evidence=(prior_evidence,),
            )
            for number in range(1, 11)
        ),
        created_at=earlier,
        provenance=prior_provenance,
    )
    plan_value = plan.model_dump_json()
    with connect_project_database(project_root) as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) "
            "VALUES ('prior-run', ?, 'succeeded', ?, ?)",
            (PROJECT_ID.root, earlier.isoformat(), earlier.isoformat()),
        )
        from novel_agent.run_models import StartRunExecutionPayload

        start_payload = StartRunExecutionPayload(
            project_id=PROJECT_ID,
            run_id="prior-run",
            idempotency_key="start-prior-run",
            user_requirements="写一本小说",
            provider="legacy_real",
        )
        connection.execute(
            "INSERT INTO commands(command_id, run_id, command_type, idempotency_key, payload_json, state, created_at) "
            "VALUES (?, 'prior-run', 'start', 'start-prior-run', ?, 'accepted', ?)",
            (
                hashlib.sha256("prior-run:start-prior-run".encode()).hexdigest(),
                start_payload.model_dump_json(),
                earlier.isoformat(),
            ),
        )
        for artifact_id, kind, value in (
            ("prior:world", "world_bible", json.dumps({"title": "晨光学院", "premise": "魔法学院。"}, ensure_ascii=False)),
            ("prior:cast", "character_set", json.dumps({"characters": [{"name": "林晚"}]}, ensure_ascii=False)),
            ("prior:plan", "plan_contract", plan_value),
            (
                f"chapter-{hashlib.sha256('prior-run:prior:chapter:2'.encode()).hexdigest()[:24]}",
                "chapter",
                json.dumps({"chapter_plan_id": "prior:chapter:2", "body": "旧的第二章正文。"}, ensure_ascii=False),
            ),
        ):
            connection.execute(
                "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
                "schema_version, is_current, created_at) VALUES (?, 1, ?, 'prior-run', ?, 'accepted', ?, ?, 1, 1, ?)",
                (
                    artifact_id,
                    PROJECT_ID.root,
                    kind,
                    value.encode(),
                    hashlib.sha256(value.encode()).hexdigest(),
                    earlier.isoformat(),
                ),
            )
        for memory_id, kind, subject, predicate, value in (
            ("prior:m1:world", "world_rule", "prior:world", "world_bible", json.dumps({"title": "晨光学院", "premise": "魔法学院。"}, ensure_ascii=False)),
            ("prior:m1:cast", "character_identity", "prior:cast", "character_identity", json.dumps({"name": "林晚", "role": "主角", "goal": "查明真相"}, ensure_ascii=False)),
        ):
            connection.execute(
                "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, "
                "subject_id, predicate, value_json, source_artifact_id, source_artifact_revision, source_span, "
                "source_hash, confidence, created_by, created_at) "
                "VALUES (?, ?, 'prior-run', 1, 'm1_canonical', ?, 'validated', ?, ?, ?, ?, 1, '0:1', ?, 1.0, "
                "'test-seed:v1', ?)",
                (
                    memory_id,
                    PROJECT_ID.root,
                    kind,
                    subject,
                    predicate,
                    value,
                    subject,
                    hashlib.sha256(value.encode()).hexdigest(),
                    earlier.isoformat(),
                ),
            )
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = ReviseTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    # The seeded prior run is already SUCCEEDED; user requests a rewrite of chapter 2.
    from novel_agent.service_models import ReviseChapterRequest

    snapshot = service.revise_chapter(
        ReviseChapterRequest(
            project_id=PROJECT_ID,
            run_id="prior-run",
            chapter_number=2,
            idempotency_key="revise-ch2",
        )
    )
    assert snapshot.status is RunStatus.QUEUED
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "prior-run")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    # The rewritten chapter awaits user confirmation as a rebase proposal.
    assert snapshot.status is RunStatus.WAITING_REVIEW
    assert snapshot.review_reason == "chapter_rebase_confirmation_required"
    assert transport.hits.count("chapter") == 1

    with connect_project_database(project_root) as connection:
        rows = connection.execute(
            "SELECT revision, is_current, raw_bytes FROM artifacts WHERE run_id = 'prior-run' "
            "AND kind = 'chapter' ORDER BY revision"
        ).fetchall()
        assert len(rows) == 2
        # The old revision is still current; the new one awaits confirmation.
        old_rev, old_current, old_bytes = rows[0]
        new_rev, new_current, new_bytes = rows[1]
        assert old_current == 1
        assert new_current == 0
        assert "旧的第二章正文" in bytes(old_bytes).decode()
        assert "重写后的第二章正文" in bytes(new_bytes).decode()

    # Confirm the rebase: the new revision becomes current.
    import novel_agent.revision_models as revision_models
    from novel_agent.revisions import confirm_chapter_rebase

    with connect_project_database(project_root) as connection:
        row = connection.execute(
            "SELECT proposal_json FROM chapter_rebase_proposals WHERE run_id = 'prior-run' "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        proposal = revision_models.RebaseProposal.model_validate_json(str(row[0]))
        confirmation = revision_models.RebaseConfirmation(
            project_id=PROJECT_ID,
            run_id="prior-run",
            proposal_id=proposal.proposal_id,
            user_confirmed=True,
            replacements=(
                revision_models.RebaseChapter(source=proposal.source, replacement=proposal.replacement),
            ),
        )
        confirm_chapter_rebase(connection, confirmation, FIXED_TIME)
        rows = connection.execute(
            "SELECT revision, is_current FROM artifacts WHERE run_id = 'prior-run' "
            "AND kind = 'chapter' ORDER BY revision"
        ).fetchall()
        assert rows[0][1] == 0  # old revision no longer current
        assert rows[1][1] == 1  # new revision is current


def test_auto_approve_false_stages_and_waits_for_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """auto_approve=False: produced artifacts are staged and the run pauses for
    user confirmation; approval persists them, regenerate discards them."""
    import hashlib

    from novel_agent.contracts import RunStatus
    from novel_agent.service_models import ApproveArtifactRequest, RegenerateArtifactRequest

    class ApprovalTransport:
        def __init__(self) -> None:
            self.hits: list[str] = []

        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                self.hits.append("brief")
                payload = {"domain": "novel", "intent": "world_only", "goal": "设定世界观", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 1}
            elif "规划器" in prompt:
                self.hits.append("plan")
                payload = {
                    "steps": [
                        {
                            "step_id": 1,
                            "description": "设定世界观",
                            "tool": "world.generate@1.0.0",
                            "goal": "生成世界观",
                            "parameters": None,
                        }
                    ]
                }
            elif "资深世界观架构师" in prompt:
                self.hits.append("world")
                payload = {"title": "晨光学院", "premise": "一座漂浮的魔法学院，魔力来自星辰。", "rules": ["魔力来自星辰"], "constraints": []}
            elif "资深角色设计师" in prompt:
                self.hits.append("cast")
                payload = {"characters": [{"name": "x", "role": "x", "goal": "y", "traits": []}]}
            elif "剧情架构师" in prompt:
                self.hits.append("outline")
                payload = {"chapters": []}
            elif "华语小说家" in prompt:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "主角抵达学院。", "events": []}
            else:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = ApprovalTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="approval-run",
            idempotency_key="start-approval-run",
            user_requirements="设定一个魔法学院世界观",
            provider="legacy_real",
            auto_approve=False,
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "approval-run")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    # auto_approve=False gates the plan first; approve it to proceed.
    assert snapshot.review_reason == "plan_approval_required"
    service.approve_plan(PROJECT_ID, "approval-run", "approve-plan")
    deadline = monotonic() + 20.0
    while (snapshot := service.snapshot(PROJECT_ID, "approval-run")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate after plan approval: {snapshot.status}"
        sleep(0.01)
    # The world was produced and staged, but the run waits for approval.
    assert snapshot.status is RunStatus.WAITING_REVIEW
    assert snapshot.review_reason == "artifact_approval_required"
    assert transport.hits.count("world") == 1
    with connect_project_database(project_root) as connection:
        staged = connection.execute(
            "SELECT artifact_id, revision, raw_bytes, kind FROM artifacts "
            "WHERE run_id = 'approval-run' AND status = 'staged'"
        ).fetchall()
        assert len(staged) == 1
        staged_artifact_id, staged_revision, staged_raw, staged_kind = staged[0]
        assert staged_kind == "world_bible"
        assert "晨光学院" in bytes(staged_raw).decode()
        accepted = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = 'approval-run' AND status = 'accepted'"
        ).fetchone()
        assert accepted == (0,)

    # Approve: the staged world is persisted to the canonical stores.
    service.approve_artifact(
        ApproveArtifactRequest(
            project_id=PROJECT_ID,
            run_id="approval-run",
            artifact_id=str(staged_artifact_id),
            revision=int(staged_revision),
            idempotency_key="approve-world",
        )
    )
    deadline = monotonic() + 20.0
    while (snapshot := service.snapshot(PROJECT_ID, "approval-run")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate after approval: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED
    with connect_project_database(project_root) as connection:
        world = connection.execute(
            "SELECT raw_bytes, status FROM artifacts WHERE run_id = 'approval-run' "
            "AND kind = 'world_bible' AND status = 'accepted'"
        ).fetchone()
        assert world is not None
        assert "晨光学院" in bytes(world[0]).decode()
        memory = connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE run_id = 'approval-run' "
            "AND layer = 'm1_canonical' AND kind = 'world_rule'"
        ).fetchone()
        assert memory[0] >= 1
        leftover = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = 'approval-run' AND status = 'staged'"
        ).fetchone()
        assert leftover == (0,)


def test_regenerate_discards_staged_and_reruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """regenerate deletes the staged artifact and requeues the run to redraft."""
    import hashlib

    from novel_agent.contracts import RunStatus
    from novel_agent.service_models import RegenerateArtifactRequest

    class RegenerateTransport:
        def __init__(self) -> None:
            self.world_calls = 0

        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                payload = {"domain": "novel", "intent": "world_only", "goal": "设定世界观", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 1}
            elif "规划器" in prompt:
                payload = {
                    "steps": [
                        {
                            "step_id": 1,
                            "description": "设定世界观",
                            "tool": "world.generate@1.0.0",
                            "goal": "生成世界观",
                            "parameters": None,
                        }
                    ]
                }
            elif "资深世界观架构师" in prompt:
                self.world_calls += 1
                payload = {
                    "title": "晨光学院",
                    "premise": "一座漂浮的魔法学院，魔力来自星辰。" if self.world_calls == 1 else "重新生成的第二版世界观。",
                    "rules": [],
                    "constraints": [],
                }
            elif "资深角色设计师" in prompt:
                payload = {"characters": [{"name": "x", "role": "x", "goal": "y", "traits": []}]}
            elif "剧情架构师" in prompt:
                payload = {"chapters": []}
            elif "华语小说家" in prompt:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "主角抵达学院。", "events": []}
            else:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = RegenerateTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="regenerate-run",
            idempotency_key="start-regenerate-run",
            user_requirements="设定一个魔法学院世界观",
            provider="legacy_real",
            auto_approve=False,
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "regenerate-run")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    # auto_approve=False gates the plan first; approve to proceed.
    assert snapshot.review_reason == "plan_approval_required"
    service.approve_plan(PROJECT_ID, "regenerate-run", "approve-plan")
    deadline = monotonic() + 20.0
    while (snapshot := service.snapshot(PROJECT_ID, "regenerate-run")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate after plan approval: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.WAITING_REVIEW
    assert transport.world_calls == 1
    with connect_project_database(project_root) as connection:
        staged = connection.execute(
            "SELECT artifact_id, revision, raw_bytes FROM artifacts "
            "WHERE run_id = 'regenerate-run' AND status = 'staged'"
        ).fetchall()
        assert len(staged) == 1
        staged_artifact_id, staged_revision, staged_raw = staged[0]
        assert "魔力来自星辰" in bytes(staged_raw).decode()

    # Regenerate: drop the staged draft and redraft the world.
    service.regenerate_artifact(
        RegenerateArtifactRequest(
            project_id=PROJECT_ID,
            run_id="regenerate-run",
            artifact_id=str(staged_artifact_id),
            revision=int(staged_revision),
            idempotency_key="regenerate-world",
        )
    )
    deadline = monotonic() + 20.0
    while (snapshot := service.snapshot(PROJECT_ID, "regenerate-run")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate after regenerate: {snapshot.status}"
        sleep(0.01)
    # Regenerate requeues: the plan gate fires again, approve it.
    if snapshot.review_reason == "plan_approval_required":
        service.approve_plan(PROJECT_ID, "regenerate-run", "approve-plan-2")
        deadline = monotonic() + 20.0
        while (snapshot := service.snapshot(PROJECT_ID, "regenerate-run")).status not in terminal:
            assert monotonic() < deadline, f"run did not terminate after replan: {snapshot.status}"
            sleep(0.01)
    assert snapshot.status is RunStatus.WAITING_REVIEW  # second draft also awaits approval
    assert transport.world_calls == 2
    with connect_project_database(project_root) as connection:
        staged = connection.execute(
            "SELECT raw_bytes FROM artifacts WHERE run_id = 'regenerate-run' AND status = 'staged'"
        ).fetchall()
        assert len(staged) == 1
        assert "重新生成的第二版世界观" in bytes(staged[0][0]).decode()


def test_cast_only_plan_gets_world_prerequisite_auto_prepended(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cast-only plan without a world step is repaired with a world step
    instead of degrading: the run generates world then cast, not degraded."""
    from novel_agent.contracts import RunStatus

    class CastOnlyTransport:
        def __init__(self) -> None:
            self.hits: list[str] = []

        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                self.hits.append("brief")
                payload = {"domain": "novel", "intent": "characters_only", "goal": "设计角色", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 1}
            elif "规划器" in prompt:
                self.hits.append("plan")
                # The model forgets the world step (like the real provider did).
                payload = {
                    "steps": [
                        {
                            "step_id": 1,
                            "description": "设计三个角色",
                            "tool": "character.generate@1.0.0",
                            "goal": "设计三个中学生角色",
                            "parameters": None,
                        }
                    ]
                }
            elif "资深世界观架构师" in prompt:
                self.hits.append("world")
                payload = {"title": "现代都市", "premise": "现代现实世界。", "rules": [], "constraints": []}
            elif "资深角色设计师" in prompt:
                self.hits.append("cast")
                payload = {"characters": [{"name": "林晚", "role": "女主", "goal": "目标", "traits": []}]}
            elif "剧情架构师" in prompt:
                self.hits.append("outline")
                payload = {"chapters": []}
            elif "华语小说家" in prompt:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "s", "events": []}
            else:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = CastOnlyTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="cast-only-repaired",
            idempotency_key="start-cast-only-repaired",
            user_requirements="设计三个17岁左右的中学生角色，一个男主角两个女主角，三角关系",
            provider="legacy_real",
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "cast-only-repaired")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED

    # The world prerequisite was prepended automatically: both world and cast
    # generators ran (no degradation to a different template).
    assert "world" in transport.hits
    assert "cast" in transport.hits
    assert "outline" not in transport.hits
    with connect_project_database(project_root) as connection:
        plan_event = connection.execute(
            "SELECT payload_json FROM run_events WHERE run_id = ? AND event_type = 'plan_generated' "
            "ORDER BY sequence DESC LIMIT 1",
            ("cast-only-repaired",),
        ).fetchone()
        assert plan_event is not None
        payload = json.loads(str(plan_event[0]))
        assert payload.get("degraded") is False
        steps = payload.get("steps", "")
        assert steps.startswith("1:world.generate@1.0.0")
        kinds = [
            str(row[0])
            for row in connection.execute(
                "SELECT kind FROM artifacts WHERE run_id = ? AND status = 'accepted' ORDER BY kind",
                ("cast-only-repaired",),
            )
        ]
        # World is a background prerequisite for cast-only runs (same as the
        # intent template): only the cast is persisted.
        assert kinds == ["character_set"]


def test_plan_failure_reason_surfaces_in_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the planner returns unparseable output, the degraded event carries
    the concrete reason so the console can explain why planning failed."""
    from novel_agent.contracts import RunStatus

    class GarbagePlanTransport:
        def generate(self, request) -> ProviderTransportResult:
            if "规划器" in request.prompt:
                return ProviderTransportResult(
                    content="抱歉，我无法完成这个任务。",
                    model_name="t",
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            return ProviderTransportResult(
                content='{"domain": "novel", "intent": "characters_only", "goal": "设计角色", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 1}',
                model_name="t",
                prompt_tokens=1,
                completion_tokens=1,
            )

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=GarbagePlanTransport())
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="plan-failure-reason",
            idempotency_key="start-plan-failure-reason",
            user_requirements="设计三个角色",
            provider="legacy_real",
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "plan-failure-reason")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    with connect_project_database(project_root) as connection:
        event = connection.execute(
            "SELECT payload_json FROM run_events WHERE run_id = ? AND event_type = 'plan_generated' "
            "ORDER BY sequence DESC LIMIT 1",
            ("plan-failure-reason",),
        ).fetchone()
        assert event is not None
        payload = json.loads(str(event[0]))
        assert payload.get("degraded") is True
        reason = payload.get("degraded_reason", "")
        assert reason.startswith("parse:")  # the raw model output was not JSON
        assert "抱歉" in reason


def test_provider_failures_recorded_for_forensics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every failing LLM call writes a structured provider_failures row."""
    from novel_agent.contracts import RunStatus
    from novel_agent.provider_types import ProviderFailureKind, ProviderTransportError

    class FlakyTransport:
        def __init__(self) -> None:
            self.plan_calls = 0

        def generate(self, request) -> ProviderTransportResult:
            if "需求分析器" in request.prompt:
                return ProviderTransportResult(
                    content='{"domain": "novel", "intent": "characters_only", "goal": "设计角色", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 1}',
                    model_name="t",
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            if "规划器" in request.prompt:
                self.plan_calls += 1
                raise ProviderTransportError(kind=ProviderFailureKind.PERMANENT, detail="model refuses")
            return ProviderTransportResult(content="x", model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=FlakyTransport())
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="provider-failures",
            idempotency_key="start-provider-failures",
            user_requirements="设计三个角色",
            provider="legacy_real",
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "provider-failures")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    with connect_project_database(project_root) as connection:
        failures = connection.execute(
            "SELECT caller, failure_kind, reason, model_name, prompt_sha256 FROM provider_failures "
            "WHERE run_id = 'provider-failures' ORDER BY created_at"
        ).fetchall()
        assert len(failures) >= 1
        caller, kind, reason, model_name, prompt_sha256 = failures[0]
        assert caller == "plan"
        assert kind == "permanent"
        assert "model refuses" in reason
        assert len(prompt_sha256) == 64


def test_outline_parse_failure_recorded_for_forensics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the outline generator returns unparseable content, the parse
    failure is recorded so the console can explain why nothing was staged."""
    from novel_agent.contracts import RunStatus

    class BadOutlineTransport:
        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                payload = {"domain": "novel", "intent": "outline_only", "goal": "写大纲", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 10}
            elif "规划器" in prompt:
                payload = {"steps": [{"step_id": 1, "description": "生成大纲", "tool": "outline.generate@1.0.0", "goal": "大纲", "parameters": None}]}
            elif "资深世界观架构师" in prompt:
                payload = {"title": "晨光学院", "premise": "魔法学院。", "rules": [], "constraints": []}
            elif "资深角色设计师" in prompt:
                payload = {"characters": [{"name": "林晚", "role": "女主", "goal": "g", "traits": []}]}
            elif "剧情架构师" in prompt:
                return ProviderTransportResult(
                    content="抱歉，我无法生成大纲。这是一段完全不符合 JSON schema 的文本回复。",
                    model_name="t",
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            elif "华语小说家" in prompt:
                return ProviderTransportResult(content="x" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "s", "events": []}
            else:
                return ProviderTransportResult(content="x" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=BadOutlineTransport())
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="bad-outline",
            idempotency_key="start-bad-outline",
            user_requirements="写一份10章故事大纲",
            provider="legacy_real",
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "bad-outline")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.WAITING_REVIEW  # plan blocked, nothing staged
    with connect_project_database(project_root) as connection:
        failures = connection.execute(
            "SELECT caller, failure_kind, reason FROM provider_failures WHERE run_id = 'bad-outline' ORDER BY created_at"
        ).fetchall()
        assert len(failures) >= 1
        caller, kind, reason = failures[-1]
        assert caller == "governed_outline"
        assert kind == "parse"
        assert "无法生成大纲" in reason  # raw model output excerpt is recorded
        staged = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = 'bad-outline' AND status = 'staged'"
        ).fetchone()
        assert staged == (0,)


def test_outline_schema_failure_with_chapters_blocks_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the outline fails validation but the graph also has chapter steps
    (approval mode), the run blocks with a clean reason instead of raising
    AttributeError on the SchemaFailure payload."""
    from novel_agent.contracts import RunStatus

    class ShortOutlineTransport:
        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                payload = {"domain": "novel", "intent": "outline_only", "goal": "写大纲", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 10}
            elif "规划器" in prompt:
                payload = {
                    "steps": [
                        {"step_id": 1, "description": "生成大纲", "tool": "outline.generate@1.0.0", "goal": "大纲", "parameters": None},
                        {"step_id": 2, "description": "写第一章", "tool": "chapter.draft@1.0.0", "goal": "正文", "parameters": None},
                    ]
                }
            elif "剧情架构师" in prompt:
                # Only 3 chapters, but 10 are required.
                payload = {"chapters": [{"title": f"第{n}章", "goal": "目标", "planned_events": ["事件"]} for n in range(1, 4)]}
            elif "华语小说家" in prompt:
                return ProviderTransportResult(content="x" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "s", "events": []}
            else:
                return ProviderTransportResult(content="x" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    (project_root / "worlds").mkdir(parents=True)
    (project_root / "characters").mkdir(parents=True)
    (project_root / "worlds" / "晨光学院.txt").write_text("魔法学院。", encoding="utf-8")
    (project_root / "characters" / "林晚.json").write_text(
        json.dumps({"name": "林晚", "role_introduction": "主角", "role_experience": "查明真相"}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=ShortOutlineTransport())
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="outline-short-chapters",
            idempotency_key="start-outline-short-chapters",
            user_requirements="写一份10章故事大纲",
            provider="legacy_real",
            auto_approve=False,
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "outline-short-chapters")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    # auto_approve=False gates the plan first; approve to proceed to outline.
    assert snapshot.review_reason == "plan_approval_required"
    service.approve_plan(PROJECT_ID, "outline-short-chapters", "approve-plan")
    deadline = monotonic() + 20.0
    while (snapshot := service.snapshot(PROJECT_ID, "outline-short-chapters")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate after plan approval: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.WAITING_REVIEW
    assert snapshot.review_reason == "outline_schema_failure"
    with connect_project_database(project_root) as connection:
        failure = connection.execute(
            "SELECT failure_kind, reason FROM provider_failures WHERE run_id = 'outline-short-chapters' "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        assert failure is not None
        assert failure[0] == "parse"
        assert "outline_chapter_count_mismatch" in failure[1]
        # No staged artifact, no partial commit.
        staged = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = 'outline-short-chapters' AND status = 'staged'"
        ).fetchone()
        assert staged == (0,)


def test_plan_approval_gate_and_intent_trimming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """outline_only intent with existing world/cast: the plan is trimmed to
    outline steps, the run pauses for plan confirmation, and approving it
    executes only the outline (no world/cast/chapter regeneration)."""
    from novel_agent.contracts import RunStatus

    class PlanGateTransport:
        def __init__(self) -> None:
            self.hits: list[str] = []

        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                self.hits.append("brief")
                payload = {"domain": "novel", "intent": "outline_only", "goal": "写大纲", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 10}
            elif "规划器" in prompt:
                self.hits.append("plan")
                # The model over-produces: world/cast/chapter steps included.
                payload = {
                    "steps": [
                        {"step_id": 1, "description": "生成世界观", "tool": "world.generate@1.0.0", "goal": "世界", "parameters": None},
                        {"step_id": 2, "description": "生成角色", "tool": "character.generate@1.0.0", "goal": "角色", "parameters": None},
                        {"step_id": 3, "description": "生成大纲", "tool": "outline.generate@1.0.0", "goal": "大纲", "parameters": None},
                        {"step_id": 4, "description": "写第一章", "tool": "chapter.draft@1.0.0", "goal": "正文", "parameters": None},
                    ]
                }
            elif "资深世界观架构师" in prompt:
                self.hits.append("world")  # must never run
                payload = {"title": "新世界", "premise": "x", "rules": [], "constraints": []}
            elif "资深角色设计师" in prompt:
                self.hits.append("cast")  # must never run
                payload = {"characters": [{"name": "x", "role": "x", "goal": "y", "traits": []}]}
            elif "剧情架构师" in prompt:
                self.hits.append("outline")
                payload = {"chapters": [{"title": f"第{n}章", "goal": "目标", "planned_events": ["事件"]} for n in range(1, 11)]}
            elif "华语小说家" in prompt:
                self.hits.append("chapter")  # must never run
                return ProviderTransportResult(content="x" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "s", "events": []}
            else:
                return ProviderTransportResult(content="x" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    (project_root / "worlds").mkdir(parents=True)
    (project_root / "characters").mkdir(parents=True)
    (project_root / "worlds" / "晨光学院.txt").write_text("魔法学院。", encoding="utf-8")
    (project_root / "characters" / "林晚.json").write_text(
        json.dumps({"name": "林晚", "role_introduction": "主角", "role_experience": "查明真相"}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = PlanGateTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="plan-gate",
            idempotency_key="start-plan-gate",
            user_requirements="基于已有角色和世界观设定，创作一部共10个章节的小说故事大纲",
            provider="legacy_real",
            auto_approve=False,
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "plan-gate")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    # The plan waits for confirmation before anything executes.
    assert snapshot.status is RunStatus.WAITING_REVIEW
    assert snapshot.review_reason == "plan_approval_required"
    assert transport.hits == ["brief", "plan"]  # nothing executed yet
    with connect_project_database(project_root) as connection:
        event = connection.execute(
            "SELECT payload_json FROM run_events WHERE run_id = ? AND event_type = 'plan_approval_required' "
            "ORDER BY sequence DESC LIMIT 1",
            ("plan-gate",),
        ).fetchone()
        assert event is not None
        payload = json.loads(str(event[0]))
        steps_text = payload.get("steps", "")
        assert "outline.generate@1.0.0" in steps_text
        assert "chapter.draft" not in steps_text  # trimmed
        assert "world.generate" not in steps_text  # trimmed
        # The steps field must be front-end parseable: semicolon-separated
        # "step_id:tool:description" entries (like plan_generated).
        parts = steps_text.split(";")
        assert len(parts) == 1
        assert parts[0].split(":")[0] == "1"
        assert parts[0].split(":")[1] == "outline.generate@1.0.0"
        assert len(parts[0].split(":")) >= 3  # includes description

    # Approve the plan: only the outline runs (world/cast/chapter never).
    service.approve_plan(PROJECT_ID, "plan-gate", "approve-plan-gate")
    deadline = monotonic() + 20.0
    while (snapshot := service.snapshot(PROJECT_ID, "plan-gate")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate after approval: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.WAITING_REVIEW  # outline staged for artifact confirmation
    assert snapshot.review_reason == "artifact_approval_required"
    assert "world" not in transport.hits
    assert "cast" not in transport.hits
    assert "chapter" not in transport.hits
    assert "outline" in transport.hits
    with connect_project_database(project_root) as connection:
        staged = connection.execute(
            "SELECT kind FROM artifacts WHERE run_id = 'plan-gate' AND status = 'staged'"
        ).fetchall()
        assert [row[0] for row in staged] == ["plan_contract"]

    # Approve the outline: the world/cast come from the legacy files (no
    # accepted artifacts of this run), so this must not raise the old
    # "world not approved yet" RunTransitionError.
    from novel_agent.service_models import ApproveArtifactRequest

    with connect_project_database(project_root) as connection:
        staged_artifact = connection.execute(
            "SELECT artifact_id, revision FROM artifacts WHERE run_id = 'plan-gate' AND status = 'staged' LIMIT 1"
        ).fetchone()
    assert staged_artifact is not None
    service.approve_artifact(
        ApproveArtifactRequest(
            project_id=PROJECT_ID,
            run_id="plan-gate",
            artifact_id=str(staged_artifact[0]),
            revision=int(staged_artifact[1]),
            idempotency_key="approve-outline",
        )
    )
    deadline = monotonic() + 20.0
    while (snapshot := service.snapshot(PROJECT_ID, "plan-gate")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate after outline approval: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED
    with connect_project_database(project_root) as connection:
        plan = connection.execute(
            "SELECT kind, status FROM artifacts WHERE run_id = 'plan-gate' AND kind = 'plan_contract' "
            "AND status = 'accepted'"
        ).fetchone()
        assert plan is not None


def test_bare_tool_names_and_revise_downgrade_repaired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real provider returns bare tool names and a stray revise step;
    the plan is normalized (versioned, deduplicated, revise->generate)
    instead of degrading."""
    from novel_agent.contracts import RunStatus

    class BareToolTransport:
        def __init__(self) -> None:
            self.hits: list[str] = []

        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                self.hits.append("brief")
                payload = {"domain": "novel", "intent": "characters_only", "goal": "设计角色", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 1}
            elif "规划器" in prompt:
                self.hits.append("plan")
                # Exactly what the live model returned: bare names, one cast
                # step per role, plus a stray revise step.
                payload = {
                    "steps": [
                        {"step_id": 1, "description": "设定世界观", "tool": "world.generate", "goal": "世界观", "parameters": None},
                        {"step_id": 2, "description": "设计男主角", "tool": "character.generate", "goal": "男主", "parameters": None},
                        {"step_id": 3, "description": "设计女主角", "tool": "character.generate", "goal": "女主1", "parameters": None},
                        {"step_id": 4, "description": "设计女主角", "tool": "character.generate", "goal": "女主2", "parameters": None},
                        {"step_id": 5, "description": "调整关系", "tool": "character.revise", "goal": "三角关系", "parameters": None},
                    ]
                }
            elif "资深世界观架构师" in prompt:
                self.hits.append("world")
                payload = {"title": "现代都市", "premise": "现代现实世界。", "rules": [], "constraints": []}
            elif "资深角色设计师" in prompt:
                self.hits.append("cast")
                payload = {"characters": [{"name": "林晚", "role": "女主", "goal": "目标", "traits": []}]}
            elif "剧情架构师" in prompt:
                self.hits.append("outline")
                payload = {"chapters": []}
            elif "华语小说家" in prompt:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "s", "events": []}
            else:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = BareToolTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="bare-tools",
            idempotency_key="start-bare-tools",
            user_requirements="设计三个17岁左右的中学生角色，包含一名男主角和两名女主角，并构成三角关系",
            provider="legacy_real",
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "bare-tools")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED
    # No degradation: the repaired plan executed world + cast normally.
    assert "world" in transport.hits
    assert "cast" in transport.hits
    with connect_project_database(project_root) as connection:
        plan_event = connection.execute(
            "SELECT payload_json FROM run_events WHERE run_id = ? AND event_type = 'plan_generated' "
            "ORDER BY sequence DESC LIMIT 1",
            ("bare-tools",),
        ).fetchone()
        assert plan_event is not None
        payload = json.loads(str(plan_event[0]))
        assert payload.get("degraded") is False
        assert payload.get("steps", "").startswith("1:world.generate@1.0.0")
        assert "character.generate@1.0.0" in payload.get("steps", "")


def test_approval_skip_prevents_redrafting_approved_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After approving the world, the requeued run skips the world node and
    proceeds to the cast; no duplicate world artifact is generated."""
    from novel_agent.contracts import RunStatus
    from novel_agent.service_models import ApproveArtifactRequest

    class TwoNodeTransport:
        def __init__(self) -> None:
            self.world_calls = 0
            self.cast_calls = 0

        def generate(self, request) -> ProviderTransportResult:
            prompt = request.prompt
            if "需求分析器" in prompt:
                payload = {"domain": "novel", "intent": "characters_only", "goal": "设计角色", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 1}
            elif "规划器" in prompt:
                payload = {
                    "steps": [
                        {"step_id": 1, "description": "设定世界观", "tool": "world.generate@1.0.0", "goal": "世界观", "parameters": None},
                        {"step_id": 2, "description": "设计角色", "tool": "character.generate@1.0.0", "goal": "角色", "parameters": None},
                    ]
                }
            elif "资深世界观架构师" in prompt:
                self.world_calls += 1
                payload = {"title": "现代都市", "premise": "现代现实世界。", "rules": [], "constraints": []}
            elif "资深角色设计师" in prompt:
                self.cast_calls += 1
                payload = {"characters": [{"name": "林晚", "role": "女主", "goal": "目标", "traits": []}]}
            elif "剧情架构师" in prompt:
                payload = {"chapters": []}
            elif "华语小说家" in prompt:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "s", "events": []}
            else:
                return ProviderTransportResult(content="我站在学院门口，晨风拂过衣角。" * 200, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    transport = TwoNodeTransport()
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="approve-skip",
            idempotency_key="start-approve-skip",
            user_requirements="设计三个角色",
            provider="legacy_real",
            auto_approve=False,
        )
    )
    deadline = monotonic() + 20.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    # auto_approve=False gates the plan first: wait for it, confirm it, then
    # wait for the world artifact to be staged for approval.
    while (snapshot := service.snapshot(PROJECT_ID, "approve-skip")).status not in terminal:
        assert monotonic() < deadline, f"run did not reach plan gate: {snapshot.status}"
        sleep(0.01)
    assert snapshot.review_reason == "plan_approval_required"
    service.approve_plan(PROJECT_ID, "approve-skip", "approve-plan")
    deadline = monotonic() + 20.0
    while (snapshot := service.snapshot(PROJECT_ID, "approve-skip")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    # First pass: world is staged (the first node output), cast not reached.
    assert snapshot.status is RunStatus.WAITING_REVIEW
    assert snapshot.review_reason == "artifact_approval_required"
    assert transport.world_calls == 1
    assert transport.cast_calls == 0
    with connect_project_database(project_root) as connection:
        staged = connection.execute(
            "SELECT artifact_id, revision FROM artifacts WHERE run_id = 'approve-skip' AND status = 'staged'"
        ).fetchall()
        assert len(staged) == 1

    # Approve the world: the requeued run must skip the world node (no
    # redraft) and stage the cast next.
    service.approve_artifact(
        ApproveArtifactRequest(
            project_id=PROJECT_ID,
            run_id="approve-skip",
            artifact_id=str(staged[0][0]),
            revision=int(staged[0][1]),
            idempotency_key="approve-world-1",
        )
    )
    deadline = monotonic() + 20.0
    while (snapshot := service.snapshot(PROJECT_ID, "approve-skip")).status not in terminal:
        if snapshot.review_reason == "plan_approval_required":
            service.approve_plan(PROJECT_ID, "approve-skip", "approve-plan-2")
        assert monotonic() < deadline, f"run did not terminate after world approval: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.WAITING_REVIEW
    assert transport.world_calls == 1  # never redrafted
    assert transport.cast_calls == 1  # cast was produced and staged
    with connect_project_database(project_root) as connection:
        worlds = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = 'approve-skip' AND kind = 'world_bible' AND status = 'accepted'"
        ).fetchone()
        assert worlds == (1,)
        staged = connection.execute(
            "SELECT artifact_id, revision FROM artifacts WHERE run_id = 'approve-skip' AND status = 'staged'"
        ).fetchall()
        assert len(staged) == 1  # the cast now awaits approval

    # Approve the cast: the run completes with no further redrafting.
    service.approve_artifact(
        ApproveArtifactRequest(
            project_id=PROJECT_ID,
            run_id="approve-skip",
            artifact_id=str(staged[0][0]),
            revision=int(staged[0][1]),
            idempotency_key="approve-cast-1",
        )
    )
    deadline = monotonic() + 20.0
    while (snapshot := service.snapshot(PROJECT_ID, "approve-skip")).status not in terminal:
        if snapshot.review_reason == "plan_approval_required":
            service.approve_plan(PROJECT_ID, "approve-skip", "approve-plan-3")
        assert monotonic() < deadline, f"run did not terminate after cast approval: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED
    assert transport.world_calls == 1
    assert transport.cast_calls == 1
    with connect_project_database(project_root) as connection:
        casts = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = 'approve-skip' AND kind = 'character_set' AND status = 'accepted'"
        ).fetchone()
        assert casts == (1,)
        leftover = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = 'approve-skip' AND status = 'staged'"
        ).fetchone()
        assert leftover == (0,)


def test_trim_to_intent_for_sub_intents() -> None:
    from novel_agent.agent.plan_generator import PlanStep, _trim_to_intent

    def step(index: int, tool: str) -> PlanStep:
        return PlanStep(step_id=index, description="步骤", tool=tool, goal="目标", parameters=None)

    steps = (
        step(1, "world.generate@1.0.0"),
        step(2, "character.generate@1.0.0"),
        step(3, "outline.generate@1.0.0"),
        step(4, "chapter.draft@1.0.0"),
    )
    assert [item.tool for item in _trim_to_intent(steps, intent="world_only", existing_world=False, existing_cast=False)] == [
        "world.generate@1.0.0"
    ]
    assert [item.tool for item in _trim_to_intent(steps, intent="characters_only", existing_world=True, existing_cast=True)] == [
        "character.generate@1.0.0"
    ]
    assert [item.tool for item in _trim_to_intent(steps, intent="add_character", existing_world=True, existing_cast=True)] == [
        "character.generate@1.0.0"
    ]


def test_trim_to_intent_handles_composite_intents() -> None:
    from novel_agent.agent.plan_generator import PlanStep, _trim_to_intent

    def step(index: int, tool: str) -> PlanStep:
        return PlanStep(step_id=index, description="步骤", tool=tool, goal="目标", parameters=None)

    steps = (
        step(1, "world.generate@1.0.0"),
        step(2, "character.generate@1.0.0"),
        step(3, "outline.generate@1.0.0"),
        step(4, "chapter.draft@1.0.0"),
    )
    trimmed = _trim_to_intent(
        steps,
        intent="characters_only",
        intents=("characters_only", "outline_only"),
        existing_world=False,
        existing_cast=False,
    )
    assert [item.tool for item in trimmed] == [
        "world.generate@1.0.0",
        "character.generate@1.0.0",
        "outline.generate@1.0.0",
    ]
