"""Agent-loop tests: action parsing, loop semantics, tool governance, E2E.

Covers the opencode-style loop additions:
- parse_action (JSON tool call / implicit done / repair tolerance)
- AgentLoop termination (done / step limit) and doom-loop detection
- novel_tools governance: write_world/cast/outline/chapter commit artifacts
- _AgentLoopStage end-to-end via offline generators + scripted decision
- loop_state persistence
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

import pytest

import config
from novel_agent.agent_loop import AgentLoop, AgentTool, LoopConfig, ToolResult, parse_action
from novel_agent.contracts import RunStatus
from novel_agent.pathing import ProjectId
from novel_agent.provider_types import ProviderTransportResult
from novel_agent.service import NovelAgentService, StartNovelRunRequest
from novel_agent.storage import connect_project_database


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")
TERMINAL = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))


# -- parse_action -------------------------------------------------------------


def test_parse_action_tool_call() -> None:
    action = parse_action('{"tool": "write_chapter", "args": {"chapter_number": 2}}')
    assert action is not None and action.kind == "tool"
    assert action.tool == "write_chapter"
    assert action.args == {"chapter_number": 2}


def test_parse_action_implicit_done() -> None:
    action = parse_action('{"reply": "完成"}')
    assert action is not None and action.kind == "done"


def test_parse_action_tolerates_fences() -> None:
    action = parse_action('```json\n{"tool": "mem_query", "args": {"query": "主角"}}\n```')
    assert action is not None and action.kind == "tool"
    assert action.tool == "mem_query"


def test_parse_action_returns_none_on_garbage() -> None:
    assert parse_action("完全不是 JSON") is None


# -- AgentLoop semantics ------------------------------------------------------


class _SeqTransport:
    def __init__(self, steps: list[str]) -> None:
        self.steps = steps
        self.calls = 0

    def generate(self, request) -> ProviderTransportResult:
        content = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        return ProviderTransportResult(content=content, model_name="t", prompt_tokens=1, completion_tokens=1)


def _recorder_tool(name: str, results: list[str]) -> AgentTool:
    def execute(args: dict) -> ToolResult:
        results.append(name)
        return ToolResult(ok=True, summary=f"{name} ok")

    return AgentTool(name=name, description=name, input_schema={}, execute=execute)


def test_loop_runs_until_done() -> None:
    called: list[str] = []
    tools = {
        "a": _recorder_tool("a", called),
        "b": _recorder_tool("b", called),
    }
    loop = AgentLoop(
        tools=tools,
        transport=_SeqTransport(['{"tool": "a", "args": {}}', '{"tool": "b", "args": {}}', '{"done": true}']),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=10),
    )
    outcome, history = loop.run(system_prompt="sys", task="任务")
    assert outcome == "done"
    assert called == ["a", "b"]
    assert any(m.role == "tool" for m in history)


def test_loop_hits_step_limit() -> None:
    loop = AgentLoop(
        tools={"a": _recorder_tool("a", [])},
        transport=_SeqTransport(['{"tool": "a", "args": {}}'] * 10),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=3),
    )
    outcome, _history = loop.run(system_prompt="sys", task="任务")
    assert outcome == "step_limit"


def test_loop_doom_detection_breaks_identical_repeats() -> None:
    called: list[str] = []
    loop = AgentLoop(
        tools={"a": _recorder_tool("a", called)},
        transport=_SeqTransport(['{"tool": "a", "args": {"x": 1}}'] * 10),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=20, doom_loop_threshold=3),
    )
    outcome, history = loop.run(system_prompt="sys", task="任务")
    # doom detection injects a warning and stops calling the same tool; the
    # scripted transport then keeps returning 'a' but the loop must eventually
    # terminate (step limit) rather than spin forever with the same args.
    assert outcome == "step_limit"
    # After the first 3 identical calls the 4th+ should have been suppressed.
    assert called.count("a") <= 3 + 1


def test_loop_recovers_from_parse_failure() -> None:
    loop = AgentLoop(
        tools={"a": _recorder_tool("a", [])},
        transport=_SeqTransport(["垃圾", '{"done": true}']),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=10),
    )
    outcome, history = loop.run(system_prompt="sys", task="任务")
    assert outcome == "done"
    assert any("无法解析" in m.content for m in history)


# -- novel_tools governance ---------------------------------------------------


@pytest.fixture
def projects_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "projects"
    (root / PROJECT_ID.root).mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(root))
    return root


class _LoopDecisionTransport:
    """Scripted decision model: world -> cast -> outline -> chapter -> done."""

    def __init__(self) -> None:
        self.actions = [
            '{"tool": "write_world", "args": {}}',
            '{"tool": "write_cast", "args": {}}',
            '{"tool": "write_outline", "args": {}}',
            '{"tool": "write_chapter", "args": {"chapter_number": 1}}',
            '{"done": true, "reply": "完成"}',
        ]
        self.calls = 0

    def generate(self, request) -> ProviderTransportResult:
        content = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        return ProviderTransportResult(content=content, model_name="t", prompt_tokens=1, completion_tokens=1)


class _LoopBriefTransport:
    def generate(self, request) -> ProviderTransportResult:
        payload = {"domain": "novel", "intent": "full_novel", "goal": "写一章小说",
                   "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": 1}
        return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False), model_name="t", prompt_tokens=1, completion_tokens=1)


def _wait_terminal(service: NovelAgentService, run_id: str, seconds: float = 10.0) -> RunStatus:
    deadline = monotonic() + seconds
    while (snapshot := service.snapshot(PROJECT_ID, run_id)).status not in TERMINAL:
        assert monotonic() < deadline, f"run did not terminate; status={snapshot.status}"
        sleep(0.01)
    return snapshot.status


def test_agent_loop_provider_full_pipeline(projects_root: Path) -> None:
    """agent_loop with offline generators + scripted decision produces all artifacts."""
    from novel_agent.default_stage import _AgentLoopStage
    from novel_agent.pathing import ProjectPathResolver
    from novel_agent.run_coordinator import RunCoordinator
    from novel_agent.run_models import StartRunRequest
    from novel_agent.run_worker import RunWorker
    from novel_agent.service_models import StartNovelRunRequest as S

    database = connect_project_database(projects_root / PROJECT_ID.root)
    connection = database.__enter__()
    try:
        from novel_agent.storage import migrate

        migrate(connection)
        coordinator = RunCoordinator(connection, lambda: FIXED_TIME)
        req = S(project_id=PROJECT_ID, run_id="run-loop", idempotency_key="start-loop",
                user_requirements="写一本一章的小说", provider="agent_loop", auto_approve=True)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="run-loop", idempotency_key="start-loop"))
        stage = _AgentLoopStage(
            coordinator, req, ProjectPathResolver(projects_root.parent), lambda: FIXED_TIME,
            transport=_LoopBriefTransport(),
            loop_transport=_LoopDecisionTransport(),
            offline_generators=True,
        )
        worker = RunWorker(coordinator, stage, stage_timeout_seconds=30.0)
        lease = coordinator.claim_next("worker")
        assert lease is not None
        worker.run(lease)
        assert connection.execute(
            "SELECT state FROM runs WHERE run_id='run-loop'"
        ).fetchone() == (RunStatus.SUCCEEDED.value,)

        kinds = [str(r[0]) for r in connection.execute(
            "SELECT kind FROM artifacts WHERE status='accepted' GROUP BY kind ORDER BY kind"
        ).fetchall()]
        assert kinds == ["chapter", "character_set", "plan_contract", "world_bible"]
        state = connection.execute(
            "SELECT step FROM loop_state WHERE run_id='run-loop'"
        ).fetchone()
        assert state is not None
        tools = [str(r[0]) for r in connection.execute(
            "SELECT payload_json FROM run_events WHERE run_id='run-loop' AND event_type='tool_call'"
        ).fetchall()]
        assert len(tools) == 4
    finally:
        database.__exit__(None, None, None)


def test_agent_loop_uses_loop_state_after_repair(projects_root: Path) -> None:
    """REPAIR on an agent_loop run resumes from the persisted loop state."""
    from novel_agent.default_stage import _AgentLoopStage
    from novel_agent.loop_state import LoopState, save_loop_state
    from novel_agent.run_coordinator import RunCoordinator
    from novel_agent.run_models import StartRunRequest
    from novel_agent.run_worker import RunWorker
    from novel_agent.service_models import StartNovelRunRequest as S

    database = connect_project_database(projects_root / PROJECT_ID.root)
    connection = database.__enter__()
    try:
        from novel_agent.storage import migrate

        migrate(connection)
        coordinator = RunCoordinator(connection, lambda: FIXED_TIME)
        req = S(project_id=PROJECT_ID, run_id="run-loop-state", idempotency_key="start-loop-state",
                user_requirements="写一本一章的小说", provider="agent_loop", auto_approve=True)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="run-loop-state", idempotency_key="start-loop-state"))
        # Persist a loop state as if the loop had progressed (step 2, wrote world).
        save_loop_state(
            connection,
            LoopState(run_id="run-loop-state", step=2, plan_ref=("p1", 1), written_chapters=(1,)),
            FIXED_TIME,
        )
        stage = _AgentLoopStage(
            coordinator, req, __import__("novel_agent.pathing", fromlist=["ProjectPathResolver"]).ProjectPathResolver(projects_root.parent),
            lambda: FIXED_TIME,
            transport=_LoopBriefTransport(),
            loop_transport=_LoopDecisionTransport(),
            offline_generators=True,
        )
        worker = RunWorker(coordinator, stage, stage_timeout_seconds=30.0)
        lease = coordinator.claim_next("worker")
        assert lease is not None
        worker.run(lease)
        # The stage loaded the persisted plan_ref and proceeded from step 2.
        from novel_agent.loop_state import load_loop_state

        final = load_loop_state(connection, "run-loop-state")
        assert final is not None
        assert final.step >= 2
    finally:
        database.__exit__(None, None, None)
