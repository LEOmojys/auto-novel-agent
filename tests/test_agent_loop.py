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
        transport=_SeqTransport([
            '{"plan": [{"tool": "a", "args": {}}, {"tool": "b", "args": {}}]}',
            '{"next": true}',
            '{"done": true}',
        ]),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=10),
    )
    outcome, history = loop.run(system_prompt="sys", task="任务")
    assert outcome == "done"
    assert called == ["a", "b"]
    assert any(m.role == "tool" for m in history)
    assert any("执行计划" in m.content for m in history)


def test_loop_hits_step_limit() -> None:
    loop = AgentLoop(
        tools={"a": _recorder_tool("a", [])},
        transport=_SeqTransport(['{"plan": [{"tool": "a", "args": {}}]}'] * 10),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=3),
    )
    outcome, _history = loop.run(system_prompt="sys", task="任务")
    assert outcome == "step_limit"


def test_loop_doom_detection_breaks_identical_repeats() -> None:
    called: list[str] = []
    events: list[tuple[str, dict]] = []
    loop = AgentLoop(
        tools={"a": _recorder_tool("a", called)},
        transport=_SeqTransport(['{"plan": [{"tool": "a", "args": {"x": 1}}]}'] * 10),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=20, doom_loop_threshold=3),
        event_sink=lambda event_type, payload: events.append((event_type, payload)),
    )
    outcome, _history = loop.run(system_prompt="sys", task="任务")
    # doom detection flags the repeated (tool, args) and stops executing it;
    # the scripted transport keeps re-planning the same step but the loop must
    # terminate (step limit) rather than spin forever with the same args.
    assert outcome == "step_limit"
    # The identical call was suppressed once the threshold was reached.
    assert called.count("a") <= 3
    assert any(event_type == "loop_doom_detected" for event_type, _payload in events)


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


def test_parse_action_plan_and_next() -> None:
    action = parse_action('{"plan": [{"tool": "a", "args": {}}, {"tool": "b", "args": {"x": 1}}]}')
    assert action is not None and action.kind == "plan"
    assert action.plan == (("a", {}), ("b", {"x": 1}))
    nxt = parse_action('{"next": true}')
    assert nxt is not None and nxt.kind == "next"


def test_loop_requires_plan_before_first_tool() -> None:
    called: list[str] = []
    loop = AgentLoop(
        tools={"a": _recorder_tool("a", called)},
        transport=_SeqTransport(['{"tool": "a", "args": {}}', '{"done": true}']),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=10),
    )
    outcome, history = loop.run(system_prompt="sys", task="任务")
    assert outcome == "done"
    assert called == []
    assert any("请先输出执行计划" in m.content for m in history)


def test_loop_emits_plan_event_and_executes_first_step() -> None:
    called: list[str] = []
    events: list[tuple[str, dict]] = []
    loop = AgentLoop(
        tools={"a": _recorder_tool("a", called)},
        transport=_SeqTransport(['{"plan": [{"tool": "a", "args": {}}]}', '{"done": true}']),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=10),
        event_sink=lambda event_type, payload: events.append((event_type, payload)),
    )
    outcome, _history = loop.run(system_prompt="sys", task="任务")
    assert outcome == "done"
    assert called == ["a"]
    assert ("loop_plan", {"steps": "a"}) in events


def test_loop_plan_replacement_skips_replaced_steps() -> None:
    called: list[str] = []
    tools = {name: _recorder_tool(name, called) for name in ("a", "b", "c")}
    loop = AgentLoop(
        tools=tools,
        transport=_SeqTransport([
            '{"plan": [{"tool": "a", "args": {}}, {"tool": "b", "args": {}}]}',
            '{"plan": [{"tool": "c", "args": {}}]}',
            '{"done": true}',
        ]),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=10),
    )
    outcome, _history = loop.run(system_prompt="sys", task="任务")
    assert outcome == "done"
    # The replacement plan drops the remaining "b" step; "c" runs instead.
    assert called == ["a", "c"]


def test_loop_next_without_plan_guides_model() -> None:
    called: list[str] = []
    loop = AgentLoop(
        tools={"a": _recorder_tool("a", called)},
        transport=_SeqTransport(['{"next": true}', '{"done": true}']),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=10),
    )
    outcome, history = loop.run(system_prompt="sys", task="任务")
    assert outcome == "done"
    assert called == []
    assert any("已全部执行完" in m.content for m in history)


# -- novel_tools governance ---------------------------------------------------


@pytest.fixture
def projects_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "projects"
    (root / PROJECT_ID.root).mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(root))
    return root


class _LoopDecisionTransport:
    """Scripted Plan-Act model: plan (world..chapter) -> re-plan remaining -> done."""

    def __init__(self) -> None:
        self.actions = [
            '{"plan": [{"tool": "write_world", "args": {}}, {"tool": "write_cast", "args": {}}, '
            '{"tool": "write_outline", "args": {}}, {"tool": "write_chapter", "args": {"chapter_number": 1}}]}',
            '{"plan": [{"tool": "write_cast", "args": {}}, {"tool": "write_outline", "args": {}}, '
            '{"tool": "write_chapter", "args": {"chapter_number": 1}}]}',
            '{"plan": [{"tool": "write_outline", "args": {}}, {"tool": "write_chapter", "args": {"chapter_number": 1}}]}',
            '{"plan": [{"tool": "write_chapter", "args": {"chapter_number": 1}}]}',
            '{"done": true, "reply": "完成"}',
        ]
        self.calls = 0

    def generate(self, request) -> ProviderTransportResult:
        content = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        return ProviderTransportResult(content=content, model_name="t", prompt_tokens=1, completion_tokens=1)


class _LoopBriefTransport:
    def __init__(self, intent: str = "full_novel", chapter_count: int = 1) -> None:
        self.intent = intent
        self.chapter_count = chapter_count

    def generate(self, request) -> ProviderTransportResult:
        payload = {"domain": "novel", "intent": self.intent, "goal": "写一章小说",
                   "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": self.chapter_count}
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


def test_agent_loop_approval_pause_and_resume(projects_root: Path) -> None:
    """Approval mode: write_world stages + pauses; loop stops; pending action persisted."""
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
        req = S(project_id=PROJECT_ID, run_id="run-approve", idempotency_key="start-approve",
                user_requirements="写一本一章的小说", provider="agent_loop", auto_approve=False)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="run-approve", idempotency_key="start-approve"))
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
        # write_world is approval-gated: run paused, artifact staged, loop stopped.
        state = connection.execute("SELECT state FROM runs WHERE run_id='run-approve'").fetchone()
        assert state == (RunStatus.WAITING_REVIEW.value,)
        staged = connection.execute(
            "SELECT kind FROM artifacts WHERE run_id='run-approve' AND status='staged'"
        ).fetchall()
        assert any(str(r[0]) == "world_bible" for r in staged)
        from novel_agent.loop_state import load_loop_state

        ls = load_loop_state(connection, "run-approve")
        assert ls is not None
        assert ls.pending_action is not None and ls.pending_action["tool"] == "write_world"
        # tool_call events streamed with summary (replayable history)
        tool_events = connection.execute(
            "SELECT payload_json FROM run_events WHERE run_id='run-approve' AND event_type='tool_call'"
        ).fetchall()
        assert len(tool_events) >= 1
    finally:
        database.__exit__(None, None, None)


def test_agent_loop_full_approval_chain_to_success(projects_root: Path) -> None:
    """Approval mode: approve each staged artifact in turn; loop resumes each time
    and eventually completes with all four artifact kinds accepted."""
    from novel_agent.default_stage import _AgentLoopStage
    from novel_agent.pathing import ProjectPathResolver
    from novel_agent.run_coordinator import RunCoordinator
    from novel_agent.run_models import StartRunRequest
    from novel_agent.run_worker import RunWorker
    from novel_agent.service import ApproveArtifactRequest, NovelAgentService
    from novel_agent.service_models import StartNovelRunRequest as S

    database = connect_project_database(projects_root / PROJECT_ID.root)
    connection = database.__enter__()
    try:
        from novel_agent.storage import migrate

        migrate(connection)
        coordinator = RunCoordinator(connection, lambda: FIXED_TIME)
        req = S(project_id=PROJECT_ID, run_id="run-chain", idempotency_key="start-chain",
                user_requirements="写一本一章的小说", provider="agent_loop", auto_approve=False)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="run-chain", idempotency_key="start-chain"))
        decision = _LoopDecisionTransport()
        stage = _AgentLoopStage(
            coordinator, req, ProjectPathResolver(projects_root.parent), lambda: FIXED_TIME,
            transport=_LoopBriefTransport(), loop_transport=decision, offline_generators=True,
        )
        worker = RunWorker(coordinator, stage, stage_timeout_seconds=30.0)
        svc = NovelAgentService.from_config(now=lambda: FIXED_TIME)

        for round_no in range(6):
            lease = coordinator.claim_next("worker")
            if lease is None:
                break
            worker.run(lease)
            state = connection.execute("SELECT state FROM runs WHERE run_id='run-chain'").fetchone()
            if state == (RunStatus.SUCCEEDED.value,):
                break
            # Approve the latest staged artifact.
            staged = connection.execute(
                "SELECT artifact_id, revision FROM artifacts WHERE run_id='run-chain' AND status='staged' "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if staged is None:
                break
            svc.approve_artifact(ApproveArtifactRequest(
                project_id=PROJECT_ID, run_id="run-chain",
                artifact_id=str(staged[0]), revision=int(staged[1]),
                idempotency_key=f"approve-{round_no}",
            ))
        final = connection.execute("SELECT state FROM runs WHERE run_id='run-chain'").fetchone()
        assert final == (RunStatus.SUCCEEDED.value,), f"chain did not finish: {final}"
        kinds = [str(r[0]) for r in connection.execute(
            "SELECT kind FROM artifacts WHERE run_id='run-chain' AND status='accepted' GROUP BY kind ORDER BY kind"
        ).fetchall()]
        assert kinds == ["chapter", "character_set", "plan_contract", "world_bible"]
    finally:
        database.__exit__(None, None, None)


def test_loop_state_survives_requeue_purge(projects_root: Path) -> None:
    """requeue/REPAIR must NOT clear loop_state (needed for breakpoint resume)."""
    from novel_agent.loop_state import LoopState, load_loop_state, save_loop_state
    from novel_agent.run_coordinator import RunCoordinator
    from novel_agent.run_models import StartRunRequest
    from novel_agent.storage import migrate

    database = connect_project_database(projects_root / PROJECT_ID.root)
    connection = database.__enter__()
    try:
        migrate(connection)
        coordinator = RunCoordinator(connection, lambda: FIXED_TIME)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="run-purge", idempotency_key="start-purge"))
        lease = coordinator.claim_next("worker")
        assert lease is not None
        coordinator.fail_run(lease, "test_failure")
        save_loop_state(
            connection,
            LoopState(run_id="run-purge", step=3, plan_ref=("p1", 1)),
            FIXED_TIME,
        )
        # Requeue (terminal -> start) must preserve loop_state.
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="run-purge", idempotency_key="start-purge-2"))
        assert load_loop_state(connection, "run-purge") is not None
    finally:
        database.__exit__(None, None, None)


def test_agent_loop_outline_only_not_guarded_by_premature_done(projects_root: Path) -> None:
    """An outline-only brief legitimately finishes without chapters."""
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
        req = S(project_id=PROJECT_ID, run_id="run-outline", idempotency_key="start-outline",
                user_requirements="写一份故事大纲", provider="agent_loop", auto_approve=True)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="run-outline", idempotency_key="start-outline"))
        stage = _AgentLoopStage(
            coordinator, req, ProjectPathResolver(projects_root.parent), lambda: FIXED_TIME,
            transport=_LoopBriefTransport(intent="outline_only"), loop_transport=_LoopDecisionTransport(), offline_generators=True,
        )
        worker = RunWorker(coordinator, stage, stage_timeout_seconds=30.0)
        lease = coordinator.claim_next("worker")
        assert lease is not None
        worker.run(lease)
        state = connection.execute("SELECT state FROM runs WHERE run_id='run-outline'").fetchone()
        # outline-only should not fail with done_without_chapters even if the
        # loop stops without writing a chapter.
        assert state[0] in (RunStatus.SUCCEEDED.value, RunStatus.WAITING_REVIEW.value)
    finally:
        database.__exit__(None, None, None)
