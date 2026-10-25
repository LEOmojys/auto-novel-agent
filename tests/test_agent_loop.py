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
    def failing(args: dict) -> ToolResult:
        called.append("a")
        return ToolResult(ok=False, error="boom")

    loop = AgentLoop(
        tools={"a": AgentTool(name="a", description="a", input_schema={}, execute=failing)},
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
    """A replacement plan after the first plan finished is a fresh plan.

    The loop auto-executes the whole plan, so a second plan is not a
    "replacement of remaining steps" but a follow-up plan (opencode-style:
    plans run to completion, then the model decides what to do next).
    """
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
    # Both plans run to completion in order (auto-advance), then done.
    assert called == ["a", "b", "c"]


def test_loop_auto_advances_entire_plan() -> None:
    """The loop executes every plan step automatically without model turns."""
    called: list[str] = []
    tools = {name: _recorder_tool(name, called) for name in ("a", "b", "c")}
    loop = AgentLoop(
        tools=tools,
        transport=_SeqTransport([
            '{"plan": [{"tool": "a", "args": {}}, {"tool": "b", "args": {}}, {"tool": "c", "args": {}}]}',
            '{"done": true}',
        ]),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=10),
    )
    outcome, history = loop.run(system_prompt="sys", task="任务")
    assert outcome == "done"
    # All three steps ran with a single plan; the model only saw the result.
    assert called == ["a", "b", "c"]
    assert sum(1 for m in history if m.role == "tool") == 3


def test_loop_pauses_auto_advance_on_failed_step() -> None:
    """A failed plan step stops the auto-advance for the model to decide."""
    called: list[str] = []

    def execute(_args: dict) -> ToolResult:
        called.append("b")
        return ToolResult(ok=False, error="b 失败了")

    tools = {
        "a": _recorder_tool("a", called),
        "b": AgentTool(name="b", description="b", input_schema={}, execute=execute),
        "c": _recorder_tool("c", called),
    }
    loop = AgentLoop(
        tools=tools,
        transport=_SeqTransport([
            '{"plan": [{"tool": "a", "args": {}}, {"tool": "b", "args": {}}, {"tool": "c", "args": {}}]}',
            '{"plan": [{"tool": "c", "args": {}}]}',
            '{"done": true}',
        ]),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=10),
    )
    outcome, _history = loop.run(system_prompt="sys", task="任务")
    assert outcome == "done"
    # "b" failed -> auto-advance stopped before "c"; the model replanned "c".
    assert called == ["a", "b", "c"]


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
    """Scripted Plan-Act model: plan (world..chapter) -> re-plan remaining -> done.

    With auto-advance, each plan runs to the first approval gate, then the
    resumed run replans the remaining steps.  This script matches the
    approval-chain E2E (auto_approve=False): plan(4) -> plan(3) -> ... -> done.
    """

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


class _LoopFullPlanTransport:
    """Scripted Plan-Act model: one full plan (world..chapter) then done.

    With auto-advance and auto_approve=True, a single plan runs every step to
    completion, so the model only confirms completion afterwards.
    """

    def __init__(self) -> None:
        self.actions = [
            '{"plan": [{"tool": "write_world", "args": {}}, {"tool": "write_cast", "args": {}}, '
            '{"tool": "write_outline", "args": {}}, {"tool": "write_chapter", "args": {"chapter_number": 1}}]}',
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
            loop_transport=_LoopFullPlanTransport(),
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


class _MultiChapterTransport:
    """Scripted Plan-Act: one plan (optionally with world/cast/outline) writing
    N chapters, then done."""

    def __init__(self, *chapter_numbers: int, with_prereqs: bool = False) -> None:
        prereqs = (
            '{"tool": "write_world", "args": {}}, {"tool": "write_cast", "args": {}}, '
            '{"tool": "write_outline", "args": {}}, '
            if with_prereqs else ""
        )
        steps = "".join(
            f'{{"tool": "write_chapter", "args": {{"chapter_number": {n}}}}}' + (", " if i < len(chapter_numbers) - 1 else "")
            for i, n in enumerate(chapter_numbers)
        )
        self.actions = [
            '{"plan": [' + prereqs + steps + ']}',
            '{"done": true, "reply": "完成"}',
        ]
        self.calls = 0

    def generate(self, request) -> ProviderTransportResult:
        content = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        return ProviderTransportResult(content=content, model_name="t", prompt_tokens=1, completion_tokens=1)


def test_agent_loop_continuation_mirrors_existing_plan(projects_root: Path) -> None:
    """A continuation run reusing a previous run's outline mirrors the plan into
    the current run so chapter staging passes (regression: ChapterPersistenceError)."""
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
        resolver = ProjectPathResolver(projects_root.parent)

        # First run: full novel producing world/cast/outline + chapters 1..3.
        first = S(project_id=PROJECT_ID, run_id="cont-1", idempotency_key="cont-1",
                  user_requirements="写一本三章的小说", provider="agent_loop", auto_approve=True)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="cont-1", idempotency_key="cont-1"))
        stage1 = _AgentLoopStage(
            coordinator, first, resolver, lambda: FIXED_TIME,
            transport=_LoopBriefTransport(chapter_count=5),
            loop_transport=_MultiChapterTransport(1, 2, 3, with_prereqs=True),
            offline_generators=True,
        )
        RunWorker(coordinator, stage1, stage_timeout_seconds=30.0).run(coordinator.claim_next("worker-1"))
        assert connection.execute("SELECT state FROM runs WHERE run_id='cont-1'").fetchone() == (RunStatus.SUCCEEDED.value,)

        # Second run: continuation reusing the existing outline (chapters 4-5).
        second = S(project_id=PROJECT_ID, run_id="cont-2", idempotency_key="cont-2",
                   user_requirements="续写第4章和第5章", provider="agent_loop", auto_approve=True)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="cont-2", idempotency_key="cont-2"))
        stage2 = _AgentLoopStage(
            coordinator, second, resolver, lambda: FIXED_TIME,
            transport=_LoopBriefTransport(chapter_count=2),
            loop_transport=_MultiChapterTransport(4, 5),
            offline_generators=True,
        )
        RunWorker(coordinator, stage2, stage_timeout_seconds=30.0).run(coordinator.claim_next("worker-2"))
        state = connection.execute("SELECT state FROM runs WHERE run_id='cont-2'").fetchone()
        assert state == (RunStatus.SUCCEEDED.value,), f"continuation failed: {state}"
        # The reused outline was mirrored into the current run and chapters committed.
        mirrored = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE artifact_id LIKE 'cont-1%plan%' AND run_id='cont-2' AND status='accepted' AND is_current=1"
        ).fetchone()
        chs = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id='cont-2' AND kind='chapter' AND status='accepted'"
        ).fetchone()
        assert chs[0] >= 2
        assert mirrored[0] >= 1
    finally:
        database.__exit__(None, None, None)


def test_memory_query_returns_confirmed_settings(projects_root: Path) -> None:
    """memory_query (agent-loop mem_query tool) returns eligible confirmed memory."""
    from novel_agent.context import memory_query

    database = connect_project_database(projects_root / PROJECT_ID.root)
    connection = database.__enter__()
    try:
        from novel_agent.storage import migrate

        migrate(connection)
        import hashlib as _hashlib
        connection.execute(
            "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("mem-run", PROJECT_ID.root, "succeeded", FIXED_TIME.isoformat(), FIXED_TIME.isoformat()),
        )
        connection.execute(
            "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
            "schema_version, is_current, created_at) VALUES (?, ?, ?, ?, 'chapter', 'accepted', ?, ?, 1, 1, ?)",
            ("mem-src", 1, PROJECT_ID.root, "mem-run", b"abcdefghij", _hashlib.sha256(b"abcdefghij").hexdigest(), FIXED_TIME.isoformat()),
        )
        import hashlib as _hashlib
        body = b"abcdefghij"  # len 10 so span 1:4 is valid
        connection.execute(
            "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, "
            "subject_id, predicate, value_json, chapter_start, chapter_end, source_artifact_id, "
            "source_artifact_revision, source_span, source_hash, confidence, created_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("mem-1", PROJECT_ID.root, "mem-run", 1, "m1_canonical", "character", "canonical",
             "程屿", "是", '"男主"', 1, 1, "mem-src", 1, "1:4", _hashlib.sha256(b"bcd").hexdigest(),
             1.0, "governed", FIXED_TIME.isoformat()),
        )
        hits = memory_query(connection, project_id=PROJECT_ID.root, query_terms=("程屿",))
        assert any(hit.subject_id == "程屿" for hit in hits)
    finally:
        database.__exit__(None, None, None)


def test_agent_loop_approval_pause_resumes_remaining_plan(projects_root: Path) -> None:
    """A plan spanning two chapters pauses at the first approval gate and, after
    approval, resumes the REMAINING plan steps automatically (regression: the
    second chapter was dropped after a pause)."""
    from novel_agent.default_stage import _AgentLoopStage
    from novel_agent.loop_state import load_loop_state
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
        resolver = ProjectPathResolver(projects_root.parent)

        # First run: full novel, chapters 1..3, auto-approved.
        first = S(project_id=PROJECT_ID, run_id="ap-1", idempotency_key="ap-1",
                  user_requirements="写一本三章的小说", provider="agent_loop", auto_approve=True)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="ap-1", idempotency_key="ap-1"))
        stage1 = _AgentLoopStage(
            coordinator, first, resolver, lambda: FIXED_TIME,
            transport=_LoopBriefTransport(chapter_count=5),
            loop_transport=_MultiChapterTransport(1, 2, 3, with_prereqs=True),
            offline_generators=True,
        )
        RunWorker(coordinator, stage1, stage_timeout_seconds=30.0).run(coordinator.claim_next("ap-w1"))
        assert connection.execute("SELECT state FROM runs WHERE run_id='ap-1'").fetchone() == (RunStatus.SUCCEEDED.value,)

        # Continuation run: plan writes chapters 4 and 5, but approval mode
        # pauses at chapter 4.
        cont = S(project_id=PROJECT_ID, run_id="ap-2", idempotency_key="ap-2",
                 user_requirements="续写第4章和第5章", provider="agent_loop", auto_approve=False)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="ap-2", idempotency_key="ap-2"))
        stage2 = _AgentLoopStage(
            coordinator, cont, resolver, lambda: FIXED_TIME,
            transport=_LoopBriefTransport(chapter_count=2),
            loop_transport=_MultiChapterTransport(4, 5),
            offline_generators=True,
        )
        RunWorker(coordinator, stage2, stage_timeout_seconds=30.0).run(coordinator.claim_next("ap-w2"))
        assert connection.execute("SELECT state FROM runs WHERE run_id='ap-2'").fetchone() == (RunStatus.WAITING_REVIEW.value,)
        state = load_loop_state(connection, "ap-2")
        assert state is not None
        # The remaining plan (chapter 5) was persisted across the pause.
        assert [tool for tool, _ in state.remaining_plan] == ["write_chapter"]

        # First pause already staged chapter 4: approve it to requeue the run.
        svc = NovelAgentService.from_config(now=lambda: FIXED_TIME)
        staged0 = connection.execute(
            "SELECT artifact_id, revision FROM artifacts WHERE run_id='ap-2' AND kind='chapter' AND status='staged' "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        assert staged0 is not None
        svc.approve_artifact(ApproveArtifactRequest(
            project_id=PROJECT_ID, run_id="ap-2",
            artifact_id=str(staged0[0]), revision=int(staged0[1]), idempotency_key="ap-appr-0",
        ))
        approved = 1
        # Remaining loop: claim -> resume (AUTO-executes chapter 5) -> approve.
        for round_no in range(1, 6):
            lease = coordinator.claim_next("ap-worker")
            if lease is None:
                break
            RunWorker(coordinator, stage2, stage_timeout_seconds=30.0).run(lease)
            state = connection.execute("SELECT state FROM runs WHERE run_id='ap-2'").fetchone()
            if state == (RunStatus.SUCCEEDED.value,):
                break
            staged = connection.execute(
                "SELECT artifact_id, revision FROM artifacts WHERE run_id='ap-2' AND kind='chapter' AND status='staged' "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if staged is None:
                break
            svc.approve_artifact(ApproveArtifactRequest(
                project_id=PROJECT_ID, run_id="ap-2",
                artifact_id=str(staged[0]), revision=int(staged[1]), idempotency_key=f"ap-appr-{round_no}",
            ))
            approved += 1
            state = connection.execute("SELECT state FROM runs WHERE run_id='ap-2'").fetchone()
            if state == (RunStatus.SUCCEEDED.value,):
                break
        assert approved >= 2
        assert connection.execute("SELECT state FROM runs WHERE run_id='ap-2'").fetchone() == (RunStatus.SUCCEEDED.value,)
        # BOTH chapters 4 and 5 accepted.
        acc = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id='ap-2' AND kind='chapter' AND status='accepted'"
        ).fetchone()
        assert acc[0] >= 2
    finally:
        database.__exit__(None, None, None)


class _CancelledToken:
    def is_cancelled(self) -> bool:
        return True


def test_loop_aborts_on_cancellation() -> None:
    """A cancelled stage token stops the loop before any tool runs."""
    called: list[str] = []
    loop = AgentLoop(
        tools={"a": _recorder_tool("a", called)},
        transport=_SeqTransport(['{"plan": [{"tool": "a", "args": {}}]}']),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=10),
        cancellation=_CancelledToken(),
    )
    outcome, _history = loop.run(system_prompt="sys", task="任务")
    assert outcome == "cancelled"
    assert called == []


class _SlowPlanTransport:
    """Slows each model step so a CANCEL can land mid-run."""

    def __init__(self, actions: list[str]) -> None:
        self.actions = actions
        self.calls = 0

    def generate(self, request) -> ProviderTransportResult:
        sleep(0.5)
        content = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        return ProviderTransportResult(content=content, model_name="t", prompt_tokens=1, completion_tokens=1)


def test_agent_loop_cancel_mid_run_stops_stage(projects_root: Path) -> None:
    """A CANCEL issued mid-run reaches the stage via worker polling and stops it."""
    from novel_agent.default_stage import _AgentLoopStage
    from novel_agent.pathing import ProjectPathResolver
    from novel_agent.run_coordinator import RunCoordinator
    from novel_agent.run_models import StartRunRequest
    from novel_agent.run_worker import RunWorker
    from novel_agent.contracts import RunCommandType
    from novel_agent.run_models import ControlRunRequest
    from novel_agent.service_models import StartNovelRunRequest as S
    from threading import Thread

    database = connect_project_database(projects_root / PROJECT_ID.root)
    connection = database.__enter__()
    try:
        from novel_agent.storage import migrate

        migrate(connection)
        coordinator = RunCoordinator(connection, lambda: FIXED_TIME)
        resolver = ProjectPathResolver(projects_root.parent)
        req = S(project_id=PROJECT_ID, run_id="cancel-mid", idempotency_key="cancel-mid",
                user_requirements="写一章", provider="agent_loop", auto_approve=True)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="cancel-mid", idempotency_key="cancel-mid"))
        stage = _AgentLoopStage(
            coordinator, req, resolver, lambda: FIXED_TIME,
            transport=_LoopBriefTransport(chapter_count=2),
            loop_transport=_SlowPlanTransport([
                '{"plan": [{"tool": "write_world", "args": {}}]}',
                '{"done": true}',
            ]),
            offline_generators=True,
        )
        worker = RunWorker(coordinator, stage, stage_timeout_seconds=20.0)
        lease = coordinator.claim_next("w")
        assert lease is not None
        result: list[object] = []
        t = Thread(target=lambda: result.append(worker.run(lease)), daemon=True)
        t.start()
        sleep(0.3)  # let the loop start and enter the slow write_world model call
        coordinator.control(ControlRunRequest(run_id="cancel-mid", command_type=RunCommandType.CANCEL, idempotency_key="cancel-1"))
        t.join(15.0)
        deadline = monotonic() + 10.0
        state = connection.execute("SELECT state FROM runs WHERE run_id='cancel-mid'").fetchone()
        while state is not None and state[0] == RunStatus.CANCEL_REQUESTED.value and monotonic() < deadline:
            sleep(0.01)
            state = connection.execute("SELECT state FROM runs WHERE run_id='cancel-mid'").fetchone()
        assert state is not None
        assert state[0] == RunStatus.CANCELLED.value, f"state={state}"
    finally:
        database.__exit__(None, None, None)


def test_cancel_waiting_review_completes_immediately() -> None:
    """Cancelling a run waiting for review (nothing executing) completes at once."""
    from novel_agent.contracts import RunCommandType
    from novel_agent.run_state import next_state

    assert next_state(RunStatus.QUEUED, RunCommandType.CANCEL) is RunStatus.CANCELLED
    assert next_state(RunStatus.PAUSED, RunCommandType.CANCEL) is RunStatus.CANCELLED
    assert next_state(RunStatus.WAITING_REVIEW, RunCommandType.CANCEL) is RunStatus.CANCELLED
    assert next_state(RunStatus.RUNNING, RunCommandType.CANCEL) is RunStatus.CANCEL_REQUESTED


def test_rollback_run_removes_artifacts_and_restores_pointer(projects_root: Path) -> None:
    """Hard-rollback of a finished run deletes its artifacts/memory and falls
    the current outline pointer back to the previous run's revision."""
    from novel_agent.default_stage import _AgentLoopStage
    from novel_agent.pathing import ProjectPathResolver
    from novel_agent.run_coordinator import RunCoordinator
    from novel_agent.run_models import StartRunRequest
    from novel_agent.run_worker import RunWorker
    from novel_agent.service import NovelAgentService
    from novel_agent.service_models import StartNovelRunRequest as S

    database = connect_project_database(projects_root / PROJECT_ID.root)
    connection = database.__enter__()
    try:
        from novel_agent.storage import migrate

        migrate(connection)
        coordinator = RunCoordinator(connection, lambda: FIXED_TIME)
        resolver = ProjectPathResolver(projects_root.parent)
        svc = NovelAgentService.from_config(now=lambda: FIXED_TIME)

        # First run: full novel (outline + chapter 1..3), auto-approved.
        r1 = S(project_id=PROJECT_ID, run_id="rb-1", idempotency_key="rb-1",
               user_requirements="写一本三章的小说", provider="agent_loop", auto_approve=True)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="rb-1", idempotency_key="rb-1"))
        stage1 = _AgentLoopStage(
            coordinator, r1, resolver, lambda: FIXED_TIME,
            transport=_LoopBriefTransport(chapter_count=5),
            loop_transport=_MultiChapterTransport(1, 2, 3, with_prereqs=True),
            offline_generators=True,
        )
        RunWorker(coordinator, stage1, stage_timeout_seconds=30.0).run(coordinator.claim_next("rb-w1"))
        assert connection.execute("SELECT state FROM runs WHERE run_id='rb-1'").fetchone() == (RunStatus.SUCCEEDED.value,)
        assert connection.execute("SELECT COUNT(*) FROM artifacts WHERE run_id='rb-1' AND kind='chapter' AND status='accepted'").fetchone()[0] >= 3

        # Second run: continuation writes chapter 4 (new outline revision, advances is_current).
        r2 = S(project_id=PROJECT_ID, run_id="rb-2", idempotency_key="rb-2",
               user_requirements="续写第4章", provider="agent_loop", auto_approve=True)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="rb-2", idempotency_key="rb-2"))
        stage2 = _AgentLoopStage(
            coordinator, r2, resolver, lambda: FIXED_TIME,
            transport=_LoopBriefTransport(chapter_count=1),
            loop_transport=_MultiChapterTransport(4),
            offline_generators=True,
        )
        RunWorker(coordinator, stage2, stage_timeout_seconds=30.0).run(coordinator.claim_next("rb-w2"))
        assert connection.execute("SELECT state FROM runs WHERE run_id='rb-2'").fetchone() == (RunStatus.SUCCEEDED.value,)
        plan_id = connection.execute(
            "SELECT artifact_id FROM artifacts WHERE kind='plan_contract' AND is_current=1"
        ).fetchone()[0]
        # rb-2 mirrored the outline under its own run and made it current.
        current_run = connection.execute(
            "SELECT run_id FROM artifacts WHERE artifact_id=? AND is_current=1", (str(plan_id),)
        ).fetchone()[0]
        assert current_run == "rb-2"
        assert connection.execute("SELECT COUNT(*) FROM artifacts WHERE run_id='rb-2' AND kind='chapter' AND status='accepted'").fetchone()[0] >= 1

        # Roll back rb-2: chapter 4 and rb-2's outline revision disappear; the
        # current outline falls back to rb-1's revision; rb-2 becomes cancelled.
        svc.rollback_run(PROJECT_ID, "rb-2")
        assert connection.execute("SELECT state FROM runs WHERE run_id='rb-2'").fetchone() == (RunStatus.CANCELLED.value,)
        assert connection.execute("SELECT COUNT(*) FROM artifacts WHERE run_id='rb-2'").fetchone()[0] == 0
        current_run_after = connection.execute(
            "SELECT run_id FROM artifacts WHERE artifact_id=? AND is_current=1", (str(plan_id),)
        ).fetchone()
        assert current_run_after is not None and current_run_after[0] == "rb-1"
        # rb-1's chapters are untouched.
        assert connection.execute("SELECT COUNT(*) FROM artifacts WHERE run_id='rb-1' AND kind='chapter' AND status='accepted'").fetchone()[0] >= 3
    finally:
        database.__exit__(None, None, None)


def test_rollback_intent_and_start_auto_rollback(projects_root: Path) -> None:
    """'去掉刚完成的内容' triggers a rollback of the latest succeeded run."""
    from novel_agent.default_stage import _AgentLoopStage
    from novel_agent.pathing import ProjectPathResolver
    from novel_agent.run_coordinator import RunCoordinator
    from novel_agent.run_models import StartRunRequest
    from novel_agent.run_worker import RunWorker
    from novel_agent.service import NovelAgentService, _is_rollback_intent
    from novel_agent.service_models import StartNovelRunRequest as S

    # Intent detection
    assert _is_rollback_intent("我觉得刚刚写的东西不对，去掉刚刚完成的内容")
    assert _is_rollback_intent("删除最近写的小说章节")
    assert not _is_rollback_intent("写一本魔法学院的小说")
    assert not _is_rollback_intent("续写第二章")

    database = connect_project_database(projects_root / PROJECT_ID.root)
    connection = database.__enter__()
    try:
        from novel_agent.storage import migrate

        migrate(connection)
        coordinator = RunCoordinator(connection, lambda: FIXED_TIME)
        resolver = ProjectPathResolver(projects_root.parent)
        r1 = S(project_id=PROJECT_ID, run_id="ri-1", idempotency_key="ri-1",
               user_requirements="写一本三章的小说", provider="agent_loop", auto_approve=True)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="ri-1", idempotency_key="ri-1"))
        stage1 = _AgentLoopStage(
            coordinator, r1, resolver, lambda: FIXED_TIME,
            transport=_LoopBriefTransport(chapter_count=5),
            loop_transport=_MultiChapterTransport(1, 2, 3, with_prereqs=True),
            offline_generators=True,
        )
        RunWorker(coordinator, stage1, stage_timeout_seconds=30.0).run(coordinator.claim_next("ri-w1"))
        assert connection.execute("SELECT state FROM runs WHERE run_id='ri-1'").fetchone() == (RunStatus.SUCCEEDED.value,)
        n_before = connection.execute("SELECT COUNT(*) FROM artifacts WHERE run_id='ri-1'").fetchone()[0]
        assert n_before > 0

        svc = NovelAgentService.from_config(now=lambda: FIXED_TIME)
        svc.start_auto(PROJECT_ID, "我觉得刚刚写的东西不对，去掉刚刚完成的内容", provider="agent_loop")
        # The latest successful run (ri-1) was rolled back.
        assert connection.execute("SELECT state FROM runs WHERE run_id='ri-1'").fetchone() == (RunStatus.CANCELLED.value,)
        assert connection.execute("SELECT COUNT(*) FROM artifacts WHERE run_id='ri-1'").fetchone()[0] == 0
    finally:
        database.__exit__(None, None, None)


class _FreshBriefTransport:
    """Brief analysis flags a brand-new story (fresh=true)."""

    def generate(self, request) -> ProviderTransportResult:
        payload = {
            "domain": "novel", "intent": "full_novel",
            "goal": "写一部全新的魔法学院小说", "constraints": [], "style_guidance": [],
            "acceptance_criteria": ["完全不同于旧故事"], "chapter_count": 2,
            "fresh": True,
        }
        return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False), model_name="t", prompt_tokens=1, completion_tokens=1)


def test_agent_loop_fresh_story_marks_brief_and_runs(projects_root: Path) -> None:
    """A 'new story' request marks the brief as fresh; the run still completes."""
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
        resolver = ProjectPathResolver(projects_root.parent)
        req = S(project_id=PROJECT_ID, run_id="fresh-run", idempotency_key="fresh-run",
                user_requirements="换个故事，写一部魔法学院小说", provider="agent_loop", auto_approve=True)
        coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="fresh-run", idempotency_key="fresh-run"))
        stage = _AgentLoopStage(
            coordinator, req, resolver, lambda: FIXED_TIME,
            transport=_FreshBriefTransport(),
            loop_transport=_MultiChapterTransport(1, 2, with_prereqs=True),
            offline_generators=True,
        )
        RunWorker(coordinator, stage, stage_timeout_seconds=30.0).run(coordinator.claim_next("fresh-w"))
        assert connection.execute("SELECT state FROM runs WHERE run_id='fresh-run'").fetchone() == (RunStatus.SUCCEEDED.value,)
        fresh = connection.execute(
            "SELECT payload_json FROM run_events WHERE run_id='fresh-run' AND event_type='brief_analyzed'"
        ).fetchone()
        assert fresh is not None
        assert json.loads(str(fresh[0])).get("fresh") is True
    finally:
        database.__exit__(None, None, None)


def test_tool_budget_caps_plan_auto_execution() -> None:
    called: list[dict] = []

    def execute(args: dict) -> ToolResult:
        called.append(args)
        return ToolResult(ok=True, summary="ok")

    plan = {"plan": [{"tool": "a", "args": {"x": i}} for i in range(5)]}
    loop = AgentLoop(
        tools={"a": AgentTool(name="a", description="a", input_schema={}, execute=execute)},
        transport=_SeqTransport([json.dumps(plan), '{"done": true}']),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=10, max_tool_calls=2, doom_loop_threshold=100),
    )
    outcome, _history = loop.run(system_prompt="sys", task="任务")

    assert outcome == "done"
    assert len(called) == 2


def test_plan_step_cap_is_enforced() -> None:
    called: list[dict] = []
    events: list[tuple[str, dict]] = []

    def execute(args: dict) -> ToolResult:
        called.append(args)
        return ToolResult(ok=True, summary="ok")

    plan = {"plan": [{"tool": "a", "args": {"x": i}} for i in range(5)]}
    loop = AgentLoop(
        tools={"a": AgentTool(name="a", description="a", input_schema={}, execute=execute)},
        transport=_SeqTransport([json.dumps(plan), '{"done": true}']),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=10, max_plan_steps=2, doom_loop_threshold=100),
        event_sink=lambda event_type, payload: events.append((event_type, payload)),
    )
    outcome, _history = loop.run(system_prompt="sys", task="任务")

    assert outcome == "done"
    assert len(called) == 2
    assert any(event_type == "loop_plan_truncated" for event_type, _ in events)


def test_repeated_successful_write_is_blocked() -> None:
    called: list[dict] = []
    events: list[tuple[str, dict]] = []

    def execute(args: dict) -> ToolResult:
        called.append(args)
        return ToolResult(ok=True, summary="ok")

    step = {"tool": "a", "args": {"x": 1}}
    plan = {"plan": [step, step, step]}
    loop = AgentLoop(
        tools={"a": AgentTool(name="a", description="a", input_schema={}, execute=execute)},
        transport=_SeqTransport([json.dumps(plan), '{"done": true}']),
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=10, max_tool_calls=20, doom_loop_threshold=100),
        event_sink=lambda event_type, payload: events.append((event_type, payload)),
    )
    outcome, _history = loop.run(system_prompt="sys", task="任务")

    assert outcome == "done"
    assert len(called) == 2
    assert any(event_type == "loop_repeat_detected" for event_type, _ in events)


def test_intent_filters_write_tools() -> None:
    from novel_agent.novel_tools import filter_tools_by_intent

    tools = {
        name: AgentTool(name=name, description=name, input_schema={}, execute=lambda args: ToolResult(ok=True))
        for name in (
            "write_world",
            "write_cast",
            "add_character",
            "write_outline",
            "write_chapter",
            "mem_query",
            "plan_query",
            "review_chapter",
            "read_chapter",
        )
    }
    world_only = filter_tools_by_intent(tools, "world_only")
    assert "write_world" in world_only
    assert "write_chapter" not in world_only
    assert "write_cast" not in world_only

    characters_only = filter_tools_by_intent(tools, "characters_only")
    assert {"write_world", "write_cast"} <= set(characters_only)
    assert "write_outline" not in characters_only
    assert "write_chapter" not in characters_only

    outline_only = filter_tools_by_intent(
        tools,
        "outline_only",
        existing_assets={"world_text": "w", "characters": [{"name": "a"}]},
    )
    assert "write_outline" in outline_only
    assert "write_world" not in outline_only
    assert "write_cast" not in outline_only
    assert "write_chapter" not in outline_only

    add_character = filter_tools_by_intent(tools, "add_character")
    assert "add_character" in add_character
    assert "write_cast" not in add_character
    assert "write_world" not in add_character

    composite = filter_tools_by_intent(
        tools,
        "characters_only",
        intents=("characters_only", "outline_only"),
    )
    assert {"write_world", "write_cast", "write_outline"} <= set(composite)
    assert "write_chapter" not in composite


def test_filtered_plan_step_is_reported_to_model() -> None:
    called: list[str] = []
    events: list[tuple[str, dict]] = []

    def execute(name: str):
        def run(args: dict) -> ToolResult:
            called.append(name)
            return ToolResult(ok=True, summary="ok")

        return run

    transport = _SeqTransport([
        json.dumps({"plan": [{"tool": "a", "args": {}}, {"tool": "b", "args": {}}]}),
        '{"done": true}',
    ])
    loop = AgentLoop(
        tools={"a": AgentTool(name="a", description="a", input_schema={}, execute=execute("a"))},
        transport=transport,
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=6, doom_loop_threshold=100),
        event_sink=lambda event_type, payload: events.append((event_type, payload)),
    )
    outcome, history = loop.run(system_prompt="sys", task="任务")

    assert outcome == "done"
    assert called == ["a"]
    assert any(event_type == "loop_plan_filtered" for event_type, _ in events)
    assert any("不在当前意图允许范围内" in message.content for message in history)


def test_plan_sequence_rejects_wrong_order_and_duplicates() -> None:
    from novel_agent.agent_loop import validate_plan_sequence

    assert "write_cast_before_write_world" in validate_plan_sequence([
        ("write_cast", {}), ("write_world", {})
    ])
    assert "write_chapter_before_write_outline" in validate_plan_sequence([
        ("write_chapter", {"chapter_number": 1}), ("write_outline", {})
    ])
    assert "duplicate_chapter_number" in validate_plan_sequence([
        ("write_chapter", {"chapter_number": 1}), ("write_chapter", {"chapter_number": 1})
    ])
    assert "chapter_numbers_out_of_order" in validate_plan_sequence([
        ("write_chapter", {"chapter_number": 2}), ("write_chapter", {"chapter_number": 1})
    ])
    assert validate_plan_sequence([
        ("write_world", {}),
        ("write_cast", {}),
        ("write_outline", {}),
        ("write_chapter", {"chapter_number": 1}),
        ("write_chapter", {"chapter_number": 2}),
    ]) == ()


def test_invalid_plan_order_is_rejected_before_execution() -> None:
    called: list[str] = []
    events: list[tuple[str, dict]] = []

    def execute(args: dict) -> ToolResult:
        called.append("cast")
        return ToolResult(ok=True, summary="ok")

    transport = _SeqTransport([
        json.dumps({"plan": [{"tool": "write_cast", "args": {}}, {"tool": "write_world", "args": {}}]}),
        '{"done": true}',
    ])
    loop = AgentLoop(
        tools={
            "write_cast": AgentTool(name="write_cast", description="c", input_schema={}, execute=execute),
            "write_world": AgentTool(
                name="write_world", description="w", input_schema={}, execute=lambda args: ToolResult(ok=True)
            ),
        },
        transport=transport,
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=6, doom_loop_threshold=100),
        event_sink=lambda event_type, payload: events.append((event_type, payload)),
    )
    outcome, _history = loop.run(system_prompt="sys", task="任务")

    assert outcome == "done"
    assert called == []
    assert any(event_type == "loop_plan_rejected" for event_type, _ in events)
