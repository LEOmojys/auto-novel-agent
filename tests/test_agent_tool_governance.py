"""Agent tool-call governance: schema validation, error observations, prompts."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace

from pydantic import BaseModel

from novel_agent.agent_loop import (
    AgentLoop,
    AgentTool,
    LoopConfig,
    ToolResult,
    build_loop_context,
    parse_action,
    render_loop_prompt,
)
from novel_agent.provider_types import ProviderTransportResult


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


class _Args(BaseModel):
    x: int


class _Transport:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, request, cancellation=None) -> ProviderTransportResult:
        content = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return ProviderTransportResult(
            content=content, model_name="scripted", prompt_tokens=1, completion_tokens=1
        )


def _event_sink(events: list[tuple[str, dict]]):
    def sink(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    return sink


def test_invalid_arguments_become_an_observation_and_tool_is_not_called() -> None:
    calls: list[dict] = []

    def execute(args: dict) -> ToolResult:
        calls.append(args)
        return ToolResult(ok=True, summary="ok")

    transport = _Transport([
        json.dumps({"plan": [{"tool": "a", "args": {"x": "not-an-int"}}]}),
        json.dumps({"done": True}),
    ])
    events: list[tuple[str, dict]] = []
    loop = AgentLoop(
        tools={"a": AgentTool(name="a", description="a", input_schema={}, execute=execute, input_model=_Args)},
        transport=transport,
        now=lambda: FIXED_TIME,
        event_sink=_event_sink(events),
        config=LoopConfig(max_steps=4),
    )
    outcome, history = loop.run(system_prompt="sys", task="任务")

    assert outcome == "done"
    assert calls == []
    assert any("参数校验失败" in message.content for message in history if message.role == "tool")
    tool_events = [payload for kind, payload in events if kind == "tool_call"]
    assert tool_events and tool_events[0]["ok"] is False
    assert tool_events[0]["error"]


def test_handler_exception_becomes_an_observation_instead_of_crashing_run() -> None:
    def execute(args: dict) -> ToolResult:
        raise RuntimeError("boom")

    transport = _Transport([
        json.dumps({"plan": [{"tool": "a", "args": {}}]}),
        json.dumps({"done": True}),
    ])
    loop = AgentLoop(
        tools={"a": AgentTool(name="a", description="a", input_schema={}, execute=execute)},
        transport=transport,
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=4),
    )
    outcome, history = loop.run(system_prompt="sys", task="任务")

    assert outcome == "done"
    assert any("工具异常" in message.content for message in history if message.role == "tool")


def test_input_schema_is_rendered_in_the_prompt() -> None:
    tool = AgentTool(
        name="write_chapter",
        description="写章节",
        input_schema={
            "type": "object",
            "properties": {
                "chapter_number": {"type": "integer"},
                "force_rewrite": {"type": "boolean"},
            },
            "required": ["chapter_number"],
        },
        execute=lambda args: ToolResult(ok=True),
    )
    prompt = render_loop_prompt(build_loop_context(system_prompt="sys", history=(), tools=[tool]), [tool])
    assert "chapter_number:integer(必填)" in prompt
    assert "force_rewrite:boolean(可选)" in prompt


def test_string_encoded_arguments_are_parsed() -> None:
    action = parse_action('{"tool": "a", "args": "{\\"x\\": 3}"}')
    assert action is not None
    assert action.tool == "a"
    assert action.args == {"x": 3}


# -- read-side tools -----------------------------------------------------------


def test_plan_query_and_read_chapter_tools(tmp_path) -> None:
    from novel_agent.contracts import (
        ChapterPlan,
        EvidenceRef,
        PlanApprovalStatus,
        PlanContract,
        Provenance,
        ProvenanceKind,
    )
    from novel_agent.novel_tools import _make_plan_query, _make_read_chapter

    provenance = Provenance(
        source_kind=ProvenanceKind.USER_REQUEST,
        source_id="test",
        source_sha256="0" * 64,
        observed_at=FIXED_TIME,
    )
    chapter = ChapterPlan(
        chapter_plan_id="chapter-1",
        chapter_number=1,
        title="开端",
        goal="主角出发",
        planned_events=(),
        evidence=(
            EvidenceRef(
                evidence_id="evidence-1",
                artifact_id="brief-1",
                artifact_revision=1,
                source_sha256="0" * 64,
                recorded_at=FIXED_TIME,
            ),
        ),
        created_at=FIXED_TIME,
        provenance=provenance,
    )
    plan = PlanContract(
        plan_id="plan-1",
        brief_id="brief-1",
        world_id="world-1",
        character_set_id="cast-1",
        approval_status=PlanApprovalStatus.VALIDATED,
        chapters=(chapter,),
        created_at=FIXED_TIME,
        provenance=provenance,
    )
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE artifacts(artifact_id TEXT, revision INTEGER, status TEXT, raw_bytes BLOB, "
        "is_current INTEGER, kind TEXT)"
    )
    connection.execute(
        "CREATE TABLE artifact_dependencies(artifact_id TEXT, artifact_revision INTEGER, "
        "depends_on_artifact_id TEXT, dependency_kind TEXT)"
    )
    connection.execute(
        "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?)",
        ("plan-1", 1, "accepted", plan.model_dump_json().encode(), 1, "plan_contract"),
    )
    connection.execute(
        "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?)",
        ("chapter-1", 1, "accepted", json.dumps({"passage": "第一章正文"}).encode(), 1, "chapter"),
    )
    connection.execute(
        "INSERT INTO artifact_dependencies VALUES (?, ?, ?, ?)",
        ("chapter-1", 1, "chapter-1", "chapter_plan"),
    )
    ctx = SimpleNamespace(
        plan_ref=("plan-1", 1),
        lease=SimpleNamespace(run_id="run-1"),
        coordinator=SimpleNamespace(connection=connection),
    )

    plan_result = _make_plan_query(ctx).execute({})
    assert plan_result.ok
    assert "第1章" in plan_result.summary

    read_result = _make_read_chapter(ctx).execute({"chapter_number": 1})
    assert read_result.ok
    assert "第一章正文" in read_result.summary
    assert read_result.artifact_ref == "chapter-1"
