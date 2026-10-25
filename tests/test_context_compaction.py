"""Tests for token-budgeted context compaction.

The previous implementation kept a fixed number of messages and summarised the
head, which could never converge when the retained messages exceeded the
budget.  These tests pin the new invariants: converge under the threshold,
keep the pinned task, carry a rolling summary with args/artifacts, and
microcompact oversized tool observations.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from novel_agent.agent_loop import (
    AgentLoop,
    AgentTool,
    LoopConfig,
    ToolResult,
    build_loop_context,
    render_loop_prompt,
    _estimate_tokens,
)
from novel_agent.context_summarizer import ProviderContextSummarizer
from novel_agent.loop_context import ContextManager, LoopMessage
from novel_agent.provider_types import ProviderTransportResult


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


def _tool_message(index: int, *, size: int = 600, text: str | None = None) -> LoopMessage:
    return LoopMessage(
        role="tool",
        content=text if text is not None else "字" * size,
        tool_name="write_chapter",
        tool_args={"chapter_number": index},
        tool_ok=True,
        artifact_ref=f"artifact-{index}",
    )


def _manager(**overrides) -> ContextManager:
    options = dict(
        context_window=2_000,
        compact_buffer=100,
        max_tail_messages=5,
        target_ratio=0.55,
        summary_max_chars=4_000,
        max_tool_chars=4_000,
    )
    options.update(overrides)
    return ContextManager(**options)


# -- manager -------------------------------------------------------------------


def test_compaction_converges_under_threshold() -> None:
    manager = _manager()
    history = [LoopMessage(role="user", content="任务", pinned=True), *(_tool_message(i) for i in range(12))]
    compacted, report = manager.maybe_compact(history, overhead_tokens=80)

    assert report is not None
    assert report.after_tokens <= manager.threshold
    assert len(compacted) < len(history)
    assert compacted[0].role == "user" and compacted[0].pinned


def test_rolling_summary_keeps_arguments_and_artifacts_across_compactions() -> None:
    manager = _manager()
    history = [LoopMessage(role="user", content="任务", pinned=True), *(_tool_message(i) for i in range(8))]
    compacted, first = manager.maybe_compact(history, overhead_tokens=80)
    assert first is not None
    assert "artifact-0" in manager.summary
    assert "chapter_number" in manager.summary

    # Continue the run: new observations must merge into the existing summary,
    # not replace it.
    compacted = [*compacted, *(_tool_message(i) for i in range(20, 26))]
    compacted, second = manager.maybe_compact(compacted, overhead_tokens=80)
    assert second is not None
    assert "artifact-0" in manager.summary
    assert "artifact-20" in manager.summary
    assert any(message.artifact_ref == "artifact-25" for message in compacted)


def test_microcompaction_trims_oversized_tool_observation() -> None:
    manager = _manager(context_window=800, compact_buffer=100, max_tool_chars=600)
    history = [
        LoopMessage(role="user", content="任务", pinned=True),
        _tool_message(1, text="开头" + "字" * 12_000 + "结尾"),
    ]
    compacted, report = manager.maybe_compact(history, overhead_tokens=50)
    assert report is not None
    tool_messages = [message for message in compacted if message.role == "tool"]
    assert tool_messages
    assert any("省略" in message.content for message in tool_messages)
    assert all(len(message.content) <= 2_000 for message in tool_messages)


def test_no_compaction_below_threshold() -> None:
    manager = _manager()
    history = [
        LoopMessage(role="user", content="任务", pinned=True),
        LoopMessage(role="assistant", content="继续"),
    ]
    compacted, report = manager.maybe_compact(history, overhead_tokens=20)
    assert report is None
    assert compacted == history


# -- agent loop integration ----------------------------------------------------


class _Transport:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, request, cancellation=None) -> ProviderTransportResult:
        self.calls += 1
        self.prompts.append(request.prompt)
        if self.calls == 1:
            content = json.dumps({"plan": [{"tool": "a", "args": {"i": 0}}]})
        else:
            content = json.dumps({"tool": "a", "args": {"i": self.calls}})
        return ProviderTransportResult(
            content=content,
            model_name="scripted",
            prompt_tokens=1,
            completion_tokens=1,
        )


def _big_tool() -> AgentTool:
    def execute(args: dict) -> ToolResult:
        return ToolResult(ok=True, summary="字" * 600, artifact_ref=f"artifact-{args.get('i')}")

    return AgentTool(name="a", description="a", input_schema={}, execute=execute)


def test_agent_loop_compacts_to_budget_and_keeps_progress() -> None:
    transport = _Transport()
    loop = AgentLoop(
        tools={"a": _big_tool()},
        transport=transport,
        now=lambda: FIXED_TIME,
        config=LoopConfig(
            max_steps=10,
            context_window=2_000,
            compact_buffer=100,
            history_window=5,
            doom_loop_threshold=1_000,
        ),
    )
    outcome, history = loop.run(system_prompt="sys", task="任务")
    prompt = render_loop_prompt(
        build_loop_context(system_prompt="sys", history=history, tools=list(loop.tools.values())),
        list(loop.tools.values()),
    )

    assert outcome == "step_limit"
    assert loop.steps_taken == 10
    assert _estimate_tokens(prompt) <= loop.config.compact_threshold
    assert history[0].pinned and history[0].role == "user"
    assert "artifact-" in loop.rolling_summary
    assert "chapter_number" not in loop.rolling_summary  # tool "a" has no such arg
    assert "{\"i\":" in loop.rolling_summary


def test_resumed_run_starts_with_previous_summary() -> None:
    class _DoneTransport:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, request, cancellation=None) -> ProviderTransportResult:
            self.prompts.append(request.prompt)
            return ProviderTransportResult(
                content='{"done": true}',
                model_name="scripted",
                prompt_tokens=1,
                completion_tokens=1,
            )

    transport = _DoneTransport()
    loop = AgentLoop(
        tools={"a": _big_tool()},
        transport=transport,
        now=lambda: FIXED_TIME,
        config=LoopConfig(max_steps=1, context_window=32_000),
    )
    outcome, _history = loop.run(
        system_prompt="sys",
        task="续跑",
        initial_summary="早期摘要：已完成世界观设定",
    )
    assert outcome == "done"
    assert "早期摘要" in transport.prompts[0]
    assert "摘要:" in transport.prompts[0]


# -- model-backed summary ------------------------------------------------------


def test_model_summary_is_used() -> None:
    calls: list[str] = []

    def summarizer(messages, previous: str) -> str:
        calls.append(previous)
        return "## 目标\n完成任务\n## 关键产物\nartifact-0"

    manager = _manager(summarizer=summarizer)
    history = [LoopMessage(role="user", content="任务", pinned=True), *(_tool_message(i) for i in range(8))]
    compacted, report = manager.maybe_compact(history, overhead_tokens=80)

    assert report is not None
    assert report.summary_source == "model"
    assert "artifact-0" in manager.summary
    assert calls  # the model summariser was actually invoked


def test_model_summary_failure_falls_back_to_deterministic() -> None:
    def failing(messages, previous: str) -> str | None:
        raise RuntimeError("provider down")

    manager = _manager(summarizer=failing)
    history = [LoopMessage(role="user", content="任务", pinned=True), *(_tool_message(i) for i in range(8))]
    _, report = manager.maybe_compact(history, overhead_tokens=80)

    assert report is not None
    assert report.summary_source == "deterministic"
    assert "artifact-0" in manager.summary


class _SummaryTransport:
    def __init__(self, content: str, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.prompts: list[str] = []

    def generate(self, request, cancellation=None):
        self.prompts.append(request.prompt)
        if self.error is not None:
            raise self.error
        return ProviderTransportResult(
            content=self.content,
            model_name="summary-model",
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_provider_summarizer_builds_bounded_prompt_and_strips_fences() -> None:
    transport = _SummaryTransport("\x60\x60\x60markdown\n## 摘要\n模型输出\n\x60\x60\x60")
    summarizer = ProviderContextSummarizer(
        transport=transport,
        now=lambda: FIXED_TIME,
        run_id="run-1",
        input_max_chars=400,
    )
    messages = [
        _tool_message(1, text="很长的观察" * 1000),
        LoopMessage(role="assistant", content="下一步写第二章"),
    ]
    summary = summarizer.summarize(messages, previous_summary="旧摘要")

    assert summary is not None
    assert summary.startswith("## 摘要")
    assert "旧摘要" in transport.prompts[0]
    assert "artifact-1" in transport.prompts[0]
    assert "chapter_number" in transport.prompts[0]
    assert len(transport.prompts[0]) < 2_000  # transcript was capped


def test_provider_summarizer_failure_returns_none() -> None:
    from novel_agent.provider_types import ProviderFailureKind, ProviderTransportError

    failures: list[tuple] = []
    summarizer = ProviderContextSummarizer(
        transport=_SummaryTransport("", error=ProviderTransportError(ProviderFailureKind.TRANSIENT, "boom")),
        now=lambda: FIXED_TIME,
        run_id="run-1",
        failure_sink=lambda *args: failures.append(args),
    )
    assert summarizer.summarize([_tool_message(1)], "old") is None
    assert failures and failures[0][0] == "context_summary"


def test_agent_loop_uses_injected_model_summary() -> None:
    transport = _Transport()
    loop = AgentLoop(
        tools={"a": _big_tool()},
        transport=transport,
        now=lambda: FIXED_TIME,
        config=LoopConfig(
            max_steps=8,
            context_window=2_000,
            compact_buffer=100,
            history_window=5,
            doom_loop_threshold=1_000,
        ),
        summarizer=lambda messages, previous: "## 模型摘要\n已写入 artifact",
    )
    outcome, history = loop.run(system_prompt="sys", task="任务")
    prompt = render_loop_prompt(
        build_loop_context(system_prompt="sys", history=history, tools=list(loop.tools.values())),
        list(loop.tools.values()),
    )
    assert outcome == "step_limit"
    assert "模型摘要" in loop.rolling_summary
    assert _estimate_tokens(prompt) <= loop.config.compact_threshold
