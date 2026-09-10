"""opencode-style agent loop for the novel domain.

Replaces the fixed Plan-and-Execute DAG with a model-driven loop: at every
step the model decides which novel tool to call next (and with what args),
observes a structured summary of the result, and repeats until it stops
emitting tool calls (implicit termination, matching opencode/codex).  All
domain governance (approval staging, memory commit, quality gate, artifact
persistence, tool idempotency/audit) lives inside the tools; this module only
decides *which* tool runs next.

Aligned with the codex/opencode consensus (see docs/agent-loop-risks.md):
- implicit termination: no tool-call in the model output == done
- model-layer retry with backoff (provider failures), tools are NOT retried
- doom-loop detection: N identical (tool, args) calls -> surface for approval
- context overflow -> compaction (head/tail + structured summary), never retry
- ToolResult.summary feeds back (truncated), full output stays in artifacts
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

from novel_agent.provider_types import ProviderRequest, ProviderTransport, ProviderTransportError


# -- loop primitives ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured result fed back to the model after a tool call."""

    ok: bool
    summary: str = ""
    artifact_ref: str | None = None
    error: str | None = None
    requires_approval: bool = False
    staged_artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class AgentTool:
    """A capability the loop model may invoke."""

    name: str
    description: str
    input_schema: dict
    execute: Callable[[dict], ToolResult]
    requires_approval: bool = False
    supports_parallel: bool = False  # read-only tools may run in parallel


@dataclass(frozen=True, slots=True)
class AgentAction:
    """The model's decision for one step."""

    kind: Literal["tool", "done"]
    tool: str | None = None
    args: dict | None = None
    thought: str | None = None


@dataclass(frozen=True, slots=True)
class LoopMessage:
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_ok: bool = True


@dataclass(frozen=True, slots=True)
class AgentLoopContext:
    """Per-step decision context (distinct from the per-chapter domain context)."""

    system_prompt: str
    history: tuple[LoopMessage, ...]
    memory_snapshot: str = ""
    recent_artifacts: tuple[str, ...] = ()


# -- loop configuration -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoopConfig:
    max_steps: int = 40
    doom_loop_threshold: int = 3
    # context compaction thresholds (token estimates)
    context_window: int = 32_000
    compact_buffer: int = 4_000
    compact_at: int | None = None  # None -> context_window - compact_buffer
    # retry (model layer)
    retry_max: int = 2
    retry_initial_delay_ms: int = 500
    retry_backoff_factor: float = 2.0
    # history window (most recent K messages survive, older head gets summarized)
    history_window: int = 20
    summary_max_chars: int = 8_000

    @property
    def compact_threshold(self) -> int:
        if self.compact_at is not None:
            return self.compact_at
        return self.context_window - self.compact_buffer


# -- model step protocol ------------------------------------------------------


class LoopModel(Protocol):
    def step(self, prompt: str) -> str: ...


# -- action parsing -----------------------------------------------------------


_ACTION_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(raw: str) -> AgentAction | None:
    """Parse the model's step output into an AgentAction.

    The model emits a JSON object: ``{"tool": "write_chapter", "args": {...}}``
    or ``{"done": true}`` / ``{"reply": "..."}``.  A JSON envelope with no
    ``tool`` key means the model finished (implicit termination).  Tolerates
    markdown code fences and stray prose like the brief/plan parsers.
    """
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    match = _ACTION_JSON.search(text)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        try:
            from json_repair import repair_json

            payload = json.loads(repair_json(match.group(0)))
        except Exception:  # noqa: BROAD_EXCEPT_OK - parsing is best-effort
            return None
    if not isinstance(payload, dict):
        return None
    tool = payload.get("tool")
    if isinstance(tool, str) and tool:
        args = payload.get("args")
        return AgentAction(
            kind="tool",
            tool=tool,
            args=args if isinstance(args, dict) else None,
            thought=str(payload.get("thought") or "") or None,
        )
    # No tool key: the model finished.
    return AgentAction(kind="done", thought=str(payload.get("reply") or "") or None)


def _truncate(text: str, limit: int = 2_000) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (CJK-aware): ~1 token per CJK char, 4 per ASCII."""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_chars = len(text) - cjk
    return cjk + ascii_chars // 4


def build_loop_context(
    *,
    system_prompt: str,
    history: Sequence[LoopMessage],
    tools: Sequence[AgentTool],
    memory_snapshot: str = "",
    recent_artifacts: Sequence[str] = (),
) -> AgentLoopContext:
    return AgentLoopContext(
        system_prompt=system_prompt,
        history=tuple(history),
        memory_snapshot=memory_snapshot,
        recent_artifacts=tuple(recent_artifacts),
    )


def render_loop_prompt(ctx: AgentLoopContext, tools: Sequence[AgentTool]) -> str:
    """Render the full model prompt for one step: system + tools + history."""
    tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    lines: list[str] = [ctx.system_prompt, "\n## 可用工具\n" + tool_lines]
    if ctx.memory_snapshot:
        lines.append("\n## 已有设定\n" + ctx.memory_snapshot)
    if ctx.recent_artifacts:
        lines.append("\n## 最近产物\n" + "\n".join(f"- {a}" for a in ctx.recent_artifacts))
    lines.append("\n## 对话历史\n")
    for msg in ctx.history:
        if msg.role == "user":
            lines.append(f"用户: {msg.content}")
        elif msg.role == "assistant":
            lines.append(f"助手: {msg.content}")
        else:  # tool observation
            lines.append(f"工具[{msg.tool_name}]结果: {_truncate(msg.content)}")
    lines.append("\n请输出下一步动作 JSON：{\"tool\": \"工具名\", \"args\": {...}} 或 {\"done\": true}")
    return "\n".join(lines)


# -- the loop ----------------------------------------------------------------


@dataclass(slots=True)
class AgentLoop:
    """Model-driven tool loop with implicit termination and compaction.

    The caller owns the tools, the transport and the event sink; the loop
    manages the conversation history, retries, doom-loop detection and
    context compaction.  Tools are invoked through the caller-provided
    ``invoke`` callable so the ToolGateway idempotency/audit stays in charge.
    """

    tools: dict[str, AgentTool]
    transport: ProviderTransport
    now: Callable[[], datetime]
    model_name: str = "deepseek-v4-pro"
    config: LoopConfig = field(default_factory=LoopConfig)
    event_sink: Callable[[str, dict[str, object]], None] | None = None
    failure_sink: Callable[[str, str], None] | None = None
    invoke: Callable[[str, dict], ToolResult] | None = None

    def __post_init__(self) -> None:
        if self.invoke is None:
            self.invoke = self._default_invoke

    def _default_invoke(self, tool: str, args: dict) -> ToolResult:
        spec = self.tools.get(tool)
        if spec is None:
            return ToolResult(ok=False, error=f"unknown tool: {tool}")
        try:
            return spec.execute(args)
        except Exception as error:  # noqa: BROAD_EXCEPT_OK - surface to model
            return ToolResult(ok=False, error=f"{type(error).__name__}: {error}")

    def _call_model(self, prompt: str) -> str | None:
        """One model step with backoff retry (model layer, not tool layer)."""
        from novel_agent.contracts import Provenance, ProvenanceKind

        timestamp = self.now()
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        request = ProviderRequest(
            provider_call_id=f"agent-loop:{prompt_sha256[:12]}",
            run_id="agent-loop",
            task_id="agent-loop:step",
            model_name=self.model_name,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            temperature=0.5,
            created_at=timestamp,
            provenance=Provenance(
                source_kind=ProvenanceKind.PROVIDER_RESPONSE,
                source_id="agent-loop:step",
                source_sha256=prompt_sha256,
                observed_at=timestamp,
            ),
        )
        delay = self.config.retry_initial_delay_ms / 1000.0
        for attempt in range(self.config.retry_max + 1):
            try:
                return self.transport.generate(request).content
            except ProviderTransportError as error:
                if self.failure_sink is not None:
                    self.failure_sink("agent_loop", str(error)[:300])
                if attempt < self.config.retry_max:
                    time.sleep(delay)
                    delay *= self.config.retry_backoff_factor
        return None

    def run(
        self,
        *,
        system_prompt: str,
        task: str,
        memory_snapshot: str = "",
        recent_artifacts: Sequence[str] = (),
    ) -> tuple[str, list[LoopMessage]]:
        """Run the loop until done / step limit. Returns (outcome, history)."""
        history: list[LoopMessage] = [LoopMessage(role="user", content=task)]
        doom_tally: list[tuple[str, str]] = []
        doomed: set[tuple[str, str]] = set()
        recent_tool_calls: list[str] = []
        total_chars = len(system_prompt) + len(task) + len(memory_snapshot)

        for step in range(self.config.max_steps):
            if self.event_sink is not None:
                self.event_sink("loop_step", {"step": step})
            # Compaction: when the accumulated history exceeds the threshold,
            # drop the oldest non-user messages and keep the recent window.
            if _estimate_tokens(render_loop_prompt(
                build_loop_context(
                    system_prompt=system_prompt,
                    history=history,
                    tools=list(self.tools.values()),
                    memory_snapshot=memory_snapshot,
                    recent_artifacts=recent_artifacts,
                ),
                list(self.tools.values()),
            )) > self.config.compact_threshold and len(history) > self.config.history_window:
                # keep task + most recent K messages (head/tail split)
                tail = history[-self.config.history_window :]
                if history[0].role == "user" and history[0] not in tail:
                    tail = [history[0], *tail]
                history = tail
                total_chars = 0
                if self.event_sink is not None:
                    self.event_sink("loop_compacted", {"kept": len(history)})

            prompt = render_loop_prompt(
                build_loop_context(
                    system_prompt=system_prompt,
                    history=history,
                    tools=list(self.tools.values()),
                    memory_snapshot=memory_snapshot,
                    recent_artifacts=recent_artifacts,
                ),
                list(self.tools.values()),
            )
            raw = self._call_model(prompt)
            if raw is None:
                if self.event_sink is not None:
                    self.event_sink("loop_provider_failed", {})
                return "provider_failed", tuple(history)
            action = parse_action(raw)
            if action is None:
                history.append(LoopMessage(role="assistant", content=f"（无法解析的响应，请输出合法 JSON）"))
                continue
            if action.kind == "done":
                if self.event_sink is not None:
                    self.event_sink("loop_done", {"reply": action.thought or ""})
                return "done", tuple(history)
            # tool call
            if action.tool is None or action.tool not in self.tools:
                history.append(LoopMessage(
                    role="assistant",
                    content=f"未知工具 {action.tool}，请从可用工具中选择。",
                ))
                continue
            args = action.args or {}
            args_json = json.dumps(args, sort_keys=True, ensure_ascii=False)
            key = (action.tool, args_json)
            # Doom-loop guard: a tool+args that already repeated N times is
            # blacklisted for the rest of the loop; asking again only wastes
            # steps, so reject it with guidance instead of executing.
            if key in doomed:
                history.append(LoopMessage(
                    role="assistant",
                    content=f"工具 {action.tool}（参数相同）此前已重复执行被判定为循环，请勿再次调用；"
                            "请换一种策略，或输出 {\"done\": true} 结束。",
                ))
                continue
            # doom-loop detection: N identical (tool, args)
            doom_tally.append(key)
            if len(doom_tally) >= self.config.doom_loop_threshold:
                last_n = doom_tally[-self.config.doom_loop_threshold :]
                if len(set(last_n)) == 1:
                    doomed.add(key)
                    doom_tally = []
                    if self.event_sink is not None:
                        self.event_sink("loop_doom_detected", {"tool": action.tool, "args": args_json})
                    history.append(LoopMessage(
                        role="assistant",
                        content=(
                            f"检测到重复调用 {action.tool}（参数相同）已达 {self.config.doom_loop_threshold} 次。"
                            "该调用已被禁用。请换一种策略，或确认已完成该步骤后输出 {\"done\": true}。"
                        ),
                    ))
                    continue
            result = self.invoke(action.tool, args)
            history.append(LoopMessage(
                role="tool",
                content=result.summary if result.ok else (result.error or "工具执行失败"),
                tool_name=action.tool,
                tool_args=args,
                tool_ok=result.ok,
            ))
            recent_tool_calls.append(action.tool)
            if self.event_sink is not None:
                self.event_sink("tool_call", {"tool": action.tool, "ok": result.ok})
        if self.event_sink is not None:
            self.event_sink("loop_step_limit", {"steps": self.config.max_steps})
        return "step_limit", tuple(history)


__all__ = (
    "AgentAction",
    "AgentLoop",
    "AgentLoopContext",
    "AgentTool",
    "LoopConfig",
    "LoopMessage",
    "ToolResult",
    "build_loop_context",
    "parse_action",
    "render_loop_prompt",
)

