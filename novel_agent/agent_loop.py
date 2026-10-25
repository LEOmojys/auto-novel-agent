"""opencode-style agent loop for the novel domain.

Replaces the fixed Plan-and-Execute DAG with a model-driven loop: at every
step the model decides which novel tool to call next (and with what args),
observes a structured summary of the result, and repeats until it stops
emitting tool calls (implicit termination, matching opencode/codex).  All
domain governance (approval staging, memory commit, quality gate, artifact
persistence, tool idempotency/audit) lives inside the tools; this module only
decides *which* tool runs next.

Plan-Act: the first decision of a fresh run must be a plan (a list of steps);
the loop executes the steps in order on {"next": true} and lets the model
deviate (re-plan / direct tool call) after each observed result.

Aligned with the codex/opencode consensus (see docs/agent-loop-risks.md):
- implicit termination: no tool-call in the model output == done
- model-layer retry with backoff (provider failures), tools are NOT retried
- doom-loop detection: N identical (tool, args) calls -> surface for approval
- context overflow -> compaction (head/tail + structured summary), never retry
- ToolResult.summary feeds back (truncated), full output stays in artifacts
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

from novel_agent.loop_context import ContextManager, LoopMessage
from novel_agent.tool_runtime import ToolCancelledError, check_tool_active, tool_scope
from novel_agent.provider_types import (
    ProviderFailureKind,
    ProviderRequest,
    ProviderTransport,
    ProviderTransportError,
)


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
    #: A transient failure that may be retried by the loop (bounded by the tool).
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class AgentTool:
    """A capability the loop model may invoke."""

    name: str
    description: str
    input_schema: dict
    execute: Callable[[dict], ToolResult]
    requires_approval: bool = False
    supports_parallel: bool = False  # read-only tools may run in parallel
    #: Optional pydantic model used to validate/normalise model-provided args.
    input_model: type | None = None
    #: Advisory per-tool deadline (enforced by the governing wrapper when possible).
    timeout_seconds: float = 0.0
    #: Bounded retries for transient tool failures (provider/transport).
    max_attempts: int = 1
    retry_delay_seconds: float = 0.5


@dataclass(frozen=True, slots=True)
class AgentAction:
    """The model's decision for one step."""

    kind: Literal["tool", "done", "plan", "next"]
    tool: str | None = None
    args: dict | None = None
    plan: tuple[tuple[str, dict], ...] = ()
    thought: str | None = None


@dataclass(frozen=True, slots=True)
class AgentLoopContext:
    """Per-step decision context (distinct from the per-chapter domain context)."""

    system_prompt: str
    history: tuple[LoopMessage, ...]
    memory_snapshot: str = ""
    recent_artifacts: tuple[str, ...] = ()
    plan_status: str = ""


# -- loop configuration -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoopConfig:
    max_steps: int = 40
    #: Hard cap on actual tool executions per loop run (plan auto-advance included).
    max_tool_calls: int = 80
    #: A single plan may not contain more than this many steps.
    max_plan_steps: int = 24
    #: Wall-clock cap for one loop run (seconds; 0 disables).
    max_wall_seconds: float = 1_800.0
    doom_loop_threshold: int = 3
    # context compaction thresholds (token estimates)
    context_window: int = 32_000
    compact_buffer: int = 4_000
    compact_at: int | None = None  # None -> context_window - compact_buffer
    # retry (model layer)
    retry_max: int = 2
    retry_initial_delay_ms: int = 500
    retry_backoff_factor: float = 2.0
    # Minimum number of recent messages kept when selecting a token budget.
    history_window: int = 20
    # Fraction of the threshold kept after compaction (token-budgeted tail).
    compact_target_ratio: float = 0.55
    # Tool observations larger than this are head/tail trimmed before compaction.
    max_tool_chars: int = 4_000
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


def _coerce_args(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


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
    # Plan-act: the model may first emit an execution plan (a list of steps),
    # then advance through it with {"next": true}, or call a tool directly.
    plan_value = payload.get("plan")
    if isinstance(plan_value, list) and plan_value:
        steps: list[tuple[str, dict]] = []
        for item in plan_value:
            if isinstance(item, dict) and isinstance(item.get("tool"), str):
                steps.append((item["tool"], _coerce_args(item.get("args"))))
        if steps:
            return AgentAction(kind="plan", plan=tuple(steps), thought=str(payload.get("thought") or "") or None)
    if payload.get("next") is True:
        return AgentAction(kind="next", thought=str(payload.get("thought") or "") or None)
    tool = payload.get("tool")
    if isinstance(tool, str) and tool:
        args = _coerce_args(payload.get("args"))
        return AgentAction(
            kind="tool",
            tool=tool,
            args=args or None,
            thought=str(payload.get("thought") or "") or None,
        )
    # No tool key: the model finished.
    return AgentAction(kind="done", thought=str(payload.get("reply") or "") or None)


def _truncate(text: str, limit: int = 2_000) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _schema_hint(schema: dict) -> str:
    """Compact one-line argument description for the model prompt."""
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict) or not properties:
        return ""
    required = set(schema.get("required") or ())
    parts: list[str] = []
    for name, value in properties.items():
        kind = value.get("type", "any") if isinstance(value, dict) else "any"
        if isinstance(kind, list):
            kind = "/".join(str(item) for item in kind)
        marker = "必填" if name in required else "可选"
        parts.append(f"{name}:{kind}({marker})")
    return "，".join(parts)


def validate_plan_sequence(
    steps: Sequence[tuple[str, dict]],
    *,
    existing_world: bool = False,
    existing_cast: bool = False,
    existing_outline: bool = False,
) -> tuple[str, ...]:
    """Preflight a model-authored plan before any step executes.

    Tools themselves still validate domain preconditions, but obvious ordering
    and uniqueness mistakes should be rejected up front: a plan that puts
    write_chapter before write_outline, or repeats chapter 1, wastes a model
    round and can leave confusing observations in history.
    """
    violations: list[str] = []
    first: dict[str, int] = {}
    chapter_numbers: list[int] = []
    outline_index = next(
        (index for index, (tool, _args) in enumerate(steps) if tool == "write_outline"),
        None,
    )
    for index, (tool, args) in enumerate(steps):
        if tool in ("write_world", "write_cast", "add_character", "write_outline"):
            if tool in first:
                violations.append(f"duplicate_step:{tool}")
            else:
                first[tool] = index
        if tool == "write_chapter":
            number = (args or {}).get("chapter_number")
            if number is not None:
                if not isinstance(number, int) or number < 1:
                    violations.append("chapter_number_invalid")
                else:
                    chapter_numbers.append(number)
            if outline_index is not None and index < outline_index:
                violations.append("write_chapter_before_write_outline")
    for cast_tool in ("write_cast", "add_character"):
        if cast_tool in first and "write_world" in first and first[cast_tool] < first["write_world"]:
            violations.append(f"{cast_tool}_before_write_world")
    if any(tool in first for tool in ("write_cast", "add_character")) and "write_world" not in first and not existing_world:
        violations.append("missing_world_prerequisite")
    if "write_outline" in first:
        if "write_world" not in first and not existing_world:
            violations.append("missing_world_prerequisite")
        if not any(tool in first for tool in ("write_cast", "add_character")) and not existing_cast:
            violations.append("missing_cast_prerequisite")
    if any(tool == "write_chapter" for tool, _args in steps) and "write_outline" not in first and not existing_outline:
        violations.append("missing_outline_prerequisite")
    if "write_outline" in first:
        for prerequisite in ("write_world", "write_cast", "add_character"):
            if prerequisite in first and first["write_outline"] < first[prerequisite]:
                violations.append(f"write_outline_before_{prerequisite}")
    if chapter_numbers:
        if len(set(chapter_numbers)) != len(chapter_numbers):
            violations.append("duplicate_chapter_number")
        if chapter_numbers != sorted(chapter_numbers):
            violations.append("chapter_numbers_out_of_order")
    return tuple(dict.fromkeys(violations))


def _json_preview(value: object, limit: int = 500) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)[:limit]
    except (TypeError, ValueError):
        return str(value)[:limit]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (CJK-aware): ~1 token per CJK char, 4 per ASCII."""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_chars = len(text) - cjk
    return cjk + ascii_chars // 4


def _accepts_cancellation(generate) -> bool:
    """Whether a transport's generate() accepts an optional cancellation arg.

    The ProviderTransport protocol allows it, but scripted test transports
    keep the two-argument signature; inspect the callable to stay compatible.
    """
    try:
        parameters = tuple(inspect.signature(generate).parameters.values())
    except (TypeError, ValueError):
        return False
    for parameter in parameters:
        if parameter.name == "cancellation":
            return True
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def build_loop_context(
    *,
    system_prompt: str,
    history: Sequence[LoopMessage],
    tools: Sequence[AgentTool],
    memory_snapshot: str = "",
    recent_artifacts: Sequence[str] = (),
    plan_status: str = "",
) -> AgentLoopContext:
    return AgentLoopContext(
        system_prompt=system_prompt,
        history=tuple(history),
        memory_snapshot=memory_snapshot,
        recent_artifacts=tuple(recent_artifacts),
        plan_status=plan_status,
    )


def render_loop_prompt(ctx: AgentLoopContext, tools: Sequence[AgentTool]) -> str:
    """Render the full model prompt for one step: system + tools + history."""
    tool_lines_parts: list[str] = []
    for tool in tools:
        line = f"- {tool.name}: {tool.description}"
        hint = _schema_hint(tool.input_schema)
        if hint:
            line += f"\n    参数: {hint}"
        if tool.requires_approval:
            line += "（执行前/后可能需要用户确认）"
        tool_lines_parts.append(line)
    tool_lines = "\n".join(tool_lines_parts)
    lines: list[str] = [ctx.system_prompt, "\n## 可用工具\n" + tool_lines]
    if ctx.memory_snapshot:
        lines.append("\n## 已有设定\n" + ctx.memory_snapshot)
    if ctx.recent_artifacts:
        lines.append("\n## 最近产物\n" + "\n".join(f"- {a}" for a in ctx.recent_artifacts))
    if ctx.plan_status:
        lines.append("\n## 计划进度\n" + ctx.plan_status)
    lines.append("\n## 对话历史\n")
    for msg in ctx.history:
        if msg.is_summary:
            lines.append(f"摘要: {msg.content}")
        elif msg.role == "user":
            lines.append(f"用户: {msg.content}")
        elif msg.role == "assistant":
            lines.append(f"助手: {msg.content}")
        else:  # tool observation
            lines.append(f"工具[{msg.tool_name}]结果: {_truncate(msg.content)}")
    if ctx.plan_status:
        lines.append(
            "\n请输出下一步动作：{\"next\": true} 推进计划下一步；"
            "如需调整可输出新的 plan 或直接 {\"tool\": \"…\", \"args\": {…}}；"
            "完成后输出 {\"done\": true, \"reply\": \"总结\"}"
        )
    else:
        lines.append(
            "\n请先输出执行计划：{\"plan\": [{\"tool\": \"…\", \"args\": {…}}, ...]}"
            "（按依赖顺序，已有设定不要重复生成）；系统将按计划依次执行。"
        )
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
    cancellation: object | None = None  # optional is_cancelled() token (stage-driven)
    # Plan state survives a run() return so the stage can persist the remaining
    # plan across an approval pause and resume it on the next run().
    current_plan: tuple[tuple[str, dict], ...] = ()
    plan_index: int = 0
    #: Rolling compaction summary; persisted by the stage so resumed runs keep
    #: the model's compressed memory instead of dropping it.
    rolling_summary: str = ""
    #: Optional model-backed summarizer: (head messages, previous summary) -> summary.
    summarizer: Callable[[list[LoopMessage], str], str | None] | None = None
    #: Number of decision steps actually taken (not len(history), which compaction shrinks).
    steps_taken: int = 0
    #: Existing assets relax plan prerequisite checks (reused across runs).
    existing_world: bool = False
    existing_cast: bool = False
    existing_outline: bool = False
    #: Persist a lightweight checkpoint at the start of each decision step.
    checkpoint: Callable[[int, tuple[tuple[str, dict], ...], int, str], None] | None = None

    def __post_init__(self) -> None:
        if self.invoke is None:
            self.invoke = self._default_invoke

    def _default_invoke(self, tool: str, args: dict) -> ToolResult:
        spec = self.tools.get(tool)
        if spec is None:
            return ToolResult(ok=False, error=f"unknown tool: {tool}")
        try:
            return spec.execute(args)
        except ToolCancelledError:
            raise
        except Exception as error:  # noqa: BROAD_EXCEPT_OK - surface to model
            retryable = isinstance(error, ProviderTransportError) and error.kind in {
                ProviderFailureKind.TRANSIENT,
                ProviderFailureKind.TIMEOUT,
            }
            return ToolResult(
                ok=False,
                error=f"工具异常 {type(error).__name__}: {str(error)[:400]}",
                retryable=retryable,
            )

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
                generate = self.transport.generate
                if self.cancellation is not None and _accepts_cancellation(generate):
                    return generate(request, cancellation=self.cancellation).content
                return generate(request).content
            except ProviderTransportError as error:
                if self.failure_sink is not None:
                    self.failure_sink("agent_loop", str(error)[:300])
                if attempt < self.config.retry_max:
                    time.sleep(delay)
                    delay *= self.config.retry_backoff_factor
        return None

    def _invoke_safely(self, tool: str, args: dict) -> ToolResult:
        try:
            return self.invoke(tool, args)
        except ToolCancelledError:
            raise
        except Exception as error:  # tool failures must become observations, not run crashes
            retryable = isinstance(error, ProviderTransportError) and error.kind in {
                ProviderFailureKind.TRANSIENT,
                ProviderFailureKind.TIMEOUT,
            }
            return ToolResult(
                ok=False,
                error=f"工具异常 {type(error).__name__}: {str(error)[:400]}",
                retryable=retryable,
            )

    def run(
        self,
        *,
        system_prompt: str,
        task: str,
        memory_snapshot: str = "",
        recent_artifacts: Sequence[str] = (),
        initial_history: Sequence[LoopMessage] = (),
        initial_summary: str = "",
        initial_plan: Sequence[tuple[str, dict]] = (),
        initial_plan_index: int = 0,
    ) -> tuple[str, list[LoopMessage]]:
        """Run the loop until done / step limit. Returns (outcome, history).

        ``initial_history`` resumes a previously-paused loop (approval gate /
        repair): the caller replays the persisted tool-call trace so the model
        keeps its context instead of starting from scratch.

        ``initial_plan`` / ``initial_plan_index`` resume a plan that was
        paused at an approval gate: the remaining steps are executed first
        (auto-advance) before the model is asked to decide anything.
        """
        history: list[LoopMessage] = [LoopMessage(role="user", content=task, pinned=True)]
        if initial_history:
            # Replayed history never contains the new task; always append it so
            # a resumed decision loop keeps the current request.
            history = [*initial_history, *history]
        if initial_summary:
            history.insert(0, LoopMessage(role="assistant", content=initial_summary, is_summary=True))
        manager = ContextManager(
            context_window=self.config.context_window,
            compact_buffer=self.config.compact_buffer,
            max_tail_messages=max(2, self.config.history_window),
            target_ratio=self.config.compact_target_ratio,
            summary_max_chars=self.config.summary_max_chars,
            max_tool_chars=self.config.max_tool_chars,
            summarizer=self.summarizer,
        )
        manager.seed_summary(initial_summary)
        self.rolling_summary = manager.summary
        doom_tally: list[tuple[str, str]] = []
        doomed: set[tuple[str, str]] = set()
        repeat_doomed: set[tuple[str, str]] = set()
        recent_tool_calls: list[str] = []
        tool_calls_made = 0
        destructive_counts: dict[tuple[str, str], int] = {}
        run_started = time.monotonic()
        # Plan-Act state: the model may first emit a plan (list of steps) and
        # the loop auto-executes it; the plan survives on the instance so an
        # approval pause can persist and later resume the remaining steps.
        self.current_plan = tuple(initial_plan) if initial_plan else ()
        self.plan_index = initial_plan_index
        current_plan: list[tuple[str, dict]] = list(self.current_plan)
        plan_index = self.plan_index

        def plan_status_text() -> str:
            if not current_plan:
                return ""
            done_n = min(plan_index, len(current_plan))
            steps = ", ".join(f"{i + 1}){tool}" for i, (tool, _args) in enumerate(current_plan))
            return f"已执行 {done_n}/{len(current_plan)} 步（{steps}）"

        def blocked_reason(tool: str, args: dict) -> str | None:
            args_json = json.dumps(args, sort_keys=True, ensure_ascii=False)
            if (tool, args_json) in doomed:
                return (
                    f"工具 {tool}（参数相同）此前已重复失败被判定为循环，请勿再次调用；"
                    "请换一种策略，或输出 {\"done\": true} 结束。"
                )
            if (tool, args_json) in repeat_doomed:
                return (
                    f"工具 {tool}（参数相同）已成功执行多次，判定为重复写入，请勿再次调用；"
                    "请继续新步骤，或输出 {\"done\": true} 结束。"
                )
            return None

        def record_outcome(tool: str, args: dict, result: ToolResult) -> None:
            nonlocal doom_tally
            args_json = json.dumps(args, sort_keys=True, ensure_ascii=False)
            key = (tool, args_json)
            if result.ok or result.requires_approval:
                # Successful calls (including legitimate polling) are progress;
                # they must not poison the failure tally.
                doom_tally.clear()
                return
            doom_tally.append(key)
            if len(doom_tally) >= self.config.doom_loop_threshold:
                last_n = doom_tally[-self.config.doom_loop_threshold :]
                if len(set(last_n)) == 1:
                    doomed.add(key)
                    doom_tally = []
                    if self.event_sink is not None:
                        self.event_sink("loop_doom_detected", {"tool": tool, "args": args_json})

        def execute_step(tool: str, args: dict) -> ToolResult:
            started = time.perf_counter()
            spec = self.tools[tool]
            call_args = args
            result = ToolResult(ok=False, error="工具执行失败")
            attempts = max(1, int(spec.max_attempts))
            tool_started = time.monotonic()
            for attempt in range(attempts):
                if spec.input_model is not None:
                    try:
                        validated = spec.input_model.model_validate(args)
                        call_args = validated.model_dump(exclude_none=True)
                    except Exception as error:  # validation error is an observation, not a crash
                        message = f"参数校验失败：{type(error).__name__}: {error}"[:600]
                        result = ToolResult(ok=False, error=message)
                        break
                try:
                    remaining_timeout = spec.timeout_seconds
                    if remaining_timeout and remaining_timeout > 0.0:
                        remaining_timeout -= time.monotonic() - tool_started
                        remaining_timeout = max(0.001, remaining_timeout)
                    with tool_scope(self.cancellation, remaining_timeout):
                        check_tool_active()
                        result = self._invoke_safely(tool, call_args)
                        if spec.supports_parallel:
                            # Read-only tools can safely be marked timed out after
                            # they return; write tools are checked before commits.
                            check_tool_active()
                except ToolCancelledError as error:
                    result = ToolResult(ok=False, error=f"工具中止：{error}")
                    break
                if result.ok or not result.retryable or attempt + 1 >= attempts:
                    break
                time.sleep(max(0.0, spec.retry_delay_seconds) * (2 ** attempt))
            history.append(LoopMessage(
                role="tool",
                content=result.summary if result.ok else (result.error or "工具执行失败"),
                tool_name=tool,
                tool_args=call_args,
                tool_ok=result.ok,
                artifact_ref=result.artifact_ref,
            ))
            recent_tool_calls.append(tool)
            if self.event_sink is not None:
                self.event_sink("tool_call", {
                    "tool": tool,
                    "ok": result.ok,
                    "summary": (result.summary or "")[:400],
                    "requires_approval": result.requires_approval,
                    "artifact_ref": result.artifact_ref,
                    "error": (result.error or "")[:300],
                    "args": _json_preview(call_args),
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "attempts": attempt + 1,
                })
            return result

        def run_tool(tool: str, args: dict) -> ToolResult | None:
            """Doom/budget guard then execute; returns None when blocked."""
            nonlocal tool_calls_made
            if tool_calls_made >= self.config.max_tool_calls:
                history.append(LoopMessage(
                    role="assistant",
                    content=f"工具调用预算已耗尽（{self.config.max_tool_calls} 次），请输出 done 或调整任务。",
                ))
                return None
            reason = blocked_reason(tool, args)
            if reason is not None:
                history.append(LoopMessage(role="assistant", content=reason))
                return None
            tool_calls_made += 1
            result = execute_step(tool, args)
            record_outcome(tool, args, result)
            if result.ok and not self.tools[tool].supports_parallel:
                args_json = json.dumps(args, sort_keys=True, ensure_ascii=False)
                key = (tool, args_json)
                destructive_counts[key] = destructive_counts.get(key, 0) + 1
                if destructive_counts[key] >= 2:
                    repeat_doomed.add(key)
                    if self.event_sink is not None:
                        self.event_sink("loop_repeat_detected", {"tool": tool, "args": args_json})
            return result

        def cancelled() -> bool:
            return self.cancellation is not None and self.cancellation.is_cancelled()

        # Resume: if a previously-paused plan remains, auto-execute the rest
        # before asking the model anything (opencode-style continuation).
        if initial_plan and plan_index < len(current_plan):
            history.append(LoopMessage(
                role="assistant",
                content="续跑剩余计划：" + " → ".join(tool for tool, _args in current_plan[plan_index:]),
            ))
            while plan_index < len(current_plan):
                if cancelled():
                    return "cancelled", tuple(history)
                tool, args = current_plan[plan_index]
                plan_index += 1
                self.plan_index = plan_index
                result = run_tool(tool, args)
                if result is None:
                    break
                if not result.ok:
                    break
                if result.requires_approval:
                    self.current_plan = tuple(current_plan)
                    if self.event_sink is not None:
                        self.event_sink("loop_waiting_approval", {"tool": tool})
                    return "waiting_approval", tuple(history)
            if plan_index >= len(current_plan):
                history.append(LoopMessage(
                    role="assistant",
                    content="当前计划已全部执行完。若任务完成请输出 {\"done\": true}；否则输出新的 plan。",
                ))

        for step in range(self.config.max_steps):
            self.steps_taken = step + 1
            if cancelled():
                if self.event_sink is not None:
                    self.event_sink("loop_cancelled", {})
                return "cancelled", tuple(history)
            if self.checkpoint is not None:
                try:
                    self.checkpoint(step, tuple(current_plan), plan_index, self.rolling_summary)
                except Exception:  # noqa: BROAD_EXCEPT_OK - checkpointing is best effort
                    if self.event_sink is not None:
                        self.event_sink("loop_checkpoint_failed", {"step": step})
            if self.event_sink is not None:
                self.event_sink("loop_step", {"step": step})
            if (
                self.config.max_wall_seconds
                and time.monotonic() - run_started > self.config.max_wall_seconds
            ):
                if self.event_sink is not None:
                    self.event_sink("loop_wall_timeout", {"seconds": self.config.max_wall_seconds})
                return "timeout", tuple(history)
            # Token-budgeted compaction: estimate the constant prompt overhead
            # (system/tools/memory/plan), then let the manager select a tail by
            # budget, merge a rolling summary and guarantee the threshold.
            overhead_prompt = render_loop_prompt(
                build_loop_context(
                    system_prompt=system_prompt,
                    history=(),
                    tools=list(self.tools.values()),
                    memory_snapshot=memory_snapshot,
                    recent_artifacts=recent_artifacts,
                    plan_status=plan_status_text(),
                ),
                list(self.tools.values()),
            )
            history, compaction = manager.maybe_compact(
                history,
                _estimate_tokens(overhead_prompt),
            )
            self.rolling_summary = manager.summary
            if compaction is not None and self.event_sink is not None:
                self.event_sink("loop_compacted", compaction.as_payload(len(history)))

            prompt = render_loop_prompt(
                build_loop_context(
                    system_prompt=system_prompt,
                    history=history,
                    tools=list(self.tools.values()),
                    memory_snapshot=memory_snapshot,
                    recent_artifacts=recent_artifacts,
                    plan_status=plan_status_text(),
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
            if cancelled():
                if self.event_sink is not None:
                    self.event_sink("loop_cancelled", {})
                return "cancelled", tuple(history)
            if action.kind == "done":
                if self.event_sink is not None:
                    self.event_sink("loop_done", {"reply": action.thought or ""})
                return "done", tuple(history)

            # Plan-Act: the model plans first; the loop then AUTOMATICALLY
            # executes the plan steps in order (opencode-style), pausing only
            # for approval gates, blocked/failed steps that need a decision,
            # or when the plan is exhausted. The model stays in control via
            # {"next": true}, replacement plans, direct tool calls, or done.
            if action.kind == "plan":
                valid = [(tool, args) for tool, args in action.plan if tool in self.tools]
                if len(valid) < len(action.plan):
                    dropped = sorted({tool for tool, _args in action.plan if tool not in self.tools})
                    if self.event_sink is not None:
                        self.event_sink(
                            "loop_plan_filtered",
                            {"dropped": ",".join(dropped[:8]), "kept": len(valid)},
                        )
                    history.append(LoopMessage(
                        role="assistant",
                        content=(
                            f"计划中有 {len(action.plan) - len(valid)} 个步骤的工具不在当前意图允许范围内"
                            f"（{', '.join(dropped[:5])}），已忽略；请只使用可用工具重新规划。"
                        ),
                    ))
                if len(valid) > self.config.max_plan_steps:
                    if self.event_sink is not None:
                        self.event_sink(
                            "loop_plan_truncated",
                            {"requested": len(valid), "kept": self.config.max_plan_steps},
                        )
                    valid = valid[: self.config.max_plan_steps]
                violations = validate_plan_sequence(
                    valid,
                    existing_world=self.existing_world,
                    existing_cast=self.existing_cast,
                    existing_outline=self.existing_outline,
                )
                if violations:
                    if self.event_sink is not None:
                        self.event_sink("loop_plan_rejected", {"violations": ",".join(violations)})
                    history.append(LoopMessage(
                        role="assistant",
                        content=(
                            "计划顺序或重复步骤不合法（" + ", ".join(violations)
                            + "），已拒绝执行；请按 世界→角色→大纲→章节 重新规划。"
                        ),
                    ))
                    continue
                if not valid:
                    history.append(LoopMessage(
                        role="assistant",
                        content="计划中的工具均不可用，请使用可用工具重新输出 plan。",
                    ))
                    continue
                current_plan[:] = valid
                plan_index = 0
                if self.event_sink is not None:
                    self.event_sink("loop_plan", {"steps": ";".join(tool for tool, _args in valid)})
                history.append(LoopMessage(
                    role="assistant",
                    content="执行计划：" + " → ".join(tool for tool, _args in valid),
                ))
                # Auto-advance: run the plan steps in order; pause only for
                # approval gates, doom-blocked steps, failed steps, or when
                # the plan is exhausted (each step's result is still recorded
                # in history as the model's observation).
                interrupted = False
                while plan_index < len(current_plan):
                    if cancelled():
                        return "cancelled", tuple(history)
                    tool, args = current_plan[plan_index]
                    plan_index += 1
                    self.plan_index = plan_index
                    result = run_tool(tool, args)
                    if result is None:  # doom-blocked: needs a model decision
                        interrupted = True
                        break
                    if not result.ok:  # failed step: needs a model decision
                        interrupted = True
                        break
                    if result.requires_approval:
                        self.current_plan = tuple(current_plan)
                        if self.event_sink is not None:
                            self.event_sink("loop_waiting_approval", {"tool": tool})
                        return "waiting_approval", tuple(history)
                if not interrupted and plan_index >= len(current_plan):
                    history.append(LoopMessage(
                        role="assistant",
                        content="当前计划已全部执行完。若任务完成请输出 {\"done\": true}；否则输出新的 plan。",
                    ))
                continue

            # Advance the active plan: auto-run the remaining steps.
            if action.kind == "next":
                interrupted = False
                while plan_index < len(current_plan):
                    if cancelled():
                        return "cancelled", tuple(history)
                    tool, args = current_plan[plan_index]
                    plan_index += 1
                    self.plan_index = plan_index
                    result = run_tool(tool, args)
                    if result is None:
                        interrupted = True
                        break
                    if not result.ok:
                        interrupted = True
                        break
                    if result.requires_approval:
                        self.current_plan = tuple(current_plan)
                        if self.event_sink is not None:
                            self.event_sink("loop_waiting_approval", {"tool": tool})
                        return "waiting_approval", tuple(history)
                if not interrupted and plan_index >= len(current_plan):
                    history.append(LoopMessage(
                        role="assistant",
                        content="当前计划已全部执行完。若任务完成请输出 {\"done\": true}；否则输出新的 plan。",
                    ))
                continue

            # Direct tool call (deviating from the plan is allowed).
            if action.tool is None or action.tool not in self.tools:
                history.append(LoopMessage(
                    role="assistant",
                    content=f"未知工具 {action.tool}，请从可用工具中选择。",
                ))
                continue
            # Plan-before-act: the first decision of a fresh run must be a plan.
            if not current_plan and not any(m.role == "tool" for m in history):
                history.append(LoopMessage(
                    role="assistant",
                    content="请先输出执行计划（{\"plan\": [{\"tool\": \"…\", \"args\": {…}}, ...]}），系统会按计划依次执行。",
                ))
                continue
            if current_plan and plan_index < len(current_plan) and current_plan[plan_index][0] == action.tool:
                plan_index += 1
            result = run_tool(action.tool, action.args or {})
            if result is None:
                continue
            if result.requires_approval:
                # An approval-gated tool staged its output and paused the run:
                # stop the loop so the user can review; REPAIR resumes from
                # the persisted loop state.
                if self.event_sink is not None:
                    self.event_sink("loop_waiting_approval", {"tool": action.tool})
                return "waiting_approval", tuple(history)
        if self.event_sink is not None:
            self.event_sink("loop_step_limit", {"steps": self.config.max_steps})
        if tool_calls_made >= self.config.max_tool_calls:
            return "tool_budget", tuple(history)
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

