"""Phase-1 plan generator: the model decomposes a brief into an executable
step list, without producing any content.

This is the Plan half of Plan-and-Execute.  ``PlanGenerator`` asks the model
for an ordered list of steps, each declaring the tool to invoke; the system
then executes those steps one by one (the Execute half).  The plan is
validated against the registered tool set and domain ordering rules before
it may become a task graph, so an invalid plan degrades to the deterministic
template instead of running blindly.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from novel_agent.agent.brief import BriefContract, intent_capabilities
from novel_agent.contracts import Provenance, ProvenanceKind, ProviderRequest
from novel_agent.provider_types import ProviderTransport, ProviderTransportError

# The registered creative tools the plan may reference.
AVAILABLE_PLAN_TOOLS: tuple[str, ...] = (
    "world.generate@1.0.0",
    "character.generate@1.0.0",
    "character.revise@1.0.0",
    "outline.generate@1.0.0",
    "chapter.draft@1.0.0",
)

_PLAN_PROMPT: str = """你是 Plan-and-Execute 架构中的规划器。请将用户的创作需求拆解为**可执行的步骤列表**——只做规划，不要生成任何内容。

## 用户需求（已分析）
{requirements}

## 意图
主意图：{intent}
全部交付意图：{intents}

## 已有设定（可直接复用，无需重新生成）
{existing}

## 可用工具（只能从以下列表中选择 tool，不得发明新工具）
{available_tools}

## 步骤约束
1. 步骤必须按执行顺序排列，每步声明：
   - step_id: 从 1 开始的连续编号
   - description: 该步骤要做什么（10-40字）
   - tool: 该步骤调用的工具（必须来自可用工具列表，**必须原样包含版本号，如 world.generate@1.0.0**）
   - goal: 该步骤的达成目标（20-60字）
   - parameters: 该步骤的结构化参数（无参数时为 null）
2. 规划原则：
   - **已有设定中已存在的内容（世界观/角色/大纲），不要再生成对应的步骤**
   - 角色设定（character.generate/character.revise）和大纲（outline.generate）都依赖世界观：
     若项目暂无已有世界观，即使需求只要求设计角色或写大纲，也必须先规划一个 world.generate 步骤
     （系统会校验并自动补全，但请尽量主动包含）
   - 用户要求重做部分角色（如"重新设计林晚""某几个角色不满意"）时，使用 character.revise 步骤，
     并在 parameters 中给出 target_names（要重做的角色名单），该步骤会保留其他角色、只重新生成名单内的角色
   - character.revise 需要已有角色可改（目标角色必须在已有设定中）
   - 世界观生成（world.generate）必须先于角色设定（character.generate/character.revise）和大纲（outline.generate）
   - 大纲（outline.generate）必须先于任何章节正文（chapter.draft）
   - 已有大纲时（无 outline.generate 步骤），章节步骤可以直接从已写章节的下一章开始，
     用 parameters.chapter_number 指定要写的章节号（如写第 2 章填 {{"chapter_number": 2}}），
     未指定时按 1 开始顺序编号
   - 每章正文是一个独立的 chapter.draft 步骤，按故事顺序排列
   - 使用 chapter.draft 步骤的数量 = 要写的章节数
3. 只做规划：步骤列表是唯一的输出，不要生成世界观/角色/大纲/正文内容

请输出**内容字段 JSON**（不要输出任何其他文字）：
{{
  "steps": [
    {{"step_id": 1, "description": "设计世界观", "tool": "world.generate@1.0.0", "goal": "生成符合需求的世界观设定", "parameters": null}},
    {{"step_id": 2, "description": "重新设计林晚", "tool": "character.revise@1.0.0", "goal": "重新设计角色林晚", "parameters": {{"target_names": ["林晚"]}}}}
  ]
}}

只输出 JSON，不要输出 markdown 代码块标记。"""

_MAX_PLAN_STEPS: int = 50


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: int = Field(ge=1)
    description: str = Field(min_length=2, max_length=80)
    tool: str
    goal: str = Field(min_length=2, max_length=120)
    parameters: dict[str, object] | None = None


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    brief_id: str
    steps: tuple[PlanStep, ...]
    created_at: datetime
    provenance: Provenance

    @property
    def chapter_steps(self) -> tuple[PlanStep, ...]:
        return tuple(step for step in self.steps if step.tool == "chapter.draft@1.0.0")


def validate_execution_plan(
    plan: ExecutionPlan,
    *,
    existing_world: bool = False,
    existing_cast: bool = False,
    existing_outline: bool = False,
) -> tuple[str, ...]:
    """Return plan violations; empty means the plan is executable.

    ``existing_world``/``existing_cast``/``existing_outline`` relax the
    world/cast/outline prerequisites when those assets already exist in the
    project and can be reused.
    """
    failures: list[str] = []
    if not plan.steps:
        return ("empty_plan",)
    if len(plan.steps) > _MAX_PLAN_STEPS:
        return ("plan_too_large",)
    registered = set(AVAILABLE_PLAN_TOOLS)
    for step in plan.steps:
        if step.tool not in registered:
            failures.append(f"unknown_tool:{step.tool}")
    first_ok = (
        plan.steps[0].tool in ("world.generate@1.0.0", "character.generate@1.0.0")
        or (existing_world and existing_cast and plan.steps[0].tool == "outline.generate@1.0.0")
        or (existing_world and existing_cast and plan.steps[0].tool == "character.revise@1.0.0")
        or (
            existing_world
            and existing_cast
            and existing_outline
            and plan.steps[0].tool == "chapter.draft@1.0.0"
        )
    )
    if not first_ok:
        failures.append("plan_must_start_with_world_or_cast")
    has_world = any(step.tool == "world.generate@1.0.0" for step in plan.steps)
    has_cast = any(step.tool == "character.generate@1.0.0" for step in plan.steps)
    has_revise = any(step.tool == "character.revise@1.0.0" for step in plan.steps)
    has_outline = any(step.tool == "outline.generate@1.0.0" for step in plan.steps)
    if has_cast and not (has_world or existing_world):
        failures.append("cast_steps_require_world")
    if has_revise:
        if not existing_cast:
            failures.append("revise_steps_require_existing_cast")
        revise_targets = (
            step.parameters.get("target_names") for step in plan.steps if step.tool == "character.revise@1.0.0"
        )
        for target in revise_targets:
            if not isinstance(target, list) or not target or not all(isinstance(item, str) and item for item in target):
                failures.append("revise_steps_require_target_names")
    if has_outline and not (has_world or existing_world):
        failures.append("outline_steps_require_world")
    chapter_steps = plan.chapter_steps
    if chapter_steps and not (has_outline or existing_outline):
        failures.append("chapter_steps_require_outline")
    if chapter_steps:
        first_chapter = next(
            index for index, step in enumerate(plan.steps) if step.tool == "chapter.draft@1.0.0"
        )
        if any(step.tool not in registered for step in plan.steps[:first_chapter]):
            failures.append("invalid_pre_chapter_steps")
        for step in chapter_steps:
            chapter_number = (step.parameters or {}).get("chapter_number")
            if chapter_number is not None and (
                not isinstance(chapter_number, int) or chapter_number < 1
            ):
                failures.append("chapter_step_invalid_number")
    # step_id must be exactly 1..N in order
    expected = tuple(range(1, len(plan.steps) + 1))
    actual = tuple(step.step_id for step in plan.steps)
    if actual != expected:
        failures.append("step_ids_not_sequential")
    return tuple(failures)


@dataclass(frozen=True, slots=True)
class PlanGenerator:
    """Phase-1 planner: model decomposes the brief into an executable step list."""

    transport: ProviderTransport
    now: Callable[[], datetime]
    run_id: str
    model_name: str = "deepseek-v4-pro"
    prompt: str | None = None
    existing_summary: str | None = None
    existing_world: bool | None = None
    existing_cast: bool | None = None
    existing_outline: bool | None = None
    failure_holder: dict[str, str] | None = None
    failure_sink: Callable[[str, str, str, str, str], None] | None = None
    call_event_sink: Callable[[str, str, str], None] | None = None
    plan_sink: Callable[[ExecutionPlan], None] | None = None

    def generate(self, brief: BriefContract) -> ExecutionPlan | None:
        intents = tuple(
            dict.fromkeys(
                ([brief.intent] if brief.intent != "unrecognized" else [])
                + list(brief.intents or ())
            )
        ) or (brief.intent,)
        prompt = (self.prompt or _PLAN_PROMPT).format(
            requirements=brief.raw_requirements,
            intent=brief.intent,
            intents=", ".join(intents),
            existing=self.existing_summary or "（项目暂无已有设定）",
            available_tools="\n".join(f"- {tool}" for tool in AVAILABLE_PLAN_TOOLS),
        )
        timestamp = self.now()
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        request = ProviderRequest(
            provider_call_id=f"{self.run_id}:plan-generation:{prompt_sha256[:12]}",
            run_id=self.run_id,
            task_id=f"{self.run_id}:plan-generation",
            model_name=self.model_name,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            temperature=0.3,
            created_at=timestamp,
            provenance=Provenance(
                source_kind=ProvenanceKind.USER_REQUEST,
                source_id=f"{self.run_id}:plan-generation",
                source_sha256=prompt_sha256,
                observed_at=timestamp,
            ),
        )
        try:
            if self.call_event_sink is not None:
                self.call_event_sink("plan", request.model_name, "started")
            raw = self.transport.generate(request).content
            if self.call_event_sink is not None:
                self.call_event_sink("plan", request.model_name, "finished")
        except ProviderTransportError as error:
            if self.call_event_sink is not None:
                self.call_event_sink("plan", request.model_name, "finished")
            self._record_failure(
                error.kind.value if hasattr(error, "kind") else type(error).__name__,
                request,
                str(error)[:400],
            )
            return None
        parsed = _parse_plan(raw)
        if parsed is None:
            self._record_failure("parse", request, repr(raw[:400]))
            return None
        try:
            plan = ExecutionPlan(
                plan_id=f"{self.run_id}:plan",
                brief_id=brief.brief_id,
                steps=parsed.steps,
                created_at=timestamp,
                provenance=Provenance(
                    source_kind=ProvenanceKind.PROVIDER_RESPONSE,
                    source_id=f"{self.run_id}:plan-generation",
                    source_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    observed_at=timestamp,
                ),
            )
        except ValidationError:
            self._record_failure("schema", request, repr(raw[:400]))
            return None
        existing = self.existing_summary or ""
        existing_world = (
            self.existing_world
            if self.existing_world is not None
            else "世界观《" in existing
        )
        existing_cast = (
            self.existing_cast if self.existing_cast is not None else "已有角色" in existing
        )
        existing_outline = (
            self.existing_outline
            if self.existing_outline is not None
            else "已有大纲" in existing
        )
        plan = _ensure_plan_prerequisites(
            plan,
            existing_world=existing_world,
            existing_cast=existing_cast,
            existing_outline=existing_outline,
            intent=brief.intent,
            intents=intents,
            requirements=brief.raw_requirements,
        )
        violations = validate_execution_plan(
            plan,
            existing_world=existing_world,
            existing_cast=existing_cast,
            existing_outline=existing_outline,
        )
        if violations:
            self._record_failure("validation", request, ",".join(violations))
            return None
        if self.plan_sink is not None:
            self.plan_sink(plan)
        return plan

    def _record_failure(self, kind: str, request: ProviderRequest, detail: str) -> None:
        if self.failure_holder is not None:
            self.failure_holder["reason"] = f"{kind}: {detail}"
        if self.failure_sink is not None:
            self.failure_sink(
                caller="plan",
                model_name=request.model_name,
                failure_kind=kind,
                reason=f"{kind}: {detail[:400]}",
                prompt_sha256=request.prompt_sha256,
            )


def _ensure_plan_prerequisites(
    plan: ExecutionPlan,
    *,
    existing_world: bool,
    existing_cast: bool,
    existing_outline: bool = False,
    intent: str | None = None,
    intents: tuple[str, ...] | None = None,
    requirements: str = "",
) -> ExecutionPlan:
    """Repair a model-produced plan before validation.

    Handles real-world model behaviours:
    - bare tool names (``world.generate`` without ``@1.0.0``): versioned
    - ``character.revise`` without an existing cast: downgraded to generate
    - duplicated cast steps (one per requested character): collapsed
    - missing world step for cast/outline-only requests: prepended
    - intent drift: an outline-only request must not plan chapter drafts,
      and existing world/cast assets must not be regenerated unless the
      request explicitly asks for it
    - continuation drift: a "续写第N章" request must not regenerate assets
      that already exist in the project
    Steps are renumbered; provenance stays that of the model's plan.
    """
    steps = _normalize_steps(plan.steps)
    steps = _collapse_duplicate_cast_steps(steps)
    steps = _downgrade_revise_without_cast(steps, existing_cast=existing_cast)
    steps = _trim_to_intent(
        steps,
        intent=intent,
        intents=intents,
        existing_world=existing_world,
        existing_cast=existing_cast,
        existing_outline=existing_outline,
        requirements=requirements,
    )
    tools = tuple(step.tool for step in steps)
    has_world = "world.generate@1.0.0" in tools
    has_cast = "character.generate@1.0.0" in tools or "character.revise@1.0.0" in tools
    has_outline = "outline.generate@1.0.0" in tools
    needs_world = (has_cast or has_outline) and not has_world and not existing_world
    needs_cast = has_outline and not has_cast and not existing_cast
    if not needs_world and not needs_cast:
        return _renumber(plan, steps)
    prepended: list[PlanStep] = []
    if needs_world:
        prepended.append(
            PlanStep(
                step_id=1,
                description="设计世界观",
                tool="world.generate@1.0.0",
                goal="生成符合需求的世界观设定",
                parameters=None,
            )
        )
    if needs_cast:
        prepended.append(
            PlanStep(
                step_id=len(prepended) + 1,
                description="设计角色",
                tool="character.generate@1.0.0",
                goal="生成符合需求的角色设定",
                parameters=None,
            )
        )
    return _renumber(plan, (*prepended, *steps))


def _trim_to_intent(
    steps: tuple[PlanStep, ...],
    *,
    intent: str | None,
    intents: tuple[str, ...] | None = None,
    existing_world: bool,
    existing_cast: bool,
    existing_outline: bool = False,
    requirements: str = "",
) -> tuple[PlanStep, ...]:
    """Trim steps that contradict the analysed intent.

    ``outline_only`` never drafts chapters.  When existing world/cast assets
    are present, an outline-only request must reuse them (the request says
    "based on existing setting") — the model often over-plans a full
    world/cast/outline pipeline, so those steps are dropped and the graph
    reuses the existing assets.

    A *continuation* request ("续写第二章", "继续写第三章", "接着写下一章")
    must likewise reuse every existing asset: the model routinely plans a
    full world/cast/outline pipeline even though the project already has all
    of them.  Those regeneration steps are dropped so the run only drafts the
    requested chapter(s); the graph then reuses the accepted assets.
    """
    requested = tuple(intents or (() if not intent else (intent,)))
    if len(requested) > 1:
        if "full_novel" in requested:
            return steps
        capability_tools = {
            "world": "world.generate@1.0.0",
            "cast": ("character.generate@1.0.0", "character.revise@1.0.0"),
            "outline": "outline.generate@1.0.0",
            "chapters": "chapter.draft@1.0.0",
            "add_character": ("character.generate@1.0.0", "character.revise@1.0.0"),
        }
        allowed: set[str] = set()
        for capability in intent_capabilities(requested):
            value = capability_tools.get(capability)
            if isinstance(value, tuple):
                allowed.update(value)
            elif value is not None:
                allowed.add(value)
        filtered = tuple(step for step in steps if step.tool in allowed)
        if existing_world and any(step.tool != "world.generate@1.0.0" for step in filtered):
            filtered = tuple(step for step in filtered if step.tool != "world.generate@1.0.0")
        return filtered
    if intent == "world_only":
        steps = tuple(step for step in steps if step.tool == "world.generate@1.0.0")
    elif intent == "characters_only":
        steps = tuple(
            step
            for step in steps
            if step.tool in (
                "world.generate@1.0.0",
                "character.generate@1.0.0",
                "character.revise@1.0.0",
            )
        )
        if existing_world:
            steps = tuple(step for step in steps if step.tool != "world.generate@1.0.0")
    elif intent == "add_character":
        steps = tuple(
            step
            for step in steps
            if step.tool in ("character.generate@1.0.0", "character.revise@1.0.0")
        )
    elif intent == "outline_only":
        steps = tuple(step for step in steps if step.tool != "chapter.draft@1.0.0")
        if existing_world:
            steps = tuple(step for step in steps if step.tool != "world.generate@1.0.0")
        if existing_cast:
            steps = tuple(
                step
                for step in steps
                if step.tool not in ("character.generate@1.0.0", "character.revise@1.0.0")
            )
    elif _is_continuation_request(requirements) and any(
        step.tool == "chapter.draft@1.0.0" for step in steps
    ):
        # Continuation: reuse every existing asset; keep only the chapter
        # drafting steps.  If the model produced no chapter steps at all the
        # plan is left untouched (safety: never empty a plan we can't fix).
        if existing_world:
            steps = tuple(step for step in steps if step.tool != "world.generate@1.0.0")
        if existing_cast:
            steps = tuple(
                step
                for step in steps
                if step.tool not in ("character.generate@1.0.0", "character.revise@1.0.0")
            )
        if existing_outline:
            steps = tuple(step for step in steps if step.tool != "outline.generate@1.0.0")
    return steps


_CONTINUATION_PATTERN = re.compile(
    r"(?:续写|继续|接着写|接着|下一章|后续|第二章|第三章|第四章|第五章|第六章|"
    r"第七\s*章|第八\s*章|第九\s*章|第[一二三四五六七八九十百\d]+章(?:的)?(?:正文|内容)?|"
    r"based on existing|next chapter|continue|chapter\s+\d+)",
    re.IGNORECASE,
)


def _is_continuation_request(requirements: str) -> bool:
    """True when the request continues an existing story rather than starting one."""
    return bool(_CONTINUATION_PATTERN.search(requirements or ""))


def _normalize_steps(steps: tuple[PlanStep, ...]) -> tuple[PlanStep, ...]:
    """Version bare tool names so the versioned registry accepts them."""

    def versioned(tool: str) -> str:
        if "@" in tool:
            return tool
        for registered in AVAILABLE_PLAN_TOOLS:
            if registered.startswith(tool + "@"):
                return registered
        return tool

    return tuple(
        PlanStep(
            step_id=step.step_id,
            description=step.description,
            tool=versioned(step.tool),
            goal=step.goal,
            parameters=step.parameters,
        )
        for step in steps
    )


def _collapse_duplicate_cast_steps(steps: tuple[PlanStep, ...]) -> tuple[PlanStep, ...]:
    """Keep only the first cast-generation step (models emit one per role)."""
    seen_cast = False
    collapsed: list[PlanStep] = []
    for step in steps:
        if step.tool in ("character.generate@1.0.0", "character.revise@1.0.0"):
            if seen_cast:
                continue
            seen_cast = True
        collapsed.append(step)
    return tuple(collapsed)


def _downgrade_revise_without_cast(
    steps: tuple[PlanStep, ...],
    *,
    existing_cast: bool,
) -> tuple[PlanStep, ...]:
    """character.revise needs existing characters; without them it becomes generate."""
    if existing_cast:
        return steps
    return tuple(
        PlanStep(
            step_id=step.step_id,
            description=step.description,
            tool="character.generate@1.0.0" if step.tool == "character.revise@1.0.0" else step.tool,
            goal=step.goal,
            parameters=None if step.tool == "character.revise@1.0.0" else step.parameters,
        )
        for step in steps
    )


def _renumber(plan: ExecutionPlan, steps: tuple[PlanStep, ...]) -> ExecutionPlan:
    renumbered = tuple(
        PlanStep(
            step_id=index,
            description=step.description,
            tool=step.tool,
            goal=step.goal,
            parameters=step.parameters,
        )
        for index, step in enumerate(steps, start=1)
    )
    return ExecutionPlan(
        plan_id=plan.plan_id,
        brief_id=plan.brief_id,
        steps=renumbered,
        created_at=plan.created_at,
        provenance=plan.provenance,
    )


class _PlanContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: tuple[PlanStep, ...]


def _parse_plan(raw_text: str) -> _PlanContent | None:
    try:
        return _PlanContent.model_validate_json(raw_text)
    except ValidationError:
        pass
    stripped = _strip_code_fence(raw_text)
    if stripped != raw_text:
        try:
            return _PlanContent.model_validate_json(stripped)
        except ValidationError:
            pass
    # Tolerate minor JSON damage (trailing prose, unescaped quotes) the same
    # way the brief/creative parsers do; a truncated-but-repairable plan must
    # not degrade the whole run.
    try:
        from json_repair import repair_json

        repaired = str(repair_json(stripped))
    except Exception:  # noqa: BROAD_EXCEPT_OK - best effort
        return None
    if repaired == stripped:
        return None
    try:
        return _PlanContent.model_validate_json(repaired)
    except ValidationError:
        return None


def _strip_code_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


__all__ = (
    "AVAILABLE_PLAN_TOOLS",
    "ExecutionPlan",
    "PlanGenerator",
    "PlanStep",
    "_ensure_plan_prerequisites",
    "validate_execution_plan",
)
