"""Novel-domain tools for the agent loop.

Wraps the existing governed planning/chapter machinery into model-facing
``AgentTool`` entries.  Each tool handler reuses the exact module-level
functions the DAG node executors use (``generate_*_step``, ``commit_*``,
``stage_artifact_for_approval``) so the loop gets the same governance
(approval staging, memory commit, artifact persistence) as the fixed graph.

``NovelToolContext`` carries the loop-time environment (plugin, coordinator,
lease, brief, existing assets, current plan ref).  Unlike the DAG's
``NodeExecutionContext`` it has no task graph: chapter/outline decisions are
driven entirely by the model and the persisted loop state.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from novel_agent.adapters import SchemaFailure
from novel_agent.agent.brief import intent_capabilities
from novel_agent.agent_loop import AgentTool, ToolResult
from novel_agent.contracts import RunStatus, StoryBrief, ToolOutcome
from novel_agent.tool_models import (
    AccessLevel,
    AgentRole,
    ApprovalPolicy,
    ArtifactPolicy,
    RetryPolicy,
    ToolInvocationRequest,
    ToolSpec,
)
from novel_agent.provider_types import ProviderFailureKind, ProviderTransportError
from novel_agent.tool_runtime import ToolCancelledError, check_tool_active
from novel_agent.tool_registry import ToolGateway, ToolRegistry
from novel_agent.planning import (
    generate_cast_step,
    generate_outline_step,
    generate_revise_cast_step,
    generate_world_step,
    select_planning_skills,
)


@dataclass(slots=True)
class NovelToolContext:
    """Environment the loop tools need, without a task graph."""

    plugin: object
    coordinator: object
    lease: object
    brief: StoryBrief
    existing_assets: dict[str, object] | None = None
    plan_ref: tuple[str, int] | None = None  # (plan_id, revision)
    auto_approve: bool = True
    now: Callable[[], object] = field(default_factory=lambda: __import__("datetime").datetime.now)
    # Wiring the DAG executor receives separately (not on the plugin):
    catalog: object | None = None
    gateway: object | None = None
    path_resolver: object | None = None

    # -- helpers shared with the DAG node executors -------------------------

    def _selected_skills(self):
        from novel_agent.planning import select_planning_skills as _sel

        return _sel(self.catalog, self.lease, self.brief)

    def _world(self):
        """Resolve the current world bible from the authoritative store.

        Prefers the newest accepted ``world_bible`` artifact (the canonical
        world), falling back to the existing-assets view when none is stored.
        """
        row = self.coordinator.connection.execute(
            "SELECT raw_bytes FROM artifacts WHERE kind = 'world_bible' AND status = 'accepted' "
            "ORDER BY created_at DESC, revision DESC LIMIT 1"
        ).fetchone()
        if row is not None:
            from novel_agent.contracts import WorldBible

            try:
                return WorldBible.model_validate_json(bytes(row[0]))
            except Exception:  # noqa: BROAD_EXCEPT_OK - fall back to asset view
                pass
        assets = self.existing_assets
        if assets is None or "world_text" not in assets:
            assets = self._refresh_assets()
        if assets is None or "world_text" not in assets:
            return None
        from novel_agent.contracts import WorldBible

        return WorldBible(
            world_id=f"{self.brief.project_id.root}:existing-world",
            brief_id=self.brief.brief_id,
            title=str(assets.get("world_title") or "world"),
            premise=str(assets["world_text"]),
            rules=(),
            constraints=(),
            created_at=self.now(),
            provenance=self.brief.provenance,
        )

    def _cast(self):
        """Resolve the current character set from the authoritative store."""
        row = self.coordinator.connection.execute(
            "SELECT raw_bytes FROM artifacts WHERE kind = 'character_set' AND status = 'accepted' "
            "ORDER BY created_at DESC, revision DESC LIMIT 1"
        ).fetchone()
        if row is not None:
            from novel_agent.contracts import CharacterSet

            try:
                return CharacterSet.model_validate_json(bytes(row[0]))
            except Exception:  # noqa: BROAD_EXCEPT_OK - fall back to asset view
                pass
        assets = self.existing_assets
        if assets is None or "characters" not in assets:
            assets = self._refresh_assets()
        if assets is None or "characters" not in assets:
            return None
        from novel_agent.contracts import CharacterCard, CharacterSet

        cards = tuple(
            CharacterCard(
                character_id=f"{self.brief.project_id.root}:existing-character:{index}",
                name=str(item.get("name", "")),
                role=str(item.get("role_introduction") or ""),
                goal=str(item.get("role_experience") or ""),
                traits=(),
                created_at=self.now(),
                provenance=self.brief.provenance,
            )
            for index, item in enumerate(assets["characters"], start=1)
            if item.get("name")
        )
        if not cards:
            return None
        return CharacterSet(
            character_set_id=f"{self.brief.project_id.root}:existing-cast",
            world_id="",
            characters=cards,
            created_at=self.now(),
            provenance=self.brief.provenance,
        )

    def _refresh_assets(self) -> dict[str, object] | None:
        """Re-detect existing assets from the store (R2.5: keep snapshot fresh)."""
        from novel_agent.domains.novel import detect_existing_assets

        project_root = None
        resolver = getattr(self, "path_resolver", None)
        if resolver is not None:
            project_root = resolver.resolve_project(self.brief.project_id)
        else:
            import sqlite3

            row = self.coordinator.connection.execute("PRAGMA database_list").fetchone()
            if row is not None and row[2]:
                project_root = Path(str(row[2])).resolve().parent
        if project_root is None:
            return None
        try:
            assets = detect_existing_assets(project_root, connection=self.coordinator.connection)
        except Exception:  # noqa: BROAD_EXCEPT_OK - reuse is best-effort
            return None
        self.existing_assets = assets
        return assets

    def _summary(self, text: str, limit: int = 400) -> str:
        return " ".join(text.split())[:limit]

    def _plan_summary(self) -> str:
        plan, _revision = _resolve_plan(self)
        if plan is None:
            return "无大纲"
        return f"共{len(plan.chapters)}章"


def _catalog(ctx: NovelToolContext):
    # The skill catalog is resolved from the wiring (not on the plugin).
    return ctx.catalog


def _commit_or_stage(ctx: NovelToolContext, kind: str, node_id: str, raw_bytes: bytes) -> ToolResult:
    """Approval-mode governance tail: stage the artifact and pause the loop.

    Used by write_* tools when ``auto_approve=False``; the generated artifact
    is written as ``staged`` and the run paused via
    ``stage_artifact_for_approval``.  The AgentLoop detects
    ``requires_approval=True`` and stops, so the user can review; REPAIR
    resumes from the persisted loop state.
    """
    from novel_agent.domains.novel import stage_artifact_for_approval

    artifact_id = stage_artifact_for_approval(
        connection=ctx.coordinator.connection,
        project_id=ctx.brief.project_id.root,
        run_id=ctx.lease.run_id,
        kind=kind,
        raw_bytes=raw_bytes,
        node_id=node_id,
        now=ctx.now(),
    )
    return ToolResult(
        ok=True,
        summary=f"{kind} 已生成，等待确认入库。",
        requires_approval=True,
        staged_artifact_id=artifact_id,
        artifact_ref=artifact_id,
    )


# -- tool factories -----------------------------------------------------------


def _make_write_world(ctx: NovelToolContext) -> AgentTool:
    def execute(args: dict) -> ToolResult:
        brief = ctx.brief
        selected = ctx._selected_skills()
        if isinstance(selected, str):
            return ToolResult(ok=False, error=f"技能选择失败: {selected}")
        result = generate_world_step(ctx.plugin.planning_generator, ctx.gateway, ctx.lease, brief, selected)
        if isinstance(result, SchemaFailure):
            return ToolResult(ok=False, error="世界观 schema 解析失败")
        check_tool_active()
        if not ctx.auto_approve:
            # Approval mode: stage the artifact and pause the loop for review.
            return _commit_or_stage(ctx, "world_bible", "loop:world", result.raw.raw_bytes)
        from novel_agent.domains.novel_commit import commit_world_only

        commit_world_only(
            ctx.coordinator.connection,
            run_id=ctx.lease.run_id,
            project_id=brief.project_id.root,
            brief_id=brief.brief_id,
            world_result=result,
            committed_at=ctx.now(),
            complete=False,
        )
        return ToolResult(
            ok=True,
            summary=f"世界观《{result.value.title}》已生成：{ctx._summary(result.value.premise)}",
            artifact_ref=result.value.world_id,
        )

    return AgentTool(
        name="write_world",
        description="生成或修订故事的世界观设定（背景、规则、约束）。若项目已有世界观，通常无需重复调用。",
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        execute=execute,
        requires_approval=True,
    )


def _make_write_cast(ctx: NovelToolContext) -> AgentTool:
    def execute(args: dict) -> ToolResult:
        brief = ctx.brief
        world = ctx._world()
        if world is None:
            return ToolResult(ok=False, error="缺少世界观：请先调用 write_world 或确认已有设定。")
        selected = ctx._selected_skills()
        if isinstance(selected, str):
            return ToolResult(ok=False, error=f"技能选择失败: {selected}")
        targets = args.get("target_names") if isinstance(args.get("target_names"), list) else None
        existing = ctx._cast()
        if targets:
            if existing is None:
                return ToolResult(ok=False, error="要重做的角色不存在于已有设定。")
            result = generate_revise_cast_step(
                ctx.plugin.planning_generator, ctx.gateway, ctx.lease, brief, world, existing, tuple(targets), selected
            )
        else:
            result = generate_cast_step(ctx.plugin.planning_generator, ctx.gateway, ctx.lease, brief, world, selected)
        if isinstance(result, SchemaFailure):
            return ToolResult(ok=False, error="角色 schema 解析失败")
        check_tool_active()
        if not ctx.auto_approve:
            return _commit_or_stage(ctx, "character_set", "loop:cast", result.raw.raw_bytes)
        from novel_agent.domains.novel_commit import commit_cast_only

        commit_cast_only(
            ctx.coordinator.connection,
            run_id=ctx.lease.run_id,
            project_id=brief.project_id.root,
            brief_id=brief.brief_id,
            cast_result=result,
            committed_at=ctx.now(),
            complete=False,
        )
        names = ", ".join(c.name for c in result.value.characters)
        return ToolResult(ok=True, summary=f"角色设定已生成：{names}", artifact_ref=result.value.character_set_id)

    return AgentTool(
        name="write_cast",
        description="生成或重做角色设定。传 target_names（数组）时只重做指定角色，其余保留。",
        input_schema={
            "type": "object",
            "properties": {
                "target_names": {"type": "array", "items": {"type": "string"}, "description": "要重做的角色名（可选）"}
            },
            "additionalProperties": True,
        },
        execute=execute,
        requires_approval=True,
    )


def _make_write_outline(ctx: NovelToolContext) -> AgentTool:
    def execute(args: dict) -> ToolResult:
        brief = ctx.brief
        world = ctx._world()
        characters = ctx._cast()
        if world is None or characters is None:
            return ToolResult(ok=False, error="缺少世界观或角色：请先调用 write_world / write_cast。")
        selected = ctx._selected_skills()
        if isinstance(selected, str):
            return ToolResult(ok=False, error=f"技能选择失败: {selected}")
        result = generate_outline_step(
            ctx.plugin.planning_generator, ctx.gateway, ctx.lease, brief, world, characters, selected
        )
        if isinstance(result, SchemaFailure):
            return ToolResult(ok=False, error="大纲 schema 解析失败")
        check_tool_active()
        plan = result.value
        if not ctx.auto_approve:
            return _commit_or_stage(ctx, "plan_contract", "loop:outline", result.raw.raw_bytes)
        from novel_agent.domains.novel_commit import commit_outline_only

        commit_outline_only(
            ctx.coordinator.connection,
            run_id=ctx.lease.run_id,
            project_id=brief.project_id.root,
            brief_id=brief.brief_id,
            world_result=_world_adaptation(ctx, world),
            cast_result=_cast_adaptation(ctx, characters),
            outline_result=result,
            committed_at=ctx.now(),
            complete=False,
        )
        ctx.plan_ref = (plan.plan_id, 1)
        chapters = f"共{len(plan.chapters)}章：{ctx._summary(plan.model_dump_json(), 300)}"
        return ToolResult(ok=True, summary=f"故事大纲已生成（{chapters}）", artifact_ref=plan.plan_id)

    return AgentTool(
        name="write_outline",
        description="生成故事大纲（章节规划）。依赖世界观与角色。生成后可通过 write_chapter 逐章撰写。",
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        execute=execute,
        requires_approval=True,
    )


def _make_write_chapter(ctx: NovelToolContext) -> AgentTool:
    def execute(args: dict) -> ToolResult:
        from novel_agent.chapter_executor import ChapterExecutionRequest, ChapterExecutionWorkflow, execute_single_chapter

        chapter_number = args.get("chapter_number")
        if not isinstance(chapter_number, int) or chapter_number < 1:
            return ToolResult(ok=False, error="chapter_number 必须是正整数。")
        plan, plan_revision = _resolve_plan(ctx)
        if plan is None:
            return ToolResult(ok=False, error="缺少大纲：请先调用 write_outline 或确认已有大纲。")
        chapter = next((c for c in plan.chapters if c.chapter_number == chapter_number), None)
        if chapter is None:
            return ToolResult(ok=False, error=f"大纲中没有第 {chapter_number} 章（大纲共 {len(plan.chapters)} 章）。")
        force_rewrite = bool(args.get("force_rewrite", False))
        if not force_rewrite:
            assets = ctx._refresh_assets() or ctx.existing_assets or {}
            written = assets.get("written_chapters") if isinstance(assets, dict) else ()
            if chapter_number in (written or ()):
                existing = _accepted_chapter_artifact_id(ctx, chapter.chapter_plan_id)
                return ToolResult(
                    ok=True,
                    summary=f"第 {chapter_number} 章已存在，跳过重复生成。",
                    artifact_ref=existing,
                )
        # A continuation run reuses a previous run's outline; the chapter staging
        # authority requires the plan to be an accepted artifact of THIS run, so
        # mirror the reused plan under the current run before writing chapters.
        plan_revision = _mirror_plan_to_run(ctx, plan, plan_revision)
        next_chapter = next(
            (c for c in plan.chapters if c.chapter_number == chapter_number + 1), None
        )
        from novel_agent.domains.novel import _story_brief

        brief = _story_brief_from_ctx(ctx)
        world = ctx._world()
        world_rules = (
            tuple(world.rules) + tuple(world.constraints) if world is not None else ()
        )
        request = ChapterExecutionRequest(
            project_id=ctx.brief.project_id,
            plan=plan,
            plan_revision=plan_revision,
            target_chinese_characters=int(getattr(ctx.brief, "target_words", None) or 3000),
            genre=brief.genre,
            style_guidance=tuple(getattr(ctx.brief, "style_guidance", ()) or ()),
            world_rules=world_rules,
            acceptance_criteria=tuple(getattr(ctx.brief, "acceptance_criteria", ()) or ()),
            activate=not force_rewrite,
            auto_approve=ctx.auto_approve,
        )
        workflow = ChapterExecutionWorkflow(
            ctx.coordinator,
            ctx.coordinator.context_builder(),
            ctx.catalog,
            ctx.plugin.chapter_generator,
            ctx.plugin.extractor,
            ctx.plugin.auditor,
            _current_reviewer(ctx),
            ctx.path_resolver,
            tool_gateway=ctx.gateway,
            stage_event_sink=None,
        )
        task_id = f"{ctx.lease.run_id}:chapter:{chapter.chapter_plan_id}"
        # Create/activate the chapter task row, mirroring what the DAG node
        # executor does before execute_single_chapter.
        timestamp = ctx.now().isoformat()
        ctx.coordinator.connection.execute(
            "INSERT INTO tasks(task_id, run_id, task_type, state, idempotency_key, created_at, updated_at) "
            "VALUES (?, ?, 'chapter', 'pending', ?, ?, ?) "
            "ON CONFLICT(task_id) DO NOTHING",
            (task_id, ctx.lease.run_id, task_id, timestamp, timestamp),
        )
        ctx.coordinator.activate_chapter_task(ctx.lease, task_id)
        result = execute_single_chapter(workflow, ctx.lease, request, chapter, next_chapter, task_id)
        match result:
            case _:
                from novel_agent.chapter_executor import (
                    ChapterExecutionCancelled,
                    ChapterExecutionSucceeded,
                    ChapterExecutionWaitingReview,
                )

                if isinstance(result, ChapterExecutionSucceeded):
                    artifact_ref = _accepted_chapter_artifact_id(ctx, chapter.chapter_plan_id)
                    return ToolResult(
                        ok=True,
                        summary=f"第 {chapter_number} 章已撰写完成并入库。",
                        artifact_ref=artifact_ref,
                    )
                if isinstance(result, ChapterExecutionWaitingReview):
                    return ToolResult(ok=True, summary=f"第 {chapter_number} 章已撰写，等待确认入库。", requires_approval=True)
                if isinstance(result, ChapterExecutionCancelled):
                    return ToolResult(ok=False, error="章节撰写被取消。")
        return ToolResult(ok=False, error=f"章节撰写未完成：{result}")

    return AgentTool(
        name="write_chapter",
        description="撰写指定章节的正文（依据当前大纲）。参数：chapter_number（必填）、force_rewrite（重写已发布章节，可选）。",
        input_schema={
            "type": "object",
            "properties": {
                "chapter_number": {"type": "integer", "minimum": 1},
                "force_rewrite": {"type": "boolean"},
            },
            "required": ["chapter_number"],
        },
        execute=execute,
        requires_approval=True,
    )


def _make_review_chapter(ctx: NovelToolContext) -> AgentTool:
    def execute(args: dict) -> ToolResult:
        # Status only: the real six-dimension review runs inside chapter
        # execution's quality gate.  Refresh the asset view so a chapter just
        # written in this run is visible immediately.
        assets = ctx._refresh_assets() or ctx.existing_assets or {}
        written = assets.get("written_chapters") if isinstance(assets, dict) else ()
        return ToolResult(
            ok=True,
            summary=f"已写章节：{sorted(written) if written else '无'}；大纲 {ctx._plan_summary()}。",
        )

    return AgentTool(
        name="review_chapter",
        description="查看当前已写章节与大纲状态，辅助判断下一步。",
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        execute=execute,
        requires_approval=False,
        supports_parallel=True,
    )


def _make_mem_query(ctx: NovelToolContext) -> AgentTool:
    def execute(args: dict) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(ok=False, error="query 必须是非空字符串。")
        from novel_agent.context import memory_query

        hits = memory_query(
            ctx.coordinator.connection,
            project_id=ctx.brief.project_id.root,
            query_terms=(query.strip(),),
            limit=5,
        )
        if not hits:
            return ToolResult(ok=True, summary=f"未找到与「{query}」相关的已确认设定。")
        text = "；".join(f"{h.subject_id}: {h.content[:120]}" for h in hits)
        return ToolResult(ok=True, summary=f"相关设定：{text}")

    return AgentTool(
        name="mem_query",
        description="查询已确认的故事设定（人物/事件/规则）。按相关性返回，用于回忆前文。",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        execute=execute,
        requires_approval=False,
        supports_parallel=True,
    )


class LoopToolInput(BaseModel):
    """Gateway envelope for one model-facing tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    args: dict = Field(default_factory=dict)


class LoopToolOutput(BaseModel):
    """Typed result persisted in the tool-call ledger and fed back to the model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    summary: str = ""
    artifact_ref: str | None = None
    error: str | None = None
    requires_approval: bool = False
    staged_artifact_id: str | None = None
    retryable: bool = False


class _WriteWorldArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

class _WriteCastArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    target_names: list[str] | None = None

class _AddCharacterArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = Field(min_length=1)


class _WriteOutlineArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

class _WriteChapterArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    chapter_number: int = Field(ge=1)
    force_rewrite: bool = False

class _ReviewChapterArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

class _MemQueryArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str = Field(min_length=1)

class _PlanQueryArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

class _ReadChapterArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    chapter_number: int = Field(ge=1)


@dataclass(frozen=True, slots=True)
class _ToolGovernance:
    tool_id: str
    role: AgentRole
    access: AccessLevel
    approval: ApprovalPolicy
    input_model: type[BaseModel]
    timeout_seconds: float = 600.0


_TOOL_GOVERNANCE: dict[str, _ToolGovernance] = {
    "write_world": _ToolGovernance("novel.write-world", AgentRole.STORY_ARCHITECT, AccessLevel.STAGE_WRITE, ApprovalPolicy.COORDINATOR, _WriteWorldArgs),
    "write_cast": _ToolGovernance("novel.write-cast", AgentRole.CAST_DESIGNER, AccessLevel.STAGE_WRITE, ApprovalPolicy.COORDINATOR, _WriteCastArgs),
    "add_character": _ToolGovernance("novel.add-character", AgentRole.CAST_DESIGNER, AccessLevel.STAGE_WRITE, ApprovalPolicy.COORDINATOR, _AddCharacterArgs),
    "write_outline": _ToolGovernance("novel.write-outline", AgentRole.OUTLINE_PLANNER, AccessLevel.STAGE_WRITE, ApprovalPolicy.COORDINATOR, _WriteOutlineArgs),
    "write_chapter": _ToolGovernance("novel.write-chapter", AgentRole.CHAPTER_EXECUTOR, AccessLevel.STAGE_WRITE, ApprovalPolicy.COORDINATOR, _WriteChapterArgs),
    "review_chapter": _ToolGovernance("novel.review-chapter", AgentRole.CONTINUITY_AUDITOR, AccessLevel.READ, ApprovalPolicy.NONE, _ReviewChapterArgs),
    "mem_query": _ToolGovernance("novel.mem-query", AgentRole.MEMORY_EXTRACTOR, AccessLevel.READ, ApprovalPolicy.NONE, _MemQueryArgs),
    "plan_query": _ToolGovernance("novel.plan-query", AgentRole.OUTLINE_PLANNER, AccessLevel.READ, ApprovalPolicy.NONE, _PlanQueryArgs),
    "read_chapter": _ToolGovernance("novel.read-chapter", AgentRole.CONTINUITY_AUDITOR, AccessLevel.READ, ApprovalPolicy.NONE, _ReadChapterArgs),
}


def _as_output(result: ToolResult) -> LoopToolOutput:
    return LoopToolOutput(
        ok=result.ok,
        summary=result.summary,
        artifact_ref=result.artifact_ref,
        error=result.error,
        requires_approval=result.requires_approval,
        staged_artifact_id=result.staged_artifact_id,
        retryable=result.retryable,
    )


def _operation_key(tool_name: str, args: dict) -> str:
    payload = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{tool_name}:{payload}".encode()).hexdigest()


def _cached_operation(
    ctx: NovelToolContext, tool_name: str, args: dict
) -> ToolResult | None:
    try:
        row = ctx.coordinator.connection.execute(
            "SELECT result_json FROM loop_operations WHERE run_id = ? AND operation_key = ? "
            "AND state = 'succeeded'",
            (ctx.lease.run_id, _operation_key(tool_name, args)),
        ).fetchone()
    except Exception:  # noqa: BROAD_EXCEPT_OK - old databases without v9 keep working
        return None
    if row is None or row[0] is None:
        return None
    try:
        payload = LoopToolOutput.model_validate_json(str(row[0]))
    except Exception:  # noqa: BROAD_EXCEPT_OK - corrupt cache is ignored
        return None
    return ToolResult(
        ok=payload.ok,
        summary=payload.summary,
        artifact_ref=payload.artifact_ref,
        error=payload.error,
        requires_approval=payload.requires_approval,
        staged_artifact_id=payload.staged_artifact_id,
        retryable=payload.retryable,
    )


def _remember_operation(
    ctx: NovelToolContext, tool_name: str, args: dict, result: ToolResult
) -> None:
    if not result.ok or result.requires_approval:
        return
    payload = LoopToolOutput(
        ok=result.ok,
        summary=result.summary,
        artifact_ref=result.artifact_ref,
        error=result.error,
        requires_approval=result.requires_approval,
        staged_artifact_id=result.staged_artifact_id,
        retryable=result.retryable,
    ).model_dump_json()
    try:
        ctx.coordinator.connection.execute(
            "INSERT INTO loop_operations(run_id, operation_key, tool_name, state, result_json, created_at) "
            "VALUES (?, ?, ?, 'succeeded', ?, ?) "
            "ON CONFLICT(run_id, operation_key) DO UPDATE SET state='succeeded', "
            "result_json=excluded.result_json, created_at=excluded.created_at",
            (
                ctx.lease.run_id,
                _operation_key(tool_name, args),
                tool_name,
                payload,
                ctx.now().isoformat(),
            ),
        )
    except Exception:  # noqa: BROAD_EXCEPT_OK - journal failure must not fail the tool
        pass


def _governed_tool(
    tool: AgentTool,
    governance: _ToolGovernance,
    gateway: ToolGateway,
    ctx: NovelToolContext,
) -> AgentTool:
    """Wrap a domain handler in the shared gateway (auth/idempotency/audit).

    The handler still performs the domain work; the gateway owns argument
    validation, role/state/skill authorization, the tool-call ledger and the
    persisted request/response hashes.  A unique idempotency nonce keeps every
    attempt auditable instead of replaying a stale failure.
    """

    tool_ref = f"{governance.tool_id}@1.0.0"

    def execute(args: dict) -> ToolResult:
        if governance.access is AccessLevel.STAGE_WRITE:
            cached = _cached_operation(ctx, tool.name, args)
            if cached is not None:
                return cached
        nonce = uuid.uuid4().hex
        request = ToolInvocationRequest(
            invocation_id=f"{ctx.lease.run_id}:loop:{tool.name}:{nonce}",
            run_id=ctx.lease.run_id,
            tool_ref=tool_ref,
            idempotency_key=f"{ctx.lease.run_id}:loop:{tool.name}:{nonce}",
            actor_role=governance.role,
            skill_ref="loop@1.0.0",
            run_status=RunStatus.RUNNING,
            input_data=LoopToolInput(args=args),
            # The novel tools stage artifacts for post-execution approval; the
            # gateway approval flag is therefore granted here and the actual
            # pause is signalled by ToolResult.requires_approval.
            approval_granted=True,
        )
        check_tool_active()
        result = gateway.invoke(request)
        if governance.access is AccessLevel.READ:
            check_tool_active()
        if result.outcome is ToolOutcome.SUCCEEDED and isinstance(result.data, LoopToolOutput):
            mapped = ToolResult(
                ok=result.data.ok,
                summary=result.data.summary,
                artifact_ref=result.data.artifact_ref,
                error=result.data.error,
                requires_approval=result.data.requires_approval,
                staged_artifact_id=result.data.staged_artifact_id,
                retryable=result.data.retryable,
            )
            if governance.access is AccessLevel.STAGE_WRITE:
                _remember_operation(ctx, tool.name, args, mapped)
            return mapped
        return ToolResult(ok=False, error=f"工具调用被拒绝/失败：{result.reason or 'TOOL_FAILED'}")

    return replace(
        tool,
        execute=execute,
        input_model=governance.input_model,
        timeout_seconds=30.0 if governance.access is AccessLevel.READ else 600.0,
        max_attempts=2 if governance.access is AccessLevel.STAGE_WRITE else 1,
        retry_delay_seconds=0.5,
    )


def build_novel_loop_tools(ctx: NovelToolContext) -> dict[str, AgentTool]:
    """Assemble the model-facing tool set, governed by ToolGateway."""
    raw_tools = [
        _make_write_world(ctx),
        _make_write_cast(ctx),
        _make_add_character(ctx),
        _make_write_outline(ctx),
        _make_write_chapter(ctx),
        _make_review_chapter(ctx),
        _make_mem_query(ctx),
        _make_plan_query(ctx),
        _make_read_chapter(ctx),
    ]
    registry = ToolRegistry()
    for tool in raw_tools:
        governance = _TOOL_GOVERNANCE[tool.name]

        def handler(payload: LoopToolInput, _tool: AgentTool = tool) -> LoopToolOutput:
            try:
                return _as_output(_tool.execute(payload.args))
            except ToolCancelledError:
                raise
            except Exception as error:  # domain failures are results, not gateway crashes
                retryable = isinstance(error, ProviderTransportError) and error.kind in {
                    ProviderFailureKind.TRANSIENT,
                    ProviderFailureKind.TIMEOUT,
                }
                return LoopToolOutput(
                    ok=False,
                    error=f"工具异常 {type(error).__name__}: {str(error)[:400]}",
                    retryable=retryable,
                )

        registry.register(
            ToolSpec(
                tool_id=governance.tool_id,
                version="1.0.0",
                input_model=LoopToolInput,
                output_model=LoopToolOutput,
                allowed_roles=frozenset((governance.role,)),
                allowed_skill_ids=frozenset(("loop@1.0.0",)),
                allowed_run_states=frozenset((RunStatus.RUNNING,)),
                access_level=governance.access,
                timeout_seconds=30.0 if governance.access is AccessLevel.READ else 600.0,
                retry_policy=RetryPolicy(max_attempts=1),
                idempotent=False,
                audit_calls=True,
                approval_policy=governance.approval,
                artifact_policy=(
                    ArtifactPolicy.NONE
                    if governance.access is AccessLevel.READ
                    else ArtifactPolicy.STAGE_ONLY
                ),
            ),
            handler,
        )
    connection = getattr(ctx.coordinator, "connection", None)
    gateway = ToolGateway(registry, connection)
    return {
        tool.name: _governed_tool(tool, _TOOL_GOVERNANCE[tool.name], gateway, ctx)
        for tool in raw_tools
    }


# -- helpers ------------------------------------------------------------------


def _resolve_plan(ctx: NovelToolContext):
    """Resolve the current plan: prefer ctx.plan_ref, else latest accepted plan."""
    if ctx.plan_ref is not None:
        plan_id, revision = ctx.plan_ref
        row = ctx.coordinator.connection.execute(
            "SELECT raw_bytes FROM artifacts WHERE artifact_id = ? AND revision = ? AND status = 'accepted'",
            (plan_id, revision),
        ).fetchone()
        if row is not None:
            from novel_agent.contracts import PlanContract

            return PlanContract.model_validate_json(bytes(row[0])), revision
    row = ctx.coordinator.connection.execute(
        "SELECT raw_bytes, revision FROM artifacts WHERE kind = 'plan_contract' AND status = 'accepted' "
        "AND is_current = 1 ORDER BY created_at DESC, revision DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None, 0
    from novel_agent.contracts import PlanContract

    return PlanContract.model_validate_json(bytes(row[0])), int(row[1])


def _mirror_plan_to_run(ctx: NovelToolContext, plan, plan_revision: int) -> int:
    """Mirror a reused plan contract into the current run as an accepted artifact.

    The chapter staging authority requires the plan to be an accepted artifact
    of the leased run (``_require_accepted_plan``).  When agent_loop reuses a
    previous run's outline (write_chapter with an existing plan), the plan is
    copied under the current run — same bytes, same artifact id, new revision —
    mirroring the DAG's ``_mirror_reused_plan`` behaviour.  Without this, a
    continuation run fails with ``ChapterPersistenceError`` on the first
    write_chapter, which the gateway surfaces as HANDLER_FAILED.
    """
    import hashlib as _hashlib

    raw_bytes = plan.model_dump_json().encode()
    connection = ctx.coordinator.connection
    row = connection.execute(
        "SELECT artifact_id, revision FROM artifacts WHERE artifact_id = ? AND run_id = ? "
        "AND kind = 'plan_contract' AND status = 'accepted' AND is_current = 1",
        (plan.plan_id, ctx.lease.run_id),
    ).fetchone()
    if row is not None:
        return int(row[1])
    next_revision = int(
        connection.execute(
            "SELECT COALESCE(MAX(revision), 0) + 1 FROM artifacts WHERE artifact_id = ?",
            (plan.plan_id,),
        ).fetchone()[0]
    )
    connection.execute(
        "UPDATE artifacts SET is_current = 0 WHERE artifact_id = ? AND is_current = 1",
        (plan.plan_id,),
    )
    connection.execute(
        "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
        "schema_version, is_current, created_at) VALUES (?, ?, ?, ?, 'plan_contract', 'accepted', ?, ?, 1, 1, ?)",
        (
            plan.plan_id,
            next_revision,
            ctx.brief.project_id.root,
            ctx.lease.run_id,
            raw_bytes,
            _hashlib.sha256(raw_bytes).hexdigest(),
            ctx.now().isoformat(),
        ),
    )
    return next_revision


def _story_brief_from_ctx(ctx: NovelToolContext) -> StoryBrief:
    requirements = getattr(ctx.brief, "raw_requirements", None) or getattr(ctx.brief, "user_requirements", "")
    return StoryBrief(
        brief_id=ctx.brief.brief_id,
        project_id=ctx.brief.project_id,
        user_requirements=requirements,
        genre="fiction",
        target_chapter_count=ctx.brief.target_chapter_count or 10,
        target_words=getattr(ctx.brief, "target_words", None),
        created_at=ctx.now(),
        provenance=ctx.brief.provenance,
    )


def _current_reviewer(ctx: NovelToolContext) -> object:
    holder = getattr(ctx.plugin, "reviewer_holder", None)
    if holder is not None and "current" in holder:
        return holder["current"]
    from novel_agent.offline_fake import OfflineQualityReviewer

    return OfflineQualityReviewer()


def _world_adaptation(ctx: NovelToolContext, world):
    from novel_agent.adapters import CreativeArtifactKind, CreativeParseRoute, RawCreativeArtifact, WorldAdaptation

    payload = world.model_dump_json()
    return WorldAdaptation(
        raw=RawCreativeArtifact(
            kind=CreativeArtifactKind.WORLD,
            raw_bytes=payload.encode(),
            content_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            adapter_version="loop:v1",
        ),
        value=world,
        parse_route=CreativeParseRoute.STRICT_JSON,
    )


def _cast_adaptation(ctx: NovelToolContext, characters):
    from novel_agent.adapters import (
        CharacterAdaptation,
        CreativeArtifactKind,
        CreativeParseRoute,
        RawCreativeArtifact,
    )

    payload = characters.model_dump_json()
    return CharacterAdaptation(
        raw=RawCreativeArtifact(
            kind=CreativeArtifactKind.CHARACTERS,
            raw_bytes=payload.encode(),
            content_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            adapter_version="loop:v1",
        ),
        value=characters,
        parse_route=CreativeParseRoute.STRICT_JSON,
    )


_READ_ONLY_LOOP_TOOLS: frozenset[str] = frozenset(
    ("mem_query", "plan_query", "review_chapter", "read_chapter")
)


def filter_tools_by_intent(
    tools: dict[str, AgentTool],
    intent: str | None,
    *,
    intents: tuple[str, ...] | None = None,
    existing_assets: dict[str, object] | None = None,
) -> dict[str, AgentTool]:
    """Expose only the write tools the requested intents permit.

    Read-only inspection tools stay available so the model can orient itself.
    Composite intents are unioned in canonical stage order; sub-intents may
    create prerequisites (world, and cast for an outline) when missing, but
    never drift into a later stage.
    """
    requested = tuple(
        dict.fromkeys(([intent] if intent else []) + list(intents or ()))
    )
    requested = tuple(item for item in requested if item != "unrecognized") or requested
    if not requested:
        return tools
        return tools
    if "full_novel" in requested:
        return tools
    capabilities = set(intent_capabilities(requested))
    assets = existing_assets or {}
    has_world = bool(assets.get("world_text") or assets.get("world_title"))
    has_cast = bool(assets.get("characters"))
    allowed: set[str] = set(_READ_ONLY_LOOP_TOOLS)
    if "world" in capabilities and not has_world:
        allowed.add("write_world")
    if "cast" in capabilities and (
        not has_cast or any(item in ("characters_only", "add_character") for item in requested)
    ):
        allowed.add("write_cast")
    if "outline" in capabilities:
        allowed.add("write_outline")
    if "chapters" in capabilities:
        allowed.add("write_chapter")
    if "add_character" in capabilities:
        allowed.add("add_character")
    return {name: tool for name, tool in tools.items() if name in allowed}


def _make_add_character(ctx: NovelToolContext) -> AgentTool:
    def execute(args: dict) -> ToolResult:
        name = args.get("name")
        if not isinstance(name, str) or not name.strip():
            return ToolResult(ok=False, error="name 必须是非空字符串。")
        name = name.strip()
        existing = ctx._cast()
        if existing is not None and any(card.name == name for card in existing.characters):
            return ToolResult(
                ok=False,
                error=f"角色「{name}」已存在；如需重做请调用 write_cast 并传 target_names。",
            )
        # The cast-revision path keeps all non-target characters and generates
        # the requested card, which is exactly the "append one character" case.
        return _make_write_cast(ctx).execute({"target_names": [name]})

    return AgentTool(
        name="add_character",
        description="为已有故事新增一个角色（传 name）；保留原有角色，不重生成整份 cast。",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "新角色名"}},
            "required": ["name"],
        },
        execute=execute,
        requires_approval=True,
    )


def _accepted_chapter_row(ctx: NovelToolContext, chapter_plan_id: str):
    """Find the accepted chapter artifact for a chapter plan.

    Chapter staging derives the artifact id deterministically from
    run_id + chapter_plan_id; older projects may still store the chapter
    plan id directly, so that remains a fallback.
    """
    connection = ctx.coordinator.connection
    seed = f"{ctx.lease.run_id}:{chapter_plan_id}"
    artifact_id = "chapter-" + hashlib.sha256(seed.encode()).hexdigest()[:24]
    row = connection.execute(
        "SELECT artifact_id, raw_bytes FROM artifacts WHERE artifact_id = ? "
        "AND kind = 'chapter' AND status = 'accepted' ORDER BY revision DESC LIMIT 1",
        (artifact_id,),
    ).fetchone()
    if row is not None:
        return row
    return connection.execute(
        "SELECT artifact_id, raw_bytes FROM artifacts WHERE artifact_id = ? "
        "AND kind = 'chapter' AND status = 'accepted' ORDER BY revision DESC LIMIT 1",
        (chapter_plan_id,),
    ).fetchone()


def _accepted_chapter_artifact_id(ctx: NovelToolContext, chapter_plan_id: str) -> str | None:
    row = _accepted_chapter_row(ctx, chapter_plan_id)
    return str(row[0]) if row is not None else None


def _make_plan_query(ctx: NovelToolContext) -> AgentTool:
    def execute(args: dict) -> ToolResult:
        plan, revision = _resolve_plan(ctx)
        if plan is None:
            return ToolResult(ok=False, error="当前没有可用大纲。")
        lines = [
            f"第{chapter.chapter_number}章《{chapter.title}》：{chapter.goal}"
            for chapter in sorted(plan.chapters, key=lambda item: item.chapter_number)
        ]
        return ToolResult(
            ok=True,
            summary=f"当前大纲（revision {revision}，共{len(plan.chapters)}章）：\n" + "\n".join(lines),
            artifact_ref=plan.plan_id,
        )

    return AgentTool(
        name="plan_query",
        description="查看当前大纲的章节编号、标题与目标，写章节前用来确认章节号。",
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        execute=execute,
        supports_parallel=True,
    )


def _make_read_chapter(ctx: NovelToolContext) -> AgentTool:
    def execute(args: dict) -> ToolResult:
        chapter_number = args.get("chapter_number")
        if not isinstance(chapter_number, int) or chapter_number < 1:
            return ToolResult(ok=False, error="chapter_number 必须是正整数。")
        plan, _revision = _resolve_plan(ctx)
        if plan is None:
            return ToolResult(ok=False, error="当前没有可用大纲。")
        chapter = next((item for item in plan.chapters if item.chapter_number == chapter_number), None)
        if chapter is None:
            return ToolResult(ok=False, error=f"大纲中没有第 {chapter_number} 章。")
        row = _accepted_chapter_row(ctx, chapter.chapter_plan_id)
        if row is None:
            return ToolResult(ok=False, error=f"第 {chapter_number} 章尚未有已接受的正文。")
        text = bytes(row[1]).decode("utf-8", errors="replace")
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            for key in ("passage", "text", "content", "summary"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    text = value
                    break
        return ToolResult(
            ok=True,
            summary=f"第 {chapter_number} 章正文（节选）：\n{text[:3000]}",
            artifact_ref=str(row[0]),
        )

    return AgentTool(
        name="read_chapter",
        description="读取已接受章节的正文节选（按章节号），用于续写或检查连续性。",
        input_schema={
            "type": "object",
            "properties": {"chapter_number": {"type": "integer", "minimum": 1}},
            "required": ["chapter_number"],
        },
        execute=execute,
        supports_parallel=True,
    )


__all__ = ("NovelToolContext", "build_novel_loop_tools", "filter_tools_by_intent")
