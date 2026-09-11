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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from novel_agent.adapters import SchemaFailure
from novel_agent.agent_loop import AgentTool, ToolResult
from novel_agent.contracts import StoryBrief
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
        next_chapter = next(
            (c for c in plan.chapters if c.chapter_number == chapter_number + 1), None
        )
        from novel_agent.domains.novel import _story_brief

        brief = _story_brief_from_ctx(ctx)
        request = ChapterExecutionRequest(
            project_id=ctx.brief.project_id,
            plan=plan,
            plan_revision=plan_revision,
            target_chinese_characters=3000,
            genre=brief.genre,
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
                    return ToolResult(ok=True, summary=f"第 {chapter_number} 章已撰写完成并入库。", artifact_ref=chapter.chapter_plan_id)
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
        # Light-weight: report current written chapters and the plan; the real
        # six-dimension review runs inside chapter execution's quality gate.
        written = ctx.existing_assets.get("written_chapters") if ctx.existing_assets else ()
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
        return ToolResult(ok=True, summary=f"相关设定：{text}", artifact_ref=query)

    return AgentTool(
        name="mem_query",
        description="查询已确认的故事设定（人物/事件/规则）。按相关性返回，用于回忆前文。",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        execute=execute,
        requires_approval=False,
        supports_parallel=True,
    )


def build_novel_loop_tools(ctx: NovelToolContext) -> dict[str, AgentTool]:
    """Assemble the model-facing tool set for the novel agent loop."""
    tools = [
        _make_write_world(ctx),
        _make_write_cast(ctx),
        _make_write_outline(ctx),
        _make_write_chapter(ctx),
        _make_review_chapter(ctx),
        _make_mem_query(ctx),
    ]
    return {tool.name: tool for tool in tools}


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


def _story_brief_from_ctx(ctx: NovelToolContext) -> StoryBrief:
    requirements = getattr(ctx.brief, "raw_requirements", None) or getattr(ctx.brief, "user_requirements", "")
    return StoryBrief(
        brief_id=ctx.brief.brief_id,
        project_id=ctx.brief.project_id,
        user_requirements=requirements,
        genre="fiction",
        target_chapter_count=ctx.brief.target_chapter_count or 10,
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


__all__ = ("NovelToolContext", "build_novel_loop_tools")
