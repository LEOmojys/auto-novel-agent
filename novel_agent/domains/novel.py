"""Novel domain plugin: the plan-and-execute graph for fiction writing.

The novel domain turns a structured brief into a validated task graph:
world generation -> cast generation -> outline generation -> one chapter
node per chapter.  Node executors reuse the exact planning/chapter machinery
(governed tool gateway, skill selection, atomic plan commit, gated single
chapter execution) so the generic executor adds orchestration without
changing any governed behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import assert_never

from pydantic import ValidationError

from novel_agent.adapters import (
    CharacterAdaptation,
    CreativeArtifactKind,
    CreativeParseRoute,
    RawCreativeArtifact,
    SchemaFailure,
    WorldAdaptation,
)
from novel_agent.agent.brief import BriefContract
from novel_agent.agent.domain_plugin import (
    NodeExecutionContext,
    NodeExecutionResult,
    NodeExecutor,
    NodeOutput,
)
from novel_agent.agent.task_graph import TaskGraph, TaskGraphNode
from novel_agent.chapter_executor import (
    ChapterExecutionCancelled,
    ChapterExecutionRequest,
    ChapterExecutionSucceeded,
    ChapterExecutionWaitingReview,
    ChapterExecutionWorkflow,
    ChapterGenerator,
    ContinuityAuditor,
    MemoryExtractor,
    execute_single_chapter,
)
from novel_agent.contracts import (
    CharacterCard,
    CharacterSet,
    PlanApprovalStatus,
    PlanContract,
    Provenance,
    ProvenanceKind,
    RunStatus,
    StoryBrief,
    WorldBible,
)
from novel_agent.offline_fake import OfflineContinuityAuditor, OfflineQualityReviewer
from novel_agent.pathing import ProjectId, ProjectPathResolver
from novel_agent.planning import (
    PlanningBlocked,
    PlanningSucceeded,
    PlanningGenerator,
    commit_plan,
    generate_cast_step,
    generate_outline_step,
    generate_revise_cast_step,
    generate_world_step,
    select_planning_skills,
)
from novel_agent.tool_models import AgentRole

_DEFAULT_TARGET_WORDS: int = 3_000


@dataclass(frozen=True, slots=True)
class NovelDomainPlugin:
    """Domain plugin wiring the novel domain into the generic agent core.

    ``reviewer_holder`` is a mutable per-run slot holding the current quality
    reviewer.  It starts as the offline acceptor and is swapped for a live
    provider-backed reviewer once the cast node has produced character
    fixtures (when ``real_reviewer_factory`` is provided).  The tool gateway
    registers ``quality.evaluate`` against the same holder, so the audited
    tool result always matches the reviewer the workflow actually used.
    """

    planning_generator: PlanningGenerator
    chapter_generator: ChapterGenerator
    extractor: MemoryExtractor
    auditor: ContinuityAuditor = OfflineContinuityAuditor()
    reviewer_holder: dict[str, object] | None = None
    real_reviewer_factory: Callable[[NodeExecutionContext], object] | None = None
    memory_names_holder: dict[str, object] | None = None
    plan_generator: object | None = None
    plan_info_holder: dict[str, object] | None = None
    existing_assets: dict[str, object] | None = None
    auto_approve: bool = True
    domain: str = "novel"

    def build_graph(
        self,
        brief: BriefContract,
        *,
        run_id: str,
        project_id: ProjectId,
        now: Callable[[], datetime],
        approved_plan: object | None = None,
    ) -> TaskGraph:
        # Plan-and-Execute: the model produces the executable step list first.
        # Only when the plan is unavailable or invalid do we fall back to the
        # deterministic intent template (and say so in the plan info holder).
        plan = approved_plan
        degraded = False
        if plan is None and self.plan_generator is not None:
            plan = self.plan_generator.generate(brief)
            degraded = plan is None
        if plan is not None:
            graph = self._graph_from_plan(plan, brief, run_id=run_id, project_id=project_id, now=now)
        else:
            intent = brief.intent
            if intent == "world_only":
                graph = self._world_only_graph(brief, run_id=run_id, project_id=project_id, now=now)
            elif intent == "characters_only":
                graph = self._characters_only_graph(brief, run_id=run_id, project_id=project_id, now=now)
            elif intent == "outline_only":
                graph = self._outline_only_graph(brief, run_id=run_id, project_id=project_id, now=now)
            else:
                graph = self._full_novel_graph(brief, run_id=run_id, project_id=project_id, now=now)
        if self.plan_info_holder is not None:
            self.plan_info_holder["degraded"] = degraded
            if plan is not None:
                # Success: clear any stale failure reason from a previous run.
                self.plan_info_holder.pop("reason", None)
            self.plan_info_holder["steps"] = (
                tuple((step.step_id, step.tool, step.description) for step in plan.steps)
                if plan is not None
                else ()
            )
        return graph

    def build_rewrite_graph(
        self,
        brief: BriefContract,
        chapter_number: int,
        *,
        run_id: str,
        project_id: ProjectId,
        now: Callable[[], datetime],
    ) -> TaskGraph:
        """Build a single-chapter rewrite graph for a REVISE command.

        The chapter node is forced (``force_rewrite=1``) so the executor
        redrafts it instead of skipping it as already written, and the new
        revision is accepted without activating (the old revision stays
        current) so a rebase proposal can ask the user for confirmation.
        """
        timestamp = now()
        nodes = (
            TaskGraphNode(
                node_id=f"chapter_{chapter_number}",
                task_type="DRAFT_CHAPTER",
                output_artifact_kind="chapter",
                output_contract="ChapterDraft",
                role=AgentRole.CHAPTER_EXECUTOR,
                skills=(
                    "chapter-scene@1.0.0",
                    "continuity-audit@1.0.0",
                    "memory-extraction@1.0.0",
                    "quality-review@1.0.0",
                    "targeted-revision@1.0.0",
                ),
                tools=(
                    "chapter.draft@1.0.0",
                    "chapter.repair@1.0.0",
                    "memory.extract@1.0.0",
                    "continuity.audit@1.0.0",
                    "quality.evaluate@1.0.0",
                ),
                acceptance=("continuity_ok", "quality_accept"),
                budget_tokens=8_000,
                metadata=(("chapter_number", str(chapter_number)), ("force_rewrite", "1")),
            ),
        )
        return self._graph(brief, run_id, project_id, timestamp, nodes)

    def _graph_from_plan(
        self,
        plan,
        brief: BriefContract,
        *,
        run_id: str,
        project_id: ProjectId,
        now: Callable[[], datetime],
    ) -> TaskGraph:
        """Map the model's step list onto semantic task-graph nodes.

        Step order is already validated (world -> cast -> outline -> chapters),
        so nodes keep their semantic ids and the deterministic dependency edges.
        """
        from novel_agent.agent.plan_generator import ExecutionPlan

        plan: ExecutionPlan = plan
        timestamp = now()
        nodes: list[TaskGraphNode] = []
        has_world = any(step.tool == "world.generate@1.0.0" for step in plan.steps)
        has_cast = any(step.tool == "character.generate@1.0.0" for step in plan.steps)
        has_revise = any(step.tool == "character.revise@1.0.0" for step in plan.steps)
        has_outline = any(step.tool == "outline.generate@1.0.0" for step in plan.steps)
        chapter_count = len(plan.chapter_steps)

        if has_world:
            nodes.append(
                TaskGraphNode(
                    node_id="world",
                    task_type="PLAN_WORLD",
                    output_artifact_kind="world_bible",
                    output_contract="WorldBible",
                    role=AgentRole.STORY_ARCHITECT,
                    skills=("world-building@1.0.0",),
                    tools=("world.generate@1.0.0",),
                    acceptance=("world_schema_valid",),
                    budget_tokens=4_000,
                )
            )
        if has_cast or has_revise:
            revise_targets = ",".join(
                sorted(
                    {
                        str(target)
                        for step in plan.steps
                        if step.tool == "character.revise@1.0.0"
                        for target in (step.parameters or {}).get("target_names", ())
                    }
                )
            )
            metadata = (("revise_targets", revise_targets),) if has_revise else ()
            nodes.append(
                TaskGraphNode(
                    node_id="cast",
                    task_type="PLAN_CAST",
                    depends_on=("world",) if has_world else (),
                    output_artifact_kind="character_set",
                    output_contract="CharacterSet",
                    role=AgentRole.CAST_DESIGNER,
                    skills=("cast-network@1.0.0",),
                    tools=(
                        ("character.generate@1.0.0", "character.revise@1.0.0")
                        if has_revise
                        else ("character.generate@1.0.0",)
                    ),
                    acceptance=("cast_schema_valid",),
                    budget_tokens=5_000,
                    metadata=metadata,
                )
            )
        if has_outline:
            dependencies = tuple(
                node_id for node_id in ("world", "cast") if node_id in {node.node_id for node in nodes}
            )
            nodes.append(
                TaskGraphNode(
                    node_id="outline",
                    task_type="PLAN_OUTLINE",
                    depends_on=dependencies,
                    output_artifact_kind="plan_contract",
                    output_contract="PlanContract",
                    role=AgentRole.OUTLINE_PLANNER,
                    skills=("story-architecture@1.0.0",),
                    tools=("outline.generate@1.0.0",),
                    acceptance=("plan_validated",),
                    budget_tokens=7_000,
                )
            )
        chapter_steps = plan.chapter_steps
        assets = self.existing_assets or {}
        written = assets.get("written_chapters")
        next_auto = max((int(number) for number in written), default=0) + 1 if written else 1
        chapter_numbers: list[int] = []
        for step in chapter_steps:
            explicit = (step.parameters or {}).get("chapter_number")
            if isinstance(explicit, int) and explicit >= 1:
                chapter_numbers.append(explicit)
                next_auto = max(next_auto, explicit + 1)
            else:
                chapter_numbers.append(next_auto)
                next_auto += 1
        for chapter_number in chapter_numbers:
            dependencies = ("outline",) if has_outline else ()
            if chapter_number > 1 and f"chapter_{chapter_number - 1}" in {
                node.node_id for node in nodes
            }:
                dependencies = (*dependencies, f"chapter_{chapter_number - 1}")
            nodes.append(
                TaskGraphNode(
                    node_id=f"chapter_{chapter_number}",
                    task_type="DRAFT_CHAPTER",
                    depends_on=dependencies,
                    output_artifact_kind="chapter",
                    output_contract="ChapterDraft",
                    role=AgentRole.CHAPTER_EXECUTOR,
                    skills=(
                        "chapter-scene@1.0.0",
                        "continuity-audit@1.0.0",
                        "memory-extraction@1.0.0",
                        "quality-review@1.0.0",
                        "targeted-revision@1.0.0",
                    ),
                    tools=(
                        "chapter.draft@1.0.0",
                        "chapter.repair@1.0.0",
                        "memory.extract@1.0.0",
                        "continuity.audit@1.0.0",
                        "quality.evaluate@1.0.0",
                    ),
                    acceptance=("continuity_ok", "quality_accept"),
                    budget_tokens=8_000,
                    metadata=(("chapter_number", str(chapter_number)),),
                )
            )
        return self._graph(brief, run_id, project_id, timestamp, tuple(nodes))

    def _world_only_graph(
        self,
        brief: BriefContract,
        *,
        run_id: str,
        project_id: ProjectId,
        now: Callable[[], datetime],
    ) -> TaskGraph:
        timestamp = now()
        nodes = (
            TaskGraphNode(
                node_id="world",
                task_type="PLAN_WORLD",
                output_artifact_kind="world_bible",
                output_contract="WorldBible",
                role=AgentRole.STORY_ARCHITECT,
                skills=("world-building@1.0.0",),
                tools=("world.generate@1.0.0",),
                acceptance=("world_schema_valid",),
                budget_tokens=4_000,
            ),
        )
        return self._graph(brief, run_id, project_id, timestamp, nodes)

    def _outline_only_graph(
        self,
        brief: BriefContract,
        *,
        run_id: str,
        project_id: ProjectId,
        now: Callable[[], datetime],
    ) -> TaskGraph:
        """world -> cast -> outline: produce a story outline, never chapters."""
        timestamp = now()
        nodes = (
            TaskGraphNode(
                node_id="world",
                task_type="PLAN_WORLD",
                output_artifact_kind="world_bible",
                output_contract="WorldBible",
                role=AgentRole.STORY_ARCHITECT,
                skills=("world-building@1.0.0",),
                tools=("world.generate@1.0.0",),
                acceptance=("world_schema_valid",),
                budget_tokens=4_000,
            ),
            TaskGraphNode(
                node_id="cast",
                task_type="PLAN_CAST",
                depends_on=("world",),
                output_artifact_kind="character_set",
                output_contract="CharacterSet",
                role=AgentRole.CAST_DESIGNER,
                skills=("cast-network@1.0.0",),
                tools=("character.generate@1.0.0",),
                acceptance=("cast_schema_valid",),
                budget_tokens=5_000,
            ),
            TaskGraphNode(
                node_id="outline",
                task_type="PLAN_OUTLINE",
                depends_on=("world", "cast"),
                output_artifact_kind="plan_contract",
                output_contract="PlanContract",
                role=AgentRole.OUTLINE_PLANNER,
                skills=("story-architecture@1.0.0",),
                tools=("outline.generate@1.0.0",),
                acceptance=("plan_validated",),
                budget_tokens=7_000,
            ),
        )
        return self._graph(brief, run_id, project_id, timestamp, nodes)

    def _characters_only_graph(
        self,
        brief: BriefContract,
        *,
        run_id: str,
        project_id: ProjectId,
        now: Callable[[], datetime],
    ) -> TaskGraph:
        timestamp = now()
        nodes = (
            TaskGraphNode(
                node_id="world",
                task_type="PLAN_WORLD",
                output_artifact_kind="world_bible",
                output_contract="WorldBible",
                role=AgentRole.STORY_ARCHITECT,
                skills=("world-building@1.0.0",),
                tools=("world.generate@1.0.0",),
                acceptance=("world_schema_valid",),
                budget_tokens=2_000,
            ),
            TaskGraphNode(
                node_id="cast",
                task_type="PLAN_CAST",
                depends_on=("world",),
                output_artifact_kind="character_set",
                output_contract="CharacterSet",
                role=AgentRole.CAST_DESIGNER,
                skills=("cast-network@1.0.0",),
                tools=("character.generate@1.0.0",),
                acceptance=("cast_schema_valid",),
                budget_tokens=4_000,
            ),
        )
        return self._graph(brief, run_id, project_id, timestamp, nodes)

    def _full_novel_graph(
        self,
        brief: BriefContract,
        *,
        run_id: str,
        project_id: ProjectId,
        now: Callable[[], datetime],
    ) -> TaskGraph:
        chapter_count = brief.target_chapter_count or 10
        timestamp = now()
        nodes: list[TaskGraphNode] = [
            TaskGraphNode(
                node_id="world",
                task_type="PLAN_WORLD",
                output_artifact_kind="world_bible",
                output_contract="WorldBible",
                role=AgentRole.STORY_ARCHITECT,
                skills=("world-building@1.0.0",),
                tools=("world.generate@1.0.0",),
                acceptance=("world_schema_valid",),
                budget_tokens=4_000,
            ),
            TaskGraphNode(
                node_id="cast",
                task_type="PLAN_CAST",
                depends_on=("world",),
                output_artifact_kind="character_set",
                output_contract="CharacterSet",
                role=AgentRole.CAST_DESIGNER,
                skills=("cast-network@1.0.0",),
                tools=("character.generate@1.0.0",),
                acceptance=("cast_schema_valid",),
                budget_tokens=5_000,
            ),
            TaskGraphNode(
                node_id="outline",
                task_type="PLAN_OUTLINE",
                depends_on=("world", "cast"),
                output_artifact_kind="plan_contract",
                output_contract="PlanContract",
                role=AgentRole.OUTLINE_PLANNER,
                skills=("story-architecture@1.0.0",),
                tools=("outline.generate@1.0.0",),
                acceptance=("plan_validated",),
                budget_tokens=7_000,
            ),
        ]
        for number in range(1, chapter_count + 1):
            dependencies = ("outline",)
            if number > 1:
                dependencies = (*dependencies, f"chapter_{number - 1}")
            nodes.append(
                TaskGraphNode(
                    node_id=f"chapter_{number}",
                    task_type="DRAFT_CHAPTER",
                    depends_on=dependencies,
                    output_artifact_kind="chapter",
                    output_contract="ChapterDraft",
                    role=AgentRole.CHAPTER_EXECUTOR,
                    skills=(
                        "chapter-scene@1.0.0",
                        "continuity-audit@1.0.0",
                        "memory-extraction@1.0.0",
                        "quality-review@1.0.0",
                        "targeted-revision@1.0.0",
                    ),
                    tools=(
                        "chapter.draft@1.0.0",
                        "chapter.repair@1.0.0",
                        "memory.extract@1.0.0",
                        "continuity.audit@1.0.0",
                        "quality.evaluate@1.0.0",
                    ),
                    acceptance=("continuity_ok", "quality_accept"),
                    budget_tokens=8_000,
                    metadata=(("chapter_number", str(number)),),
                )
            )
        return self._graph(brief, run_id, project_id, timestamp, tuple(nodes))

    def _graph(
        self,
        brief: BriefContract,
        run_id: str,
        project_id: ProjectId,
        timestamp: datetime,
        nodes: tuple[TaskGraphNode, ...],
    ) -> TaskGraph:
        return TaskGraph(
            graph_id=f"{run_id}:graph",
            brief_id=brief.brief_id,
            run_id=run_id,
            project_id=project_id,
            domain=self.domain,
            nodes=nodes,
            created_at=timestamp,
            provenance=Provenance(
                source_kind=ProvenanceKind.USER_REQUEST,
                source_id=run_id,
                source_sha256="0" * 64,
                observed_at=timestamp,
            ),
        )

    def node_executor(self, node: TaskGraphNode) -> NodeExecutor | None:
        return _NovelNodeExecutor(self, node)


class _NovelNodeExecutor(NodeExecutor):
    def __init__(self, plugin: NovelDomainPlugin, node: TaskGraphNode) -> None:
        self._plugin = plugin
        self._node = node

    def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        match self._node.task_type:
            case "PLAN_WORLD":
                return self._world(context)
            case "PLAN_CAST":
                return self._cast(context)
            case "PLAN_OUTLINE":
                return self._outline(context)
            case "DRAFT_CHAPTER":
                return self._chapter(context)
            case unreachable:
                assert_never(unreachable)

    # -- planning nodes ------------------------------------------------------

    def _already_approved(self, context: NodeExecutionContext, kind: str) -> bool:
        """True when the run already has an accepted artifact of this kind.

        In approval mode the run is requeued after each confirmation; nodes
        whose output was already approved must be skipped, otherwise the
        generator would redraft the same artifact and the second approval
        would collide on the artifact id.
        """
        if self._plugin.auto_approve:
            return False
        row = context.coordinator.connection.execute(
            "SELECT 1 FROM artifacts WHERE project_id = ? AND run_id = ? AND kind = ? "
            "AND status = 'accepted' LIMIT 1",
            (context.project_id.root, context.run_id, kind),
        ).fetchone()
        return row is not None

    def _world(self, context: NodeExecutionContext) -> NodeExecutionResult:
        if self._already_approved(context, "world_bible"):
            # Approved earlier in approval mode: skip without producing an
            # output, so downstream nodes fall back to the existing asset.
            return NodeExecutionResult(
                status="succeeded",
                output=None,
            )
        brief = _story_brief(context)
        selected = select_planning_skills(context.catalog, context.lease, brief)
        match selected:
            case tuple() as skills:
                pass
            case str() as reason:
                _block(context, reason)
                return NodeExecutionResult(status="blocked", reason=reason)
            case unreachable:
                assert_never(unreachable)
        result = generate_world_step(
            self._plugin.planning_generator,
            context.gateway,
            context.lease,
            brief,
            skills,
        )
        match result:
            case WorldAdaptation():
                if not self._plugin.auto_approve:
                    return self._stage_and_pause(context, "world_bible", result.raw.raw_bytes)
                committed = False
                if _world_only_graph(context):
                    from novel_agent.domains.novel_commit import commit_world_only

                    commit_world_only(
                        context.coordinator.connection,
                        run_id=context.run_id,
                        project_id=context.project_id.root,
                        brief_id=context.brief.brief_id,
                        world_result=result,
                        committed_at=context.now(),
                    )
                    committed = True
                return NodeExecutionResult(
                    status="succeeded",
                    output=NodeOutput(node_id=self._node.node_id, value=result, committed=committed),
                )
            case SchemaFailure():
                _block(context, "world_schema_failure")
                return NodeExecutionResult(status="blocked", reason="world_schema_failure")
            case unreachable:
                assert_never(unreachable)

    def _cast(self, context: NodeExecutionContext) -> NodeExecutionResult:
        if self._already_approved(context, "character_set"):
            return NodeExecutionResult(
                status="succeeded",
                output=None,
            )
        brief = _story_brief(context)
        world_output = context.outputs.get("world")
        if world_output is not None and world_output.value is not None:
            world: WorldBible = world_output.value.value
        else:
            # No world node in the graph: reuse an existing world asset.
            world = self._resolve_world(context)
            if world is None:
                _block(context, "world_artifact_missing")
                return NodeExecutionResult(status="blocked", reason="world_artifact_missing")

        selected = select_planning_skills(context.catalog, context.lease, brief)
        match selected:
            case tuple() as skills:
                pass
            case str() as reason:
                _block(context, reason)
                return NodeExecutionResult(status="blocked", reason=reason)
            case unreachable:
                assert_never(unreachable)
        revise_targets = dict(self._node.metadata).get("revise_targets")
        if revise_targets:
            targets = tuple(name for name in str(revise_targets).split(",") if name)
            existing = self._resolve_cast(context)
            if existing is None:
                _block(context, "cast_assets_missing")
                return NodeExecutionResult(status="blocked", reason="cast_assets_missing")
            result = generate_revise_cast_step(
                self._plugin.planning_generator,
                context.gateway,
                context.lease,
                brief,
                world,
                existing,
                targets,
                skills,
            )
        else:
            result = generate_cast_step(
                self._plugin.planning_generator,
                context.gateway,
                context.lease,
                brief,
                world,
                skills,
            )
        match result:
            case CharacterAdaptation():
                self._swap_real_reviewer(context, result)
                self._register_character_names(context, result)
                if not self._plugin.auto_approve:
                    return self._stage_and_pause(context, "character_set", result.raw.raw_bytes)
                committed = False
                if _cast_only_graph(context):
                    from novel_agent.domains.novel_commit import commit_cast_only

                    commit_cast_only(
                        context.coordinator.connection,
                        run_id=context.run_id,
                        project_id=context.project_id.root,
                        brief_id=context.brief.brief_id,
                        cast_result=result,
                        committed_at=context.now(),
                    )
                    committed = True
                return NodeExecutionResult(
                    status="succeeded",
                    output=NodeOutput(node_id=self._node.node_id, value=result, committed=committed),
                )
            case SchemaFailure():
                _block(context, "cast_schema_failure")
                return NodeExecutionResult(status="blocked", reason="cast_schema_failure")
            case unreachable:
                assert_never(unreachable)

    def _swap_real_reviewer(
        self,
        context: NodeExecutionContext,
        cast_result: CharacterAdaptation,
    ) -> None:
        factory = self._plugin.real_reviewer_factory
        holder = self._plugin.reviewer_holder
        if factory is None or holder is None:
            return
        holder["current"] = factory(context, cast_result)
        holder["swapped"] = True

    def _ensure_real_reviewer(self, context: NodeExecutionContext) -> None:
        """Install the provider-backed reviewer when the cast node never ran.

        Continuation runs (existing outline/chapters) skip the cast node, so
        ``_swap_real_reviewer`` is never invoked and the chapter quality gate
        would silently fall back to the offline fake reviewer.  Reuse the
        existing cast assets to build the real evaluator exactly once.
        """
        factory = self._plugin.real_reviewer_factory
        holder = self._plugin.reviewer_holder
        if factory is None or holder is None or holder.get("swapped"):
            return
        cast = self._resolve_cast(context)
        if cast is None:
            return
        import hashlib as _hashlib

        from novel_agent.adapters import (
            CharacterAdaptation,
            CreativeArtifactKind,
            CreativeParseRoute,
            RawCreativeArtifact,
        )

        adaptation = CharacterAdaptation(
            raw=RawCreativeArtifact(
                kind=CreativeArtifactKind.CHARACTERS,
                raw_bytes=cast.model_dump_json().encode(),
                content_sha256=_hashlib.sha256(cast.model_dump_json().encode()).hexdigest(),
                adapter_version="reuse:v1",
            ),
            value=cast,
            parse_route=CreativeParseRoute.STRICT_JSON,
        )
        self._swap_real_reviewer(context, adaptation)

    def _resolve_world(self, context: NodeExecutionContext) -> WorldBible | None:
        """Prefer a world produced by the graph; fall back to existing assets."""
        world_output = context.outputs.get("world")
        if world_output is not None and world_output.value is not None:
            return world_output.value.value
        assets = self._plugin.existing_assets
        if assets is None or "world_text" not in assets:
            return None
        timestamp = context.now()
        return WorldBible(
            world_id=f"{context.run_id}:existing-world",
            brief_id=context.brief.brief_id,
            title=str(assets.get("world_title") or "world"),
            premise=str(assets["world_text"]),
            rules=(),
            constraints=(),
            created_at=timestamp,
            provenance=context.brief.provenance,
        )

    def _resolve_cast(self, context: NodeExecutionContext) -> CharacterSet | None:
        """Prefer a cast produced by the graph; fall back to existing assets."""
        cast_output = context.outputs.get("cast")
        if cast_output is not None and cast_output.value is not None:
            return cast_output.value.value
        assets = self._plugin.existing_assets
        if assets is None or "characters" not in assets:
            return None
        timestamp = context.now()
        cards = tuple(
            CharacterCard(
                character_id=f"{context.run_id}:existing-character:{index}",
                name=str(item.get("name", "")),
                role=str(item.get("role_introduction") or ""),
                goal=str(item.get("role_experience") or ""),
                traits=(),
                created_at=timestamp,
                provenance=context.brief.provenance,
            )
            for index, item in enumerate(assets["characters"], start=1)
            if item.get("name")
        )
        if not cards:
            return None
        return CharacterSet(
            character_set_id=f"{context.run_id}:existing-cast",
            world_id="",
            characters=cards,
            created_at=timestamp,
            provenance=context.brief.provenance,
        )

    @staticmethod
    def _world_adaptation(context: NodeExecutionContext, world: WorldBible):
        from novel_agent.adapters import WorldAdaptation

        world_output = context.outputs.get("world")
        if world_output is not None and world_output.value is not None:
            return world_output.value
        import hashlib as _hashlib

        payload = world.model_dump_json()
        return WorldAdaptation(
            raw=RawCreativeArtifact(
                kind=CreativeArtifactKind.WORLD,
                raw_bytes=payload.encode(),
                content_sha256=_hashlib.sha256(payload.encode()).hexdigest(),
                adapter_version="existing-assets-v1",
            ),
            value=world,
            parse_route=CreativeParseRoute.LEGACY_NORMALIZATION,
        )

    @staticmethod
    def _cast_adaptation(context: NodeExecutionContext, characters: CharacterSet):
        from novel_agent.adapters import CharacterAdaptation

        cast_output = context.outputs.get("cast")
        if cast_output is not None and cast_output.value is not None:
            return cast_output.value
        import hashlib as _hashlib

        payload = characters.model_dump_json()
        return CharacterAdaptation(
            raw=RawCreativeArtifact(
                kind=CreativeArtifactKind.CHARACTERS,
                raw_bytes=payload.encode(),
                content_sha256=_hashlib.sha256(payload.encode()).hexdigest(),
                adapter_version="existing-assets-v1",
            ),
            value=characters,
            parse_route=CreativeParseRoute.LEGACY_NORMALIZATION,
        )

    def _resolve_outline(self, context: NodeExecutionContext) -> tuple[PlanContract, int] | None:
        """Reuse the existing accepted plan contract when the graph has no outline node."""
        assets = self._plugin.existing_assets
        if assets is None:
            return None
        outline = assets.get("outline")
        if outline is None:
            return None
        try:
            plan = PlanContract.model_validate_json(str(outline["plan_json"]))
        except ValidationError:
            return None
        return plan, int(outline["revision"])

    def _propose_rewrite_rebase(
        self,
        context: NodeExecutionContext,
        plan: PlanContract,
        chapter_number: int,
        task_id: str,
    ) -> None:
        """After a forced rewrite, propose swapping the current chapter for the
        new revision; the run enters WAITING_REVIEW for user confirmation.

        Uses the raw proposal writer instead of the lease-guarded coordinator
        path: the chapter accept already completed the run (SUCCEEDED), so no
        RUNNING lease is available — the rewrite moves a finished run back
        into review.
        """
        import hashlib as _hashlib

        from novel_agent.pathing import ArtifactId
        from novel_agent.revision_models import ChapterReplacementRequest, ChapterRevisionRef
        from novel_agent.revisions import propose_chapter_rebase as _propose_rebase
        from novel_agent.run_event_store import append_run_event as _append_run_event
        from novel_agent.run_state import RunEvent as _RunEvent

        chapter = next((c for c in plan.chapters if c.chapter_number == chapter_number), None)
        if chapter is None:
            return
        connection = context.coordinator.connection
        seed = f"{context.run_id}:{chapter.chapter_plan_id}"
        artifact_id = ArtifactId(f"chapter-{_hashlib.sha256(seed.encode()).hexdigest()[:24]}")
        rows = connection.execute(
            "SELECT revision, is_current FROM artifacts WHERE artifact_id = ? AND run_id = ? "
            "AND kind = 'chapter' AND status = 'accepted' ORDER BY revision",
            (artifact_id.root, context.run_id),
        ).fetchall()
        current_revision = next((int(row[0]) for row in rows if int(row[1]) == 1), None)
        newest_revision = max((int(row[0]) for row in rows), default=None)
        if current_revision is None or newest_revision is None or current_revision == newest_revision:
            return
        proposal = _propose_rebase(
            connection,
            ChapterReplacementRequest(
                project_id=context.project_id,
                run_id=context.run_id,
                source=ChapterRevisionRef(
                    artifact_id=artifact_id,
                    revision=current_revision,
                    chapter_number=chapter_number,
                ),
                replacement=ChapterRevisionRef(
                    artifact_id=artifact_id,
                    revision=newest_revision,
                    chapter_number=chapter_number,
                ),
            ),
            context.now(),
        )
        timestamp = context.now()
        connection.execute(
            "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
            (RunStatus.WAITING_REVIEW.value, timestamp.isoformat(), context.run_id),
        )
        _append_run_event(
            connection,
            _RunEvent(
                context.run_id,
                "chapter_rebase_proposed",
                {"proposal": proposal.proposal_id, "branch": proposal.branch_id},
                timestamp,
            ),
        )

    def _stage_and_pause(
        self,
        context: NodeExecutionContext,
        kind: str,
        raw_bytes: bytes,
    ) -> NodeExecutionResult:
        """Persist a produced artifact as staged and pause the run for approval.

        The artifact is written with a deterministic staged id (the node's
        artifact id) but marked ``staged``/``is_current=0``; the run moves to
        WAITING_REVIEW with an ``artifact_approval_required`` event so the
        console can show the content and let the user approve, edit, or
        regenerate it.  Approval persists the artifact through the normal
        commit paths.
        """
        import hashlib as _hashlib

        from novel_agent.pathing import ArtifactId
        from novel_agent.run_event_store import append_run_event as _append_run_event
        from novel_agent.run_state import RunEvent as _RunEvent

        artifact_id = ArtifactId(
            f"staged-{kind}-{_hashlib.sha256(f'{context.run_id}:{self._node.node_id}'.encode()).hexdigest()[:24]}"
        )
        timestamp = context.now()
        connection = context.coordinator.connection
        connection.execute(
            "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
            "schema_version, is_current, created_at) VALUES (?, 1, ?, ?, ?, 'staged', ?, ?, 1, 0, ?) "
            "ON CONFLICT(artifact_id, revision) DO UPDATE SET raw_bytes = excluded.raw_bytes, "
            "sha256 = excluded.sha256, created_at = excluded.created_at",
            (
                artifact_id.root,
                context.project_id.root,
                context.run_id,
                kind,
                raw_bytes,
                _hashlib.sha256(raw_bytes).hexdigest(),
                timestamp.isoformat(),
            ),
        )
        connection.execute(
            "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
            (RunStatus.WAITING_REVIEW.value, timestamp.isoformat(), context.run_id),
        )
        _append_run_event(
            connection,
            _RunEvent(
                context.run_id,
                "artifact_approval_required",
                {"artifact_id": artifact_id.root, "kind": kind, "node": self._node.node_id},
                timestamp,
            ),
        )
        return NodeExecutionResult(
            status="blocked",
            reason="artifact_approval_required",
            output=NodeOutput(node_id=self._node.node_id, value=artifact_id.root, committed=False),
        )

    def _is_written_chapter(self, context: NodeExecutionContext, chapter_number: int) -> bool:
        assets = self._plugin.existing_assets
        if assets is None:
            return False
        written = assets.get("written_chapters")
        return isinstance(written, (tuple, list)) and chapter_number in written

    def _mirror_reused_plan(
        self,
        context: NodeExecutionContext,
        plan: PlanContract,
        plan_revision: int,
    ) -> None:
        """Mirror an earlier run's accepted plan contract into the current run.

        The chapter staging authority requires the plan to be an accepted
        artifact of the leased run, so a reused plan is copied (same bytes,
        same artifact id) under the current run before chapters execute.  The
        prior run's current pointer is moved to the mirrored row so the
        per-artifact current uniqueness is preserved.
        """
        import hashlib as _hashlib

        raw_bytes = plan.model_dump_json().encode()
        connection = context.coordinator.connection
        row = connection.execute(
            "SELECT artifact_id, revision FROM artifacts WHERE artifact_id = ? AND run_id = ? "
            "AND kind = 'plan_contract' AND status = 'accepted' AND is_current = 1",
            (plan.plan_id, context.run_id),
        ).fetchone()
        if row is not None:
            return
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
                context.project_id.root,
                context.run_id,
                raw_bytes,
                _hashlib.sha256(raw_bytes).hexdigest(),
                context.now().isoformat(),
            ),
        )

    def _register_character_names(
        self,
        context: NodeExecutionContext,
        cast_result: CharacterAdaptation,
    ) -> None:
        holder = self._plugin.memory_names_holder
        if holder is None:
            return
        holder["names"] = tuple(character.name for character in cast_result.value.characters)

    def _outline(self, context: NodeExecutionContext) -> NodeExecutionResult:
        if self._already_approved(context, "plan_contract"):
            return NodeExecutionResult(
                status="succeeded",
                output=None,
            )
        brief = _story_brief(context)
        world = self._resolve_world(context)
        characters = self._resolve_cast(context)
        if world is None or characters is None:
            _block(context, "plan_prerequisites_missing")
            return NodeExecutionResult(status="blocked", reason="plan_prerequisites_missing")
        selected = select_planning_skills(context.catalog, context.lease, brief)
        match selected:
            case tuple() as skills:
                pass
            case str() as reason:
                _block(context, reason)
                return NodeExecutionResult(status="blocked", reason=reason)
            case unreachable:
                assert_never(unreachable)
        outline_result = generate_outline_step(
            self._plugin.planning_generator,
            context.gateway,
            context.lease,
            brief,
            world,
            characters,
            skills,
        )
        has_chapters = any(node.node_id.startswith("chapter_") for node in context.graph.nodes)
        if not has_chapters:
            # Plan without chapter steps (outline-only): commit the package
            # and finish the run; no phantom chapter tasks are created.
            from novel_agent.adapters import SchemaFailure
            from novel_agent.domains.novel_commit import commit_outline_only

            match outline_result:
                case SchemaFailure():
                    _block(context, "outline_schema_failure")
                    return NodeExecutionResult(status="blocked", reason="outline_schema_failure")
                case _:
                    pass
            if not self._plugin.auto_approve:
                return self._stage_and_pause(context, "plan_contract", outline_result.raw.raw_bytes)
            commit_outline_only(
                context.coordinator.connection,
                run_id=context.run_id,
                project_id=context.project_id.root,
                brief_id=context.brief.brief_id,
                world_result=self._world_adaptation(context, world),
                cast_result=self._cast_adaptation(context, characters),
                outline_result=outline_result,
                committed_at=context.now(),
            )
            return NodeExecutionResult(
                status="succeeded",
                output=NodeOutput(node_id=self._node.node_id, value=outline_result, committed=True),
            )
        if not self._plugin.auto_approve:
            from novel_agent.adapters import SchemaFailure as _SchemaFailure

            match outline_result:
                case _SchemaFailure():
                    _block(context, "outline_schema_failure")
                    return NodeExecutionResult(status="blocked", reason="outline_schema_failure")
                case _:
                    pass
            return self._stage_and_pause(context, "plan_contract", outline_result.raw.raw_bytes)
        outcome = commit_plan(
            context.coordinator,
            context.lease,
            brief,
            self._world_adaptation(context, world),
            self._cast_adaptation(context, characters),
            outline_result,
            skills,
        )
        match outcome:
            case PlanningSucceeded():
                return NodeExecutionResult(
                    status="succeeded",
                    output=NodeOutput(node_id=self._node.node_id, value=outline_result, committed=True),
                )
            case PlanningBlocked(reason_code=reason):
                return NodeExecutionResult(status="blocked", reason=reason)
            case unreachable:
                assert_never(unreachable)

    # -- chapter nodes -------------------------------------------------------

    def _chapter(self, context: NodeExecutionContext) -> NodeExecutionResult:
        self._ensure_real_reviewer(context)
        outline_output = context.outputs.get("outline")
        reused_outline = outline_output is None or outline_output.value is None
        if reused_outline:
            # No outline node in the graph: reuse the existing accepted plan.
            resolved = self._resolve_outline(context)
            if resolved is None:
                _block(context, "plan_artifact_missing")
                return NodeExecutionResult(status="blocked", reason="plan_artifact_missing")
            plan, plan_revision = resolved
        else:
            plan: PlanContract = outline_output.value.value
            accepted = context.coordinator.load_accepted_plan(context.lease, plan.plan_id)
            if accepted is None:
                _block(context, "plan_artifact_missing")
                return NodeExecutionResult(status="blocked", reason="plan_artifact_missing")
            plan_revision = accepted.artifact_revision
        chapter_number = int(dict(self._node.metadata)["chapter_number"])
        chapter = next(
            (candidate for candidate in plan.chapters if candidate.chapter_number == chapter_number),
            None,
        )
        if chapter is None:
            _block(context, "chapter_plan_missing")
            return NodeExecutionResult(status="blocked", reason="chapter_plan_missing")
        force_rewrite = dict(self._node.metadata).get("force_rewrite") == "1"
        if not force_rewrite and reused_outline and self._is_written_chapter(context, chapter_number):
            # Already produced under the reused outline: skip, do not rewrite.
            # The run continues when later chapter nodes exist; when this is
            # the final node, the executor's graph-completion path finishes it.
            self._mirror_reused_plan(context, plan, plan_revision)
            return NodeExecutionResult(
                status="succeeded",
                output=NodeOutput(node_id=self._node.node_id, value=chapter.chapter_plan_id, committed=False),
            )
        has_next_node = any(
            node.node_id == f"chapter_{chapter_number + 1}" for node in context.graph.nodes
        )
        next_chapter = (
            next(
                (candidate for candidate in plan.chapters if candidate.chapter_number == chapter_number + 1),
                None,
            )
            if has_next_node
            else None
        )
        brief = _story_brief(context)
        request = ChapterExecutionRequest(
            project_id=context.project_id,
            plan=plan,
            plan_revision=plan_revision,
            target_chinese_characters=_DEFAULT_TARGET_WORDS,
            genre=brief.genre,
            activate=not force_rewrite,
            auto_approve=self._plugin.auto_approve,
        )
        workflow = ChapterExecutionWorkflow(
            context.coordinator,
            context.coordinator.context_builder(),
            context.catalog,
            self._plugin.chapter_generator,
            self._plugin.extractor,
            self._plugin.auditor,
            _current_reviewer(self),
            context.path_resolver,
            tool_gateway=context.gateway,
            stage_event_sink=context.stage_event_sink,
        )
        task_id = f"{context.run_id}:chapter:{chapter.chapter_plan_id}"
        if reused_outline:
            # The reused plan was committed by an earlier run: mirror it as an
            # accepted artifact of *this* run so the chapter staging authority
            # (plan must belong to the leased run) and the memory provenance
            # checks accept it, then create the pending chapter task row that
            # plan-commit would normally create.
            self._mirror_reused_plan(context, plan, plan_revision)
            timestamp = context.now().isoformat()
            context.coordinator.connection.execute(
                "INSERT INTO tasks(task_id, run_id, task_type, state, idempotency_key, created_at, updated_at) "
                "VALUES (?, ?, 'chapter', 'pending', ?, ?, ?) "
                "ON CONFLICT(task_id) DO NOTHING",
                (task_id, context.run_id, task_id, timestamp, timestamp),
            )
        context.coordinator.activate_chapter_task(context.lease, task_id)
        result = execute_single_chapter(workflow, context.lease, request, chapter, next_chapter, task_id)
        match result:
            case ChapterExecutionSucceeded():
                if force_rewrite:
                    self._propose_rewrite_rebase(context, plan, chapter_number, task_id)
                return NodeExecutionResult(
                    status="succeeded",
                    output=NodeOutput(
                        node_id=self._node.node_id,
                        value=chapter.chapter_plan_id,
                        committed=True,
                    ),
                )
            case ChapterExecutionWaitingReview(reason_code=reason):
                return NodeExecutionResult(status="blocked", reason=reason)
            case ChapterExecutionCancelled():
                return NodeExecutionResult(status="blocked", reason="cancelled")
            case unreachable:
                assert_never(unreachable)


def detect_existing_assets(project_root, connection=None) -> dict[str, object]:
    """Detect existing world/character assets for reuse by later tasks.

    Checks legacy-format files (``worlds/*.txt``, ``characters/*.json``) and,
    when a project database connection is supplied, validated M1 canonical
    memory (``world_rule`` / ``character_identity``) plus the current accepted
    plan contract (outline) and written chapters.  File entries take
    precedence; SQLite memory fills gaps so governed runs reuse canonical
    settings instead of regenerating them.  Returns a dict with optional
    ``world_title``/``world_text``, ``characters`` (list of legacy character
    dicts), ``outline`` (plan payload + revision) and ``written_chapters``
    (chapter numbers already produced under the current outline).  Empty when
    nothing exists.
    """
    import json as json_module

    assets: dict[str, object] = {}
    worlds_dir = project_root / "worlds"
    characters_dir = project_root / "characters"
    if worlds_dir.is_dir():
        worlds = sorted(worlds_dir.glob("*.txt"))
        if worlds:
            assets["world_title"] = worlds[0].stem
            assets["world_text"] = worlds[0].read_text(encoding="utf-8")
    if characters_dir.is_dir():
        cards: list[dict] = []
        for path in sorted(characters_dir.glob("*.json")):
            try:
                data = json_module.loads(path.read_text(encoding="utf-8"))
            except (json_module.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(data, dict) and data.get("name"):
                cards.append(data)
        if cards:
            assets["characters"] = cards
    _merge_sqlite_assets(assets, connection)
    _merge_sqlite_outline(assets, connection)
    return assets


def _merge_sqlite_outline(assets: dict[str, object], connection) -> None:
    """Merge the current accepted plan contract and written chapters.

    The newest ``plan_contract`` artifact supplies the outline (structured
    JSON, so it can be parsed back into a ``PlanContract``); accepted chapter
    artifacts whose ``chapter_plan_id`` belongs to that plan identify the
    chapters already written.  Memory failures are best-effort.
    """
    if connection is None:
        return
    import json as json_module

    try:
        row = connection.execute(
            "SELECT raw_bytes, revision FROM artifacts "
            "WHERE kind = 'plan_contract' AND status = 'accepted' AND is_current = 1 "
            "ORDER BY created_at DESC, revision DESC LIMIT 1"
        ).fetchone()
    except Exception:  # noqa: BROAD_EXCEPT_OK - reuse is best-effort
        return
    if row is None:
        return
    try:
        plan = json_module.loads(bytes(row[0]).decode("utf-8"))
    except (json_module.JSONDecodeError, UnicodeDecodeError):
        return
    chapters = plan.get("chapters") or []
    if not chapters:
        return
    plan_id_to_number = {
        str(chapter.get("chapter_plan_id") or ""): int(chapter.get("chapter_number") or 0)
        for chapter in chapters
    }
    assets["outline"] = {
        "plan_json": json_module.dumps(plan, ensure_ascii=False),
        "revision": int(row[1]),
        "chapter_count": len(chapters),
        "chapters_text": "；".join(
            f"第{chapter.get('chapter_number')}章《{chapter.get('title')}》" for chapter in chapters
        ),
    }
    written: list[int] = []
    try:
        chapter_rows = connection.execute(
            "SELECT artifact_id, run_id, raw_bytes FROM artifacts "
            "WHERE kind = 'chapter' AND status = 'accepted'"
        ).fetchall()
        for artifact_id, run_id, raw_bytes in chapter_rows:
            number = 0
            try:
                draft = json_module.loads(bytes(raw_bytes).decode("utf-8"))
                if isinstance(draft, dict):
                    number = plan_id_to_number.get(str(draft.get("chapter_plan_id") or ""), 0)
            except (json_module.JSONDecodeError, UnicodeDecodeError):
                number = 0
            if not number:
                # Legacy plain-text chapter (raw is the prose, not a JSON
                # envelope): recover the chapter number from the artifact id,
                # which stage_chapter derives as sha256(run_id:plan_id)[:24].
                import hashlib as _hashlib

                suffix = str(artifact_id).removeprefix("chapter-")
                for plan_id, candidate_number in plan_id_to_number.items():
                    if _hashlib.sha256(f"{run_id}:{plan_id}".encode()).hexdigest()[:24] == suffix:
                        number = candidate_number
                        break
            if number and number not in written:
                written.append(number)
    except Exception:  # noqa: BROAD_EXCEPT_OK - reuse is best-effort
        written = []
    if written:
        assets["written_chapters"] = tuple(sorted(written))


def _merge_sqlite_assets(assets: dict[str, object], connection) -> None:
    """Merge validated M1 canonical memory into the file-based asset view.

    The newest ``world_rule`` supplies the world premise when no world file
    exists; ``character_identity`` records fill in characters whose names are
    not already covered by the files.  Memory failures are best-effort: a
    locked or malformed database must not block planning.
    """
    if connection is None:
        return
    import json as json_module

    try:
        rows = connection.execute(
            "SELECT kind, value_json FROM memory_records "
            "WHERE layer = 'm1_canonical' AND status = 'validated' "
            "AND kind IN ('world_rule', 'character_identity') "
            "ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
    except Exception:  # noqa: BROAD_EXCEPT_OK - memory reuse is best-effort
        return
    world_title: str | None = None
    world_text: str | None = None
    for kind, value_json in rows:
        try:
            data = json_module.loads(str(value_json))
        except (json_module.JSONDecodeError, TypeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if kind == "world_rule":
            if world_text is not None:
                continue  # newest world only
            title = str(data.get("title") or "")
            premise = str(data.get("premise") or "")
            rules = data.get("rules") or ()
            rule_lines = [str(rule) for rule in rules] if isinstance(rules, (tuple, list)) else []
            text = " ".join(part for part in (premise, *rule_lines) if part)
            if title or text:
                world_title = title or "world"
                world_text = text
        elif kind == "character_identity":
            name = str(data.get("name") or "")
            if not name:
                continue
            cards: list[dict] = list(assets.get("characters") or [])
            if any(str(card.get("name", "")) == name for card in cards):
                continue  # file wins over memory
            cards.append(
                {
                    "name": name,
                    "role_introduction": str(data.get("role") or ""),
                    "role_experience": str(data.get("goal") or ""),
                }
            )
            assets["characters"] = cards
    if world_text is not None:
        existing = str(assets.get("world_text") or "")
        if existing:
            assets["world_text"] = f"{existing} {world_text}"
            assets["world_title"] = str(assets.get("world_title") or world_title)
        else:
            assets["world_title"] = world_title
            assets["world_text"] = world_text


def existing_assets_summary(assets: dict[str, object] | None) -> str:
    """Human-readable summary of existing assets for the planner prompt."""
    if not assets:
        return "（项目暂无已有设定）"
    parts: list[str] = []
    world_text = assets.get("world_text")
    if world_text:
        title = assets.get("world_title") or "world"
        compact = " ".join(str(world_text).split())
        parts.append(f"世界观《{title}》：{compact[:200]}")
    characters = assets.get("characters")
    if characters:
        names = [str(item.get("name", "")) for item in characters]
        parts.append("已有角色：" + ",".join(names))
    outline = assets.get("outline")
    if outline:
        parts.append(
            f"已有大纲（{outline.get('chapter_count')}章）：{str(outline.get('chapters_text') or '')[:200]}"
        )
    written = assets.get("written_chapters")
    if written:
        parts.append("已写章节：" + ",".join(str(number) for number in written))
    return "\n".join(parts) if parts else "（项目暂无已有设定）"


def _story_brief(context: NodeExecutionContext) -> StoryBrief:
    return StoryBrief(
        brief_id=context.brief.brief_id,
        project_id=context.brief.project_id,
        user_requirements=context.brief.raw_requirements,
        genre="fiction",
        target_chapter_count=context.brief.target_chapter_count or 10,
        created_at=context.now(),
        provenance=context.brief.provenance,
    )


def _current_reviewer(self: _NovelNodeExecutor) -> object:
    holder = self._plugin.reviewer_holder
    if holder is None or "current" not in holder:
        return OfflineQualityReviewer()
    return holder["current"]


def _block(context: NodeExecutionContext, reason_code: str) -> None:
    context.coordinator.block_planning(context.lease, reason_code)


def _world_only_graph(context: NodeExecutionContext) -> bool:
    """True when the graph consists of exactly the world node (no downstream work)."""
    return len(context.graph.nodes) == 1 and context.graph.nodes[0].node_id == "world"


def _cast_only_graph(context: NodeExecutionContext) -> bool:
    """True when the graph stops after the cast node (no outline, no chapters)."""
    return not any(
        node.node_id == "outline" or node.node_id.startswith("chapter_")
        for node in context.graph.nodes
    )


__all__ = ("NovelDomainPlugin",)
