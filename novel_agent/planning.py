from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol, assert_never

from novel_agent.adapters import (
    CharacterAdaptation,
    CreativeArtifactKind,
    CreativeParseRoute,
    OutlineAdaptation,
    RawCreativeArtifact,
    SchemaFailure,
    WorldAdaptation,
)
from novel_agent.contracts import CharacterSet, PlanApprovalStatus, PlanContract, RunStatus, StoryBrief, WorldBible
from novel_agent.planning_commit import PlanningCommitReceipt, PlanningCommitRequest
from novel_agent.run_models import RunLease
from novel_agent.skill_manifests.models import ActiveSkillPin, SkillPause, SkillSelection, TaskDescriptor
from novel_agent.skills import SkillCatalog, SkillSelector
from novel_agent.tool_models import AgentRole, ToolInvocationRequest
from novel_agent.tool_registry import ToolCallResult, ToolGateway
from novel_agent.workflow_tools import (
    CastGenerateInput,
    CastReviseInput,
    OutlineGenerateInput,
    WorldGenerateInput,
    is_schema_invalid,
    unwrap_tool_result,
    workflow_tool_request,
)


_MAX_AUTOMATIC_CHAPTERS: Final = 100


class PlanningGenerator(Protocol):
    def generate_world(self, brief: StoryBrief) -> WorldAdaptation | SchemaFailure:
        ...

    def generate_characters(self, brief: StoryBrief, world: WorldBible) -> CharacterAdaptation | SchemaFailure:
        ...

    def revise_characters(
        self,
        brief: StoryBrief,
        world: WorldBible,
        existing: CharacterSet,
        target_names: tuple[str, ...],
    ) -> CharacterAdaptation | SchemaFailure:
        ...

    def generate_outline(
        self,
        brief: StoryBrief,
        world: WorldBible,
        characters: CharacterSet,
    ) -> OutlineAdaptation | SchemaFailure:
        ...


class PlanningCoordinator(Protocol):
    def commit_validated_plan(self, lease: RunLease, request: PlanningCommitRequest) -> PlanningCommitReceipt:
        ...

    def block_planning(self, lease: RunLease, reason_code: str) -> None:
        ...


@dataclass(frozen=True, slots=True)
class PlanningPolicy:
    maximum_automatic_chapters: int = _MAX_AUTOMATIC_CHAPTERS


@dataclass(frozen=True, slots=True)
class PlanningSucceeded:
    plan_id: str
    first_task_id: str
    plan_revision: int


@dataclass(frozen=True, slots=True)
class PlanningBlocked:
    reason_code: str


@dataclass(frozen=True, slots=True)
class _PlanningStage:
    suffix: str
    task_type: str
    role: AgentRole
    context_slices: frozenset[str]
    artifact_kinds: frozenset[str]
    tool_ids: frozenset[str]
    context_budget: int


_PLANNING_STAGES: Final[tuple[_PlanningStage, ...]] = (
    _PlanningStage(
        "brief",
        "INTAKE",
        AgentRole.BRIEF_PLANNER,
        frozenset(("raw_brief", "defaults")),
        frozenset(),
        frozenset(),
        2_000,
    ),
    _PlanningStage(
        "world",
        "PLAN_WORLD",
        AgentRole.STORY_ARCHITECT,
        frozenset(("story_brief",)),
        frozenset(("story_brief",)),
        frozenset(("world.generate@1.0.0",)),
        4_000,
    ),
    _PlanningStage(
        "cast",
        "PLAN_CAST",
        AgentRole.CAST_DESIGNER,
        frozenset(("story_brief", "world_bible")),
        frozenset(("story_brief", "world_bible")),
        frozenset(("character.generate@1.0.0",)),
        5_000,
    ),
    _PlanningStage(
        "outline",
        "PLAN_OUTLINE",
        AgentRole.OUTLINE_PLANNER,
        frozenset(("story_brief", "world_bible", "character_set")),
        frozenset(("story_brief", "world_bible", "character_set")),
        frozenset(("outline.generate@1.0.0",)),
        7_000,
    ),
)


@dataclass(frozen=True, slots=True)
class PlanningWorkflow:
    coordinator: PlanningCoordinator
    generator: PlanningGenerator
    catalog: SkillCatalog
    policy: PlanningPolicy = PlanningPolicy()
    tool_gateway: ToolGateway | None = None
    stage_event_sink: Callable[[str], None] | None = None

    def execute(self, lease: RunLease, brief: StoryBrief) -> PlanningSucceeded | PlanningBlocked:
        if brief.target_chapter_count > self.policy.maximum_automatic_chapters:
            return self._block(lease, "plan_budget_exceeded")
        selected = select_planning_skills(self.catalog, lease, brief)
        match selected:
            case tuple() as skills:
                pass
            case str() as reason_code:
                return self._block(lease, reason_code)
            case unreachable:
                assert_never(unreachable)
        world_result = self._generate_world(lease, brief, skills)
        self._emit_stage("plan_world_complete")
        match world_result:
            case WorldAdaptation(value=world):
                pass
            case SchemaFailure():
                return self._block(lease, "world_schema_failure")
            case unreachable:
                assert_never(unreachable)
        characters_result = self._generate_characters(lease, brief, world, skills)
        self._emit_stage("plan_cast_complete")
        match characters_result:
            case CharacterAdaptation(value=characters):
                pass
            case SchemaFailure():
                return self._block(lease, "cast_schema_failure")
            case unreachable:
                assert_never(unreachable)
        outline_result = self._generate_outline(lease, brief, world, characters, skills)
        self._emit_stage("plan_outline_complete")
        match outline_result:
            case OutlineAdaptation(value=plan):
                pass
            case SchemaFailure():
                return self._block(lease, "outline_schema_failure")
            case unreachable:
                assert_never(unreachable)
        return commit_plan(
            self.coordinator,
            lease,
            brief,
            world_result,
            characters_result,
            outline_result,
            skills,
        )

    def _select_skills(self, lease: RunLease, brief: StoryBrief) -> tuple[ActiveSkillPin, ...] | str:
        return select_planning_skills(self.catalog, lease, brief)

    def _block(self, lease: RunLease, reason_code: str) -> PlanningBlocked:
        self.coordinator.block_planning(lease, reason_code)
        return PlanningBlocked(reason_code)

    def _emit_stage(self, stage: str) -> None:
        if self.stage_event_sink is not None:
            self.stage_event_sink(stage)

    def _generate_world(
        self,
        lease: RunLease,
        brief: StoryBrief,
        skills: tuple[ActiveSkillPin, ...],
    ) -> WorldAdaptation | SchemaFailure:
        return generate_world_step(self.generator, self.tool_gateway, lease, brief, skills)

    def _generate_characters(
        self,
        lease: RunLease,
        brief: StoryBrief,
        world: WorldBible,
        skills: tuple[ActiveSkillPin, ...],
    ) -> CharacterAdaptation | SchemaFailure:
        return generate_cast_step(self.generator, self.tool_gateway, lease, brief, world, skills)

    def _generate_outline(
        self,
        lease: RunLease,
        brief: StoryBrief,
        world: WorldBible,
        characters: CharacterSet,
        skills: tuple[ActiveSkillPin, ...],
    ) -> OutlineAdaptation | SchemaFailure:
        return generate_outline_step(self.generator, self.tool_gateway, lease, brief, world, characters, skills)


def select_planning_skills(
    catalog: SkillCatalog,
    lease: RunLease,
    brief: StoryBrief,
) -> tuple[ActiveSkillPin, ...] | str:
    """Deterministically select the planning skills for the four intake stages."""
    selector = SkillSelector(catalog)
    pins: list[ActiveSkillPin] = []
    for stage in _PLANNING_STAGES:
        selection = selector.select(_descriptor(lease, brief, stage))
        match selection:
            case SkillSelection(primary=primary):
                pins.append(catalog.pin(lease.run_id, primary))
            case SkillPause(code=code):
                return f"skill_{code.value.lower()}"
            case unreachable:
                assert_never(unreachable)
    return tuple(pins)


def generate_world_step(
    generator: PlanningGenerator,
    gateway: ToolGateway | None,
    lease: RunLease,
    brief: StoryBrief,
    skills: tuple[ActiveSkillPin, ...],
) -> WorldAdaptation | SchemaFailure:
    """Run the world-generation planning step (direct or through the tool gateway)."""
    if gateway is None:
        return generator.generate_world(brief)
    result = gateway.invoke(
        workflow_tool_request(
            run_id=lease.run_id,
            task_id=None,
            tool_ref="world.generate@1.0.0",
            skill_ref=_skill_ref(skills, "world-building"),
            role=AgentRole.STORY_ARCHITECT,
            input_data=WorldGenerateInput(brief=brief),
        )
    )
    return _tool_adaptation(result, "world.generate@1.0.0")


def generate_cast_step(
    generator: PlanningGenerator,
    gateway: ToolGateway | None,
    lease: RunLease,
    brief: StoryBrief,
    world: WorldBible,
    skills: tuple[ActiveSkillPin, ...],
) -> CharacterAdaptation | SchemaFailure:
    """Run the cast-generation planning step (direct or through the tool gateway)."""
    if gateway is None:
        return generator.generate_characters(brief, world)
    result = gateway.invoke(
        workflow_tool_request(
            run_id=lease.run_id,
            task_id=None,
            tool_ref="character.generate@1.0.0",
            skill_ref=_skill_ref(skills, "cast-network"),
            role=AgentRole.CAST_DESIGNER,
            input_data=CastGenerateInput(brief=brief, world=world),
        )
    )
    return _tool_adaptation(result, "character.generate@1.0.0")


def generate_revise_cast_step(
    generator: PlanningGenerator,
    gateway: ToolGateway | None,
    lease: RunLease,
    brief: StoryBrief,
    world: WorldBible,
    existing: CharacterSet,
    target_names: tuple[str, ...],
    skills: tuple[ActiveSkillPin, ...],
) -> CharacterAdaptation | SchemaFailure:
    """Run the targeted cast-revision planning step (direct or through the tool gateway)."""
    if gateway is None:
        return generator.revise_characters(brief, world, existing, target_names)
    result = gateway.invoke(
        workflow_tool_request(
            run_id=lease.run_id,
            task_id=None,
            tool_ref="character.revise@1.0.0",
            skill_ref=_skill_ref(skills, "cast-network"),
            role=AgentRole.CAST_DESIGNER,
            input_data=CastReviseInput(
                brief=brief,
                world=world,
                existing=existing,
                target_names=target_names,
            ),
        )
    )
    return _tool_adaptation(result, "character.revise@1.0.0")


def generate_outline_step(
    generator: PlanningGenerator,
    gateway: ToolGateway | None,
    lease: RunLease,
    brief: StoryBrief,
    world: WorldBible,
    characters: CharacterSet,
    skills: tuple[ActiveSkillPin, ...],
) -> OutlineAdaptation | SchemaFailure:
    """Run the outline-generation planning step (direct or through the tool gateway)."""
    if gateway is None:
        return generator.generate_outline(brief, world, characters)
    result = gateway.invoke(
        workflow_tool_request(
            run_id=lease.run_id,
            task_id=None,
            tool_ref="outline.generate@1.0.0",
            skill_ref=_skill_ref(skills, "story-architecture"),
            role=AgentRole.OUTLINE_PLANNER,
            input_data=OutlineGenerateInput(brief=brief, world=world, characters=characters),
        )
    )
    return _tool_adaptation(result, "outline.generate@1.0.0")


def commit_plan(
    coordinator: PlanningCoordinator,
    lease: RunLease,
    brief: StoryBrief,
    world_result: WorldAdaptation | SchemaFailure,
    characters_result: CharacterAdaptation | SchemaFailure,
    outline_result: OutlineAdaptation | SchemaFailure,
    skills: tuple[ActiveSkillPin, ...],
) -> PlanningSucceeded | PlanningBlocked:
    """Validate the assembled package and atomically commit the plan (or block)."""
    match world_result:
        case WorldAdaptation(value=world):
            pass
        case SchemaFailure():
            coordinator.block_planning(lease, "world_schema_failure")
            return PlanningBlocked("world_schema_failure")
        case unreachable:
            assert_never(unreachable)
    match characters_result:
        case CharacterAdaptation(value=characters):
            pass
        case SchemaFailure():
            coordinator.block_planning(lease, "cast_schema_failure")
            return PlanningBlocked("cast_schema_failure")
        case unreachable:
            assert_never(unreachable)
    match outline_result:
        case OutlineAdaptation(value=plan):
            pass
        case SchemaFailure():
            coordinator.block_planning(lease, "outline_schema_failure")
            return PlanningBlocked("outline_schema_failure")
        case unreachable:
            assert_never(unreachable)
    reason_code = _validate_graph(brief, world, characters, plan)
    if reason_code is not None:
        coordinator.block_planning(lease, reason_code)
        return PlanningBlocked(reason_code)
    receipt = coordinator.commit_validated_plan(
        lease,
        PlanningCommitRequest(
            run_id=lease.run_id,
            project_id=brief.project_id,
            brief=brief,
            world=world_result,
            characters=characters_result,
            outline=outline_result,
            skills=skills,
        ),
    )
    return PlanningSucceeded(receipt.plan_id, receipt.first_task_id, receipt.plan_revision)


def _skill_ref(pins: tuple[ActiveSkillPin, ...], skill_id: str) -> str:
    for pin in pins:
        if pin.manifest.skill_id == skill_id:
            return f"{pin.manifest.skill_id}@{pin.manifest.version}"
    return f"{skill_id}@1.0.0"


def _tool_adaptation(
    result: ToolCallResult,
    tool_ref: str,
) -> WorldAdaptation | CharacterAdaptation | OutlineAdaptation | SchemaFailure:
    if is_schema_invalid(result):
        return SchemaFailure(
            raw_artifact=RawCreativeArtifact(
                kind=CreativeArtifactKind.WORLD,
                raw_bytes=b"",
                content_sha256=hashlib.sha256(b"").hexdigest(),
                adapter_version="workflow-gateway-v1",
            ),
            expected_contract="creative_artifact",
            parse_routes=(CreativeParseRoute.STRICT_JSON, CreativeParseRoute.JSON_REPAIR),
        )
    unwrapped = unwrap_tool_result(result, tool_ref)
    adaptation = getattr(unwrapped, "adaptation", None)
    if adaptation is None:
        raise RuntimeError(f"tool {tool_ref} returned no adaptation payload")
    return adaptation


def _descriptor(lease: RunLease, brief: StoryBrief, stage: _PlanningStage) -> TaskDescriptor:
    return TaskDescriptor(
        run_id=lease.run_id,
        plan_revision=1,
        task_id=f"{lease.run_id}:planning:{stage.suffix}",
        task_type=stage.task_type,
        run_state=RunStatus.RUNNING,
        actor_role=stage.role,
        input_schema_version=1,
        expected_output_schema_version=1,
        available_context_slices=stage.context_slices,
        artifact_kinds=stage.artifact_kinds,
        genre_tags=frozenset((brief.genre,)),
        authorized_tool_ids=stage.tool_ids,
        remaining_context_tokens=stage.context_budget,
        remaining_tool_calls=3,
        remaining_repairs=1,
        remaining_timeout_seconds=180.0,
    )


def _validate_graph(
    brief: StoryBrief,
    world: WorldBible,
    characters: CharacterSet,
    plan: PlanContract,
) -> str | None:
    checks = (
        (world.brief_id == brief.brief_id, "world_brief_mismatch"),
        (characters.world_id == world.world_id, "cast_world_mismatch"),
        (plan.brief_id == brief.brief_id, "plan_brief_mismatch"),
        (plan.world_id == world.world_id, "plan_world_mismatch"),
        (plan.character_set_id == characters.character_set_id, "plan_cast_mismatch"),
        (len(plan.chapters) == brief.target_chapter_count, "chapter_count_mismatch"),
        (
            tuple(chapter.chapter_number for chapter in plan.chapters) == tuple(range(1, len(plan.chapters) + 1)),
            "chapter_order_invalid",
        ),
    )
    for valid, reason_code in checks:
        if not valid:
            return reason_code
    match plan.approval_status:
        case PlanApprovalStatus.VALIDATED:
            return None
        case PlanApprovalStatus.DRAFT | PlanApprovalStatus.NEEDS_REVIEW:
            return "plan_not_validated"
        case unreachable:
            assert_never(unreachable)


__all__ = (
    "PlanningBlocked",
    "PlanningCoordinator",
    "PlanningGenerator",
    "PlanningPolicy",
    "PlanningSucceeded",
    "PlanningWorkflow",
)
