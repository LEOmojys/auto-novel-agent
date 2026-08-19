"""Workflow tool registration: wire planning/chapter workflows through ToolGateway.

The governed creative workflows call generator/extractor/auditor/reviewer
protocols directly.  This module registers those same calls as typed
``ToolSpec`` entries so production runs emit ``tool_calls`` audit rows with
request/response hashes, role/state/skill authorization and idempotency —
the ToolGateway invocation graph required by the design (draft 3.1/3.5).

Workflows that do not receive a gateway (tests, embedded usage) keep their
direct protocol calls; behaviour is byte-identical in both paths.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from novel_agent.adapters import (
    ChapterAdaptation,
    CharacterAdaptation,
    OutlineAdaptation,
    SchemaFailure,
    WorldAdaptation,
)
from novel_agent.chapter_models import (
    ChapterDraftRequest,
    ContinuityAuditReport,
    ContinuityAuditRequest,
    MemoryExtractionRequest,
)
from novel_agent.contracts import (
    CharacterSet,
    RunStatus,
    StoryBrief,
    ToolOutcome,
    WorldBible,
)
from novel_agent.memory_models import MemoryProposal
from novel_agent.provider_types import ProviderFailureKind, ProviderTransportError
from novel_agent.quality import ChapterQualityRequest, QualityDecision
from novel_agent.provider_types import ProviderTransportError
from novel_agent.tool_models import (
    AccessLevel,
    AgentRole,
    ApprovalPolicy,
    ArtifactPolicy,
    RetryPolicy,
    ToolInvocationRequest,
    ToolSpec,
)
from novel_agent.tool_registry import ToolCallResult, ToolGateway, ToolRegistry

if TYPE_CHECKING:
    from novel_agent.chapter_executor import (
        ChapterGenerator,
        ContinuityAuditor,
        MemoryExtractor,
        QualityReviewer,
    )
    from novel_agent.chapter_models import ChapterDraftRequest, ChapterQualityRequest, ContinuityAuditRequest, MemoryExtractionRequest
    from novel_agent.planning import PlanningGenerator


class _ToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# -- planning tools ---------------------------------------------------------


class WorldGenerateInput(_ToolModel):
    brief: StoryBrief


class WorldGenerateOutput(_ToolModel):
    adaptation: WorldAdaptation | SchemaFailure


class CastGenerateInput(_ToolModel):
    brief: StoryBrief
    world: WorldBible


class CastGenerateOutput(_ToolModel):
    adaptation: CharacterAdaptation | SchemaFailure


class CastReviseInput(_ToolModel):
    brief: StoryBrief
    world: WorldBible
    existing: CharacterSet
    target_names: tuple[str, ...]


class CastReviseOutput(_ToolModel):
    adaptation: CharacterAdaptation | SchemaFailure


class OutlineGenerateInput(_ToolModel):
    brief: StoryBrief
    world: WorldBible
    characters: CharacterSet


class OutlineGenerateOutput(_ToolModel):
    adaptation: OutlineAdaptation | SchemaFailure


# -- chapter tools ----------------------------------------------------------


class ChapterDraftInput(_ToolModel):
    request: ChapterDraftRequest


class ChapterDraftOutput(_ToolModel):
    adaptation: ChapterAdaptation | SchemaFailure


class MemoryExtractInput(_ToolModel):
    request: MemoryExtractionRequest


class MemoryExtractOutput(_ToolModel):
    proposals: tuple[MemoryProposal, ...]


class ContinuityAuditInput(_ToolModel):
    request: ContinuityAuditRequest


class ContinuityAuditOutput(_ToolModel):
    report: ContinuityAuditReport


class QualityReviewInput(_ToolModel):
    request: ChapterQualityRequest


class QualityReviewOutput(_ToolModel):
    decision: QualityDecision


def register_workflow_tools(
    registry: ToolRegistry,
    *,
    planning_generator: PlanningGenerator,
    chapter_generator: ChapterGenerator,
    extractor: MemoryExtractor,
    auditor: ContinuityAuditor,
    reviewer: QualityReviewer | Callable[[], QualityReviewer],
) -> None:
    """Register the eight creative tools with handlers bound to the given components.

    ``reviewer`` may be a callable returning the current reviewer; production
    stages pass a holder-backed callable so the audited quality tool always
    matches the reviewer the workflow used for that chapter.
    """

    def _reviewer() -> QualityReviewer:
        return reviewer() if callable(reviewer) else reviewer
    registry.register(
        ToolSpec(
            tool_id="world.generate",
            version="1.0.0",
            input_model=WorldGenerateInput,
            output_model=WorldGenerateOutput,
            allowed_roles=frozenset((AgentRole.STORY_ARCHITECT,)),
            allowed_skill_ids=frozenset(("world-building@1.0.0",)),
            allowed_run_states=frozenset((RunStatus.RUNNING,)),
            access_level=AccessLevel.PROVIDER,
            timeout_seconds=300.0,
            retry_policy=RetryPolicy(max_attempts=2, retryable_error_codes=frozenset(("TRANSIENT",))),
            idempotent=True,
            approval_policy=ApprovalPolicy.NONE,
            artifact_policy=ArtifactPolicy.NONE,
        ),
        lambda _input: WorldGenerateOutput(adaptation=planning_generator.generate_world(_input.brief)),
    )
    registry.register(
        ToolSpec(
            tool_id="character.generate",
            version="1.0.0",
            input_model=CastGenerateInput,
            output_model=CastGenerateOutput,
            allowed_roles=frozenset((AgentRole.CAST_DESIGNER,)),
            allowed_skill_ids=frozenset(("cast-network@1.0.0",)),
            allowed_run_states=frozenset((RunStatus.RUNNING,)),
            access_level=AccessLevel.PROVIDER,
            timeout_seconds=300.0,
            retry_policy=RetryPolicy(max_attempts=2, retryable_error_codes=frozenset(("TRANSIENT",))),
            idempotent=True,
            approval_policy=ApprovalPolicy.NONE,
            artifact_policy=ArtifactPolicy.NONE,
        ),
        lambda _input: CastGenerateOutput(
            adaptation=planning_generator.generate_characters(_input.brief, _input.world)
        ),
    )
    registry.register(
        ToolSpec(
            tool_id="character.revise",
            version="1.0.0",
            input_model=CastReviseInput,
            output_model=CastReviseOutput,
            allowed_roles=frozenset((AgentRole.CAST_DESIGNER,)),
            allowed_skill_ids=frozenset(("cast-network@1.0.0",)),
            allowed_run_states=frozenset((RunStatus.RUNNING,)),
            access_level=AccessLevel.PROVIDER,
            timeout_seconds=300.0,
            retry_policy=RetryPolicy(max_attempts=2, retryable_error_codes=frozenset(("TRANSIENT",))),
            idempotent=True,
            approval_policy=ApprovalPolicy.NONE,
            artifact_policy=ArtifactPolicy.NONE,
        ),
        lambda _input: CastReviseOutput(
            adaptation=planning_generator.revise_characters(
                _input.brief,
                _input.world,
                _input.existing,
                _input.target_names,
            )
        ),
    )
    registry.register(
        ToolSpec(
            tool_id="outline.generate",
            version="1.0.0",
            input_model=OutlineGenerateInput,
            output_model=OutlineGenerateOutput,
            allowed_roles=frozenset((AgentRole.OUTLINE_PLANNER,)),
            allowed_skill_ids=frozenset(("story-architecture@1.0.0",)),
            allowed_run_states=frozenset((RunStatus.RUNNING,)),
            access_level=AccessLevel.PROVIDER,
            timeout_seconds=300.0,
            retry_policy=RetryPolicy(max_attempts=2, retryable_error_codes=frozenset(("TRANSIENT",))),
            idempotent=True,
            approval_policy=ApprovalPolicy.NONE,
            artifact_policy=ArtifactPolicy.NONE,
        ),
        lambda _input: OutlineGenerateOutput(
            adaptation=planning_generator.generate_outline(_input.brief, _input.world, _input.characters)
        ),
    )
    registry.register(
        ToolSpec(
            tool_id="chapter.draft",
            version="1.0.0",
            input_model=ChapterDraftInput,
            output_model=ChapterDraftOutput,
            allowed_roles=frozenset((AgentRole.CHAPTER_EXECUTOR,)),
            allowed_skill_ids=frozenset(("chapter-scene@1.0.0",)),
            allowed_run_states=frozenset((RunStatus.RUNNING,)),
            access_level=AccessLevel.STAGE_WRITE,
            timeout_seconds=300.0,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
            approval_policy=ApprovalPolicy.COORDINATOR,
            artifact_policy=ArtifactPolicy.STAGE_ONLY,
        ),
        lambda _input: ChapterDraftOutput(adaptation=chapter_generator.generate(_input.request)),
    )
    registry.register(
        ToolSpec(
            tool_id="chapter.repair",
            version="1.0.0",
            input_model=ChapterDraftInput,
            output_model=ChapterDraftOutput,
            allowed_roles=frozenset((AgentRole.CHAPTER_EXECUTOR,)),
            allowed_skill_ids=frozenset(("targeted-revision@1.0.0",)),
            allowed_run_states=frozenset((RunStatus.RUNNING,)),
            access_level=AccessLevel.STAGE_WRITE,
            timeout_seconds=300.0,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
            approval_policy=ApprovalPolicy.COORDINATOR,
            artifact_policy=ArtifactPolicy.STAGE_ONLY,
        ),
        lambda _input: ChapterDraftOutput(adaptation=chapter_generator.generate(_input.request)),
    )
    registry.register(
        ToolSpec(
            tool_id="memory.extract",
            version="1.0.0",
            input_model=MemoryExtractInput,
            output_model=MemoryExtractOutput,
            allowed_roles=frozenset((AgentRole.MEMORY_EXTRACTOR,)),
            allowed_skill_ids=frozenset(("memory-extraction@1.0.0",)),
            allowed_run_states=frozenset((RunStatus.RUNNING,)),
            access_level=AccessLevel.CANDIDATE_WRITE,
            timeout_seconds=120.0,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
            approval_policy=ApprovalPolicy.COORDINATOR,
            artifact_policy=ArtifactPolicy.CANDIDATE_ONLY,
        ),
        lambda _input: MemoryExtractOutput(proposals=extractor.extract(_input.request)),
    )
    registry.register(
        ToolSpec(
            tool_id="continuity.audit",
            version="1.0.0",
            input_model=ContinuityAuditInput,
            output_model=ContinuityAuditOutput,
            allowed_roles=frozenset((AgentRole.CONTINUITY_AUDITOR,)),
            allowed_skill_ids=frozenset(("continuity-audit@1.0.0",)),
            allowed_run_states=frozenset((RunStatus.RUNNING,)),
            access_level=AccessLevel.READ,
            timeout_seconds=120.0,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
            approval_policy=ApprovalPolicy.NONE,
            artifact_policy=ArtifactPolicy.NONE,
        ),
        lambda _input: ContinuityAuditOutput(report=auditor.audit(_input.request)),
    )
    registry.register(
        ToolSpec(
            tool_id="quality.evaluate",
            version="1.0.0",
            input_model=QualityReviewInput,
            output_model=QualityReviewOutput,
            allowed_roles=frozenset((AgentRole.QUALITY_REVIEWER,)),
            allowed_skill_ids=frozenset(("quality-review@1.0.0",)),
            allowed_run_states=frozenset((RunStatus.RUNNING,)),
            access_level=AccessLevel.READ,
            timeout_seconds=120.0,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
            approval_policy=ApprovalPolicy.NONE,
            artifact_policy=ArtifactPolicy.NONE,
        ),
        lambda _input: QualityReviewOutput(decision=_reviewer().review(_input.request)),
    )


def workflow_tool_request(
    *,
    run_id: str,
    task_id: str | None,
    tool_ref: str,
    skill_ref: str,
    role: AgentRole,
    input_data: BaseModel,
) -> ToolInvocationRequest:
    """Build a typed gateway envelope for one workflow tool call.

    The idempotency key includes the input hash so repeated chapters (or a
    crash-resumed run) reuse the original audit row instead of conflicting.
    Planning-stage calls have no durable task row yet and pass ``task_id=None``.
    """
    import hashlib

    input_hash = hashlib.sha256(input_data.model_dump_json().encode()).hexdigest()[:16]
    return ToolInvocationRequest(
        invocation_id=f"{run_id}:{task_id or 'planning'}:{tool_ref}:{skill_ref}:{input_hash}",
        run_id=run_id,
        task_id=task_id,
        tool_ref=tool_ref,
        idempotency_key=f"{run_id}:{tool_ref}:{skill_ref}:{input_hash}",
        actor_role=role,
        skill_ref=skill_ref,
        run_status=RunStatus.RUNNING,
        input_data=input_data,
        approval_granted=True,
    )


def unwrap_tool_result(result: ToolCallResult, tool_ref: str) -> BaseModel:
    if result.outcome is not ToolOutcome.SUCCEEDED:
        raise ProviderTransportError(
            kind=ProviderFailureKind.PERMANENT,
            detail=f"tool {tool_ref} failed: {result.reason}",
        )
    if result.data is None:
        raise ProviderTransportError(
            kind=ProviderFailureKind.PERMANENT,
            detail=f"tool {tool_ref} returned no data",
        )
    return result.data


def is_schema_invalid(result: ToolCallResult) -> bool:
    return result.outcome is not ToolOutcome.SUCCEEDED and result.reason == "SCHEMA_INVALID"


__all__ = (
    "CastGenerateInput",
    "CastGenerateOutput",
    "ChapterDraftInput",
    "ChapterDraftOutput",
    "ContinuityAuditInput",
    "ContinuityAuditOutput",
    "MemoryExtractInput",
    "MemoryExtractOutput",
    "OutlineGenerateInput",
    "OutlineGenerateOutput",
    "QualityReviewInput",
    "QualityReviewOutput",
    "WorldGenerateInput",
    "WorldGenerateOutput",
    "is_schema_invalid",
    "register_workflow_tools",
    "unwrap_tool_result",
    "workflow_tool_request",
)
