from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, assert_never

from pydantic import ValidationError

from novel_agent.adapters import ChapterAdaptation, SchemaFailure
from novel_agent.chapter_lifecycle import (
    context_request,
    pin_chapter_skills,
    quality_report_request,
    quality_request,
    repair_brief,
)
from novel_agent.chapter_models import (
    ChapterAcceptanceBlocked,
    ChapterAcceptanceReceipt,
    ChapterAcceptanceRequest,
    ChapterCandidateStageRequest,
    ChapterCoordinator,
    ChapterDraftRequest,
    ChapterExecutionCancelled,
    ChapterExecutionPolicy,
    ChapterExecutionRequest,
    ChapterExecutionSucceeded,
    ChapterExecutionWaitingReview,
    ChapterPauseRequest,
    ChapterStageRequest,
    ContinuityAuditReport,
    ContinuityAuditRequest,
    MemoryExtractionRequest,
    RepairBrief,
    StagedChapter,
)
from novel_agent.context import ContextBuilder
from novel_agent.contracts import ChapterPlan, ToolOutcome
from novel_agent.memory_models import MemoryProposal
from novel_agent.memory_store import MemoryGovernanceError
from novel_agent.pathing import ProjectPathResolver
from novel_agent.provider_types import ProviderFailureKind, ProviderTransportError
from novel_agent.quality import ChapterQualityRequest, QualityDecision, QualityOutcome
from novel_agent.run_models import RunLease
from novel_agent.skills import SkillCatalog
from novel_agent.tool_models import AgentRole, ToolInvocationRequest
from novel_agent.tool_registry import ToolCallResult, ToolGateway
from novel_agent.workflow_tools import (
    ChapterDraftInput,
    ContinuityAuditInput,
    MemoryExtractInput,
    QualityReviewInput,
    is_schema_invalid,
    unwrap_tool_result,
    workflow_tool_request,
)


class ChapterGenerator(Protocol):
    def generate(self, request: ChapterDraftRequest) -> ChapterAdaptation | SchemaFailure: ...


class MemoryExtractor(Protocol):
    def extract(self, request: MemoryExtractionRequest) -> tuple[MemoryProposal, ...]: ...


class ContinuityAuditor(Protocol):
    def audit(self, request: ContinuityAuditRequest) -> ContinuityAuditReport: ...


class QualityReviewer(Protocol):
    def review(self, request: ChapterQualityRequest) -> QualityDecision: ...


@dataclass(frozen=True, slots=True)
class ChapterExecutionWorkflow:
    coordinator: ChapterCoordinator
    context_builder: ContextBuilder
    catalog: SkillCatalog
    generator: ChapterGenerator
    extractor: MemoryExtractor
    continuity_auditor: ContinuityAuditor
    quality_reviewer: QualityReviewer
    path_resolver: ProjectPathResolver
    policy: ChapterExecutionPolicy = ChapterExecutionPolicy()
    tool_gateway: ToolGateway | None = None
    stage_event_sink: Callable[[str], None] | None = None

    def execute(
        self,
        lease: RunLease,
        request: ChapterExecutionRequest,
    ) -> ChapterExecutionSucceeded | ChapterExecutionWaitingReview | ChapterExecutionCancelled:
        first_task_id = self._task_id(lease.run_id, request.plan.chapters[0])
        if self.coordinator.cancellation_requested(lease):
            self.coordinator.cancel_chapter(lease, first_task_id)
            return ChapterExecutionCancelled(first_task_id)
        if len(request.plan.chapters) > self.policy.maximum_automatic_chapters:
            return self._pause(lease, first_task_id, "chapter_budget_exceeded")
        completed = 0
        for index, chapter in enumerate(request.plan.chapters):
            task_id = self._task_id(lease.run_id, chapter)
            self.coordinator.activate_chapter_task(lease, task_id)
            next_chapter = request.plan.chapters[index + 1] if index + 1 < len(request.plan.chapters) else None
            result = self._execute_chapter(lease, request, chapter, next_chapter, task_id)
            match result:
                case ChapterExecutionSucceeded():
                    completed += 1
                case ChapterExecutionWaitingReview() | ChapterExecutionCancelled():
                    return result
                case unreachable:
                    assert_never(unreachable)
        return ChapterExecutionSucceeded(completed)

    def _execute_chapter(
        self,
        lease: RunLease,
        request: ChapterExecutionRequest,
        chapter: ChapterPlan,
        next_chapter: ChapterPlan | None,
        task_id: str,
    ) -> ChapterExecutionSucceeded | ChapterExecutionWaitingReview | ChapterExecutionCancelled:
        return execute_single_chapter(self, lease, request, chapter, next_chapter, task_id)

    # -- gateway-aware tool calls --------------------------------------------

    def _quality_decision(
        self,
        lease: RunLease,
        request: ChapterExecutionRequest,
        chapter: ChapterPlan,
        staged: StagedChapter,
        continuity: ContinuityAuditReport,
        repair_attempt: int,
        task_id: str,
        skills: tuple[ActiveSkillPin, ...],
    ) -> QualityDecision:
        quality = quality_request(
            request,
            chapter,
            staged,
            continuity.located_beats,
            continuity.hard_conflicts,
            repair_attempt,
        )
        if continuity.hard_conflicts:
            return QualityDecision(
                outcome=QualityOutcome.PAUSE,
                score=None,
                reason_codes=("continuity_hard_conflict",),
                evaluator_attempts=0,
                policy_version="chapter-execution-v1",
                metrics=None,
            )
        try:
            return self._review_quality(lease, quality, task_id, skills)
        except ValidationError:
            return QualityDecision(
                outcome=QualityOutcome.EVALUATOR_ERROR,
                score=None,
                reason_codes=("evaluator_schema_failure",),
                evaluator_attempts=1,
                policy_version="chapter-execution-v1",
                metrics=None,
            )

    def _pause(self, lease: RunLease, task_id: str, reason: str) -> ChapterExecutionWaitingReview:
        self.coordinator.pause_chapter(lease, ChapterPauseRequest(task_id, reason))
        return ChapterExecutionWaitingReview(reason)

    def _emit_stage(self, stage: str) -> None:
        if self.stage_event_sink is not None:
            self.stage_event_sink(stage)

    # -- gateway-aware tool calls --------------------------------------------

    def _generate_draft(
        self,
        lease: RunLease,
        request: ChapterExecutionRequest,
        chapter: ChapterPlan,
        context: ContextBuildResult,
        repair_brief: RepairBrief | None,
        task_id: str,
        skills: tuple[ActiveSkillPin, ...],
    ) -> ChapterAdaptation | SchemaFailure:
        draft_request = ChapterDraftRequest(
            lease.run_id,
            task_id,
            chapter,
            context,
            request.target_chinese_characters,
            repair_brief,
        )
        if self.tool_gateway is None:
            return self.generator.generate(draft_request)
        repair = repair_brief is not None
        result = self.tool_gateway.invoke(
            workflow_tool_request(
                run_id=lease.run_id,
                task_id=task_id,
                tool_ref="chapter.repair@1.0.0" if repair else "chapter.draft@1.0.0",
                skill_ref=_chapter_skill_ref(skills, "targeted-revision" if repair else "chapter-scene"),
                role=AgentRole.CHAPTER_EXECUTOR,
                input_data=ChapterDraftInput(request=draft_request),
            )
        )
        if is_schema_invalid(result):
            raise ValidationError.from_exception_data("chapter_draft", [])
        unwrapped = unwrap_tool_result(result, "chapter.draft@1.0.0")
        adaptation = getattr(unwrapped, "adaptation", None)
        if adaptation is None:
            raise RuntimeError("chapter draft tool returned no adaptation payload")
        return adaptation

    def _extract_memory(
        self,
        lease: RunLease,
        request: MemoryExtractionRequest,
        task_id: str,
        skills: tuple[ActiveSkillPin, ...],
    ) -> tuple[MemoryProposal, ...]:
        if self.tool_gateway is None:
            return self.extractor.extract(request)
        result = self.tool_gateway.invoke(
            workflow_tool_request(
                run_id=lease.run_id,
                task_id=task_id,
                tool_ref="memory.extract@1.0.0",
                skill_ref=_chapter_skill_ref(skills, "memory-extraction"),
                role=AgentRole.MEMORY_EXTRACTOR,
                input_data=MemoryExtractInput(request=request),
            )
        )
        if is_schema_invalid(result):
            raise ValidationError.from_exception_data("memory_extract", [])
        unwrapped = unwrap_tool_result(result, "memory.extract@1.0.0")
        return tuple(getattr(unwrapped, "proposals", ()))

    def _audit_continuity(
        self,
        lease: RunLease,
        request: ContinuityAuditRequest,
        task_id: str,
        skills: tuple[ActiveSkillPin, ...],
    ) -> ContinuityAuditReport:
        if self.tool_gateway is None:
            return self.continuity_auditor.audit(request)
        result = self.tool_gateway.invoke(
            workflow_tool_request(
                run_id=lease.run_id,
                task_id=task_id,
                tool_ref="continuity.audit@1.0.0",
                skill_ref=_chapter_skill_ref(skills, "continuity-audit"),
                role=AgentRole.CONTINUITY_AUDITOR,
                input_data=ContinuityAuditInput(request=request),
            )
        )
        if is_schema_invalid(result):
            raise ValidationError.from_exception_data("continuity_audit", [])
        unwrapped = unwrap_tool_result(result, "continuity.audit@1.0.0")
        report = getattr(unwrapped, "report", None)
        if report is None:
            raise RuntimeError("continuity audit tool returned no report")
        return report

    def _review_quality(
        self,
        lease: RunLease,
        request: ChapterQualityRequest,
        task_id: str,
        skills: tuple[ActiveSkillPin, ...],
    ) -> QualityDecision:
        if self.tool_gateway is None:
            return self.quality_reviewer.review(request)
        result = self.tool_gateway.invoke(
            workflow_tool_request(
                run_id=lease.run_id,
                task_id=task_id,
                tool_ref="quality.evaluate@1.0.0",
                skill_ref=_chapter_skill_ref(skills, "quality-review"),
                role=AgentRole.QUALITY_REVIEWER,
                input_data=QualityReviewInput(request=request),
            )
        )
        if is_schema_invalid(result):
            raise ValidationError.from_exception_data("quality_review", [])
        unwrapped = unwrap_tool_result(result, "quality.evaluate@1.0.0")
        decision = getattr(unwrapped, "decision", None)
        if decision is None:
            raise RuntimeError("quality tool returned no decision")
        return decision

    @staticmethod
    def _task_id(run_id: str, chapter: ChapterPlan) -> str:
        return f"{run_id}:chapter:{chapter.chapter_plan_id}"


__all__ = (
    "ChapterDraftRequest",
    "ChapterExecutionCancelled",
    "ChapterExecutionPolicy",
    "ChapterExecutionRequest",
    "ChapterExecutionSucceeded",
    "ChapterExecutionWaitingReview",
    "ChapterExecutionWorkflow",
    "ContinuityAuditReport",
    "ContinuityAuditRequest",
    "MemoryExtractionRequest",
    "execute_single_chapter",
)


def execute_single_chapter(
    workflow: ChapterExecutionWorkflow,
    lease: RunLease,
    request: ChapterExecutionRequest,
    chapter: ChapterPlan,
    next_chapter: ChapterPlan | None,
    task_id: str,
) -> ChapterExecutionSucceeded | ChapterExecutionWaitingReview | ChapterExecutionCancelled:
    """Run the full gated lifecycle of exactly one chapter.

    Extracted from ``ChapterExecutionWorkflow._execute_chapter`` so a task-graph
    node executor can drive chapters one at a time; the workflow's own loop
    delegates here, keeping both paths byte-identical.
    """
    context = workflow.context_builder.build(
        context_request(lease.run_id, request, chapter, workflow.policy.context_budget)
    )
    workflow._emit_stage("chapter_prepare_complete")
    current_brief: RepairBrief | None = None
    repair_attempt = 0
    while True:
        if workflow.coordinator.cancellation_requested(lease):
            workflow.coordinator.cancel_chapter(lease, task_id)
            return ChapterExecutionCancelled(task_id)
        skills = pin_chapter_skills(workflow.catalog, lease.run_id, current_brief is not None)
        try:
            generated = workflow._generate_draft(lease, request, chapter, context, current_brief, task_id, skills)
        except ValidationError:
            return workflow._pause(lease, task_id, "chapter_schema_failure")
        match generated:
            case ChapterAdaptation() as adaptation:
                pass
            case SchemaFailure():
                return workflow._pause(lease, task_id, "chapter_schema_failure")
            case unreachable:
                assert_never(unreachable)
        staged = workflow.coordinator.stage_chapter(
            lease,
            ChapterStageRequest(
                request.project_id,
                lease.run_id,
                task_id,
                request.plan.plan_id,
                chapter,
                adaptation,
                context.digest,
                skills,
                repair_attempt + 1,
            ),
        )
        workflow._emit_stage("chapter_repair_staged" if current_brief is not None else "chapter_draft_staged")
        try:
            proposals = workflow._extract_memory(
                lease,
                MemoryExtractionRequest(lease.run_id, request.plan_revision, chapter, staged),
                task_id,
                skills,
            )
            workflow.coordinator.stage_memory_candidates(
                lease,
                ChapterCandidateStageRequest(request.project_id, lease.run_id, staged, proposals),
            )
        except (MemoryGovernanceError, ValidationError):
            return workflow._pause(lease, task_id, "memory_schema_failure")
        workflow._emit_stage("chapter_continuity_checked")
        try:
            continuity = workflow._audit_continuity(
                lease,
                ContinuityAuditRequest(chapter, staged, context),
                task_id,
                skills,
            )
        except ValidationError:
            return workflow._pause(lease, task_id, "continuity_schema_failure")
        decision = workflow._quality_decision(
            lease,
            request,
            chapter,
            staged,
            continuity,
            repair_attempt,
            task_id,
            skills,
        )
        workflow._emit_stage("chapter_quality_reviewed")
        workflow.coordinator.record_chapter_quality(
            lease,
            quality_report_request(lease.run_id, task_id, staged, decision),
        )
        match decision.outcome:
            case QualityOutcome.ACCEPT:
                if workflow.coordinator.cancellation_requested(lease):
                    workflow.coordinator.cancel_chapter(lease, task_id)
                    return ChapterExecutionCancelled(task_id)
                if not request.auto_approve:
                    # Approval mode: keep the chapter staged and pause the run
                    # so the console user can review, edit, or regenerate it
                    # before it is persisted to the canonical stores.
                    return workflow._pause(lease, task_id, "artifact_approval_required")
                receipt = workflow.coordinator.accept_chapter(
                    lease,
                    ChapterAcceptanceRequest(
                        request.project_id,
                        lease.run_id,
                        task_id,
                        request.plan.plan_id,
                        request.plan_revision,
                        chapter,
                        staged,
                        next_chapter,
                        context.digest,
                        skills,
                        decision,
                        activate=request.activate,
                    ),
                    workflow.path_resolver,
                )
                match receipt:
                    case ChapterAcceptanceBlocked(reason_code=reason):
                        return workflow._pause(lease, task_id, reason)
                    case ChapterAcceptanceReceipt():
                        workflow._emit_stage("chapter_committed")
                        return ChapterExecutionSucceeded(1)
                    case unreachable:
                        assert_never(unreachable)
            case QualityOutcome.REPAIR:
                if repair_attempt >= workflow.policy.maximum_repairs:
                    return workflow._pause(lease, task_id, "repair_budget_exhausted")
                repair_attempt += 1
                current_brief = repair_brief(chapter, repair_attempt, decision.reason_codes, context, staged)
            case QualityOutcome.PAUSE | QualityOutcome.EVALUATOR_ERROR:
                reason = decision.reason_codes[0] if decision.reason_codes else "quality_review_failed"
                return workflow._pause(lease, task_id, reason)
            case unreachable:
                assert_never(unreachable)


def _chapter_skill_ref(pins: tuple[ActiveSkillPin, ...], skill_id: str) -> str:
    for pin in pins:
        if pin.manifest.skill_id == skill_id:
            return f"{pin.manifest.skill_id}@{pin.manifest.version}"
    return f"{skill_id}@1.0.0"
