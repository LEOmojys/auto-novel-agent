from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from novel_agent.adapters import ChapterAdaptation
from novel_agent.context_models import ContextBudget, ContextBuildResult
from novel_agent.contracts import ChapterPlan, PlanContract
from novel_agent.memory_models import MemoryProposal
from novel_agent.pathing import ArtifactId, ProjectId, ProjectPathResolver
from novel_agent.quality import QualityDecision
from novel_agent.run_models import RunLease
from novel_agent.skill_manifests.models import ActiveSkillPin


class ChapterExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: ProjectId
    plan: PlanContract
    plan_revision: int = Field(ge=1)
    target_chinese_characters: int = Field(gt=0)
    genre: str = Field(min_length=1)
    style_guidance: tuple[str, ...] = ()
    activate: bool = True
    auto_approve: bool = True


class ChapterExecutionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_budget: ContextBudget = ContextBudget()
    maximum_automatic_chapters: int = Field(default=100, ge=1)
    maximum_repairs: int = Field(default=2, ge=0, le=2)


class ContinuityAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hard_conflicts: tuple[str, ...] = ()
    located_beats: tuple[str, ...] = ()


class RepairBrief(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chapter_plan: ChapterPlan
    repair_attempt: int = Field(ge=1, le=2)
    failure_codes: tuple[str, ...]
    context_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_artifact_id: ArtifactId
    evidence_artifact_revision: int = Field(ge=1)
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class StagedChapter:
    artifact_id: ArtifactId
    revision: int
    task_id: str
    chapter_plan_id: str
    chapter_number: int
    raw_bytes: bytes
    raw_sha256: str
    body: str
    adapter_version: str = "not-recorded"
    parse_route: str = "not-recorded"
    generation_attempt: int = 1


@dataclass(frozen=True, slots=True)
class ChapterDraftRequest:
    run_id: str
    task_id: str
    chapter_plan: ChapterPlan
    context: ContextBuildResult
    target_chinese_characters: int
    repair_brief: RepairBrief | None


@dataclass(frozen=True, slots=True)
class MemoryExtractionRequest:
    run_id: str
    plan_revision: int
    chapter_plan: ChapterPlan
    staged: StagedChapter


@dataclass(frozen=True, slots=True)
class ContinuityAuditRequest:
    chapter_plan: ChapterPlan
    staged: StagedChapter
    context: ContextBuildResult


@dataclass(frozen=True, slots=True)
class ChapterStageRequest:
    project_id: ProjectId
    run_id: str
    task_id: str
    plan_id: str
    chapter_plan: ChapterPlan
    adaptation: ChapterAdaptation
    context_digest: str
    skills: tuple[ActiveSkillPin, ...]
    attempt: int


@dataclass(frozen=True, slots=True)
class ChapterCandidateStageRequest:
    project_id: ProjectId
    run_id: str
    staged: StagedChapter
    proposals: tuple[MemoryProposal, ...]


@dataclass(frozen=True, slots=True)
class ChapterQualityReportRequest:
    run_id: str
    task_id: str
    staged: StagedChapter
    decision: QualityDecision


@dataclass(frozen=True, slots=True)
class ChapterAcceptanceRequest:
    project_id: ProjectId
    run_id: str
    task_id: str
    plan_id: str
    plan_revision: int
    chapter_plan: ChapterPlan
    staged: StagedChapter
    next_chapter: ChapterPlan | None
    context_digest: str
    skills: tuple[ActiveSkillPin, ...]
    decision: QualityDecision
    activate: bool = True


@dataclass(frozen=True, slots=True)
class ChapterPauseRequest:
    task_id: str
    reason_code: str


@dataclass(frozen=True, slots=True)
class ChapterAcceptanceReceipt:
    artifact_id: ArtifactId
    revision: int
    next_task_id: str | None
    promoted_memory_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChapterAcceptanceBlocked:
    reason_code: str


@dataclass(frozen=True, slots=True)
class ChapterExecutionSucceeded:
    completed_chapters: int


@dataclass(frozen=True, slots=True)
class ChapterExecutionWaitingReview:
    reason_code: str


@dataclass(frozen=True, slots=True)
class ChapterExecutionCancelled:
    task_id: str


class ChapterCoordinator(Protocol):
    def activate_chapter_task(self, lease: RunLease, task_id: str) -> None:
        ...

    def stage_chapter(self, lease: RunLease, request: ChapterStageRequest) -> StagedChapter:
        ...

    def stage_memory_candidates(self, lease: RunLease, request: ChapterCandidateStageRequest) -> tuple[str, ...]:
        ...

    def record_chapter_quality(self, lease: RunLease, request: ChapterQualityReportRequest) -> None:
        ...

    def accept_chapter(
        self,
        lease: RunLease,
        request: ChapterAcceptanceRequest,
        path_resolver: ProjectPathResolver,
    ) -> ChapterAcceptanceReceipt | ChapterAcceptanceBlocked:
        ...

    def pause_chapter(self, lease: RunLease, request: ChapterPauseRequest) -> None:
        ...

    def cancellation_requested(self, lease: RunLease) -> bool:
        ...

    def cancel_chapter(self, lease: RunLease, task_id: str) -> None:
        ...


__all__ = (
    "ChapterAcceptanceBlocked",
    "ChapterAcceptanceReceipt",
    "ChapterAcceptanceRequest",
    "ChapterCandidateStageRequest",
    "ChapterCoordinator",
    "ChapterDraftRequest",
    "ChapterExecutionCancelled",
    "ChapterExecutionPolicy",
    "ChapterExecutionRequest",
    "ChapterExecutionSucceeded",
    "ChapterExecutionWaitingReview",
    "ChapterPauseRequest",
    "ChapterQualityReportRequest",
    "ChapterStageRequest",
    "ContinuityAuditReport",
    "ContinuityAuditRequest",
    "MemoryExtractionRequest",
    "RepairBrief",
    "StagedChapter",
)
