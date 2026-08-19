from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from novel_agent.pathing import ArtifactId, ProjectId


class RevisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChapterRevisionRef(RevisionModel):
    artifact_id: ArtifactId
    revision: int = Field(ge=1)
    chapter_number: int = Field(ge=1)


class ChapterReplacementRequest(RevisionModel):
    project_id: ProjectId
    run_id: str = Field(min_length=1)
    source: ChapterRevisionRef
    replacement: ChapterRevisionRef

    @model_validator(mode="after")
    def validate_target(self) -> ChapterReplacementRequest:
        same_chapter = (
            self.source.artifact_id == self.replacement.artifact_id
            and self.source.chapter_number == self.replacement.chapter_number
        )
        if not same_chapter or self.source.revision == self.replacement.revision:
            raise PydanticCustomError(
                "chapter_replacement_target",
                "replacement must be a distinct revision of the same chapter",
            )
        return self


class RebaseImpact(RevisionModel):
    affected_chapters: tuple[ChapterRevisionRef, ...]
    affected_m3_memory_ids: tuple[str, ...]
    affected_plan_memory_ids: tuple[str, ...]
    invalidated_context_layers: tuple[Literal["m4_retrieval", "m5_working"], ...]


@dataclass(frozen=True, slots=True)
class RebaseApplied:
    proposal_id: str
    branch_id: str
    replacement_count: int
    invalidated_m3_memory_ids: tuple[str, ...]
    invalidated_plan_memory_ids: tuple[str, ...]


class RebaseProposal(RevisionModel):
    proposal_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    project_id: ProjectId
    run_id: str = Field(min_length=1)
    source: ChapterRevisionRef
    replacement: ChapterRevisionRef
    impact: RebaseImpact
    requires_downstream_rebase: bool
    created_at: datetime

    def applied(self) -> RebaseApplied:
        return RebaseApplied(
            proposal_id=self.proposal_id,
            branch_id=self.branch_id,
            replacement_count=len(self.impact.affected_chapters),
            invalidated_m3_memory_ids=self.impact.affected_m3_memory_ids,
            invalidated_plan_memory_ids=self.impact.affected_plan_memory_ids,
        )


class RebaseChapter(RevisionModel):
    source: ChapterRevisionRef
    replacement: ChapterRevisionRef

    @model_validator(mode="after")
    def validate_target(self) -> RebaseChapter:
        same_chapter = (
            self.source.artifact_id == self.replacement.artifact_id
            and self.source.chapter_number == self.replacement.chapter_number
        )
        if not same_chapter or self.source.revision == self.replacement.revision:
            raise PydanticCustomError(
                "chapter_replacement_target",
                "replacement must be a distinct revision of the same chapter",
            )
        return self


class RebaseConfirmation(RevisionModel):
    project_id: ProjectId
    run_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    user_confirmed: Literal[True]
    replacements: tuple[RebaseChapter, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distinct_sources(self) -> RebaseConfirmation:
        source_numbers = tuple(item.source.chapter_number for item in self.replacements)
        if len(source_numbers) != len(set(source_numbers)):
            raise PydanticCustomError("rebase_duplicate_chapter", "rebase chain contains duplicate chapters")
        return self


@dataclass(slots=True)  # contextlib assigns tracebacks while rolling back transactions
class ChapterRevisionError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


__all__ = (
    "ChapterReplacementRequest",
    "ChapterRevisionError",
    "ChapterRevisionRef",
    "RebaseApplied",
    "RebaseChapter",
    "RebaseConfirmation",
    "RebaseImpact",
    "RebaseProposal",
)
