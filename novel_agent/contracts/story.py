from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import field_validator, model_validator

from novel_agent.contracts.common import (
    ContractErrorCode,
    ContractModel,
    EvidenceBackedContract,
    PlanApprovalStatus,
    Provenance,
    contract_error,
)
from novel_agent.pathing import ProjectId


class StoryBrief(ContractModel):
    brief_id: str
    project_id: ProjectId
    user_requirements: str
    genre: str
    target_chapter_count: int
    created_at: datetime
    provenance: Provenance

    @field_validator("target_chapter_count")
    @classmethod
    def validate_target_chapter_count(cls, value: int) -> int:
        if value < 1:
            raise contract_error(ContractErrorCode.INVALID_CONTEXT_BUDGET, "target chapter count must be positive")
        return value


class WorldBible(ContractModel):
    world_id: str
    brief_id: str
    title: str
    premise: str
    rules: tuple[str, ...]
    constraints: tuple[str, ...]
    created_at: datetime
    provenance: Provenance


class CharacterCard(ContractModel):
    character_id: str
    name: str
    role: str
    goal: str
    traits: tuple[str, ...]
    created_at: datetime
    provenance: Provenance


class CharacterSet(ContractModel):
    character_set_id: str
    world_id: str
    characters: tuple[CharacterCard, ...]
    created_at: datetime
    provenance: Provenance

    @model_validator(mode="after")
    def validate_character_ids(self) -> Self:
        character_ids = tuple(character.character_id for character in self.characters)
        if len(character_ids) != len(frozenset(character_ids)):
            raise contract_error(ContractErrorCode.DUPLICATE_CHAPTER, "character identifiers must be unique")
        return self


class ChapterPlan(EvidenceBackedContract):
    chapter_plan_id: str
    chapter_number: int
    title: str
    goal: str
    planned_events: tuple[str, ...]
    created_at: datetime
    provenance: Provenance

    @field_validator("chapter_number")
    @classmethod
    def validate_chapter_number(cls, value: int) -> int:
        if value < 1:
            raise contract_error(ContractErrorCode.INVALID_CONTEXT_BUDGET, "chapter number must be positive")
        return value


class PlanContract(ContractModel):
    plan_id: str
    brief_id: str
    world_id: str
    character_set_id: str
    approval_status: PlanApprovalStatus
    chapters: tuple[ChapterPlan, ...]
    created_at: datetime
    provenance: Provenance

    @model_validator(mode="after")
    def validate_chapter_numbers(self) -> Self:
        chapter_numbers = tuple(chapter.chapter_number for chapter in self.chapters)
        if not chapter_numbers or len(chapter_numbers) != len(frozenset(chapter_numbers)):
            raise contract_error(ContractErrorCode.DUPLICATE_CHAPTER, "chapter numbers must be unique and nonempty")
        return self
