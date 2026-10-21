from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from novel_agent.pathing import ProjectId


class ContextBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    total_tokens: int = Field(default=4096, ge=1)
    canonical_tokens: int = Field(default=1024, ge=0)
    plan_tokens: int = Field(default=1024, ge=0)
    continuity_tokens: int = Field(default=1024, ge=0)
    long_term_tokens: int = Field(default=768, ge=0)
    style_tokens: int = Field(default=256, ge=0)

    @model_validator(mode="after")
    def validate_category_limits(self) -> ContextBudget:
        limits = (
            self.canonical_tokens,
            self.plan_tokens,
            self.continuity_tokens,
            self.long_term_tokens,
            self.style_tokens,
        )
        if all(limit <= self.total_tokens for limit in limits):
            return self
        raise PydanticCustomError(
            "context_category_budget",
            "category budgets cannot exceed the total budget",
        )


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    project_id: ProjectId
    run_id: str = Field(min_length=1)
    plan_revision: int = Field(ge=0)
    chapter_plan_id: str = Field(min_length=1)
    chapter_number: int = Field(ge=1)
    direct_memory_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    hook_ids: tuple[str, ...] = ()
    location_ids: tuple[str, ...] = ()
    query_terms: tuple[str, ...] = ()
    style_guidance: tuple[str, ...] = ()
    budget: ContextBudget = ContextBudget()
    # "chapter" keeps the strict per-chapter view (no future chapters, only the
    # active plan revision).  "project" is the explicit-retrieval view used by
    # the agent's mem_query tool: it may read any confirmed chapter and any
    # plan revision because the caller asks for project knowledge directly.
    scope: Literal["chapter", "project"] = "chapter"


@dataclass(frozen=True, slots=True)
class ContextItem:
    memory_id: str
    content: str
    token_count: int
    category: str
    tier: int
    relevance: int
    narrative_time: int
    source_artifact_id: str
    source_artifact_revision: int
    artifact_hash: str
    source_hash: str
    fts_score: float = 0.0
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ContextExclusion:
    memory_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class BudgetLedger:
    total_tokens: int
    canonical_tokens: int
    plan_tokens: int
    continuity_tokens: int
    long_term_tokens: int
    style_tokens: int


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    items: tuple[ContextItem, ...]
    exclusions: tuple[ContextExclusion, ...]
    prompt: str
    digest: str
    tokenizer_version: str
    budget_ledger: BudgetLedger
    pack_id: str | None = None

    @property
    def memory_ids(self) -> tuple[str, ...]:
        return tuple(item.memory_id for item in self.items if not item.memory_id.startswith("style:"))


__all__ = (
    "BudgetLedger",
    "ContextBudget",
    "ContextBuildResult",
    "ContextExclusion",
    "ContextItem",
    "ContextRequest",
)
