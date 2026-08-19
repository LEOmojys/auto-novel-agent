from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import assert_never

from novel_agent.adapters.models import (
    ChapterAdaptation,
    ChapterDraft,
    CharacterAdaptation,
    OutlineAdaptation,
    SchemaFailure,
    WorldAdaptation,
)
from novel_agent.adapters.structured import SchemaRepairer, StructuredCreativeAdapter
from novel_agent.contracts import CharacterSet, PlanContract, WorldBible
from novel_agent.contracts.common import ContractModel


def _append_contract(instruction: str, contract_type: type[ContractModel]) -> str:
    schema = json.dumps(contract_type.model_json_schema(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"{instruction}\n\nReturn only JSON that validates against this governed contract schema: {schema}"


class WorldGenerationRequest(ContractModel):
    idea: str
    world_id: str
    brief_id: str
    title: str
    created_at: datetime
    source_id: str


class CharacterGenerationRequest(ContractModel):
    project: str
    world_name: str
    requirements: str
    count: int
    character_set_id: str
    world_id: str
    created_at: datetime
    source_id: str


class OutlineGenerationRequest(ContractModel):
    project: str
    world_name: str
    preferences: str
    chapter_count: int
    plan_id: str
    brief_id: str
    world_id: str
    character_set_id: str
    created_at: datetime
    source_id: str


class ChapterGenerationRequest(ContractModel):
    last_passage: str
    this_passage_request: str
    article_name: str
    chapter_number: str
    chapter_name: str
    characters: tuple[str, ...]
    world: str
    styles: str
    words: int
    is_npc: bool
    project: str
    chapter_draft_id: str
    chapter_plan_id: str
    created_at: datetime
    source_id: str


@dataclass(frozen=True, slots=True)
class LegacyCreativeAdapter:
    structured: StructuredCreativeAdapter
    repairer: SchemaRepairer | None = None

    def __post_init__(self) -> None:
        if self.repairer is not None and self.structured.repairer is None:
            object.__setattr__(
                self,
                "structured",
                StructuredCreativeAdapter(
                    repairer=self.repairer,
                    adapter_version=self.structured.adapter_version,
                ),
            )

    def generate_world(self, request: WorldGenerationRequest) -> WorldAdaptation | SchemaFailure:
        from generate_world import expand_world

        raw_text = expand_world(_append_contract(request.idea, WorldBible))
        parsed = self.structured.adapt_world(raw_text)
        match parsed:
            case WorldAdaptation():
                return parsed
            case SchemaFailure():
                return parsed
            case unreachable:
                assert_never(unreachable)

    def generate_characters(self, request: CharacterGenerationRequest) -> CharacterAdaptation | SchemaFailure:
        from generate_characters import CHARACTER_PROMPT, access_deepseek, read_world

        prompt = CHARACTER_PROMPT.format(
            world_text=read_world(request.world_name, request.project),
            user_requirements=_append_contract(request.requirements, CharacterSet),
            count=request.count,
        )
        raw_text = access_deepseek(prompt, model_name="deepseek-v4-pro", max_retry=1)
        parsed = self.structured.adapt_characters(raw_text)
        match parsed:
            case CharacterAdaptation():
                return parsed
            case SchemaFailure():
                return parsed
            case unreachable:
                assert_never(unreachable)

    def generate_outline(self, request: OutlineGenerationRequest) -> OutlineAdaptation | SchemaFailure:
        from generate_outline import generate_outline

        raw_text = generate_outline(
            request.project,
            request.world_name,
            _append_contract(request.preferences, PlanContract),
            request.chapter_count,
        )
        parsed = self.structured.adapt_outline(raw_text)
        match parsed:
            case OutlineAdaptation():
                return parsed
            case SchemaFailure():
                return parsed
            case unreachable:
                assert_never(unreachable)

    def generate_chapter(self, request: ChapterGenerationRequest) -> ChapterAdaptation | SchemaFailure:
        from tools_writting import writting_novel

        raw_text = writting_novel(
            request.last_passage,
            _append_contract(request.this_passage_request, ChapterDraft),
            request.article_name,
            request.chapter_number,
            request.chapter_name,
            list(request.characters),
            request.world,
            request.styles,
            request.words,
            request.is_npc,
            request.project,
        )
        parsed = self.structured.adapt_chapter(raw_text)
        match parsed:
            case ChapterAdaptation():
                return parsed
            case SchemaFailure():
                return parsed
            case unreachable:
                assert_never(unreachable)


__all__ = (
    "ChapterGenerationRequest",
    "CharacterGenerationRequest",
    "LegacyCreativeAdapter",
    "OutlineGenerationRequest",
    "WorldGenerationRequest",
)
