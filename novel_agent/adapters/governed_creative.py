"""Governed real-model creative pipeline.

Bridges the governed run workflows (PlanningGenerator / ChapterGenerator /
MemoryExtractor) to live provider calls.  The model is asked for *content
fields only*; identifiers, timestamps, provenance and approval status are
assembled by this adapter from request metadata, because those are system
facts that a model cannot reproduce (digests, ISO clocks, lease-scoped ids).

All provider I/O flows through ``ProviderTransport`` (the same wire layer the
``legacy_real`` stage uses), so token accounting and transport failures stay
inside the governed boundary.  This module never imports legacy writer tools;
the architecture guard test only exempts ``adapters/legacy.py`` for direct
legacy imports.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from json_repair import repair_json
from pydantic import BaseModel, ConfigDict, ValidationError

from novel_agent.adapters.governance import NovelSaveCandidateAdapter
from novel_agent.adapters.models import (
    ChapterAdaptation,
    ChapterDraft,
    CharacterAdaptation,
    CreativeArtifactKind,
    CreativeParseRoute,
    M0CandidateExtractionRequest,
    OutlineAdaptation,
    RawCreativeArtifact,
    SchemaFailure,
    WorldAdaptation,
)
from novel_agent.contracts import (
    CharacterCard,
    CharacterSet,
    ChapterPlan,
    EvidenceRef,
    MemoryLayer,
    PlanApprovalStatus,
    PlanContract,
    Provenance,
    ProvenanceKind,
    ProviderRequest,
    StoryBrief,
    WorldBible,
)
from novel_agent.memory_models import (
    EntityResolution,
    InferenceMode,
    MemoryFactKind,
    MemoryProposal,
    PromotionAuthority,
    TemporalCertainty,
)
from novel_agent.pathing import ProjectId
from novel_agent.provider_types import ProviderTransport

if TYPE_CHECKING:
    from novel_agent.chapter_models import ChapterDraftRequest, MemoryExtractionRequest

_DEFAULT_CHARACTER_COUNT: int = 5
_TEMPERATURE: float = 0.7

_WORLD_PROMPT: str = """你是一位资深世界观架构师。请根据下面的创作需求，设计一个完整的小说世界观。

## 创作需求
{requirements}

## 时代与题材
{genre}

请输出**内容字段 JSON**（不要输出任何其他文字），结构如下：
{{
  "title": "世界观名称（4-8字，中文）",
  "premise": "800-1500字的连贯世界观描述，包含：时代与背景、地理与势力、社会结构、核心冲突、独特法则、叙事基调。写成有画面感的描述性文字，不要列表，给创作者留白",
  "rules": ["世界观必须遵守的硬规则，3-5条，每条例句形式"],
  "constraints": ["创作时不得违反的边界，1-3条"]
}}

只输出 JSON，不要输出 markdown 代码块标记。"""

_CHARACTERS_PROMPT: str = """你是一位资深角色设计师。请根据下面的世界观和创作需求，设计 {count} 个角色。

## 世界观
标题：{world_title}
设定：{world_premise}
硬规则：{world_rules}

## 创作需求
{requirements}

## 设计要求
- 角色之间要有关系网（师徒、对手、暗恋、宿敌等），不要各自孤立
- 性格要有冲突空间——两个性格相反的角色放在一起自然能产生剧情
- 至少有一个角色带着秘密或隐藏身份
- 角色动机要合理，不要为了推动剧情而强行安排行为
- role 用简短中文标注定位（如：主角、挚友、宿敌、导师）

请输出**内容字段 JSON**（不要输出任何其他文字），结构如下：
{{
  "characters": [
    {{"name": "角色名（2-4字）", "role": "定位", "goal": "该角色的核心目标（20-50字）", "traits": ["性格特质，3-5个"]}}
  ]
}}

必须且只能包含 {count} 个角色。只输出 JSON，不要输出 markdown 代码块标记。"""

_REVISE_CHARACTERS_PROMPT: str = """你是一位资深角色设计师。用户对部分角色不满意，要求**重新设计以下角色**，其余角色保持不变。

## 世界观
标题：{world_title}
设定：{world_premise}
硬规则：{world_rules}

## 创作需求
{requirements}

## 保持不变的角色（不要修改或重复设计这些角色）
{keep_characters_text}

## 需要重新设计的角色
{target_names_text}

## 设计要求
- 只输出需要重新设计的角色，**一个都不要多、一个都不要少**
- 新设计与原有角色之间的关系网要保持自洽
- 新设计要有新意，不要与旧版本雷同
- role 用简短中文标注定位（如：主角、挚友、宿敌、导师）

请输出**内容字段 JSON**（不要输出任何其他文字），结构如下：
{{
  "characters": [
    {{"name": "角色名（2-4字）", "role": "定位", "goal": "该角色的核心目标（20-50字）", "traits": ["性格特质，3-5个"]}}
  ]
}}

必须且只能包含 {target_count} 个角色（与"需要重新设计的角色"名单一一对应）。只输出 JSON，不要输出 markdown 代码块标记。"""

_OUTLINE_PROMPT: str = """你是一位资深小说编辑和剧情架构师。请根据世界观、角色和创作需求，生成一份完整的章节大纲。

## 世界观
标题：{world_title}
设定：{world_premise}
硬规则：{world_rules}

## 角色列表
{characters_text}

## 创作需求
{requirements}

## 大纲要求
1. 生成 {chapter_count} 章的大纲，每章包含：
   - title: 章节标题（4-8字）
   - goal: 本章核心目标（30-80字，说明本章要完成的情节推进）
   - planned_events: 本章发生的主要事件（2-4条，每条一句话）
2. 剧情设计原则：
   - 前 1-3 章建立角色关系和日常，让读者认识角色、喜欢角色
   - 第 4-6 章引入第一个主要冲突或转折
   - 中段持续升级冲突，角色关系发生变化
   - 结尾章节收束主线，留下余韵或新钩子
   - 每章事件必须让出场角色产生变化，不能原地打转

请输出**内容字段 JSON**（不要输出任何其他文字），结构如下：
{{
  "chapters": [
    {{"title": "章节标题", "goal": "本章目标", "planned_events": ["事件1", "事件2"]}}
  ]
}}

必须且只能包含 {chapter_count} 章，按故事先后顺序排列。只输出 JSON，不要输出 markdown 代码块标记。"""

_CHAPTER_PROMPT: str = """你是一位才华横溢的华语小说家。请根据本章规划、世界观与角色设定、以及之前的剧情记忆，写出本章正文。

## 本章规划
第 {chapter_number} 章：{chapter_title}
本章目标：{chapter_goal}
计划事件：{planned_events}

## 世界观、角色与剧情记忆
{context_prompt}

{repair_instruction}## 写作要求
1. 字数：约 {target_words} 字（中文字符）
2. 严格的第一人称视角（"我"），只写"我"的所见所闻所思，不要上帝视角
3. 用行动、细节和环境描写推进剧情，禁止用旁白说明信息（禁止"我看到了…我想到…"式直白叙述）
4. 完成本章规划中的所有计划事件，每章要有推进感和情感变化
5. 遵循世界观硬规则，角色言行必须符合其设定与性格
6. 对话要有潜台词和节奏感，善用动作与神态描写
7. 全角标点，杜绝AI腔（"仿佛""宛如""不禁"等词慎用，避免排比堆砌）
8. 只输出正文本身，不要任何标题、解释或 markdown 标记"""

_SUMMARY_PROMPT: str = """请为下面的小说章节写一段 100-200 字的摘要，保留关键剧情、出场角色及其行为动机，供后续章节的上下文记忆使用。

只输出内容字段 JSON：{{"summary": "摘要"}}，不要输出其他任何文字。

## 章节正文
{chapter_body}"""

# Default instruction templates, keyed by Skill id.  Production stages inject
# the versioned Skill templates (``load_builtin_templates``); these defaults
# keep embedded/test usage self-contained.
_DEFAULT_PROMPTS: Final[dict[str, str]] = {
    "world-building": _WORLD_PROMPT,
    "cast-network": _CHARACTERS_PROMPT,
    "cast-network-revise": _REVISE_CHARACTERS_PROMPT,
    "story-architecture": _OUTLINE_PROMPT,
    "chapter-scene": _CHAPTER_PROMPT,
    "memory-extraction": _SUMMARY_PROMPT,
}


class _ContentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _WorldContent(_ContentModel):
    title: str
    premise: str
    rules: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()


class _CharacterContent(_ContentModel):
    name: str
    role: str
    goal: str
    traits: tuple[str, ...] = ()


class _CharactersContent(_ContentModel):
    characters: tuple[_CharacterContent, ...]


class _ChapterContent(_ContentModel):
    title: str
    goal: str
    planned_events: tuple[str, ...] = ()


class _OutlineContent(_ContentModel):
    chapters: tuple[_ChapterContent, ...]


class _MemoryEventContent(_ContentModel):
    subject: str
    kind: str
    predicate: str
    object: str = ""


class _MemoryExtractionContent(_ContentModel):
    summary: str
    events: tuple[_MemoryEventContent, ...] = ()


@dataclass(frozen=True, slots=True)
class GovernedCreativeAdapter:
    """Real-model implementation of the planning, chapter, and memory protocols.

    The single instance can serve as ``PlanningGenerator``, ``ChapterGenerator``
    and ``MemoryExtractor`` at the same time; each protocol method is distinct
    so no name collision occurs.
    """

    transport: ProviderTransport
    now: Callable[[], datetime]
    run_id: str
    project_id: ProjectId
    # Plan-and-Execute model split:
    #   Plan phase (world / cast / outline generation) -> deepseek-v4-pro
    #   Execute phase (chapter prose / memory summary) -> deepseek-v4-flash
    world_model: str = "deepseek-v4-pro"
    characters_model: str = "deepseek-v4-pro"
    outline_model: str = "deepseek-v4-pro"
    chapter_model: str = "deepseek-v4-flash"
    summary_model: str = "deepseek-v4-flash"
    default_character_count: int = _DEFAULT_CHARACTER_COUNT
    prompts: Mapping[str, str] | None = None
    character_names_holder: dict[str, object] | None = None
    failure_sink: Callable[[str, str, str, str, str], None] | None = None
    call_event_sink: Callable[[str, str, str], None] | None = None

    # -- PlanningGenerator -------------------------------------------------

    def generate_world(self, brief: StoryBrief) -> WorldAdaptation | SchemaFailure:
        prompt = self._prompt("world-building").format(requirements=brief.user_requirements, genre=brief.genre)
        raw_text = self._call(prompt, self.world_model, "governed_world")
        parsed = _parse_content(_WorldContent, raw_text)
        if parsed is None:
            return _schema_failure(CreativeArtifactKind.WORLD, raw_text, "WorldBible")
        content, route = parsed
        timestamp = self.now()
        world = WorldBible(
            world_id=f"{brief.brief_id}:world",
            brief_id=brief.brief_id,
            title=content.title,
            premise=content.premise,
            rules=content.rules,
            constraints=content.constraints,
            created_at=timestamp,
            provenance=_provider_provenance(f"{brief.brief_id}:world", raw_text, timestamp),
        )
        return WorldAdaptation(
            raw=_raw(CreativeArtifactKind.WORLD, world.model_dump_json()),
            value=world,
            parse_route=route,
        )

    def generate_characters(
        self,
        brief: StoryBrief,
        world: WorldBible,
    ) -> CharacterAdaptation | SchemaFailure:
        count = _requested_character_count(brief.user_requirements) or self.default_character_count
        prompt = self._prompt("cast-network").format(
            count=count,
            world_title=world.title,
            world_premise=world.premise,
            world_rules="\n".join(f"- {rule}" for rule in world.rules),
            requirements=brief.user_requirements,
        )
        raw_text = self._call(prompt, self.characters_model, "governed_cast")
        parsed = _parse_content(_CharactersContent, raw_text)
        if parsed is None:
            return _schema_failure(CreativeArtifactKind.CHARACTERS, raw_text, "CharacterSet")
        content, route = parsed
        timestamp = self.now()
        characters = tuple(
            CharacterCard(
                character_id=f"{brief.brief_id}:character:{index}",
                name=character.name,
                role=character.role,
                goal=character.goal,
                traits=character.traits,
                created_at=timestamp,
                provenance=_provider_provenance(
                    f"{brief.brief_id}:character:{index}",
                    raw_text,
                    timestamp,
                ),
            )
            for index, character in enumerate(content.characters, start=1)
        )
        value = CharacterSet(
            character_set_id=f"{brief.brief_id}:cast",
            world_id=world.world_id,
            characters=characters,
            created_at=timestamp,
            provenance=_provider_provenance(f"{brief.brief_id}:cast", raw_text, timestamp),
        )
        return CharacterAdaptation(
            raw=_raw(CreativeArtifactKind.CHARACTERS, value.model_dump_json()),
            value=value,
            parse_route=route,
        )

    def revise_characters(
        self,
        brief: StoryBrief,
        world: WorldBible,
        existing: CharacterSet,
        target_names: tuple[str, ...],
    ) -> CharacterAdaptation | SchemaFailure:
        """Re-design only the named characters; keep the rest unchanged.

        The provider is asked for the target characters' new cards only; the
        merged set (kept characters + revised cards) becomes the new cast.
        """
        target_set = frozenset(target_names)
        kept = tuple(
            character for character in existing.characters if character.name not in target_set
        )
        prompt = self._prompt("cast-network-revise").format(
            world_title=world.title,
            world_premise=world.premise,
            world_rules="\n".join(f"- {rule}" for rule in world.rules),
            requirements=brief.user_requirements,
            keep_characters_text="\n".join(
                f"- {character.name}（{character.role}）：{character.goal}"
                for character in kept
            )
            or "（无）",
            target_names_text="\n".join(f"- {name}" for name in target_names),
            target_count=len(target_names),
        )
        raw_text = self._call(prompt, self.characters_model, "governed_cast_revise")
        parsed = _parse_content(_CharactersContent, raw_text)
        if parsed is None:
            return _schema_failure(CreativeArtifactKind.CHARACTERS, raw_text, "CharacterSet")
        content, route = parsed
        produced = frozenset(character.name for character in content.characters)
        if produced != target_set:
            return _schema_failure(
                CreativeArtifactKind.CHARACTERS,
                raw_text,
                "CharacterSet",
                reason_code="revise_target_names_mismatch",
            )
        timestamp = self.now()
        revised = tuple(
            CharacterCard(
                character_id=f"{brief.brief_id}:character:{index}",
                name=character.name,
                role=character.role,
                goal=character.goal,
                traits=character.traits,
                created_at=timestamp,
                provenance=_provider_provenance(
                    f"{brief.brief_id}:character:{index}",
                    raw_text,
                    timestamp,
                ),
            )
            for index, character in enumerate(content.characters, start=1)
        )
        merged = CharacterSet(
            character_set_id=f"{brief.brief_id}:cast",
            world_id=world.world_id,
            characters=kept + revised,
            created_at=timestamp,
            provenance=_provider_provenance(f"{brief.brief_id}:cast", raw_text, timestamp),
        )
        return CharacterAdaptation(
            raw=_raw(CreativeArtifactKind.CHARACTERS, merged.model_dump_json()),
            value=merged,
            parse_route=route,
        )

    def generate_outline(
        self,
        brief: StoryBrief,
        world: WorldBible,
        characters: CharacterSet,
    ) -> OutlineAdaptation | SchemaFailure:
        prompt = self._prompt("story-architecture").format(
            world_title=world.title,
            world_premise=world.premise,
            world_rules="\n".join(f"- {rule}" for rule in world.rules),
            characters_text="\n".join(
                f"- {character.name}（{character.role}）：{character.goal}"
                for character in characters.characters
            ),
            requirements=brief.user_requirements,
            chapter_count=brief.target_chapter_count,
        )
        raw_text = self._call(prompt, self.outline_model, "governed_outline")
        parsed = _parse_content(_OutlineContent, raw_text)
        if parsed is None:
            self._record_parse_failure("governed_outline", self.outline_model, raw_text)
            return _schema_failure(CreativeArtifactKind.OUTLINE, raw_text, "PlanContract")
        content, route = parsed
        if len(content.chapters) != brief.target_chapter_count:
            self._record_parse_failure(
                "governed_outline",
                self.outline_model,
                raw_text,
                reason=f"outline_chapter_count_mismatch: got {len(content.chapters)}, want {brief.target_chapter_count}",
            )
            return _schema_failure(
                CreativeArtifactKind.OUTLINE,
                raw_text,
                "PlanContract",
                reason_code="outline_chapter_count_mismatch",
            )
        timestamp = self.now()
        evidence_sha256 = hashlib.sha256(world.model_dump_json().encode()).hexdigest()
        chapters = tuple(
            ChapterPlan(
                chapter_plan_id=f"{brief.brief_id}:chapter:{number}",
                chapter_number=number,
                title=chapter.title,
                goal=chapter.goal,
                planned_events=chapter.planned_events,
                created_at=timestamp,
                provenance=_provider_provenance(f"{brief.brief_id}:chapter:{number}", raw_text, timestamp),
                evidence=(
                    EvidenceRef(
                        evidence_id=f"{brief.brief_id}:evidence:{number}",
                        artifact_id=world.world_id,
                        artifact_revision=1,
                        source_sha256=evidence_sha256,
                        recorded_at=timestamp,
                    ),
                ),
            )
            for number, chapter in enumerate(content.chapters, start=1)
        )
        plan = PlanContract(
            plan_id=f"{brief.brief_id}:plan",
            brief_id=brief.brief_id,
            world_id=world.world_id,
            character_set_id=characters.character_set_id,
            approval_status=PlanApprovalStatus.VALIDATED,
            chapters=chapters,
            created_at=timestamp,
            provenance=_provider_provenance(f"{brief.brief_id}:plan", raw_text, timestamp),
        )
        return OutlineAdaptation(
            raw=_raw(CreativeArtifactKind.OUTLINE, plan.model_dump_json()),
            value=plan,
            parse_route=route,
        )

    # -- ChapterGenerator --------------------------------------------------

    def generate(self, request: ChapterDraftRequest) -> ChapterAdaptation | SchemaFailure:
        chapter = request.chapter_plan
        repair_instruction = ""
        if request.repair_brief is not None:
            repair_instruction = (
                f"## 上次修订反馈（第 {request.repair_brief.repair_attempt} 次）\n"
                f"失败原因：{', '.join(request.repair_brief.failure_codes)}\n"
                "请针对上述问题重写本章，不要重复之前的写法。\n\n"
            )
        prompt = self._prompt("chapter-scene").format(
            chapter_number=chapter.chapter_number,
            chapter_title=chapter.title,
            chapter_goal=chapter.goal,
            planned_events="\n".join(f"- {event}" for event in chapter.planned_events),
            context_prompt=request.context.prompt or "（暂无历史记忆，本章是新故事的开端。）",
            repair_instruction=repair_instruction,
            target_words=request.target_chinese_characters,
        )
        raw_text = self._call(prompt, self.chapter_model, "governed_chapter")
        body = _strip_code_fence(raw_text).strip()
        if not body:
            return _schema_failure(CreativeArtifactKind.CHAPTER, raw_text, "ChapterDraft")
        timestamp = self.now()
        draft = ChapterDraft(
            chapter_draft_id=f"{request.task_id}:draft",
            chapter_plan_id=chapter.chapter_plan_id,
            body=body,
            created_at=timestamp,
            provenance=_provider_provenance(request.task_id, raw_text, timestamp),
        )
        # Chapters are stored as a structured JSON envelope (like every other
        # artifact kind): the prose body lives in ``ChapterDraft.body`` and
        # the envelope carries the plan linkage.  Storing the raw model text
        # verbatim made downstream consumers (approval, export, written-
        # chapter detection) diverge between providers, so the envelope is
        # canonical for all providers.
        return ChapterAdaptation(
            raw=_raw(CreativeArtifactKind.CHAPTER, draft.model_dump_json()),
            value=draft,
            parse_route=CreativeParseRoute.STRICT_JSON,
        )

    # -- MemoryExtractor ---------------------------------------------------

    def extract(self, request: MemoryExtractionRequest) -> tuple[MemoryProposal, ...]:
        body = request.staged.body
        summary = _fallback_summary(body)
        character_names = self._character_names()
        characters_text = "\n".join(f"- {name}" for name in character_names) or "（暂无角色名单，本环节不提取事件）"
        prompt = self._prompt("memory-extraction").format(
            chapter_body=body,
            characters=characters_text,
        )
        events: tuple[_MemoryEventContent, ...] = ()
        try:
            raw_text = self._call(prompt, self.summary_model, "governed_memory")
            parsed = _parse_content(_MemoryExtractionContent, raw_text)
            if parsed is not None:
                if parsed[0].summary.strip():
                    summary = parsed[0].summary.strip()
                events = parsed[0].events
        except Exception:  # noqa: BROAD_EXCEPT_OK - extraction degrades to summary-only.
            pass
        proposals: list[MemoryProposal] = [
            self._summary_proposal(request, summary),
        ]
        name_set = frozenset(character_names)
        for index, event in enumerate(events, start=1):
            if event.subject not in name_set:
                continue  # unresolved entities never enter canonical memory
            proposals.append(self._event_proposal(request, event, index))
        return tuple(proposals)

    def _summary_proposal(self, request: MemoryExtractionRequest, summary: str) -> MemoryProposal:
        extraction = NovelSaveCandidateAdapter().extract(
            M0CandidateExtractionRequest(
                run_id=request.run_id,
                project_id=self.project_id,
                source_artifact_id=request.staged.artifact_id,
                source_artifact_revision=request.staged.revision,
                chapter_body=request.staged.body,
                source_text=request.staged.raw_bytes.decode("utf-8", errors="replace"),
                subject_id=request.chapter_plan.chapter_plan_id,
                predicate="summary",
                value_json=json.dumps({"summary": summary}, ensure_ascii=False, separators=(",", ":")),
                target_layer=MemoryLayer.M3_CONFIRMED,
                created_at=self.now(),
            )
        )
        return extraction.proposal.model_copy(update={"plan_revision": request.plan_revision})

    def _event_proposal(
        self,
        request: MemoryExtractionRequest,
        event: _MemoryEventContent,
        index: int,
    ) -> MemoryProposal:
        body = request.staged.raw_bytes.decode("utf-8", errors="replace")
        chapter_number = request.chapter_plan.chapter_number
        return MemoryProposal(
            run_id=request.run_id,
            project_id=self.project_id,
            source_artifact_id=request.staged.artifact_id,
            source_artifact_revision=request.staged.revision,
            extraction_schema_version=1,
            target_layer=MemoryLayer.M3_CONFIRMED,
            kind=MemoryFactKind.DIRECT_EVENT,
            subject_id=event.subject,
            predicate=f"direct_event:chapter_{chapter_number}:{index}",
            value_json=json.dumps(
                {
                    "event": event.predicate,
                    "object": event.object,
                    "chapter": chapter_number,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            source_start=0,
            source_end=len(body),
            source_excerpt=body,
            entity_resolution=EntityResolution.EXACT,
            inference_mode=InferenceMode.DIRECT,
            temporal_certainty=TemporalCertainty.EXACT,
            confidence=1.0,
            authority=PromotionAuthority.ACCEPTED_CHAPTER,
            plan_revision=request.plan_revision,
            supersedes_memory_id=None,
            created_at=self.now(),
        )

    def _character_names(self) -> tuple[str, ...]:
        if self.character_names_holder is None:
            return ()
        names = self.character_names_holder.get("names")
        if not isinstance(names, tuple):
            return ()
        return names

    # -- internals ----------------------------------------------------------

    def _prompt(self, skill_id: str) -> str:
        if self.prompts is not None:
            template = self.prompts.get(skill_id)
            if template:
                return template
        return _DEFAULT_PROMPTS[skill_id]

    def _call(self, prompt: str, model_name: str, caller: str) -> str:
        timestamp = self.now()
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        request = ProviderRequest(
            provider_call_id=f"{self.run_id}:{caller}:{prompt_sha256[:12]}",
            run_id=self.run_id,
            task_id=f"{self.run_id}:{caller}",
            model_name=model_name,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            temperature=_TEMPERATURE,
            created_at=timestamp,
            provenance=Provenance(
                source_kind=ProvenanceKind.USER_REQUEST,
                source_id=f"{self.run_id}:{caller}",
                source_sha256=prompt_sha256,
                observed_at=timestamp,
            ),
        )
        try:
            if self.call_event_sink is not None:
                self.call_event_sink(caller, model_name, "started")
            content = self.transport.generate(request).content
            if self.call_event_sink is not None:
                self.call_event_sink(caller, model_name, "finished")
            return content
        except ProviderTransportError as error:
            if self.call_event_sink is not None:
                self.call_event_sink(caller, model_name, "finished")
            self._record_failure(caller, request, error)
            raise
        except Exception as error:  # noqa: BROAD_EXCEPT_OK - transport boundary
            if self.call_event_sink is not None:
                self.call_event_sink(caller, model_name, "finished")
            self._record_failure(caller, request, error)
            raise

    def _record_failure(self, caller: str, request: ProviderRequest, error: Exception) -> None:
        if self.failure_sink is None:
            return
        from novel_agent.provider_types import ProviderFailureKind

        kind = error.kind.value if isinstance(error, ProviderTransportError) else type(error).__name__
        detail = error.detail if isinstance(error, ProviderTransportError) else str(error)
        self.failure_sink(
            caller=caller,
            model_name=request.model_name,
            failure_kind=kind,
            reason=f"transport: {detail[:400]}",
            prompt_sha256=request.prompt_sha256,
        )

    def _record_parse_failure(
        self,
        caller: str,
        model_name: str,
        raw_text: str,
        reason: str | None = None,
    ) -> None:
        """Record a schema/parse failure so the console can explain why a
        generated artifact never became staged/accepted."""
        if self.failure_sink is None:
            return
        excerpt = " ".join(raw_text.split())[:300]
        self.failure_sink(
            caller=caller,
            model_name=model_name,
            failure_kind="parse",
            reason=f"parse: {reason or 'schema validation failed'} | {excerpt!r}",
            prompt_sha256="",
        )


def _requested_character_count(requirements: str) -> int | None:
    """Parse an explicit character count from the user's requirements.

    Supports Arabic numerals and common Chinese numerals (e.g. "3个角色",
    "三名角色", "六个角色").  Returns None when the requirement does not
    state a count, letting the caller use its default.
    """
    import re

    arabic = re.search(r"(\d+)\s*个?(?:主要)?角色", requirements)
    if arabic:
        return int(arabic.group(1))
    chinese = {"一": 1, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    match = re.search(r"([一二两三四五六七八九十])\s*(?:个|名)\s*(?:主要|女主|男主)?角(?:色)?", requirements)
    if match:
        return chinese[match.group(1)]
    return None


def _parse_content(model_type: type[_ContentModel], raw_text: str) -> tuple[_ContentModel, CreativeParseRoute] | None:
    strict = _validate_content(model_type, raw_text)
    if strict is not None:
        return strict, CreativeParseRoute.STRICT_JSON
    try:
        repaired = str(repair_json(raw_text))
    except ValueError:
        return None
    repaired_parse = _validate_content(model_type, repaired)
    if repaired_parse is not None:
        return repaired_parse, CreativeParseRoute.JSON_REPAIR
    return None


def _validate_content(model_type: type[_ContentModel], raw_text: str) -> _ContentModel | None:
    try:
        return model_type.model_validate_json(raw_text)
    except ValidationError:
        return None


def _schema_failure(
    kind: CreativeArtifactKind,
    raw_text: str,
    expected_contract: str,
    *,
    reason_code: str = "creative_schema_failure",
) -> SchemaFailure:
    return SchemaFailure(
        raw_artifact=_raw(kind, raw_text),
        expected_contract=expected_contract,
        parse_routes=(CreativeParseRoute.STRICT_JSON, CreativeParseRoute.JSON_REPAIR),
        reason_code=reason_code,
    )


def _raw(kind: CreativeArtifactKind, raw_text: str) -> RawCreativeArtifact:
    return RawCreativeArtifact(
        kind=kind,
        raw_bytes=raw_text.encode(),
        content_sha256=hashlib.sha256(raw_text.encode()).hexdigest(),
        adapter_version="governed-creative-v1",
    )


def _provider_provenance(source_id: str, raw_text: str, observed_at: datetime) -> Provenance:
    return Provenance(
        source_kind=ProvenanceKind.PROVIDER_RESPONSE,
        source_id=source_id,
        source_sha256=hashlib.sha256(raw_text.encode()).hexdigest(),
        observed_at=observed_at,
    )


def _strip_code_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _fallback_summary(body: str) -> str:
    compact = " ".join(body.split())
    return compact[:200]


__all__ = ("GovernedCreativeAdapter",)
