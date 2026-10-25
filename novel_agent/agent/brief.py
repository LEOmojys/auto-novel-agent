"""Requirement analysis: turn a raw user request into a structured brief.

This is the first stage of the plan-and-execute loop.  ``BriefAnalyzer`` asks
the provider to distil goals, constraints, style guidance, acceptance
criteria and a domain hint from the raw request.  When the provider is
unavailable or returns malformed output the analyzer degrades to a
deterministic fallback so a run can still proceed (the raw text is always
preserved verbatim in ``BriefContract.raw_requirements``).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from json_repair import repair_json
from pydantic import BaseModel, ConfigDict, ValidationError

from novel_agent.contracts import Provenance, ProvenanceKind, ProviderRequest
from novel_agent.pathing import ProjectId
from novel_agent.provider_types import ProviderTransport, ProviderTransportError

_DEFAULT_DOMAIN: str = "novel"
_DEFAULT_INTENT: str = "full_novel"
#: Sentinel intent for out-of-distribution requests: the provider could not
#: extract a real creative goal AND no deterministic rule matched either.
#: The run must fail with an explicit "intent recognition failed" reason
#: instead of silently treating chatter/gibberish as a full-novel request.
_INTENT_UNRECOGNIZED: str = "unrecognized"
_DEFAULT_CHAPTER_COUNT: int = 10
_SUPPORTED_INTENTS: frozenset[str] = frozenset(
    (
        "full_novel",
        "characters_only",
        "world_only",
        "outline_only",
        "add_character",
        "unrecognized",
    )
)
INTENT_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "world_only": ("world",),
    "characters_only": ("world", "cast"),
    "outline_only": ("world", "cast", "outline"),
    "full_novel": ("world", "cast", "outline", "chapters"),
    "add_character": ("add_character",),
}


_ONLY_WORLD = re.compile(r"(?:只|仅|只要|单单)(?:写|要|生成|设定|设计)?(?:世界观|世界设定|背景设定|背景)")
_ONLY_CAST = re.compile(r"(?:只|仅|只要|单单)(?:写|要|生成|设计)?(?:角色|人物)")
_ONLY_OUTLINE = re.compile(r"(?:只|仅|只要|单单)(?:写|要|生成|设计)?(?:大纲|故事大纲|章节规划)")
_NO_CHAPTER = re.compile(r"(?:不要|不写|无需|不用|别)(?:写)?(?:正文|章节|正文内容|章节内容)")
_NO_OUTLINE = re.compile(r"(?:不要|不写|无需|不用|别)(?:写)?(?:大纲|故事大纲|章节规划)")


def detect_intent_conflicts(requirements: str, intents: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Return explicit intent conflicts that require a user decision.

    Composite intents are normally valid (e.g. cast + outline).  A conflict is
    only raised when the request contains an explicit exclusivity phrase
    ("只写世界观", "不要章节") that contradicts the analysed deliverables.
    """
    text = requirements or ""
    capabilities = set(intent_capabilities(intents))
    conflicts: list[str] = []
    if _ONLY_WORLD.search(text) and capabilities - {"world"}:
        conflicts.append("only_world_contradicts_downstream")
    if _ONLY_CAST.search(text) and capabilities & {"outline", "chapters"}:
        conflicts.append("only_cast_contradicts_outline_or_chapters")
    if _ONLY_OUTLINE.search(text) and "chapters" in capabilities:
        conflicts.append("only_outline_contradicts_chapters")
    if _NO_CHAPTER.search(text) and "chapters" in capabilities:
        conflicts.append("no_chapter_contradicts_chapters")
    if _NO_OUTLINE.search(text) and "outline" in capabilities:
        conflicts.append("no_outline_contradicts_outline")
    return tuple(dict.fromkeys(conflicts))


def _clarification_question(conflicts: tuple[str, ...]) -> str:
    detail = "；".join(conflicts)
    return (
        f"检测到意图冲突（{detail}）。请确认你要的交付范围："
        "1) 只按当前限制生成；2) 继续完整小说流程。"
        "回复新的需求后我会重新规划。"
    )


def intent_capabilities(intents: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    ordered = ("world", "cast", "outline", "add_character", "chapters")
    available: set[str] = set()
    for intent in intents or ():
        available.update(INTENT_CAPABILITIES.get(intent, ()))
    return tuple(item for item in ordered if item in available)
_CHAPTER_COUNT_PATTERN = re.compile(r"(\d+)\s*章")
_CHINESE_CHAPTER_PATTERN = re.compile(r"([零〇一二两三四五六七八九十百]+)\s*章")
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

_INTENT_HINTS: str = (
    '"intents": 用户要求的全部交付意图数组（可多选，按主次排列）；单意图时只含一个元素。取值同上。'
    '"intent": 主意图（与 intents 第一项一致）：'
    '"full_novel"（写一部完整小说，含世界观/角色/大纲/章节），'
    '"characters_only"（只设计角色/角色卡），'
    '"world_only"（只设定世界观/背景），'
    '"outline_only"（只写故事大纲/章节规划），'
    '"add_character"（为已有故事新增角色），'
    '"unrecognized"（用户需求与小说创作无关、是日常闲聊、无法理解或乱码时选择；'
    '拿不准但看起来确实是创作需求时选 "full_novel"）'
)

#: Deterministic "new story" signals: the user explicitly wants a DIFFERENT
#: story than whatever already exists in the project, so old assets must not
#: be reused.  Used for the brief's ``fresh`` flag (and as a degraded-mode
#: fallback when the provider analysis fails).
_FRESH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:换|另|新)(?:一个|一部|本)?(?:故事|小说|题材|主题)"),
    re.compile(r"(?:写|开|创作|来)(?:一部|一本|个)?(?:全新|新的|另外|别的|不同的|另一|完全不同)(?:故事|小说|题材|主题|作品)"),
    re.compile(r"(?:不是|别于|不同于|不要用)(?:之前|原来|上次|刚才).{0,8}(?:那个|这个)?(?:故事|设定|题材|小说)"),
    re.compile(r"(?:重新|从头)(?:写|开始|构思|创作)"),
    re.compile(r"(?:抛弃|丢弃|扔掉|清空|重置).{0,10}(?:设定|故事|世界观)"),
    re.compile(r"(?:不要|舍弃|放弃)(?:之前|原来|以前的|上次的).{0,8}(?:设定|世界观|故事|大纲|角色)"),
    re.compile(r"(?:和|与)(?:之前|原来|旧).{0,4}(?:故事|设定).{0,6}(?:不同|无关|完全不一样)"),
)


def _is_fresh_story(requirements: str) -> bool:
    """Whether the request asks for a brand-new story independent of existing assets."""
    text = requirements.strip()
    return any(pattern.search(text) for pattern in _FRESH_PATTERNS)

_ANALYSIS_PROMPT: str = """你是一个需求分析器。请分析用户的创作需求，提炼出结构化信息。

## 用户需求
{requirements}

请输出**内容字段 JSON**（不要输出任何其他文字）：
{{
  "domain": "领域，当前仅支持 novel（小说创作）",
  "goal": "用户的核心目标（一句话，20-80字）",
  "constraints": ["用户明确的约束条件，3-8条；无则空数组"],
  "style_guidance": ["风格指引，2-5条；如第一人称、文风参考、篇幅偏好"],
  "acceptance_criteria": ["验收标准，2-5条；如何判断创作符合需求"],
  "chapter_count": 章节数（用户未指定则省略此字段，由系统默认）,
  "target_words": 单章目标字数（用户说"每章5000字"时填 5000；未指定则省略）,

  {intent_hints}
  "fresh": 是否要求全新故事（true：用户明确要"换一个/新写一个/全新/不同于之前"的故事，此时不要沿用已有世界观/角色/大纲；false：续写或基于已有设定；用户未提则省略或 false）
}}

只输出 JSON，不要输出 markdown 代码块标记。"""


class BriefContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    brief_id: str
    project_id: ProjectId
    raw_requirements: str
    domain: str = _DEFAULT_DOMAIN
    intent: str = _DEFAULT_INTENT
    intents: tuple[str, ...] = ()
    degraded: bool = False
    degraded_reason: str | None = None
    degraded_detail: str | None = None
    goal: str
    constraints: tuple[str, ...] = ()
    style_guidance: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    target_chapter_count: int | None = None
    target_words: int | None = None
    fresh: bool = False  # True: a brand-new story; do not reuse existing assets
    intent_conflicts: tuple[str, ...] = ()
    needs_clarification: bool = False
    clarification_question: str | None = None
    created_at: datetime
    provenance: Provenance


class _AnalysisContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: str = _DEFAULT_DOMAIN
    intent: Literal["full_novel", "characters_only", "world_only", "outline_only", "add_character", "unrecognized"] = _DEFAULT_INTENT
    goal: str
    intents: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    style_guidance: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    chapter_count: int | None = None
    target_words: int | None = None
    fresh: bool = False


@dataclass(frozen=True, slots=True)
class BriefAnalyzer:
    """Distil a structured brief from raw user requirements.

    Requirement analysis is a planning task: the stronger model is the
    default so goals, constraints and intent are understood precisely.
    """

    transport: ProviderTransport
    now: Callable[[], datetime]
    model_name: str = "deepseek-v4-pro"
    prompt: str | None = None
    failure_sink: Callable[[str, str, str, str, str], None] | None = None
    call_event_sink: Callable[[str, str, str], None] | None = None

    def analyze(
        self,
        requirements: str,
        *,
        project_id: ProjectId,
        run_id: str,
        brief_id: str,
        explicit_chapter_count: int | None = None,
    ) -> BriefContract:
        timestamp = self.now()
        content, degraded_reason, degraded_detail = self._analyze_content(requirements, run_id)
        if content is not None:
            chapter_count = explicit_chapter_count or content.chapter_count
            goal = content.goal.strip() or _fallback_goal(requirements)
            if chapter_count is not None and not 1 <= chapter_count <= 200:
                chapter_count = None
            target_words = content.target_words
            if target_words is not None and not 300 <= target_words <= 30_000:
                target_words = None
            domain = content.domain
            raw_intents = content.intents or ((content.intent,) if content.intent else ())
            if not content.intents and content.intent != _INTENT_UNRECOGNIZED:
                heuristic = _heuristic_intents(requirements)
                if len(heuristic) > 1:
                    # The provider sometimes returns only the primary intent for
                    # an explicitly compound request ("design a cast and draft
                    # chapter 1").  Add the deterministic multi-intent matches.
                    raw_intents = (content.intent, *heuristic)
            intents = tuple(
                item for item in dict.fromkeys(raw_intents) if item in _SUPPORTED_INTENTS
            )
            if not intents:
                intents = (content.intent,)
            if content.intent != _INTENT_UNRECOGNIZED and content.intent not in intents:
                intents = (content.intent, *intents)
            if len(intents) > 1 and _INTENT_UNRECOGNIZED in intents:
                intents = tuple(item for item in intents if item != _INTENT_UNRECOGNIZED)
            intent = _primary_intent(intents, preferred=content.intent)
            intent_conflicts = detect_intent_conflicts(requirements, intents)
            constraints = content.constraints
            style_guidance = content.style_guidance
            acceptance_criteria = content.acceptance_criteria
            fresh = content.fresh or _is_fresh_story(requirements)
            degraded = False
        else:
            chapter_count = explicit_chapter_count
            target_words = _parse_target_words(requirements)
            goal = _fallback_goal(requirements)
            domain = _DEFAULT_DOMAIN
            heuristic = _heuristic_intents(requirements)
            intents = heuristic or (_INTENT_UNRECOGNIZED,)
            intent = _primary_intent(intents, preferred=intents[0])
            intent_conflicts = detect_intent_conflicts(requirements, intents)
            constraints = ()
            style_guidance = ()
            acceptance_criteria = ()
            fresh = _is_fresh_story(requirements)
            degraded = True
        if chapter_count is None:
            chapter_count = _parse_chapter_count(requirements)
        return BriefContract(
            brief_id=brief_id,
            project_id=project_id,
            raw_requirements=requirements,
            domain=domain,
            intent=intent,
            intents=intents,
            degraded=degraded,
            degraded_reason=degraded_reason,
            degraded_detail=degraded_detail,
            goal=goal,
            constraints=constraints,
            style_guidance=style_guidance,
            acceptance_criteria=acceptance_criteria,
            target_chapter_count=chapter_count,
            target_words=target_words,
            fresh=fresh,
            intent_conflicts=intent_conflicts,
            needs_clarification=bool(intent_conflicts),
            clarification_question=_clarification_question(intent_conflicts) if intent_conflicts else None,
            created_at=timestamp,
            provenance=Provenance(
                source_kind=ProvenanceKind.USER_REQUEST,
                source_id=run_id,
                source_sha256=hashlib.sha256(requirements.encode()).hexdigest(),
                observed_at=timestamp,
            ),
        )

    def _analyze_content(self, requirements: str, run_id: str) -> tuple[_AnalysisContent | None, str | None, str | None]:
        """Return (parsed analysis, degraded reason, provider detail)."""
        template = self.prompt or _ANALYSIS_PROMPT
        prompt = template.format(requirements=requirements, intent_hints=_INTENT_HINTS)
        timestamp = self.now()
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        request = ProviderRequest(
            provider_call_id=f"{run_id}:brief-analysis:{prompt_sha256[:12]}",
            run_id=run_id,
            task_id=f"{run_id}:brief-analysis",
            model_name=self.model_name,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            temperature=0.2,
            created_at=timestamp,
            provenance=Provenance(
                source_kind=ProvenanceKind.USER_REQUEST,
                source_id=f"{run_id}:brief-analysis",
                source_sha256=prompt_sha256,
                observed_at=timestamp,
            ),
        )
        try:
            if self.call_event_sink is not None:
                self.call_event_sink("brief", request.model_name, "started")
            raw = self.transport.generate(request).content
            if self.call_event_sink is not None:
                self.call_event_sink("brief", request.model_name, "finished")
        except ProviderTransportError as error:
            if self.call_event_sink is not None:
                self.call_event_sink("brief", request.model_name, "finished")
            self._record_failure(request, error)
            return None, "provider_failure", error.detail
        parsed = _parse_analysis(raw)
        if parsed is not None:
            return parsed, None, None
        self._record_failure(request, None, raw)
        return None, "parse_failure", None

    def _record_failure(self, request: ProviderRequest, error: Exception | None, raw: str | None = None) -> None:
        if self.failure_sink is None:
            return
        if error is not None:
            kind = error.kind.value if hasattr(error, "kind") else type(error).__name__
            reason = f"transport: {str(error)[:400]}"
        else:
            kind = "parse"
            reason = f"parse: {raw[:400]!r}"
        self.failure_sink(
            caller="brief",
            model_name=request.model_name,
            failure_kind=kind,
            reason=reason,
            prompt_sha256=request.prompt_sha256,
        )


def _parse_analysis(raw_text: str) -> _AnalysisContent | None:
    """Parse the analysis JSON with the same tolerance as creative output.

    Strips markdown code fences and falls back to ``json_repair`` so a
    well-formed provider reply is never misreported as a model failure.
    """
    strict = _validate_analysis(raw_text)
    if strict is not None:
        return strict
    stripped = _strip_code_fence(raw_text)
    if stripped != raw_text:
        strict = _validate_analysis(stripped)
        if strict is not None:
            return strict
    try:
        repaired = str(repair_json(raw_text))
    except ValueError:
        return None
    return _validate_analysis(repaired)


def _validate_analysis(raw_text: str) -> _AnalysisContent | None:
    try:
        return _AnalysisContent.model_validate_json(raw_text)
    except ValidationError:
        return None


def _strip_code_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _parse_target_words(requirements: str) -> int | None:
    matched = re.search(r"(?:每章|一章|每章节)?\s*(\d{3,5})\s*字", requirements)
    if matched is None:
        return None
    value = int(matched.group(1))
    return value if 300 <= value <= 30_000 else None


def _parse_chapter_count(requirements: str) -> int:
    matched = _CHAPTER_COUNT_PATTERN.search(requirements)
    if matched is not None:
        return max(1, int(matched.group(1)))
    chinese = _CHINESE_CHAPTER_PATTERN.search(requirements)
    if chinese is not None:
        value = _chinese_number(chinese.group(1))
        if value is not None and value >= 1:
            return value
    return _DEFAULT_CHAPTER_COUNT


def _chinese_number(text: str) -> int | None:
    total = 0
    section = 0
    number = 0
    for character in text:
        if character in _CN_DIGITS:
            number = _CN_DIGITS[character]
        elif character == "十":
            section += (number or 1) * 10
            number = 0
        elif character == "百":
            section += (number or 1) * 100
            number = 0
        else:
            return None
    return total + section + number


def _fallback_goal(requirements: str) -> str:
    compact = " ".join(requirements.split())
    return compact[:120]


# Deterministic intent fallback, ordered from most specific to least.  Used
# only when the provider analysis fails, so a sub-task request (e.g. "设计3个
# 角色") never degrades into a full-novel plan.  Returns ``None`` when nothing
# matches: an out-of-distribution request (everyday chatter, gibberish) that
# must surface as an explicit intent-recognition failure rather than being
# treated as a full-novel request by default.
_INTENT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:新增|添加|加一个).{0,4}角色"), "add_character"),
    (re.compile(r"角色(?:设定|卡)"), "characters_only"),
    (
        re.compile(r"(?:设计|创建|生成|来|添加)\s*(?:[一二两三四五六七八九十百\d]+\s*个?)?\s*[^，。；、]{0,14}(?:角色|女主角|男主角)"),
        "characters_only",
    ),
    (re.compile(r"(?:设计|创建|生成|写|创作|设定)\s*(?:[^，。；、]{0,20}?世界观|世界观)|世界设定|背景设定"), "world_only"),
    (re.compile(r"(?:设计|创建|生成|写|创作|制定)\s*(?:[^，。；、]{0,20}?大纲|大纲)|大纲(?:设定|规划)"), "outline_only"),
    (re.compile(r"写.{0,12}(?:正文|章节|小说|故事|短篇|长篇|网文)(?!大纲)"), "full_novel"),
    (re.compile(r"继续.{0,10}(?:正文|章节|小说)"), "full_novel"),
    # Continuation requests ("续写第二章", "接着写下一章") are creative work
    # even though they carry no explicit 写小说 vocabulary — they must not be
    # misjudged as out-of-distribution when the provider analysis degrades.
    (re.compile(r"(?:续写|接着写|继续写|写.{0,8}(?:下一章|第[一二三四五六七八九十百\d]+章))"), "full_novel"),
)


_CREATIVE_SIGNAL = re.compile(
    r"(?:故事|小说|长篇|短篇|网文|正文|章节|大纲|角色|世界观|旅程|冒险|奇幻|科幻|"
    r"穿越|重生|言情|武侠|悬疑|推理|都市|校园|修仙|玄幻|剧情|情节|作品|篇章|续写|"
    r"第[一二三四五六七八九十百\d]+章|下一章|"
    r"story|novel|mystery|fiction|fantasy|adventure|continue|next chapter)",
    re.IGNORECASE,
)


def _heuristic_intents(requirements: str) -> tuple[str, ...]:
    found: list[str] = []
    for pattern, intent in _INTENT_PATTERNS:
        if pattern.search(requirements) and intent not in found:
            found.append(intent)
    if found:
        return tuple(found)
    # No precise rule matched: fall back to a loose creative signal so a
    # legitimate noun-phrase request ("一个关于魔法学院的成长故事") is still
    # treated as a full novel, while chatter/gibberish with no creative
    # vocabulary at all is rejected as out-of-distribution.
    if _CREATIVE_SIGNAL.search(requirements):
        return (_DEFAULT_INTENT,)
    return ()


def _primary_intent(intents: tuple[str, ...], *, preferred: str | None = None) -> str:
    if preferred in intents:
        return preferred
    priority = ("full_novel", "outline_only", "characters_only", "world_only", "add_character", "unrecognized")
    for intent in priority:
        if intent in intents:
            return intent
    return intents[0] if intents else _INTENT_UNRECOGNIZED


def _heuristic_intent(requirements: str) -> str | None:
    intents = _heuristic_intents(requirements)
    return _primary_intent(intents) if intents else None



__all__ = ("BriefAnalyzer", "BriefContract")
