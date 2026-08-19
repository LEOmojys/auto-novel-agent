"""Provider-backed quality evaluators for the novel domain.

Implements the two evaluator protocols from ``quality.py`` with live provider
calls, mirroring the "independent judge" pattern used by production novel
pipelines (autonovel keeps the writer and judge on different models).  The
writer generates prose; the evaluator scores six dimensions on a 1-5 scale.
A provider failure returns an empty string so the quality policy degrades to
``EVALUATOR_ERROR`` (pause), never to a fabricated score.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from novel_agent.contracts import Provenance, ProvenanceKind, ProviderRequest
from novel_agent.quality import (
    ChapterMetrics,
    CharacterFixture,
    EvaluatorAdapterRequest,
    EvaluatorReport,
    MetricsEvaluator,
    StructuredEvaluator,
)
from novel_agent.provider_types import ProviderTransport, ProviderTransportError

_CHINESE_CHARACTER_PATTERN = re.compile(r"[\u4e00-\u9fff]")

_EVALUATION_PROMPT: str = """你是一位苛刻的小说评审编辑。请评估下面的章节草稿，在六个维度上各给出 1-5 分（5 为最高）和一句理由。

## 章节要求
{beats}

## 角色
{characters}

## 章节正文
{body}

请输出**内容字段 JSON**（不要输出任何其他文字）：
{{
  "style_consistency": {{"score": 1-5, "reason": "理由"}},
  "character_consistency": {{"score": 1-5, "reason": "理由"}},
  "prose_fluency": {{"score": 1-5, "reason": "理由"}},
  "scene_completeness": {{"score": 1-5, "reason": "理由"}},
  "dialogue_naturalness": {{"score": 1-5, "reason": "理由"}},
  "content_alignment": {{"score": 1-5, "reason": "理由"}}
}}

评分要严格：文笔生硬、角色言行不符、情节推进不足都要给低分。只输出 JSON。"""


@dataclass(frozen=True, slots=True)
class ProviderStructuredEvaluator:
    """A ``StructuredEvaluator`` that asks the provider for a six-dimension report."""

    transport: ProviderTransport
    now: Callable[[], datetime]
    run_id: str
    model_name: str = "deepseek-v4-flash"
    prompt: str | None = None
    failure_sink: Callable[[str, str, str, str, str], None] | None = None
    call_event_sink: Callable[[str, str, str], None] | None = None

    def evaluate(self, request: EvaluatorAdapterRequest) -> str:
        characters_text = "\n".join(
            f"- {fixture.name}（{'/'.join(fixture.aliases) if fixture.aliases else '无别名'}）"
            for fixture in request.characters
        ) or "（无角色信息）"
        beats_text = "\n".join(f"- {beat}" for beat in request.required_beats) or "（无明确事件要求）"
        template = self.prompt or _EVALUATION_PROMPT
        prompt = template.format(
            beats=beats_text,
            characters=characters_text,
            body=request.body[:8_000],
        )
        timestamp = self.now()
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        provider_request = ProviderRequest(
            provider_call_id=f"{self.run_id}:quality:{request.chapter_plan_id}:{prompt_sha256[:12]}",
            run_id=self.run_id,
            task_id=f"{self.run_id}:quality:{request.chapter_plan_id}",
            model_name=self.model_name,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            temperature=0.2,
            created_at=timestamp,
            provenance=Provenance(
                source_kind=ProvenanceKind.USER_REQUEST,
                source_id=f"{self.run_id}:quality:{request.chapter_plan_id}",
                source_sha256=prompt_sha256,
                observed_at=timestamp,
            ),
        )
        try:
            if self.call_event_sink is not None:
                self.call_event_sink("quality", provider_request.model_name, "started")
            raw = self.transport.generate(provider_request).content
            if self.call_event_sink is not None:
                self.call_event_sink("quality", provider_request.model_name, "finished")
        except ProviderTransportError as error:
            if self.call_event_sink is not None:
                self.call_event_sink("quality", provider_request.model_name, "finished")
            if self.failure_sink is not None:
                self.failure_sink(
                    caller="quality",
                    model_name=provider_request.model_name,
                    failure_kind=error.kind.value if hasattr(error, "kind") else type(error).__name__,
                    reason=f"transport: {str(error)[:400]}",
                    prompt_sha256=provider_request.prompt_sha256,
                )
            return ""
        try:
            return EvaluatorReport.model_validate_json(raw).model_dump_json()
        except ValidationError:
            if self.failure_sink is not None:
                self.failure_sink(
                    caller="quality",
                    model_name=provider_request.model_name,
                    failure_kind="parse",
                    reason=f"parse: {raw[:400]!r}",
                    prompt_sha256=provider_request.prompt_sha256,
                )
            return ""


@dataclass(frozen=True, slots=True)
class CharCountMetricsEvaluator:
    """Deterministic metrics: Chinese character count and mentioned character ids."""

    def measure(
        self,
        body: str,
        target_chinese_characters: int,
        characters: tuple[CharacterFixture, ...],
    ) -> ChapterMetrics:
        mentioned: list[str] = []
        for fixture in characters:
            names = (fixture.name, *fixture.aliases)
            if any(name and name in body for name in names):
                mentioned.append(fixture.character_id)
        return ChapterMetrics(
            chinese_characters=len(_CHINESE_CHARACTER_PATTERN.findall(body)),
            target_chinese_characters=target_chinese_characters,
            mentioned_character_ids=tuple(mentioned),
        )


__all__ = ("CharCountMetricsEvaluator", "ProviderStructuredEvaluator")
