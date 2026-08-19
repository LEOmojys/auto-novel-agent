from __future__ import annotations

import json
from datetime import UTC, datetime

from novel_agent.domains.quality_evaluators import CharCountMetricsEvaluator, ProviderStructuredEvaluator
from novel_agent.provider_types import ProviderFailureKind, ProviderTransportError, ProviderTransportResult
from novel_agent.quality import CharacterFixture, EvaluatorAdapterRequest


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


class StubTransport:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def generate(self, request) -> ProviderTransportResult:
        self.prompts.append(request.prompt)
        return ProviderTransportResult(content=self.content, model_name="t", prompt_tokens=1, completion_tokens=1)


class BrokenTransport:
    def generate(self, _request) -> ProviderTransportResult:
        raise ProviderTransportError(kind=ProviderFailureKind.TRANSIENT, detail="wire down")


def _request() -> EvaluatorAdapterRequest:
    return EvaluatorAdapterRequest(
        project_id="novel-one",
        artifact_id="chapter-1",
        body="主角林晚抵达魔法学院，与阿泽重逢。",
        plan_id="plan-1",
        chapter_plan_id="chapter-plan-1",
        chapter_number=1,
        required_beats=("抵达学院", "遇见阿泽"),
        characters=(
            CharacterFixture(character_id="c-1", name="林晚", aliases=("晚晚",)),
            CharacterFixture(character_id="c-2", name="阿泽"),
        ),
    )


def test_provider_evaluator_returns_validated_report_json() -> None:
    report = {
        "style_consistency": {"score": 4, "reason": "风格一致"},
        "character_consistency": {"score": 5, "reason": "言行符合"},
        "prose_fluency": {"score": 4, "reason": "流畅"},
        "scene_completeness": {"score": 3, "reason": "场景可补"},
        "dialogue_naturalness": {"score": 4, "reason": "自然"},
        "content_alignment": {"score": 4, "reason": "符合目标"},
    }
    transport = StubTransport(json.dumps(report, ensure_ascii=False))
    evaluator = ProviderStructuredEvaluator(transport=transport, now=lambda: FIXED_TIME, run_id="run-001")

    raw = evaluator.evaluate(_request())

    assert json.loads(raw)["character_consistency"]["score"] == 5
    assert len(transport.prompts) == 1
    assert "苛刻的小说评审编辑" in transport.prompts[0]
    assert "林晚" in transport.prompts[0]


def test_provider_evaluator_failure_returns_empty_string() -> None:
    evaluator = ProviderStructuredEvaluator(transport=BrokenTransport(), now=lambda: FIXED_TIME, run_id="run-001")

    assert evaluator.evaluate(_request()) == ""


def test_provider_evaluator_rejects_malformed_report() -> None:
    evaluator = ProviderStructuredEvaluator(transport=StubTransport("not json"), now=lambda: FIXED_TIME, run_id="run-001")

    assert evaluator.evaluate(_request()) == ""


def test_metrics_evaluator_counts_chinese_characters_and_mentions() -> None:
    metrics = CharCountMetricsEvaluator().measure(
        body="主角林晚抵达魔法学院，与阿泽重逢。ABC123",
        target_chinese_characters=100,
        characters=_request().characters,
    )

    assert metrics.chinese_characters == 15
    assert metrics.target_chinese_characters == 100
    assert set(metrics.mentioned_character_ids) == {"c-1", "c-2"}
    assert 0.14 < metrics.length_ratio < 0.16
