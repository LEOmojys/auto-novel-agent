from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from novel_agent.agent.brief import BriefAnalyzer, BriefContract
from novel_agent.pathing import ProjectId
from novel_agent.provider_types import ProviderFailureKind, ProviderTransportError, ProviderTransportResult


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")


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


def _analyzer(transport) -> BriefAnalyzer:
    return BriefAnalyzer(transport=transport, now=lambda: FIXED_TIME)


def test_analyzer_distils_structured_brief() -> None:
    transport = StubTransport(
        json.dumps(
            {
                "domain": "novel",
                "goal": "写一本魔法学院的成长小说",
                "constraints": ["主角是孤儿", "不使用现代科技"],
                "style_guidance": ["第一人称", "温暖文风"],
                "acceptance_criteria": ["20章完整", "人物动机合理"],
                "chapter_count": 20,
            },
            ensure_ascii=False,
        )
    )

    brief = _analyzer(transport).analyze(
        "写一本魔法学院的小说，20章",
        project_id=PROJECT_ID,
        run_id="run-001",
        brief_id="run-001:brief",
    )

    assert brief.domain == "novel"
    assert brief.goal.startswith("写一本魔法学院")
    assert "主角是孤儿" in brief.constraints
    assert brief.style_guidance == ("第一人称", "温暖文风")
    assert brief.target_chapter_count == 20
    assert brief.raw_requirements == "写一本魔法学院的小说，20章"
    assert brief.provenance.source_sha256 is not None
    assert len(transport.prompts) == 1


def test_analyzer_prefers_explicit_chapter_count() -> None:
    transport = StubTransport(
        json.dumps({"goal": "写小说", "chapter_count": 5}, ensure_ascii=False)
    )

    brief = _analyzer(transport).analyze(
        "写一本小说",
        project_id=PROJECT_ID,
        run_id="run-002",
        brief_id="run-002:brief",
        explicit_chapter_count=30,
    )

    assert brief.target_chapter_count == 30


def test_analyzer_falls_back_when_provider_unavailable() -> None:
    brief = _analyzer(BrokenTransport()).analyze(
        "写一本20章的奇幻小说",
        project_id=PROJECT_ID,
        run_id="run-003",
        brief_id="run-003:brief",
    )

    assert brief.domain == "novel"
    assert brief.raw_requirements == "写一本20章的奇幻小说"
    assert brief.target_chapter_count == 20  # parsed from the raw text
    assert brief.constraints == ()


def test_analyzer_defaults_chapter_count() -> None:
    brief = _analyzer(BrokenTransport()).analyze(
        "写一本奇幻小说",
        project_id=PROJECT_ID,
        run_id="run-004",
        brief_id="run-004:brief",
    )

    assert brief.target_chapter_count == 10
    assert isinstance(brief, BriefContract)


def test_analyzer_accepts_fence_wrapped_json_without_degrading() -> None:
    """A markdown-fenced JSON reply is a successful analysis, not a failure."""
    wrapped = (
        "```json\n"
        + json.dumps(
            {"domain": "novel", "intent": "characters_only", "goal": "设计三个角色"},
            ensure_ascii=False,
        )
        + "\n```"
    )
    transport = StubTransport(wrapped)

    brief = _analyzer(transport).analyze(
        "设计3个角色",
        project_id=PROJECT_ID,
        run_id="run-005",
        brief_id="run-005:brief",
    )

    assert brief.degraded is False
    assert brief.degraded_reason is None
    assert brief.intent == "characters_only"


def test_analyzer_reports_parse_failure_reason() -> None:
    """Malformed provider output degrades with an explicit parse reason."""
    brief = _analyzer(StubTransport("这不是 JSON")).analyze(
        "设计3个角色",
        project_id=PROJECT_ID,
        run_id="run-006",
        brief_id="run-006:brief",
    )

    assert brief.degraded is True
    assert brief.degraded_reason == "parse_failure"
    assert brief.intent == "characters_only"  # heuristic fallback still applies


def test_analyzer_reports_provider_failure_reason() -> None:
    brief = _analyzer(BrokenTransport()).analyze(
        "写一本小说",
        project_id=PROJECT_ID,
        run_id="run-007",
        brief_id="run-007:brief",
    )

    assert brief.degraded is True
    assert brief.degraded_reason == "provider_failure"


# -- out-of-distribution defense ---------------------------------------------


@pytest.mark.parametrize(
    "requirements",
    [
        "今天天气真好，中午吃什么？",  # everyday chatter
        "asdfghjklqwertyuiop",  # gibberish
        "哈哈哈哈哈哈",  # laughter
        "你好",  # greeting
        "1234567890",  # digits
        " ",  # whitespace only
    ],
)
def test_heuristic_intent_rejects_out_of_distribution(requirements: str) -> None:
    from novel_agent.agent.brief import _heuristic_intent

    assert _heuristic_intent(requirements) is None


@pytest.mark.parametrize(
    "requirements",
    [
        "帮我写个故事",
        "写一部短篇",
        "写一本20章的奇幻小说",
        "基于已有大纲写第一章正文",
    ],
)
def test_heuristic_intent_keeps_recognizing_creative_requests(requirements: str) -> None:
    from novel_agent.agent.brief import _heuristic_intent

    assert _heuristic_intent(requirements) == "full_novel"


@pytest.mark.parametrize(
    "requirements",
    [
        "今天天气真好，中午吃什么？",
        "asdfghjklqwertyuiop",
    ],
)
def test_analyzer_marks_unrecognized_intent_when_provider_fails(requirements: str) -> None:
    brief = _analyzer(BrokenTransport()).analyze(
        requirements,
        project_id=PROJECT_ID,
        run_id="run-ood-1",
        brief_id="run-ood-1:brief",
    )

    assert brief.degraded is True
    assert brief.intent == "unrecognized"


def test_analyzer_accepts_provider_declared_unrecognized_intent() -> None:
    """The provider may itself conclude the request is out-of-distribution."""
    transport = StubTransport(
        json.dumps(
            {"intent": "unrecognized", "goal": "无关内容", "constraints": []},
            ensure_ascii=False,
        )
    )

    brief = _analyzer(transport).analyze(
        "今天天气不错",
        project_id=PROJECT_ID,
        run_id="run-ood-2",
        brief_id="run-ood-2:brief",
    )

    assert brief.degraded is False
    assert brief.intent == "unrecognized"


def test_analyzer_rejects_invalid_intent_from_provider() -> None:
    """An intent outside the known enum degrades and falls back to heuristics."""
    transport = StubTransport(
        json.dumps({"intent": "write_a_poem", "goal": "写诗"}, ensure_ascii=False)
    )

    brief = _analyzer(transport).analyze(
        "今天天气真好，中午吃什么？",
        project_id=PROJECT_ID,
        run_id="run-ood-3",
        brief_id="run-ood-3:brief",
    )

    assert brief.degraded is True
    assert brief.intent == "unrecognized"
