from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import assert_never

import pytest

from novel_agent.adapters import (
    ChapterAdaptation,
    CharacterAdaptation,
    CreativeParseRoute,
    GovernedCreativeAdapter,
    OutlineAdaptation,
    SchemaFailure,
    WorldAdaptation,
)
from novel_agent.chapter_models import ChapterDraftRequest, MemoryExtractionRequest, StagedChapter
from novel_agent.context_models import BudgetLedger, ContextBuildResult
from novel_agent.contracts import (
    ChapterPlan,
    EvidenceRef,
    MemoryLayer,
    PlanApprovalStatus,
    ProvenanceKind,
    StoryBrief,
)
from novel_agent.memory_models import MemoryFactKind, PromotionAuthority
from novel_agent.pathing import ArtifactId, ProjectId
from novel_agent.provider_types import ProviderTransportError, ProviderTransportResult


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")


class StubTransport:
    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, request) -> ProviderTransportResult:
        self.calls.append(request.model_name)
        content = self._responses.pop(0)
        return ProviderTransportResult(content=content, model_name="test-model", prompt_tokens=1, completion_tokens=1)


class FailingTransport:
    def generate(self, _request) -> ProviderTransportResult:
        raise ProviderTransportError(kind=ProviderTransportError.kind.TRANSIENT, detail="wire down")


def _adapter(transport) -> GovernedCreativeAdapter:
    return GovernedCreativeAdapter(
        transport=transport,
        now=lambda: FIXED_TIME,
        run_id="run-001",
        project_id=PROJECT_ID,
        default_character_count=2,
    )


_CHARACTERS_RAW: str = json.dumps(
    {
        "characters": [
            {"name": "林晚", "role": "主角", "goal": "查明学院禁地之谜", "traits": ["冷静", "执着"]},
            {"name": "阿泽", "role": "挚友", "goal": "守护林晚", "traits": ["热情"]},
        ]
    },
    ensure_ascii=False,
)


def _brief(chapter_count: int = 2) -> StoryBrief:
    return StoryBrief(
        brief_id="run-001:brief",
        project_id=PROJECT_ID,
        user_requirements="一个关于魔法学院的成长故事",
        genre="fiction",
        target_chapter_count=chapter_count,
        created_at=FIXED_TIME,
        provenance={
            "source_kind": "user_request",
            "source_id": "run-001",
            "source_sha256": "a" * 64,
            "observed_at": FIXED_TIME.isoformat(),
        },
    )


def _world() -> WorldAdaptation:
    adapter = _adapter(
        StubTransport(
            json.dumps(
                {
                    "title": "晨曦学院",
                    "premise": "一座漂浮在云海之上的魔法学院。",
                    "rules": ["魔力来自星辰", "禁忌魔法不可触碰"],
                    "constraints": ["不出现现代科技"],
                },
                ensure_ascii=False,
            )
        )
    )
    result = adapter.generate_world(_brief())
    match result:
        case WorldAdaptation():
            return result
        case SchemaFailure():
            pytest.fail("valid world content unexpectedly failed")
        case unreachable:
            assert_never(unreachable)


def test_generate_characters_uses_requested_count() -> None:
    from novel_agent.adapters.governed_creative import _requested_character_count

    assert _requested_character_count("为我设计3个角色，两女一男") == 3
    assert _requested_character_count("三名角色，一个男主角") == 3
    assert _requested_character_count("六个角色") == 6
    assert _requested_character_count("两个女主角") == 2
    assert _requested_character_count("写一本魔法学院小说") is None


def test_generate_world_assembles_governed_world_bible() -> None:
    raw = json.dumps(
        {
            "title": "晨曦学院",
            "premise": "一座漂浮在云海之上的魔法学院。",
            "rules": ["魔力来自星辰", "禁忌魔法不可触碰"],
            "constraints": ["不出现现代科技"],
        },
        ensure_ascii=False,
    )
    adapter = _adapter(StubTransport(raw))

    result = adapter.generate_world(_brief())

    match result:
        case WorldAdaptation(raw=artifact, value=world, parse_route=route):
            assert world.world_id == "run-001:brief:world"
            assert world.brief_id == "run-001:brief"
            assert world.title == "晨曦学院"
            assert world.premise.startswith("一座漂浮")
            assert world.rules == ("魔力来自星辰", "禁忌魔法不可触碰")
            assert world.constraints == ("不出现现代科技",)
            assert world.created_at == FIXED_TIME
            assert world.provenance.source_kind is ProvenanceKind.PROVIDER_RESPONSE
            assert artifact.raw_bytes == world.model_dump_json().encode()
            assert route is CreativeParseRoute.STRICT_JSON
        case SchemaFailure():
            pytest.fail("valid world content unexpectedly failed")
        case unreachable:
            assert_never(unreachable)


def test_generate_world_recovers_wrapped_json_via_json_repair() -> None:
    raw = "```json\n" + json.dumps({"title": "晨曦学院", "premise": "设定", "rules": [], "constraints": []}) + "\n```"
    adapter = _adapter(StubTransport(raw))

    result = adapter.generate_world(_brief())

    match result:
        case WorldAdaptation(value=world, parse_route=route):
            assert world.title == "晨曦学院"
            assert route is CreativeParseRoute.JSON_REPAIR
        case SchemaFailure():
            pytest.fail("repairable world content unexpectedly failed")
        case unreachable:
            assert_never(unreachable)


def test_generate_world_schema_failure_retains_raw_bytes() -> None:
    raw_world = "Legacy world prose with no governed shape."
    adapter = _adapter(StubTransport(raw_world))

    result = adapter.generate_world(_brief())

    match result:
        case SchemaFailure(raw_artifact=artifact, expected_contract="WorldBible"):
            assert artifact.raw_bytes == raw_world.encode()
        case WorldAdaptation():
            pytest.fail("prose unexpectedly became a world contract")
        case unreachable:
            assert_never(unreachable)


def test_generate_characters_assembles_character_set() -> None:
    raw = json.dumps(
        {
            "characters": [
                {"name": "林晚", "role": "主角", "goal": "查明学院禁地之谜", "traits": ["冷静", "执着"]},
                {"name": "阿泽", "role": "挚友", "goal": "守护林晚", "traits": ["热情"]},
            ]
        },
        ensure_ascii=False,
    )
    adapter = _adapter(StubTransport(raw))
    world = _world().value

    result = adapter.generate_characters(_brief(), world)

    match result:
        case CharacterAdaptation(raw=artifact, value=characters, parse_route=route):
            assert characters.character_set_id == "run-001:brief:cast"
            assert characters.world_id == world.world_id
            assert tuple(c.name for c in characters.characters) == ("林晚", "阿泽")
            assert characters.characters[0].character_id == "run-001:brief:character:1"
            assert characters.characters[1].character_id == "run-001:brief:character:2"
            assert characters.characters[0].traits == ("冷静", "执着")
            assert artifact.raw_bytes == characters.model_dump_json().encode()
            assert route is CreativeParseRoute.STRICT_JSON
        case SchemaFailure():
            pytest.fail("valid cast content unexpectedly failed")
        case unreachable:
            assert_never(unreachable)


def test_generate_outline_assembles_validated_plan() -> None:
    raw = json.dumps(
        {
            "chapters": [
                {"title": "初入学院", "goal": "主角抵达魔法学院", "planned_events": ["报到", "认识室友"]},
                {"title": "禁地之谜", "goal": "第一个冲突出现", "planned_events": ["深夜发现禁地"]},
            ]
        },
        ensure_ascii=False,
    )
    adapter = _adapter(StubTransport(_CHARACTERS_RAW, raw))
    brief = _brief(chapter_count=2)
    world = _world().value
    characters = adapter.generate_characters(brief, world)

    result = adapter.generate_outline(brief, world, characters.value)

    match result:
        case OutlineAdaptation(raw=artifact, value=plan, parse_route=route):
            assert plan.plan_id == "run-001:brief:plan"
            assert plan.brief_id == brief.brief_id
            assert plan.world_id == world.world_id
            assert plan.character_set_id == characters.value.character_set_id
            assert plan.approval_status is PlanApprovalStatus.VALIDATED
            assert len(plan.chapters) == 2
            assert tuple(c.chapter_number for c in plan.chapters) == (1, 2)
            assert tuple(c.chapter_plan_id for c in plan.chapters) == (
                "run-001:brief:chapter:1",
                "run-001:brief:chapter:2",
            )
            assert plan.chapters[0].title == "初入学院"
            assert plan.chapters[0].planned_events == ("报到", "认识室友")
            assert len(plan.chapters[0].evidence) == 1
            assert plan.chapters[0].evidence[0].artifact_id == world.world_id
            assert route is CreativeParseRoute.STRICT_JSON
        case SchemaFailure():
            pytest.fail("valid outline content unexpectedly failed")
        case unreachable:
            assert_never(unreachable)


def test_generate_outline_rejects_chapter_count_mismatch() -> None:
    raw = json.dumps(
        {
            "chapters": [
                {"title": "只有一章", "goal": "不足两章", "planned_events": ["事件"]},
            ]
        },
        ensure_ascii=False,
    )
    adapter = _adapter(StubTransport(_CHARACTERS_RAW, raw))
    brief = _brief(chapter_count=2)
    world = _world().value
    characters = adapter.generate_characters(brief, world)

    result = adapter.generate_outline(brief, world, characters.value)

    match result:
        case SchemaFailure(reason_code="outline_chapter_count_mismatch"):
            pass
        case OutlineAdaptation():
            pytest.fail("chapter count mismatch unexpectedly produced a plan")
        case unreachable:
            assert_never(unreachable)


def test_generate_chapter_uses_prose_body_and_strips_fences() -> None:
    chapter_plan = ChapterPlan(
        chapter_plan_id="run-001:brief:chapter:1",
        chapter_number=1,
        title="初入学院",
        goal="主角抵达魔法学院",
        planned_events=("报到",),
        created_at=FIXED_TIME,
        provenance={
            "source_kind": "provider_response",
            "source_id": "run-001:brief:chapter:1",
            "source_sha256": "b" * 64,
            "observed_at": FIXED_TIME.isoformat(),
        },
        evidence=(
            EvidenceRef(
                evidence_id="evidence-1",
                artifact_id="run-001:brief:world",
                artifact_revision=1,
                source_sha256="c" * 64,
                recorded_at=FIXED_TIME,
            ),
        ),
    )
    request = ChapterDraftRequest(
        run_id="run-001",
        task_id="run-001:chapter:run-001:brief:chapter:1",
        chapter_plan=chapter_plan,
        context=ContextBuildResult(
            items=(),
            exclusions=(),
            prompt="世界观记忆：晨光学院。",
            digest="d" * 64,
            tokenizer_version="test",
            budget_ledger=BudgetLedger(0, 0, 0, 0, 0, 0),
        ),
        target_chinese_characters=3000,
        repair_brief=None,
    )
    adapter = _adapter(StubTransport("```\n我站在学院门口，晨风拂过衣角。\n```"))

    result = adapter.generate(request)

    match result:
        case ChapterAdaptation(raw=artifact, value=draft, parse_route=route):
            assert draft.body == "我站在学院门口，晨风拂过衣角。"
            assert draft.chapter_draft_id == "run-001:chapter:run-001:brief:chapter:1:draft"
            assert draft.chapter_plan_id == "run-001:brief:chapter:1"
            # Chapters are stored as a structured JSON envelope (canonical
            # format since the JSON unification), with the prose in the body.
            assert route is CreativeParseRoute.STRICT_JSON
            envelope = json.loads(artifact.raw_bytes.decode("utf-8"))
            assert envelope["chapter_plan_id"] == "run-001:brief:chapter:1"
            assert envelope["body"] == "我站在学院门口，晨风拂过衣角。"
        case SchemaFailure():
            pytest.fail("valid prose chapter unexpectedly failed")
        case unreachable:
            assert_never(unreachable)


def test_extract_uses_llm_summary_inside_candidate_envelope() -> None:
    chapter_plan = ChapterPlan(
        chapter_plan_id="run-001:brief:chapter:1",
        chapter_number=1,
        title="初入学院",
        goal="主角抵达魔法学院",
        planned_events=("报到",),
        created_at=FIXED_TIME,
        provenance={
            "source_kind": "provider_response",
            "source_id": "run-001:brief:chapter:1",
            "source_sha256": "b" * 64,
            "observed_at": FIXED_TIME.isoformat(),
        },
        evidence=(
            EvidenceRef(
                evidence_id="evidence-1",
                artifact_id="run-001:brief:world",
                artifact_revision=1,
                source_sha256="c" * 64,
                recorded_at=FIXED_TIME,
            ),
        ),
    )
    staged = StagedChapter(
        artifact_id=ArtifactId("chapter-001"),
        revision=1,
        task_id="run-001:chapter:run-001:brief:chapter:1",
        chapter_plan_id=chapter_plan.chapter_plan_id,
        chapter_number=1,
        raw_bytes="正文内容".encode(),
        raw_sha256="e" * 64,
        body="正文内容",
    )
    request = MemoryExtractionRequest(
        run_id="run-001",
        plan_revision=1,
        chapter_plan=chapter_plan,
        staged=staged,
    )
    adapter = _adapter(StubTransport(json.dumps({"summary": "主角抵达学院并完成报到。"}, ensure_ascii=False)))

    proposals = adapter.extract(request)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.kind is MemoryFactKind.CHAPTER_SUMMARY
    assert proposal.target_layer is MemoryLayer.M3_CONFIRMED
    assert proposal.authority is PromotionAuthority.ACCEPTED_CHAPTER
    assert proposal.subject_id == chapter_plan.chapter_plan_id
    assert proposal.predicate == "summary"
    assert json.loads(proposal.value_json) == {"summary": "主角抵达学院并完成报到。"}
    assert proposal.plan_revision == 1
    assert proposal.source_artifact_id == staged.artifact_id


def test_extract_falls_back_to_truncation_when_summary_call_fails() -> None:
    chapter_plan = ChapterPlan(
        chapter_plan_id="run-001:brief:chapter:2",
        chapter_number=2,
        title="第二幕",
        goal="推进",
        planned_events=(),
        created_at=FIXED_TIME,
        provenance={
            "source_kind": "provider_response",
            "source_id": "run-001:brief:chapter:2",
            "source_sha256": "f" * 64,
            "observed_at": FIXED_TIME.isoformat(),
        },
        evidence=(
            EvidenceRef(
                evidence_id="evidence-2",
                artifact_id="run-001:brief:world",
                artifact_revision=1,
                source_sha256="c" * 64,
                recorded_at=FIXED_TIME,
            ),
        ),
    )
    staged = StagedChapter(
        artifact_id=ArtifactId("chapter-002"),
        revision=1,
        task_id="run-001:chapter:run-001:brief:chapter:2",
        chapter_plan_id=chapter_plan.chapter_plan_id,
        chapter_number=2,
        raw_bytes="很长很长的正文内容".encode(),
        raw_sha256="g" * 64,
        body="很长很长的正文内容",
    )
    adapter = _adapter(FailingTransport())

    proposals = adapter.extract(
        MemoryExtractionRequest(run_id="run-001", plan_revision=1, chapter_plan=chapter_plan, staged=staged)
    )

    assert len(proposals) == 1
    assert json.loads(proposals[0].value_json) == {"summary": "很长很长的正文内容"}
