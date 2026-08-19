from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from types import ModuleType
from typing import assert_never

import pytest

from novel_agent.adapters import (
    CharacterAdaptation,
    CharacterGenerationRequest,
    CreativeParseRoute,
    LegacyCreativeAdapter,
    LegacyEvaluationAdapter,
    LegacyEvaluationRequest,
    M0CandidateExtractionRequest,
    NovelSaveCandidateAdapter,
    SchemaFailure,
    StructuredCreativeAdapter,
    WorldAdaptation,
    WorldGenerationRequest,
)
from novel_agent.contracts import MemoryLayer
from novel_agent.pathing import ArtifactId, ProjectId


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_legacy_world_adapter_rejects_prose_without_legacy_write(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a legacy generator emits prose rather than the governed JSON contract.
    raw_world = "Legacy world prose with one stable rule."
    writes: list[str] = []
    module = ModuleType("generate_world")
    module.expand_world = lambda _idea: raw_world
    module.save_world = lambda *_args: writes.append("save_world")
    monkeypatch.setitem(sys.modules, "generate_world", module)
    request = WorldGenerationRequest(
        idea="A governed idea",
        world_id="world-legacy-001",
        brief_id="brief-001",
        title="Legacy World",
        created_at=FIXED_TIME,
        source_id="legacy-world-call",
    )

    # When: the governed legacy wrapper invokes the existing read/generate callable.
    result = LegacyCreativeAdapter(StructuredCreativeAdapter()).generate_world(request)

    # Then: byte-exact source evidence is retained but cannot become a typed contract.
    match result:
        case SchemaFailure(raw_artifact=raw, parse_routes=parse_routes):
            assert raw.raw_bytes == raw_world.encode()
            assert parse_routes == (CreativeParseRoute.STRICT_JSON, CreativeParseRoute.JSON_REPAIR)
        case WorldAdaptation():
            pytest.fail("legacy prose unexpectedly became a world contract")
        case unreachable:
            assert_never(unreachable)
    assert writes == []


def test_legacy_character_adapter_rejects_unversioned_list(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: an old list-shaped character response without the governed CharacterSet envelope.
    import generate_characters as character_module

    raw_characters = json.dumps([{"name": "Hero", "role": "lead"}])
    monkeypatch.setattr(character_module, "read_world", lambda *_args: "A governed city.")
    monkeypatch.setattr(character_module, "access_deepseek", lambda *_args, **_kwargs: raw_characters)
    request = CharacterGenerationRequest(
        project="novel-one",
        world_name="legacy-world",
        requirements="Create a cast.",
        count=1,
        character_set_id="cast-001",
        world_id="world-001",
        created_at=FIXED_TIME,
        source_id="legacy-cast-call",
    )

    # When: the governed adapter receives the historic list output.
    result = LegacyCreativeAdapter(StructuredCreativeAdapter()).generate_characters(request)

    # Then: no heuristic conversion turns it into M2-ready state.
    match result:
        case SchemaFailure(raw_artifact=raw):
            assert raw.raw_bytes == raw_characters.encode()
        case CharacterAdaptation():
            pytest.fail("legacy character list unexpectedly became a character set")
        case unreachable:
            assert_never(unreachable)


def test_legacy_evaluator_and_m0_candidate_adapter_do_not_write_legacy_state(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: an evaluator response and a legacy writer sentinel.
    evaluator_module = ModuleType("evaluate")
    evaluator_module.llm_evaluate_chapter = lambda *_args, **_kwargs: {
        "scores": {"continuity": 4.0},
        "reasons": {"continuity": "consistent"},
        "avg_score": 4.0,
    }
    history_module = ModuleType("tools_history_save")
    writes: list[str] = []
    history_module.novel_save = lambda *_args, **_kwargs: writes.append("novel_save")
    monkeypatch.setitem(sys.modules, "evaluate", evaluator_module)
    monkeypatch.setitem(sys.modules, "tools_history_save", history_module)

    # When: governed evaluation and candidate extraction consume the draft.
    evaluation = LegacyEvaluationAdapter().evaluate(
        LegacyEvaluationRequest(
            chapter_body="A chapter that keeps continuity.",
            characters_json='{"hero":{"name":"Hero"}}',
            chapter_label="chapter-1",
            chapter_requirement="Keep continuity.",
        )
    )
    candidate = NovelSaveCandidateAdapter().extract(
        M0CandidateExtractionRequest(
            run_id="run-001",
            project_id=ProjectId("novel-one"),
            source_artifact_id=ArtifactId("chapter-001"),
            source_artifact_revision=1,
            chapter_body="A chapter that keeps continuity.",
            subject_id="chapter-001",
            predicate="summary",
            value_json='{"summary":"A chapter that keeps continuity."}',
            target_layer=MemoryLayer.M3_CONFIRMED,
            created_at=FIXED_TIME,
        )
    )

    # Then: results are typed M0-ready data, never an implicit save or promotion.
    assert evaluation.average_score == 4.0
    assert evaluation.raw_bytes == b'{"avg_score":4.0,"reasons":{"continuity":"consistent"},"scores":{"continuity":4.0}}'
    assert candidate.proposal.source_excerpt == candidate.request.chapter_body
    assert candidate.proposal.source_start == 0
    assert candidate.proposal.source_end == len(candidate.request.chapter_body)
    assert writes == []
