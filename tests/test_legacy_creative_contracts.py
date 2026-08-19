from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from types import ModuleType
from typing import assert_never

import pytest

from novel_agent.adapters import (
    ChapterAdaptation,
    ChapterDraft,
    ChapterGenerationRequest,
    CharacterAdaptation,
    CharacterGenerationRequest,
    LegacyCreativeAdapter,
    OutlineAdaptation,
    OutlineGenerationRequest,
    SchemaFailure,
    StructuredCreativeAdapter,
)
from novel_agent.contracts import CharacterSet


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


def _provenance(source_id: str) -> dict[str, str]:
    return {
        "source_kind": "provider_response",
        "source_id": source_id,
        "source_sha256": "a" * 64,
        "observed_at": FIXED_TIME.isoformat(),
    }


def test_legacy_character_outline_and_chapter_adapters_preserve_raw_outputs_without_writers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: legacy generation modules that honor the governed output contracts and writer sentinels.
    import generate_characters as characters_module

    writes: list[str] = []
    captured: list[str] = []
    character_prompts: list[str] = []
    character_raw = json.dumps(
        {
            "character_set_id": "cast-001",
            "world_id": "world-001",
            "characters": [
                {
                    "character_id": "character-001",
                    "name": "Hero",
                    "role": "lead",
                    "goal": "solve the mystery",
                    "traits": ["curious"],
                    "created_at": FIXED_TIME.isoformat(),
                    "provenance": _provenance("legacy-cast-call"),
                }
            ],
            "created_at": FIXED_TIME.isoformat(),
            "provenance": _provenance("legacy-cast-call"),
        }
    )
    monkeypatch.setattr(characters_module, "read_world", lambda *_args: "A governed city.")
    monkeypatch.setattr(
        characters_module,
        "access_deepseek",
        lambda prompt, **_kwargs: character_prompts.append(prompt) or character_raw,
    )
    monkeypatch.setattr(characters_module, "save_character", lambda *_args: writes.append("save_character"))
    outline_module = ModuleType("generate_outline")
    outline_module.generate_outline = lambda *_args: json.dumps(
        {
            "plan_id": "plan-001",
            "brief_id": "brief-001",
            "world_id": "world-001",
            "character_set_id": "cast-001",
            "approval_status": "validated",
            "chapters": [
                {
                    "chapter_plan_id": "chapter-plan-001",
                    "chapter_number": 1,
                    "title": "Opening",
                    "goal": "Introduce the mystery.",
                    "planned_events": ["A clue arrives."],
                    "created_at": FIXED_TIME.isoformat(),
                    "provenance": _provenance("legacy-outline-call"),
                    "evidence": [
                        {
                            "evidence_id": "evidence-001",
                            "artifact_id": "world-raw",
                            "artifact_revision": 1,
                            "source_sha256": "a" * 64,
                            "recorded_at": FIXED_TIME.isoformat(),
                        }
                    ],
                }
            ],
            "created_at": FIXED_TIME.isoformat(),
            "provenance": _provenance("legacy-outline-call"),
        }
    )
    outline_module.save_outline = lambda *_args: writes.append("save_outline")
    writing_module = ModuleType("tools_writting")
    writing_module.writting_novel = lambda _last, request, *_args: captured.append(request) or json.dumps(
        {
            "chapter_draft_id": "draft-001",
            "chapter_plan_id": "chapter-plan-001",
            "body": "A governed chapter.",
            "created_at": FIXED_TIME.isoformat(),
            "provenance": _provenance("legacy-chapter-call"),
        }
    )
    history_module = ModuleType("tools_history_save")
    history_module.novel_save = lambda *_args, **_kwargs: writes.append("novel_save")
    monkeypatch.setitem(sys.modules, "generate_outline", outline_module)
    monkeypatch.setitem(sys.modules, "tools_writting", writing_module)
    monkeypatch.setitem(sys.modules, "tools_history_save", history_module)
    adapter = LegacyCreativeAdapter(StructuredCreativeAdapter())

    # When: each legacy source is called through its governed wrapper.
    characters = adapter.generate_characters(
        CharacterGenerationRequest(
            project="novel-one",
            world_name="legacy-world",
            requirements="Create a cast.",
            count=1,
            character_set_id="cast-001",
            world_id="world-001",
            created_at=FIXED_TIME,
            source_id="legacy-cast-call",
        )
    )
    outline = adapter.generate_outline(
        OutlineGenerationRequest(
            project="novel-one",
            world_name="legacy-world",
            preferences="Create an outline.",
            chapter_count=1,
            plan_id="plan-001",
            brief_id="brief-001",
            world_id="world-001",
            character_set_id="cast-001",
            created_at=FIXED_TIME,
            source_id="legacy-outline-call",
        )
    )
    chapter = adapter.generate_chapter(
        ChapterGenerationRequest(
            last_passage="",
            this_passage_request="Write chapter one.",
            article_name="Novel",
            chapter_number="1",
            chapter_name="Opening",
            characters=("Hero",),
            world="legacy-world",
            styles="plain",
            words=200,
            is_npc=False,
            project="novel-one",
            chapter_draft_id="draft-001",
            chapter_plan_id="chapter-plan-001",
            created_at=FIXED_TIME,
            source_id="legacy-chapter-call",
        )
    )

    # Then: every wrapper accepts only strict versioned JSON and leaves writers untouched.
    match characters:
        case CharacterAdaptation(raw=raw, value=value):
            assert raw.raw_bytes == character_raw.encode()
            assert value.characters[0].name == "Hero"
        case SchemaFailure() as failure:
            pytest.fail(f"legacy character adaptation failed: {failure.reason_code}")
        case unreachable:
            assert_never(unreachable)
    match outline:
        case OutlineAdaptation(raw=raw, value=value):
            assert raw.raw_bytes.startswith(b"{")
            assert value.approval_status.value == "validated"
        case SchemaFailure() as failure:
            pytest.fail(f"legacy outline adaptation failed: {failure.reason_code}")
        case unreachable:
            assert_never(unreachable)
    match chapter:
        case ChapterAdaptation(raw=raw, value=value):
            assert raw.raw_bytes.startswith(b"{")
            assert value.body == "A governed chapter."
        case SchemaFailure() as failure:
            pytest.fail(f"legacy chapter adaptation failed: {failure.reason_code}")
        case unreachable:
            assert_never(unreachable)
    chapter_schema = json.dumps(
        ChapterDraft.model_json_schema(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    character_schema = json.dumps(
        CharacterSet.model_json_schema(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    assert chapter_schema in captured[0]
    assert character_schema in character_prompts[0]
    assert writes == []
