from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import assert_never

import pytest

from novel_agent.adapters import (
    ChapterAdaptation,
    CharacterAdaptation,
    CreativeParseRoute,
    OutlineAdaptation,
    SchemaFailure,
    SchemaRepairRequest,
    StructuredCreativeAdapter,
    WorldAdaptation,
)
from novel_agent.contracts import ProvenanceKind
from novel_agent.pathing import ProjectId
from novel_agent.storage import connect_project_database, migrate


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
FIXED_HASH = "a" * 64


def _provenance(source_id: str) -> dict[str, str]:
    return {
        "source_kind": ProvenanceKind.PROVIDER_RESPONSE.value,
        "source_id": source_id,
        "source_sha256": FIXED_HASH,
        "observed_at": FIXED_TIME.isoformat(),
    }


def _world_json() -> str:
    return json.dumps(
        {
            "world_id": "world-001",
            "brief_id": "brief-001",
            "title": "Test World",
            "premise": "A governed premise.",
            "rules": ["Rule one"],
            "constraints": ["Constraint one"],
            "created_at": FIXED_TIME.isoformat(),
            "provenance": _provenance("world-response"),
        }
    )


def _characters_json() -> str:
    return json.dumps(
        {
            "character_set_id": "cast-001",
            "world_id": "world-001",
            "characters": [
                {
                    "character_id": "character-001",
                    "name": "Hero",
                    "role": "protagonist",
                    "goal": "Solve the mystery.",
                    "traits": ["curious"],
                    "created_at": FIXED_TIME.isoformat(),
                    "provenance": _provenance("character-response"),
                }
            ],
            "created_at": FIXED_TIME.isoformat(),
            "provenance": _provenance("cast-response"),
        }
    )


def _outline_json() -> str:
    return json.dumps(
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
                    "provenance": _provenance("outline-response"),
                    "evidence": [
                        {
                            "evidence_id": "evidence-001",
                            "artifact_id": "world-raw",
                            "artifact_revision": 1,
                            "source_sha256": FIXED_HASH,
                            "recorded_at": FIXED_TIME.isoformat(),
                        }
                    ],
                }
            ],
            "created_at": FIXED_TIME.isoformat(),
            "provenance": _provenance("plan-response"),
        }
    )


def _chapter_json() -> str:
    return json.dumps(
        {
            "chapter_draft_id": "chapter-draft-001",
            "chapter_plan_id": "chapter-plan-001",
            "body": "A deterministic first chapter.",
            "created_at": FIXED_TIME.isoformat(),
            "provenance": _provenance("chapter-response"),
        }
    )


@dataclass(frozen=True, slots=True)
class RecordingRepairer:
    response: str
    calls: list[SchemaRepairRequest]

    def repair(self, request: SchemaRepairRequest) -> str:
        self.calls.append(request)
        return self.response


def test_world_cast_outline_and_chapter_adapters_validate() -> None:
    # Given: valid versioned JSON returned by each legacy-compatible creative stage.
    adapter = StructuredCreativeAdapter()

    # When: the governed adapter validates the raw response for every stage.
    world = adapter.adapt_world(_world_json())
    characters = adapter.adapt_characters(_characters_json())
    outline = adapter.adapt_outline(_outline_json())
    chapter = adapter.adapt_chapter(_chapter_json())

    # Then: each result is typed and preserves immutable byte-exact raw evidence.
    match world:
        case WorldAdaptation(raw=world_raw, value=world_value):
            assert world_value.title == "Test World"
            assert world_raw.raw_bytes == _world_json().encode()
            assert world_raw.content_sha256 == hashlib.sha256(_world_json().encode()).hexdigest()
        case SchemaFailure() as failure:
            pytest.fail(f"world adaptation failed: {failure.reason_code}")
        case unreachable:
            assert_never(unreachable)
    match characters:
        case CharacterAdaptation(value=character_value):
            assert character_value.characters[0].name == "Hero"
        case SchemaFailure() as failure:
            pytest.fail(f"character adaptation failed: {failure.reason_code}")
        case unreachable:
            assert_never(unreachable)
    match outline:
        case OutlineAdaptation(value=outline_value):
            assert outline_value.chapters[0].chapter_number == 1
        case SchemaFailure() as failure:
            pytest.fail(f"outline adaptation failed: {failure.reason_code}")
        case unreachable:
            assert_never(unreachable)
    match chapter:
        case ChapterAdaptation(value=chapter_value):
            assert chapter_value.body == "A deterministic first chapter."
        case SchemaFailure() as failure:
            pytest.fail(f"chapter adaptation failed: {failure.reason_code}")
        case unreachable:
            assert_never(unreachable)


def test_repairable_malformed_json_uses_only_bounded_json_repair() -> None:
    # Given: a syntactically repairable response and a repair prompt that must remain unused.
    raw = _world_json().replace('"premise":', "premise:")
    calls: list[SchemaRepairRequest] = []
    adapter = StructuredCreativeAdapter(RecordingRepairer(_world_json(), calls))

    # When: strict JSON fails but json_repair restores the bounded contract shape.
    result = adapter.adapt_world(raw)

    # Then: the repaired result is typed without a schema-repair provider call.
    match result:
        case WorldAdaptation(parse_route=parse_route):
            assert parse_route is CreativeParseRoute.JSON_REPAIR
        case SchemaFailure() as failure:
            pytest.fail(f"repairable response failed: {failure.reason_code}")
        case unreachable:
            assert_never(unreachable)
    assert calls == []


def test_unrepaired_schema_failure_runs_once_without_governed_write(tmp_path: Path) -> None:
    # Given: a schema-invalid response, one bounded repair response, and a migrated project database.
    calls: list[SchemaRepairRequest] = []
    adapter = StructuredCreativeAdapter(RecordingRepairer("{still: invalid}", calls))
    project_dir = tmp_path / "projects" / "novel-one"
    with connect_project_database(project_dir) as connection:
        migrate(connection)

        # When: all parse paths are exhausted without a valid contract.
        result = adapter.adapt_world('{"world_id":"world-001"}')

        # Then: it pauses as a typed schema failure and cannot create state or current pointers.
        match result:
            case SchemaFailure(parse_routes=parse_routes):
                assert parse_routes == (
                    CreativeParseRoute.STRICT_JSON,
                    CreativeParseRoute.JSON_REPAIR,
                    CreativeParseRoute.SCHEMA_REPAIR,
                )
            case WorldAdaptation():
                pytest.fail("invalid response unexpectedly became a world contract")
            case unreachable:
                assert_never(unreachable)
        assert len(calls) == 1
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM memory_records").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM artifact_outbox").fetchone() == (0,)


def test_schema_repair_response_is_revalidated_before_becoming_a_contract() -> None:
    # Given: schema-invalid JSON and a targeted repair provider that returns a valid world contract.
    calls: list[SchemaRepairRequest] = []
    adapter = StructuredCreativeAdapter(RecordingRepairer(_world_json(), calls))

    # When: the single schema repair response crosses the same Pydantic boundary.
    result = adapter.adapt_world('{"world_id":"world-001"}')

    # Then: it is accepted only after the strict revalidation and has one repair trace.
    match result:
        case WorldAdaptation(parse_route=parse_route):
            assert parse_route is CreativeParseRoute.SCHEMA_REPAIR
        case SchemaFailure() as failure:
            pytest.fail(f"schema repair failed: {failure.reason_code}")
        case unreachable:
            assert_never(unreachable)
    assert len(calls) == 1
    assert calls[0].expected_contract == "WorldBible"
