from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import assert_never

import pytest

from novel_agent.adapters import (
    CreativeParseRoute,
    LegacyCreativeAdapter,
    SchemaFailure,
    SchemaRepairRequest,
    StructuredCreativeAdapter,
    WorldAdaptation,
    WorldGenerationRequest,
)
from novel_agent.adapters.schema_repair import ProviderSchemaRepairer
from novel_agent.provider_types import (
    ProviderFailureKind,
    ProviderTransportError,
    ProviderTransportResult,
)


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


class RepairTransport:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, request) -> ProviderTransportResult:
        self.prompts.append(request.prompt)
        return ProviderTransportResult(
            content=self.response,
            model_name="repair-test",
            prompt_tokens=3,
            completion_tokens=2,
        )


class BrokenTransport:
    def generate(self, _request) -> ProviderTransportResult:
        raise ProviderTransportError(kind=ProviderFailureKind.TRANSIENT, detail="wire down")


def _repairer(transport) -> ProviderSchemaRepairer:
    return ProviderSchemaRepairer(transport=transport, now=lambda: FIXED_TIME, run_id="run-001")


def _world_request() -> WorldGenerationRequest:
    return WorldGenerationRequest(
        idea="A governed idea",
        world_id="world-001",
        brief_id="brief-001",
        title="Legacy World",
        created_at=FIXED_TIME,
        source_id="legacy-world-call",
    )


def test_provider_schema_repairer_forwards_schema_and_raw_text() -> None:
    # Given: a provider transport that answers with repaired JSON.
    repaired = json.dumps(
        {
            "world_id": "world-001",
            "brief_id": "brief-001",
            "title": "Fixed World",
            "premise": "A coherent premise.",
            "rules": ("rule-one",),
            "constraints": (),
            "created_at": FIXED_TIME.isoformat(),
            "provenance": {
                "source_kind": "provider_response",
                "source_id": "legacy-world-call",
                "source_sha256": "a" * 64,
                "observed_at": FIXED_TIME.isoformat(),
            },
        },
        ensure_ascii=False,
    )
    transport = RepairTransport(repaired)

    # When: the repairer is asked to repair malformed output.
    result = _repairer(transport).repair(
        SchemaRepairRequest(
            raw_artifact=_malformed_artifact(),
            expected_contract="WorldBible",
            contract_schema_json='{"type":"object"}',
        )
    )

    # Then: the provider saw both the schema and the broken text.
    assert result == repaired
    assert len(transport.prompts) == 1
    assert '"type":"object"' in transport.prompts[0]
    assert "broken world text" in transport.prompts[0]


def test_repairer_failure_returns_empty_string_for_fallback() -> None:
    # Given: an unavailable provider.
    repairer = _repairer(BrokenTransport())

    # When: repair is attempted.
    result = repairer.repair(
        SchemaRepairRequest(
            raw_artifact=_malformed_artifact(),
            expected_contract="WorldBible",
            contract_schema_json="{}",
        )
    )

    # Then: the caller falls through to a typed schema failure.
    assert result == ""


def test_legacy_adapter_uses_repairer_as_third_parse_rung() -> None:
    # Given: legacy output that is malformed JSON but repairable by the provider.
    broken = (
        '{"world_id": "world-001", "brief_id": "brief-001", "title": "Fixed World", '
        '"premise": "A coherent premise.", "rules": ["rule-one"], "constraints": [],'
    )
    repaired = json.dumps(
        {
            "world_id": "world-001",
            "brief_id": "brief-001",
            "title": "Fixed World",
            "premise": "A coherent premise.",
            "rules": ("rule-one",),
            "constraints": (),
            "created_at": FIXED_TIME.isoformat(),
            "provenance": {
                "source_kind": "provider_response",
                "source_id": "legacy-world-call",
                "source_sha256": "a" * 64,
                "observed_at": FIXED_TIME.isoformat(),
            },
        },
        ensure_ascii=False,
    )
    module = pytest.importorskip("generate_world")
    import sys
    from types import ModuleType

    fake_module = ModuleType("generate_world")
    fake_module.expand_world = lambda _idea: broken
    sys.modules["generate_world"] = fake_module

    adapter = LegacyCreativeAdapter(
        StructuredCreativeAdapter(),
        repairer=_repairer(RepairTransport(repaired)),
    )

    # When: the legacy world generator returns malformed output.
    result = adapter.generate_world(_world_request())

    # Then: the bounded repair prompt recovers a typed contract with the repair route.
    match result:
        case WorldAdaptation(value=world, parse_route=route):
            assert world.title == "Fixed World"
            assert route is CreativeParseRoute.SCHEMA_REPAIR
        case SchemaFailure():
            pytest.fail("repairable output unexpectedly failed")
        case unreachable:
            assert_never(unreachable)


def _malformed_artifact():
    from novel_agent.adapters import CreativeArtifactKind, RawCreativeArtifact

    return RawCreativeArtifact(
        kind=CreativeArtifactKind.WORLD,
        raw_bytes=b"broken world text",
        content_sha256="496f1f3c2e62a16b30c0377d57d95d83e0df3ce082c0bb4e625142177408a0aa",
        adapter_version="test-v1",
    )
