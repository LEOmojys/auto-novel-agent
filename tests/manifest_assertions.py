from __future__ import annotations

import json


def assert_planning_manifest_contract(
    manifest_json: str,
    expected_prompt_sha256: str,
    expected_response_sha256: str,
) -> None:
    payload = json.loads(manifest_json)
    assert {item["skill_ref"] for item in payload["skills"]} == {
        "brief-normalization@1.0.0",
        "world-building@1.0.0",
        "cast-network@1.0.0",
        "story-architecture@1.0.0",
    }
    assert payload["model_name"] == "not-recorded"
    assert payload["parameters"] == {
        "characters_parse_route": "strict_json",
        "outline_parse_route": "strict_json",
        "world_parse_route": "strict_json",
    }
    assert payload["prompt_sha256"] == expected_prompt_sha256
    assert payload["response_sha256"] == expected_response_sha256
    assert payload["adapter_version"] == "test-adapter-v1"
    assert payload["artifact_revision"] == 1
