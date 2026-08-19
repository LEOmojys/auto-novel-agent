from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from novel_agent.contracts import (
    AuditReplayManifest,
    CharacterCard,
    CharacterSet,
    ChapterPlan,
    ContextItem,
    ContextPack,
    ContractErrorCode,
    EvidenceRef,
    MemoryCandidate,
    MemoryLayer,
    PlanApprovalStatus,
    PlanContract,
    ProviderRequest,
    Provenance,
    ProvenanceKind,
    RunCommand,
    RunCommandType,
    RunStatus,
    StoryBrief,
    WorldBible,
)
from novel_agent.pathing import ProjectId


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
FIXED_HASH = "a" * 64


def _provenance() -> Provenance:
    return Provenance(
        source_kind=ProvenanceKind.USER_REQUEST,
        source_id="request-001",
        source_sha256=FIXED_HASH,
        observed_at=FIXED_TIME,
    )


def _evidence() -> EvidenceRef:
    return EvidenceRef(
        evidence_id="evidence-001",
        artifact_id="artifact-001",
        artifact_revision=1,
        source_sha256=FIXED_HASH,
        recorded_at=FIXED_TIME,
    )


def _story_brief() -> StoryBrief:
    return StoryBrief(
        brief_id="brief-001",
        project_id=ProjectId("project-alpha"),
        user_requirements="Create a bounded mystery arc.",
        genre="mystery",
        target_chapter_count=1,
        created_at=FIXED_TIME,
        provenance=_provenance(),
    )


def test_story_package_round_trip() -> None:
    # Given: a valid brief, world, cast, and evidence-backed chapter plan.
    brief = _story_brief()
    world = WorldBible(
        world_id="world-001",
        brief_id=brief.brief_id,
        title="Harbor of Ash",
        premise="A cartographer investigates a vanished island.",
        rules=("Maps change only at dawn.",),
        constraints=("Do not contradict the established tide cycle.",),
        created_at=FIXED_TIME,
        provenance=_provenance(),
    )
    card = CharacterCard(
        character_id="character-001",
        name="Lin",
        role="investigator",
        goal="Recover the missing chart.",
        traits=("precise", "guarded"),
        created_at=FIXED_TIME,
        provenance=_provenance(),
    )
    cast = CharacterSet(
        character_set_id="cast-001",
        world_id=world.world_id,
        characters=(card,),
        created_at=FIXED_TIME,
        provenance=_provenance(),
    )
    chapter = ChapterPlan(
        chapter_plan_id="chapter-plan-001",
        chapter_number=1,
        title="Low Tide",
        goal="Find the first clue.",
        planned_events=("Lin studies the harbor map.",),
        evidence=(_evidence(),),
        created_at=FIXED_TIME,
        provenance=_provenance(),
    )
    plan = PlanContract(
        plan_id="plan-001",
        brief_id=brief.brief_id,
        world_id=world.world_id,
        character_set_id=cast.character_set_id,
        approval_status=PlanApprovalStatus.VALIDATED,
        chapters=(chapter,),
        created_at=FIXED_TIME,
        provenance=_provenance(),
    )

    # When: the persisted plan crosses the JSON boundary.
    restored = PlanContract.model_validate_json(plan.model_dump_json())

    # Then: all governed identity and evidence fields survive unchanged.
    assert restored == plan
    assert restored.chapters[0].evidence[0].source_sha256 == FIXED_HASH


def test_contract_rejects_unknown_field_at_boundary() -> None:
    # Given: a valid serialized brief with an undeclared external field.
    invalid_json = _story_brief().model_dump_json()[:-1] + ',"untrusted":true}'

    # When: the user-supplied JSON enters the contract boundary.
    with pytest.raises(ValidationError) as raised:
        StoryBrief.model_validate_json(invalid_json)

    # Then: Pydantic rejects it before any repository interaction is possible.
    assert raised.value.errors()[0]["type"] == "extra_forbidden"


def test_memory_candidate_rejects_missing_evidence() -> None:
    # Given: a candidate fact without an immutable evidence reference.
    candidate = {
        "candidate_id": "candidate-001",
        "project_id": "project-alpha",
        "memory_layer": MemoryLayer.M0_CANDIDATE,
        "subject": "Lin",
        "predicate": "owns",
        "value": "a brass compass",
        "evidence": (),
        "created_at": FIXED_TIME,
        "provenance": _provenance(),
    }

    # When: it crosses the memory boundary.
    with pytest.raises(ValidationError) as raised:
        MemoryCandidate.model_validate(candidate)

    # Then: it remains quarantined instead of becoming a usable fact.
    assert raised.value.errors()[0]["type"] == ContractErrorCode.MISSING_EVIDENCE


def test_run_command_rejects_illegal_expected_state() -> None:
    # Given: a pause command that claims a queued run can be paused.
    invalid_command = {
        "command_id": "command-001",
        "run_id": "run-001",
        "command_type": RunCommandType.PAUSE,
        "expected_status": RunStatus.QUEUED,
        "issued_at": FIXED_TIME,
        "provenance": _provenance(),
    }

    # When: it enters the durable command boundary.
    with pytest.raises(ValidationError) as raised:
        RunCommand.model_validate(invalid_command)

    # Then: a state-machine-invalid payload cannot reach a coordinator.
    assert raised.value.errors()[0]["type"] == ContractErrorCode.INVALID_STATE_TRANSITION


def test_contract_rejects_unsupported_schema_version() -> None:
    # Given: a brief encoded by a future, unsupported contract revision.
    payload = _story_brief().model_dump(mode="json")
    payload["schema_version"] = 2

    # When: it is parsed by the current process.
    with pytest.raises(ValidationError) as raised:
        StoryBrief.model_validate(payload)

    # Then: the boundary fails deterministically before persistence.
    assert raised.value.errors()[0]["type"] == "literal_error"


def test_provider_request_rejects_nonmatching_prompt_digest() -> None:
    # Given: a provider request whose prompt text does not match its declared digest.
    payload = {
        "provider_call_id": "provider-call-001",
        "run_id": "run-001",
        "task_id": "task-001",
        "model_name": "fake-model",
        "prompt": "Write a chapter.",
        "prompt_sha256": FIXED_HASH,
        "temperature": 0.2,
        "created_at": FIXED_TIME,
        "provenance": _provenance(),
    }

    # When: it enters the boundary used for a provider call.
    with pytest.raises(ValidationError) as raised:
        ProviderRequest.model_validate(payload)

    # Then: text can never be sent under a mismatched audit digest.
    assert raised.value.errors()[0]["type"] == ContractErrorCode.INVALID_REQUEST_DIGEST


def test_context_and_audit_models_round_trip() -> None:
    # Given: a bounded context pack and audit manifest emitted by a governed run.
    context = ContextPack(
        context_pack_id="context-001",
        run_id="run-001",
        chapter_plan_id="chapter-plan-001",
        max_tokens=400,
        items=(
            ContextItem(
                memory_id="memory-001",
                content="Lin has a brass compass.",
                token_count=7,
                evidence=(_evidence(),),
                included_at=FIXED_TIME,
                provenance=_provenance(),
            ),
        ),
        assembled_at=FIXED_TIME,
        provenance=_provenance(),
    )
    manifest = AuditReplayManifest(
        manifest_id="manifest-001",
        run_id="run-001",
        model_name="fake-model",
        temperature=0.2,
        prompt_template_sha256=FIXED_HASH,
        response_sha256=FIXED_HASH,
        adapter_version="v1",
        created_at=FIXED_TIME,
        provenance=_provenance(),
    )

    # When: both persisted records cross their JSON contracts.
    restored_context = ContextPack.model_validate_json(context.model_dump_json())
    restored_manifest = AuditReplayManifest.model_validate_json(manifest.model_dump_json())

    # Then: retrieval provenance and replay audit metadata are retained.
    assert restored_context == context
    assert restored_manifest == manifest
