from __future__ import annotations

import hashlib
import sqlite3
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import assert_never

import pytest

from novel_agent.adapters import CharacterAdaptation, OutlineAdaptation, RawCreativeArtifact, WorldAdaptation
from novel_agent.contracts import CharacterCard, CharacterSet, ChapterPlan, EvidenceRef, PlanApprovalStatus, PlanContract, Provenance, ProvenanceKind, StoryBrief, WorldBible
from novel_agent.pathing import ProjectId
from novel_agent.planning_authority import PlanningCommitAuthorityError
from novel_agent.planning_commit import PlanningCommitRequest
from novel_agent.planning import PlanningBlocked, PlanningSucceeded, PlanningWorkflow
from novel_agent.run_coordinator import RunCoordinator
from novel_agent.run_models import StartRunRequest
from novel_agent.skills import load_builtin_catalog
from novel_agent.storage import connect_project_database, migrate
from tests.manifest_assertions import assert_planning_manifest_contract


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
FIXED_HASH = "b" * 64


@dataclass(frozen=True, slots=True)
class FixedClock:
    current: datetime

    def now(self) -> datetime:
        return self.current


@dataclass(frozen=True, slots=True)
class FakePlanningGenerator:
    world: WorldAdaptation
    characters: CharacterAdaptation
    outline: OutlineAdaptation

    def generate_world(self, _brief: StoryBrief) -> WorldAdaptation:
        return self.world

    def generate_characters(self, _brief: StoryBrief, _world: WorldBible) -> CharacterAdaptation:
        return self.characters

    def generate_outline(self, _brief: StoryBrief, _world: WorldBible, _characters: CharacterSet) -> OutlineAdaptation:
        return self.outline


def _provenance(source_id: str) -> Provenance:
    return Provenance(source_kind=ProvenanceKind.PROVIDER_RESPONSE, source_id=source_id, source_sha256=FIXED_HASH, observed_at=FIXED_TIME)


def _raw(kind: str, body: str) -> RawCreativeArtifact:
    raw_bytes = body.encode()
    return RawCreativeArtifact(kind=kind, raw_bytes=raw_bytes, content_sha256=hashlib.sha256(raw_bytes).hexdigest(), adapter_version="test-adapter-v1")


def _brief(target_chapter_count: int = 1) -> StoryBrief:
    return StoryBrief(brief_id="brief-001", project_id=ProjectId("novel-one"), user_requirements="A mystery in a governed city.", genre="mystery", target_chapter_count=target_chapter_count, created_at=FIXED_TIME, provenance=_provenance("brief"))


def _generator(*, wrong_world_reference: bool = False) -> FakePlanningGenerator:
    world_raw = _raw("world", "world source")
    world = WorldBible(
        world_id="world-001",
        brief_id="brief-001",
        title="Governed City",
        premise="A city keeps its own records.",
        rules=("Every clue has provenance.",),
        constraints=("No unexplained retcons.",),
        created_at=FIXED_TIME,
        provenance=_provenance("world"),
    )
    cast_raw = _raw("characters", "character source")
    characters = CharacterSet(
        character_set_id="cast-001",
        world_id="world-other" if wrong_world_reference else "world-001",
        characters=(
            CharacterCard(
                character_id="character-001",
                name="Hero",
                role="detective",
                goal="Find the missing record.",
                traits=("careful",),
                created_at=FIXED_TIME,
                provenance=_provenance("cast"),
            ),
        ),
        created_at=FIXED_TIME,
        provenance=_provenance("cast"),
    )
    plan_raw = _raw("outline", "plan source")
    plan = PlanContract(
        plan_id="plan-001",
        brief_id="brief-001",
        world_id="world-001",
        character_set_id="cast-001",
        approval_status=PlanApprovalStatus.VALIDATED,
        chapters=(
            ChapterPlan(
                chapter_plan_id="chapter-plan-001",
                chapter_number=1,
                title="Opening",
                goal="Introduce the missing record.",
                planned_events=("A clue arrives.",),
                created_at=FIXED_TIME,
                provenance=_provenance("plan"),
                evidence=(
                    EvidenceRef(
                        evidence_id="plan-evidence-001",
                        artifact_id="world-001",
                        artifact_revision=1,
                        source_sha256=FIXED_HASH,
                        recorded_at=FIXED_TIME,
                    ),
                ),
            ),
        ),
        created_at=FIXED_TIME,
        provenance=_provenance("plan"),
    )
    return FakePlanningGenerator(
        world=WorldAdaptation(raw=world_raw, value=world, parse_route="strict_json"),
        characters=CharacterAdaptation(raw=cast_raw, value=characters, parse_route="strict_json"),
        outline=OutlineAdaptation(raw=plan_raw, value=plan, parse_route="strict_json"),
    )


def _request(run_id: str, project_id: ProjectId) -> PlanningCommitRequest:
    generated = _generator()
    return PlanningCommitRequest(run_id=run_id, project_id=project_id, brief=_brief(), world=generated.world, characters=generated.characters, outline=generated.outline, skills=())


def _governed_counts(connection: sqlite3.Connection, run_id: str) -> tuple[int, int, int, int, int]:
    row = connection.execute(
        "SELECT (SELECT COUNT(*) FROM artifacts WHERE run_id = ?), "
        "(SELECT COUNT(*) FROM memory_records WHERE run_id = ?), "
        "(SELECT COUNT(*) FROM tasks WHERE run_id = ?), "
        "(SELECT COUNT(*) FROM audit_manifests WHERE run_id = ?), "
        "(SELECT COUNT(*) FROM run_events WHERE run_id = ?)",
        (run_id, run_id, run_id, run_id, run_id),
    ).fetchone()
    assert row is not None
    return (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]))


def _open(
    tmp_path: Path,
) -> tuple[sqlite3.Connection, AbstractContextManager[sqlite3.Connection], RunCoordinator, FixedClock]:
    context = connect_project_database(tmp_path / "projects" / "novel-one")
    connection = context.__enter__()
    migrate(connection)
    clock = FixedClock(FIXED_TIME)
    coordinator = RunCoordinator(connection, clock.now)
    coordinator.submit_start(StartRunRequest(project_id=ProjectId("novel-one"), run_id="run-001", idempotency_key="start"))
    return connection, context, coordinator, clock


def test_validated_plan_autostarts_first_chapter(tmp_path: Path) -> None:
    # Given: a claimed run and fake creative stages that produce compatible validated contracts.
    connection, context, coordinator, _clock = _open(tmp_path)
    try:
        lease = coordinator.claim_next("planner")
        assert lease is not None
        workflow = PlanningWorkflow(coordinator, _generator(), load_builtin_catalog())

        # When: the planner executes its explicit world, cast, and outline task graph.
        result = workflow.execute(lease, _brief())

        # Then: the immutable plan is committed, exact Skills are audited, and chapter one is queued.
        match result:
            case PlanningSucceeded(plan_id=plan_id, first_task_id=task_id):
                assert plan_id == "plan-001"
                assert task_id.endswith("chapter-plan-001")
            case PlanningBlocked() as blocked:
                pytest.fail(f"planning unexpectedly blocked: {blocked.reason_code}")
            case unreachable:
                assert_never(unreachable)
        replayed = workflow.execute(lease, _brief())
        assert replayed == result
        assert connection.execute("SELECT plan_revision FROM runs WHERE run_id = 'run-001'").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone() == (3,)
        assert connection.execute("SELECT COUNT(*) FROM memory_records").fetchone() == (3,)
        assert connection.execute("SELECT task_type, state FROM tasks WHERE run_id = 'run-001'").fetchone() == (
            "chapter",
            "pending",
        )
        events = connection.execute("SELECT event_type FROM run_events WHERE run_id = 'run-001' ORDER BY sequence").fetchall()
        assert [row[0] for row in events][-1] == "plan_validated"
        assert [row[0] for row in events].count("plan_validated") == 1
        manifest = connection.execute("SELECT manifest_json FROM audit_manifests WHERE run_id = 'run-001'").fetchone()
        assert manifest is not None
        assert_planning_manifest_contract(
            str(manifest[0]),
            hashlib.sha256(_brief().user_requirements.encode()).hexdigest(),
            _generator().outline.raw.content_sha256,
        )
    finally:
        context.__exit__(None, None, None)


def test_plan_budget_breach_blocks_before_any_governed_commit(tmp_path: Path) -> None:
    # Given: a valid generated graph whose requested chapter count exceeds the automatic planning budget.
    connection, context, coordinator, _clock = _open(tmp_path)
    try:
        lease = coordinator.claim_next("planner")
        assert lease is not None
        workflow = PlanningWorkflow(coordinator, _generator(), load_builtin_catalog())

        # When: the planner evaluates the brief before authorizing any world, cast, or plan write.
        result = workflow.execute(lease, _brief(target_chapter_count=101))

        # Then: the run pauses for review and all governed write tables remain empty.
        match result:
            case PlanningBlocked(reason_code="plan_budget_exceeded"):
                pass
            case PlanningSucceeded():
                pytest.fail("over-budget plan unexpectedly committed")
            case unreachable:
                assert_never(unreachable)
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM memory_records").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone() == (0,)
        assert connection.execute("SELECT state FROM runs WHERE run_id = 'run-001'").fetchone() == ("waiting_review",)
    finally:
        context.__exit__(None, None, None)


def test_invalid_cross_artifact_reference_blocks_plan_without_current_pointer(tmp_path: Path) -> None:
    # Given: a claimed run whose cast points to a different world than the generated world.
    connection, context, coordinator, _clock = _open(tmp_path)
    try:
        lease = coordinator.claim_next("planner")
        assert lease is not None
        generated = _generator(wrong_world_reference=True)
        workflow = PlanningWorkflow(coordinator, generated, load_builtin_catalog())

        # When: cross-artifact validation rejects the candidate plan before commit.
        result = workflow.execute(lease, _brief())

        # Then: review is visible and no M1/M2/current plan or chapter task is written.
        match result:
            case PlanningBlocked(reason_code="cast_world_mismatch"):
                pass
            case PlanningSucceeded():
                pytest.fail("invalid plan unexpectedly committed")
            case unreachable:
                assert_never(unreachable)
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM memory_records").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone() == (0,)
        assert connection.execute("SELECT state FROM runs WHERE run_id = 'run-001'").fetchone() == ("waiting_review",)
    finally:
        context.__exit__(None, None, None)


@pytest.mark.parametrize(
    ("request_run_id", "request_project_id"),
    (
        ("run-002", ProjectId("novel-one")),
        ("run-001", ProjectId("novel-other")),
        ("", ProjectId("novel-one")),
    ),
)
def test_planning_commit_rejects_request_outside_claimed_run_authority(
    tmp_path: Path,
    request_run_id: str,
    request_project_id: ProjectId,
) -> None:
    # Given: run-001 is claimed while another queued run shares the same project database.
    connection, context, coordinator, _clock = _open(tmp_path)
    try:
        coordinator.submit_start(StartRunRequest(project_id=ProjectId("novel-one"), run_id="run-002", idempotency_key="start-002"))
        lease = coordinator.claim_next("planner")
        assert lease is not None
        before_runs = connection.execute(
            "SELECT run_id, state, plan_revision FROM runs ORDER BY run_id"
        ).fetchall()
        before_counts = _governed_counts(connection, request_run_id)

        # When: a lease for run-001 attempts to commit a valid request for another run or project.
        with pytest.raises(PlanningCommitAuthorityError):
            coordinator.commit_validated_plan(lease, _request(request_run_id, request_project_id))

        # Then: neither the target run nor the leased run receives a planning write or revision change.
        assert connection.execute("SELECT run_id, state, plan_revision FROM runs ORDER BY run_id").fetchall() == before_runs
        assert _governed_counts(connection, request_run_id) == before_counts
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM memory_records").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM audit_manifests").fetchone() == (0,)
    finally:
        context.__exit__(None, None, None)
