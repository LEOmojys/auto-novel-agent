from __future__ import annotations

import hashlib
import sqlite3
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path

from novel_agent.adapters import ChapterAdaptation, ChapterDraft, CreativeArtifactKind, CreativeParseRoute, RawCreativeArtifact
from novel_agent.chapter_executor import (
    ChapterGenerator,
    ChapterDraftRequest,
    ChapterExecutionPolicy,
    ChapterExecutionRequest,
    ChapterExecutionWorkflow,
    ContinuityAuditReport,
    ContinuityAuditRequest,
    MemoryExtractionRequest,
)
from novel_agent.context import ContextBuilder
from novel_agent.contracts import ChapterPlan, EvidenceRef, MemoryLayer, PlanApprovalStatus, PlanContract, Provenance, ProvenanceKind
from novel_agent.memory import EntityResolution, InferenceMode, MemoryFactKind, MemoryProposal, PromotionAuthority, TemporalCertainty
from novel_agent.pathing import ProjectId, ProjectPathResolver
from novel_agent.quality import ChapterQualityRequest, QualityDecision, QualityOutcome
from novel_agent.run_coordinator import RunCoordinator
from novel_agent.run_models import RunLease, StartRunRequest
from novel_agent.skills import load_builtin_catalog
from novel_agent.storage import ArtifactRevisionWrite, commit_artifact_revision, connect_project_database, migrate


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
FIXED_HASH = "a" * 64
PROJECT_ID = ProjectId("novel-one")


class FixedClock:
    def __init__(self) -> None:
        self.current = FIXED_TIME

    def now(self) -> datetime:
        return self.current


class FakeChapterGenerator:
    def __init__(self) -> None:
        self.context_memory_ids: list[tuple[str, ...]] = []
        self.chapter_numbers: list[int] = []

    def generate(self, request: ChapterDraftRequest) -> ChapterAdaptation:
        chapter = request.chapter_plan
        self.chapter_numbers.append(chapter.chapter_number)
        self.context_memory_ids.append(request.context.memory_ids)
        repair = 0 if request.repair_brief is None else request.repair_brief.repair_attempt
        body = f"chapter-{chapter.chapter_number}-attempt-{repair}"
        raw_bytes = f'{{"chapter_plan_id":"{chapter.chapter_plan_id}","body":"{body}"}}'.encode()
        return ChapterAdaptation(
            raw=RawCreativeArtifact(
                kind=CreativeArtifactKind.CHAPTER,
                raw_bytes=raw_bytes,
                content_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                adapter_version="fake-chapter-v1",
            ),
            value=ChapterDraft(
                chapter_draft_id=f"draft-{chapter.chapter_number}-{repair}",
                chapter_plan_id=chapter.chapter_plan_id,
                body=body,
                created_at=FIXED_TIME,
                provenance=provenance(f"draft-{chapter.chapter_number}-{repair}"),
            ),
            parse_route=CreativeParseRoute.STRICT_JSON,
        )


class FakeMemoryExtractor:
    def extract(self, request: MemoryExtractionRequest) -> tuple[MemoryProposal, ...]:
        staged = request.staged
        chapter = request.chapter_plan
        return (
            MemoryProposal(
                run_id=request.run_id,
                project_id=PROJECT_ID,
                source_artifact_id=staged.artifact_id,
                source_artifact_revision=staged.revision,
                extraction_schema_version=1,
                target_layer=MemoryLayer.M3_CONFIRMED,
                kind=MemoryFactKind.DIRECT_EVENT,
                subject_id=f"chapter-{chapter.chapter_number}",
                predicate=f"event-{chapter.chapter_number}",
                value_json=f'{{"chapter":{chapter.chapter_number}}}',
                source_start=0,
                source_end=len(staged.raw_bytes),
                source_excerpt=staged.raw_bytes.decode(),
                entity_resolution=EntityResolution.EXACT,
                inference_mode=InferenceMode.DIRECT,
                temporal_certainty=TemporalCertainty.EXACT,
                confidence=1.0,
                authority=PromotionAuthority.ACCEPTED_CHAPTER,
                plan_revision=request.plan_revision,
                supersedes_memory_id=None,
                created_at=FIXED_TIME,
            ),
        )


class FakeContinuityAuditor:
    def __init__(self, hard_conflict: bool = False) -> None:
        self.hard_conflict = hard_conflict

    def audit(self, request: ContinuityAuditRequest) -> ContinuityAuditReport:
        return ContinuityAuditReport(
            hard_conflicts=("continuity_conflict",) if self.hard_conflict else (),
            located_beats=request.chapter_plan.planned_events,
        )


class FakeQualityReviewer:
    def __init__(self, repairs_before_accept: int) -> None:
        self.repairs_before_accept = repairs_before_accept
        self.calls = 0

    def review(self, _request: ChapterQualityRequest) -> QualityDecision:
        self.calls += 1
        if self.calls <= self.repairs_before_accept:
            return QualityDecision(
                outcome=QualityOutcome.REPAIR,
                score=3.0,
                reason_codes=("missing_plan_beat",),
                evaluator_attempts=1,
                policy_version="fake-quality-v1",
                metrics=None,
            )
        return QualityDecision(
            outcome=QualityOutcome.ACCEPT,
            score=4.0,
            reason_codes=(),
            evaluator_attempts=1,
            policy_version="fake-quality-v1",
            metrics=None,
        )


def provenance(source_id: str) -> Provenance:
    return Provenance(
        source_kind=ProvenanceKind.SYSTEM,
        source_id=source_id,
        source_sha256=FIXED_HASH,
        observed_at=FIXED_TIME,
    )


def plan(chapter_count: int = 2) -> PlanContract:
    chapters = tuple(
        ChapterPlan(
            chapter_plan_id=f"chapter-plan-{number:03d}",
            chapter_number=number,
            title=f"Chapter {number}",
            goal=f"Advance event {number}",
            planned_events=(f"event-{number}",),
            created_at=FIXED_TIME,
            provenance=provenance(f"chapter-plan-{number}"),
            evidence=(
                EvidenceRef(
                    evidence_id=f"evidence-{number}",
                    artifact_id="world-001",
                    artifact_revision=1,
                    source_sha256=FIXED_HASH,
                    recorded_at=FIXED_TIME,
                ),
            ),
        )
        for number in range(1, chapter_count + 1)
    )
    return PlanContract(
        plan_id="plan-001",
        brief_id="brief-001",
        world_id="world-001",
        character_set_id="cast-001",
        approval_status=PlanApprovalStatus.VALIDATED,
        chapters=chapters,
        created_at=FIXED_TIME,
        provenance=provenance("plan"),
    )


def open_run(
    tmp_path: Path,
    plan_contract: PlanContract,
) -> tuple[sqlite3.Connection, AbstractContextManager[sqlite3.Connection], RunCoordinator, RunLease]:
    context = connect_project_database(tmp_path / "projects" / PROJECT_ID.root)
    connection = context.__enter__()
    migrate(connection)
    clock = FixedClock()
    coordinator = RunCoordinator(connection, clock.now)
    coordinator.submit_start(StartRunRequest(project_id=PROJECT_ID, run_id="run-001", idempotency_key="start"))
    lease = coordinator.claim_next("chapter-worker")
    assert lease is not None
    commit_artifact_revision(
        connection,
        ArtifactRevisionWrite(
            artifact_id=plan_contract.plan_id,
            revision=1,
            project_id=PROJECT_ID.root,
            run_id="run-001",
            kind="plan_contract",
            status="accepted",
            raw_bytes=plan_contract.model_dump_json().encode(),
            schema_version=1,
            created_at=FIXED_TIME.isoformat(),
        ),
    )
    first_task = f"run-001:chapter:{plan_contract.chapters[0].chapter_plan_id}"
    connection.execute(
        "INSERT INTO tasks(task_id, run_id, task_type, state, idempotency_key, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (first_task, "run-001", "chapter", "pending", first_task, FIXED_TIME.isoformat(), FIXED_TIME.isoformat()),
    )
    return connection, context, coordinator, lease


def workflow(
    connection: sqlite3.Connection,
    coordinator: RunCoordinator,
    generator: ChapterGenerator,
    reviewer: FakeQualityReviewer,
    auditor: FakeContinuityAuditor,
    tmp_path: Path,
    policy: ChapterExecutionPolicy = ChapterExecutionPolicy(),
) -> ChapterExecutionWorkflow:
    return ChapterExecutionWorkflow(
        coordinator=coordinator,
        context_builder=ContextBuilder(connection),
        catalog=load_builtin_catalog(),
        generator=generator,
        extractor=FakeMemoryExtractor(),
        continuity_auditor=auditor,
        quality_reviewer=reviewer,
        path_resolver=ProjectPathResolver(tmp_path / "projects"),
        policy=policy,
    )


def execution_request(plan_contract: PlanContract) -> ChapterExecutionRequest:
    return ChapterExecutionRequest(
        project_id=PROJECT_ID,
        plan=plan_contract,
        plan_revision=1,
        target_chinese_characters=20,
        genre="mystery",
    )
