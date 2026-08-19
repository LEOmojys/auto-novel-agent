from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from novel_agent.adapters import (
    ChapterAdaptation,
    ChapterDraft,
    CharacterAdaptation,
    CreativeArtifactKind,
    CreativeParseRoute,
    OutlineAdaptation,
    RawCreativeArtifact,
    WorldAdaptation,
)
from novel_agent.chapter_executor import ContinuityAuditReport
from novel_agent.chapter_models import ChapterDraftRequest, ContinuityAuditRequest, MemoryExtractionRequest
from novel_agent.contracts import (
    CharacterCard,
    CharacterSet,
    ChapterPlan,
    EvidenceRef,
    PlanApprovalStatus,
    PlanContract,
    Provenance,
    ProvenanceKind,
    StoryBrief,
    WorldBible,
)
from novel_agent.memory_models import (
    EntityResolution,
    InferenceMode,
    MemoryFactKind,
    MemoryLayer,
    MemoryProposal,
    PromotionAuthority,
    TemporalCertainty,
)
from novel_agent.pathing import ProjectId
from novel_agent.quality import ChapterQualityRequest, QualityDecision, QualityOutcome


def offline_provenance(source: str, now: datetime) -> Provenance:
    return Provenance(
        source_kind=ProvenanceKind.PROVIDER_RESPONSE,
        source_id=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        observed_at=now,
    )


def _raw(kind: CreativeArtifactKind, payload: bytes) -> RawCreativeArtifact:
    return RawCreativeArtifact(
        kind=kind,
        raw_bytes=payload,
        content_sha256=hashlib.sha256(payload).hexdigest(),
        adapter_version="offline-fake-v1",
    )


@dataclass(frozen=True, slots=True)
class OfflinePlanningGenerator:
    now: datetime

    def generate_world(self, brief: StoryBrief) -> WorldAdaptation:
        world = WorldBible(
            world_id=f"{brief.brief_id}:world",
            brief_id=brief.brief_id,
            title="Offline World",
            premise=brief.user_requirements,
            rules=("Every accepted fact has provenance.",),
            constraints=(),
            created_at=self.now,
            provenance=offline_provenance("offline-world", self.now),
        )
        payload = world.model_dump_json().encode()
        return WorldAdaptation(
            raw=_raw(CreativeArtifactKind.WORLD, payload),
            value=world,
            parse_route=CreativeParseRoute.STRICT_JSON,
        )

    def generate_characters(self, brief: StoryBrief, world: WorldBible) -> CharacterAdaptation:
        character = CharacterCard(
            character_id=f"{brief.brief_id}:protagonist",
            name="Alex",
            role="protagonist",
            goal="solve the mystery",
            traits=("curious",),
            created_at=self.now,
            provenance=offline_provenance("offline-character", self.now),
        )
        value = CharacterSet(
            character_set_id=f"{brief.brief_id}:cast",
            world_id=world.world_id,
            characters=(character,),
            created_at=self.now,
            provenance=offline_provenance("offline-cast", self.now),
        )
        payload = value.model_dump_json().encode()
        return CharacterAdaptation(
            raw=_raw(CreativeArtifactKind.CHARACTERS, payload),
            value=value,
            parse_route=CreativeParseRoute.STRICT_JSON,
        )

    def revise_characters(
        self,
        brief: StoryBrief,
        world: WorldBible,
        existing: CharacterSet,
        target_names: tuple[str, ...],
    ) -> CharacterAdaptation:
        target_set = frozenset(target_names)
        kept = tuple(character for character in existing.characters if character.name not in target_set)
        revised = tuple(
            CharacterCard(
                character_id=f"{brief.brief_id}:revised:{index}",
                name=name,
                role="protagonist",
                goal="revised goal",
                traits=("revised",),
                created_at=self.now,
                provenance=offline_provenance("offline-character-revise", self.now),
            )
            for index, name in enumerate(target_names, start=1)
        )
        value = CharacterSet(
            character_set_id=f"{brief.brief_id}:cast",
            world_id=world.world_id,
            characters=kept + revised,
            created_at=self.now,
            provenance=offline_provenance("offline-cast-revise", self.now),
        )
        payload = value.model_dump_json().encode()
        return CharacterAdaptation(
            raw=_raw(CreativeArtifactKind.CHARACTERS, payload),
            value=value,
            parse_route=CreativeParseRoute.STRICT_JSON,
        )

    def generate_outline(
        self,
        brief: StoryBrief,
        world: WorldBible,
        characters: CharacterSet,
    ) -> OutlineAdaptation:
        evidence = (
            EvidenceRef(
                evidence_id="offline-world",
                artifact_id=world.world_id,
                artifact_revision=1,
                source_sha256=hashlib.sha256(world.model_dump_json().encode()).hexdigest(),
                recorded_at=self.now,
            ),
        )
        chapters = tuple(
            ChapterPlan(
                chapter_plan_id=f"{brief.brief_id}:chapter:{number}",
                chapter_number=number,
                title=f"Chapter {number}",
                goal="advance the story",
                planned_events=(f"event-{number}",),
                created_at=self.now,
                provenance=offline_provenance(f"offline-plan-{number}", self.now),
                evidence=evidence,
            )
            for number in range(1, brief.target_chapter_count + 1)
        )
        plan = PlanContract(
            plan_id=f"{brief.brief_id}:plan",
            brief_id=brief.brief_id,
            world_id=world.world_id,
            character_set_id=characters.character_set_id,
            approval_status=PlanApprovalStatus.VALIDATED,
            chapters=chapters,
            created_at=self.now,
            provenance=offline_provenance("offline-outline", self.now),
        )
        payload = plan.model_dump_json().encode()
        return OutlineAdaptation(
            raw=_raw(CreativeArtifactKind.OUTLINE, payload),
            value=plan,
            parse_route=CreativeParseRoute.STRICT_JSON,
        )


@dataclass(frozen=True, slots=True)
class OfflineChapterGenerator:
    now: datetime

    def generate(self, request: ChapterDraftRequest) -> ChapterAdaptation:
        events = ", ".join(request.chapter_plan.planned_events)
        body = (
            f"Chapter {request.chapter_plan.chapter_number}: {request.chapter_plan.goal}. "
            f"The governed story advances through {events}."
        )
        draft = ChapterDraft(
            chapter_draft_id=f"{request.task_id}:draft",
            chapter_plan_id=request.chapter_plan.chapter_plan_id,
            body=body,
            created_at=self.now,
            provenance=offline_provenance(request.task_id, self.now),
        )
        payload = draft.model_dump_json().encode()
        return ChapterAdaptation(
            raw=_raw(CreativeArtifactKind.CHAPTER, payload),
            value=draft,
            parse_route=CreativeParseRoute.STRICT_JSON,
        )


@dataclass(frozen=True, slots=True)
class OfflineMemoryExtractor:
    now: datetime
    project_id: ProjectId

    def extract(self, request: MemoryExtractionRequest) -> tuple[MemoryProposal, ...]:
        staged = request.staged
        return (
            MemoryProposal(
                run_id=request.run_id,
                project_id=self.project_id,
                source_artifact_id=staged.artifact_id,
                source_artifact_revision=staged.revision,
                extraction_schema_version=1,
                target_layer=MemoryLayer.M3_CONFIRMED,
                kind=MemoryFactKind.DIRECT_EVENT,
                subject_id=staged.chapter_plan_id,
                predicate="offline_event",
                value_json='{"offline":true}',
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
                created_at=self.now,
            ),
        )


@dataclass(frozen=True, slots=True)
class OfflineContinuityAuditor:
    def audit(self, request: ContinuityAuditRequest) -> ContinuityAuditReport:
        return ContinuityAuditReport(located_beats=request.chapter_plan.planned_events)


@dataclass(frozen=True, slots=True)
class OfflineQualityReviewer:
    def review(self, _request: ChapterQualityRequest) -> QualityDecision:
        return QualityDecision(
            outcome=QualityOutcome.ACCEPT,
            score=5.0,
            reason_codes=(),
            evaluator_attempts=1,
            policy_version="offline-fake-v1",
            metrics=None,
        )


__all__ = (
    "OfflineChapterGenerator",
    "OfflineContinuityAuditor",
    "OfflineMemoryExtractor",
    "OfflinePlanningGenerator",
    "OfflineQualityReviewer",
    "offline_provenance",
)
