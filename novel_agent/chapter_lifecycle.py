from __future__ import annotations

from novel_agent.chapter_models import (
    ChapterExecutionRequest,
    ChapterQualityReportRequest,
    RepairBrief,
    StagedChapter,
)
from novel_agent.context_models import ContextBudget, ContextBuildResult, ContextRequest
from novel_agent.contracts import ChapterPlan
from novel_agent.quality import ChapterQualityRequest, QualityDecision
from novel_agent.skill_manifests.models import ActiveSkillPin, SelectedSkill, SkillManifest
from novel_agent.skills import SkillCatalog, SkillCatalogError


# The chapter execution chain is a fixed, deterministic pipeline (draft ->
# memory extraction -> continuity audit -> quality review, plus targeted
# revision on repair).  This mirrors the fixed-workflow pattern used by
# production novel agents (e.g. ainovel-cli's writer sequence) rather than a
# free-form skill competition: each chapter runs every gate exactly once, so
# the SkillSelector is intentionally not consulted per chapter sub-step.
# Every skill in the chain is still validated against its manifest
# (``_pin_skill`` rejects disabled/deprecated entries).
_BASE_SKILLS = ("chapter-scene", "continuity-audit", "memory-extraction", "quality-review")


def pin_chapter_skills(
    catalog: SkillCatalog,
    run_id: str,
    repair: bool,
) -> tuple[ActiveSkillPin, ...]:
    """Pin the deterministic chapter skill chain, rejecting non-enabled manifests."""
    skill_ids = (*_BASE_SKILLS, "targeted-revision") if repair else _BASE_SKILLS
    manifests = {manifest.skill_id: manifest for manifest in catalog.manifests}
    try:
        return tuple(
            _pin_skill(catalog, run_id, manifests, skill_id)
            for skill_id in skill_ids
        )
    except KeyError as error:
        raise SkillCatalogError(detail=f"required chapter Skill is unavailable: {error.args[0]}") from error


def _pin_skill(
    catalog: SkillCatalog,
    run_id: str,
    manifests: dict[str, SkillManifest],
    skill_id: str,
) -> ActiveSkillPin:
    from novel_agent.skill_manifests.models import SkillStatus

    manifest = manifests[skill_id]
    if manifest.status is not SkillStatus.ENABLED:
        raise SkillCatalogError(detail=f"required chapter Skill is not enabled: {manifest.skill_ref}")
    return catalog.pin(
        run_id,
        SelectedSkill(manifest=manifest, score=100, score_evidence=("governed_chapter",)),
    )


def context_request(
    run_id: str,
    request: ChapterExecutionRequest,
    chapter: ChapterPlan,
    budget: ContextBudget,
) -> ContextRequest:
    return ContextRequest(
        project_id=request.project_id,
        run_id=run_id,
        plan_revision=request.plan_revision,
        chapter_plan_id=chapter.chapter_plan_id,
        chapter_number=chapter.chapter_number,
        query_terms=(chapter.title, chapter.goal, *chapter.planned_events),
        style_guidance=request.style_guidance,
        budget=budget,
    )


def quality_request(
    request: ChapterExecutionRequest,
    chapter: ChapterPlan,
    staged: StagedChapter,
    located_beats: tuple[str, ...],
    hard_conflicts: tuple[str, ...],
    repair_attempt: int,
) -> ChapterQualityRequest:
    return ChapterQualityRequest(
        project_id=request.project_id,
        artifact_id=staged.artifact_id.root,
        plan_id=request.plan.plan_id,
        expected_plan_id=request.plan.plan_id,
        chapter_plan_id=staged.chapter_plan_id,
        expected_chapter_plan_id=chapter.chapter_plan_id,
        chapter_number=staged.chapter_number,
        expected_chapter_number=chapter.chapter_number,
        body=staged.body,
        target_chinese_characters=request.target_chinese_characters,
        required_beats=chapter.planned_events,
        located_beats=located_beats,
        continuity_hard_conflicts=hard_conflicts,
        repair_attempts=repair_attempt,
    )


def repair_brief(
    chapter: ChapterPlan,
    repair_attempt: int,
    failure_codes: tuple[str, ...],
    context: ContextBuildResult,
    staged: StagedChapter,
) -> RepairBrief:
    return RepairBrief(
        chapter_plan=chapter,
        repair_attempt=repair_attempt,
        failure_codes=failure_codes,
        context_digest=context.digest,
        evidence_artifact_id=staged.artifact_id,
        evidence_artifact_revision=staged.revision,
        evidence_sha256=staged.raw_sha256,
    )


def quality_report_request(
    run_id: str,
    task_id: str,
    staged: StagedChapter,
    decision: QualityDecision,
) -> ChapterQualityReportRequest:
    return ChapterQualityReportRequest(run_id, task_id, staged, decision)


__all__ = (
    "context_request",
    "pin_chapter_skills",
    "quality_report_request",
    "quality_request",
    "repair_brief",
)
