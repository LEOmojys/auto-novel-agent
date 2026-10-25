from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field

from novel_agent.pathing import ProjectId


POLICY_VERSION: Final = "1.1.0"


class QualityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@unique
class QualityOutcome(StrEnum):
    ACCEPT = "ACCEPT"
    REPAIR = "REPAIR"
    PAUSE = "PAUSE"
    EVALUATOR_ERROR = "EVALUATOR_ERROR"


class QualityPolicy(QualityModel):
    version: str = POLICY_VERSION
    minimum_length_ratio: float = 0.80
    maximum_length_ratio: float = 1.25
    minimum_dimension_score: int = 3
    minimum_mean_score: float = 3.5
    maximum_repairs: int = Field(default=2, ge=0)
    maximum_evaluator_attempts: int = Field(default=2, gt=0)
    #: When a cast is available, at least this many cast members must appear.
    minimum_mentioned_characters: int = Field(default=0, ge=0)


class CharacterFixture(QualityModel):
    character_id: str
    name: str
    aliases: tuple[str, ...] = ()
    role: str = ""
    goal: str = ""
    traits: tuple[str, ...] = ()


class ChapterQualityRequest(QualityModel):
    project_id: ProjectId
    artifact_id: str
    plan_id: str
    expected_plan_id: str
    chapter_plan_id: str
    expected_chapter_plan_id: str
    chapter_number: int = Field(ge=1)
    expected_chapter_number: int = Field(ge=1)
    body: str
    target_chinese_characters: int = Field(gt=0)
    required_beats: tuple[str, ...] = ()
    located_beats: tuple[str, ...] = ()
    continuity_hard_conflicts: tuple[str, ...] = ()
    repair_attempts: int = Field(default=0, ge=0)
    style_guidance: tuple[str, ...] = ()
    world_rules: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    previous_summary: str = ""


class ChapterMetrics(QualityModel):
    chinese_characters: int = Field(ge=0)
    target_chinese_characters: int = Field(gt=0)
    mentioned_character_ids: tuple[str, ...] = ()

    @property
    def length_ratio(self) -> float:
        return self.chinese_characters / self.target_chinese_characters


class DimensionScore(QualityModel):
    score: int = Field(ge=1, le=5)
    reason: str


class EvaluatorReport(QualityModel):
    style_consistency: DimensionScore
    character_consistency: DimensionScore
    prose_fluency: DimensionScore
    scene_completeness: DimensionScore
    dialogue_naturalness: DimensionScore
    content_alignment: DimensionScore

    @property
    def dimensions(self) -> tuple[DimensionScore, ...]:
        return (
            self.style_consistency,
            self.character_consistency,
            self.prose_fluency,
            self.scene_completeness,
            self.dialogue_naturalness,
            self.content_alignment,
        )

    @property
    def mean_score(self) -> float:
        return sum(dimension.score for dimension in self.dimensions) / len(self.dimensions)


class EvaluatorAdapterRequest(QualityModel):
    project_id: ProjectId
    artifact_id: str
    body: str
    plan_id: str
    chapter_plan_id: str
    chapter_number: int
    required_beats: tuple[str, ...]
    characters: tuple[CharacterFixture, ...]
    style_guidance: tuple[str, ...] = ()
    world_rules: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    previous_summary: str = ""


class StructuredEvaluator(Protocol):
    def evaluate(self, request: EvaluatorAdapterRequest) -> str: ...


class MetricsEvaluator(Protocol):
    def measure(
        self,
        body: str,
        target_chinese_characters: int,
        characters: tuple[CharacterFixture, ...],
    ) -> ChapterMetrics: ...


class QualityDecision(QualityModel):
    outcome: QualityOutcome
    score: float | None
    reason_codes: tuple[str, ...]
    evaluator_attempts: int
    policy_version: str
    metrics: ChapterMetrics | None
    #: Human/model-readable reasons for the failing dimensions (repair input).
    feedback: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectScopedEvaluationAdapter:
    project_id: ProjectId
    characters: tuple[CharacterFixture, ...]
    evaluator: StructuredEvaluator
    metrics: MetricsEvaluator

    def review(self, request: ChapterQualityRequest, policy: QualityPolicy = QualityPolicy()) -> QualityDecision:
        hard_failure = self._hard_failure(request)
        if hard_failure is not None:
            return QualityDecision(
                outcome=QualityOutcome.PAUSE,
                score=None,
                reason_codes=(hard_failure,),
                evaluator_attempts=0,
                policy_version=policy.version,
                metrics=None,
            )

        metrics = self.metrics.measure(request.body, request.target_chinese_characters, self.characters)
        adapter_request = EvaluatorAdapterRequest(
            project_id=self.project_id,
            artifact_id=request.artifact_id,
            body=request.body,
            plan_id=request.plan_id,
            chapter_plan_id=request.chapter_plan_id,
            chapter_number=request.chapter_number,
            required_beats=request.required_beats,
            characters=self.characters,
            style_guidance=request.style_guidance,
            world_rules=request.world_rules,
            acceptance_criteria=request.acceptance_criteria,
            previous_summary=request.previous_summary,
        )
        report: EvaluatorReport | None = None
        attempts = 0
        while attempts < policy.maximum_evaluator_attempts:
            attempts += 1
            try:
                report = EvaluatorReport.model_validate_json(self.evaluator.evaluate(adapter_request))
                break
            except Exception:  # noqa: BROAD_EXCEPT_OK - external evaluator boundary
                continue
        if report is None:
            return QualityDecision(
                outcome=QualityOutcome.EVALUATOR_ERROR,
                score=None,
                reason_codes=("evaluator_error",),
                evaluator_attempts=attempts,
                policy_version=policy.version,
                metrics=metrics,
            )

        failures: list[str] = []
        feedback: list[str] = []
        if not policy.minimum_length_ratio <= metrics.length_ratio <= policy.maximum_length_ratio:
            failures.append("length_ratio")
            feedback.append(
                f"length_ratio: {metrics.chinese_characters}/{metrics.target_chinese_characters}"
            )
        below = [
            (name, dimension)
            for name, dimension in zip(_DIMENSION_NAMES, report.dimensions)
            if dimension.score < policy.minimum_dimension_score
        ]
        if below:
            failures.append("dimension_below_minimum")
            feedback.extend(f"{name}: {dimension.reason}" for name, dimension in below)
        if report.mean_score < policy.minimum_mean_score:
            failures.append("mean_below_minimum")
            feedback.append(f"mean_score: {report.mean_score:.2f}")
        missing_beats = [
            beat for beat in request.required_beats if not _beat_covered(beat, request.located_beats)
        ]
        if missing_beats:
            failures.append("missing_plan_beat")
            feedback.extend(f"missing_plan_beat: {beat}" for beat in missing_beats)
        if self.characters and len(metrics.mentioned_character_ids) < policy.minimum_mentioned_characters:
            failures.append("missing_character_coverage")
            feedback.append(
                f"missing_character_coverage: {len(metrics.mentioned_character_ids)}"
                f"/{policy.minimum_mentioned_characters}"
            )
        if not failures:
            return QualityDecision(
                outcome=QualityOutcome.ACCEPT,
                score=report.mean_score,
                reason_codes=(),
                evaluator_attempts=attempts,
                policy_version=policy.version,
                metrics=metrics,
                feedback=(),
            )
        if request.repair_attempts >= policy.maximum_repairs:
            return QualityDecision(
                outcome=QualityOutcome.PAUSE,
                score=report.mean_score,
                reason_codes=("repair_budget_exhausted",),
                evaluator_attempts=attempts,
                policy_version=policy.version,
                metrics=metrics,
                feedback=tuple(feedback),
            )
        return QualityDecision(
            outcome=QualityOutcome.REPAIR,
            score=report.mean_score,
            reason_codes=tuple(failures),
            evaluator_attempts=attempts,
            policy_version=policy.version,
            metrics=metrics,
            feedback=tuple(feedback),
        )

    def _hard_failure(self, request: ChapterQualityRequest) -> str | None:
        if request.project_id != self.project_id:
            return "project_mismatch"
        if not request.body.strip():
            return "empty_body"
        identity_matches = (
            request.plan_id == request.expected_plan_id
            and request.chapter_plan_id == request.expected_chapter_plan_id
            and request.chapter_number == request.expected_chapter_number
        )
        if not identity_matches:
            return "chapter_identity_mismatch"
        if request.continuity_hard_conflicts:
            return "continuity_hard_conflict"
        return None

_DIMENSION_NAMES: Final = (
    "style_consistency",
    "character_consistency",
    "prose_fluency",
    "scene_completeness",
    "dialogue_naturalness",
    "content_alignment",
)


def _normalise_beat(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def _beat_covered(required: str, located: tuple[str, ...]) -> bool:
    """Match a planned beat even when the auditor paraphrased it."""
    target = _normalise_beat(required)
    if not target:
        return True
    for candidate in located:
        normalised = _normalise_beat(candidate)
        if not normalised:
            continue
        if target in normalised or normalised in target:
            return True
        target_bigrams = {target[index : index + 2] for index in range(max(1, len(target) - 1))}
        candidate_bigrams = {
            normalised[index : index + 2] for index in range(max(1, len(normalised) - 1))
        }
        if target_bigrams and len(target_bigrams & candidate_bigrams) / len(target_bigrams) >= 0.6:
            return True
    return False


__all__ = (
    "CharacterFixture",
    "ChapterMetrics",
    "ChapterQualityRequest",
    "DimensionScore",
    "EvaluatorAdapterRequest",
    "EvaluatorReport",
    "MetricsEvaluator",
    "ProjectScopedEvaluationAdapter",
    "QualityDecision",
    "QualityOutcome",
    "QualityPolicy",
    "StructuredEvaluator",
)
