from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from novel_agent.pathing import ProjectId
from novel_agent.quality_gate import quality_accepted
from novel_agent.quality import (
    CharacterFixture,
    ChapterMetrics,
    ChapterQualityRequest,
    EvaluatorAdapterRequest,
    QualityOutcome,
    QualityPolicy,
    ProjectScopedEvaluationAdapter,
)


VALID_SCORES = """{
  "style_consistency":{"score":4,"reason":"ok"},
  "character_consistency":{"score":4,"reason":"ok"},
  "prose_fluency":{"score":4,"reason":"ok"},
  "scene_completeness":{"score":4,"reason":"ok"},
  "dialogue_naturalness":{"score":4,"reason":"ok"},
  "content_alignment":{"score":4,"reason":"ok"}
}"""


@dataclass(frozen=True, slots=True)
class RecordingEvaluator:
    response: str = VALID_SCORES
    error: RuntimeError | None = None
    seen: list[EvaluatorAdapterRequest] | None = None

    def evaluate(self, request: EvaluatorAdapterRequest) -> str:
        if self.seen is not None:
            self.seen.append(request)
        if self.error is not None:
            raise self.error
        return self.response


@dataclass(frozen=True, slots=True)
class FixtureMetrics:
    def measure(
        self,
        body: str,
        target_chinese_characters: int,
        characters: tuple[CharacterFixture, ...],
    ) -> ChapterMetrics:
        return ChapterMetrics(
            chinese_characters=len(body),
            target_chinese_characters=target_chinese_characters,
            mentioned_character_ids=tuple(character.character_id for character in characters if character.name in body),
        )


def _request(**updates: str | int | tuple[str, ...] | ProjectId) -> ChapterQualityRequest:
    values: dict[str, str | int | tuple[str, ...] | ProjectId] = {
        "project_id": ProjectId("project-a"),
        "artifact_id": "chapter-1",
        "plan_id": "plan-1",
        "expected_plan_id": "plan-1",
        "chapter_plan_id": "chapter-plan-1",
        "expected_chapter_plan_id": "chapter-plan-1",
        "chapter_number": 1,
        "expected_chapter_number": 1,
        "body": "中" * 100,
        "target_chinese_characters": 100,
        "required_beats": ("arrival",),
        "located_beats": ("arrival",),
        "continuity_hard_conflicts": (),
        "repair_attempts": 0,
    }
    values.update(updates)
    return ChapterQualityRequest.model_validate(values)


def _adapter(
    evaluator: RecordingEvaluator | None = None,
    *,
    project_id: str = "project-a",
    characters: tuple[CharacterFixture, ...] = (),
) -> ProjectScopedEvaluationAdapter:
    return ProjectScopedEvaluationAdapter(
        project_id=ProjectId(project_id),
        characters=characters,
        evaluator=evaluator or RecordingEvaluator(),
        metrics=FixtureMetrics(),
    )


def test_thresholds_accept_valid_chapter() -> None:
    # Given: a matching chapter at its target length with all six scores above policy floors.
    request = _request()
    # When: the project-scoped adapter reviews it.
    decision = _adapter().review(request, QualityPolicy())
    # Then: the chapter is accepted with its real evaluator mean.
    assert (decision.outcome, decision.score, decision.policy_version) == (QualityOutcome.ACCEPT, 4.0, "1.1.0")


def test_length_and_locatable_beat_gaps_are_repairable() -> None:
    # Given: a short chapter with an explicitly missing planned beat.
    request = _request(body="中" * 79, located_beats=())
    # When: quality policy is applied before the repair budget is exhausted.
    decision = _adapter().review(request, QualityPolicy())
    # Then: targeted repair is requested for both deterministic gaps.
    assert decision.outcome is QualityOutcome.REPAIR
    assert set(decision.reason_codes) == {"length_ratio", "missing_plan_beat"}


def test_low_dimension_and_mean_are_repairable() -> None:
    # Given: structured scores with one dimension below 3 and a mean below 3.5.
    response = VALID_SCORES.replace('"score":4', '"score":2', 1).replace('"score":4', '"score":3')
    # When: the evaluator result is reviewed.
    decision = _adapter(RecordingEvaluator(response=response)).review(_request(), QualityPolicy())
    # Then: both score policy failures request repair.
    assert decision.outcome is QualityOutcome.REPAIR
    assert set(decision.reason_codes) == {"dimension_below_minimum", "mean_below_minimum"}


def test_empty_identity_mismatch_and_continuity_conflict_pause() -> None:
    # Given: each deterministic hard-failure input.
    requests = (
        _request(body="   "),
        _request(expected_plan_id="other-plan"),
        _request(continuity_hard_conflicts=("hero-dead-and-speaking",)),
    )
    # When: each request is reviewed.
    decisions = tuple(_adapter().review(request, QualityPolicy()) for request in requests)
    # Then: each pauses without inventing a score.
    assert tuple(decision.outcome for decision in decisions) == (QualityOutcome.PAUSE,) * 3
    assert all(decision.score is None for decision in decisions)


def test_malformed_output_is_explicit_evaluator_error() -> None:
    # Given: evaluator output that cannot satisfy the six-dimension schema.
    evaluator = RecordingEvaluator(response='{"style_consistency":{"score":4,"reason":"ok"}}')
    # When: both parser attempts fail.
    decision = _adapter(evaluator).review(_request(), QualityPolicy())
    # Then: the result is an operational error, never a fabricated zero.
    assert (decision.outcome, decision.score, decision.evaluator_attempts) == (QualityOutcome.EVALUATOR_ERROR, None, 2)


def test_evaluator_error_retries_then_pauses_explicitly() -> None:
    # Given: a provider boundary that always raises.
    evaluator = RecordingEvaluator(error=RuntimeError("offline"))
    # When: the adapter exhausts its evaluator attempts.
    decision = _adapter(evaluator).review(_request(), QualityPolicy())
    # Then: it emits EVALUATOR_ERROR with no numeric score.
    assert (decision.outcome, decision.score, decision.evaluator_attempts) == (QualityOutcome.EVALUATOR_ERROR, None, 2)


def test_exhausted_repair_budget_pauses() -> None:
    # Given: a repairable length failure after two targeted repairs.
    request = _request(body="中" * 79, repair_attempts=2)
    # When: the policy evaluates the exhausted request.
    decision = _adapter().review(request, QualityPolicy())
    # Then: it pauses instead of scheduling a third repair.
    assert (decision.outcome, decision.reason_codes) == (QualityOutcome.PAUSE, ("repair_budget_exhausted",))


def test_project_adapters_keep_character_fixtures_isolated() -> None:
    # Given: two adapters with distinct project-scoped character fixtures and evaluator traces.
    seen_a: list[EvaluatorAdapterRequest] = []
    seen_b: list[EvaluatorAdapterRequest] = []
    adapter_a = _adapter(RecordingEvaluator(seen=seen_a), characters=(CharacterFixture(character_id="a", name="甲"),))
    adapter_b = _adapter(
        RecordingEvaluator(seen=seen_b),
        project_id="project-b",
        characters=(CharacterFixture(character_id="b", name="乙"),),
    )
    # When: each adapter reviews its own project's chapter.
    decision_a = adapter_a.review(_request(body="甲" * 100), QualityPolicy())
    decision_b = adapter_b.review(_request(project_id=ProjectId("project-b"), body="乙" * 100), QualityPolicy())
    # Then: metrics and evaluator dependencies observe only their owning fixture.
    assert (decision_a.metrics.mentioned_character_ids, decision_b.metrics.mentioned_character_ids) == (("a",), ("b",))
    assert (seen_a[0].characters[0].name, seen_b[0].characters[0].name) == ("甲", "乙")


def test_beat_match_tolerates_paraphrase() -> None:
    request = _request(
        required_beats=("抵达学院",),
        located_beats=("林晚抵达了魔法学院门口",),
    )
    decision = _adapter().review(request, QualityPolicy())
    assert decision.outcome is QualityOutcome.ACCEPT


def test_missing_character_coverage_repairs() -> None:
    request = _request(body="中" * 100)
    characters = (CharacterFixture(character_id="c-1", name="林晚"),)
    decision = _adapter(characters=characters).review(
        request, QualityPolicy(minimum_mentioned_characters=1)
    )
    assert decision.outcome is QualityOutcome.REPAIR
    assert "missing_character_coverage" in decision.reason_codes
    assert any("missing_character_coverage" in item for item in decision.feedback)


def test_quality_context_is_passed_to_evaluator() -> None:
    seen: list[EvaluatorAdapterRequest] = []
    evaluator = RecordingEvaluator(seen=seen)
    request = _request(
        style_guidance=("冷峻克制",),
        world_rules=("魔法不可复活死者",),
        acceptance_criteria=("主角必须出场",),
        previous_summary="上一章主角抵达学院。",
    )
    decision = _adapter(evaluator).review(request, QualityPolicy())
    assert decision.outcome is QualityOutcome.ACCEPT
    assert seen
    assert seen[0].style_guidance == ("冷峻克制",)
    assert seen[0].world_rules == ("魔法不可复活死者",)
    assert seen[0].acceptance_criteria == ("主角必须出场",)
    assert seen[0].previous_summary == "上一章主角抵达学院。"


def test_low_dimension_feedback_is_actionable() -> None:
    response = VALID_SCORES.replace('"score":4', '"score":2', 1)
    decision = _adapter(RecordingEvaluator(response=response)).review(_request(), QualityPolicy())
    assert decision.outcome is QualityOutcome.REPAIR
    assert decision.feedback
    assert any("style_consistency" in item for item in decision.feedback)


def test_quality_gate_uses_latest_report() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE quality_reports(artifact_id TEXT, artifact_revision INTEGER, "
        "decision TEXT, created_at TEXT)"
    )
    connection.execute(
        "INSERT INTO quality_reports VALUES ('chapter-1', 1, 'REPAIR', '2030-01-01')"
    )
    assert not quality_accepted(connection, "chapter-1", 1)
    connection.execute(
        "INSERT INTO quality_reports VALUES ('chapter-1', 1, 'ACCEPT', '2030-01-02')"
    )
    assert quality_accepted(connection, "chapter-1", 1)
    connection.execute(
        "INSERT INTO quality_reports VALUES ('chapter-1', 1, 'PAUSE', '2030-01-03')"
    )
    assert not quality_accepted(connection, "chapter-1", 1)
