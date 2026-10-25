"""Lazy provider-backed quality reviewer for the agent loop.

The agent loop writes chapters through tools, so it never visits the DAG cast
node where `real_reviewer_factory` is normally swapped in.  As a result the
loop's quality gate stayed on `OfflineQualityReviewer` (always ACCEPT) in
production.

This reviewer resolves the character cast at first use (the cast artifact is
already accepted by the time a chapter is written) and then delegates to the
same `ProjectScopedEvaluationAdapter` + `ProviderStructuredEvaluator` used by
the DAG path.  If the cast cannot be resolved the chapter is paused instead of
being accepted by a fabricated score.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from novel_agent.contracts import CharacterSet
from novel_agent.domains.quality_evaluators import (
    CharCountMetricsEvaluator,
    ProviderStructuredEvaluator,
)
from novel_agent.pathing import ProjectId
from novel_agent.provider_types import ProviderTransport
from novel_agent.quality import (
    CharacterFixture,
    ChapterQualityRequest,
    ProjectScopedEvaluationAdapter,
    QualityDecision,
    QualityOutcome,
)


@dataclass(slots=True)
class LazyProviderQualityReviewer:
    """Build the live six-dimension reviewer on first use."""

    connection: object
    project_id: ProjectId
    run_id: str
    transport: ProviderTransport
    now: Callable[[], datetime]
    prompt: str | None = None
    failure_sink: Callable[[str, str, str, str, str], None] | None = None
    call_event_sink: Callable[[str, str, str], None] | None = None
    _delegate: ProjectScopedEvaluationAdapter | None = field(default=None, init=False, repr=False)
    _resolved: bool = field(default=False, init=False, repr=False)

    def review(self, request: ChapterQualityRequest) -> QualityDecision:
        if not self._resolved:
            self._delegate = self._build()
            self._resolved = True
        if self._delegate is None:
            return QualityDecision(
                outcome=QualityOutcome.PAUSE,
                score=None,
                reason_codes=("reviewer_unavailable",),
                evaluator_attempts=0,
                policy_version="quality-runtime-v1",
                metrics=None,
            )
        return self._delegate.review(request)

    def _build(self) -> ProjectScopedEvaluationAdapter | None:
        row = self.connection.execute(
            "SELECT raw_bytes FROM artifacts WHERE kind = 'character_set' AND status = 'accepted' "
            "ORDER BY created_at DESC, revision DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        try:
            cast = CharacterSet.model_validate_json(bytes(row[0]))
        except Exception:  # noqa: BROAD_EXCEPT_OK - malformed cast must pause, not pass
            return None
        fixtures = tuple(
            CharacterFixture(
                character_id=character.character_id,
                name=character.name,
                role=character.role,
                goal=character.goal,
                traits=character.traits,
            )
            for character in cast.characters
        )
        return ProjectScopedEvaluationAdapter(
            project_id=self.project_id,
            characters=fixtures,
            evaluator=ProviderStructuredEvaluator(
                transport=self.transport,
                now=self.now,
                run_id=self.run_id,
                prompt=self.prompt,
                failure_sink=self.failure_sink,
                call_event_sink=self.call_event_sink,
            ),
            metrics=CharCountMetricsEvaluator(),
        )


__all__ = ("LazyProviderQualityReviewer",)
