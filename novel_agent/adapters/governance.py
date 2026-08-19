from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from novel_agent.adapters.models import (
    LegacyEvaluationFailure,
    LegacyEvaluationRequest,
    LegacyEvaluationResult,
    M0CandidateExtraction,
    M0CandidateExtractionRequest,
)
from novel_agent.memory_models import (
    EntityResolution,
    InferenceMode,
    MemoryFactKind,
    MemoryProposal,
    PromotionAuthority,
    TemporalCertainty,
)


class _LegacyEvaluationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scores: dict[str, float]
    reasons: dict[str, str]
    avg_score: float


@dataclass(frozen=True, slots=True)
class NovelSaveCandidateAdapter:
    def extract(self, request: M0CandidateExtractionRequest) -> M0CandidateExtraction:
        # Memory spans are measured against the artifact's verbatim text (the
        # staged raw).  ``source_text`` is that text; it falls back to the
        # prose body only for legacy callers that never pass the raw.
        source_text = request.source_text if request.source_text is not None else request.chapter_body
        proposal = MemoryProposal(
            run_id=request.run_id,
            project_id=request.project_id,
            source_artifact_id=request.source_artifact_id,
            source_artifact_revision=request.source_artifact_revision,
            extraction_schema_version=1,
            target_layer=request.target_layer,
            kind=MemoryFactKind.CHAPTER_SUMMARY,
            subject_id=request.subject_id,
            predicate=request.predicate,
            value_json=request.value_json,
            source_start=0,
            source_end=len(source_text),
            source_excerpt=source_text,
            entity_resolution=EntityResolution.EXACT,
            inference_mode=InferenceMode.DIRECT,
            temporal_certainty=TemporalCertainty.EXACT,
            confidence=1.0,
            authority=PromotionAuthority.ACCEPTED_CHAPTER,
            plan_revision=0,
            supersedes_memory_id=None,
            created_at=request.created_at,
        )
        return M0CandidateExtraction(request=request, proposal=proposal)


@dataclass(frozen=True, slots=True)
class LegacyEvaluationAdapter:
    def evaluate(self, request: LegacyEvaluationRequest) -> LegacyEvaluationResult | LegacyEvaluationFailure:
        try:
            characters = json.loads(request.characters_json)
        except json.JSONDecodeError:
            return _failure(b"", "characters_json_invalid")
        from evaluate import llm_evaluate_chapter

        response = llm_evaluate_chapter(
            request.chapter_body,
            characters,
            request.chapter_label,
            request.chapter_requirement,
        )
        raw_bytes = json.dumps(response, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        try:
            payload = _LegacyEvaluationPayload.model_validate_json(raw_bytes)
        except ValidationError:
            return _failure(raw_bytes, "legacy_evaluation_shape_invalid")
        return LegacyEvaluationResult(
            raw_bytes=raw_bytes,
            raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            scores_json=json.dumps(payload.scores, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            reasons_json=json.dumps(payload.reasons, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            average_score=payload.avg_score,
        )


def _failure(raw_bytes: bytes, reason_code: str) -> LegacyEvaluationFailure:
    return LegacyEvaluationFailure(
        raw_bytes=raw_bytes,
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        reason_code=reason_code,
    )


__all__ = ("LegacyEvaluationAdapter", "NovelSaveCandidateAdapter")
