from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from novel_agent.contracts.common import MemoryLayer
from novel_agent.memory_models import (
    CandidateEvidence,
    CandidateStatus,
    EntityResolution,
    InferenceMode,
    MemoryFactKind,
    PromotionAuthority,
    PromotionOutcome,
    ReviewCode,
    SourceArtifactStatus,
    TemporalCertainty,
)


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    outcome: PromotionOutcome
    candidate_status: CandidateStatus
    review_code: ReviewCode | None
    requires_pause: bool


def decide_promotion(evidence: CandidateEvidence, has_conflict: bool) -> PromotionDecision:
    review_code = _review_code(evidence)
    if review_code is not None:
        return PromotionDecision(
            outcome=PromotionOutcome.REVIEW_REQUIRED,
            candidate_status=CandidateStatus.NEEDS_REVIEW,
            review_code=review_code,
            requires_pause=False,
        )
    if has_conflict and not _is_explicit_retcon(evidence):
        return PromotionDecision(
            outcome=PromotionOutcome.CONFLICTED,
            candidate_status=CandidateStatus.CONFLICTED,
            review_code=ReviewCode.CONFLICT,
            requires_pause=True,
        )
    return PromotionDecision(
        outcome=PromotionOutcome.PROMOTED,
        candidate_status=CandidateStatus.PROMOTED,
        review_code=None,
        requires_pause=False,
    )


def _review_code(evidence: CandidateEvidence) -> ReviewCode | None:
    proposal = evidence.proposal
    if evidence.source_status is not SourceArtifactStatus.ACCEPTED:
        return ReviewCode.SOURCE_NOT_ACCEPTED
    if not _has_valid_authority(proposal.target_layer, proposal.authority, proposal.supersedes_memory_id):
        return ReviewCode.INVALID_PROMOTION
    fact_code = _fact_code(proposal.target_layer, proposal.kind)
    if fact_code is not None:
        return fact_code
    entity_code = _entity_code(proposal.entity_resolution)
    if entity_code is not None:
        return entity_code
    inference_code = _inference_code(proposal.inference_mode)
    if inference_code is not None:
        return inference_code
    temporal_code = _temporal_code(proposal.temporal_certainty)
    if temporal_code is not None:
        return temporal_code
    if proposal.confidence < 0.95:
        return ReviewCode.LOW_CONFIDENCE
    return None


def _has_valid_authority(
    layer: MemoryLayer,
    authority: PromotionAuthority,
    supersedes_memory_id: str | None,
) -> bool:
    match layer:
        case MemoryLayer.M1_CANONICAL:
            match authority:
                case PromotionAuthority.VALIDATED_WORLD:
                    return supersedes_memory_id is None
                case PromotionAuthority.USER_RETCON:
                    return supersedes_memory_id is not None
                case PromotionAuthority.VALIDATED_PLAN | PromotionAuthority.ACCEPTED_CHAPTER:
                    return False
                case unreachable:
                    assert_never(unreachable)
        case MemoryLayer.M2_PLAN:
            match authority:
                case PromotionAuthority.VALIDATED_PLAN:
                    return supersedes_memory_id is None
                case (
                    PromotionAuthority.VALIDATED_WORLD
                    | PromotionAuthority.USER_RETCON
                    | PromotionAuthority.ACCEPTED_CHAPTER
                ):
                    return False
                case unreachable:
                    assert_never(unreachable)
        case MemoryLayer.M3_CONFIRMED:
            match authority:
                case PromotionAuthority.ACCEPTED_CHAPTER:
                    return supersedes_memory_id is None
                case (
                    PromotionAuthority.VALIDATED_WORLD
                    | PromotionAuthority.USER_RETCON
                    | PromotionAuthority.VALIDATED_PLAN
                ):
                    return False
                case unreachable:
                    assert_never(unreachable)
        case MemoryLayer.M0_CANDIDATE | MemoryLayer.M4_RETRIEVAL | MemoryLayer.M5_WORKING:
            return False
        case unreachable:
            assert_never(unreachable)


def _fact_code(layer: MemoryLayer, kind: MemoryFactKind) -> ReviewCode | None:
    match layer:
        case MemoryLayer.M1_CANONICAL:
            match kind:
                case MemoryFactKind.WORLD_RULE | MemoryFactKind.CHARACTER_IDENTITY:
                    return None
                case (
                    MemoryFactKind.PLAN_GOAL
                    | MemoryFactKind.PLAN_HOOK
                    | MemoryFactKind.CHAPTER_SUMMARY
                    | MemoryFactKind.DIRECT_EVENT
                    | MemoryFactKind.STATE_CHANGE
                    | MemoryFactKind.HOOK_STATUS
                    | MemoryFactKind.RELATIONSHIP
                    | MemoryFactKind.CAUSALITY
                    | MemoryFactKind.MOTIVATION
                ):
                    return ReviewCode.INVALID_PROMOTION
                case unreachable:
                    assert_never(unreachable)
        case MemoryLayer.M2_PLAN:
            match kind:
                case MemoryFactKind.PLAN_GOAL | MemoryFactKind.PLAN_HOOK:
                    return None
                case (
                    MemoryFactKind.WORLD_RULE
                    | MemoryFactKind.CHARACTER_IDENTITY
                    | MemoryFactKind.CHAPTER_SUMMARY
                    | MemoryFactKind.DIRECT_EVENT
                    | MemoryFactKind.STATE_CHANGE
                    | MemoryFactKind.HOOK_STATUS
                    | MemoryFactKind.RELATIONSHIP
                    | MemoryFactKind.CAUSALITY
                    | MemoryFactKind.MOTIVATION
                ):
                    return ReviewCode.INVALID_PROMOTION
                case unreachable:
                    assert_never(unreachable)
        case MemoryLayer.M3_CONFIRMED:
            match kind:
                case (
                    MemoryFactKind.CHAPTER_SUMMARY
                    | MemoryFactKind.DIRECT_EVENT
                    | MemoryFactKind.STATE_CHANGE
                    | MemoryFactKind.HOOK_STATUS
                ):
                    return None
                case (
                    MemoryFactKind.WORLD_RULE
                    | MemoryFactKind.CHARACTER_IDENTITY
                    | MemoryFactKind.PLAN_GOAL
                    | MemoryFactKind.PLAN_HOOK
                    | MemoryFactKind.RELATIONSHIP
                    | MemoryFactKind.CAUSALITY
                    | MemoryFactKind.MOTIVATION
                ):
                    return ReviewCode.INVALID_PROMOTION
                case unreachable:
                    assert_never(unreachable)
        case MemoryLayer.M0_CANDIDATE | MemoryLayer.M4_RETRIEVAL | MemoryLayer.M5_WORKING:
            return ReviewCode.INVALID_PROMOTION
        case unreachable:
            assert_never(unreachable)


def _entity_code(resolution: EntityResolution) -> ReviewCode | None:
    match resolution:
        case EntityResolution.EXACT:
            return None
        case EntityResolution.AMBIGUOUS:
            return ReviewCode.AMBIGUOUS_ENTITY
        case EntityResolution.NEW:
            return ReviewCode.NEW_ENTITY
        case EntityResolution.UNRESOLVED:
            return ReviewCode.UNRESOLVED_ENTITY
        case unreachable:
            assert_never(unreachable)


def _inference_code(mode: InferenceMode) -> ReviewCode | None:
    match mode:
        case InferenceMode.DIRECT:
            return None
        case InferenceMode.RELATIONSHIP:
            return ReviewCode.INFERRED_RELATIONSHIP
        case InferenceMode.CAUSAL:
            return ReviewCode.INFERRED_CAUSALITY
        case InferenceMode.MOTIVATION:
            return ReviewCode.INFERRED_MOTIVATION
        case unreachable:
            assert_never(unreachable)


def _temporal_code(certainty: TemporalCertainty) -> ReviewCode | None:
    match certainty:
        case TemporalCertainty.EXACT:
            return None
        case TemporalCertainty.UNCERTAIN:
            return ReviewCode.UNCERTAIN_TIME
        case unreachable:
            assert_never(unreachable)


def _is_explicit_retcon(evidence: CandidateEvidence) -> bool:
    match evidence.proposal.authority:
        case PromotionAuthority.USER_RETCON:
            return evidence.proposal.supersedes_memory_id is not None
        case (
            PromotionAuthority.VALIDATED_WORLD
            | PromotionAuthority.VALIDATED_PLAN
            | PromotionAuthority.ACCEPTED_CHAPTER
        ):
            return False
        case unreachable:
            assert_never(unreachable)


__all__ = ("PromotionDecision", "decide_promotion")
