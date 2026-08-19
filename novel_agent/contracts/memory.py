from __future__ import annotations

from datetime import datetime
from typing import Self, assert_never

from pydantic import field_validator, model_validator

from novel_agent.contracts.common import (
    ContractErrorCode,
    ContractModel,
    EvidenceBackedContract,
    MemoryLayer,
    Provenance,
    contract_error,
    validate_sha256,
)
from novel_agent.pathing import ProjectId


class MemoryCandidate(EvidenceBackedContract):
    candidate_id: str
    project_id: ProjectId
    memory_layer: MemoryLayer
    subject: str
    predicate: str
    value: str
    created_at: datetime
    provenance: Provenance

    @model_validator(mode="after")
    def validate_candidate_layer(self) -> Self:
        match self.memory_layer:
            case MemoryLayer.M0_CANDIDATE:
                return self
            case (
                MemoryLayer.M1_CANONICAL
                | MemoryLayer.M2_PLAN
                | MemoryLayer.M3_CONFIRMED
                | MemoryLayer.M4_RETRIEVAL
                | MemoryLayer.M5_WORKING
            ):
                raise contract_error(ContractErrorCode.INVALID_MEMORY_LAYER, "candidates must remain in M0")
            case unreachable:
                assert_never(unreachable)


class MemoryRecord(EvidenceBackedContract):
    memory_id: str
    project_id: ProjectId
    memory_layer: MemoryLayer
    subject: str
    predicate: str
    value: str
    supersedes_memory_id: str | None
    created_at: datetime
    provenance: Provenance

    @model_validator(mode="after")
    def validate_record_layer(self) -> Self:
        match self.memory_layer:
            case MemoryLayer.M1_CANONICAL | MemoryLayer.M2_PLAN | MemoryLayer.M3_CONFIRMED:
                return self
            case MemoryLayer.M0_CANDIDATE | MemoryLayer.M4_RETRIEVAL | MemoryLayer.M5_WORKING:
                raise contract_error(ContractErrorCode.INVALID_MEMORY_LAYER, "records must use a promotable memory layer")
            case unreachable:
                assert_never(unreachable)


class ContextItem(EvidenceBackedContract):
    memory_id: str
    content: str
    token_count: int
    included_at: datetime
    provenance: Provenance

    @field_validator("token_count")
    @classmethod
    def validate_token_count(cls, value: int) -> int:
        if value < 1:
            raise contract_error(ContractErrorCode.INVALID_CONTEXT_BUDGET, "token count must be positive")
        return value


class ContextPack(ContractModel):
    context_pack_id: str
    run_id: str
    chapter_plan_id: str
    max_tokens: int
    items: tuple[ContextItem, ...]
    assembled_at: datetime
    provenance: Provenance

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int) -> int:
        if value < 1:
            raise contract_error(ContractErrorCode.INVALID_CONTEXT_BUDGET, "maximum tokens must be positive")
        return value

    @model_validator(mode="after")
    def validate_context_budget(self) -> Self:
        total_tokens = sum(item.token_count for item in self.items)
        if total_tokens > self.max_tokens:
            raise contract_error(ContractErrorCode.INVALID_CONTEXT_BUDGET, "context items exceed maximum token budget")
        return self


class AuditReplayManifest(ContractModel):
    manifest_id: str
    run_id: str
    model_name: str
    temperature: float
    prompt_template_sha256: str
    response_sha256: str
    adapter_version: str
    created_at: datetime
    provenance: Provenance

    @field_validator("prompt_template_sha256", "response_sha256")
    @classmethod
    def validate_manifest_hashes(cls, value: str) -> str:
        return validate_sha256(value)
