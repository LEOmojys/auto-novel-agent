from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, TypeVar, assert_never

from json_repair import repair_json
from pydantic import ValidationError

from novel_agent.adapters.models import (
    ChapterAdaptation,
    ChapterDraft,
    CharacterAdaptation,
    CreativeArtifactKind,
    CreativeParseRoute,
    OutlineAdaptation,
    RawCreativeArtifact,
    SchemaFailure,
    SchemaRepairRequest,
    WorldAdaptation,
)
from novel_agent.contracts import CharacterSet, ContractModel, PlanContract, WorldBible


_ADAPTER_VERSION = "creative-adapter-v1"
TContract = TypeVar("TContract", bound=ContractModel)


class SchemaRepairer(Protocol):
    def repair(self, request: SchemaRepairRequest) -> str:
        ...


@dataclass(frozen=True, slots=True)
class StructuredCreativeAdapter:
    repairer: SchemaRepairer | None = None
    adapter_version: str = _ADAPTER_VERSION

    def adapt_world(self, raw_text: str) -> WorldAdaptation | SchemaFailure:
        raw, parsed, route = self._parse(raw_text, CreativeArtifactKind.WORLD, WorldBible)
        match parsed:
            case WorldBible():
                return WorldAdaptation(raw=raw, value=parsed, parse_route=route)
            case SchemaFailure():
                return parsed
            case unreachable:
                assert_never(unreachable)

    def adapt_characters(self, raw_text: str) -> CharacterAdaptation | SchemaFailure:
        raw, parsed, route = self._parse(raw_text, CreativeArtifactKind.CHARACTERS, CharacterSet)
        match parsed:
            case CharacterSet():
                return CharacterAdaptation(raw=raw, value=parsed, parse_route=route)
            case SchemaFailure():
                return parsed
            case unreachable:
                assert_never(unreachable)

    def adapt_outline(self, raw_text: str) -> OutlineAdaptation | SchemaFailure:
        raw, parsed, route = self._parse(raw_text, CreativeArtifactKind.OUTLINE, PlanContract)
        match parsed:
            case PlanContract():
                return OutlineAdaptation(raw=raw, value=parsed, parse_route=route)
            case SchemaFailure():
                return parsed
            case unreachable:
                assert_never(unreachable)

    def adapt_chapter(self, raw_text: str) -> ChapterAdaptation | SchemaFailure:
        raw, parsed, route = self._parse(raw_text, CreativeArtifactKind.CHAPTER, ChapterDraft)
        match parsed:
            case ChapterDraft():
                return ChapterAdaptation(raw=raw, value=parsed, parse_route=route)
            case SchemaFailure():
                return parsed
            case unreachable:
                assert_never(unreachable)

    def _parse(
        self,
        raw_text: str,
        kind: CreativeArtifactKind,
        contract_type: type[TContract],
    ) -> tuple[RawCreativeArtifact, TContract | SchemaFailure, CreativeParseRoute]:
        raw = RawCreativeArtifact(
            kind=kind,
            raw_bytes=raw_text.encode(),
            content_sha256=hashlib.sha256(raw_text.encode()).hexdigest(),
            adapter_version=self.adapter_version,
        )
        strict = _validate(raw_text, contract_type)
        if strict is not None:
            return raw, strict, CreativeParseRoute.STRICT_JSON
        repaired = _repair_json(raw_text)
        parsed_repair = _validate(repaired, contract_type)
        if parsed_repair is not None:
            return raw, parsed_repair, CreativeParseRoute.JSON_REPAIR
        repairer = self.repairer
        if repairer is None:
            return raw, _failure(raw, contract_type, False), CreativeParseRoute.JSON_REPAIR
        request = SchemaRepairRequest(
            raw_artifact=raw,
            expected_contract=contract_type.__name__,
            contract_schema_json=json.dumps(contract_type.model_json_schema(), sort_keys=True),
        )
        repaired_contract = _validate(repairer.repair(request), contract_type)
        if repaired_contract is not None:
            return raw, repaired_contract, CreativeParseRoute.SCHEMA_REPAIR
        return raw, _failure(raw, contract_type, True), CreativeParseRoute.SCHEMA_REPAIR


def _validate(raw_text: str, contract_type: type[TContract]) -> TContract | None:
    try:
        return contract_type.model_validate_json(raw_text)
    except ValidationError:
        return None


def _repair_json(raw_text: str) -> str:
    try:
        return str(repair_json(raw_text))
    except ValueError:
        return ""


def _failure(
    raw: RawCreativeArtifact,
    contract_type: type[ContractModel],
    schema_repair_attempted: bool,
) -> SchemaFailure:
    routes = (CreativeParseRoute.STRICT_JSON, CreativeParseRoute.JSON_REPAIR)
    if schema_repair_attempted:
        routes = routes + (CreativeParseRoute.SCHEMA_REPAIR,)
    return SchemaFailure(
        raw_artifact=raw,
        expected_contract=contract_type.__name__,
        parse_routes=routes,
    )


__all__ = ("SchemaRepairer", "StructuredCreativeAdapter")
