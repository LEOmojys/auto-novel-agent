from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, assert_never

from novel_agent.skill_manifests.models import (
    ActiveSkillPin,
    BUILTIN_SKILL_IDS,
    CandidateEvidence,
    SchemaCompatibility,
    SelectedSkill,
    SelectionCode,
    SkillKind,
    SkillManifest,
    SkillPause,
    SkillSelection,
    SkillStatus,
    TaskDescriptor,
)


_MANIFEST_ROOT: Final = Path(__file__).with_name("skill_manifests")


@dataclass(frozen=True, slots=True)
class SkillCatalogError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail
class SkillCatalog:
    def __init__(self, manifests: tuple[SkillManifest, ...]) -> None:
        self._manifests = {manifest.skill_ref: manifest for manifest in manifests}
        self._pins: dict[tuple[str, str], ActiveSkillPin] = {}

    @property
    def manifests(self) -> tuple[SkillManifest, ...]:
        return tuple(sorted(self._manifests.values(), key=lambda item: item.skill_ref))

    @property
    def pins(self) -> tuple[ActiveSkillPin, ...]:
        return tuple(self._pins.values())

    def set_status(self, skill_ref: str, status: SkillStatus) -> None:
        manifest = self._manifests[skill_ref]
        self._manifests[skill_ref] = manifest.model_copy(update={"status": status})

    def pin(self, run_id: str, selected: SelectedSkill) -> ActiveSkillPin:
        key = (run_id, selected.manifest.skill_id)
        prior = self._pins.get(key)
        if prior is not None:
            return prior
        pin = ActiveSkillPin(
            run_id=run_id,
            manifest=selected.manifest,
            manifest_hash=selected.manifest_hash,
            template_hash=selected.template_hash,
        )
        self._pins[key] = pin
        return pin

    def resolve_pin(self, run_id: str, skill_id: str) -> ActiveSkillPin:
        return self._pins[(run_id, skill_id)]


class SkillSelector:
    def __init__(self, catalog: SkillCatalog) -> None:
        self._catalog = catalog

    def select(self, descriptor: TaskDescriptor, *, shadow: bool = False) -> SkillSelection | SkillPause:
        evidence = tuple(self._evaluate(item, descriptor, shadow) for item in self._catalog.manifests)
        primary = self._choose(self._catalog.manifests, evidence, SkillKind.PRIMARY)
        match primary:
            case SelectionCode.SKILL_AMBIGUOUS | SelectionCode.SKILL_NOT_FOUND:
                return SkillPause(
                    code=primary,
                    descriptor_hash=descriptor.descriptor_hash,
                    reason=(
                        "no exact Skill"
                        if primary is SelectionCode.SKILL_NOT_FOUND
                        else "Skill score or lead is ambiguous"
                    ),
                    candidates=evidence,
                )
            case SelectionCode.SELECTED:
                raise SkillCatalogError(detail="selector returned a non-terminal selection code")
            case SelectedSkill() as selected_primary:
                pass
            case unreachable:
                assert_never(unreachable)
        audit_manifests = tuple(
            item for item in self._catalog.manifests if selected_primary.manifest.skill_id in item.audit_for
        )
        audit_evidence = tuple(self._evaluate(item, descriptor, shadow) for item in audit_manifests)
        audit = self._choose(audit_manifests, audit_evidence, SkillKind.AUDIT)
        match audit:
            case SelectionCode.SKILL_AMBIGUOUS:
                return SkillPause(
                    code=audit,
                    descriptor_hash=descriptor.descriptor_hash,
                    reason="audit Skill score or lead is ambiguous",
                    candidates=(*evidence, *audit_evidence),
                )
            case SelectionCode.SKILL_NOT_FOUND:
                selected_audit = None
            case SelectionCode.SELECTED:
                raise SkillCatalogError(detail="selector returned a non-terminal audit code")
            case SelectedSkill() as selected_audit:
                pass
            case unreachable:
                assert_never(unreachable)
        return SkillSelection(
            descriptor_hash=descriptor.descriptor_hash,
            primary=selected_primary,
            audit=selected_audit,
            candidates=(*evidence, *audit_evidence),
            shadow=shadow,
        )

    @staticmethod
    def _evaluate(manifest: SkillManifest, task: TaskDescriptor, shadow: bool) -> CandidateEvidence:
        valid_statuses = {SkillStatus.SHADOW} if shadow else {SkillStatus.ENABLED}
        reasons: list[str] = []
        checks = (
            (manifest.status in valid_statuses, "status"),
            (task.task_type in manifest.task_types, "task_type"),
            (task.run_state in manifest.run_states, "run_state"),
            (task.actor_role in manifest.allowed_roles, "actor_role"),
            (
                manifest.compatibility.minimum_input <= task.input_schema_version <= manifest.compatibility.maximum_input
                and manifest.compatibility.minimum_output
                <= task.expected_output_schema_version
                <= manifest.compatibility.maximum_output,
                "schema",
            ),
            (manifest.required_context_slices <= task.available_context_slices, "required_context"),
            (manifest.required_artifact_kinds <= task.artifact_kinds, "required_artifacts"),
            (manifest.allowed_tool_ids <= task.authorized_tool_ids, "allowed_tools"),
            (manifest.context_budget <= task.remaining_context_tokens, "context_budget"),
            (manifest.max_tool_calls <= task.remaining_tool_calls, "tool_budget"),
            (manifest.max_repairs <= task.remaining_repairs, "repair_budget"),
            (manifest.timeout_budget <= task.remaining_timeout_seconds, "timeout_budget"),
        )
        reasons.extend(name for passed, name in checks if not passed)
        exact_schema = (
            manifest.compatibility.minimum_input
            == manifest.compatibility.maximum_input
            == task.input_schema_version
            and manifest.compatibility.minimum_output
            == manifest.compatibility.maximum_output
            == task.expected_output_schema_version
        )
        genre_match = not manifest.genre_tags or bool(manifest.genre_tags & task.genre_tags)
        stable_preference = int(manifest.version.split(".", maxsplit=1)[0]) >= 1
        score_parts = (
            (40, "task_type=40"),
            (25, "run_state=25"),
            (15, "role=15"),
            (10 if exact_schema else 0, f"schema_context={10 if exact_schema else 0}"),
            (5 if genre_match else 0, f"genre={5 if genre_match else 0}"),
            (5 if stable_preference else 0, f"stable={5 if stable_preference else 0}"),
        )
        score = sum(points for points, _ in score_parts)
        return CandidateEvidence(
            skill_ref=manifest.skill_ref,
            score=None if reasons else score,
            score_evidence=() if reasons else tuple(item for _, item in score_parts),
            reasons=tuple(reasons),
        )

    @staticmethod
    def _choose(
        manifests: tuple[SkillManifest, ...], evidence: tuple[CandidateEvidence, ...], kind: SkillKind
    ) -> SelectedSkill | SelectionCode:
        by_ref = {item.skill_ref: item for item in manifests}
        ranked = sorted(
            (item for item in evidence if item.score is not None and by_ref[item.skill_ref].kind is kind),
            key=lambda item: (-int(item.score or 0), item.skill_ref),
        )
        if not ranked:
            return SelectionCode.SKILL_NOT_FOUND
        lead = ranked[0].score or 0
        if lead < 80 or (len(ranked) > 1 and lead - int(ranked[1].score or 0) < 10):
            return SelectionCode.SKILL_AMBIGUOUS
        winner = ranked[0]
        return SelectedSkill(manifest=by_ref[winner.skill_ref], score=lead, score_evidence=winner.score_evidence)


def load_builtin_catalog() -> SkillCatalog:
    manifests: list[SkillManifest] = []
    for path in sorted(_MANIFEST_ROOT.glob("*.json")):
        manifest = SkillManifest.model_validate_json(path.read_text(encoding="utf-8"))
        template_path = _MANIFEST_ROOT / manifest.prompt_template_path
        template_hash = hashlib.sha256(template_path.read_bytes()).hexdigest()
        if template_hash != manifest.prompt_template_hash:
            raise SkillCatalogError(detail=f"template hash mismatch for {manifest.skill_ref}")
        manifests.append(manifest)
    if frozenset(item.skill_id for item in manifests) != BUILTIN_SKILL_IDS:
        raise SkillCatalogError(detail="built-in Skill catalog is incomplete")
    return SkillCatalog(tuple(manifests))


def load_builtin_templates(catalog: SkillCatalog) -> dict[str, str]:
    """Load every enabled Skill's instruction template, keyed by skill id.

    Templates are the actual task instructions: a domain adapter receives
    them from the catalog and formats them with its own placeholders, so the
    governed creative prompts live and evolve as versioned Skill assets.
    """
    templates: dict[str, str] = {}
    for manifest in catalog.manifests:
        template_path = _MANIFEST_ROOT / manifest.prompt_template_path
        templates[manifest.skill_id] = template_path.read_text(encoding="utf-8")
    return templates


__all__ = (
    "ActiveSkillPin",
    "BUILTIN_SKILL_IDS",
    "CandidateEvidence",
    "SchemaCompatibility",
    "SelectedSkill",
    "SelectionCode",
    "SkillCatalog",
    "SkillCatalogError",
    "SkillKind",
    "SkillManifest",
    "SkillPause",
    "SkillSelection",
    "SkillSelector",
    "SkillStatus",
    "TaskDescriptor",
    "load_builtin_catalog",
    "load_builtin_templates",
)
