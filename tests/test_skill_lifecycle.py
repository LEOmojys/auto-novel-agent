from __future__ import annotations

from novel_agent.skills import (
    SelectionCode,
    SkillPause,
    SkillSelection,
    SkillSelector,
    SkillCatalog,
    SkillStatus,
    load_builtin_catalog,
)

from test_skill_selector import _draft_descriptor


def test_active_run_pin_survives_deprecation() -> None:
    # Given: a selected Skill pinned for an active run.
    catalog = load_builtin_catalog()
    selected = SkillSelector(catalog).select(_draft_descriptor())
    assert isinstance(selected, SkillSelection)
    pin = catalog.pin("run-1", selected.primary)

    # When: that version is deprecated for new selections.
    catalog.set_status("chapter-scene@1.0.0", SkillStatus.DEPRECATED)

    # Then: the active run resolves the original immutable hashes.
    assert catalog.resolve_pin("run-1", "chapter-scene").manifest_hash == pin.manifest_hash
    paused = SkillSelector(catalog).select(_draft_descriptor().model_copy(update={"run_id": "run-2"}))
    assert isinstance(paused, SkillPause)
    assert paused.code is SelectionCode.SKILL_NOT_FOUND


def test_disabled_skill_is_rejected_with_evidence() -> None:
    # Given: the only primary draft Skill is disabled.
    catalog = load_builtin_catalog()
    catalog.set_status("chapter-scene@1.0.0", SkillStatus.DISABLED)

    # When: a new run requests selection.
    result = SkillSelector(catalog).select(_draft_descriptor())

    # Then: the result pauses and records the lifecycle rejection.
    assert isinstance(result, SkillPause)
    assert any("status" in item.reasons for item in result.candidates)


def test_shadow_selection_has_no_catalog_side_effect() -> None:
    # Given: enabled production and shadow versions with identical applicability.
    source = next(item for item in load_builtin_catalog().manifests if item.skill_id == "chapter-scene")
    shadow = source.model_copy(update={"version": "1.1.0", "status": SkillStatus.SHADOW})
    catalog = SkillCatalog((source, shadow))
    before = catalog.pins

    # When: shadow selection is explicitly evaluated.
    result = SkillSelector(catalog).select(_draft_descriptor(), shadow=True)

    # Then: evidence is returned without pinning or invoking any effect.
    assert isinstance(result, SkillSelection)
    assert result.shadow is True
    assert result.primary.skill_ref == "chapter-scene@1.1.0"
    assert catalog.pins == before
