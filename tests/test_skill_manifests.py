from __future__ import annotations

from pathlib import Path

from novel_agent.skills import BUILTIN_SKILL_IDS, SkillStatus, load_builtin_catalog


def test_builtin_manifests_are_complete_and_hash_verified() -> None:
    # Given: the packaged first-party manifest directory.
    catalog = load_builtin_catalog()

    # When: all manifests and templates are parsed at the trust boundary.
    manifests = catalog.manifests

    # Then: all nine governed IDs are enabled, versioned, and hash pinned.
    assert frozenset(manifest.skill_id for manifest in manifests) == BUILTIN_SKILL_IDS
    assert all(manifest.status is SkillStatus.ENABLED for manifest in manifests)
    assert all(manifest.skill_ref.endswith("@1.0.0") for manifest in manifests)
    assert all(len(manifest.prompt_template_hash) == 64 for manifest in manifests)
    assert all(len(manifest.manifest_hash) == 64 for manifest in manifests)


def test_manifests_do_not_expose_direct_resource_capabilities() -> None:
    # Given: validated built-in manifests.
    manifests = load_builtin_catalog().manifests

    # When: declared tools and template paths are inspected.
    declarations = tuple(tool for manifest in manifests for tool in manifest.allowed_tool_ids)

    # Then: Skills expose tool references only, never paths, URLs, DB handles, or providers.
    assert all("@" in tool for tool in declarations)
    assert all(not Path(manifest.prompt_template_path).is_absolute() for manifest in manifests)
    assert all("artifact.commit" not in tool for tool in declarations)

