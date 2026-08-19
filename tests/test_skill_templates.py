from __future__ import annotations

from datetime import UTC, datetime

import pytest

from novel_agent.adapters import GovernedCreativeAdapter
from novel_agent.agent.brief import BriefAnalyzer, BriefContract
from novel_agent.contracts import Provenance, ProvenanceKind, StoryBrief
from novel_agent.pathing import ProjectId
from novel_agent.provider_types import ProviderTransportResult
from novel_agent.skills import SkillCatalogError, load_builtin_catalog, load_builtin_templates


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")


class CapturingTransport:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, request) -> ProviderTransportResult:
        self.prompts.append(request.prompt)
        return ProviderTransportResult(
            content='{"title":"t","premise":"p","rules":[],"constraints":[]}',
            model_name="t",
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_builtin_templates_are_real_instructions() -> None:
    templates = load_builtin_templates(load_builtin_catalog())

    assert "资深世界观架构师" in templates["world-building"]
    assert "资深角色设计师" in templates["cast-network"]
    assert "剧情架构师" in templates["story-architecture"]
    assert "华语小说家" in templates["chapter-scene"]
    assert "摘要" in templates["memory-extraction"]
    assert "需求分析器" in templates["brief-normalization"]
    assert "评审编辑" in templates["quality-review"]
    assert "连续性审计员" in templates["continuity-audit"]
    assert "修订型小说家" in templates["targeted-revision"]
    # Templates carry their format placeholders.
    assert "{requirements}" in templates["world-building"]
    assert "{chapter_body}" in templates["memory-extraction"]


def test_governed_adapter_uses_injected_skill_template() -> None:
    templates = load_builtin_templates(load_builtin_catalog())
    transport = CapturingTransport()
    adapter = GovernedCreativeAdapter(
        transport=transport,
        now=lambda: FIXED_TIME,
        run_id="run-001",
        project_id=PROJECT_ID,
        prompts=templates,
    )
    brief = StoryBrief(
        brief_id="b",
        project_id=PROJECT_ID,
        user_requirements="一个魔法学院的故事",
        genre="fiction",
        target_chapter_count=1,
        created_at=FIXED_TIME,
        provenance=Provenance(
            source_kind=ProvenanceKind.USER_REQUEST,
            source_id="run-001",
            source_sha256="a" * 64,
            observed_at=FIXED_TIME,
        ),
    )

    adapter.generate_world(brief)

    # The world prompt came from the skill template asset, not the code default.
    assert "资深世界观架构师" in transport.prompts[0]
    assert transport.prompts[0].startswith("你是一位资深世界观架构师")


def test_brief_analyzer_uses_injected_skill_template() -> None:
    templates = load_builtin_templates(load_builtin_catalog())
    transport = CapturingTransport()

    brief: BriefContract = BriefAnalyzer(
        transport=transport,
        now=lambda: FIXED_TIME,
        prompt=templates["brief-normalization"],
    ).analyze(
        "写一本小说",
        project_id=PROJECT_ID,
        run_id="run-001",
        brief_id="run-001:brief",
    )

    assert brief.goal == "" or brief.goal is not None
    assert transport.prompts[0].startswith("你是一个需求分析器")


def test_disabled_chapter_skill_blocks_pinning() -> None:
    from novel_agent.chapter_lifecycle import pin_chapter_skills
    from novel_agent.skill_manifests.models import SkillStatus

    catalog = load_builtin_catalog()
    catalog.set_status("chapter-scene@1.0.0", SkillStatus.DISABLED)

    with pytest.raises(SkillCatalogError):
        pin_chapter_skills(catalog, "run-001", repair=False)
