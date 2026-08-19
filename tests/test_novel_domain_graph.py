from __future__ import annotations

from datetime import UTC, datetime

from novel_agent.agent.brief import BriefContract
from novel_agent.agent.task_graph import topological_order, validate_task_graph
from novel_agent.contracts import Provenance
from novel_agent.domains.novel import NovelDomainPlugin
from novel_agent.offline_fake import (
    OfflineChapterGenerator,
    OfflineContinuityAuditor,
    OfflineMemoryExtractor,
    OfflinePlanningGenerator,
    OfflineQualityReviewer,
)
from novel_agent.pathing import ProjectId


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")


def _plugin() -> NovelDomainPlugin:
    return NovelDomainPlugin(
        planning_generator=OfflinePlanningGenerator(FIXED_TIME),
        chapter_generator=OfflineChapterGenerator(FIXED_TIME),
        extractor=OfflineMemoryExtractor(FIXED_TIME, PROJECT_ID),
    )


def _brief(chapter_count: int = 3) -> BriefContract:
    return BriefContract(
        brief_id="brief-001",
        project_id=PROJECT_ID,
        raw_requirements="写一本三章的魔法学院小说",
        goal="目标",
        target_chapter_count=chapter_count,
        created_at=FIXED_TIME,
        provenance=Provenance(
            source_kind="user_request",
            source_id="run-001",
            source_sha256="a" * 64,
            observed_at=FIXED_TIME,
        ),
    )


def test_novel_graph_structure_and_dependencies() -> None:
    graph = _plugin().build_graph(_brief(chapter_count=3), run_id="run-001", project_id=PROJECT_ID, now=lambda: FIXED_TIME)

    # Three planning nodes plus one chapter node per chapter.
    assert len(graph.nodes) == 6
    ids = [node.node_id for node in graph.nodes]
    assert ids[:3] == ["world", "cast", "outline"]
    assert ids[3:] == ["chapter_1", "chapter_2", "chapter_3"]

    by_id = graph.nodes_by_id
    assert by_id["world"].depends_on == ()
    assert by_id["cast"].depends_on == ("world",)
    assert by_id["outline"].depends_on == ("world", "cast")
    assert by_id["chapter_1"].depends_on == ("outline",)
    assert by_id["chapter_2"].depends_on == ("outline", "chapter_1")
    assert by_id["chapter_3"].depends_on == ("outline", "chapter_2")

    # Chapter nodes declare the gated tool chain and skills.
    chapter = by_id["chapter_2"]
    assert "chapter.draft@1.0.0" in chapter.tools
    assert "quality.evaluate@1.0.0" in chapter.tools
    assert "chapter-scene@1.0.0" in chapter.skills
    assert dict(chapter.metadata)["chapter_number"] == "2"

    # The graph is topologically valid and acyclic.
    assert validate_task_graph(graph) == ()
    ordered = topological_order(graph)
    assert ordered is not None
    ordered_ids = [node.node_id for node in ordered]
    assert ordered_ids.index("world") < ordered_ids.index("cast") < ordered_ids.index("outline")
    assert ordered_ids.index("outline") < ordered_ids.index("chapter_1") < ordered_ids.index("chapter_2")


def test_novel_graph_has_domain_executors_for_every_node() -> None:
    plugin = _plugin()
    graph = plugin.build_graph(_brief(chapter_count=2), run_id="run-001", project_id=PROJECT_ID, now=lambda: FIXED_TIME)

    for node in graph.nodes:
        executor = plugin.node_executor(node)
        assert executor is not None, f"missing executor for {node.node_id}"
