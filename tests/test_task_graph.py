from __future__ import annotations

from datetime import UTC, datetime

import pytest

from novel_agent.agent.task_graph import TaskGraph, TaskGraphNode, topological_order, validate_task_graph
from novel_agent.contracts import ProvenanceKind
from novel_agent.pathing import ProjectId
from novel_agent.tool_models import AgentRole


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")


def _node(node_id: str, task_type: str, depends_on: tuple[str, ...] = ()) -> TaskGraphNode:
    return TaskGraphNode(
        node_id=node_id,
        task_type=task_type,
        depends_on=depends_on,
        output_artifact_kind="artifact",
        output_contract="Contract",
        role=AgentRole.CHAPTER_EXECUTOR,
        skills=("chapter-scene@1.0.0",),
        tools=("chapter.draft@1.0.0",),
    )


def _graph(*nodes: TaskGraphNode) -> TaskGraph:
    return TaskGraph(
        graph_id="graph-001",
        brief_id="brief-001",
        run_id="run-001",
        project_id=PROJECT_ID,
        domain="novel",
        nodes=nodes,
        created_at=FIXED_TIME,
        provenance={
            "source_kind": "user_request",
            "source_id": "run-001",
            "source_sha256": "a" * 64,
            "observed_at": FIXED_TIME.isoformat(),
        },
    )


def test_graph_rejects_duplicate_node_ids() -> None:
    with pytest.raises(ValueError):
        _graph(_node("a", "t1"), _node("a", "t2"))


def test_graph_rejects_empty_node_list() -> None:
    with pytest.raises(ValueError):
        _graph()


def test_validate_rejects_unknown_and_self_dependencies() -> None:
    graph = _graph(
        _node("a", "t1", depends_on=("missing",)),
        _node("b", "t2", depends_on=("b",)),
    )
    failures = validate_task_graph(graph)
    assert "a: unknown dependency missing" in failures
    assert "b: cannot depend on itself" in failures


def test_validate_rejects_nodes_without_skills_or_tools() -> None:
    bare = TaskGraphNode(
        node_id="bare",
        task_type="t",
        output_artifact_kind="artifact",
        output_contract="Contract",
        role=AgentRole.CHAPTER_EXECUTOR,
    )
    failures = validate_task_graph(_graph(bare))
    assert "bare: no candidate skills" in failures
    assert "bare: no allowed tools" in failures


def test_validate_rejects_cycles() -> None:
    graph = _graph(
        _node("a", "t1", depends_on=("b",)),
        _node("b", "t2", depends_on=("a",)),
    )
    assert "task graph contains a dependency cycle" in validate_task_graph(graph)
    assert topological_order(graph) is None


def test_topological_order_respects_dependencies() -> None:
    graph = _graph(
        _node("world", "PLAN_WORLD"),
        _node("cast", "PLAN_CAST", depends_on=("world",)),
        _node("outline", "PLAN_OUTLINE", depends_on=("world", "cast")),
        _node("chapter_1", "DRAFT_CHAPTER", depends_on=("outline",)),
        _node("chapter_2", "DRAFT_CHAPTER", depends_on=("chapter_1",)),
    )
    ordered = topological_order(graph)
    assert ordered is not None
    ids = [node.node_id for node in ordered]
    assert ids[0] == "world"
    assert ids.index("cast") > ids.index("world")
    assert ids.index("outline") > ids.index("cast")
    assert ids.index("chapter_1") > ids.index("outline")
    assert ids.index("chapter_2") > ids.index("chapter_1")
