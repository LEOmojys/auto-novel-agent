"""Generic task-graph contract for the plan-and-execute agent core.

A ``TaskGraph`` is the machine-readable plan: an ordered set of nodes with
dependencies, each declaring its output contract, role, candidate skills,
allowed tools, acceptance criteria and budget.  Domain plugins build graphs
from a ``BriefContract``; the generic executor runs them topologically.
Graphs are validated before execution so an invalid plan can never reach a
tool call.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import Field, field_validator

from novel_agent.contracts.common import ContractModel, Provenance
from novel_agent.pathing import ProjectId
from novel_agent.tool_models import AgentRole

_NODE_ID_PATTERN: Final[str] = r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$"


class TaskGraphNode(ContractModel):
    model_config = {"extra": "forbid", "frozen": True}

    node_id: str = Field(pattern=_NODE_ID_PATTERN)
    task_type: str = Field(min_length=1)
    depends_on: tuple[str, ...] = ()
    output_artifact_kind: str = Field(min_length=1)
    output_contract: str = Field(min_length=1)
    role: AgentRole
    skills: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    acceptance: tuple[str, ...] = ()
    budget_tokens: int = Field(default=4_000, gt=0)
    max_repairs: int = Field(default=1, ge=0, le=2)
    metadata: tuple[tuple[str, str], ...] = ()

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(item == "" for item in value):
            raise ValueError("dependency node ids must be non-empty")
        if len(value) != len(frozenset(value)):
            raise ValueError("dependency node ids must be unique")
        return value


class TaskGraph(ContractModel):
    model_config = {"extra": "forbid", "frozen": True}

    graph_id: str = Field(min_length=1)
    brief_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    project_id: ProjectId
    domain: str = Field(min_length=1)
    nodes: tuple[TaskGraphNode, ...]
    created_at: datetime
    provenance: Provenance

    @field_validator("nodes")
    @classmethod
    def validate_nodes(cls, value: tuple[TaskGraphNode, ...]) -> tuple[TaskGraphNode, ...]:
        if not value:
            raise ValueError("task graph must contain at least one node")
        ids = tuple(node.node_id for node in value)
        if len(ids) != len(frozenset(ids)):
            raise ValueError("task graph node ids must be unique")
        return value

    @property
    def nodes_by_id(self) -> dict[str, TaskGraphNode]:
        return {node.node_id: node for node in self.nodes}


def validate_task_graph(graph: TaskGraph) -> tuple[str, ...]:
    """Return every validation failure for a task graph (empty means valid)."""
    failures: list[str] = []
    by_id = graph.nodes_by_id
    for node in graph.nodes:
        for dependency in node.depends_on:
            if dependency not in by_id:
                failures.append(f"{node.node_id}: unknown dependency {dependency}")
            if dependency == node.node_id:
                failures.append(f"{node.node_id}: cannot depend on itself")
        if not node.skills:
            failures.append(f"{node.node_id}: no candidate skills")
        if not node.tools:
            failures.append(f"{node.node_id}: no allowed tools")
    if _topological_order(graph) is None:
        failures.append("task graph contains a dependency cycle")
    return tuple(failures)


def topological_order(graph: TaskGraph) -> tuple[TaskGraphNode, ...] | None:
    """Return nodes in dependency order, or None when the graph has a cycle."""
    return _topological_order(graph)


def _topological_order(graph: TaskGraph) -> tuple[TaskGraphNode, ...] | None:
    by_id = graph.nodes_by_id
    ready = [node for node in graph.nodes if not node.depends_on]
    remaining = {node.node_id: set(node.depends_on) for node in graph.nodes}
    ordered: list[TaskGraphNode] = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for candidate in graph.nodes:
            if candidate.node_id in remaining and node.node_id in remaining[candidate.node_id]:
                remaining[candidate.node_id].discard(node.node_id)
                if not remaining[candidate.node_id]:
                    ready.append(candidate)
    if len(ordered) != len(graph.nodes):
        return None
    return tuple(ordered)


__all__ = (
    "TaskGraph",
    "TaskGraphNode",
    "topological_order",
    "validate_task_graph",
)
