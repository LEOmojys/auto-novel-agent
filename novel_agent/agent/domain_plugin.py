"""Domain plugin protocol: how a domain contributes to the generic agent core.

A domain plugin owns the mapping from a ``BriefContract`` to a validated
``TaskGraph`` and provides a node executor for each task type it declares.
The generic ``TaskGraphExecutor`` only orchestrates: it resolves ready nodes,
invokes the plugin's executor with a rich execution context, and routes the
outcome.  All governed side effects (tool calls through the gateway, memory
promotion, coordinator commits) stay inside the domain node executors, which
reuse the existing planning/chapter machinery.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from novel_agent.agent.brief import BriefContract
from novel_agent.agent.task_graph import TaskGraph, TaskGraphNode
from novel_agent.pathing import ProjectId, ProjectPathResolver
from novel_agent.run_models import RunLease
from novel_agent.skills import SkillCatalog
from novel_agent.tool_registry import ToolGateway, ToolRegistry


class NodeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    value: object | None = None
    committed: bool = False


class NodeExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["succeeded", "blocked"]
    output: NodeOutput | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class NodeExecutionContext:
    graph: TaskGraph
    node: TaskGraphNode
    brief: BriefContract
    lease: RunLease
    outputs: Mapping[str, NodeOutput]
    coordinator: object
    gateway: ToolGateway | None
    catalog: SkillCatalog
    path_resolver: ProjectPathResolver
    now: Callable[[], datetime]
    run_id: str
    project_id: ProjectId
    stage_event_sink: Callable[[str], None] | None = None


class NodeExecutor(Protocol):
    def execute(self, context: NodeExecutionContext) -> NodeExecutionResult: ...


class DomainPlugin(Protocol):
    """A domain's contribution to the generic plan-and-execute core."""

    domain: str

    def build_graph(
        self,
        brief: BriefContract,
        *,
        run_id: str,
        project_id: ProjectId,
        now: Callable[[], datetime],
    ) -> TaskGraph: ...

    def node_executor(self, node: TaskGraphNode) -> NodeExecutor | None: ...

    def register_tools(self, registry: ToolRegistry) -> None:
        """Optional: register domain tools (called once per stage)."""
        return None


__all__ = (
    "DomainPlugin",
    "NodeExecutionContext",
    "NodeExecutionResult",
    "NodeExecutor",
    "NodeOutput",
)
