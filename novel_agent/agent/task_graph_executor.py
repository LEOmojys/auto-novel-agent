"""Generic task-graph executor: the plan-and-execute engine.

Runs a validated ``TaskGraph`` in topological order.  For every ready node it
builds a ``NodeExecutionContext`` (including the outputs of completed
dependencies), delegates to the domain plugin's node executor, and routes the
result.  Blocked nodes stop the graph; the domain executor is responsible for
persisting the run state transition (block/fail/pause) through the
coordinator, exactly as the current planning/chapter workflows do.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from novel_agent.agent.brief import BriefContract
from novel_agent.agent.domain_plugin import (
    DomainPlugin,
    NodeExecutionContext,
    NodeExecutionResult,
    NodeOutput,
)
from novel_agent.agent.task_graph import (
    TaskGraph,
    TaskGraphNode,
    topological_order,
    validate_task_graph,
)
from novel_agent.pathing import ProjectId, ProjectPathResolver
from novel_agent.run_models import RunLease
from novel_agent.skills import SkillCatalog
from novel_agent.tool_registry import ToolGateway


@dataclass(frozen=True, slots=True)
class GraphExecutionResult:
    status: Literal["succeeded", "blocked"]
    reason: str | None = None
    outputs: tuple[NodeOutput, ...] = ()


@dataclass(slots=True)
class TaskGraphExecutor:
    """Runs a domain's task graph through its node executors."""

    plugin: DomainPlugin
    coordinator: object
    path_resolver: ProjectPathResolver
    now: Callable[[], datetime]
    catalog: SkillCatalog
    gateway: ToolGateway | None = None
    stage_event_sink: Callable[[str], None] | None = None
    _completed: dict[str, NodeOutput] = field(default_factory=dict)

    def execute(
        self,
        graph: TaskGraph,
        brief: BriefContract,
        lease: RunLease,
    ) -> GraphExecutionResult:
        failures = validate_task_graph(graph)
        if failures:
            return GraphExecutionResult(status="blocked", reason="task_graph_invalid")
        ordered = topological_order(graph)
        if ordered is None:
            return GraphExecutionResult(status="blocked", reason="task_graph_cycle")
        for node in ordered:
            executor = self.plugin.node_executor(node)
            if executor is None:
                return GraphExecutionResult(
                    status="blocked",
                    reason=f"node_executor_missing:{node.task_type}",
                )
            context = NodeExecutionContext(
                graph=graph,
                node=node,
                brief=brief,
                lease=lease,
                outputs=dict(self._completed),
                coordinator=self.coordinator,
                gateway=self.gateway,
                catalog=self.catalog,
                path_resolver=self.path_resolver,
                now=self.now,
                run_id=graph.run_id,
                project_id=graph.project_id,
                stage_event_sink=self.stage_event_sink,
            )
            result = executor.execute(context)
            if result.status == "blocked":
                return GraphExecutionResult(
                    status="blocked",
                    reason=result.reason,
                    outputs=tuple(self._completed.values()),
                )
            if result.output is not None:
                self._completed[node.node_id] = result.output
            if self.stage_event_sink is not None:
                self.stage_event_sink(f"node_complete:{node.node_id}")
        return GraphExecutionResult(
            status="succeeded",
            outputs=tuple(self._completed.values()),
        )


__all__ = ("GraphExecutionResult", "TaskGraphExecutor")
