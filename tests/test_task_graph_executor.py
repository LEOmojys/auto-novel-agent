from __future__ import annotations

from datetime import UTC, datetime

from novel_agent.agent.brief import BriefContract
from novel_agent.agent.domain_plugin import (
    DomainPlugin,
    NodeExecutionContext,
    NodeExecutionResult,
    NodeExecutor,
    NodeOutput,
)
from novel_agent.agent.task_graph import TaskGraph, TaskGraphNode, topological_order
from novel_agent.agent.task_graph_executor import TaskGraphExecutor
from novel_agent.contracts import Provenance
from novel_agent.pathing import ProjectId
from novel_agent.run_models import RunLease
from novel_agent.tool_models import AgentRole


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")


class _RecordingNode(NodeExecutor):
    def __init__(self, task_type: str, fail: bool = False) -> None:
        self.task_type = task_type
        self.fail = fail
        self.calls: list[str] = []

    def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        self.calls.append(context.node.node_id)
        if self.fail:
            return NodeExecutionResult(status="blocked", reason="fake_failure")
        value = f"output:{context.node.node_id}"
        return NodeExecutionResult(
            status="succeeded",
            output=NodeOutput(node_id=context.node.node_id, value=value, committed=False),
        )


class FakePlugin:
    domain = "fake"

    def __init__(self, fail_node: str | None = None) -> None:
        self.fail_node = fail_node
        self.executors: dict[str, _RecordingNode] = {}

    def build_graph(self, brief, *, run_id, project_id, now) -> TaskGraph:
        nodes = (
            TaskGraphNode(
                node_id="a",
                task_type="step_a",
                output_artifact_kind="a",
                output_contract="A",
                role=AgentRole.CHAPTER_EXECUTOR,
                skills=("skill-a",),
                tools=("tool-a",),
            ),
            TaskGraphNode(
                node_id="b",
                task_type="step_b",
                depends_on=("a",),
                output_artifact_kind="b",
                output_contract="B",
                role=AgentRole.CHAPTER_EXECUTOR,
                skills=("skill-b",),
                tools=("tool-b",),
            ),
        )
        return TaskGraph(
            graph_id="graph-fake",
            brief_id=brief.brief_id,
            run_id=run_id,
            project_id=project_id,
            domain="fake",
            nodes=nodes,
            created_at=now(),
            provenance=Provenance(
                source_kind="system",
                source_id=run_id,
                source_sha256="b" * 64,
                observed_at=now(),
            ),
        )

    def node_executor(self, node: TaskGraphNode) -> NodeExecutor | None:
        if node.node_id not in self.executors:
            self.executors[node.node_id] = _RecordingNode(node.task_type, fail=node.node_id == self.fail_node)
        return self.executors[node.node_id]


def _brief() -> BriefContract:
    return BriefContract(
        brief_id="brief-001",
        project_id=PROJECT_ID,
        raw_requirements="需求",
        goal="目标",
        created_at=FIXED_TIME,
        provenance=Provenance(
            source_kind="user_request",
            source_id="run-001",
            source_sha256="c" * 64,
            observed_at=FIXED_TIME,
        ),
    )


def _lease() -> RunLease:
    return RunLease(run_id="run-001", holder_id="h", fencing_token=1, expires_at=FIXED_TIME)


def test_executor_runs_nodes_in_topological_order() -> None:
    plugin = FakePlugin()
    graph = plugin.build_graph(_brief(), run_id="run-001", project_id=PROJECT_ID, now=lambda: FIXED_TIME)
    executor = TaskGraphExecutor(
        plugin=plugin,
        coordinator=object(),
        path_resolver=object(),  # type: ignore[arg-type]
        now=lambda: FIXED_TIME,
        catalog=object(),  # type: ignore[arg-type]
    )

    result = executor.execute(graph, _brief(), _lease())

    assert result.status == "succeeded"
    assert [node.node_id for node in topological_order(graph)] == ["a", "b"]
    assert plugin.executors["a"].calls == ["a"]
    assert plugin.executors["b"].calls == ["b"]
    assert len(result.outputs) == 2


def test_executor_stops_at_blocked_node_and_preserves_dependencies() -> None:
    plugin = FakePlugin(fail_node="a")
    graph = plugin.build_graph(_brief(), run_id="run-001", project_id=PROJECT_ID, now=lambda: FIXED_TIME)
    executor = TaskGraphExecutor(
        plugin=plugin,
        coordinator=object(),
        path_resolver=object(),  # type: ignore[arg-type]
        now=lambda: FIXED_TIME,
        catalog=object(),  # type: ignore[arg-type]
    )

    result = executor.execute(graph, _brief(), _lease())

    assert result.status == "blocked"
    assert result.reason == "fake_failure"
    # The downstream node was never resolved or executed after the block.
    assert plugin.executors.get("b") is None


def test_executor_rejects_invalid_graph() -> None:
    plugin = FakePlugin()
    graph = plugin.build_graph(_brief(), run_id="run-001", project_id=PROJECT_ID, now=lambda: FIXED_TIME)
    graph = graph.model_copy(update={"nodes": (graph.nodes[0],)})
    graph = graph.model_copy(
        update={
            "nodes": (
                graph.nodes[0].model_copy(update={"depends_on": ("missing",)}),
            )
        }
    )
    executor = TaskGraphExecutor(
        plugin=plugin,
        coordinator=object(),
        path_resolver=object(),  # type: ignore[arg-type]
        now=lambda: FIXED_TIME,
        catalog=object(),  # type: ignore[arg-type]
    )

    result = executor.execute(graph, _brief(), _lease())

    assert result.status == "blocked"
    assert result.reason == "task_graph_invalid"


def test_executor_emits_stage_events_per_node() -> None:
    plugin = FakePlugin()
    graph = plugin.build_graph(_brief(), run_id="run-001", project_id=PROJECT_ID, now=lambda: FIXED_TIME)
    events: list[str] = []
    executor = TaskGraphExecutor(
        plugin=plugin,
        coordinator=object(),
        path_resolver=object(),  # type: ignore[arg-type]
        now=lambda: FIXED_TIME,
        catalog=object(),  # type: ignore[arg-type]
        stage_event_sink=events.append,
    )

    executor.execute(graph, _brief(), _lease())

    assert "node_complete:a" in events
    assert "node_complete:b" in events
