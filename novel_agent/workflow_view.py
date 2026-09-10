"""Workflow view: project the event ledger into a human-readable agent trace.

The UI shows the agent's work as a task-graph progress board plus an
auditable reasoning trace ("chain of thought" equivalent).  This module is a
pure read-side projection over ``run_events``, ``tool_calls`` and
``quality_reports``; it never mutates state.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from novel_agent.contracts import RunStatus

_CHAPTER_ACTIVITY: dict[str, str] = {
    "chapter_prepare_complete": "正在构建章节上下文…",
    "chapter_draft_staged": "正在生成章节草稿…",
    "chapter_repair_staged": "正在按评审意见修订章节…",
    "chapter_continuity_checked": "正在做连续性检查…",
    "chapter_quality_reviewed": "质量评审完成，等待判定…",
    "chapter_committed": "章节已提交",
}

_INTENT_LABELS: dict[str, str] = {
    "full_novel": "完整小说创作",
    "characters_only": "角色设定",
    "world_only": "世界观设定",
    "outline_only": "大纲规划",
    "add_character": "新增角色",
}

_DEGRADE_LABELS: dict[str, str] = {
    "provider_failure": "模型调用失败",
    "parse_failure": "模型返回无法解析（已按宽松解析重试仍失败）",
}

_PLAN_NODE_ACTIVITY: dict[str, str] = {
    "world": "规划中：世界观生成",
    "cast": "规划中：角色设定",
    "outline": "规划中：章节大纲",
}

_LOOP_TOOL_LABELS: dict[str, str] = {
    "write_world": "写世界观",
    "write_cast": "写角色",
    "write_outline": "写大纲",
    "write_chapter": "写章节",
    "review_chapter": "评审章节",
    "mem_query": "查询设定记忆",
}


@dataclass(frozen=True, slots=True)
class WorkflowNodeView:
    node_id: str
    status: str  # done | running | pending | blocked


@dataclass(frozen=True, slots=True)
class WorkflowView:
    phase: str  # brief | planning | executing | review | finished | failed | cancelled
    current_activity: str
    nodes: tuple[WorkflowNodeView, ...]
    chain: tuple[str, ...]


def build_workflow_view(connection: sqlite3.Connection, run_id: str) -> WorkflowView:
    events = _events(connection, run_id)
    run_state = _run_state(connection, run_id)

    node_ids = _node_ids(events)
    done = {event["payload"].get("stage", "").removeprefix("node_complete:") for event in events}
    done = {node_id for node_id in done if node_id in set(node_ids)}
    node_views = _node_views(node_ids, done, run_state)

    phase, activity = _phase_and_activity(run_state, events, node_ids, done)
    chain = _chain(events, run_state, connection, run_id)
    return WorkflowView(
        phase=phase,
        current_activity=activity,
        nodes=node_views,
        chain=chain,
    )


def _node_views(
    node_ids: list[str],
    done: set[str],
    run_state: str | None,
) -> tuple[WorkflowNodeView, ...]:
    """Node badges reflect the run terminal state.

    A failed/cancelled run never emits node_complete for the node that was
    in flight; without this, the task list keeps showing it as "running"
    forever after the run ended.
    """
    if not node_ids:
        return ()
    terminal_failed = run_state in (RunStatus.FAILED.value, RunStatus.CANCELLED.value)
    terminal_ok = run_state == RunStatus.SUCCEEDED.value
    views: list[WorkflowNodeView] = []
    running_seen = False
    for node_id in node_ids:
        if node_id in done or terminal_ok:
            views.append(WorkflowNodeView(node_id=node_id, status="done"))
        elif terminal_failed and not running_seen:
            views.append(WorkflowNodeView(node_id=node_id, status="blocked"))
            running_seen = True
        elif not running_seen:
            views.append(WorkflowNodeView(node_id=node_id, status="running"))
            running_seen = True
        else:
            views.append(WorkflowNodeView(node_id=node_id, status="pending"))
    return tuple(views)


def _phase_and_activity(
    run_state: str | None,
    events: list[dict],
    node_ids: list[str],
    done: set[str],
) -> tuple[str, str]:
    if run_state == RunStatus.WAITING_REVIEW.value:
        reason = _latest_payload_value(
            events,
            {"chapter_waiting_review", "plan_blocked", "artifact_approval_required"},
            "reason",
        ) or "待人工审查"
        if reason == "artifact_approval_required":
            return "review", "等待确认产物入库"
        return "review", f"等待人工审查：{reason}"
    if run_state == RunStatus.FAILED.value:
        reason = _latest_payload_value(events, {"run_failed"}, "reason") or "未知原因"
        return "failed", f"运行失败：{reason}"
    if run_state == RunStatus.CANCELLED.value:
        return "cancelled", "运行已取消"
    if run_state == RunStatus.SUCCEEDED.value:
        return "finished", "全部任务完成"
    if not node_ids:
        # Before the task graph is built the run is still in the planning
        # phase (brief analysis / model plan generation); only show
        # "需求分析中" until the brief event exists.
        has_brief = any(event["type"] == "brief_analyzed" for event in events)
        # Agent-loop runs never build a task graph; show its latest activity.
        latest_loop = next(
            (event for event in reversed(events) if event["type"] in ("tool_call", "loop_done", "loop_step")),
            None,
        )
        if latest_loop is not None and has_brief:
            if latest_loop["type"] == "loop_done":
                return "finished", "Agent 已完成创作"
            tool = latest_loop["payload"].get("tool", "")
            label = _LOOP_TOOL_LABELS.get(str(tool), str(tool))
            return "executing", f"Agent 创作中：{label}"
        if has_brief:
            return "planning", "规划中（模型正在生成执行计划）…"
        return "brief", "需求分析中…"

    running = next((node_id for node_id in node_ids if node_id not in done), None)
    if running is None:
        return "finished", "全部任务完成"
    if running == "world" or running == "cast" or running == "outline":
        return "planning", _PLAN_NODE_ACTIVITY[running]
    if running.startswith("chapter_"):
        number = running.removeprefix("chapter_")
        detail = _chapter_detail(events, number)
        return "executing", f"创作中：第 {number} 章{detail}"
    return "executing", f"执行中：{running}"


def _chapter_detail(events: list[dict], number: str) -> str:
    """Latest in-chapter activity before the chapter's node completes."""
    target = f"node_complete:chapter_{number}"
    latest: str | None = None
    for event in events:
        if event["type"] == "stage_entered":
            stage = event["payload"].get("stage", "")
            if stage == target:
                latest = None
            elif stage in _CHAPTER_ACTIVITY:
                latest = stage
    if latest is None:
        return ""
    return f"（{_CHAPTER_ACTIVITY[latest]}）"


def _chain(events: list[dict], run_state: str | None, connection: sqlite3.Connection, run_id: str) -> tuple[str, ...]:
    lines: list[str] = []
    for event in events:
        payload = event["payload"]
        match event["type"]:
            case "brief_analyzed":
                goal = payload.get("goal", "")
                intent = payload.get("intent", "")
                degraded = payload.get("degraded", False)
                constraints = payload.get("constraints", "")
                style = payload.get("style", "")
                acceptance = payload.get("acceptance", "")
                count = payload.get("chapter_count", 0)
                intent_label = _INTENT_LABELS.get(str(intent), str(intent))
                parts = [f"需求分析：{goal}", f"意图：{intent_label}"]
                if degraded:
                    reason = str(payload.get("degraded_reason", "") or "未知原因")
                    label = _DEGRADE_LABELS.get(reason, f"模型调用失败（{reason}）")
                    detail = str(payload.get("degraded_detail", "") or "")
                    if detail:
                        label = f"{label}：{detail}"
                    parts.append(f"（分析降级：{label}，按规则兜底处理）")
                if constraints:
                    parts.append(f"约束：{constraints}")
                if style:
                    parts.append(f"风格：{style}")
                if acceptance:
                    parts.append(f"验收：{acceptance}")
                parts.append(f"目标章数：{count}")
                lines.append(" | ".join(parts))
            case "plan_generated":
                steps = str(payload.get("steps", ""))
                if payload.get("degraded"):
                    lines.append("执行计划：模型规划失败，已按规则模板兜底")
                elif steps:
                    lines.append(f"执行计划（LLM 生成）：{steps}")
                else:
                    lines.append("执行计划：已生成")
            case "task_graph_built":
                lines.append(f"任务图已构建：{payload.get('nodes', '')}")
            case "loop_step":
                step = payload.get("step", "")
                lines.append(f"◆ Agent 决策（步 {step}）")
            case "tool_call":
                tool = payload.get("tool", "")
                ok = payload.get("ok", True)
                mark = "✓" if ok else "✗"
                label = _LOOP_TOOL_LABELS.get(str(tool), str(tool))
                lines.append(f"{mark} 调用工具：{label}")
            case "loop_done":
                reply = payload.get("reply", "")
                lines.append(f"✓ Agent 完成{('：' + str(reply)) if reply else ''}")
            case "loop_compacted":
                kept = payload.get("kept", 0)
                lines.append(f"↻ 上下文已压缩（保留最近 {kept} 条）")
            case "stage_entered":
                stage = payload.get("stage", "")
                if stage.startswith("node_complete:"):
                    lines.append(f"✓ 节点完成：{stage.removeprefix('node_complete:')}")
                elif stage in _CHAPTER_ACTIVITY:
                    lines.append(_CHAPTER_ACTIVITY[stage])
            case "plan_validated":
                lines.append(f"✓ 计划验证通过（{payload.get('plan_id', '')} v{payload.get('plan_revision', '')}）")
            case "chapter_accepted":
                lines.append(f"✓ 第 {payload.get('chapter', '')} 章已接受并提交")
            case "chapter_staged":
                lines.append(f"第 {payload.get('chapter', '')} 章草稿已暂存（尝试 {payload.get('attempt', 1)}）")
            case "chapter_quality_recorded":
                decision = payload.get("decision", "")
                if decision == "ACCEPT":
                    lines.append("质量评审：通过（ACCEPT）")
                elif decision == "REPAIR":
                    lines.append("质量评审：需修订（REPAIR）")
                else:
                    lines.append(f"质量评审：{decision}")
            case "plan_blocked":
                lines.append(f"⛔ 规划受阻：{payload.get('reason', '')}")
            case "run_failed":
                lines.append(f"⛔ 运行失败：{payload.get('reason', '')}")
            case "run_cancelled" | "run_cancelled_requested":
                lines.append("运行取消")
    lines.extend(_tool_chain(connection, run_id))
    lines.extend(_quality_chain(connection, run_id))
    return tuple(lines)


def _tool_chain(connection: sqlite3.Connection, run_id: str) -> tuple[str, ...]:
    rows = connection.execute(
        "SELECT tool_id, state FROM tool_calls WHERE run_id = ? ORDER BY created_at, call_id",
        (run_id,),
    ).fetchall()
    return tuple(f"工具调用：{row[0]} → {row[1]}" for row in rows)


def _quality_chain(connection: sqlite3.Connection, run_id: str) -> tuple[str, ...]:
    rows = connection.execute(
        "SELECT task_id, decision, report_json FROM quality_reports WHERE run_id = ? "
        "ORDER BY created_at, report_id",
        (run_id,),
    ).fetchall()
    lines: list[str] = []
    for task_id, decision, report_json in rows:
        summary = _quality_summary(str(report_json))
        lines.append(f"评审（{str(task_id)}）：{summary} → {decision}")
    return tuple(lines)


def _quality_summary(report_json: str) -> str:
    try:
        data = json.loads(report_json)
    except (json.JSONDecodeError, TypeError):
        return "报告不可解析"
    scores = data.get("scores") or data.get("dimensions") or {}
    if isinstance(scores, dict):
        parts = [f"{key}={value}" for key, value in scores.items()]
        return " ".join(parts) if parts else "无评分"
    return "无评分"


def _events(connection: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = connection.execute(
        "SELECT event_type, payload_json FROM run_events WHERE run_id = ? ORDER BY sequence",
        (run_id,),
    ).fetchall()
    events: list[dict] = []
    for event_type, payload_json in rows:
        try:
            payload = json.loads(str(payload_json))
        except json.JSONDecodeError:
            payload = {}
        events.append({"type": str(event_type), "payload": payload})
    return events


def _run_state(connection: sqlite3.Connection, run_id: str) -> str | None:
    row = connection.execute("SELECT state FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return None if row is None else str(row[0])


def _node_ids(events: list[dict]) -> list[str]:
    for event in events:
        if event["type"] == "task_graph_built":
            raw = event["payload"].get("nodes", "")
            return [node_id for node_id in raw.split(",") if node_id]
    return []


def _latest_payload_value(events: list[dict], event_types: set[str], key: str) -> str | None:
    for event in reversed(events):
        if event["type"] in event_types:
            value = event["payload"].get(key)
            if value is not None:
                return str(value)
    return None


__all__ = ("WorkflowNodeView", "WorkflowView", "build_workflow_view")
