from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from novel_agent.contracts import RunStatus
from novel_agent.run_coordinator import RunCoordinator
from novel_agent.run_event_store import append_run_event
from novel_agent.run_state import RunEvent
from novel_agent.storage import connect_project_database, migrate
from novel_agent.workflow_view import WorkflowView, build_workflow_view


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
RUN_ID = "run-001"


def _run(connection, state: str = RunStatus.RUNNING.value) -> None:
    connection.execute(
        "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, 'novel-one', ?, ?, ?)",
        (RUN_ID, state, FIXED_TIME.isoformat(), FIXED_TIME.isoformat()),
    )


def _event(connection, event_type: str, payload: dict) -> None:
    append_run_event(connection, RunEvent(RUN_ID, event_type, payload, FIXED_TIME))


def _seed_planning_run(connection) -> None:
    _run(connection)
    _event(connection, "brief_analyzed", {"goal": "写魔法学院小说", "constraints": "主角是孤儿", "style": "第一人称", "acceptance": "20章", "chapter_count": 2})
    _event(connection, "task_graph_built", {"nodes": "world,cast,outline,chapter_1,chapter_2"})
    _event(connection, "stage_entered", {"stage": "node_complete:world"})
    _event(connection, "stage_entered", {"stage": "node_complete:cast"})


def test_planning_phase_shows_current_node(tmp_path: Path) -> None:
    project_path = tmp_path / "p1"
    with connect_project_database(project_path) as connection:
        migrate(connection)
        _seed_planning_run(connection)

        view = build_workflow_view(connection, RUN_ID)

    assert view.phase == "planning"
    assert view.current_activity == "规划中：章节大纲"
    assert [node.status for node in view.nodes] == ["done", "done", "running", "pending", "pending"]
    assert view.chain[0].startswith("需求分析：写魔法学院小说")
    assert "任务图已构建" in view.chain[1]
    assert "节点完成：world" in view.chain[2]


def test_executing_phase_tracks_chapter_activity(tmp_path: Path) -> None:
    project_path = tmp_path / "p2"
    with connect_project_database(project_path) as connection:
        migrate(connection)
        _seed_planning_run(connection)
        _event(connection, "plan_validated", {"plan_id": "p-1", "plan_revision": 1})
        _event(connection, "stage_entered", {"stage": "node_complete:outline"})
        _event(connection, "stage_entered", {"stage": "chapter_prepare_complete"})
        _event(connection, "stage_entered", {"stage": "chapter_draft_staged"})

        view = build_workflow_view(connection, RUN_ID)

    assert view.phase == "executing"
    assert view.current_activity == "创作中：第 1 章（正在生成章节草稿…）"
    assert [node.status for node in view.nodes] == ["done", "done", "done", "running", "pending"]


def test_finished_phase_and_quality_chain(tmp_path: Path) -> None:
    project_path = tmp_path / "p3"
    with connect_project_database(project_path) as connection:
        migrate(connection)
        _run(connection, RunStatus.SUCCEEDED.value)
        _event(connection, "task_graph_built", {"nodes": "world,cast,outline,chapter_1"})
        for node in ("world", "cast", "outline", "chapter_1"):
            _event(connection, "stage_entered", {"stage": f"node_complete:{node}"})
        _event(connection, "chapter_quality_recorded", {"decision": "ACCEPT", "task": "t"})
        connection.execute(
            "INSERT INTO tasks(task_id, run_id, task_type, state, idempotency_key, created_at, updated_at) "
            "VALUES ('t', ?, 'chapter', 'succeeded', 't', ?, ?)",
            (RUN_ID, FIXED_TIME.isoformat(), FIXED_TIME.isoformat()),
        )
        connection.execute(
            "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
            "schema_version, is_current, created_at) "
            "VALUES ('a', 1, 'novel-one', ?, 'chapter', 'accepted', ?, ?, 1, 1, ?)",
            (RUN_ID, b"body", "f" * 64, FIXED_TIME.isoformat()),
        )
        connection.execute(
            "INSERT INTO quality_reports(report_id, run_id, task_id, artifact_id, artifact_revision, decision, "
            "report_json, created_at) VALUES ('q-1', ?, 't', 'a', 1, 'ACCEPT', ?, ?)",
            (
                RUN_ID,
                json.dumps({"scores": {"风格": 4, "角色": 5}}, ensure_ascii=False),
                FIXED_TIME.isoformat(),
            ),
        )

        view: WorkflowView = build_workflow_view(connection, RUN_ID)

    assert view.phase == "finished"
    assert view.current_activity == "全部任务完成"
    assert all(node.status == "done" for node in view.nodes)
    assert any(line.startswith("评审（") and "ACCEPT" in line for line in view.chain)


def test_review_phase_reports_reason(tmp_path: Path) -> None:
    project_path = tmp_path / "p4"
    with connect_project_database(project_path) as connection:
        migrate(connection)
        _run(connection, RunStatus.WAITING_REVIEW.value)
        _event(connection, "task_graph_built", {"nodes": "world,cast,outline,chapter_1"})
        for node in ("world", "cast", "outline"):
            _event(connection, "stage_entered", {"stage": f"node_complete:{node}"})
        _event(connection, "chapter_waiting_review", {"reason": "chapter_schema_failure", "task": "t"})

        view = build_workflow_view(connection, RUN_ID)

    assert view.phase == "review"
    assert "chapter_schema_failure" in view.current_activity
    assert [node.status for node in view.nodes] == ["done", "done", "done", "running"]


def test_brief_phase_before_graph(tmp_path: Path) -> None:
    project_path = tmp_path / "p5"
    with connect_project_database(project_path) as connection:
        migrate(connection)
        _run(connection)

        view = build_workflow_view(connection, RUN_ID)

    assert view.phase == "brief"
    assert view.nodes == ()
