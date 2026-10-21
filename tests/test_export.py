from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

import pytest

import config
from novel_agent.export import export_project
from novel_agent.pathing import ProjectId, ProjectPathResolver
from novel_agent.service import NovelAgentService, StartNovelRunRequest
from novel_agent.storage import connect_project_database, migrate
from tests.test_workflow_gateway import FIXED_TIME


PROJECT_ID = ProjectId("novel-one")


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    """Export is a best-effort compatibility view written after the run ends."""
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return True
        sleep(0.01)
    return predicate()


def _seed(connection) -> None:
    migrate(connection)
    connection.execute(
        "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES ('r', 'novel-one', 'succeeded', ?, ?)",
        (FIXED_TIME.isoformat(), FIXED_TIME.isoformat()),
    )
    # world_bible artifact
    connection.execute(
        "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
        "schema_version, is_current, created_at) VALUES (?, 1, 'novel-one', 'r', 'world_bible', 'accepted', ?, ?, 1, 1, ?)",
        (
            "r:brief:world",
            json.dumps(
                {"title": "晨光学院", "premise": "漂浮的魔法学院。", "rules": ["魔力来自星辰"], "constraints": ["无科技"]},
                ensure_ascii=False,
            ).encode(),
            "f" * 64,
            FIXED_TIME.isoformat(),
        ),
    )
    # character_set artifact
    cast = {
        "characters": [
            {"character_id": "c1", "name": "林晚", "role": "主角", "goal": "查明真相", "traits": ["冷静", "执着"]},
        ]
    }
    connection.execute(
        "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
        "schema_version, is_current, created_at) VALUES (?, 1, 'novel-one', 'r', 'character_set', 'accepted', ?, ?, 1, 1, ?)",
        ("r:brief:cast", json.dumps(cast, ensure_ascii=False).encode(), "e" * 64, FIXED_TIME.isoformat()),
    )
    # plan_contract artifact
    plan = {
        "chapters": [
            {"chapter_plan_id": "p:1", "chapter_number": 1, "title": "初入学院", "goal": "抵达", "planned_events": ["报到"]},
        ]
    }
    connection.execute(
        "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
        "schema_version, is_current, created_at) VALUES (?, 1, 'novel-one', 'r', 'plan_contract', 'accepted', ?, ?, 1, 1, ?)",
        ("r:brief:plan", json.dumps(plan, ensure_ascii=False).encode(), "d" * 64, FIXED_TIME.isoformat()),
    )
    # chapter artifact
    chapter = {"chapter_draft_id": "d1", "chapter_plan_id": "p:1", "body": "我站在学院门口。", "created_at": FIXED_TIME.isoformat(), "provenance": {"source_kind": "provider_response", "source_id": "s", "source_sha256": "c" * 64, "observed_at": FIXED_TIME.isoformat()}}
    connection.execute(
        "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
        "schema_version, is_current, created_at) VALUES (?, 1, 'novel-one', 'r', 'chapter', 'accepted', ?, ?, 1, 1, ?)",
        ("chapter-abc", json.dumps(chapter, ensure_ascii=False).encode(), "b" * 64, FIXED_TIME.isoformat()),
    )
    # M3 summary memory
    connection.execute(
        "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, subject_id, "
        "predicate, value_json, source_artifact_id, source_artifact_revision, source_span, source_hash, confidence, "
        "created_by, created_at) VALUES ('m1', 'novel-one', 'r', 1, 'm3_confirmed', 'chapter_summary', 'confirmed', "
        "'p:1', 'summary', ?, 'chapter-abc', 1, '0:8', ?, 1.0, 't', ?)",
        ('{"summary": "主角抵达学院。"}', "a" * 64, FIXED_TIME.isoformat()),
    )
    # DIRECT_EVENT memory for the character
    connection.execute(
        "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, subject_id, "
        "predicate, value_json, source_artifact_id, source_artifact_revision, source_span, source_hash, confidence, "
        "created_by, created_at) VALUES ('m2', 'novel-one', 'r', 1, 'm3_confirmed', 'direct_event', 'confirmed', "
        "'林晚', 'direct_event:chapter_1:1', ?, 'chapter-abc', 1, '0:8', ?, 1.0, 't', ?)",
        ('{"event": "抵达学院", "object": "学院", "chapter": 1}', "a" * 64, FIXED_TIME.isoformat()),
    )


def test_export_writes_legacy_format_files(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    with connect_project_database(project_root) as connection:
        _seed(connection)
        report = export_project(connection, PROJECT_ID, ProjectPathResolver(projects_root))

    # world txt
    world_file = project_root / "worlds" / "晨光学院.txt"
    assert str(world_file) in report.files
    world_text = world_file.read_text(encoding="utf-8")
    assert "漂浮的魔法学院" in world_text
    assert "魔力来自星辰" in world_text

    # character json (legacy shape)
    character_file = project_root / "characters" / "林晚.json"
    assert str(character_file) in report.files
    character = json.loads(character_file.read_text(encoding="utf-8"))
    assert set(character) == {"name", "role_introduction", "role_experience", "role_personality", "plot_history"}
    assert character["name"] == "林晚"
    assert character["role_introduction"] == "主角"
    assert character["plot_history"] == [{"初入学院": "抵达学院"}]

    # outline txt
    outline_file = project_root / "novel" / "00_故事大纲.txt"
    assert outline_file.is_file()
    assert "初入学院" in outline_file.read_text(encoding="utf-8")

    # chapter json (legacy shape: passage + summary)
    chapter_file = project_root / "novel" / "1 初入学院.json"
    assert chapter_file.is_file()
    chapter = json.loads(chapter_file.read_text(encoding="utf-8"))
    assert set(chapter) == {"passage", "summary"}
    assert chapter["passage"] == "我站在学院门口。"
    assert chapter["summary"] == "主角抵达学院。"


def test_export_follows_updates(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    resolver = ProjectPathResolver(projects_root)
    with connect_project_database(project_root) as connection:
        _seed(connection)
        export_project(connection, PROJECT_ID, resolver)
        # update the chapter body in SQLite
        connection.execute(
            "UPDATE artifacts SET raw_bytes = ?, sha256 = ? WHERE artifact_id = 'chapter-abc'",
            (
                json.dumps(
                    {
                        "chapter_draft_id": "d1",
                        "chapter_plan_id": "p:1",
                        "body": "修订后的正文。",
                        "created_at": FIXED_TIME.isoformat(),
                        "provenance": {"source_kind": "provider_response", "source_id": "s", "source_sha256": "c" * 64, "observed_at": FIXED_TIME.isoformat()},
                    },
                    ensure_ascii=False,
                ).encode(),
                "b" * 64,
            ),
        )
        export_project(connection, PROJECT_ID, resolver)

    chapter = json.loads((project_root / "novel" / "1 初入学院.json").read_text(encoding="utf-8"))
    assert chapter["passage"] == "修订后的正文。"


def test_auto_export_after_run_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="auto-export",
            idempotency_key="start-auto-export",
            user_requirements="写一本一章的小说",
            provider="offline_fake",
            target_chapter_count=1,
        )
    )
    deadline = monotonic() + 5.0
    from novel_agent.contracts import RunStatus

    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "auto-export")).status not in terminal:
        assert monotonic() < deadline
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED

    # The scheduler syncs legacy-format files asynchronously after the run
    # reaches its terminal state, so wait for the derived view instead of
    # assuming it is already on disk.
    assert _wait_for(lambda: any((project_root / "worlds").glob("*.txt"))), "world export missing after run"
    assert _wait_for(lambda: any((project_root / "characters").glob("*.json"))), "character export missing after run"
    assert _wait_for(lambda: any((project_root / "novel").glob("*.json"))), "chapter export missing after run"


def test_webui_export_endpoint(console) -> None:
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", console.port, timeout=5)
    try:
        connection.request(
            "POST",
            "/api/export",
            body=json.dumps({"project": "novel-one"}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()
    assert response.status == 200
    assert isinstance(data["files"], list)


def test_outline_only_run_produces_plan_without_chapters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A story-outline request finishes with the plan package and no phantom chapter tasks."""
    from novel_agent.contracts import RunStatus

    projects_root = tmp_path / "projects"
    project_root = projects_root / PROJECT_ID.root
    project_root.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME)
    service.start(
        StartNovelRunRequest(
            project_id=PROJECT_ID,
            run_id="outline-only",
            idempotency_key="start-outline-only",
            user_requirements="写一份故事大纲，现代都市题材",
            provider="offline_fake",
        )
    )
    deadline = monotonic() + 5.0
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(PROJECT_ID, "outline-only")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate: {snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED

    with connect_project_database(project_root) as connection:
        kinds = [
            str(row[0])
            for row in connection.execute(
                "SELECT kind FROM artifacts WHERE run_id = ? AND status = 'accepted' ORDER BY kind",
                ("outline-only",),
            )
        ]
        assert kinds == ["character_set", "plan_contract", "world_bible"]
        chapter_tasks = connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE run_id = ? AND task_type = 'chapter'",
            ("outline-only",),
        ).fetchone()
        assert chapter_tasks == (0,)
        plan_memory = connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE run_id = ? AND layer = 'm2_plan'",
            ("outline-only",),
        ).fetchone()
        assert plan_memory[0] >= 1
    # Legacy-format outline is exported asynchronously after the run ends.
    outline_file = project_root / "novel" / "00_故事大纲.txt"
    assert _wait_for(outline_file.is_file), "outline export missing after run"
    assert outline_file.read_text(encoding="utf-8").strip()
