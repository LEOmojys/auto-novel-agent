"""Asset-reuse robustness: placeholder outlines must not shadow the real one.

Regression for the case where an offline/test run left a 1-chapter
``plan_contract`` with ``is_current=1`` while the canonical multi-chapter
outline also exists — the newest-created placeholder used to win, causing
``chapter_plan_missing`` on continuation.  ``_merge_sqlite_outline`` now
prefers the is_current outline with the most chapters.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from novel_agent.pathing import ProjectId
from novel_agent.storage import connect_project_database, migrate


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")


def _plan_payload(run_id: str, n_chapters: int) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "plan_id": f"{run_id}:brief:plan",
            "brief_id": f"{run_id}:brief",
            "chapters": [
                {
                    "chapter_plan_id": f"{run_id}:brief:chapter:{n}",
                    "chapter_number": n,
                    "title": f"第{n}章",
                    "goal": "目标",
                    "planned_events": [],
                }
                for n in range(1, n_chapters + 1)
            ],
        },
        ensure_ascii=False,
    ).encode()


def _insert_plan(connection, run_id: str, n_chapters: int, created_at: datetime, is_current: int) -> None:
    raw = _plan_payload(run_id, n_chapters)
    connection.execute(
        "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, 'succeeded', ?, ?) "
        "ON CONFLICT(run_id) DO NOTHING",
        (run_id, PROJECT_ID.root, created_at.isoformat(), created_at.isoformat()),
    )
    connection.execute(
        "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
        "schema_version, is_current, created_at) VALUES (?, 1, ?, ?, 'plan_contract', 'accepted', ?, ?, 1, ?, ?) "
        "ON CONFLICT(artifact_id, revision) DO NOTHING",
        (f"{run_id}:brief:plan", PROJECT_ID.root, run_id, raw, __import__("hashlib").sha256(raw).hexdigest(), is_current, created_at.isoformat()),
    )


def test_placeholder_plan_does_not_shadow_real_outline(tmp_path: Path) -> None:
    from novel_agent.domains.novel import _merge_sqlite_outline

    db = connect_project_database(tmp_path / "projects" / PROJECT_ID.root)
    connection = db.__enter__()
    try:
        migrate(connection)
        # Real 10-chapter outline (older), then a placeholder 1-chapter plan (newer).
        _insert_plan(connection, "real-run", 10, datetime(2030, 1, 1, tzinfo=UTC), 1)
        _insert_plan(connection, "placeholder-run", 1, datetime(2030, 1, 2, tzinfo=UTC), 1)

        assets: dict[str, object] = {}
        _merge_sqlite_outline(assets, connection)

        outline = assets.get("outline")
        assert outline is not None, "expected an outline to be selected"
        assert outline["chapter_count"] == 10, f"placeholder 1-chapter plan shadowed the real one: {outline['chapter_count']}"
        assert "placeholder" not in outline.get("chapters_text", "")
    finally:
        db.__exit__(None, None, None)


def test_placeholder_plan_selected_when_no_real_outline(tmp_path: Path) -> None:
    """Without a larger real outline, the is_current placeholder is still usable."""
    from novel_agent.domains.novel import _merge_sqlite_outline

    db = connect_project_database(tmp_path / "projects" / PROJECT_ID.root)
    connection = db.__enter__()
    try:
        migrate(connection)
        _insert_plan(connection, "only-run", 1, FIXED_TIME, 1)

        assets: dict[str, object] = {}
        _merge_sqlite_outline(assets, connection)

        outline = assets.get("outline")
        assert outline is not None
        assert outline["chapter_count"] == 1
    finally:
        db.__exit__(None, None, None)
