from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from novel_agent.memory_store import rebuild_retrieval_index
from novel_agent.storage import CURRENT_SCHEMA_VERSION, connect_project_database, migrate


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


def _seed_memory(connection) -> tuple[str, str]:
    connection.execute(
        "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, 'succeeded', ?, ?)",
        ("run-001", "novel-one", FIXED_TIME.isoformat(), FIXED_TIME.isoformat()),
    )
    connection.execute(
        "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, sha256, "
        "schema_version, is_current, created_at) "
        "VALUES ('artifact-1', 1, 'novel-one', 'run-001', 'chapter', 'accepted', ?, ?, 1, 1, ?)",
        (b"body bytes", "f" * 64, FIXED_TIME.isoformat()),
    )
    connection.execute(
        "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, layer, kind, status, "
        "subject_id, predicate, value_json, source_artifact_id, source_artifact_revision, source_span, "
        "source_hash, confidence, created_by, created_at) "
        "VALUES ('m-001', 'novel-one', 'run-001', 1, 'm3_confirmed', 'chapter_summary', 'confirmed', "
        "'chapter-1', 'summary', ?, 'artifact-1', 1, '0:8', ?, 1.0, 'test', ?)",
        (
            '{"summary": "主角林晚抵达魔法学院完成报到。"}',
            "f" * 64,
            FIXED_TIME.isoformat(),
        ),
    )
    return "m-001", "魔法学院"


def test_migration_v5_creates_fts_and_conflict_tables(tmp_path: Path) -> None:
    # Given: a fresh project database.
    project_path = tmp_path / "projects" / "novel-one"
    project_path.mkdir(parents=True)

    # When: the database migrates.
    with connect_project_database(project_path) as connection:
        migrate(connection)

        # Then: v5 tables exist and the schema version is current.
        assert connection.execute(
            "SELECT current_version FROM schema_state WHERE singleton = 1"
        ).fetchone() == (CURRENT_SCHEMA_VERSION,)
        fts = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'memory_fts'"
        ).fetchone()
        conflicts = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'memory_conflicts'"
        ).fetchone()
        assert fts is not None
        assert conflicts is not None


def test_retrieval_index_is_rebuildable_from_memory_records(tmp_path: Path) -> None:
    # Given: a migrated database whose FTS index is intentionally dropped.
    project_path = tmp_path / "projects" / "novel-one"
    project_path.mkdir(parents=True)
    with connect_project_database(project_path) as connection:
        migrate(connection)
        memory_id, term = _seed_memory(connection)
        connection.execute("DELETE FROM memory_fts")

        # When: the derived index is rebuilt from canonical memory records.
        rebuilt = rebuild_retrieval_index(connection, "novel-one")

        # Then: the index is fully restored without touching story truth.
        assert rebuilt == 1
        match = connection.execute(
            "SELECT memory_id FROM memory_fts WHERE memory_fts MATCH ?",
            (f'"{term}"',),
        ).fetchone()
        assert match is not None
        assert str(match[0]) == memory_id
