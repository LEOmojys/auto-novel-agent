from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from novel_agent.storage import (
    CURRENT_SCHEMA_VERSION,
    MIGRATIONS,
    ArtifactRevisionWrite,
    Migration,
    commit_artifact_revision,
    connect_project_database,
    migrate,
    write_transaction,
)


class ForcedTransactionFailure(RuntimeError):
    pass


def test_migrate_and_commit_artifact_revision(tmp_path: Path) -> None:
    # Given: a fresh project directory and arbitrary non-text artifact bytes.
    project_dir = tmp_path / "data" / "projects" / "novel-one"
    raw_bytes = b"chapter:\x00\xff\r\nexact bytes"
    revision = ArtifactRevisionWrite(
        artifact_id="chapter-1",
        revision=1,
        project_id="novel-one",
        run_id=None,
        kind="chapter",
        status="accepted",
        raw_bytes=raw_bytes,
        schema_version=1,
        created_at="2030-01-02T03:04:05+00:00",
    )

    # When: the store is opened, migrated, and an artifact revision is committed.
    with connect_project_database(project_dir) as connection:
        applied = migrate(connection)
        receipt = commit_artifact_revision(connection, revision)
        row = connection.execute(
            "SELECT raw_bytes, sha256, is_current FROM artifacts "
            "WHERE artifact_id = ? AND revision = ?",
            (revision.artifact_id, revision.revision),
        ).fetchone()

    # Then: schema state and the exact governed bytes are authoritative in SQLite.
    assert applied == tuple(migration.version for migration in MIGRATIONS)
    assert receipt.sha256 == hashlib.sha256(raw_bytes).hexdigest()
    assert row == (raw_bytes, receipt.sha256, 1)
    assert project_dir.joinpath("agent.sqlite").is_file()


def test_required_pragmas_are_enabled(tmp_path: Path) -> None:
    # Given: a new project database.
    project_dir = tmp_path / "project"

    # When: it is opened through the public storage boundary.
    with connect_project_database(project_dir) as connection:
        values = (
            connection.execute("PRAGMA foreign_keys").fetchone()[0],
            connection.execute("PRAGMA journal_mode").fetchone()[0],
            connection.execute("PRAGMA busy_timeout").fetchone()[0],
        )

    # Then: referential integrity, WAL, and bounded lock waiting are active.
    assert values == (1, "wal", 5000)


def test_migration_twice_is_idempotent_and_stale_connection_observes_version(
    tmp_path: Path,
) -> None:
    # Given: two connections to the same fresh project database.
    project_dir = tmp_path / "project"
    with (
        connect_project_database(project_dir) as first,
        connect_project_database(project_dir) as second,
    ):
        # When: the first connection migrates and the second retries migration.
        first_applied = migrate(first)
        second_applied = migrate(second)
        versions = second.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        current = second.execute(
            "SELECT current_version FROM schema_state WHERE singleton = 1"
        ).fetchone()[0]

    # Then: only one journal row and one current version exist.
    assert first_applied == tuple(migration.version for migration in MIGRATIONS)
    assert second_applied == ()
    assert versions == [(migration.version,) for migration in MIGRATIONS]
    assert current == CURRENT_SCHEMA_VERSION


def test_foreign_key_rejects_task_for_unknown_run(tmp_path: Path) -> None:
    # Given: a migrated database with no runs.
    with connect_project_database(tmp_path / "project") as connection:
        migrate(connection)

        # When/Then: a task cannot reference an absent run.
        with pytest.raises(sqlite3.IntegrityError):
            with write_transaction(connection):
                connection.execute(
                    "INSERT INTO tasks(task_id, run_id, task_type, state, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("task-1", "missing-run", "chapter", "pending", "2030-01-02"),
                )


def test_rollback_leaves_no_partial_rows_or_current_pointer(tmp_path: Path) -> None:
    # Given: a migrated database and a valid artifact write.
    revision = ArtifactRevisionWrite(
        artifact_id="chapter-1",
        revision=1,
        project_id="novel-one",
        run_id=None,
        kind="chapter",
        status="accepted",
        raw_bytes=b"must roll back",
        schema_version=1,
        created_at="2030-01-02T03:04:05+00:00",
    )
    with connect_project_database(tmp_path / "project") as connection:
        migrate(connection)

        # When: the caller fails after inserting inside an explicit write transaction.
        with pytest.raises(ForcedTransactionFailure, match="forced failure"):
            with write_transaction(connection):
                connection.execute(
                    "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, "
                    "kind, status, raw_bytes, sha256, schema_version, is_current, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                    (
                        revision.artifact_id,
                        revision.revision,
                        revision.project_id,
                        revision.run_id,
                        revision.kind,
                        revision.status,
                        revision.raw_bytes,
                        hashlib.sha256(revision.raw_bytes).hexdigest(),
                        revision.schema_version,
                        revision.created_at,
                    ),
                )
                raise ForcedTransactionFailure("forced failure")

        rows = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
        current_rows = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE is_current = 1"
        ).fetchone()[0]

    # Then: rollback removes both the revision and its current marker.
    assert rows == 0
    assert current_rows == 0


def test_failed_migration_rolls_back_journal_pointer_and_schema(tmp_path: Path) -> None:
    # Given: a fresh database and a migration that fails after a schema mutation.
    failing = Migration(
        version=1,
        name="forced_failure",
        statements=(
            "CREATE TABLE partial_row(id INTEGER PRIMARY KEY)",
            "INSERT INTO missing_table(id) VALUES (1)",
        ),
    )
    with connect_project_database(tmp_path / "project") as connection:
        # When: migration execution raises from SQLite.
        with pytest.raises(sqlite3.OperationalError):
            migrate(connection, migrations=(failing,))
        tables = connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND name = 'partial_row'"
        ).fetchall()
        journal_rows = connection.execute(
            "SELECT COUNT(*) FROM schema_migrations"
        ).fetchone()[0]
        current_rows = connection.execute("SELECT COUNT(*) FROM schema_state").fetchone()[0]

    # Then: neither the DDL nor migration metadata survives the failed transaction.
    assert tables == []
    assert journal_rows == 0
    assert current_rows == 0
