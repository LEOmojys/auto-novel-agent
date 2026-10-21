from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from novel_agent.migrations import INITIAL_SCHEMA_STATEMENTS
from novel_agent.migrations.v2_run_engine import RUN_ENGINE_SCHEMA_STATEMENTS
from novel_agent.migrations.v3_revisions import REVISION_SCHEMA_STATEMENTS
from novel_agent.migrations.v4_tool_call_results import TOOL_CALL_RESULT_SCHEMA_STATEMENTS
from novel_agent.migrations.v5_m4_retrieval import M4_RETRIEVAL_SCHEMA_STATEMENTS
from novel_agent.migrations.v6_provider_failures import PROVIDER_FAILURE_SCHEMA_STATEMENTS
from novel_agent.migrations.v7_memory_completion import MEMORY_COMPLETION_SCHEMA_STATEMENTS
from novel_agent.sqlite_transactions import TransactionStateError, write_transaction
from novel_agent.tool_call_ledger import (
    ToolCallLedgerWrite,
    ToolCallReservation,
    complete_tool_call,
    persist_tool_call,
    reserve_tool_call,
)

DATABASE_FILENAME: Final = "agent.sqlite"
CURRENT_SCHEMA_VERSION: Final = 7


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArtifactRevisionWrite:
    artifact_id: str
    revision: int
    project_id: str
    run_id: str | None
    kind: str
    status: str
    raw_bytes: bytes
    schema_version: int
    created_at: str


@dataclass(frozen=True, slots=True)
class ArtifactCommitReceipt:
    artifact_id: str
    revision: int
    sha256: str


class MigrationPlanError(RuntimeError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


MIGRATIONS: Final[tuple[Migration, ...]] = (
    Migration(
        version=1,
        name="initial_authoritative_store",
        statements=INITIAL_SCHEMA_STATEMENTS,
    ),
    Migration(
        version=2,
        name="run_engine_checkpoints",
        statements=RUN_ENGINE_SCHEMA_STATEMENTS,
    ),
    Migration(
        version=3,
        name="chapter_revision_branches",
        statements=REVISION_SCHEMA_STATEMENTS,
    ),
    Migration(
        version=4,
        name="tool_call_result_payloads",
        statements=TOOL_CALL_RESULT_SCHEMA_STATEMENTS,
    ),
    Migration(
        version=5,
        name="m4_retrieval_and_conflicts",
        statements=M4_RETRIEVAL_SCHEMA_STATEMENTS,
    ),
    Migration(
        version=6,
        name="provider_failure_traceability",
        statements=PROVIDER_FAILURE_SCHEMA_STATEMENTS,
    ),
    Migration(
        version=7,
        name="memory_completion",
        statements=MEMORY_COMPLETION_SCHEMA_STATEMENTS,
    ),
)


@contextmanager
def connect_project_database(project_dir: Path) -> Iterator[sqlite3.Connection]:
    project_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        project_dir / DATABASE_FILENAME,
        timeout=5.0,
        isolation_level=None,
        check_same_thread=False,
    )
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        yield connection
    finally:
        connection.close()


def migrate(
    connection: sqlite3.Connection,
    migrations: Sequence[Migration] = MIGRATIONS,
) -> tuple[int, ...]:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL "
        "DEFAULT CURRENT_TIMESTAMP)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_state ("
        "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
        "current_version INTEGER NOT NULL)"
    )
    ordered = tuple(sorted(migrations, key=lambda item: item.version))
    applied_versions: list[int] = []
    for migration in ordered:
        with write_transaction(connection):
            current_row = connection.execute(
                "SELECT current_version FROM schema_state WHERE singleton = 1"
            ).fetchone()
            current_version = 0 if current_row is None else int(current_row[0])
            if migration.version <= current_version:
                journaled = connection.execute(
                    "SELECT name FROM schema_migrations WHERE version = ?",
                    (migration.version,),
                ).fetchone()
                if journaled is None or journaled[0] != migration.name:
                    raise MigrationPlanError(
                        f"Migration {migration.version} does not match its journal entry"
                    )
                continue
            if migration.version != current_version + 1:
                raise MigrationPlanError(
                    f"Expected migration {current_version + 1}, got {migration.version}"
                )
            for statement in migration.statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, name) VALUES (?, ?)",
                (migration.version, migration.name),
            )
            connection.execute(
                "INSERT INTO schema_state(singleton, current_version) VALUES (1, ?) "
                "ON CONFLICT(singleton) DO UPDATE SET current_version = excluded.current_version",
                (migration.version,),
            )
            applied_versions.append(migration.version)
    return tuple(applied_versions)


def commit_artifact_revision(
    connection: sqlite3.Connection,
    revision: ArtifactRevisionWrite,
) -> ArtifactCommitReceipt:
    digest = hashlib.sha256(revision.raw_bytes).hexdigest()
    with write_transaction(connection):
        connection.execute(
            "UPDATE artifacts SET is_current = 0 "
            "WHERE artifact_id = ? AND is_current = 1",
            (revision.artifact_id,),
        )
        connection.execute(
            "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, "
            "raw_bytes, sha256, schema_version, is_current, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
            (
                revision.artifact_id,
                revision.revision,
                revision.project_id,
                revision.run_id,
                revision.kind,
                revision.status,
                revision.raw_bytes,
                digest,
                revision.schema_version,
                revision.created_at,
            ),
        )
    return ArtifactCommitReceipt(
        artifact_id=revision.artifact_id,
        revision=revision.revision,
        sha256=digest,
    )


__all__ = (
    "CURRENT_SCHEMA_VERSION",
    "ArtifactCommitReceipt",
    "ArtifactRevisionWrite",
    "Migration",
    "MigrationPlanError",
    "commit_artifact_revision",
    "complete_tool_call",
    "connect_project_database",
    "migrate",
    "persist_tool_call",
    "reserve_tool_call",
    "ToolCallReservation",
    "ToolCallLedgerWrite",
    "write_transaction",
)
