from typing import Final

OPERATIONAL_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE runs (
        run_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        state TEXT NOT NULL,
        plan_revision INTEGER NOT NULL DEFAULT 0 CHECK (plan_revision >= 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE run_events (
        event_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        sequence INTEGER NOT NULL CHECK (sequence >= 0),
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (run_id, sequence)
    )
    """,
    """
    CREATE TABLE tasks (
        task_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        task_type TEXT NOT NULL,
        state TEXT NOT NULL,
        idempotency_key TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT,
        UNIQUE (run_id, idempotency_key)
    )
    """,
    """
    CREATE TABLE commands (
        command_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        command_type TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        state TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (run_id, idempotency_key)
    )
    """,
    """
    CREATE TABLE run_leases (
        run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
        holder_id TEXT NOT NULL,
        fencing_token INTEGER NOT NULL CHECK (fencing_token > 0),
        expires_at TEXT NOT NULL,
        acquired_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE artifacts (
        artifact_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision > 0),
        project_id TEXT NOT NULL,
        run_id TEXT REFERENCES runs(run_id) ON DELETE RESTRICT,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        raw_bytes BLOB NOT NULL,
        sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
        schema_version INTEGER NOT NULL CHECK (schema_version > 0),
        is_current INTEGER NOT NULL DEFAULT 0 CHECK (is_current IN (0, 1)),
        created_at TEXT NOT NULL,
        PRIMARY KEY (artifact_id, revision)
    )
    """,
    """
    CREATE UNIQUE INDEX one_current_artifact_revision
    ON artifacts(artifact_id) WHERE is_current = 1
    """,
    """
    CREATE INDEX artifacts_by_run ON artifacts(run_id, kind, status)
    """,
    """
    CREATE TABLE artifact_dependencies (
        artifact_id TEXT NOT NULL,
        artifact_revision INTEGER NOT NULL,
        depends_on_artifact_id TEXT NOT NULL,
        depends_on_revision INTEGER NOT NULL,
        dependency_kind TEXT NOT NULL,
        PRIMARY KEY (
            artifact_id, artifact_revision,
            depends_on_artifact_id, depends_on_revision, dependency_kind
        ),
        FOREIGN KEY (artifact_id, artifact_revision)
            REFERENCES artifacts(artifact_id, revision) ON DELETE CASCADE,
        FOREIGN KEY (depends_on_artifact_id, depends_on_revision)
            REFERENCES artifacts(artifact_id, revision) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE artifact_outbox (
        outbox_id TEXT PRIMARY KEY,
        artifact_id TEXT NOT NULL,
        artifact_revision INTEGER NOT NULL,
        target_path TEXT NOT NULL,
        state TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (artifact_id, artifact_revision, target_path),
        FOREIGN KEY (artifact_id, artifact_revision)
            REFERENCES artifacts(artifact_id, revision) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX artifact_outbox_pending ON artifact_outbox(state, created_at)
    """,
    """
    CREATE TABLE provider_calls (
        call_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        task_id TEXT REFERENCES tasks(task_id) ON DELETE SET NULL,
        attempt INTEGER NOT NULL CHECK (attempt > 0),
        request_sha256 TEXT NOT NULL,
        response_sha256 TEXT,
        state TEXT NOT NULL,
        error_code TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (task_id, attempt)
    )
    """,
    """
    CREATE TABLE tool_calls (
        call_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        task_id TEXT REFERENCES tasks(task_id) ON DELETE SET NULL,
        tool_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        response_sha256 TEXT,
        state TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (run_id, idempotency_key)
    )
    """,
)

__all__ = ("OPERATIONAL_SCHEMA_STATEMENTS",)
