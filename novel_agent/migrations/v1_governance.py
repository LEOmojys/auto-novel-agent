from typing import Final

GOVERNANCE_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE quality_reports (
        report_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        task_id TEXT REFERENCES tasks(task_id) ON DELETE SET NULL,
        artifact_id TEXT NOT NULL,
        artifact_revision INTEGER NOT NULL,
        decision TEXT NOT NULL,
        report_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (artifact_id, artifact_revision)
            REFERENCES artifacts(artifact_id, revision) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE memory_candidates (
        candidate_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        source_artifact_id TEXT NOT NULL,
        source_artifact_revision INTEGER NOT NULL,
        extraction_schema_version INTEGER NOT NULL CHECK (extraction_schema_version > 0),
        normalized_fact_hash TEXT NOT NULL,
        status TEXT NOT NULL,
        candidate_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (
            source_artifact_id, source_artifact_revision,
            extraction_schema_version, normalized_fact_hash
        ),
        FOREIGN KEY (source_artifact_id, source_artifact_revision)
            REFERENCES artifacts(artifact_id, revision) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE memory_records (
        memory_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        run_id TEXT REFERENCES runs(run_id) ON DELETE RESTRICT,
        plan_revision INTEGER NOT NULL CHECK (plan_revision >= 0),
        layer TEXT NOT NULL,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        predicate TEXT NOT NULL,
        value_json TEXT NOT NULL,
        chapter_start INTEGER,
        chapter_end INTEGER,
        source_artifact_id TEXT,
        source_artifact_revision INTEGER,
        source_span TEXT,
        source_hash TEXT,
        confidence REAL,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        supersedes_id TEXT REFERENCES memory_records(memory_id) ON DELETE RESTRICT,
        FOREIGN KEY (source_artifact_id, source_artifact_revision)
            REFERENCES artifacts(artifact_id, revision) ON DELETE RESTRICT
    )
    """,
    """
    CREATE INDEX memory_records_current
    ON memory_records(project_id, layer, status, subject_id, predicate)
    """,
    """
    CREATE TABLE memory_edges (
        edge_id TEXT PRIMARY KEY,
        source_memory_id TEXT NOT NULL REFERENCES memory_records(memory_id) ON DELETE CASCADE,
        target_memory_id TEXT NOT NULL REFERENCES memory_records(memory_id) ON DELETE RESTRICT,
        relation TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (source_memory_id, target_memory_id, relation)
    )
    """,
    """
    CREATE TABLE legacy_imports (
        import_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        source_relative_path TEXT NOT NULL,
        raw_bytes BLOB NOT NULL,
        sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
        import_schema_version INTEGER NOT NULL CHECK (import_schema_version > 0),
        parsed_json TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (project_id, source_relative_path, import_schema_version)
    )
    """,
    """
    CREATE INDEX legacy_imports_by_hash
    ON legacy_imports(project_id, sha256, import_schema_version)
    """,
    """
    CREATE TABLE import_issues (
        issue_id TEXT PRIMARY KEY,
        import_id TEXT REFERENCES legacy_imports(import_id) ON DELETE CASCADE,
        issue_code TEXT NOT NULL,
        detail_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (import_id, issue_code, detail_json)
    )
    """,
    """
    CREATE TABLE audit_manifests (
        manifest_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        schema_version INTEGER NOT NULL CHECK (schema_version > 0),
        manifest_json TEXT NOT NULL,
        sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
        created_at TEXT NOT NULL
    )
    """,
)

__all__ = ("GOVERNANCE_SCHEMA_STATEMENTS",)
