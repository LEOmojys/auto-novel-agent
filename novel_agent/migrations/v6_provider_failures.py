from typing import Final

PROVIDER_FAILURE_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS provider_failures (
        failure_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
        caller TEXT NOT NULL,
        model_name TEXT NOT NULL,
        failure_kind TEXT NOT NULL,
        reason TEXT NOT NULL,
        detail TEXT NOT NULL DEFAULT '',
        prompt_sha256 TEXT NOT NULL,
        request_excerpt TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_provider_failures_run
    ON provider_failures(run_id, created_at)
    """,
)

__all__ = ("PROVIDER_FAILURE_SCHEMA_STATEMENTS",)
