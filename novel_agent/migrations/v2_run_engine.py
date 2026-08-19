from typing import Final


RUN_ENGINE_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE task_results (
        task_id TEXT PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        result_sha256 TEXT NOT NULL CHECK (length(result_sha256) = 64),
        committed_fence INTEGER NOT NULL CHECK (committed_fence > 0),
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE run_checkpoints (
        checkpoint_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
        stage TEXT NOT NULL,
        result_sha256 TEXT NOT NULL CHECK (length(result_sha256) = 64),
        fencing_token INTEGER NOT NULL CHECK (fencing_token > 0),
        created_at TEXT NOT NULL,
        UNIQUE (task_id, stage)
    )
    """,
    "CREATE INDEX task_results_by_run ON task_results(run_id, created_at)",
    "CREATE INDEX run_checkpoints_by_run ON run_checkpoints(run_id, created_at)",
)


__all__ = ("RUN_ENGINE_SCHEMA_STATEMENTS",)
