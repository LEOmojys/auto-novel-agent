from typing import Final

M4_RETRIEVAL_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS memory_conflicts (
        conflict_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
        candidate_id TEXT NOT NULL REFERENCES memory_candidates(candidate_id) ON DELETE CASCADE,
        existing_memory_id TEXT REFERENCES memory_records(memory_id) ON DELETE RESTRICT,
        reason_code TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        memory_id UNINDEXED,
        content,
        tokenize='trigram'
    )
    """,
)

__all__ = ("M4_RETRIEVAL_SCHEMA_STATEMENTS",)
