"""Memory completion: status normalization, review ledger, context packs.

Version 7 closes the gaps between the memory design and its wiring:

* legacy M1 rows written by partial planning commits used `status='validated'`
  while the context reader required `'canonical'`; normalize the vocabulary;
* record human review decisions for needs_review/conflicted candidates;
* persist assembled context packs (the M4 retrieval view) for audit;
* rebuild the derived FTS index so it cannot keep orphan/duplicate rows.
"""

from typing import Final

MEMORY_COMPLETION_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    "ALTER TABLE memory_candidates ADD COLUMN chapter_number INTEGER",
    """
    CREATE TABLE IF NOT EXISTS memory_reviews (
        review_id TEXT PRIMARY KEY,
        candidate_id TEXT NOT NULL REFERENCES memory_candidates(candidate_id) ON DELETE CASCADE,
        project_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        decision TEXT NOT NULL,
        actor TEXT NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        memory_id TEXT,
        decided_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS memory_reviews_by_candidate
    ON memory_reviews(candidate_id, decided_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS context_packs (
        pack_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        chapter_plan_id TEXT NOT NULL,
        digest TEXT NOT NULL,
        tokenizer_version TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS context_packs_by_run
    ON context_packs(project_id, run_id, chapter_plan_id)
    """,
    # One status vocabulary for M1: "canonical" (the record_status contract).
    "UPDATE memory_records SET status = 'canonical' WHERE layer = 'm1_canonical' AND status = 'validated'",
    # Fully derived index: rebuild from story truth so orphans/duplicates vanish.
    "DELETE FROM memory_fts",
    "INSERT INTO memory_fts(memory_id, content) SELECT memory_id, value_json FROM memory_records",
)

__all__ = ("MEMORY_COMPLETION_SCHEMA_STATEMENTS",)
