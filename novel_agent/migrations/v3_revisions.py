from typing import Final


REVISION_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE chapter_branches (
        branch_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        parent_branch_id TEXT REFERENCES chapter_branches(branch_id) ON DELETE RESTRICT,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        activated_at TEXT
    )
    """,
    """
    CREATE TABLE chapter_branch_members (
        branch_id TEXT NOT NULL REFERENCES chapter_branches(branch_id) ON DELETE CASCADE,
        chapter_number INTEGER NOT NULL CHECK (chapter_number > 0),
        artifact_id TEXT NOT NULL,
        artifact_revision INTEGER NOT NULL CHECK (artifact_revision > 0),
        PRIMARY KEY (branch_id, chapter_number),
        FOREIGN KEY (artifact_id, artifact_revision)
            REFERENCES artifacts(artifact_id, revision) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE chapter_rebase_proposals (
        proposal_id TEXT PRIMARY KEY,
        branch_id TEXT NOT NULL REFERENCES chapter_branches(branch_id) ON DELETE RESTRICT,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        project_id TEXT NOT NULL,
        status TEXT NOT NULL,
        proposal_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        confirmed_at TEXT,
        UNIQUE (run_id, branch_id)
    )
    """,
    "CREATE INDEX chapter_branches_by_run ON chapter_branches(run_id, status)",
    "CREATE INDEX chapter_rebase_proposals_by_run ON chapter_rebase_proposals(run_id, status)",
)


__all__ = ("REVISION_SCHEMA_STATEMENTS",)
