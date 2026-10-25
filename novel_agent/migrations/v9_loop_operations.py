"""Loop write-operation journal.

Production recovery needs a durable record of completed model-facing write
operations, separate from the provider-tool idempotency ledger.  A loop tool
may finish its domain commit and then lose the process before the model sees
the result; caching the successful operation lets a resumed loop return the
existing result instead of creating another artifact revision.
"""

from typing import Final

LOOP_OPERATION_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS loop_operations (
        run_id TEXT NOT NULL,
        operation_key TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        state TEXT NOT NULL,
        result_json TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (run_id, operation_key)
    )
    """,
)
