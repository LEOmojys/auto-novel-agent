"""Agent-loop state persistence: resume a loop after approval / repair.

The fixed DAG re-runs the whole stage after a repair; an agent loop must
resume from the exact step it paused at (approval gate) or crashed at.  This
module persists the minimal loop state — step counter, history event
pointer, pending action, current plan ref and written chapters — in the run
database so ``_AgentLoopStage`` can rebuild the loop from where it stopped.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class LoopState:
    """Persisted agent-loop state for one run."""

    run_id: str
    step: int = 0
    history_event_seq: int | None = None  # last run_event sequence consumed
    pending_action: dict | None = None    # {"tool": ..., "args": ...} to continue
    plan_ref: tuple[str, int] | None = None  # (plan_id, revision)
    written_chapters: tuple[int, ...] = ()
    summary: str = ""
    remaining_plan: tuple[tuple[str, dict], ...] = ()  # unexecuted plan steps (approval pause)
    plan_index: int = 0  # steps already executed of the original plan

    def model_dump(self) -> dict:
        return {
            "run_id": self.run_id,
            "step": self.step,
            "history_event_seq": self.history_event_seq,
            "pending_action": self.pending_action,
            "plan_ref": list(self.plan_ref) if self.plan_ref else None,
            "written_chapters": list(self.written_chapters),
            "summary": self.summary,
            "remaining_plan": [list(step) for step in self.remaining_plan],
            "plan_index": self.plan_index,
        }


_LOOP_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS loop_state (
    run_id TEXT PRIMARY KEY,
    step INTEGER NOT NULL DEFAULT 0,
    history_event_seq INTEGER,
    pending_action_json TEXT,
    plan_ref_json TEXT,
    written_chapters_json TEXT NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL DEFAULT '',
    remaining_plan_json TEXT,
    plan_index INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
)
"""


def ensure_loop_state_table(connection: sqlite3.Connection) -> None:
    connection.execute(_LOOP_STATE_SCHEMA)
    # Backfill columns for databases created before plan persistence.
    columns = {row[1] for row in connection.execute("PRAGMA table_info(loop_state)").fetchall()}
    if "remaining_plan_json" not in columns:
        connection.execute("ALTER TABLE loop_state ADD COLUMN remaining_plan_json TEXT")
    if "plan_index" not in columns:
        connection.execute("ALTER TABLE loop_state ADD COLUMN plan_index INTEGER NOT NULL DEFAULT 0")


def save_loop_state(
    connection: sqlite3.Connection,
    state: LoopState,
    now: datetime,
) -> None:
    ensure_loop_state_table(connection)
    connection.execute(
        "INSERT INTO loop_state(run_id, step, history_event_seq, pending_action_json, plan_ref_json, "
        "written_chapters_json, summary, remaining_plan_json, plan_index, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(run_id) DO UPDATE SET step = excluded.step, "
        "history_event_seq = excluded.history_event_seq, pending_action_json = excluded.pending_action_json, "
        "plan_ref_json = excluded.plan_ref_json, written_chapters_json = excluded.written_chapters_json, "
        "summary = excluded.summary, remaining_plan_json = excluded.remaining_plan_json, "
        "plan_index = excluded.plan_index, updated_at = excluded.updated_at",
        (
            state.run_id,
            state.step,
            state.history_event_seq,
            json.dumps(state.pending_action) if state.pending_action else None,
            json.dumps(state.plan_ref) if state.plan_ref else None,
            json.dumps(state.written_chapters),
            state.summary,
            json.dumps([list(step) for step in state.remaining_plan]) if state.remaining_plan else None,
            state.plan_index,
            now.isoformat(),
        ),
    )


def load_loop_state(connection: sqlite3.Connection, run_id: str) -> LoopState | None:
    ensure_loop_state_table(connection)
    row = connection.execute(
        "SELECT step, history_event_seq, pending_action_json, plan_ref_json, written_chapters_json, summary, "
        "remaining_plan_json, plan_index FROM loop_state WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return None

    def _parse(payload: str | None, default):
        if payload is None:
            return default
        try:
            return json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return default

    plan_ref = _parse(row[3], None)
    remaining = _parse(row[6], [])
    return LoopState(
        run_id=run_id,
        step=int(row[0] or 0),
        history_event_seq=int(row[1]) if row[1] is not None else None,
        pending_action=_parse(row[2], None),
        plan_ref=tuple(plan_ref) if isinstance(plan_ref, list) and len(plan_ref) == 2 else None,
        written_chapters=tuple(int(n) for n in _parse(row[4], [])),
        summary=str(row[5] or ""),
        remaining_plan=tuple(
            (str(step[0]), dict(step[1])) for step in remaining
            if isinstance(step, list) and len(step) == 2 and isinstance(step[0], str)
        ),
        plan_index=int(row[7] or 0),
    )


def clear_loop_state(connection: sqlite3.Connection, run_id: str) -> None:
    ensure_loop_state_table(connection)
    connection.execute("DELETE FROM loop_state WHERE run_id = ?", (run_id,))


__all__ = ("LoopState", "clear_loop_state", "ensure_loop_state_table", "load_loop_state", "save_loop_state")
