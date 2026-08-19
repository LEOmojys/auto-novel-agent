from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from novel_agent.pathing import ProjectPathResolver


@dataclass(frozen=True, slots=True)
class ChapterCommitContext:
    connection: sqlite3.Connection
    path_resolver: ProjectPathResolver
    committed_at: datetime
    fencing_token: int


__all__ = ("ChapterCommitContext",)
