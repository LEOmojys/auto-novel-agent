from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)  # contextlib assigns tracebacks while rolling back transactions
class ChapterPersistenceError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(slots=True)  # contextlib assigns tracebacks while rolling back transactions
class ChapterTaskStateError(Exception):
    task_id: str

    def __str__(self) -> str:
        return f"chapter task is not active: {self.task_id}"


@dataclass(slots=True)  # contextlib assigns tracebacks while rolling back transactions
class ChapterCommitBlocked(Exception):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


__all__ = ("ChapterCommitBlocked", "ChapterPersistenceError", "ChapterTaskStateError")
