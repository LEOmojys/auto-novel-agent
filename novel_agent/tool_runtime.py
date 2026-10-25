"""Cooperative tool deadline and cancellation propagation.

Python cannot safely kill a thread that is holding a SQLite write
transaction, so a tool deadline is enforced at explicit boundaries:

* before a tool starts and after a read-only tool returns;
* before every provider attempt and before commit points in write tools;
* during streaming HTTP calls, by passing a deadline/cancellation token to
  the provider transport (the legacy transport checks it between chunks).

The context variable means existing tool/provider signatures do not change.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


class ToolCancelledError(RuntimeError):
    """The run was cancelled while a tool was executing."""


class ToolTimedOut(ToolCancelledError):
    """A tool exceeded its per-call deadline."""


@dataclass(frozen=True, slots=True)
class ToolExecutionToken:
    parent: object | None
    deadline: float | None

    def expired(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    def cancelled_by_parent(self) -> bool:
        return self.parent is not None and bool(getattr(self.parent, "is_cancelled", lambda: False)())

    def is_cancelled(self) -> bool:
        return self.cancelled_by_parent() or self.expired()


_CURRENT: ContextVar[ToolExecutionToken | None] = ContextVar("novel_agent_tool_scope", default=None)


@contextmanager
def tool_scope(cancellation: object | None, timeout_seconds: float | None) -> Iterator[ToolExecutionToken]:
    deadline = (
        time.monotonic() + float(timeout_seconds)
        if timeout_seconds is not None and float(timeout_seconds) > 0.0
        else None
    )
    token = ToolExecutionToken(cancellation, deadline)
    reset = _CURRENT.set(token)
    try:
        yield token
    finally:
        _CURRENT.reset(reset)


def current_tool_token() -> ToolExecutionToken | None:
    return _CURRENT.get()


def check_tool_active() -> None:
    token = _CURRENT.get()
    if token is None:
        return
    if token.cancelled_by_parent():
        raise ToolCancelledError("任务已被取消")
    if token.expired():
        raise ToolTimedOut("工具执行超时")


class CombinedToken:
    def __init__(self, *tokens: object | None) -> None:
        self._tokens = tuple(token for token in tokens if token is not None)

    def is_cancelled(self) -> bool:
        for token in self._tokens:
            checker = getattr(token, "is_cancelled", None)
            if checker is not None and checker():
                return True
        return False


def combine_tokens(*tokens: object | None) -> CombinedToken:
    return CombinedToken(*tokens)


__all__ = (
    "CombinedToken",
    "ToolCancelledError",
    "ToolExecutionToken",
    "ToolTimedOut",
    "check_tool_active",
    "combine_tokens",
    "current_tool_token",
    "tool_scope",
)
