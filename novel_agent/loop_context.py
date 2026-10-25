"""Loop message model and adaptive context compaction.

The first iteration of the agent loop compacted by message count: keep the
last N messages and replace the head with a one-line tool-name summary.  That
does not converge when the retained messages are themselves large (the window
can exceed the token budget), and repeated compactions summarised the previous
summary away because summaries are not tool messages.

This module implements the approach used by production coding agents
(Claude Code / Codex / OpenCode): a token-budgeted tail, tool-output
microcompaction, a rolling structured summary, and a hard cap that guarantees
the prompt ends up under the compaction threshold.

Design:

* **Budget, not count** — the retained tail is selected newest-first until the
  target token budget is reached (with a minimum number of recent messages).
* **Rolling summary** — summaries accumulate across compactions; older lines
  are folded only when the summary itself exceeds its character budget.
* **Microcompaction** — oversized tool observations keep head+tail with an
  explicit elision marker before full compaction is attempted.
* **Convergence** — compaction repeats until the estimate is at or below the
  target, then falls back to a hard trim that guarantees the threshold.
* **Pinned messages** — the original user task is never dropped.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from novel_agent.memory_text import estimate_tokens


@dataclass(frozen=True, slots=True)
class LoopMessage:
    """One message in the loop's working memory."""

    role: Literal["user", "assistant", "tool"]
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_ok: bool = True
    #: Artifact produced/affected by a tool call; kept so compaction summaries
    #: and resumed runs can still point the model at durable outputs.
    artifact_ref: str | None = None
    #: The original task must survive every compaction.
    pinned: bool = False
    #: A rolling compaction summary, rendered as a distinct prompt section.
    is_summary: bool = False


@dataclass(frozen=True, slots=True)
class CompactionReport:
    before_tokens: int
    after_tokens: int
    threshold: int
    dropped_messages: int
    rounds: int
    summary_chars: int
    hard_limited: bool
    summary_text: str = ""
    summary_source: str = "deterministic"

    def as_payload(self, kept: int) -> dict[str, object]:
        return {
            # Kept for compatibility with the previous event schema.
            "kept": kept,
            "summary": self.summary_text,
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "threshold": self.threshold,
            "dropped": self.dropped_messages,
            "rounds": self.rounds,
            "hard_limited": self.hard_limited,
        }


_SUMMARY_HEADER = "以下为较早步骤的压缩摘要（滚动累积）"
_FOLDED = "（更早的摘要已折叠）"


class ContextManager:
    """Token-budgeted compaction for one loop run."""

    def __init__(
        self,
        *,
        context_window: int,
        compact_buffer: int,
        compact_at: int | None = None,
        max_tail_messages: int = 20,
        target_ratio: float = 0.55,
        summary_max_chars: int = 4_000,
        max_tool_chars: int = 4_000,
        summarizer: Callable[[Sequence[LoopMessage], str], str | None] | None = None,
    ) -> None:
        configured = int(compact_at) if compact_at is not None else int(context_window) - int(compact_buffer)
        self.threshold = max(256, configured)
        self.target = max(128, int(self.threshold * float(target_ratio)))
        self.max_tail_messages = max(1, int(max_tail_messages))
        self.summary_max_chars = max(200, int(summary_max_chars))
        self.max_tool_chars = max(200, int(max_tool_chars))
        self.summary = ""
        self.summarizer = summarizer
        self.summary_source = "deterministic"
        self.compactions = 0

    # -- lifecycle ---------------------------------------------------------

    def seed_summary(self, summary: str | None) -> None:
        if summary:
            self.summary = summary.strip()

    # -- public API --------------------------------------------------------

    def maybe_compact(
        self,
        history: list[LoopMessage],
        overhead_tokens: int,
    ) -> tuple[list[LoopMessage], CompactionReport | None]:
        """Return (possibly compacted history, report when compaction ran)."""
        if not history:
            return history, None
        if not self.summary:
            existing = next((message.content for message in history if message.is_summary), "")
            if existing:
                self.summary = existing
        before_tokens = overhead_tokens + sum(self._tokens(message) for message in history)
        if before_tokens <= self.threshold:
            return list(history), None

        work = [self._microcompact(message) for message in history]
        total = overhead_tokens + sum(self._tokens(message) for message in work)
        original_count = len(work)
        rounds = 0
        dropped_count = 0
        hard_limited = False
        summary_attempted = False

        while total > self.target and rounds < 8:
            pinned = [message for message in work if message.pinned]
            candidates = [
                message
                for message in work
                if not message.pinned and not message.is_summary
            ]
            tail, head = self._split_tail(
                candidates,
                budget=max(0, self.target - overhead_tokens - 256),
            )
            if not head:
                break
            self._absorb(
                head,
                allow_model=self.summarizer is not None and not summary_attempted,
            )
            summary_attempted = True
            dropped_count += len(head)
            summary_message = LoopMessage(
                role="assistant",
                content=self.summary,
                is_summary=True,
            )
            work = [*pinned, summary_message, *tail]
            new_total = overhead_tokens + sum(self._tokens(message) for message in work)
            rounds += 1
            if new_total >= total:
                # No progress (budget too small to summarise anything useful);
                # stop and let the hard cap guarantee the threshold.
                total = new_total
                break
            total = new_total
            if total <= self.target:
                break

        if total > self.threshold:
            hard_limited = True
            work, total = self._hard_trim(work, overhead_tokens)

        self.compactions += 1
        report = CompactionReport(
            before_tokens=before_tokens,
            after_tokens=total,
            threshold=self.threshold,
            dropped_messages=dropped_count,
            rounds=rounds,
            summary_chars=len(self.summary),
            hard_limited=hard_limited,
            summary_text=self.summary,
            summary_source=self.summary_source,
        )
        return work, report

    # -- selection ---------------------------------------------------------

    def _split_tail(
        self,
        messages: list[LoopMessage],
        *,
        budget: int,
    ) -> tuple[list[LoopMessage], list[LoopMessage]]:
        """Take newest messages until the token budget or the count cap.

        The size floor is a hard token budget, not a message count: forcing a
        minimum number of large messages is exactly what made the previous
        implementation non-convergent.  At least one message is always kept so
        the model retains the most recent observation.
        """
        if not messages:
            return [], []
        tail: list[LoopMessage] = []
        used = 0
        index = len(messages) - 1
        while index >= 0:
            message = messages[index]
            cost = self._tokens(message)
            if tail and (used + cost > budget or len(tail) >= self.max_tail_messages):
                break
            tail.append(message)
            used += cost
            index -= 1
        tail.reverse()
        head = messages[: index + 1]
        return tail, head

    @staticmethod
    def _tokens(message: LoopMessage) -> int:
        if message.role == "tool":
            text = f"工具[{message.tool_name or ''}]结果: {message.content}"
        elif message.is_summary:
            text = f"摘要: {message.content}"
        else:
            text = message.content
        return estimate_tokens(text) + 4

    # -- microcompaction ---------------------------------------------------

    def _microcompact(self, message: LoopMessage) -> LoopMessage:
        if message.role != "tool" or len(message.content) <= self.max_tool_chars:
            return message
        keep = max(100, self.max_tool_chars // 2)
        omitted = len(message.content) - keep * 2
        content = (
            message.content[:keep]
            + f"\n…（中间省略 {omitted} 字，完整内容见产物/事件流）…\n"
            + message.content[-keep:]
        )
        return replace(message, content=content)

    # -- summarisation -----------------------------------------------------

    def _absorb(self, messages: list[LoopMessage], *, allow_model: bool) -> None:
        if allow_model and self.summarizer is not None:
            try:
                produced = self.summarizer(messages, self.summary)
            except Exception:  # noqa: BROAD_EXCEPT_OK - fall back to deterministic summary
                produced = None
            if produced and produced.strip():
                self.summary = produced.strip()
                self._cap_summary()
                self.summary_source = "model"
                return
        self._absorb_deterministic(messages)
        self.summary_source = "deterministic"

    def _absorb_deterministic(self, messages: list[LoopMessage]) -> None:
        lines: list[str] = []
        artifacts: list[str] = []
        for message in messages:
            if message.is_summary:
                continue
            if message.role == "tool":
                args = self._compact_args(message.tool_args)
                outcome = "ok" if message.tool_ok else "failed"
                line = f"- {message.tool_name}{args} → {outcome}"
                if message.artifact_ref:
                    line += f" [产物 {message.artifact_ref}]"
                    if message.artifact_ref not in artifacts:
                        artifacts.append(message.artifact_ref)
                lines.append(line)
            elif message.role == "assistant" and message.content.strip():
                text = " ".join(message.content.split())
                if len(text) > 120:
                    text = text[:120] + "…"
                lines.append(f"- 决策：{text}")
        if not lines and not artifacts:
            return
        block: list[str] = []
        if lines:
            block.append("\n".join(self._dedupe(lines)))
        if artifacts:
            block.append("关键产物：" + "、".join(artifacts))
        addition = "\n".join(block)
        if self.summary:
            self.summary = self.summary.rstrip() + "\n" + addition
        else:
            self.summary = _SUMMARY_HEADER + "\n" + addition
        self._cap_summary()

    @staticmethod
    def _compact_args(args: dict | None) -> str:
        if not args:
            return "()"
        try:
            text = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            text = str(args)
        if len(text) > 80:
            text = text[:80] + "…"
        return f"({text})"

    @staticmethod
    def _dedupe(lines: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for line in lines:
            if line in seen:
                continue
            seen.add(line)
            result.append(line)
        return result

    def _cap_summary(self) -> None:
        if len(self.summary) <= self.summary_max_chars:
            return
        keep = self.summary_max_chars - len(_SUMMARY_HEADER) - len(_FOLDED) - 2
        self.summary = _SUMMARY_HEADER + "\n" + _FOLDED + "\n" + self.summary[-max(0, keep):]

    # -- hard cap ----------------------------------------------------------

    def _hard_trim(
        self,
        history: list[LoopMessage],
        overhead_tokens: int,
    ) -> tuple[list[LoopMessage], int]:
        """Guarantee the threshold even when overhead + summary are large."""
        work = list(history)
        total = overhead_tokens + sum(self._tokens(message) for message in work)

        # 1) Drop the oldest tail messages, keeping pinned items and the summary.
        while total > self.threshold:
            drop_index = next(
                (
                    index
                    for index, message in enumerate(work)
                    if not message.pinned and not message.is_summary
                ),
                None,
            )
            if drop_index is None:
                break
            work.pop(drop_index)
            total = overhead_tokens + sum(self._tokens(message) for message in work)

        # 2) Shrink the summary itself.
        if total > self.threshold and self.summary:
            room = max(0, self.threshold - overhead_tokens - 64)
            if room <= 0:
                self.summary = ""
            else:
                self.summary = self.summary[:room]
            work = [
                replace(message, content=self.summary) if message.is_summary else message
                for message in work
            ]
            total = overhead_tokens + sum(self._tokens(message) for message in work)

        # 3) Last resort: keep only pinned messages, truncated to fit.
        if total > self.threshold:
            pinned = [message for message in work if message.pinned]
            if not pinned:
                return [], overhead_tokens
            room = max(32, self.threshold - overhead_tokens)
            budget_each = max(16, room // len(pinned))
            trimmed = [
                replace(message, content=message.content[: budget_each * 4])
                for message in pinned
            ]
            work = trimmed
            total = overhead_tokens + sum(self._tokens(message) for message in work)
        return work, total


__all__ = ("CompactionReport", "ContextManager", "LoopMessage")
