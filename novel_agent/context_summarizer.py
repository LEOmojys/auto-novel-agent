"""Model-generated context summaries for loop compaction.

Claude Code / Codex summarise the compacted head with the model instead of a
mechanical list of tool names.  This module implements that path with explicit
guards:

* the transcript is compacted line-by-line and capped before it reaches the
  summariser, so the summary call cannot itself overflow the window;
* the prompt carries the previous rolling summary so information is merged,
  not replaced;
* any provider/parse failure returns None and the caller falls back to the
  deterministic summariser (a failed summarisation must never lose the run).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from novel_agent.contracts import Provenance, ProvenanceKind, ProviderRequest
from novel_agent.loop_context import LoopMessage
from novel_agent.provider_types import ProviderTransport, ProviderTransportError


_SUMMARY_PROMPT: str = """你是长对话的上下文压缩器。请把下面的早期对话压缩成一份可以继续工作的摘要，供后续决策使用。

要求：
- 保留原始目标、已完成的关键步骤、关键产物（artifact 标识）、重要约束/设定、失败与修复、尚未完成的计划。
- 工具调用要保留能区分它们的参数（例如章节号、工具名）。
- 不要编造未出现的事实；不要输出与摘要无关的解释。
- 使用简洁的 Markdown 小节：## 目标、## 已完成、## 关键产物、## 失败与修复、## 待办。

## 上一次摘要
{previous}

## 需要合并的早期对话
{transcript}
"""


@dataclass(frozen=True, slots=True)
class ProviderContextSummarizer:
    """One bounded provider call that turns compacted messages into a summary."""

    transport: ProviderTransport
    now: Callable[[], datetime]
    run_id: str
    model_name: str = "deepseek-v4-flash"
    prompt: str | None = None
    failure_sink: Callable[[str, str, str, str, str], None] | None = None
    call_event_sink: Callable[[str, str, str], None] | None = None
    input_max_chars: int = 12_000
    output_max_chars: int = 6_000

    def summarize(self, messages: Sequence[LoopMessage], previous_summary: str) -> str | None:
        transcript = self._transcript(messages)
        if not transcript.strip():
            return None
        previous = previous_summary.strip()
        if len(previous) > self.input_max_chars // 2:
            previous = previous[-(self.input_max_chars // 2) :]
        prompt = (self.prompt or _SUMMARY_PROMPT).format(
            previous=previous or "（无）",
            transcript=transcript,
        )
        timestamp = self.now()
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        request = ProviderRequest(
            provider_call_id=f"{self.run_id}:summary:{prompt_sha256[:12]}",
            run_id=self.run_id,
            task_id=f"{self.run_id}:summary",
            model_name=self.model_name,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            temperature=0.2,
            created_at=timestamp,
            provenance=Provenance(
                source_kind=ProvenanceKind.USER_REQUEST,
                source_id=f"{self.run_id}:summary",
                source_sha256=prompt_sha256,
                observed_at=timestamp,
            ),
        )
        try:
            if self.call_event_sink is not None:
                self.call_event_sink("summary", request.model_name, "started")
            content = self.transport.generate(request).content
            if self.call_event_sink is not None:
                self.call_event_sink("summary", request.model_name, "finished")
        except ProviderTransportError as error:
            self._record_failure(request, "transport", str(error))
            return None
        except Exception as error:  # noqa: BROAD_EXCEPT_OK - fall back to deterministic summary
            self._record_failure(request, "unexpected", f"{type(error).__name__}: {error}")
            return None
        summary = _clean_summary(content)
        if not summary:
            self._record_failure(request, "empty", "summary was empty")
            return None
        if len(summary) > self.output_max_chars:
            summary = summary[: self.output_max_chars].rstrip() + "…"
        return summary

    def _transcript(self, messages: Sequence[LoopMessage]) -> str:
        lines: list[str] = []
        for message in messages:
            if message.is_summary:
                continue
            if message.role == "tool":
                args = self._compact_args(message.tool_args)
                outcome = "ok" if message.tool_ok else "failed"
                line = f"[tool] {message.tool_name}{args} → {outcome}"
                if message.artifact_ref:
                    line += f" artifact={message.artifact_ref}"
                content = " ".join(message.content.split())
                if content:
                    line += " :: " + content[:600]
                lines.append(line)
            elif message.role == "assistant":
                text = " ".join(message.content.split())
                if text:
                    lines.append("[assistant] " + text[:300])
            else:
                text = " ".join(message.content.split())
                if text:
                    lines.append("[user] " + text[:300])
        transcript = "\n".join(lines)
        if len(transcript) <= self.input_max_chars:
            return transcript
        half = max(100, self.input_max_chars // 2)
        return transcript[:half] + "\n…（中间省略）…\n" + transcript[-half:]

    @staticmethod
    def _compact_args(args: dict | None) -> str:
        if not args:
            return "()"
        try:
            text = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            text = str(args)
        if len(text) > 120:
            text = text[:120] + "…"
        return f"({text})"

    def _record_failure(self, request: ProviderRequest, kind: str, reason: str) -> None:
        if self.call_event_sink is not None:
            self.call_event_sink("summary", request.model_name, "finished")
        if self.failure_sink is not None:
            self.failure_sink(
                "context_summary",
                request.model_name,
                kind,
                reason[:400],
                request.prompt_sha256,
            )


def _clean_summary(content: str) -> str:
    text = content.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        newline = text.find("\n")
        text = text[newline + 1 :] if newline >= 0 else text[len(fence) :]
    if text.endswith(fence):
        text = text[: -len(fence)]
    return text.strip()


__all__ = ("ProviderContextSummarizer",)
