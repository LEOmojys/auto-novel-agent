"""Provider-backed continuity auditor for chapter execution.

The governed chapter pipeline needs an independent check that the drafted
chapter actually covers the planned beats and does not contradict the context
pack.  Before this module the only implementation was
`OfflineContinuityAuditor`, which claimed every planned beat was present and
never reported a conflict — silently disabling both the continuity pause and
the `missing_plan_beat` quality rule.

`ProviderContinuityAuditor` asks the provider for a structured report:

    {"located_beats": ["..."], "hard_conflicts": ["..."]}

Unknown beat strings are dropped (the report cannot invent planned events).
A provider/parse failure is reported as a hard conflict so the chapter pauses
for review instead of being accepted with a fabricated "no conflicts" result.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from novel_agent.contracts import Provenance, ProvenanceKind, ProviderRequest
from novel_agent.chapter_models import ContinuityAuditReport, ContinuityAuditRequest
from novel_agent.provider_types import ProviderTransport, ProviderTransportError


_CONTINUITY_PROMPT: str = """你是小说连续性审稿人。请核对本章草稿是否落实了计划事件，并找出与既有设定的硬冲突。

## 本章计划事件（只能从这些事件中选择已落实项）
{beats}

## 既有设定与上下文
{context}

## 本章正文
{body}

请输出内容字段 JSON（不要输出其他文字）：
{{
  "located_beats": ["正文中确实落实的计划事件（原文照抄计划事件）"],
  "hard_conflicts": ["与设定冲突的事实（没有则空数组）"]
}}
只输出 JSON。"""


class _ContinuityContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    located_beats: tuple[str, ...] = Field(default_factory=tuple)
    hard_conflicts: tuple[str, ...] = Field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ProviderContinuityAuditor:
    """A `ContinuityAuditor` backed by a live provider call."""

    transport: ProviderTransport
    now: Callable[[], datetime]
    run_id: str
    model_name: str = "deepseek-v4-flash"
    prompt: str | None = None
    failure_sink: Callable[[str, str, str, str, str], None] | None = None
    call_event_sink: Callable[[str, str, str], None] | None = None

    def audit(self, request: ContinuityAuditRequest) -> ContinuityAuditReport:
        planned = tuple(request.chapter_plan.planned_events)
        prompt = (self.prompt or _CONTINUITY_PROMPT).format(
            beats="\n".join(f"- {beat}" for beat in planned) or "（本章无计划事件）",
            context=request.context.prompt or "（无额外设定）",
            body=request.staged.body[:8_000],
        )
        timestamp = self.now()
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        provider_request = ProviderRequest(
            provider_call_id=f"{self.run_id}:continuity:{request.chapter_plan.chapter_plan_id}:{prompt_sha256[:12]}",
            run_id=self.run_id,
            task_id=f"{self.run_id}:continuity:{request.chapter_plan.chapter_plan_id}",
            model_name=self.model_name,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            temperature=0.1,
            created_at=timestamp,
            provenance=Provenance(
                source_kind=ProvenanceKind.USER_REQUEST,
                source_id=f"{self.run_id}:continuity:{request.chapter_plan.chapter_plan_id}",
                source_sha256=prompt_sha256,
                observed_at=timestamp,
            ),
        )
        try:
            if self.call_event_sink is not None:
                self.call_event_sink("continuity", provider_request.model_name, "started")
            raw = self.transport.generate(provider_request).content
            if self.call_event_sink is not None:
                self.call_event_sink("continuity", provider_request.model_name, "finished")
        except ProviderTransportError as error:
            self._record_failure(provider_request, "transport", str(error))
            return ContinuityAuditReport(hard_conflicts=("continuity_auditor_error",))
        except Exception as error:  # noqa: BROAD_EXCEPT_OK - never fabricate a "no conflict" result
            self._record_failure(provider_request, "unexpected", f"{type(error).__name__}: {error}")
            return ContinuityAuditReport(hard_conflicts=("continuity_auditor_error",))
        try:
            parsed = _ContinuityContent.model_validate_json(raw)
        except ValidationError:
            self._record_failure(provider_request, "parse", raw[:400])
            return ContinuityAuditReport(hard_conflicts=("continuity_auditor_error",))
        allowed = frozenset(planned)
        located = tuple(beat for beat in parsed.located_beats if beat in allowed)
        conflicts = tuple(conflict.strip() for conflict in parsed.hard_conflicts if conflict.strip())
        return ContinuityAuditReport(located_beats=located, hard_conflicts=conflicts)

    def _record_failure(self, request: ProviderRequest, kind: str, reason: str) -> None:
        if self.call_event_sink is not None:
            self.call_event_sink("continuity", request.model_name, "finished")
        if self.failure_sink is not None:
            self.failure_sink(
                "continuity",
                request.model_name,
                kind,
                reason[:400],
                request.prompt_sha256,
            )


__all__ = ("ProviderContinuityAuditor",)
