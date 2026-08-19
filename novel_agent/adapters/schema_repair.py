"""Provider-backed schema repairer for the structured creative adapter.

Implements the third rung of the creative parse ladder (strict JSON ->
bounded json_repair -> one directed schema-repair prompt) described by the
work plan.  A malformed model response is sent back to the provider once
with the expected contract schema; the repairer returns the repaired text
or an empty string when the provider is unavailable, so the caller falls
through to a typed ``SchemaFailure`` instead of inventing content.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from novel_agent.adapters.structured import SchemaRepairRequest, SchemaRepairer
from novel_agent.contracts import Provenance, ProvenanceKind, ProviderRequest
from novel_agent.provider_types import ProviderTransport, ProviderTransportError

_REPAIR_PROMPT: Final[str] = """你是一个结构化 JSON 修复器。下面的文本试图表示一个契约对象，但格式或结构不合法。

## 期望的 JSON Schema
{schema}

## 需要修复的文本
{raw_text}

请输出一个完全符合上述 JSON Schema 的修复后 JSON 对象，不要输出任何其他文字、解释或 markdown 标记。"""


@dataclass(frozen=True, slots=True)
class ProviderSchemaRepairer:
    """A ``SchemaRepairer`` that asks the provider to repair malformed JSON."""

    transport: ProviderTransport
    now: Callable[[], datetime]
    run_id: str
    model_name: str = "deepseek-v4-flash"
    failure_sink: Callable[[str, str, str, str, str], None] | None = None
    call_event_sink: Callable[[str, str, str], None] | None = None

    def repair(self, request: SchemaRepairRequest) -> str:
        prompt = _REPAIR_PROMPT.format(
            schema=request.contract_schema_json,
            raw_text=request.raw_artifact.raw_bytes.decode(errors="replace"),
        )
        timestamp = self.now()
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        provider_request = ProviderRequest(
            provider_call_id=f"{self.run_id}:schema-repair:{prompt_sha256[:12]}",
            run_id=self.run_id,
            task_id=f"{self.run_id}:schema-repair",
            model_name=self.model_name,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            temperature=0.0,
            created_at=timestamp,
            provenance=Provenance(
                source_kind=ProvenanceKind.USER_REQUEST,
                source_id=f"{self.run_id}:schema-repair",
                source_sha256=prompt_sha256,
                observed_at=timestamp,
            ),
        )
        try:
            if self.call_event_sink is not None:
                self.call_event_sink("schema_repair", provider_request.model_name, "started")
            content = self.transport.generate(provider_request).content
            if self.call_event_sink is not None:
                self.call_event_sink("schema_repair", provider_request.model_name, "finished")
            return content
        except ProviderTransportError as error:
            if self.call_event_sink is not None:
                self.call_event_sink("schema_repair", provider_request.model_name, "finished")
            if self.failure_sink is not None:
                self.failure_sink(
                    caller="schema_repair",
                    model_name=provider_request.model_name,
                    failure_kind=error.kind.value if hasattr(error, "kind") else type(error).__name__,
                    reason=f"transport: {str(error)[:400]}",
                    prompt_sha256=provider_request.prompt_sha256,
                )
            return ""


__all__ = ("ProviderSchemaRepairer",)
