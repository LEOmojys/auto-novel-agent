from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Final

from novel_agent.context_models import (
    BudgetLedger,
    ContextBudget,
    ContextBuildResult,
    ContextExclusion,
    ContextItem,
    ContextRequest,
)


TOKENIZER_VERSION: Final = "char-v1"
_SOURCE_SPAN_PATTERN: Final = re.compile(r"([0-9]+):([0-9]+)")


@dataclass(frozen=True, slots=True)
class _MemoryRow:
    memory_id: str
    plan_revision: int
    layer: str
    kind: str
    status: str
    subject_id: str
    predicate: str
    content: str
    chapter_start: int | None
    source_artifact_id: str | None
    source_artifact_revision: int | None
    source_span: str | None
    source_hash: str | None
    artifact_status: str | None
    artifact_bytes: bytes | None
    artifact_hash: str | None
    has_successor: bool


@dataclass(frozen=True, slots=True)
class _BudgetState:
    total: int = 0
    canonical: int = 0
    plan: int = 0
    continuity: int = 0
    long_term: int = 0
    style: int = 0

    def used(self, category: str) -> int:
        return {
            "canonical": self.canonical,
            "plan": self.plan,
            "continuity": self.continuity,
            "long_term": self.long_term,
            "style": self.style,
        }[category]

    def add(self, category: str, amount: int) -> _BudgetState:
        return _BudgetState(
            total=self.total + amount,
            canonical=self.canonical + (amount if category == "canonical" else 0),
            plan=self.plan + (amount if category == "plan" else 0),
            continuity=self.continuity + (amount if category == "continuity" else 0),
            long_term=self.long_term + (amount if category == "long_term" else 0),
            style=self.style + (amount if category == "style" else 0),
        )


class ContextBuilder:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def build(self, request: ContextRequest) -> ContextBuildResult:
        candidates: list[ContextItem] = []
        exclusions: list[ContextExclusion] = []
        for row in self._load_rows(request.project_id.root):
            reason = self._exclusion_reason(row, request)
            if reason is not None:
                exclusions.append(ContextExclusion(row.memory_id, reason))
                continue
            candidates.append(self._memory_item(row, request))
        candidates.extend(self._style_items(request))
        ordered = sorted(
            candidates,
            key=lambda item: (item.tier, item.relevance, item.narrative_time, item.memory_id),
        )
        selected: list[ContextItem] = []
        state = _BudgetState()
        category_limits = {
            "canonical": request.budget.canonical_tokens,
            "plan": request.budget.plan_tokens,
            "continuity": request.budget.continuity_tokens,
            "long_term": request.budget.long_term_tokens,
            "style": request.budget.style_tokens,
        }
        for item in ordered:
            limit = category_limits[item.category]
            if state.total + item.token_count > request.budget.total_tokens:
                exclusions.append(ContextExclusion(item.memory_id, "total_budget_exceeded"))
            elif state.used(item.category) + item.token_count > limit:
                exclusions.append(ContextExclusion(item.memory_id, f"{item.category}_budget_exceeded"))
            else:
                selected.append(item)
                state = state.add(item.category, item.token_count)
        items = tuple(selected)
        prompt = "\n".join(item.content for item in items)
        digest_payload = json.dumps(
            {
                "chapter_plan_id": request.chapter_plan_id,
                "items": [item.memory_id for item in items],
                "plan_revision": request.plan_revision,
                "prompt": prompt,
                "tokenizer": TOKENIZER_VERSION,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return ContextBuildResult(
            items=items,
            exclusions=tuple(sorted(exclusions, key=lambda item: (item.memory_id, item.reason))),
            prompt=prompt,
            digest=hashlib.sha256(digest_payload.encode()).hexdigest(),
            tokenizer_version=TOKENIZER_VERSION,
            budget_ledger=BudgetLedger(
                state.total,
                state.canonical,
                state.plan,
                state.continuity,
                state.long_term,
                state.style,
            ),
        )

    def _load_rows(self, project_id: str) -> tuple[_MemoryRow, ...]:
        rows = self._connection.execute(
            "SELECT m.memory_id, m.plan_revision, m.layer, m.kind, m.status, m.subject_id, "
            "m.predicate, m.value_json, m.chapter_start, m.source_artifact_id, "
            "m.source_artifact_revision, m.source_span, m.source_hash, a.status, a.raw_bytes, a.sha256, "
            "EXISTS(SELECT 1 FROM memory_records s WHERE s.supersedes_id = m.memory_id) "
            "FROM memory_records m LEFT JOIN artifacts a ON a.artifact_id = m.source_artifact_id "
            "AND a.revision = m.source_artifact_revision WHERE m.project_id = ? ORDER BY m.memory_id",
            (project_id,),
        ).fetchall()
        return tuple(
            _MemoryRow(
                memory_id=str(row[0]), plan_revision=int(row[1]), layer=str(row[2]),
                kind=str(row[3]), status=str(row[4]), subject_id=str(row[5]),
                predicate=str(row[6]), content=str(row[7]),
                chapter_start=None if row[8] is None else int(row[8]),
                source_artifact_id=None if row[9] is None else str(row[9]),
                source_artifact_revision=None if row[10] is None else int(row[10]),
                source_span=None if row[11] is None else str(row[11]),
                source_hash=None if row[12] is None else str(row[12]),
                artifact_status=None if row[13] is None else str(row[13]),
                artifact_bytes=None if row[14] is None else bytes(row[14]),
                artifact_hash=None if row[15] is None else str(row[15]),
                has_successor=bool(row[16]),
            )
            for row in rows
        )

    @staticmethod
    def _exclusion_reason(row: _MemoryRow, request: ContextRequest) -> str | None:
        eligible = {"m1_canonical": "canonical", "m2_plan": "validated", "m3_confirmed": "confirmed"}
        if row.layer not in eligible:
            return "disallowed_layer"
        if row.status != eligible[row.layer]:
            return "status_not_eligible"
        if row.has_successor:
            return "superseded"
        if row.layer == "m2_plan" and row.plan_revision != request.plan_revision:
            return "stale_plan_revision"
        if row.layer == "m3_confirmed" and row.chapter_start is not None and row.chapter_start > request.chapter_number:
            return "future_memory"
        if row.kind.startswith("quality") or row.kind in {"review", "critique", "opinion"}:
            return "quality_opinion"
        if (
            row.source_artifact_id is None
            or row.source_artifact_revision is None
            or row.source_span is None
            or row.source_hash is None
            or row.artifact_bytes is None
            or row.artifact_hash is None
        ):
            return "missing_provenance"
        if row.artifact_status != "accepted":
            return "source_artifact_not_accepted"
        if hashlib.sha256(row.artifact_bytes).hexdigest() != row.artifact_hash:
            return "source_artifact_hash_mismatch"
        span = _SOURCE_SPAN_PATTERN.fullmatch(row.source_span)
        if span is None:
            return "source_span_invalid"
        start, end = (int(value) for value in span.groups())
        if start >= end or end > len(row.artifact_bytes):
            return "source_span_invalid"
        if hashlib.sha256(row.artifact_bytes[start:end]).hexdigest() != row.source_hash:
            return "source_span_hash_mismatch"
        return None

    @staticmethod
    def _memory_item(row: _MemoryRow, request: ContextRequest) -> ContextItem:
        direct_subjects = frozenset((*request.entity_ids, *request.hook_ids, *request.location_ids, request.chapter_plan_id))
        direct = row.memory_id in request.direct_memory_ids or row.subject_id in direct_subjects
        text = " ".join((row.subject_id, row.predicate, row.kind, row.content)).casefold()
        matches = sum(text.count(term.casefold()) for term in request.query_terms if term)
        tier = 0 if direct else (1 if matches else 2)
        category = {"m1_canonical": "canonical", "m2_plan": "plan"}.get(row.layer)
        if category is None:
            is_recent = row.chapter_start is not None and row.chapter_start >= max(
                0,
                request.chapter_number - 2,
            )
            category = "continuity" if is_recent else "long_term"
        narrative_time = -(row.chapter_start or 0)
        return ContextItem(
            row.memory_id, row.content, len(row.content), category, tier, -matches,
            narrative_time, row.source_artifact_id or "", row.source_artifact_revision or 0,
            row.artifact_hash or "", row.source_hash or "",
        )

    @staticmethod
    def _style_items(request: ContextRequest) -> tuple[ContextItem, ...]:
        items: list[ContextItem] = []
        for index, guidance in enumerate(request.style_guidance):
            source_hash = hashlib.sha256(guidance.encode()).hexdigest()
            items.append(ContextItem(
                f"style:{index:04d}:{source_hash}", guidance, len(guidance), "style", 3, 0,
                0, f"request:{request.run_id}", request.plan_revision, source_hash, source_hash,
            ))
        return tuple(items)


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """A memory row returned by :func:`memory_query` (model-facing)."""

    memory_id: str
    subject_id: str
    predicate: str
    kind: str
    content: str
    source_artifact_id: str


def memory_query(
    connection: sqlite3.Connection,
    *,
    project_id: str,
    query_terms: tuple[str, ...],
    limit: int = 5,
) -> tuple[MemoryHit, ...]:
    """Explicit memory lookup for the agent-loop ``mem_query`` tool.

    Reuses the same row loading and eligibility filtering as
    ``ContextBuilder`` (eligibility is project-wide here, no chapter context),
    then ranks by term matches and returns the top ``limit`` hits.
    """
    builder = ContextBuilder(connection)
    rows = builder._load_rows(project_id)
    request = ContextRequest(
        project_id=ProjectId(project_id),
        run_id="mem-query",
        plan_revision=0,
        chapter_plan_id="mem-query",
        chapter_number=0,
        query_terms=query_terms,
    )
    hits: list[MemoryHit] = []
    for row in rows:
        if builder._exclusion_reason(row, request) is not None:
            continue
        text = " ".join((row.subject_id, row.predicate, row.kind, row.content)).casefold()
        matches = sum(text.count(term.casefold()) for term in query_terms if term)
        if matches == 0 and not any(
            term.casefold() in row.subject_id.casefold() for term in query_terms if term
        ):
            continue
        hits.append(
            MemoryHit(
                memory_id=row.memory_id,
                subject_id=row.subject_id,
                predicate=row.predicate,
                kind=row.kind,
                content=row.content,
                source_artifact_id=row.source_artifact_id or "",
            )
        )
    hits.sort(key=lambda hit: _query_relevance(hit, query_terms), reverse=True)
    return tuple(hits[:limit])


def _query_relevance(hit: MemoryHit, query_terms: tuple[str, ...]) -> int:
    text = " ".join((hit.subject_id, hit.predicate, hit.kind, hit.content)).casefold()
    return sum(text.count(term.casefold()) for term in query_terms if term)


__all__ = (
    "BudgetLedger",
    "ContextBudget",
    "ContextBuilder",
    "ContextBuildResult",
    "ContextExclusion",
    "ContextItem",
    "ContextRequest",
    "TOKENIZER_VERSION",
)
