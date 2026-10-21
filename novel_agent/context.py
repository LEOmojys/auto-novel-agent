"""Layered context assembly for chapter writing and explicit memory lookup.

Retrieval is deterministic: eligible rows are filtered by the memory contract
(layer/status/supersession/provenance hashes), rendered into model-facing text,
ranked by (tier, lexical matches, FTS5/BM25 score, recency), then packed into a
per-category token budget.  No embeddings are involved; BM25 comes from the
`memory_fts` index and is used only as a ranking signal, never as a filter.
"""

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
from novel_agent.memory_text import (
    MIN_FRAGMENT_TOKENS,
    TOKENIZER_VERSION,
    estimate_tokens,
    render_memory_text,
    truncate_text,
)
from novel_agent.pathing import ProjectId


_SOURCE_SPAN_PATTERN: Final = re.compile(r"([0-9]+):([0-9]+)")
_FTS_MIN_TERM_CHARACTERS: Final = 3  # trigram tokenizer matches 3+ characters


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
    artifact_sha256: str | None
    has_successor: bool


@dataclass(frozen=True, slots=True)
class _ArtifactSource:
    status: str
    raw_bytes: bytes
    sha256: str
    digest_valid: bool


class _ArtifactCache:
    """Per-build artifact lookup: one read and one full digest per revision."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._cache: dict[tuple[str, int], _ArtifactSource | None] = {}

    def get(self, artifact_id: str, revision: int) -> _ArtifactSource | None:
        key = (artifact_id, revision)
        if key not in self._cache:
            row = self._connection.execute(
                "SELECT status, raw_bytes, sha256 FROM artifacts WHERE artifact_id = ? AND revision = ?",
                (artifact_id, revision),
            ).fetchone()
            if row is None:
                self._cache[key] = None
            else:
                raw_bytes = bytes(row[1])
                digest = str(row[2])
                self._cache[key] = _ArtifactSource(
                    status=str(row[0]),
                    raw_bytes=raw_bytes,
                    sha256=digest,
                    digest_valid=hashlib.sha256(raw_bytes).hexdigest() == digest,
                )
        return self._cache[key]


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

    def build(self, request: ContextRequest, *, persist: bool = False) -> ContextBuildResult:
        artifacts = _ArtifactCache(self._connection)
        fts_scores = self._fts_scores(request.query_terms)
        candidates: list[ContextItem] = []
        exclusions: list[ContextExclusion] = []
        for row in self._load_rows(request.project_id.root):
            reason = self._exclusion_reason(row, request, artifacts)
            if reason is not None:
                exclusions.append(ContextExclusion(row.memory_id, reason))
                continue
            candidates.append(self._memory_item(row, request, fts_scores.get(row.memory_id, 0.0)))
        candidates.extend(self._style_items(request))
        ordered = sorted(
            candidates,
            key=lambda item: (
                item.tier,
                item.relevance,
                item.fts_score,
                item.narrative_time,
                item.memory_id,
            ),
        )
        selected, budget_state = self._select(ordered, request, exclusions)
        items = tuple(selected)
        prompt = "\n".join(item.content for item in items)
        digest_payload = json.dumps(
            {
                "chapter_plan_id": request.chapter_plan_id,
                "items": [item.memory_id for item in items],
                "plan_revision": request.plan_revision,
                "prompt": prompt,
                "scope": request.scope,
                "tokenizer": TOKENIZER_VERSION,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(digest_payload.encode()).hexdigest()
        pack_id = hashlib.sha256(
            f"{request.run_id}:{request.chapter_plan_id}:{digest}".encode()
        ).hexdigest()[:32]
        if persist:
            self._persist_pack(request, pack_id, digest, items, exclusions, budget_state)
        return ContextBuildResult(
            items=items,
            exclusions=tuple(sorted(exclusions, key=lambda item: (item.memory_id, item.reason))),
            prompt=prompt,
            digest=digest,
            tokenizer_version=TOKENIZER_VERSION,
            budget_ledger=BudgetLedger(
                budget_state.total,
                budget_state.canonical,
                budget_state.plan,
                budget_state.continuity,
                budget_state.long_term,
                budget_state.style,
            ),
            pack_id=pack_id if persist else None,
        )

    # -- selection ---------------------------------------------------------

    @staticmethod
    def _select(
        ordered: list[ContextItem],
        request: ContextRequest,
        exclusions: list[ContextExclusion],
    ) -> tuple[list[ContextItem], _BudgetState]:
        limits = {
            "canonical": request.budget.canonical_tokens,
            "plan": request.budget.plan_tokens,
            "continuity": request.budget.continuity_tokens,
            "long_term": request.budget.long_term_tokens,
            "style": request.budget.style_tokens,
        }
        selected: list[ContextItem] = []
        state = _BudgetState()
        for item in ordered:
            category = item.category
            remaining_total = request.budget.total_tokens - state.total
            remaining_category = limits[category] - state.used(category)
            if remaining_total <= 0:
                exclusions.append(ContextExclusion(item.memory_id, "total_budget_exceeded"))
                continue
            if remaining_category <= 0:
                exclusions.append(ContextExclusion(item.memory_id, f"{category}_budget_exceeded"))
                continue
            room = min(remaining_total, remaining_category)
            if item.token_count <= room:
                selected.append(item)
                state = state.add(category, item.token_count)
                continue
            # A single oversized fact must not erase its whole category: when
            # nothing was selected yet, keep a truncated fragment instead.
            if state.used(category) == 0 and room >= MIN_FRAGMENT_TOKENS:
                fragment = truncate_text(item.content, room)
                if fragment:
                    selected.append(
                        ContextItem(
                            memory_id=item.memory_id,
                            content=fragment,
                            token_count=estimate_tokens(fragment),
                            category=category,
                            tier=item.tier,
                            relevance=item.relevance,
                            narrative_time=item.narrative_time,
                            source_artifact_id=item.source_artifact_id,
                            source_artifact_revision=item.source_artifact_revision,
                            artifact_hash=item.artifact_hash,
                            source_hash=item.source_hash,
                            fts_score=item.fts_score,
                            truncated=True,
                        )
                    )
                    state = state.add(category, estimate_tokens(fragment))
                    continue
            reason = (
                "total_budget_exceeded"
                if remaining_total <= remaining_category
                else f"{category}_budget_exceeded"
            )
            exclusions.append(ContextExclusion(item.memory_id, reason))
        return selected, state

    # -- loading -----------------------------------------------------------

    def _load_rows(self, project_id: str) -> tuple[_MemoryRow, ...]:
        rows = self._connection.execute(
            "SELECT m.memory_id, m.plan_revision, m.layer, m.kind, m.status, m.subject_id, "
            "m.predicate, m.value_json, m.chapter_start, m.source_artifact_id, "
            "m.source_artifact_revision, m.source_span, m.source_hash, a.status, a.sha256, "
            "EXISTS(SELECT 1 FROM memory_records s WHERE s.supersedes_id = m.memory_id) "
            "FROM memory_records m LEFT JOIN artifacts a ON a.artifact_id = m.source_artifact_id "
            "AND a.revision = m.source_artifact_revision WHERE m.project_id = ? ORDER BY m.memory_id",
            (project_id,),
        ).fetchall()
        return tuple(
            _MemoryRow(
                memory_id=str(row[0]),
                plan_revision=int(row[1]),
                layer=str(row[2]),
                kind=str(row[3]),
                status=str(row[4]),
                subject_id=str(row[5]),
                predicate=str(row[6]),
                content=str(row[7]),
                chapter_start=None if row[8] is None else int(row[8]),
                source_artifact_id=None if row[9] is None else str(row[9]),
                source_artifact_revision=None if row[10] is None else int(row[10]),
                source_span=None if row[11] is None else str(row[11]),
                source_hash=None if row[12] is None else str(row[12]),
                artifact_status=None if row[13] is None else str(row[13]),
                artifact_sha256=None if row[14] is None else str(row[14]),
                has_successor=bool(row[15]),
            )
            for row in rows
        )

    # -- eligibility -------------------------------------------------------

    @staticmethod
    def _exclusion_reason(
        row: _MemoryRow,
        request: ContextRequest,
        artifacts: _ArtifactCache,
    ) -> str | None:
        eligible = {
            "m1_canonical": ("canonical", "validated"),
            "m2_plan": ("validated",),
            "m3_confirmed": ("confirmed",),
        }
        if row.layer not in eligible:
            return "disallowed_layer"
        if row.status not in eligible[row.layer]:
            return "status_not_eligible"
        if row.has_successor:
            return "superseded"
        if request.scope == "chapter":
            if row.layer == "m2_plan" and row.plan_revision != request.plan_revision:
                return "stale_plan_revision"
            if (
                row.layer == "m3_confirmed"
                and row.chapter_start is not None
                and row.chapter_start > request.chapter_number
            ):
                return "future_memory"
        if row.kind.startswith("quality") or row.kind in {"review", "critique", "opinion"}:
            return "quality_opinion"
        if (
            row.source_artifact_id is None
            or row.source_artifact_revision is None
            or row.source_span is None
            or row.source_hash is None
        ):
            return "missing_provenance"
        source = artifacts.get(row.source_artifact_id, row.source_artifact_revision)
        if source is None:
            return "source_artifact_not_accepted"
        if source.status != "accepted":
            return "source_artifact_not_accepted"
        if not source.digest_valid:
            return "source_artifact_hash_mismatch"
        span = _SOURCE_SPAN_PATTERN.fullmatch(row.source_span)
        if span is None:
            return "source_span_invalid"
        start, end = (int(value) for value in span.groups())
        if start >= end or end > len(source.raw_bytes):
            return "source_span_invalid"
        if hashlib.sha256(source.raw_bytes[start:end]).hexdigest() != row.source_hash:
            return "source_span_hash_mismatch"
        return None

    # -- ranking -----------------------------------------------------------

    @staticmethod
    def _memory_item(row: _MemoryRow, request: ContextRequest, fts_score: float) -> ContextItem:
        direct_subjects = frozenset(
            (*request.entity_ids, *request.hook_ids, *request.location_ids, request.chapter_plan_id)
        )
        direct = row.memory_id in request.direct_memory_ids or row.subject_id in direct_subjects
        content = render_memory_text(row.layer, row.kind, row.content)
        text = " ".join((row.subject_id, row.predicate, row.kind, content)).casefold()
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
            memory_id=row.memory_id,
            content=content,
            token_count=estimate_tokens(content),
            category=category,
            tier=tier,
            relevance=-matches,
            narrative_time=narrative_time,
            source_artifact_id=row.source_artifact_id or "",
            source_artifact_revision=row.source_artifact_revision or 0,
            artifact_hash=row.artifact_sha256 or "",
            source_hash=row.source_hash or "",
            fts_score=fts_score,
        )

    def _fts_scores(self, query_terms: tuple[str, ...]) -> dict[str, float]:
        usable = [term.strip() for term in query_terms if len(term.strip()) >= _FTS_MIN_TERM_CHARACTERS]
        if not usable:
            return {}
        match = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in usable)
        try:
            rows = self._connection.execute(
                "SELECT memory_id, bm25(memory_fts) FROM memory_fts WHERE memory_fts MATCH ?",
                (match,),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        scores: dict[str, float] = {}
        for memory_id, score in rows:
            identifier = str(memory_id)
            value = float(score)
            if identifier not in scores or value < scores[identifier]:
                scores[identifier] = value
        return scores

    @staticmethod
    def _style_items(request: ContextRequest) -> tuple[ContextItem, ...]:
        items: list[ContextItem] = []
        for index, guidance in enumerate(request.style_guidance):
            source_hash = hashlib.sha256(guidance.encode()).hexdigest()
            items.append(
                ContextItem(
                    memory_id=f"style:{index:04d}:{source_hash}",
                    content=guidance,
                    token_count=estimate_tokens(guidance),
                    category="style",
                    tier=3,
                    relevance=0,
                    narrative_time=0,
                    source_artifact_id=f"request:{request.run_id}",
                    source_artifact_revision=request.plan_revision,
                    artifact_hash=source_hash,
                    source_hash=source_hash,
                )
            )
        return tuple(items)

    # -- persistence -------------------------------------------------------

    def _persist_pack(
        self,
        request: ContextRequest,
        pack_id: str,
        digest: str,
        items: tuple[ContextItem, ...],
        exclusions: list[ContextExclusion],
        state: _BudgetState,
    ) -> None:
        payload = {
            "items": [
                {
                    "memory_id": item.memory_id,
                    "category": item.category,
                    "tier": item.tier,
                    "tokens": item.token_count,
                    "truncated": item.truncated,
                }
                for item in items
            ],
            "exclusions": [
                {"memory_id": exclusion.memory_id, "reason": exclusion.reason}
                for exclusion in sorted(exclusions, key=lambda entry: (entry.memory_id, entry.reason))
            ],
            "ledger": {
                "total": state.total,
                "canonical": state.canonical,
                "plan": state.plan,
                "continuity": state.continuity,
                "long_term": state.long_term,
                "style": state.style,
            },
        }
        try:
            self._connection.execute(
                "INSERT INTO context_packs(pack_id, project_id, run_id, chapter_plan_id, digest, "
                "tokenizer_version, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(pack_id) DO NOTHING",
                (
                    pack_id,
                    request.project_id.root,
                    request.run_id,
                    request.chapter_plan_id,
                    digest,
                    TOKENIZER_VERSION,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    self._now_iso(),
                ),
            )
        except sqlite3.OperationalError:
            # A database created before the context_packs migration must keep
            # working; pack persistence is audit-only.
            return

    @staticmethod
    def _now_iso() -> str:
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat()


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
    chapter_number: int | None = None,
    plan_revision: int | None = None,
) -> tuple[MemoryHit, ...]:
    """Explicit memory lookup for the agent-loop `mem_query` tool.

    Unlike chapter assembly this is a *project-scope* lookup: it may read any
    confirmed chapter and any plan revision, because the caller is explicitly
    asking for project knowledge rather than writing a specific chapter.
    """
    builder = ContextBuilder(connection)
    artifacts = _ArtifactCache(connection)
    fts_scores = builder._fts_scores(query_terms)
    request = ContextRequest(
        project_id=ProjectId(project_id),
        run_id="mem-query",
        plan_revision=0 if plan_revision is None else plan_revision,
        chapter_plan_id="mem-query",
        chapter_number=1 if chapter_number is None else chapter_number,
        query_terms=query_terms,
        scope="project",
    )
    hits: list[MemoryHit] = []
    for row in builder._load_rows(project_id):
        if builder._exclusion_reason(row, request, artifacts) is not None:
            continue
        content = render_memory_text(row.layer, row.kind, row.content)
        text = " ".join((row.subject_id, row.predicate, row.kind, content)).casefold()
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
                content=content,
                source_artifact_id=row.source_artifact_id or "",
            )
        )
    hits.sort(
        key=lambda hit: (
            _query_relevance(hit, query_terms),
            -fts_scores.get(hit.memory_id, 0.0),
        ),
        reverse=True,
    )
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
    "MemoryHit",
    "TOKENIZER_VERSION",
    "memory_query",
)
