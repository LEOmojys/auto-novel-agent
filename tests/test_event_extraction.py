from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

import pytest

import config
from novel_agent.adapters import GovernedCreativeAdapter
from novel_agent.chapter_models import ChapterDraftRequest, MemoryExtractionRequest, StagedChapter
from novel_agent.contracts import ChapterPlan, EvidenceRef, Provenance, ProvenanceKind, StoryBrief
from novel_agent.context_models import BudgetLedger, ContextBuildResult
from novel_agent.memory_models import MemoryFactKind
from novel_agent.pathing import ArtifactId, ProjectId
from novel_agent.provider_types import ProviderTransportResult
from novel_agent.service import NovelAgentService, StartNovelRunRequest
from novel_agent.storage import connect_project_database


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")


class EventTransport:
    def __init__(self, summary: str, events: list[dict]) -> None:
        self.payload = {"summary": summary, "events": events}

    def generate(self, _request) -> ProviderTransportResult:
        return ProviderTransportResult(
            content=json.dumps(self.payload, ensure_ascii=False, separators=(",", ":")),
            model_name="t",
            prompt_tokens=1,
            completion_tokens=1,
        )


def _adapter(transport, names: tuple[str, ...] = ("林晚", "阿泽")) -> GovernedCreativeAdapter:
    holder: dict[str, object] = {"names": names}
    return GovernedCreativeAdapter(
        transport=transport,
        now=lambda: FIXED_TIME,
        run_id="run-001",
        project_id=PROJECT_ID,
        character_names_holder=holder,
    )


def _chapter_plan(number: int) -> ChapterPlan:
    return ChapterPlan(
        chapter_plan_id=f"run-001:brief:chapter:{number}",
        chapter_number=number,
        title="t",
        goal="g",
        planned_events=("e",),
        created_at=FIXED_TIME,
        provenance={
            "source_kind": "provider_response",
            "source_id": f"run-001:brief:chapter:{number}",
            "source_sha256": "b" * 64,
            "observed_at": FIXED_TIME.isoformat(),
        },
        evidence=(
            EvidenceRef(
                evidence_id="ev",
                artifact_id="w",
                artifact_revision=1,
                source_sha256="c" * 64,
                recorded_at=FIXED_TIME,
            ),
        ),
    )


def _extraction_request(chapter: ChapterPlan) -> MemoryExtractionRequest:
    staged = StagedChapter(
        artifact_id=ArtifactId("chapter-001"),
        revision=1,
        task_id="run-001:chapter:t",
        chapter_plan_id=chapter.chapter_plan_id,
        chapter_number=chapter.chapter_number,
        raw_bytes="林晚在教室遇见了阿泽。".encode(),
        raw_sha256="e" * 64,
        body="林晚在教室遇见了阿泽。",
    )
    return MemoryExtractionRequest(run_id="run-001", plan_revision=1, chapter_plan=chapter, staged=staged)


def test_extract_emits_summary_plus_direct_events() -> None:
    adapter = _adapter(
        EventTransport(
            "摘要内容",
            [
                {"subject": "林晚", "kind": "direct_event", "predicate": "在教室遇见阿泽", "object": "阿泽"},
                {"subject": "阿泽", "kind": "direct_event", "predicate": "约林晚放学后见面", "object": "林晚"},
            ],
        )
    )
    chapter = _chapter_plan(1)

    proposals = adapter.extract(_extraction_request(chapter))

    assert len(proposals) == 3
    kinds = [proposal.kind for proposal in proposals]
    assert kinds[0] is MemoryFactKind.CHAPTER_SUMMARY
    assert kinds[1:] == [MemoryFactKind.DIRECT_EVENT, MemoryFactKind.DIRECT_EVENT]
    first_event = proposals[1]
    assert first_event.subject_id == "林晚"
    assert first_event.predicate == "direct_event:chapter_1:1"
    assert json.loads(first_event.value_json) == {"event": "在教室遇见阿泽", "object": "阿泽", "chapter": 1}
    assert first_event.source_artifact_id == ArtifactId("chapter-001")
    assert first_event.plan_revision == 1


def test_extract_drops_events_with_unresolved_subjects() -> None:
    adapter = _adapter(
        EventTransport(
            "摘要",
            [
                {"subject": "神秘人", "kind": "direct_event", "predicate": "出现在走廊", "object": ""},
                {"subject": "林晚", "kind": "direct_event", "predicate": "察觉异常", "object": ""},
            ],
        ),
        names=("林晚", "阿泽"),
    )

    proposals = adapter.extract(_extraction_request(_chapter_plan(1)))

    # The unresolved "神秘人" event never enters canonical memory.
    assert len(proposals) == 2  # summary + 林晚 event
    assert proposals[1].subject_id == "林晚"
    assert proposals[1].predicate == "direct_event:chapter_1:2"  # original index retained


def test_extract_without_character_roster_emits_summary_only() -> None:
    adapter = _adapter(EventTransport("摘要", [{"subject": "林晚", "kind": "direct_event", "predicate": "行动", "object": ""}]))
    adapter = GovernedCreativeAdapter(
        transport=adapter.transport,
        now=lambda: FIXED_TIME,
        run_id="run-001",
        project_id=PROJECT_ID,
    )

    proposals = adapter.extract(_extraction_request(_chapter_plan(1)))

    assert len(proposals) == 1
    assert proposals[0].kind is MemoryFactKind.CHAPTER_SUMMARY


class E2ETransport:
    """Dispatch transport with real memory-extraction events."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request) -> ProviderTransportResult:
        self.calls += 1
        prompt = request.prompt
        if "需求分析器" in prompt:
            payload = {
                "domain": "novel",
                "intent": "full_novel",
                "goal": "写一本两章的小说",
                "constraints": [],
                "style_guidance": [],
                "acceptance_criteria": [],
                "chapter_count": 2,
            }
        elif "资深世界观架构师" in prompt:
            payload = {"title": "晨光学院", "premise": "魔法学院。", "rules": ["魔力来自星辰"], "constraints": []}
        elif "资深角色设计师" in prompt:
            payload = {
                "characters": [
                    {"name": "林晚", "role": "主角", "goal": "查明真相", "traits": ["冷静"]},
                    {"name": "阿泽", "role": "挚友", "goal": "守护", "traits": ["热情"]},
                ]
            }
        elif "剧情架构师" in prompt:
            payload = {
                "chapters": [
                    {"title": f"第{n}章", "goal": f"目标{n}", "planned_events": [f"事件{n}"]}
                    for n in range(1, 3)
                ]
            }
        elif "苛刻的小说评审编辑" in prompt:
            payload = {
                "style_consistency": {"score": 4, "reason": "r"},
                "character_consistency": {"score": 4, "reason": "r"},
                "prose_fluency": {"score": 4, "reason": "r"},
                "scene_completeness": {"score": 4, "reason": "r"},
                "dialogue_naturalness": {"score": 4, "reason": "r"},
                "content_alignment": {"score": 4, "reason": "r"},
            }
        elif "摘要" in prompt:
            payload = {
                "summary": "主角抵达学院。",
                "events": [
                    {"subject": "林晚", "kind": "direct_event", "predicate": "抵达魔法学院", "object": "学院"},
                    {"subject": "阿泽", "kind": "direct_event", "predicate": "迎接林晚", "object": "林晚"},
                ],
            }
        else:
            return ProviderTransportResult(
                content="我站在学院门口，晨风拂过衣角。" * 200,
                model_name="t",
                prompt_tokens=1,
                completion_tokens=1,
            )
        return ProviderTransportResult(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            model_name="t",
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_legacy_real_run_persists_direct_event_memories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    projects_root = tmp_path / "projects"
    project_id = ProjectId("novel-one")
    (projects_root / project_id.root).mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=E2ETransport())
    service.start(
        StartNovelRunRequest(
            project_id=project_id,
            run_id="run-events",
            idempotency_key="start-events",
            user_requirements="写一本两章的小说",
            provider="legacy_real",
            target_chapter_count=2,
        )
    )
    deadline = monotonic() + 5.0
    terminal = frozenset(("succeeded", "failed", "waiting_review", "cancelled"))
    from novel_agent.contracts import RunStatus

    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snapshot := service.snapshot(project_id, "run-events")).status not in terminal:
        assert monotonic() < deadline, f"run did not terminate; status={snapshot.status}"
        sleep(0.01)
    assert snapshot.status is RunStatus.SUCCEEDED

    with connect_project_database(projects_root / project_id.root) as connection:
        kinds = [
            str(row[0])
            for row in connection.execute(
                "SELECT kind FROM memory_records WHERE run_id = ? AND layer = 'm3_confirmed' ORDER BY memory_id",
                ("run-events",),
            )
        ]
        assert kinds.count("chapter_summary") == 2
        assert kinds.count("direct_event") == 4  # two events per chapter
        fts = connection.execute("SELECT COUNT(*) FROM memory_fts").fetchone()
        assert fts == (len(kinds) + 5,)  # events + summaries + M1 world/cast + M2 plans
        events = connection.execute(
            "SELECT subject_id, predicate FROM memory_records WHERE run_id = ? AND kind = 'direct_event'",
            ("run-events",),
        ).fetchall()
        predicates = [str(row[1]) for row in events]
        assert len(predicates) == len(set(predicates))  # unique per chapter/index
