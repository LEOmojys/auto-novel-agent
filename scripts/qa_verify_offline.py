# -*- coding: utf-8 -*-
"""QA hands-on offline verification script (temporary). Deleted after run."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def wait_terminal(service, project_id, run_id, seconds=20.0):
    from novel_agent.contracts import RunStatus

    deadline = time.monotonic() + seconds
    terminal = frozenset((RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_REVIEW, RunStatus.CANCELLED))
    while (snap := service.snapshot(project_id, run_id)).status not in terminal:
        assert time.monotonic() < deadline, f"run {run_id} stuck at {snap.status}"
        time.sleep(0.01)
    return snap


def dump_events(service, project_id, run_id, limit=25):
    events = list(service.events(project_id, run_id))
    return "; ".join(f"{e.event_type}" for e in events[-limit:])


def new_projects_root():
    root = Path(tempfile.mkdtemp(prefix="qa-iso-"))
    projects_root = root / "projects"
    (projects_root / "novel-one").mkdir(parents=True)
    return root, projects_root


LONG_TEXT = "我站在学院门口，晨风拂过衣角，图书馆的钟声在远处响起。" * 120


class PlannerLegacyTransport:
    """legacy_real transport WITH an LLM planner (plan gate fires)."""

    def __init__(self, *, chapter_count=1):
        self.chapter_count = chapter_count
        self.planner_calls = 0

    def generate(self, request):
        from novel_agent.provider_types import ProviderTransportResult

        prompt = request.prompt
        if "需求分析器" in prompt:
            payload = {"domain": "novel", "intent": "full_novel", "goal": "写小说", "constraints": [], "style_guidance": [], "acceptance_criteria": [], "chapter_count": self.chapter_count}
        elif "规划器" in prompt:
            self.planner_calls += 1
            payload = {"steps": [
                {"step_id": 1, "description": "设计世界观", "tool": "world.generate@1.0.0", "goal": "世界观", "parameters": None},
                {"step_id": 2, "description": "设计角色", "tool": "character.generate@1.0.0", "goal": "角色", "parameters": None},
                {"step_id": 3, "description": "生成大纲", "tool": "outline.generate@1.0.0", "goal": "大纲", "parameters": None},
                {"step_id": 4, "description": "写章节", "tool": "chapter.draft@1.0.0", "goal": "章节", "parameters": None},
            ]}
        elif "资深世界观架构师" in prompt:
            payload = {"title": "现代校园", "premise": "一所普通中学。", "rules": ["现实世界规则"], "constraints": []}
        elif "资深角色设计师" in prompt:
            payload = {"characters": [{"name": "林一", "role": "男主角", "goal": "解开误会", "traits": ["温和"]}]}
        elif "剧情架构师" in prompt:
            payload = {"chapters": [{"title": f"第{n}章", "goal": "目标" * 10, "planned_events": ["事件"]} for n in range(1, self.chapter_count + 1)]}
        elif "华语小说家" in prompt:
            return ProviderTransportResult(content=LONG_TEXT, model_name="t", prompt_tokens=1, completion_tokens=1)
        elif "苛刻的小说评审编辑" in prompt:
            payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
        elif "提取章节摘要" in prompt:
            payload = {"summary": "s", "events": []}
        else:
            return ProviderTransportResult(content=LONG_TEXT, model_name="t", prompt_tokens=1, completion_tokens=1)
        return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)


def approve_pending(service, projects_root, project_id, run_id, cycle):
    from novel_agent.service import ApproveArtifactRequest
    from novel_agent.storage import connect_project_database

    with connect_project_database(projects_root / project_id.root) as conn:
        row = conn.execute(
            "SELECT artifact_id, revision FROM artifacts WHERE run_id=? AND status='staged' "
            "ORDER BY created_at DESC LIMIT 1", (run_id,)).fetchone()
    assert row is not None, "no staged artifact"
    service.approve_artifact(ApproveArtifactRequest(
        project_id=project_id, run_id=run_id, artifact_id=str(row[0]),
        revision=int(row[1]), idempotency_key=f"ap-{cycle}"))


def main() -> int:
    from novel_agent.pathing import ProjectId
    from novel_agent.service import (
        NovelAgentService,
        RepairNovelRunRequest,
        StartNovelRunRequest,
    )
    from novel_agent.contracts import RunStatus
    from novel_agent.storage import connect_project_database

    project_id = ProjectId("novel-one")

    def fresh_service(projects_root, transport=None):
        import config
        config.PROJECTS_DIR = str(projects_root)
        return NovelAgentService.from_config(now=lambda: FIXED_TIME, provider_transport=transport)

    # ---- S4: offline_fake e2e auto_approve=True -> SUCCEEDED ----
    try:
        root, projects_root = new_projects_root()
        service = fresh_service(projects_root)
        snap = service.start(StartNovelRunRequest(
            project_id=project_id, run_id="qa-s4", idempotency_key="k-s4",
            user_requirements="写一本两章的奇幻小说", provider="offline_fake",
            target_chapter_count=2, auto_approve=True))
        assert snap.status is RunStatus.QUEUED, snap.status
        final = wait_terminal(service, project_id, "qa-s4")
        assert final.status is RunStatus.SUCCEEDED, f"{final.status} events={dump_events(service, project_id, 'qa-s4')}"
        with connect_project_database(projects_root / project_id.root) as conn:
            kinds = sorted(str(r[0]) for r in conn.execute("SELECT kind FROM artifacts WHERE run_id='qa-s4' AND status='accepted'"))
            ch = conn.execute("SELECT COUNT(*) FROM artifacts WHERE run_id='qa-s4' AND kind='chapter' AND status='accepted'").fetchone()[0]
            ev = [str(r[0]) for r in conn.execute("SELECT event_type FROM run_events WHERE run_id='qa-s4'")]
        assert ch == 2, f"chapters={ch}"
        record("S4 offline_fake auto_approve=True e2e", True,
               f"status={final.status.value} kinds={kinds} chapters={ch} events={len(ev)} run_succeeded={'run_succeeded' in ev}")
        shutil.rmtree(root, ignore_errors=True)
    except Exception as e:
        record("S4 offline_fake auto_approve=True e2e", False, f"{type(e).__name__}: {e}")

    # ---- S5a: offline_fake auto_approve=False -> artifact gate chain -> SUCCEEDED ----
    try:
        root, projects_root = new_projects_root()
        service = fresh_service(projects_root)
        service.start(StartNovelRunRequest(
            project_id=project_id, run_id="qa-s5", idempotency_key="k-s5",
            user_requirements="写一本一章的奇幻小说", provider="offline_fake",
            target_chapter_count=1, auto_approve=False))
        final = wait_terminal(service, project_id, "qa-s5")
        gates = []
        assert final.status is RunStatus.WAITING_REVIEW, f"{final.status} events={dump_events(service, project_id, 'qa-s5')}"
        # offline_fake has no LLM planner: no plan gate; artifact gates directly.
        assert final.review_reason == "artifact_approval_required", final.review_reason
        gates.append(final.review_reason)
        cycle = 0
        while True:
            cycle += 1
            approve_pending(service, projects_root, project_id, "qa-s5", cycle)
            final = wait_terminal(service, project_id, "qa-s5")
            if final.status is RunStatus.SUCCEEDED:
                break
            assert final.status is RunStatus.WAITING_REVIEW, f"{final.status} events={dump_events(service, project_id, 'qa-s5')}"
            gates.append(final.review_reason)
            assert cycle < 10, "too many approval cycles"
        with connect_project_database(projects_root / project_id.root) as conn:
            kinds = sorted(str(r[0]) for r in conn.execute("SELECT kind FROM artifacts WHERE run_id='qa-s5' AND status='accepted'"))
        record("S5a offline_fake approval chain (artifact gates)", True, f"gates={gates} accepted={kinds}")
        shutil.rmtree(root, ignore_errors=True)
    except Exception as e:
        record("S5a offline_fake approval chain (artifact gates)", False, f"{type(e).__name__}: {e}")

    # ---- S5b: legacy_real auto_approve=False -> plan gate -> approve_plan -> artifact gates ----
    try:
        root, projects_root = new_projects_root()
        transport = PlannerLegacyTransport(chapter_count=1)
        service = fresh_service(projects_root, transport=transport)
        service.start(StartNovelRunRequest(
            project_id=project_id, run_id="qa-s5b", idempotency_key="k-s5b",
            user_requirements="设定一个魔法学院世界观，写一章", provider="legacy_real",
            target_chapter_count=1, auto_approve=False))
        final = wait_terminal(service, project_id, "qa-s5b")
        assert final.status is RunStatus.WAITING_REVIEW, f"{final.status} events={dump_events(service, project_id, 'qa-s5b')}"
        assert final.review_reason == "plan_approval_required", final.review_reason
        service.approve_plan(project_id, "qa-s5b", idempotency_key="k-s5b-plan")
        cycle = 0
        gates = ["plan_approval_required"]
        while True:
            cycle += 1
            final = wait_terminal(service, project_id, "qa-s5b")
            if final.status is RunStatus.SUCCEEDED:
                break
            assert final.status is RunStatus.WAITING_REVIEW, f"{final.status} events={dump_events(service, project_id, 'qa-s5b')}"
            assert final.review_reason == "artifact_approval_required", final.review_reason
            gates.append(final.review_reason)
            approve_pending(service, projects_root, project_id, "qa-s5b", cycle)
            assert cycle < 10
        record("S5b legacy_real plan gate + artifact approval chain", True,
               f"gates={gates} planner_calls={transport.planner_calls}")
        shutil.rmtree(root, ignore_errors=True)
    except Exception as e:
        record("S5b legacy_real plan gate + artifact approval chain", False, f"{type(e).__name__}: {e}")

    # ---- S6: plain-text chapter (legacy_real governed adapter) approval + envelope ----
    try:
        root, projects_root = new_projects_root()
        transport = PlannerLegacyTransport(chapter_count=1)
        service = fresh_service(projects_root, transport=transport)
        service.start(StartNovelRunRequest(
            project_id=project_id, run_id="qa-s6", idempotency_key="k-s6",
            user_requirements="写一本一章的小说", provider="legacy_real", auto_approve=False))
        final = wait_terminal(service, project_id, "qa-s6")
        assert final.status is RunStatus.WAITING_REVIEW, f"{final.status} events={dump_events(service, project_id, 'qa-s6')}"
        assert final.review_reason == "plan_approval_required", final.review_reason
        service.approve_plan(project_id, "qa-s6", idempotency_key="k-s6-plan")
        staged_env = None
        cycle = 0
        while True:
            cycle += 1
            final = wait_terminal(service, project_id, "qa-s6")
            if final.status is RunStatus.SUCCEEDED:
                break
            assert final.status is RunStatus.WAITING_REVIEW, f"{final.status} events={dump_events(service, project_id, 'qa-s6')}"
            with connect_project_database(projects_root / project_id.root) as conn:
                row = conn.execute(
                    "SELECT artifact_id, revision, kind, raw_bytes FROM artifacts WHERE run_id='qa-s6' AND status='staged' "
                    "ORDER BY created_at DESC LIMIT 1").fetchone()
            assert row is not None, "no staged artifact"
            if str(row[2]) == "chapter":
                env = json.loads(bytes(row[3]).decode("utf-8"))
                assert isinstance(env, dict) and "body" in env, f"staged chapter not JSON envelope: {bytes(row[3])[:100]!r}"
                assert env["body"] == LONG_TEXT, "envelope body != original prose"
                staged_env = env
            approve_pending(service, projects_root, project_id, "qa-s6", cycle)
            assert cycle < 10
        assert staged_env is not None, "chapter never staged"
        with connect_project_database(projects_root / project_id.root) as conn:
            acc = conn.execute("SELECT raw_bytes FROM artifacts WHERE run_id='qa-s6' AND kind='chapter' AND status='accepted'").fetchone()
        assert acc is not None and LONG_TEXT[:40] in bytes(acc[0]).decode("utf-8", errors="replace"), f"accepted chapter head={bytes(acc[0])[:60]!r}"
        record("S6 plain-text chapter approval + JSON envelope", True,
               f"staged chapter keys={sorted(staged_env.keys())} body_len={len(staged_env['body'])} accepted keeps verbatim prose")
        shutil.rmtree(root, ignore_errors=True)
    except Exception as e:
        record("S6 plain-text chapter approval + JSON envelope", False, f"{type(e).__name__}: {e}")

    # ---- S7: _is_continuation_request boundaries ----
    try:
        from novel_agent.agent.plan_generator import _is_continuation_request
        cases = [("写一本小说", False), ("续写第二章", True), ("帮我接着写", True), ("write a new story", False)]
        got = [(text, _is_continuation_request(text)) for text, _ in cases]
        for (text, expected), (_, actual) in zip(cases, got):
            assert actual is expected, f"{text!r} -> {actual}, expected {expected}"
        record("S7 _is_continuation_request boundaries", True, f"{got}")
    except Exception as e:
        record("S7 _is_continuation_request boundaries", False, f"{type(e).__name__}: {e}")

    # ---- S8/S9: OOD (broken brief transport) ----
    from novel_agent.provider_types import ProviderFailureKind, ProviderTransportError

    class OODTransport:
        def generate(self, request):
            from novel_agent.provider_types import ProviderTransportResult

            prompt = request.prompt
            if "需求分析器" in prompt:
                raise ProviderTransportError(kind=ProviderFailureKind.TRANSIENT, detail="wire down")
            if "资深世界观架构师" in prompt:
                payload = {"title": "现代校园", "premise": "一所普通中学。", "rules": ["现实世界规则"], "constraints": []}
            elif "资深角色设计师" in prompt:
                payload = {"characters": [{"name": "林一", "role": "男主角", "goal": "解开误会", "traits": ["温和"]}]}
            elif "剧情架构师" in prompt:
                payload = {"chapters": [{"title": f"第{n}章", "goal": f"目标{n}", "planned_events": [f"事件{n}"]} for n in range(1, 21)]}
            elif "苛刻的小说评审编辑" in prompt:
                payload = {k: {"score": 4, "reason": "r"} for k in ("style_consistency", "character_consistency", "prose_fluency", "scene_completeness", "dialogue_naturalness", "content_alignment")}
            elif "提取章节摘要" in prompt:
                payload = {"summary": "s", "events": []}
            elif "规划器" in prompt:
                payload = {"steps": [{"step_id": 1, "description": "生成大纲", "tool": "outline.generate@1.0.0", "goal": "大纲", "parameters": None}, {"step_id": 2, "description": "写章节", "tool": "chapter.draft@1.0.0", "goal": "章节", "parameters": None}]}
            else:
                return ProviderTransportResult(content=LONG_TEXT, model_name="t", prompt_tokens=1, completion_tokens=1)
            return ProviderTransportResult(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), model_name="t", prompt_tokens=1, completion_tokens=1)

    try:
        root, projects_root = new_projects_root()
        service = fresh_service(projects_root, transport=OODTransport())
        service.start(StartNovelRunRequest(
            project_id=project_id, run_id="qa-s8", idempotency_key="k-s8",
            user_requirements="asdfghjklqwertyuiop", provider="legacy_real"))
        final = wait_terminal(service, project_id, "qa-s8")
        assert final.status is RunStatus.FAILED, f"{final.status} events={dump_events(service, project_id, 'qa-s8')}"
        with connect_project_database(projects_root / project_id.root) as conn:
            fail = conn.execute("SELECT payload_json FROM run_events WHERE run_id='qa-s8' AND event_type='run_failed'").fetchone()
            unrec = conn.execute("SELECT payload_json FROM run_events WHERE run_id='qa-s8' AND event_type='brief_unrecognized'").fetchone()
            tasks = conn.execute("SELECT COUNT(*) FROM tasks WHERE run_id='qa-s8' AND task_type='chapter'").fetchone()[0]
        assert fail is not None and json.loads(fail[0])["reason"] == "intent_recognition_failed"
        assert unrec is not None and json.loads(unrec[0])["requirements"] == "asdfghjklqwertyuiop"
        assert tasks == 0
        record("S8 OOD gibberish -> run FAILED", True,
               f"run_failed.reason={json.loads(fail[0])['reason']} brief_unrecognized.reason={json.loads(unrec[0])['reason']} chapter_tasks={tasks}")
        shutil.rmtree(root, ignore_errors=True)
    except Exception as e:
        record("S8 OOD gibberish -> run FAILED", False, f"{type(e).__name__}: {e}")

    try:
        root, projects_root = new_projects_root()
        service = fresh_service(projects_root, transport=OODTransport())
        service.start(StartNovelRunRequest(
            project_id=project_id, run_id="qa-s9", idempotency_key="k-s9",
            user_requirements="写一本20章的奇幻小说", provider="legacy_real"))
        final = wait_terminal(service, project_id, "qa-s9", seconds=40)
        assert final.status is RunStatus.SUCCEEDED, f"{final.status} review={final.review_reason} events={dump_events(service, project_id, 'qa-s9')}"
        record("S9 valid brief + broken brief transport heuristic fallback", True, f"status={final.status.value}")
        shutil.rmtree(root, ignore_errors=True)
    except Exception as e:
        record("S9 valid brief + broken brief transport heuristic fallback", False, f"{type(e).__name__}: {e}")

    # ---- S10: empty requirements -> ValidationError ----
    try:
        from pydantic import ValidationError
        for blank in ("", "   "):
            try:
                StartNovelRunRequest(project_id=project_id, run_id="qa-blank", idempotency_key="k-blank", user_requirements=blank)
                raise AssertionError(f"accepted blank {blank!r}")
            except ValidationError:
                pass
        trimmed = StartNovelRunRequest(project_id=project_id, run_id="qa-trim", idempotency_key="k-trim", user_requirements="  写一本小说  ")
        assert trimmed.user_requirements == "写一本小说"
        record("S10 service.start blank requirements -> ValidationError", True, "'' and '   ' rejected; surrounding whitespace trimmed")
    except Exception as e:
        record("S10 service.start blank requirements -> ValidationError", False, f"{type(e).__name__}: {e}")

    # ---- S12: repair e2e ----
    class FlakyWorldTransport(PlannerLegacyTransport):
        def __init__(self):
            super().__init__(chapter_count=1)
            self.world_calls = 0

        def generate(self, request):
            from novel_agent.provider_types import ProviderTransportResult

            prompt = request.prompt
            if "资深世界观架构师" in prompt:
                self.world_calls += 1
                if self.world_calls == 1:
                    return ProviderTransportResult(content="完全不是JSON", model_name="t", prompt_tokens=1, completion_tokens=1)
            return super().generate(request)

    try:
        root, projects_root = new_projects_root()
        transport = FlakyWorldTransport()
        service = fresh_service(projects_root, transport=transport)
        service.start(StartNovelRunRequest(
            project_id=project_id, run_id="qa-s12", idempotency_key="k-s12",
            user_requirements="写一本一章的小说", provider="legacy_real", auto_approve=True))
        final = wait_terminal(service, project_id, "qa-s12")
        assert final.status is RunStatus.WAITING_REVIEW, f"{final.status} events={dump_events(service, project_id, 'qa-s12')}"
        assert transport.world_calls == 1, transport.world_calls
        reason0 = final.review_reason
        snap = service.repair(RepairNovelRunRequest(project_id=project_id, run_id="qa-s12", idempotency_key="k-s12-repair"))
        assert snap.status is RunStatus.QUEUED, f"{snap.status} events={dump_events(service, project_id, 'qa-s12')}"
        final = wait_terminal(service, project_id, "qa-s12")
        assert final.status is RunStatus.SUCCEEDED, f"{final.status} review={final.review_reason} events={dump_events(service, project_id, 'qa-s12')}"
        assert transport.world_calls == 2, transport.world_calls
        record("S12 repair from world schema failure", True,
               f"first stop={reason0} world_calls={transport.world_calls} final={final.status.value}")
        shutil.rmtree(root, ignore_errors=True)
    except Exception as e:
        record("S12 repair from world schema failure", False, f"{type(e).__name__}: {e}")

    # ---- S13: start_auto reuse / next run id ----
    try:
        root, projects_root = new_projects_root()
        service = fresh_service(projects_root)
        first = service.start_auto(project_id, "写一本一章的奇幻小说", provider="offline_fake", target_chapter_count=1, auto_approve=True)
        assert first.run_id == "novel-one-run-1", first.run_id
        wait_terminal(service, project_id, first.run_id)
        retry = service.start_auto(project_id, "写一本一章的奇幻小说", provider="offline_fake", target_chapter_count=1, auto_approve=True)
        assert retry.run_id == "novel-one-run-1", retry.run_id
        assert retry.status is RunStatus.QUEUED
        with connect_project_database(projects_root / project_id.root) as conn:
            requeued = conn.execute("SELECT COUNT(*) FROM run_events WHERE run_id='novel-one-run-1' AND event_type='run_requeued'").fetchone()[0]
        assert requeued == 1, requeued
        next_run = service.start_auto(project_id, "换一个新故事：写一本两章的科幻小说", provider="offline_fake", target_chapter_count=2, auto_approve=True)
        assert next_run.run_id == "novel-one-run-2", next_run.run_id
        wait_terminal(service, project_id, "novel-one-run-2")
        record("S13 start_auto reuse + next run id", True,
               f"first={first.run_id} retry={retry.run_id} (run_requeued={requeued}) next={next_run.run_id}")
        shutil.rmtree(root, ignore_errors=True)
    except Exception as e:
        record("S13 start_auto reuse + next run id", False, f"{type(e).__name__}: {e}")

    # ---- P1-17: regenerate_plan (legacy_real plan gate) ----
    try:
        root, projects_root = new_projects_root()
        transport = PlannerLegacyTransport(chapter_count=1)
        service = fresh_service(projects_root, transport=transport)
        service.start(StartNovelRunRequest(
            project_id=project_id, run_id="qa-p1-regen", idempotency_key="k-regen",
            user_requirements="写一本一章的小说", provider="legacy_real",
            target_chapter_count=1, auto_approve=False))
        final = wait_terminal(service, project_id, "qa-p1-regen")
        assert final.status is RunStatus.WAITING_REVIEW, f"{final.status} events={dump_events(service, project_id, 'qa-p1-regen')}"
        assert final.review_reason == "plan_approval_required", final.review_reason
        calls0 = transport.planner_calls
        snap = service.regenerate_plan(project_id, "qa-p1-regen", idempotency_key="k-regen-1")
        assert snap.status is RunStatus.QUEUED, snap.status
        final = wait_terminal(service, project_id, "qa-p1-regen")
        assert final.status is RunStatus.WAITING_REVIEW, f"{final.status} events={dump_events(service, project_id, 'qa-p1-regen')}"
        assert final.review_reason == "plan_approval_required", final.review_reason
        assert transport.planner_calls > calls0, f"planner not re-called: {calls0}->{transport.planner_calls}"
        record("P1 regenerate_plan discards proposal and replans", True,
               f"requeued={snap.status.value} planner_calls {calls0}->{transport.planner_calls} next gate={final.review_reason}")
        shutil.rmtree(root, ignore_errors=True)
    except Exception as e:
        record("P1 regenerate_plan discards proposal and replans", False, f"{type(e).__name__}: {e}")

    # ---- P1-19: cancel path ----
    try:
        from novel_agent.service_models import ControlNovelRunRequest
        from novel_agent.contracts import RunCommandType

        root, projects_root = new_projects_root()
        service = fresh_service(projects_root)
        service.start(StartNovelRunRequest(
            project_id=project_id, run_id="qa-p1-cancel", idempotency_key="k-cancel",
            user_requirements="写一本一章的小说", provider="offline_fake",
            target_chapter_count=1, auto_approve=False))
        final = wait_terminal(service, project_id, "qa-p1-cancel")
        assert final.status is RunStatus.WAITING_REVIEW
        snap = service.control(ControlNovelRunRequest(
            project_id=project_id, run_id="qa-p1-cancel",
            command_type=RunCommandType.CANCEL, idempotency_key="k-cancel-1"))
        assert snap.status in (RunStatus.CANCELLED, RunStatus.CANCEL_REQUESTED), snap.status
        record("P1 cancel waiting_review run", True, f"control -> {snap.status.value}")
        shutil.rmtree(root, ignore_errors=True)
    except Exception as e:
        record("P1 cancel waiting_review run", False, f"{type(e).__name__}: {e}")

    # ---- P1-20: concurrent double-start same key -> reused ----
    try:
        root, projects_root = new_projects_root()
        service = fresh_service(projects_root)
        req = dict(project_id=project_id, run_id="qa-p1-dbl", idempotency_key="k-dbl",
                   user_requirements="写一本一章的小说", provider="offline_fake", target_chapter_count=1, auto_approve=False)
        first = service.start(StartNovelRunRequest(**req))
        second = service.start(StartNovelRunRequest(**req))
        assert first.reused is False
        assert second.reused is True, f"second.reused={second.reused}"
        record("P1 double-start same idempotency key reused", True, f"first.reused={first.reused} second.reused={second.reused}")
        shutil.rmtree(root, ignore_errors=True)
    except Exception as e:
        record("P1 double-start same idempotency key reused", False, f"{type(e).__name__}: {e}")

    print("\n==== SUMMARY ====")
    failed = [r for r in RESULTS if not r[1]]
    print(f"total={len(RESULTS)} passed={len(RESULTS) - len(failed)} failed={len(failed)}")
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
