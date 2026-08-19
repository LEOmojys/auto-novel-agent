from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, assert_never

from novel_agent.adapters import GovernedCreativeAdapter
from novel_agent.chapter_executor import (
    ChapterExecutionCancelled,
    ChapterExecutionRequest,
    ChapterExecutionSucceeded,
    ChapterExecutionWorkflow,
    ChapterExecutionWaitingReview,
    ChapterGenerator,
    ContinuityAuditor,
    MemoryExtractor,
)
from novel_agent.domains.novel import detect_existing_assets, existing_assets_summary
from novel_agent.contracts import Provenance, ProvenanceKind, RunStatus, StoryBrief
from novel_agent.offline_fake import (
    OfflineChapterGenerator,
    OfflineContinuityAuditor,
    OfflineMemoryExtractor,
    OfflinePlanningGenerator,
    OfflineQualityReviewer,
    offline_provenance,
)
from novel_agent.pathing import ProjectId, ProjectPathResolver
from novel_agent.planning import PlanningBlocked, PlanningSucceeded, PlanningWorkflow
from novel_agent.provider import LegacyDeepSeekTransport
from novel_agent.provider_types import (
    ProviderFailureKind,
    ProviderTransport,
    ProviderTransportError,
    ProviderTransportResult,
)
from novel_agent.tool_registry import ToolGateway, ToolRegistry
from novel_agent.workflow_tools import register_workflow_tools
from novel_agent.run_coordinator import RunCoordinator
from novel_agent.run_event_store import append_run_event
from novel_agent.run_models import RunRecoveryState
from novel_agent.run_state import RunEvent
from novel_agent.run_worker import RunStageExecutor, StageCancellationToken
from novel_agent.skills import load_builtin_catalog, load_builtin_templates

if TYPE_CHECKING:
    from novel_agent.service import StartNovelRunRequest


@dataclass(frozen=True, slots=True)
class _OfflineStage:
    coordinator: RunCoordinator
    request: StartNovelRunRequest
    path_resolver: ProjectPathResolver
    now: Callable[[], datetime]

    def execute(
        self,
        recovery: RunRecoveryState,
        cancellation: StageCancellationToken | None = None,
    ) -> None:
        if cancellation is not None and cancellation.is_cancelled():
            return
        timestamp = self.now()
        reviewer_holder: dict[str, object] = {"current": OfflineQualityReviewer()}
        generator = OfflinePlanningGenerator(timestamp)
        gateway = _build_tool_gateway(
            self.coordinator,
            planning_generator=generator,
            chapter_generator=OfflineChapterGenerator(timestamp),
            extractor=OfflineMemoryExtractor(timestamp, self.request.project_id),
            auditor=OfflineContinuityAuditor(),
            reviewer=lambda: reviewer_holder["current"],
        )
        _execute_graph_stage(
            coordinator=self.coordinator,
            request=self.request,
            path_resolver=self.path_resolver,
            now=self.now,
            transport=_UnavailableTransport(),
            explicit_chapter_count=self.request.target_chapter_count or 1,
            brief_analysis=False,
            cancellation=cancellation,
            recovery=recovery,
            gateway=gateway,
            planning_generator=generator,
            chapter_generator=OfflineChapterGenerator(timestamp),
            extractor=OfflineMemoryExtractor(timestamp, self.request.project_id),
            auditor=OfflineContinuityAuditor(),
            reviewer_holder=reviewer_holder,
            real_reviewer_factory=None,
        )


def _call_deepseek(
    text: str,
    *,
    max_retry: int,
    model_name: str,
    return_tokens: Literal[True],
    caller: str,
) -> tuple[str, Mapping[str, int | str]]:
    from tools_call_deepseek import access_deepseek

    return access_deepseek(
        text,
        max_retry=max_retry,
        model_name=model_name,
        return_tokens=return_tokens,
        caller=caller,
    )


_DEFAULT_TARGET_CHAPTERS: int = 10
_DEFAULT_TARGET_WORDS: int = 3_000


def _build_tool_gateway(
    coordinator: RunCoordinator,
    *,
    planning_generator: PlanningGenerator,
    chapter_generator: ChapterGenerator,
    extractor: MemoryExtractor,
    auditor: ContinuityAuditor,
    reviewer: QualityReviewer | Callable[[], QualityReviewer],
) -> ToolGateway:
    """Build the audited tool gateway that production stages run their calls through."""
    registry = ToolRegistry()
    register_workflow_tools(
        registry,
        planning_generator=planning_generator,
        chapter_generator=chapter_generator,
        extractor=extractor,
        auditor=auditor,
        reviewer=reviewer,
    )
    return ToolGateway(registry, coordinator.connection)


def _stage_event_sink(coordinator: RunCoordinator, now: Callable[[], datetime], run_id: str) -> Callable[[str], None]:
    """Append an auditable stage event for a governed run."""

    def record(stage: str) -> None:
        append_run_event(
            coordinator.connection,
            RunEvent(run_id, "stage_entered", {"stage": stage}, now()),
        )

    return record


def _failure_sink_factory(
    coordinator: RunCoordinator,
    project_id: ProjectId,
    run_id: str,
) -> Callable[[str, str, str, str, str], None]:
    """Build the shared provider-failure recorder for all LLM call sites."""

    def record(
        caller: str,
        model_name: str,
        failure_kind: str,
        reason: str,
        prompt_sha256: str,
    ) -> None:
        from novel_agent.provider_failure import record_provider_failure

        record_provider_failure(
            coordinator.connection,
            project_id=project_id.root,
            run_id=run_id,
            caller=caller,
            model_name=model_name,
            failure_kind=failure_kind,
            reason=reason,
            prompt_sha256=prompt_sha256,
        )

    return record


def _plan_proposal_sink(
    coordinator: RunCoordinator,
    run_id: str,
    now: Callable[[], datetime],
) -> Callable[[object], None]:
    """Persist a generated plan as a pending proposal for user confirmation."""

    def persist(plan: object) -> None:
        import hashlib as _hashlib

        plan_json = plan.model_dump_json()
        command_id = _hashlib.sha256(f"{run_id}:plan-proposed".encode()).hexdigest()
        timestamp = now()
        coordinator.connection.execute(
            "INSERT INTO commands(command_id, run_id, command_type, idempotency_key, payload_json, state, created_at) "
            "VALUES (?, ?, 'plan_proposed', 'plan-proposed', ?, 'proposed', ?) "
            "ON CONFLICT(command_id) DO UPDATE SET payload_json = excluded.payload_json",
            (command_id, run_id, plan_json, timestamp.isoformat()),
        )

    return persist


def _call_event_sink_factory(
    coordinator: RunCoordinator,
    now: Callable[[], datetime],
    run_id: str,
) -> Callable[[str, str, str], None]:
    """Build the shared LLM call-lifecycle event recorder.

    Emits ``provider_call_started`` before each model invocation and
    ``provider_call_finished`` after it returns (or fails), so the console
    can show exactly which model call is in flight and for how long.
    """

    def record(caller: str, model_name: str, event: str) -> None:
        if event not in {"started", "finished"}:
            return
        append_run_event(
            coordinator.connection,
            RunEvent(
                run_id,
                f"provider_call_{event}",
                {"caller": caller, "model": model_name},
                now(),
            ),
        )

    return record


def _revise_target_chapter(connection, run_id: str) -> int | None:
    """Return the chapter number the latest REVISE command targets, if any."""
    import json as json_module

    row = connection.execute(
        "SELECT payload_json FROM commands WHERE run_id = ? AND command_type = 'revise' "
        "ORDER BY created_at DESC, command_id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        payload = json_module.loads(str(row[0]))
        chapter_number = int(payload.get("chapter_number") or 0)
    except (json_module.JSONDecodeError, TypeError, ValueError):
        return None
    return chapter_number if chapter_number >= 1 else None


def _plan_approval_pending(connection, run_id: str) -> bool:
    """True when a generated plan is waiting for user confirmation."""
    row = connection.execute(
        "SELECT state FROM commands WHERE run_id = ? AND command_type = 'plan_proposed' "
        "ORDER BY created_at DESC, command_id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return row is not None and str(row[0]) == "proposed"


def _load_approved_plan(connection, run_id: str):
    """Return the approved plan JSON (as model steps) or None."""
    import json as json_module

    from novel_agent.agent.plan_generator import ExecutionPlan

    row = connection.execute(
        "SELECT payload_json FROM commands WHERE run_id = ? AND command_type = 'plan_proposed' "
        "AND state = 'approved' ORDER BY created_at DESC, command_id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        return ExecutionPlan.model_validate_json(str(row[0]))
    except Exception:  # noqa: BROAD_EXCEPT_OK - a corrupt approval must not block
        return None


def _load_proposed_plan(connection, run_id: str):
    """Return the latest proposed (unconfirmed) plan, if any."""
    import json as json_module

    from novel_agent.agent.plan_generator import ExecutionPlan

    row = connection.execute(
        "SELECT payload_json FROM commands WHERE run_id = ? AND command_type = 'plan_proposed' "
        "AND state = 'proposed' ORDER BY created_at DESC, command_id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        return ExecutionPlan.model_validate_json(str(row[0]))
    except Exception:  # noqa: BROAD_EXCEPT_OK - a corrupt proposal must not block
        return None


def _request_plan_approval(
    coordinator: RunCoordinator,
    run_id: str,
    now: Callable[[], datetime],
    plan,
) -> None:
    """Persist the generated plan and pause the run for user confirmation."""
    import json as json_module

    import hashlib as _hashlib

    from novel_agent.run_event_store import append_run_event as _append_event
    from novel_agent.run_state import RunEvent as _RunEvent

    plan_json = plan.model_dump_json()
    command_id = _hashlib.sha256(f"{run_id}:plan-proposed".encode()).hexdigest()
    timestamp = now()
    coordinator.connection.execute(
        "INSERT INTO commands(command_id, run_id, command_type, idempotency_key, payload_json, state, created_at) "
        "VALUES (?, ?, 'plan_proposed', 'plan-proposed', ?, 'proposed', ?) "
        "ON CONFLICT(command_id) DO UPDATE SET payload_json = excluded.payload_json, state = 'proposed'",
        (command_id, run_id, plan_json, timestamp.isoformat()),
    )
    coordinator.connection.execute(
        "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
        (RunStatus.WAITING_REVIEW.value, timestamp.isoformat(), run_id),
    )
    _append_event(
        coordinator.connection,
        _RunEvent(
            run_id,
            "plan_approval_required",
            {"steps": ";".join(f"{step.step_id}:{step.tool}:{step.description}" for step in plan.steps)},
            timestamp,
        ),
    )


def _finish_run_if_pending(
    coordinator: RunCoordinator,
    run_id: str,
    now: Callable[[], datetime],
) -> None:
    """Complete a run whose graph finished but left no commit trail.

    Normal runs are completed by their final chapter commit; runs whose
    chapters were all skipped (already written under a reused outline) have
    no commit and must be completed here.
    """
    row = coordinator.connection.execute(
        "SELECT state FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None or str(row[0]) != RunStatus.RUNNING.value:
        return
    timestamp = now()
    coordinator.connection.execute(
        "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
        (RunStatus.SUCCEEDED.value, timestamp.isoformat(), run_id),
    )
    append_run_event(
        coordinator.connection,
        RunEvent(run_id, "run_succeeded", {}, timestamp),
    )


class _UnavailableTransport:
    """Transport that always fails, forcing BriefAnalyzer's deterministic fallback."""

    def generate(self, request) -> ProviderTransportResult:
        raise ProviderTransportError(
            kind=ProviderFailureKind.CONFIGURATION,
            detail="provider transport disabled for offline stage",
        )


def _real_reviewer_factory(
    transport: ProviderTransport,
    now: Callable[[], datetime],
    run_id: str,
    quality_prompt: str | None = None,
    failure_sink: Callable[[str, str, str, str, str], None] | None = None,
    call_event_sink: Callable[[str, str, str], None] | None = None,
) -> Callable[[object, object], object]:
    """Build the live provider-backed quality reviewer once the cast is ready."""

    def factory(context: object, cast_result: object) -> object:
        from novel_agent.agent.domain_plugin import NodeExecutionContext
        from novel_agent.adapters import CharacterAdaptation
        from novel_agent.domains.quality_evaluators import (
            CharCountMetricsEvaluator,
            ProviderStructuredEvaluator,
        )
        from novel_agent.quality import CharacterFixture, ProjectScopedEvaluationAdapter

        cast: CharacterAdaptation = cast_result
        fixtures = tuple(
            CharacterFixture(character_id=character.character_id, name=character.name)
            for character in cast.value.characters
        )
        return ProjectScopedEvaluationAdapter(
            project_id=context.project_id,
            characters=fixtures,
            evaluator=ProviderStructuredEvaluator(
                transport=transport,
                now=now,
                run_id=run_id,
                prompt=quality_prompt,
                failure_sink=failure_sink,
                call_event_sink=call_event_sink,
            ),
            metrics=CharCountMetricsEvaluator(),
        )

    return factory


def _execute_graph_stage(
    *,
    coordinator: RunCoordinator,
    request: StartNovelRunRequest,
    path_resolver: ProjectPathResolver,
    now: Callable[[], datetime],
    transport: ProviderTransport,
    explicit_chapter_count: int | None,
    brief_analysis: bool,
    cancellation: StageCancellationToken | None,
    recovery: RunRecoveryState,
    gateway: ToolGateway,
    planning_generator: PlanningGenerator,
    chapter_generator: ChapterGenerator,
    extractor: MemoryExtractor,
    auditor: ContinuityAuditor,
    reviewer_holder: dict[str, object],
    memory_names_holder: dict[str, object] | None = None,
    plan_generator: object | None = None,
    plan_info_holder: dict[str, object] | None = None,
    real_reviewer_factory: Callable[[object, object], object] | None = None,
) -> None:
    """Run the plan-and-execute task graph for the request through the generic core.

    Shared by the offline and legacy_real stages: builds the brief (with or
    without LLM requirement analysis), builds and records the domain task
    graph, then runs the generic executor.  All governed behavior (tool
    audit, plan commit, chapter gates, memory promotion) lives in the domain
    node executors.
    """
    from novel_agent.agent.brief import BriefAnalyzer, _INTENT_UNRECOGNIZED
    from novel_agent.agent.task_graph_executor import TaskGraphExecutor
    from novel_agent.domains.novel import NovelDomainPlugin

    catalog = load_builtin_catalog()
    templates = load_builtin_templates(catalog)
    analyzer_transport = transport if brief_analysis else _UnavailableTransport()
    failure_sink = _failure_sink_factory(coordinator, request.project_id, request.run_id)
    call_event_sink = _call_event_sink_factory(coordinator, now, request.run_id)
    brief = BriefAnalyzer(
        transport=analyzer_transport,
        now=now,
        prompt=templates.get("brief-normalization"),
        failure_sink=failure_sink,
        call_event_sink=call_event_sink,
    ).analyze(
        request.user_requirements,
        project_id=request.project_id,
        run_id=request.run_id,
        brief_id=f"{request.run_id}:brief",
        explicit_chapter_count=explicit_chapter_count,
    )
    append_run_event(
        coordinator.connection,
        RunEvent(
            request.run_id,
            "brief_analyzed",
            {
                "goal": brief.goal,
                "domain": brief.domain,
                "intent": brief.intent,
                "degraded": brief.degraded,
                "degraded_reason": brief.degraded_reason or "",
                "degraded_detail": brief.degraded_detail or "",
                "constraints": ",".join(brief.constraints),
                "style": ",".join(brief.style_guidance),
                "acceptance": ",".join(brief.acceptance_criteria),
                "chapter_count": brief.target_chapter_count or 0,
            },
            now(),
        ),
    )
    if brief.intent == _INTENT_UNRECOGNIZED and brief_analysis:
        # Out-of-distribution request: the provider could not extract a real
        # creative goal and no deterministic rule matched either (everyday
        # chatter, gibberish).  Fail the run explicitly instead of silently
        # planning a full novel from noise.  Offline stages deliberately
        # disable requirement analysis (``brief_analysis=False``), so their
        # placeholder requirements are not rejected here.
        append_run_event(
            coordinator.connection,
            RunEvent(
                request.run_id,
                "brief_unrecognized",
                {
                    "requirements": request.user_requirements,
                    "reason": brief.degraded_reason or "unrecognized_request",
                    "detail": brief.degraded_detail or "",
                },
                now(),
            ),
        )
        coordinator.fail_run(recovery.lease, "intent_recognition_failed")
        return
    from novel_agent.domains.novel import detect_existing_assets, existing_assets_summary

    existing_assets = detect_existing_assets(
        path_resolver.resolve_project(request.project_id),
        connection=coordinator.connection,
    )
    plugin = NovelDomainPlugin(        planning_generator=planning_generator,
        chapter_generator=chapter_generator,
        extractor=extractor,
        auditor=auditor,
        reviewer_holder=reviewer_holder,
        memory_names_holder=memory_names_holder,
        plan_generator=plan_generator,
        plan_info_holder=plan_info_holder,
        existing_assets=existing_assets,
        auto_approve=request.auto_approve,
        real_reviewer_factory=real_reviewer_factory,
    )
    revise_target = _revise_target_chapter(coordinator.connection, request.run_id)
    if revise_target is not None:
        graph = plugin.build_rewrite_graph(
            brief,
            revise_target,
            run_id=request.run_id,
            project_id=request.project_id,
            now=now,
        )
    else:
        approved_plan = _load_approved_plan(coordinator.connection, request.run_id)
        if approved_plan is not None:
            graph = plugin.build_graph(
                brief,
                run_id=request.run_id,
                project_id=request.project_id,
                now=now,
                approved_plan=approved_plan,
            )
        else:
            graph = plugin.build_graph(brief, run_id=request.run_id, project_id=request.project_id, now=now)
            if (
                not request.auto_approve
                and plan_generator is not None
                and not plan_info_holder.get("degraded", False)
            ):
                plan = _load_proposed_plan(coordinator.connection, request.run_id)
                if plan is not None:
                    _request_plan_approval(coordinator, request.run_id, now, plan)
                    return
    if plan_info_holder is not None:
        steps = plan_info_holder.get("steps")
        steps_text = (
            ";".join(f"{item[0]}:{item[1]}:{item[2]}" for item in steps)
            if isinstance(steps, tuple) and steps
            else ""
        )
        append_run_event(
            coordinator.connection,
            RunEvent(
                request.run_id,
                "plan_generated",
                {
                    "degraded": bool(plan_info_holder.get("degraded", False)),
                    "degraded_reason": str(plan_info_holder.get("reason") or ""),
                    "steps": steps_text,
                },
                now(),
            ),
        )
    append_run_event(
        coordinator.connection,
        RunEvent(
            request.run_id,
            "task_graph_built",
            {
                "graph_id": graph.graph_id,
                "domain": graph.domain,
                "nodes": ",".join(node.node_id for node in graph.nodes),
            },
            now(),
        ),
    )
    executor = TaskGraphExecutor(
        plugin=plugin,
        coordinator=coordinator,
        path_resolver=path_resolver,
        now=now,
        catalog=catalog,
        gateway=gateway,
        stage_event_sink=_stage_event_sink(coordinator, now, request.run_id),
    )
    try:
        result = executor.execute(graph, brief, recovery.lease)
    except ProviderTransportError:
        coordinator.fail_run(recovery.lease, "provider_transport_failure")
        return
    if cancellation is not None and cancellation.is_cancelled():
        return
    match result.status:
        case "succeeded":
            _finish_run_if_pending(coordinator, request.run_id, now)
            return
        case "blocked":
            return
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class _LegacyStage:
    """Real creative pipeline: world -> cast -> outline -> chapters via live models.

    ``provider="legacy_real"`` runs the governed planning and chapter workflows
    with ``GovernedCreativeAdapter`` as the generator/extractor.  The adapter
    assembles governed contracts (ids, provenance, timestamps) around model
    content, so accepted artifacts are real novels rather than offline fakes.
    """

    coordinator: RunCoordinator
    request: StartNovelRunRequest
    path_resolver: ProjectPathResolver
    now: Callable[[], datetime]
    transport: ProviderTransport | None

    def execute(
        self,
        recovery: RunRecoveryState,
        cancellation: StageCancellationToken | None = None,
    ) -> None:
        if cancellation is not None and cancellation.is_cancelled():
            return
        transport = self.transport or LegacyDeepSeekTransport(call=_call_deepseek)
        catalog = load_builtin_catalog()
        templates = load_builtin_templates(catalog)
        memory_names_holder: dict[str, object] = {}
        failure_sink = _failure_sink_factory(
            self.coordinator,
            self.request.project_id,
            self.request.run_id,
        )
        call_event_sink = _call_event_sink_factory(self.coordinator, self.now, self.request.run_id)
        generator = GovernedCreativeAdapter(
            transport=transport,
            now=self.now,
            run_id=self.request.run_id,
            project_id=self.request.project_id,
            prompts=templates,
            character_names_holder=memory_names_holder,
            failure_sink=failure_sink,
            call_event_sink=call_event_sink,
        )
        reviewer_holder: dict[str, object] = {"current": OfflineQualityReviewer()}
        plan_info_holder: dict[str, object] = {}
        existing_assets = detect_existing_assets(
            self.path_resolver.resolve_project(self.request.project_id),
            connection=self.coordinator.connection,
        )
        from novel_agent.agent.plan_generator import PlanGenerator

        gateway = _build_tool_gateway(
            self.coordinator,
            planning_generator=generator,
            chapter_generator=generator,
            extractor=generator,
            auditor=OfflineContinuityAuditor(),
            reviewer=lambda: reviewer_holder["current"],
        )
        _execute_graph_stage(
            coordinator=self.coordinator,
            request=self.request,
            path_resolver=self.path_resolver,
            now=self.now,
            transport=transport,
            explicit_chapter_count=self.request.target_chapter_count,
            brief_analysis=True,
            cancellation=cancellation,
            recovery=recovery,
            gateway=gateway,
            planning_generator=generator,
            chapter_generator=generator,
            extractor=generator,
            auditor=OfflineContinuityAuditor(),
            reviewer_holder=reviewer_holder,
            memory_names_holder=memory_names_holder,
            plan_generator=PlanGenerator(
                transport=transport,
                now=self.now,
                run_id=self.request.run_id,
                prompt=templates.get("plan-execution"),
                existing_summary=existing_assets_summary(existing_assets),
                failure_holder=plan_info_holder,
                failure_sink=failure_sink,
                call_event_sink=call_event_sink,
                plan_sink=_plan_proposal_sink(self.coordinator, self.request.run_id, self.now),
            ),
            plan_info_holder=plan_info_holder,
            real_reviewer_factory=_real_reviewer_factory(
                transport,
                self.now,
                self.request.run_id,
                quality_prompt=templates.get("quality-review"),
                failure_sink=failure_sink,
                call_event_sink=call_event_sink,
            ),
        )


@dataclass(frozen=True, slots=True)
class ProviderSetupStage:
    coordinator: RunCoordinator
    reason_code: str = "provider_executor_not_configured"

    def execute(
        self,
        recovery: RunRecoveryState,
        cancellation: StageCancellationToken | None = None,
    ) -> None:
        if cancellation is not None and cancellation.is_cancelled():
            return
        self.coordinator.block_planning(recovery.lease, self.reason_code)


def default_stage_executor_factory(
    coordinator: RunCoordinator,
    request: StartNovelRunRequest,
    *,
    path_resolver: ProjectPathResolver | None = None,
    now: Callable[[], datetime] | None = None,
    provider_transport: ProviderTransport | None = None,
) -> RunStageExecutor:
    mode = request.provider or os.environ.get("NOVEL_AGENT_PROVIDER_MODE")
    if mode in {"offline_fake", "legacy_real"}:
        clock = now or (lambda: datetime.now(UTC))
        if mode == "legacy_real":
            return _LegacyStage(
                coordinator,
                request,
                path_resolver or ProjectPathResolver.from_config(),
                clock,
                provider_transport,
            )
        return _OfflineStage(
            coordinator,
            request,
            path_resolver or ProjectPathResolver.from_config(),
            clock,
        )
    return ProviderSetupStage(coordinator)


__all__ = ("ProviderSetupStage", "default_stage_executor_factory")
