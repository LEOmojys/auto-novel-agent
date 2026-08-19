from __future__ import annotations

import hashlib
from typing import Literal, TypedDict

from pydantic import ValidationError

from novel_agent.audit import AuditManifestView, ArtifactNotFoundError, RunProjectAuditError, project_audit_manifest
from novel_agent.contracts import RunCommandType, RunStatus, is_run_command_allowed
from novel_agent.pathing import ProjectId
from novel_agent.revision_models import ChapterRevisionError, RebaseConfirmation
from novel_agent.run_models import RunLeaseError, RunTransitionError
from novel_agent.service import (
    ControlNovelRunRequest,
    NovelAgentService,
    RebaseWorkerUnavailableError,
    RepairNovelRunRequest,
    RunNotFoundError,
    RunRepairUnavailableError,
    RunUiSnapshot,
    StartNovelRunRequest,
)


class InteractiveUpdate(TypedDict):
    interactive: bool
    __type__: Literal["update"]


RunUiResult = tuple[
    str,
    str,
    str,
    InteractiveUpdate,
    InteractiveUpdate,
    InteractiveUpdate,
    InteractiveUpdate,
]


def run_control_updates(
    snapshot: RunUiSnapshot | None,
) -> tuple[InteractiveUpdate, InteractiveUpdate, InteractiveUpdate, InteractiveUpdate]:
    if snapshot is None:
        return tuple(_interactive(False) for _ in range(4))
    repair_allowed = (
        snapshot.status is RunStatus.WAITING_REVIEW
        and snapshot.review_reason != "chapter_rebase_confirmation_required"
    )
    return (
        _interactive(is_run_command_allowed(RunCommandType.PAUSE, snapshot.status)),
        _interactive(is_run_command_allowed(RunCommandType.RESUME, snapshot.status)),
        _interactive(is_run_command_allowed(RunCommandType.CANCEL, snapshot.status)),
        _interactive(repair_allowed),
    )


def reset_run_controls() -> tuple[InteractiveUpdate, InteractiveUpdate, InteractiveUpdate, InteractiveUpdate]:
    return run_control_updates(None)


def rebase_control_updates(
    snapshot: RunUiSnapshot | None,
) -> tuple[InteractiveUpdate, InteractiveUpdate]:
    rebase_allowed = (
        snapshot is not None
        and snapshot.status is RunStatus.WAITING_REVIEW
        and snapshot.review_reason == "chapter_rebase_confirmation_required"
    )
    return _interactive(rebase_allowed), _interactive(rebase_allowed)


def reset_rebase_controls() -> tuple[InteractiveUpdate, InteractiveUpdate]:
    return rebase_control_updates(None)


class RunUiController:
    def __init__(self, service: NovelAgentService) -> None:
        self._service = service

    def start_run(
        self,
        project_name: str,
        run_id: str,
        requirements: str,
        provider: Literal["offline_fake", "legacy_real"],
    ) -> RunUiResult:
        requirements_sha256 = hashlib.sha256(requirements.encode("utf-8")).hexdigest()
        try:
            snapshot = self._service.start(
                StartNovelRunRequest(
                    project_id=ProjectId(project_name),
                    run_id=run_id,
                    idempotency_key=f"ui-start:{run_id}:{requirements_sha256}",
                    user_requirements=requirements,
                    provider=provider,
                )
            )
        except (ValidationError, RunNotFoundError, RunTransitionError) as error:
            return _render_error(error)
        return self._render(snapshot)

    def control_run(self, project_name: str, run_id: str, command_type: RunCommandType) -> RunUiResult:
        try:
            project_id = ProjectId(project_name)
            current = self._service.snapshot(project_id, run_id)
            snapshot = self._service.control(
                ControlNovelRunRequest(
                    project_id=project_id,
                    run_id=run_id,
                    command_type=command_type,
                    idempotency_key=f"ui-{command_type.value}:{run_id}:{current.event_count}",
                )
            )
        except (ValidationError, RunNotFoundError, RunTransitionError) as error:
            return _render_error(error)
        return self._render(snapshot)

    def repair_run(self, project_name: str, run_id: str) -> RunUiResult:
        try:
            project_id = ProjectId(project_name)
            current = self._service.snapshot(project_id, run_id)
            review_event_id = current.review_event_id
            if review_event_id is None:
                raise RunRepairUnavailableError(run_id, current.review_reason)
            snapshot = self._service.repair(
                RepairNovelRunRequest(
                    project_id=project_id,
                    run_id=run_id,
                    idempotency_key=f"ui-repair:{run_id}:{review_event_id}",
                )
            )
        except (ValidationError, RunNotFoundError, RunRepairUnavailableError, RunTransitionError) as error:
            return _render_error(error)
        return self._render(snapshot)

    def refresh_run(self, project_name: str, run_id: str) -> RunUiResult:
        try:
            snapshot = self._service.snapshot(ProjectId(project_name), run_id)
        except (ValidationError, RunNotFoundError) as error:
            return _render_error(error)
        return self._render(snapshot)

    def refresh_rebase_controls(
        self,
        project_name: str,
        run_id: str,
    ) -> tuple[InteractiveUpdate, InteractiveUpdate]:
        try:
            snapshot = self._service.snapshot(ProjectId(project_name), run_id)
        except (ValidationError, RunNotFoundError):
            return reset_rebase_controls()
        return rebase_control_updates(snapshot)

    def view_artifact(self, project_name: str, run_id: str, artifact_id: str, revision: int) -> tuple[str, str]:
        try:
            artifact = self._service.artifact(ProjectId(project_name), run_id, artifact_id, revision)
        except (ArtifactNotFoundError, RunNotFoundError, ValidationError) as error:
            return _error_message(error), ""
        metadata = (
            f"{artifact.kind} / {artifact.status}\n"
            f"artifact_id={artifact.artifact_id.root}\nrevision={artifact.revision}\nSHA-256={artifact.sha256}"
        )
        return metadata, artifact.raw_text

    def refresh_audit(self, project_name: str, run_id: str) -> tuple[str, str]:
        try:
            project_id = ProjectId(project_name)
            snapshot = self._service.snapshot(project_id, run_id)
            audit_text = _audit_and_quality(self._service, project_id, run_id)
        except (RunNotFoundError, RunProjectAuditError, ValidationError) as error:
            return _error_message(error), ""
        return _snapshot_message(snapshot), audit_text

    def confirm_rebase(self, project_name: str, run_id: str, confirmation_json: str) -> RunUiResult:
        try:
            confirmation = RebaseConfirmation.model_validate_json(confirmation_json)
            project_id = ProjectId(project_name)
            if confirmation.project_id != project_id or confirmation.run_id != run_id:
                raise RunNotFoundError(project_id, run_id)
            snapshot = self._service.confirm_rebase(confirmation, "gradio-rebase-worker")
        except (
            ChapterRevisionError,
            RebaseWorkerUnavailableError,
            RunLeaseError,
            RunNotFoundError,
            RunTransitionError,
            ValidationError,
        ) as error:
            return _render_error(error)
        return self._render(snapshot)

    def refresh_workflow(self, project_name: str, run_id: str) -> tuple[str, str, str]:
        """Render the agent workflow board: phase, task graph, and reasoning trace."""
        try:
            view = self._service.workflow_view(ProjectId(project_name), run_id)
        except (ValidationError, RunNotFoundError) as error:
            return _error_message(error), "", ""
        return (
            f"当前阶段：{_PHASE_LABELS.get(view.phase, view.phase)}｜{view.current_activity}",
            _render_graph(view),
            "思维链（可审计执行轨迹）\n" + "\n".join(f"- {line}" for line in view.chain) or "（暂无轨迹）",
        )

    def _render(self, snapshot: RunUiSnapshot) -> RunUiResult:
        events = self._service.events(snapshot.project_id, snapshot.run_id)
        event_text = "\n".join(
            f"#{event.sequence}  {event.created_at.isoformat()}  {event.event_type}  {event.payload_json}"
            for event in events
        )
        return (
            _snapshot_message(snapshot),
            snapshot.model_dump_json(indent=2),
            event_text,
            *run_control_updates(snapshot),
        )


def _interactive(interactive: bool) -> InteractiveUpdate:
    return {"interactive": interactive, "__type__": "update"}


_PHASE_LABELS: dict[str, str] = {
    "brief": "需求分析",
    "planning": "规划（Plan）",
    "executing": "执行（Execute）",
    "review": "等待审查",
    "finished": "已完成",
    "failed": "已失败",
    "cancelled": "已取消",
}

_BADGE_STYLE: dict[str, str] = {
    "done": 'style="color:#15803D;font-weight:700"',
    "running": 'style="color:#1D4ED8;font-weight:700"',
    "pending": 'style="color:#94A3B8"',
    "blocked": 'style="color:#B91C1C;font-weight:700"',
}

_BADGE_GLYPH: dict[str, str] = {
    "done": "✓",
    "running": "●",
    "pending": "○",
    "blocked": "✗",
}


def _render_graph(view) -> str:
    """Render the task graph as an inline HTML badge chain."""
    if not view.nodes:
        return "（任务图尚未构建）"
    badges: list[str] = []
    for node in view.nodes:
        style = _BADGE_STYLE.get(node.status, _BADGE_STYLE["pending"])
        glyph = _BADGE_GLYPH.get(node.status, "○")
        label = node.node_id
        badges.append(f'<span {style}>{glyph} {label}</span>')
    return "&nbsp;&nbsp;→&nbsp;&nbsp;".join(badges)


def _snapshot_message(snapshot: RunUiSnapshot) -> str:
    review = "" if snapshot.review_reason is None else f"；复核原因：{snapshot.review_reason}"
    return f"运行 {snapshot.run_id} 的持久化状态：{snapshot.status.value}{review}"


def _error_message(error: Exception) -> str:
    return f"请求未执行：{error}"


def _render_error(error: Exception) -> RunUiResult:
    return _error_message(error), "", "", *run_control_updates(None)


def _audit_and_quality(service: NovelAgentService, project_id: ProjectId, run_id: str) -> str:
    manifests = service.audit_manifests(project_id, run_id)
    reports = service.quality_reports(project_id, run_id)
    manifest_text = "\n\n".join(_manifest_text(manifest) for manifest in manifests)
    report_text = "\n\n".join(
        f"Quality {report.report_id} ({report.decision})\n{report.report_json}" for report in reports
    )
    return f"Audit manifests\n{manifest_text or 'none'}\n\nQuality reports\n{report_text or 'none'}"


def _manifest_text(manifest: AuditManifestView) -> str:
    projection = project_audit_manifest(manifest.manifest_json)
    return "\n".join(
        (
            f"Manifest {manifest.manifest_id}",
            f"SHA-256: {manifest.sha256}",
            f"model: {projection.model}",
            f"parameters: {projection.parameters}",
            f"prompt/template hash: {projection.prompt_template_hash}",
            f"response hash: {projection.response_hash}",
            f"adapter/Skill version: {projection.adapter_skill_version}",
            f"artifact revision: {projection.artifact_revision}",
            "Model output is not deterministically replayable; these fields support audit comparison only.",
        )
    )
    return "\n\n".join(
        (
            "审计清单\n" + manifest_text if manifest_text else "审计清单\n暂无记录",
            "质量报告\n" + report_text if report_text else "质量报告\n暂无记录",
        )
    )


__all__ = (
    "RunUiController",
    "reset_rebase_controls",
    "reset_run_controls",
    "rebase_control_updates",
    "run_control_updates",
)
