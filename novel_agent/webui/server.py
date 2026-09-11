"""Agent web console: a zero-dependency HTTP frontend for the governed runtime.

Serves a static single-page console and a small REST API that maps directly
onto ``NovelAgentService`` (the same service boundary the Gradio console
uses).  The UI polls snapshot/workflow/event endpoints on a timer, so no
websocket or SSE infrastructure is required; all responses are JSON, and the
console never touches SQLite or legacy writers directly.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, unquote, urlparse

from novel_agent.contracts import RunCommandType
from novel_agent.pathing import ProjectId
from novel_agent.service import (
    ControlNovelRunRequest,
    NovelAgentService,
    RepairNovelRunRequest,
    StartNovelRunRequest,
)

_STATIC_ROOT: Final[Path] = Path(__file__).with_name("static")
_MIME: Final[dict[str, str]] = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}


class AgentConsoleHandler(BaseHTTPRequestHandler):
    server_version = "NovelAgentConsole/1.0"

    service: NovelAgentService  # set by the server factory

    # -- HTTP plumbing ------------------------------------------------------

    def log_message(self, fmt: str, *args: object) -> None:  # keep console quiet
        return

    def _request_allowed(self) -> bool:
        """Reject cross-site requests (CSRF/DNS-rebinding defence).

        A malicious webpage can issue simple POST requests (e.g. text/plain)
        to a loopback service; the Origin/Host check ensures only the console's
        own page (or a same-host client) can drive the agent.
        """
        origin = self.headers.get("Origin")
        if origin is None:
            return True  # same-origin / non-browser client
        try:
            parsed = urlparse(origin)
        except ValueError:
            return False
        host = parsed.netloc or parsed.path
        hostname = host.split(":", 1)[0].lower()
        return hostname in ("127.0.0.1", "localhost", "::1", "0.0.0.0")

    def do_GET(self) -> None:  # noqa: N802 - http.server contract
        try:
            if not self._request_allowed():
                return self._json({"error": "cross-site request rejected"}, 403)
            self._route_get(urlparse(self.path))
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Client closed the connection mid-poll; nothing to respond to.
            return
        except Exception as error:  # noqa: BROAD_EXCEPT_OK - API boundary
            self._json({"error": f"{type(error).__name__}: {error}"}, 500)

    def do_POST(self) -> None:  # noqa: N802 - http.server contract
        try:
            if not self._request_allowed():
                return self._json({"error": "cross-site request rejected"}, 403)
            self._route_post(urlparse(self.path))
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return
        except Exception as error:  # noqa: BROAD_EXCEPT_OK - API boundary
            self._json({"error": f"{type(error).__name__}: {error}"}, 500)

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server contract
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Client went away; do not let a dead socket poison the handler.
            return

    def _json(self, payload: object, status: int = 200) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode())

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # -- routes -------------------------------------------------------------

    def _route_get(self, url: urlparse) -> None:
        path = url.path
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        if path == "/" or path == "/index.html":
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path.removeprefix("/static/"))
        if path == "/api/projects":
            return self._json({"projects": [project.root for project in self.server.service.project_ids()]})
        if path == "/api/health":
            return self._json(_provider_health())
        if path == "/api/runs":
            return self._json({"runs": _list_runs(self.server.service)})
        if path == "/api/run/snapshot":
            return self._json(self._snapshot_payload(query))
        if path == "/api/run/workflow":
            return self._json(self._workflow_payload(query))
        if path == "/api/run/artifacts":
            return self._json(self._artifacts_payload(query))
        if path == "/api/run/artifact":
            return self._json(self._artifact_payload(query))
        if path == "/api/run/audit":
            return self._json(self._audit_payload(query))
        if path == "/api/run/events":
            return self._json(self._events_payload(query))
        return self._json({"error": f"unknown route: {path}"}, 404)

    def _route_post(self, url: urlparse) -> None:
        body = self._body()
        if url.path == "/api/run/start":
            return self._json(self._start_payload(body))
        if url.path == "/api/run/control":
            return self._json(self._control_payload(body))
        if url.path == "/api/export":
            return self._json(self._export_payload(body))
        if url.path == "/api/run/rebase":
            return self._json(self._rebase_payload(body))
        if url.path == "/api/run/rewrite":
            return self._json(self._rewrite_payload(body))
        if url.path == "/api/run/approve":
            return self._json(self._approve_payload(body))
        if url.path == "/api/run/regenerate":
            return self._json(self._regenerate_payload(body))
        if url.path == "/api/run/plan/approve":
            return self._json(self._plan_approve_payload(body))
        if url.path == "/api/run/plan/regenerate":
            return self._json(self._plan_regenerate_payload(body))
        return self._json({"error": f"unknown route: {url.path}"}, 404)

    def _static(self, name: str) -> None:
        target = (_STATIC_ROOT / name).resolve()
        if not str(target).startswith(str(_STATIC_ROOT.resolve())) or not target.is_file():
            return self._json({"error": "not found"}, 404)
        mime = _MIME.get(target.suffix, "application/octet-stream")
        self._send(200, mime, target.read_bytes())

    # -- service projections ------------------------------------------------

    def _snapshot_payload(self, query: dict) -> dict:
        project, run_id = _project_run(query)
        snapshot = self.server.service.snapshot(project, run_id)
        return {
            "status": snapshot.status.value,
            "review_reason": snapshot.review_reason,
            "event_count": snapshot.event_count,
            "active_task_id": snapshot.active_task_id,
            "events": [
                {
                    "sequence": event.sequence,
                    "event_type": event.event_type,
                    "payload": json.loads(event.payload_json),
                    "created_at": event.created_at.isoformat(),
                }
                for event in self.server.service.events(project, run_id)
            ],
        }

    def _workflow_payload(self, query: dict) -> dict:
        project, run_id = _project_run(query)
        view = self.server.service.workflow_view(project, run_id)
        return {
            "phase": view.phase,
            "current_activity": view.current_activity,
            "nodes": [
                {"node_id": node.node_id, "status": node.status}
                for node in view.nodes
            ],
            "chain": list(view.chain),
        }

    def _artifacts_payload(self, query: dict) -> dict:
        project, run_id = _project_run(query)
        artifacts = []
        for artifact in self.server.service.artifacts(project, run_id):
            artifacts.append(
                {
                    "artifact_id": artifact.artifact_id.root,
                    "kind": artifact.kind,
                    "status": artifact.status,
                    "revision": artifact.revision,
                    "size": len(artifact.raw_text.encode()),
                }
            )
        return {"artifacts": artifacts}

    def _artifact_payload(self, query: dict) -> dict:
        project, run_id = _project_run(query)
        artifact_id = query.get("artifact_id", "")
        revision = int(query.get("revision", "1"))
        artifact = self.server.service.artifact(project, run_id, artifact_id, revision)
        return {
            "artifact_id": artifact.artifact_id.root,
            "kind": artifact.kind,
            "status": artifact.status,
            "revision": artifact.revision,
            "sha256": artifact.sha256,
            "text": artifact.raw_text,
        }

    def _audit_payload(self, query: dict) -> dict:
        project, run_id = _project_run(query)
        manifests = []
        for manifest in self.server.service.audit_manifests(project, run_id):
            manifests.append(
                {
                    "manifest_id": manifest.manifest_id,
                    "sha256": manifest.sha256,
                    "report_json": json.loads(manifest.manifest_json),
                }
            )
        quality = []
        for report in self.server.service.quality_reports(project, run_id):
            quality.append(
                {
                    "report_id": report.report_id,
                    "decision": report.decision,
                    "report": json.loads(report.report_json),
                }
            )
        failures = list(self.server.service.provider_failures(project, run_id))
        return {"manifests": manifests, "quality": quality, "provider_failures": failures}

    def _events_payload(self, query: dict) -> dict:
        project, run_id = _project_run(query)
        after = int(query.get("after", "-1"))
        events = [
            {
                "sequence": event.sequence,
                "event_type": event.event_type,
                "payload": json.loads(event.payload_json),
            }
            for event in self.server.service.events(project, run_id)
            if event.sequence > after
        ]
        next_seq = max((event["sequence"] for event in events), default=after)
        return {"events": events, "next_seq": next_seq}

    def _export_payload(self, body: dict) -> dict:
        import sqlite3

        import config
        from novel_agent.export import export_project

        project = ProjectId(str(body["project"]))
        path = Path(config.PROJECTS_DIR) / project.root / "agent.sqlite"
        with sqlite3.connect(path) as connection:
            report = export_project(connection, project)
        return {"files": list(report.files)}

    def _rebase_payload(self, body: dict) -> dict:
        """Confirm the pending branch-rebase proposal for a run.

        The confirmation payload is assembled server-side from the persisted
        proposal, so the console user only approves — never hand-crafts JSON.
        """
        import sqlite3

        import config
        from novel_agent.revision_models import (
            RebaseChapter,
            RebaseConfirmation,
            RebaseProposal,
        )

        project = ProjectId(str(body["project"]))
        run_id = str(body["run_id"])
        path = Path(config.PROJECTS_DIR) / project.root / "agent.sqlite"
        with sqlite3.connect(path) as connection:
            row = connection.execute(
                "SELECT proposal_id, proposal_json FROM chapter_rebase_proposals "
                "WHERE run_id = ? AND project_id = ? AND status = 'proposed' "
                "ORDER BY created_at DESC LIMIT 1",
                (run_id, project.root),
            ).fetchone()
        if row is None:
            raise ValueError("no pending rebase proposal for this run")
        proposal = RebaseProposal.model_validate_json(str(row[1]))
        confirmation = RebaseConfirmation(
            project_id=project,
            run_id=run_id,
            proposal_id=proposal.proposal_id,
            user_confirmed=True,
            replacements=(
                RebaseChapter(source=proposal.source, replacement=proposal.replacement),
            ),
        )
        snapshot = self.server.service.confirm_rebase(confirmation, "webui-rebase-worker")
        return _snapshot_dict(snapshot)

    def _rewrite_payload(self, body: dict) -> dict:
        from novel_agent.service_models import ReviseChapterRequest

        snapshot = self.server.service.revise_chapter(
            ReviseChapterRequest(
                project_id=ProjectId(str(body["project"])),
                run_id=str(body["run_id"]),
                chapter_number=int(body["chapter_number"]),
                idempotency_key=f"webui-revise:{body['run_id']}:{body['chapter_number']}",
            )
        )
        return _snapshot_dict(snapshot)

    def _approve_payload(self, body: dict) -> dict:
        from novel_agent.service_models import ApproveArtifactRequest

        snapshot = self.server.service.approve_artifact(
            ApproveArtifactRequest(
                project_id=ProjectId(str(body["project"])),
                run_id=str(body["run_id"]),
                artifact_id=str(body["artifact_id"]),
                revision=int(body["revision"]),
                idempotency_key=f"webui-approve:{body['run_id']}:{body['artifact_id']}:{body['revision']}",
            )
        )
        return _snapshot_dict(snapshot)

    def _regenerate_payload(self, body: dict) -> dict:
        from novel_agent.service_models import RegenerateArtifactRequest

        snapshot = self.server.service.regenerate_artifact(
            RegenerateArtifactRequest(
                project_id=ProjectId(str(body["project"])),
                run_id=str(body["run_id"]),
                artifact_id=str(body["artifact_id"]),
                revision=int(body["revision"]),
                idempotency_key=f"webui-regenerate:{body['run_id']}:{body['artifact_id']}:{body['revision']}",
            )
        )
        return _snapshot_dict(snapshot)

    def _plan_approve_payload(self, body: dict) -> dict:
        snapshot = self.server.service.approve_plan(
            ProjectId(str(body["project"])),
            str(body["run_id"]),
            idempotency_key=f"webui-plan-approve:{body['run_id']}",
        )
        return _snapshot_dict(snapshot)

    def _plan_regenerate_payload(self, body: dict) -> dict:
        snapshot = self.server.service.regenerate_plan(
            ProjectId(str(body["project"])),
            str(body["run_id"]),
            idempotency_key=f"webui-plan-regenerate:{body['run_id']}",
        )
        return _snapshot_dict(snapshot)

    def _start_payload(self, body: dict) -> dict:
        run_id = str(body["run_id"]).strip() if body.get("run_id") else ""
        requirements = str(body["requirements"])
        project_id = ProjectId(str(body["project"]))
        if run_id:
            # Explicit run_id (tests / advanced use); idempotency keyed on the
            # requirement so resubmitting identical text is a no-op while a
            # changed requirement gets a fresh command.
            snapshot = self.server.service.start(
                StartNovelRunRequest(
                    project_id=project_id,
                    run_id=run_id,
                    idempotency_key=f"webui-start:{run_id}:{hashlib.sha256(requirements.encode()).hexdigest()[:12]}",
                    user_requirements=requirements,
                    provider=str(body.get("provider", "offline_fake")),
                    target_chapter_count=int(body["chapter_count"]) if body.get("chapter_count") else None,
                    auto_approve=bool(body.get("auto_approve", True)),
                )
            )
        else:
            # Multi-turn console mode: the service assigns the run id
            # automatically (reusing a terminal run for retries).
            snapshot = self.server.service.start_auto(
                project_id,
                requirements,
                provider=str(body.get("provider", "offline_fake")),
                target_chapter_count=int(body["chapter_count"]) if body.get("chapter_count") else None,
                auto_approve=bool(body.get("auto_approve", True)),
            )
        return _snapshot_dict(snapshot)

    def _control_payload(self, body: dict) -> dict:
        project = ProjectId(str(body["project"]))
        run_id = str(body["run_id"])
        command = RunCommandType(str(body["command"]))
        if command is RunCommandType.REPAIR:
            snapshot = self.server.service.repair(
                RepairNovelRunRequest(
                    project_id=project,
                    run_id=run_id,
                    idempotency_key=f"webui-repair:{run_id}",
                )
            )
        else:
            snapshot = self.server.service.control(
                ControlNovelRunRequest(
                    project_id=project,
                    run_id=run_id,
                    command_type=command,
                    idempotency_key=f"webui-{command.value}:{run_id}",
                )
            )
        return _snapshot_dict(snapshot)


def _provider_health(call=None) -> dict:
    """Live LLM connectivity probe: key presence + one minimal real call.

    ``call`` is injectable for tests; the default is the real DeepSeek
    adapter.  Never blocks the console: the UI polls this endpoint and the
    probe itself runs inside the request with a short deadline.
    """
    import os
    import time

    if not (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY_1")
    ):
        return {
            "ok": False,
            "reason": "missing_key",
            "hint": "请设置环境变量 LLM_API_KEY（或 DEEPSEEK_API_KEY）后重启",
            "latency_ms": None,
        }
    from tools_call_deepseek import access_deepseek, get_model_name

    probe = call or access_deepseek
    started = time.monotonic()
    try:
        probe("请只回复两个字：正常", max_retry=1)
        return {
            "ok": True,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "model": get_model_name(),
        }
    except Exception as error:  # noqa: BROAD_EXCEPT_OK - surface the raw provider failure
        return {
            "ok": False,
            "reason": f"{type(error).__name__}: {error}",
            "latency_ms": round((time.monotonic() - started) * 1000),
        }


def _project_run(query: dict) -> tuple[ProjectId, str]:
    project = ProjectId(str(query.get("project", "default")))
    run_id = str(query.get("run_id", ""))
    if not run_id:
        raise ValueError("run_id is required")
    return project, run_id


def _snapshot_dict(snapshot) -> dict:
    return {
        "run_id": snapshot.run_id,
        "status": snapshot.status.value,
        "review_reason": snapshot.review_reason,
        "event_count": snapshot.event_count,
        "command_id": snapshot.command_id,
        "reused": snapshot.reused,
    }


def _list_runs(service: NovelAgentService) -> list[dict]:
    import sqlite3

    import config

    runs: list[dict] = []
    for project in service.project_ids():
        path = Path(config.PROJECTS_DIR) / project.root / "agent.sqlite"
        if not path.is_file():
            continue
        try:
            with sqlite3.connect(path) as connection:
                rows = connection.execute(
                    "SELECT run_id, state, updated_at FROM runs ORDER BY updated_at DESC LIMIT 20"
                ).fetchall()
        except sqlite3.Error:
            continue
        runs.extend(
            {"project": project.root, "run_id": str(row[0]), "state": str(row[1]), "updated_at": str(row[2])}
            for row in rows
        )
    return runs


def create_console_server(
    service: NovelAgentService,
    host: str = "127.0.0.1",
    port: int = 7860,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), AgentConsoleHandler)
    server.service = service  # type: ignore[attr-defined]
    return server


def launch_console(host: str = "127.0.0.1", port: int = 7860) -> None:
    import os

    service = NovelAgentService.from_config()
    server = create_console_server(service, host=host, port=port)
    print(f"Agent console running at http://{host}:{port}")
    health = _provider_health()
    if health["ok"]:
        print(f"LLM API: OK ({health['latency_ms']}ms)")
    else:
        print(f"LLM API: UNAVAILABLE - {health.get('reason', 'unknown')}")
    timeout = os.environ.get("NOVEL_AGENT_STAGE_TIMEOUT_SECONDS")
    if timeout is None:
        print(
            "警告: 未设置 NOVEL_AGENT_STAGE_TIMEOUT_SECONDS（默认 30 秒）。"
            "legacy_real 真实模型运行会被超时中断，请设置如 7200。"
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ("AgentConsoleHandler", "create_console_server", "launch_console")
