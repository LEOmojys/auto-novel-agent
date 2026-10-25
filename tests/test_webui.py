from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

import pytest

from novel_agent.pathing import ProjectId
from tests.conftest import _ConsoleClient as _Client


FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT_ID = ProjectId("novel-one")


def _wait_succeeded(console: _Client, run_id: str) -> dict:
    deadline = monotonic() + 20.0
    while True:
        status, snapshot = console.get(f"/api/run/snapshot?project={PROJECT_ID.root}&run_id={run_id}")
        assert status == 200
        if snapshot["status"] in ("succeeded", "failed", "waiting_review", "cancelled"):
            return snapshot
        assert monotonic() < deadline, f"run did not terminate: {snapshot['status']}"
        sleep(0.05)


def test_projects_and_runs_endpoints(console: _Client) -> None:
    status, data = console.get("/api/projects")
    assert status == 200
    assert PROJECT_ID.root in data["projects"]

    status, data = console.get("/api/runs")
    assert status == 200
    assert isinstance(data["runs"], list)


def test_static_index_served(console: _Client) -> None:
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", console.port, timeout=5)
    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        assert response.status == 200
        assert "Agent 小说创作控制台" in body
    finally:
        connection.close()


def test_start_snapshot_workflow_artifacts_audit_events(console: _Client) -> None:
    status, started = console.post(
        "/api/run/start",
        {
            "project": PROJECT_ID.root,
            "run_id": "webui-run-1",
            "requirements": "写一本一章的小说",
            "provider": "offline_fake",
            "chapter_count": 1,
        },
    )
    assert status == 200
    assert started["status"] == "queued"

    snapshot = _wait_succeeded(console, "webui-run-1")
    assert snapshot["status"] == "succeeded"
    assert snapshot["event_count"] >= 5
    assert any(event["event_type"] == "plan_validated" for event in snapshot["events"])
    assert any(event["event_type"] == "run_succeeded" for event in snapshot["events"])

    status, workflow = console.get(f"/api/run/workflow?project={PROJECT_ID.root}&run_id=webui-run-1")
    assert status == 200
    assert workflow["phase"] == "finished"
    assert [node["node_id"] for node in workflow["nodes"]] == ["world", "cast", "outline", "chapter_1"]
    assert all(node["status"] == "done" for node in workflow["nodes"])
    assert any("任务图已构建" in line for line in workflow["chain"])

    status, artifacts = console.get(f"/api/run/artifacts?project={PROJECT_ID.root}&run_id=webui-run-1")
    assert status == 200
    kinds = {artifact["kind"] for artifact in artifacts["artifacts"]}
    assert {"world_bible", "character_set", "plan_contract", "chapter"} <= kinds
    chapter = next(artifact for artifact in artifacts["artifacts"] if artifact["kind"] == "chapter")

    status, artifact = console.get(
        f"/api/run/artifact?project={PROJECT_ID.root}&run_id=webui-run-1"
        f"&artifact_id={chapter['artifact_id']}&revision=1"
    )
    assert status == 200
    assert artifact["kind"] == "chapter"
    assert len(artifact["text"]) > 0

    status, audit = console.get(f"/api/run/audit?project={PROJECT_ID.root}&run_id=webui-run-1")
    assert status == 200
    assert isinstance(audit["manifests"], list)
    assert isinstance(audit["quality"], list)

    status, events = console.get(f"/api/run/events?project={PROJECT_ID.root}&run_id=webui-run-1&after=0")
    assert status == 200
    assert events["next_seq"] >= 1
    assert events["events"]


def test_control_cancel(console: _Client) -> None:
    console.post(
        "/api/run/start",
        {
            "project": PROJECT_ID.root,
            "run_id": "webui-cancel",
            "requirements": "写一本小说",
            "provider": "offline_fake",
            "chapter_count": 1,
        },
    )
    status, cancelled = console.post(
        "/api/run/control",
        {"project": PROJECT_ID.root, "run_id": "webui-cancel", "command": "cancel"},
    )
    # The offline run may finish before the cancel lands; both outcomes are legal.
    assert status in (200, 500)
    if status == 200:
        assert cancelled["status"] in ("cancelled", "cancel_requested", "queued")


def test_unknown_route_returns_404(console: _Client) -> None:
    status, data = console.get("/api/not-a-route")
    assert status == 404
    assert "error" in data


def test_health_endpoint_missing_key(console: _Client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY_1", raising=False)

    status, data = console.get("/api/health")

    assert status == 200
    assert data["ok"] is False
    assert data["reason"] == "missing_key"


def test_health_endpoint_ok(console: _Client, monkeypatch: pytest.MonkeyPatch) -> None:
    from novel_agent.webui import server as server_module

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    calls: list[str] = []
    monkeypatch.setattr(
        "tools_call_deepseek.access_deepseek",
        lambda prompt, **kwargs: calls.append(prompt) or "正常",
    )

    status, data = console.get("/api/health")

    assert status == 200
    assert data["ok"] is True
    assert data["latency_ms"] >= 0
    assert calls, "the health probe must make a real provider call"


def test_health_endpoint_provider_failure(console: _Client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    def failing(_prompt, **_kwargs):
        raise RuntimeError("injected provider failure")

    monkeypatch.setattr("tools_call_deepseek.access_deepseek", failing)

    status, data = console.get("/api/health")

    assert status == 200
    assert data["ok"] is False
    assert "injected provider failure" in data["reason"]


def test_rebase_endpoint_reports_missing_proposal(console: _Client) -> None:
    status, data = console.post("/api/run/rebase", {"project": PROJECT_ID.root, "run_id": "no-such-run"})
    assert status == 500
    assert "no pending rebase proposal" in data["error"]


def test_frontend_script_ids_match_dom() -> None:
    """Every element app.js reads must exist in index.html."""
    from novel_agent.webui import server as server_module

    html = (server_module._STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (server_module._STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    import re

    referenced = set(re.findall(r'\$\("([a-z-]+)"\)', script))
    for element_id in referenced:
        assert f'id="{element_id}"' in html, f"app.js references missing element id: {element_id}"
    # The requirement input is the primary user entry point.
    assert 'id="requirements"' in html
    assert 'id="start-btn"' in html


def test_start_requires_non_blank_requirements() -> None:
    """A start with empty or whitespace-only requirements is rejected."""
    from pydantic import ValidationError

    from novel_agent.pathing import ProjectId
    from novel_agent.service_models import StartNovelRunRequest

    for blank in ("", "   ", "\t\n"):
        with pytest.raises(ValidationError):
            StartNovelRunRequest(
                project_id=ProjectId("novel-one"),
                run_id="run-blank",
                idempotency_key="start-blank",
                user_requirements=blank,
            )
    # Surrounding whitespace is trimmed, not rejected.
    request = StartNovelRunRequest(
        project_id=ProjectId("novel-one"),
        run_id="run-trim",
        idempotency_key="start-trim",
        user_requirements="  写一本小说  ",
    )
    assert request.user_requirements == "写一本小说"


def test_create_project_endpoint(console: _Client) -> None:
    """POST /api/projects/create adds a project and returns the refreshed list."""
    status, data = console.post("/api/projects/create", {"name": "brand-new-project"})
    assert status == 200
    assert data.get("project") == "brand-new-project"
    assert "brand-new-project" in data.get("projects", [])


def test_create_project_rejects_bad_name(console: _Client) -> None:
    status, data = console.post("/api/projects/create", {"name": "../escape"})
    assert status == 200  # handled as an error payload, not a crash
    assert "error" in data
    assert data["error"]


def test_provider_select_includes_agent_loop() -> None:
    from novel_agent.webui import server as server_module

    html = (server_module._STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    assert 'value="agent_loop"' in html
    assert 'value="legacy_real"' in html
