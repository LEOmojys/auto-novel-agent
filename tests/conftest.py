from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import pytest
import requests


class NetworkForbidden(RuntimeError):
    def __init__(self, method: str, url: str) -> None:
        super().__init__(f"Outbound HTTP is forbidden: {method} {url}")
        self.method = method
        self.url = url


@dataclass(frozen=True, slots=True)
class DeterministicFakeProvider:
    response: str = '{"status":"offline"}'

    def generate(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("Fake provider requires a prompt")
        return self.response


@dataclass(frozen=True, slots=True)
class FixedClock:
    instant: datetime = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)

    def now(self) -> datetime:
        return self.instant


@dataclass(frozen=True, slots=True)
class TemporaryProject:
    root: Path
    project_path: Path


@pytest.fixture
def fake_provider() -> DeterministicFakeProvider:
    return DeterministicFakeProvider()


@pytest.fixture
def fixed_clock() -> FixedClock:
    return FixedClock()


@pytest.fixture
def temporary_project(tmp_path: Path) -> TemporaryProject:
    root = tmp_path / "projects"
    project_path = root / "default"
    project_path.mkdir(parents=True)
    return TemporaryProject(root=root, project_path=project_path)


@pytest.fixture
def isolated_project(monkeypatch: pytest.MonkeyPatch, temporary_project: TemporaryProject) -> TemporaryProject:
    import config

    monkeypatch.setattr(config, "PROJECTS_DIR", str(temporary_project.root))
    return temporary_project


@pytest.fixture(autouse=True)
def deterministic_continuity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the LLM continuity auditor out of the deterministic test suite.

    Production defaults to the provider-backed auditor; dedicated tests opt in
    with `monkeypatch.setenv("NOVEL_AGENT_CONTINUITY_AUDIT", "llm")`.
    """
    monkeypatch.setenv("NOVEL_AGENT_CONTINUITY_AUDIT", "offline")


@pytest.fixture(autouse=True)
def block_outbound_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbid_request(
        _session: requests.Session,
        method: str,
        url: str,
        **_kwargs: str,
    ) -> NoReturn:
        raise NetworkForbidden(method=method, url=url)

    monkeypatch.setattr(requests.sessions.Session, "request", forbid_request)


@pytest.fixture()
def console(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A live web console server over an isolated project."""
    import http.client
    import json
    import threading

    import config
    from novel_agent.service import NovelAgentService
    from novel_agent.webui.server import create_console_server

    projects_root = tmp_path / "projects"
    (projects_root / "novel-one").mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    service = NovelAgentService.from_config(now=lambda: datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC))
    server = create_console_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _ConsoleClient(server.server_port)
    finally:
        server.shutdown()
        server.server_close()


class _ConsoleClient:
    def __init__(self, port: int) -> None:
        self.port = port

    def get(self, path: str):
        import http.client
        import json

        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)
        finally:
            connection.close()

    def post(self, path: str, payload: dict):
        import http.client
        import json

        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)
        finally:
            connection.close()
