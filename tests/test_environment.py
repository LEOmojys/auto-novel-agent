import importlib
import sqlite3
import sys
import tomllib
from pathlib import Path

import pydantic
import pytest
import requests

from conftest import (
    DeterministicFakeProvider,
    FixedClock,
    NetworkForbidden,
    TemporaryProject,
)


EXPECTED_DEPENDENCIES = (
    "requests>=2.28,<3",
    "gradio>=4,<7",
    "json-repair>=0.30,<1",
    "pydantic>=2,<3",
    "pytest>=8,<10",
)


def assert_governed_dependency_bounds(dependencies: tuple[str, ...]) -> None:
    assert dependencies == EXPECTED_DEPENDENCIES


def test_declared_dependency_and_python_bounds_are_exact() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    manifest = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert_governed_dependency_bounds(tuple(requirements))
    assert_governed_dependency_bounds(tuple(manifest["project"]["dependencies"]))
    assert manifest["project"]["requires-python"] == ">=3.11,<3.14"


def test_dependency_assertion_rejects_missing_requirement() -> None:
    incomplete_dependencies = EXPECTED_DEPENDENCIES[:-1]

    with pytest.raises(AssertionError):
        assert_governed_dependency_bounds(incomplete_dependencies)


def test_runtime_prerequisites_are_supported() -> None:
    assert (3, 11) <= sys.version_info[:2] < (3, 14)
    assert sqlite3.sqlite_version_info >= (3, 35, 0)
    assert pydantic.VERSION.split(".", maxsplit=1)[0] == "2"


def test_gradio_and_json_repair_import_offline() -> None:
    assert importlib.import_module("gradio") is not None
    assert importlib.import_module("json_repair") is not None


def test_fake_provider_is_used(fake_provider: DeterministicFakeProvider, fixed_clock: FixedClock) -> None:
    assert fake_provider.generate("outline") == '{"status":"offline"}'
    assert fixed_clock.now().isoformat() == "2030-01-02T03:04:05+00:00"


def test_temporary_project_is_isolated(isolated_project: TemporaryProject) -> None:
    repository_root = Path.cwd().resolve()

    assert isolated_project.project_path.is_dir()
    assert not isolated_project.project_path.resolve().is_relative_to(repository_root)


def test_api_v1_builds_offline_in_temporary_project(isolated_project: TemporaryProject) -> None:
    import api_v1

    app = api_v1.create_novel_app()

    assert type(app).__name__ == "Blocks"
    assert isolated_project.project_path.is_dir()


def test_outbound_http_is_blocked() -> None:
    with pytest.raises(NetworkForbidden, match="Outbound HTTP is forbidden"):
        requests.Session().get("https://example.invalid")
