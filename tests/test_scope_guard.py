from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from novel_agent.legacy_import import import_legacy_project
from novel_agent.pathing import ProjectId, ProjectPathResolver
from novel_agent.storage import connect_project_database, migrate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_project"
FORBIDDEN_DEPENDENCIES = (
    "langgraph",
    "pydantic-ai",
    "chromadb",
    "qdrant-client",
    "weaviate-client",
    "pinecone-client",
    "celery",
    "dramatiq",
    "rq",
)
FORBIDDEN_GOVERNED_WRITERS = (
    "writting_novel",
    "novel_save",
    "save_character",
    "plot_history",
)


def test_legacy_files_and_dependencies_stay_in_scope(tmp_path: Path) -> None:
    # Given: the real legacy fixture has an immutable byte manifest.
    before = _byte_manifest(LEGACY_FIXTURE_ROOT)
    projects_root = tmp_path / "projects"
    project_path = projects_root / "legacy-one"
    shutil.copytree(LEGACY_FIXTURE_ROOT, project_path)
    resolver = ProjectPathResolver(projects_root=projects_root)

    # When: the governed importer journals a copied legacy project twice.
    with connect_project_database(project_path) as connection:
        migrate(connection)
        first = import_legacy_project(connection, resolver, ProjectId("legacy-one"))
        second = import_legacy_project(connection, resolver, ProjectId("legacy-one"))
        imported_rows = connection.execute(
            "SELECT source_relative_path, sha256 FROM legacy_imports ORDER BY source_relative_path"
        ).fetchall()

    # Then: source bytes stay immutable, imports are idempotent, and scope stays local-only.
    assert first.imported_count == len(imported_rows)
    assert second.imported_count == 0
    assert _byte_manifest(LEGACY_FIXTURE_ROOT) == before
    assert _source_manifest(project_path, before) == before
    assert not _forbidden_dependency_names(_dependency_declarations())
    assert not _forbidden_governed_writer_references()


def test_scope_guards_detect_temporary_fixture_and_dependency_drift(tmp_path: Path) -> None:
    # Given: an isolated fixture copy and its original byte manifest.
    real_fixture_before = _byte_manifest(LEGACY_FIXTURE_ROOT)
    copied_fixture = tmp_path / "legacy-project"
    shutil.copytree(LEGACY_FIXTURE_ROOT, copied_fixture)
    before = _byte_manifest(copied_fixture)

    # When: the isolated copy and an input dependency declaration are intentionally made invalid.
    changed_path = copied_fixture / "worlds" / "setting.txt"
    changed_path.write_bytes(changed_path.read_bytes() + b"\nchanged")
    forbidden = _forbidden_dependency_names("pydantic-ai>=1")

    # Then: the guard detects both forms of drift without touching the real fixture.
    assert _byte_manifest(copied_fixture) != before
    assert forbidden == ("pydantic-ai",)
    assert _byte_manifest(LEGACY_FIXTURE_ROOT) == real_fixture_before


def _byte_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _source_manifest(project_path: Path, expected: dict[str, str]) -> dict[str, str]:
    return {
        relative_path: hashlib.sha256((project_path / relative_path).read_bytes()).hexdigest()
        for relative_path in expected
    }


def _dependency_declarations() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "requirements.txt", PROJECT_ROOT / "pyproject.toml")
    )


def _forbidden_dependency_names(declarations: str) -> tuple[str, ...]:
    lowered = declarations.casefold()
    return tuple(name for name in FORBIDDEN_DEPENDENCIES if name in lowered)


def _forbidden_governed_writer_references() -> tuple[str, ...]:
    sources = (
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "novel_agent").rglob("*.py")
        if path.relative_to(PROJECT_ROOT / "novel_agent").as_posix()
        not in ("adapters/legacy.py", "export.py")  # export.py: sanctioned legacy-format writer
    )
    joined = "\n".join(sources)
    return tuple(name for name in FORBIDDEN_GOVERNED_WRITERS if name in joined)
