from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from novel_agent.legacy_import import IMPORT_SCHEMA_VERSION, import_legacy_project
from novel_agent.pathing import ProjectId, ProjectPathResolver
from novel_agent.storage import connect_project_database, migrate


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_project"


def _byte_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_import_is_idempotent_and_lossless(tmp_path: Path) -> None:
    # Given: a copied legacy project and a byte manifest from every fixture file.
    projects_root = tmp_path / "projects"
    project_root = projects_root / "legacy-one"
    shutil.copytree(FIXTURE_ROOT, project_root)
    before = _byte_manifest(project_root)
    resolver = ProjectPathResolver(projects_root=projects_root)
    project_id = ProjectId("legacy-one")

    # When: the same project is imported twice into its real SQLite journal.
    with connect_project_database(project_root) as connection:
        migrate(connection)
        first = import_legacy_project(connection, resolver, project_id)
        first_rows = connection.execute(
            "SELECT import_id, source_relative_path, raw_bytes, sha256, parsed_json "
            "FROM legacy_imports ORDER BY source_relative_path"
        ).fetchall()
        first_issues = connection.execute(
            "SELECT issue_id, issue_code, detail_json FROM import_issues ORDER BY issue_id"
        ).fetchall()
        second = import_legacy_project(connection, resolver, project_id)
        second_rows = connection.execute(
            "SELECT import_id, source_relative_path, raw_bytes, sha256, parsed_json "
            "FROM legacy_imports ORDER BY source_relative_path"
        ).fetchall()
        second_issues = connection.execute(
            "SELECT issue_id, issue_code, detail_json FROM import_issues ORDER BY issue_id"
        ).fetchall()

    # Then: IDs and hashes are deterministic, retry is a no-op, and sources are untouched.
    assert first.imported_count == 8
    assert first.issue_count == 3
    assert second.imported_count == 0
    assert second.issue_count == 0
    assert second_rows == first_rows
    assert second_issues == first_issues
    after = _byte_manifest(project_root)
    assert {relative: after[relative] for relative in before} == before
    assert all(raw == (project_root / relative).read_bytes() for _, relative, raw, _, _ in first_rows)
    assert (
        project_root / "characters" / "02_duplicate.json"
    ).read_bytes() == (project_root / "characters" / "01_hero.json").read_bytes()
    assert all(import_id == f"legacy:{kind}:{digest}" for import_id, kind, digest in connectionless_ids(first_rows))
    assert "ignored-root.txt" not in {row[1] for row in first_rows}
    assert len({row[3] for row in first_rows}) == len(first_rows)


def connectionless_ids(rows: list[tuple[str, str, bytes, str, str | None]]) -> list[tuple[str, str, str]]:
    kinds_by_root = {
        "00_故事大纲.txt": "outline",
        "worlds": "world",
        "characters": "character",
        "novel": "novel",
        "evaluate": "evaluation",
    }
    return [
        (import_id, kinds_by_root[relative.split("/", maxsplit=1)[0]], digest)
        for import_id, relative, _, digest, _ in rows
    ]


def test_malformed_json_is_retained_with_structured_issue(tmp_path: Path) -> None:
    # Given: a copied project containing malformed character JSON.
    projects_root = tmp_path / "projects"
    project_root = projects_root / "legacy-one"
    shutil.copytree(FIXTURE_ROOT, project_root)
    malformed_path = project_root / "characters" / "04_malformed.json"
    malformed_bytes = malformed_path.read_bytes()

    # When: the legacy project is imported.
    with connect_project_database(project_root) as connection:
        migrate(connection)
        import_legacy_project(
            connection,
            ProjectPathResolver(projects_root=projects_root),
            ProjectId("legacy-one"),
        )
        imported = connection.execute(
            "SELECT raw_bytes, parsed_json FROM legacy_imports WHERE source_relative_path = ?",
            ("characters/04_malformed.json",),
        ).fetchone()
        issue = connection.execute(
            "SELECT issue_code, detail_json FROM import_issues WHERE issue_code = ?",
            ("malformed_json",),
        ).fetchone()

    # Then: raw bytes survive, no repaired view exists, and the error is machine-readable.
    assert imported == (malformed_bytes, None)
    assert issue[0] == "malformed_json"
    assert json.loads(issue[1]) == {"source_relative_path": "characters/04_malformed.json"}
    assert malformed_path.read_bytes() == malformed_bytes


def test_duplicate_content_collision_and_unmappable_character_are_reported(
    tmp_path: Path,
) -> None:
    # Given: duplicate character bytes and a record without explicit identity.
    projects_root = tmp_path / "projects"
    project_root = projects_root / "legacy-one"
    shutil.copytree(FIXTURE_ROOT, project_root)

    # When: the project is imported.
    with connect_project_database(project_root) as connection:
        migrate(connection)
        import_legacy_project(
            connection,
            ProjectPathResolver(projects_root=projects_root),
            ProjectId("legacy-one"),
        )
        character_paths = connection.execute(
            "SELECT source_relative_path FROM legacy_imports WHERE kind = 'character' "
            "ORDER BY source_relative_path"
        ).fetchall()
        issues = connection.execute(
            "SELECT issue_code, detail_json FROM import_issues "
            "WHERE issue_code IN ('duplicate_content_collision', 'unmappable_character') "
            "ORDER BY issue_code"
        ).fetchall()

    # Then: no identity is guessed and the duplicate path is described without a second row.
    assert character_paths == [
        ("characters/01_hero.json",),
        ("characters/03_unknown.json",),
        ("characters/04_malformed.json",),
    ]
    assert [row[0] for row in issues] == [
        "duplicate_content_collision",
        "unmappable_character",
    ]
    assert json.loads(issues[0][1]) == {
        "source_relative_path": "characters/02_duplicate.json",
        "stored_source_relative_path": "characters/01_hero.json",
    }
    assert json.loads(issues[1][1]) == {
        "source_relative_path": "characters/03_unknown.json"
    }
    assert IMPORT_SCHEMA_VERSION == 1


def test_source_path_collision_preserves_first_import(tmp_path: Path) -> None:
    # Given: a project whose world file has already been imported.
    projects_root = tmp_path / "projects"
    project_root = projects_root / "legacy-one"
    shutil.copytree(FIXTURE_ROOT, project_root)
    resolver = ProjectPathResolver(projects_root=projects_root)
    project_id = ProjectId("legacy-one")
    source = project_root / "worlds" / "setting.txt"

    # When: that legacy path changes and import is retried twice.
    with connect_project_database(project_root) as connection:
        migrate(connection)
        import_legacy_project(connection, resolver, project_id)
        stored_before = connection.execute(
            "SELECT raw_bytes, sha256 FROM legacy_imports WHERE source_relative_path = ?",
            ("worlds/setting.txt",),
        ).fetchone()
        source.write_bytes(b"changed legacy bytes")
        changed = import_legacy_project(connection, resolver, project_id)
        repeated = import_legacy_project(connection, resolver, project_id)
        stored_after = connection.execute(
            "SELECT raw_bytes, sha256 FROM legacy_imports WHERE source_relative_path = ?",
            ("worlds/setting.txt",),
        ).fetchone()
        issues = connection.execute(
            "SELECT issue_code FROM import_issues WHERE issue_code = 'source_path_collision'"
        ).fetchall()

    # Then: stale authoritative bytes stay intact and the stable issue is inserted once.
    assert changed == type(changed)(imported_count=0, issue_count=1)
    assert repeated == type(repeated)(imported_count=0, issue_count=0)
    assert stored_after == stored_before
    assert issues == [("source_path_collision",)]


def test_symlinked_source_is_not_read(tmp_path: Path) -> None:
    # Given: a legacy source directory link that points outside the validated project.
    projects_root = tmp_path / "projects"
    project_root = projects_root / "legacy-one"
    project_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.json"
    secret.write_bytes(b'{"outside":true}')
    linked_worlds = project_root / "worlds"
    try:
        linked_worlds.symlink_to(outside, target_is_directory=True)
    except OSError:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked_worlds), str(outside)],
            check=True,
            capture_output=True,
        )

    # When: import enumerates the project through the validated resolver.
    with connect_project_database(project_root) as connection:
        migrate(connection)
        report = import_legacy_project(
            connection,
            ProjectPathResolver(projects_root=projects_root),
            ProjectId("legacy-one"),
        )
        imports = connection.execute("SELECT source_relative_path FROM legacy_imports").fetchall()
        issues = connection.execute(
            "SELECT issue_code, detail_json FROM import_issues"
        ).fetchall()

    # Then: external bytes never enter the journal and the rejected root is explicit.
    assert report.imported_count == 0
    assert report.issue_count == 1
    assert imports == []
    assert issues[0][0] == "unsafe_source_link"
    assert json.loads(issues[0][1]) == {"source_relative_path": "worlds"}
    assert secret.read_bytes() == b'{"outside":true}'
