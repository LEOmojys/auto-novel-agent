import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from novel_agent.pathing import (
    ArtifactExtension,
    ArtifactId,
    ArtifactName,
    PathBoundaryError,
    PathingErrorCode,
    ProjectId,
    ProjectPathResolver,
)


VALID_HASH = "a" * 64


def _validation_code(error: ValidationError) -> str:
    return error.errors()[0]["type"]


def test_valid_project_and_hash_artifact_path(tmp_path: Path) -> None:
    # Given: an isolated projects root and governed identifiers.
    projects_root = tmp_path / "data" / "projects"
    resolver = ProjectPathResolver(projects_root=projects_root)
    project_id = ProjectId("project-alpha")
    artifact_name = ArtifactName(
        artifact_id=ArtifactId("chapter-001"),
        content_hash=VALID_HASH,
        extension=ArtifactExtension.JSON,
    )

    # When: the governed artifact path is resolved.
    resolved = resolver.resolve_export(project_id, artifact_name)

    # Then: it is contained and named only from governed fields.
    expected = (
        projects_root.resolve()
        / "project-alpha"
        / ".governed"
        / "artifacts"
        / f"chapter-001-{VALID_HASH}.json"
    )
    assert resolved == expected
    assert resolved.is_relative_to(projects_root.resolve())
    assert "reader supplied finale" not in resolved.name


@pytest.mark.parametrize("value", ["..", "../project", "project/../other", r"project\..\other"])
def test_project_traversal_is_rejected(value: str) -> None:
    # Given: a project identifier containing traversal syntax.
    # When: it crosses the Pydantic boundary.
    with pytest.raises(ValidationError) as raised:
        ProjectId(value)

    # Then: traversal has a stable rejection code.
    assert _validation_code(raised.value) == PathingErrorCode.TRAVERSAL


@pytest.mark.parametrize("value", ["/tmp/project", r"C:\project", r"\\server\share\project"])
def test_absolute_project_path_is_rejected_on_every_platform(value: str) -> None:
    # Given: POSIX, drive-qualified, or UNC absolute input.
    # When: it crosses the Pydantic boundary.
    with pytest.raises(ValidationError) as raised:
        ProjectId(value)

    # Then: host-independent absolute-path validation rejects it.
    assert _validation_code(raised.value) == PathingErrorCode.ABSOLUTE


@pytest.mark.parametrize("value", ["CON", "con.txt", "CON .txt", "LPT9.log", "aux.JSON", "COM1"])
def test_windows_reserved_project_name_is_rejected(value: str) -> None:
    # Given: a Windows device name, including extension forms.
    # When: it crosses the Pydantic boundary.
    with pytest.raises(ValidationError) as raised:
        ProjectId(value)

    # Then: it is rejected even on non-Windows hosts.
    assert _validation_code(raised.value) == PathingErrorCode.RESERVED


@pytest.mark.parametrize("value", ['project:name', 'project?name', 'project"name', 'project\x1fname'])
def test_windows_invalid_filename_characters_are_rejected(value: str) -> None:
    # Given: a logical identifier containing a Windows filename metacharacter or control.
    # When: it crosses the Pydantic boundary.
    with pytest.raises(ValidationError) as raised:
        ProjectId(value)

    # Then: Windows-incompatible names have a stable rejection code.
    assert _validation_code(raised.value) == PathingErrorCode.INVALID_CHARACTER


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("", PathingErrorCode.EMPTY_SEGMENT),
        ("project/name", PathingErrorCode.SEPARATOR),
        (r"project\name", PathingErrorCode.SEPARATOR),
        ("project\x00name", PathingErrorCode.NUL),
        ("project.", PathingErrorCode.TRAILING_DOT_SPACE),
        ("project ", PathingErrorCode.TRAILING_DOT_SPACE),
    ],
)
def test_malformed_project_identifier_has_typed_rejection(value: str, code: str) -> None:
    # Given: malformed logical identifier input.
    # When: it crosses the Pydantic boundary.
    with pytest.raises(ValidationError) as raised:
        ProjectId(value)

    # Then: the exact malformed-input category is observable.
    assert _validation_code(raised.value) == code


@pytest.mark.parametrize("value", ["../chapter", "/chapter", r"C:\chapter", "NUL.txt"])
def test_artifact_logical_name_reuses_the_safe_identifier_boundary(value: str) -> None:
    # Given: an unsafe logical artifact identifier.
    # When: it crosses the Pydantic boundary.
    with pytest.raises(ValidationError):
        ArtifactId(value)

    # Then: no ArtifactId is constructed.


def test_invalid_hash_is_rejected_before_path_resolution() -> None:
    # Given: a logical artifact name with a non-SHA-256 digest.
    # When: it crosses the Pydantic boundary.
    with pytest.raises(ValidationError) as raised:
        ArtifactName(
            artifact_id=ArtifactId("chapter-001"),
            content_hash="reader supplied finale.txt",
            extension=ArtifactExtension.TXT,
        )

    # Then: the digest has a typed rejection code.
    assert _validation_code(raised.value) == PathingErrorCode.INVALID_HASH


def test_raw_user_filename_is_not_part_of_logical_artifact_naming() -> None:
    # Given: otherwise valid governed fields plus raw user filename text.
    raw_artifact = {
        "artifact_id": ArtifactId("chapter-001"),
        "content_hash": VALID_HASH,
        "extension": ArtifactExtension.JSON,
        "user_filename": "reader supplied finale.txt",
    }

    # When: the artifact naming boundary parses the input.
    with pytest.raises(ValidationError) as raised:
        ArtifactName.model_validate(raw_artifact)

    # Then: raw filename text is forbidden rather than entering an output name.
    assert _validation_code(raised.value) == "extra_forbidden"


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    # Given: a project entry symlinked outside the configured root.
    projects_root = tmp_path / "projects"
    outside = tmp_path / "outside"
    projects_root.mkdir()
    outside.mkdir()
    link = projects_root / "project-alpha"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            check=True,
            capture_output=True,
        )
    resolver = ProjectPathResolver(projects_root=projects_root)

    # When: a governed export is resolved through the symlink.
    with pytest.raises(PathBoundaryError) as raised:
        resolver.resolve_export(
            ProjectId("project-alpha"),
            ArtifactName(
                artifact_id=ArtifactId("chapter-001"),
                content_hash=VALID_HASH,
                extension=ArtifactExtension.JSON,
            ),
        )

    # Then: resolution fails with the symlink-escape code.
    assert raised.value.code is PathingErrorCode.SYMLINK_ESCAPE


def test_resolver_reads_the_current_configured_projects_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: the configured projects root is replaced after module import.
    import config

    projects_root = tmp_path / "configured-projects"
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))

    # When: a resolver is created from configuration.
    resolver = ProjectPathResolver.from_config()
    resolved = resolver.resolve_project(ProjectId("default"))

    # Then: it uses the fixture root rather than a stale import-time value.
    assert resolved == Path(config.PROJECTS_DIR).resolve() / "default"
