from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, Self

import config
from pydantic import BaseModel, ConfigDict, RootModel, field_validator, model_validator
from pydantic_core import PydanticCustomError


_WINDOWS_RESERVED_NAMES: Final[frozenset[str]] = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
# ``:`` is deliberately allowed: governed artifact ids are generated as
# ``{run_id}:{kind}:{suffix}`` (e.g. ``run-001:brief:world``).  Colon is not a
# path traversal or absolute-path hazard, and the export filename layer
# sanitizes it for Windows filenames.
_WINDOWS_INVALID_CHARACTERS: Final = frozenset('<>"|?*')
_SHA256_LENGTH: Final = 64


@unique
class PathingErrorCode(StrEnum):
    ABSOLUTE = "path_absolute"
    EMPTY_SEGMENT = "path_empty_segment"
    INVALID_CHARACTER = "path_invalid_character"
    INVALID_HASH = "path_invalid_hash"
    NUL = "path_nul"
    RESERVED = "path_windows_reserved"
    SEPARATOR = "path_separator"
    SYMLINK_ESCAPE = "path_symlink_escape"
    TRAILING_DOT_SPACE = "path_trailing_dot_space"
    TRAVERSAL = "path_traversal"


def _path_error(code: PathingErrorCode, message: str) -> PydanticCustomError:
    return PydanticCustomError(code.value, message)


class _LogicalIdentifier(RootModel[str]):
    model_config = ConfigDict(frozen=True, strict=True)

    @model_validator(mode="after")
    def validate_logical_identifier(self) -> Self:
        value = self.root
        if not value:
            raise _path_error(PathingErrorCode.EMPTY_SEGMENT, "identifier cannot be empty")
        if "\x00" in value:
            raise _path_error(PathingErrorCode.NUL, "identifier cannot contain NUL")

        windows_path = PureWindowsPath(value)
        if PurePosixPath(value).is_absolute() or windows_path.is_absolute() or windows_path.drive:
            raise _path_error(PathingErrorCode.ABSOLUTE, "identifier cannot be an absolute path")
        if any(ord(character) < 32 or character in _WINDOWS_INVALID_CHARACTERS for character in value):
            raise _path_error(
                PathingErrorCode.INVALID_CHARACTER,
                "identifier contains a Windows-invalid filename character",
            )

        segments = value.replace("\\", "/").split("/")
        if ".." in segments:
            raise _path_error(PathingErrorCode.TRAVERSAL, "identifier cannot contain traversal")
        if "/" in value or "\\" in value:
            raise _path_error(PathingErrorCode.SEPARATOR, "identifier cannot contain separators")
        if value.endswith((".", " ")):
            raise _path_error(
                PathingErrorCode.TRAILING_DOT_SPACE,
                "identifier cannot end with a dot or space",
            )
        if value.split(".", maxsplit=1)[0].rstrip(" ").upper() in _WINDOWS_RESERVED_NAMES:
            raise _path_error(PathingErrorCode.RESERVED, "identifier is a Windows device name")
        return self


class ProjectId(_LogicalIdentifier):
    """Project identifiers also become directory names, so ``:`` stays rejected."""

    @model_validator(mode="after")
    def validate_project_directory_name(self) -> Self:
        if ":" in self.root:
            raise _path_error(
                PathingErrorCode.INVALID_CHARACTER,
                "project identifier cannot contain ':' (directory name)",
            )
        return self


class ArtifactId(_LogicalIdentifier):
    pass


@unique
class ArtifactExtension(StrEnum):
    JSON = "json"
    TXT = "txt"


class ArtifactName(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    artifact_id: ArtifactId
    content_hash: str
    extension: ArtifactExtension

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
            raise _path_error(PathingErrorCode.INVALID_HASH, "content hash must be lowercase SHA-256")
        return value

    @property
    def filename(self) -> str:
        safe_id = self.artifact_id.root.replace(":", "_")
        return f"{safe_id}-{self.content_hash}.{self.extension.value}"


@dataclass(frozen=True, slots=True)
class PathBoundaryError(Exception):
    code: PathingErrorCode
    candidate: Path
    boundary: Path

    def __str__(self) -> str:
        return f"{self.code.value}: resolved path is outside its governed boundary"


@dataclass(frozen=True, slots=True)
class ProjectPathResolver:
    projects_root: Path

    @classmethod
    def from_config(cls) -> ProjectPathResolver:
        return cls(projects_root=Path(config.PROJECTS_DIR))

    def resolve_project(self, project_id: ProjectId) -> Path:
        projects_root = self.projects_root.resolve()
        return self._resolve_contained(projects_root, projects_root / project_id.root)

    def resolve_export(self, project_id: ProjectId, artifact_name: ArtifactName) -> Path:
        project_root = self.resolve_project(project_id)
        candidate = project_root / ".governed" / "artifacts" / artifact_name.filename
        return self._resolve_contained(project_root, candidate)

    @staticmethod
    def _resolve_contained(boundary: Path, candidate: Path) -> Path:
        resolved_boundary = boundary.resolve()
        resolved_candidate = candidate.resolve()
        if not resolved_candidate.is_relative_to(resolved_boundary):
            raise PathBoundaryError(
                code=PathingErrorCode.SYMLINK_ESCAPE,
                candidate=resolved_candidate,
                boundary=resolved_boundary,
            )
        return resolved_candidate


__all__ = (
    "ArtifactExtension",
    "ArtifactId",
    "ArtifactName",
    "PathBoundaryError",
    "PathingErrorCode",
    "ProjectId",
    "ProjectPathResolver",
)
