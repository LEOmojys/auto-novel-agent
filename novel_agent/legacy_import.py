from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, JsonValue, StringConstraints, TypeAdapter, ValidationError

from novel_agent.pathing import ProjectId, ProjectPathResolver
from novel_agent.storage import write_transaction

IMPORT_SCHEMA_VERSION: Final = 1
_JSON_ADAPTER: Final = TypeAdapter(JsonValue)


class _CharacterIdentity(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)

    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@dataclass(frozen=True, slots=True)
class ImportReport:
    imported_count: int
    issue_count: int


@dataclass(frozen=True, slots=True)
class _Source:
    path: Path
    relative_path: str
    kind: str


@dataclass(frozen=True, slots=True)
class _RejectedSource:
    relative_path: str


@dataclass(frozen=True, slots=True)
class _ImportContext:
    project_id: str
    created_at: str


@dataclass(frozen=True, slots=True)
class _ImportResult:
    imported_count: int
    issue_count: int


@dataclass(frozen=True, slots=True)
class _Issue:
    import_id: str | None
    code: str
    details: Mapping[str, JsonValue]


_DIRECTORY_KINDS: Final[tuple[tuple[str, str], ...]] = (
    ("worlds", "world"),
    ("characters", "character"),
    ("novel", "novel"),
    ("evaluate", "evaluation"),
)


def import_legacy_project(
    connection: sqlite3.Connection,
    resolver: ProjectPathResolver,
    project_id: ProjectId,
) -> ImportReport:
    project_root = resolver.resolve_project(project_id)
    sources, rejected_sources = _discover_sources(project_root)
    context = _ImportContext(
        project_id=project_id.root,
        created_at=datetime.now(UTC).isoformat(),
    )
    imported_count = 0
    issue_count = 0
    with write_transaction(connection):
        for rejected in rejected_sources:
            issue_count += _insert_issue(
                connection,
                _Issue(
                    import_id=None,
                    code="unsafe_source_link",
                    details={"source_relative_path": rejected.relative_path},
                ),
                context,
            )
        for source in sources:
            result = _import_source(connection, source, context)
            imported_count += result.imported_count
            issue_count += result.issue_count
    return ImportReport(imported_count=imported_count, issue_count=issue_count)


def _discover_sources(project_root: Path) -> tuple[tuple[_Source, ...], tuple[_RejectedSource, ...]]:
    sources: list[_Source] = []
    rejected: list[_RejectedSource] = []
    outline = project_root / "00_故事大纲.txt"
    if _is_link(outline):
        rejected.append(_RejectedSource(relative_path=outline.name))
    elif outline.is_file():
        sources.append(_Source(path=outline, relative_path=outline.name, kind="outline"))

    for directory_name, kind in _DIRECTORY_KINDS:
        source_root = project_root / directory_name
        if _is_link(source_root):
            rejected.append(_RejectedSource(relative_path=directory_name))
            continue
        if not source_root.is_dir():
            continue
        for current, directory_names, file_names in os.walk(source_root, followlinks=False):
            current_path = Path(current)
            for directory_name_nested in tuple(sorted(directory_names)):
                nested = current_path / directory_name_nested
                if _is_link(nested):
                    directory_names.remove(directory_name_nested)
                    rejected.append(
                        _RejectedSource(
                            relative_path=nested.relative_to(project_root).as_posix()
                        )
                    )
            for file_name in sorted(file_names):
                candidate = current_path / file_name
                relative_path = candidate.relative_to(project_root).as_posix()
                if _is_link(candidate):
                    rejected.append(_RejectedSource(relative_path=relative_path))
                elif candidate.is_file():
                    sources.append(
                        _Source(path=candidate, relative_path=relative_path, kind=kind)
                    )
    return (
        tuple(sorted(sources, key=lambda source: source.relative_path)),
        tuple(sorted(rejected, key=lambda source: source.relative_path)),
    )


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _import_source(
    connection: sqlite3.Connection,
    source: _Source,
    context: _ImportContext,
) -> _ImportResult:
    raw_bytes = source.path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    import_id = f"legacy:{source.kind}:{digest}"
    existing_id = connection.execute(
        "SELECT source_relative_path FROM legacy_imports WHERE import_id = ?",
        (import_id,),
    ).fetchone()
    if existing_id is not None:
        stored_path = str(existing_id[0])
        if stored_path == source.relative_path:
            return _ImportResult(imported_count=0, issue_count=0)
        issue_count = _insert_issue(
            connection,
            _Issue(
                import_id=import_id,
                code="duplicate_content_collision",
                details={
                    "source_relative_path": source.relative_path,
                    "stored_source_relative_path": stored_path,
                },
            ),
            context,
        )
        return _ImportResult(imported_count=0, issue_count=issue_count)

    existing_path = connection.execute(
        "SELECT import_id, sha256 FROM legacy_imports "
        "WHERE project_id = ? AND source_relative_path = ? AND import_schema_version = ?",
        (context.project_id, source.relative_path, IMPORT_SCHEMA_VERSION),
    ).fetchone()
    if existing_path is not None:
        issue_count = _insert_issue(
            connection,
            _Issue(
                import_id=str(existing_path[0]),
                code="source_path_collision",
                details={
                    "source_relative_path": source.relative_path,
                    "stored_sha256": str(existing_path[1]),
                    "candidate_sha256": digest,
                },
            ),
            context,
        )
        return _ImportResult(imported_count=0, issue_count=issue_count)

    parsed_json, issue_code = _parse_json_view(source, raw_bytes)
    connection.execute(
        "INSERT INTO legacy_imports(import_id, project_id, kind, source_relative_path, "
        "raw_bytes, sha256, import_schema_version, parsed_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            import_id,
            context.project_id,
            source.kind,
            source.relative_path,
            raw_bytes,
            digest,
            IMPORT_SCHEMA_VERSION,
            parsed_json,
            context.created_at,
        ),
    )
    issue_count = 0
    if issue_code is not None:
        issue_count = _insert_issue(
            connection,
            _Issue(
                import_id=import_id,
                code=issue_code,
                details={"source_relative_path": source.relative_path},
            ),
            context,
        )
    return _ImportResult(imported_count=1, issue_count=issue_count)


def _parse_json_view(source: _Source, raw_bytes: bytes) -> tuple[str | None, str | None]:
    if source.path.suffix.casefold() != ".json":
        return None, None
    try:
        parsed = _JSON_ADAPTER.validate_json(raw_bytes)
    except ValidationError:
        return None, "malformed_json"
    parsed_json = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if source.kind == "character":
        try:
            _CharacterIdentity.model_validate(parsed)
        except ValidationError:
            return parsed_json, "unmappable_character"
    return parsed_json, None


def _insert_issue(
    connection: sqlite3.Connection,
    issue: _Issue,
    context: _ImportContext,
) -> int:
    detail_json = json.dumps(
        issue.details,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    identity = "\x00".join(
        (context.project_id, issue.import_id or "", issue.code, detail_json)
    )
    issue_id = f"legacy-issue:{hashlib.sha256(identity.encode()).hexdigest()}"
    cursor = connection.execute(
        "INSERT OR IGNORE INTO import_issues"
        "(issue_id, import_id, issue_code, detail_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (issue_id, issue.import_id, issue.code, detail_json, context.created_at),
    )
    return cursor.rowcount


__all__ = ("IMPORT_SCHEMA_VERSION", "ImportReport", "import_legacy_project")
