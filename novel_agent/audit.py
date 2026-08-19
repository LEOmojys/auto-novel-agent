from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Final

from novel_agent.pathing import ArtifactId, ProjectId


@dataclass(frozen=True, slots=True)
class ArtifactView:
    artifact_id: ArtifactId
    revision: int
    kind: str
    status: str
    sha256: str
    raw_text: str


@dataclass(frozen=True, slots=True)
class AuditManifestView:
    manifest_id: str
    sha256: str
    manifest_json: str
    created_at: str


@dataclass(frozen=True, slots=True)
class AuditManifestProjection:
    model: str
    parameters: str
    prompt_template_hash: str
    response_hash: str
    adapter_skill_version: str
    artifact_revision: str


_SAFE_PARAMETER_KEYS: Final = frozenset(
    {
        "connect_timeout_seconds",
        "logical_deadline_seconds",
        "max_tokens",
        "maximum_attempts",
        "read_timeout_seconds",
        "temperature",
        "top_p",
    }
)


@dataclass(frozen=True, slots=True)
class QualityReportView:
    report_id: str
    decision: str
    report_json: str
    created_at: str


@dataclass(slots=True)  # noqa: MUTABLE_OK
class ArtifactNotFoundError(Exception):
    artifact_id: ArtifactId
    revision: int

    def __str__(self) -> str:
        return f"governed artifact is not available: {self.artifact_id.root}@{self.revision}"


def read_artifact(
    connection: sqlite3.Connection,
    project_id: ProjectId,
    run_id: str,
    artifact_id: ArtifactId,
    revision: int,
) -> ArtifactView:
    row = connection.execute(
        "SELECT kind, status, sha256, raw_bytes FROM artifacts WHERE project_id = ? AND run_id = ? "
        "AND artifact_id = ? AND revision = ?",
        (project_id.root, run_id, artifact_id.root, revision),
    ).fetchone()
    if row is None:
        raise ArtifactNotFoundError(artifact_id, revision)
    return ArtifactView(
        artifact_id=artifact_id,
        revision=revision,
        kind=str(row[0]),
        status=str(row[1]),
        sha256=str(row[2]),
        raw_text=bytes(row[3]).decode("utf-8", errors="replace"),
    )


def read_artifacts(
    connection: sqlite3.Connection,
    project_id: ProjectId,
    run_id: str,
) -> tuple[ArtifactView, ...]:
    _require_run_project(connection, project_id, run_id)
    rows = connection.execute(
        "SELECT artifact_id, revision, kind, status, sha256, raw_bytes FROM artifacts "
        "WHERE project_id = ? AND run_id = ? ORDER BY created_at, artifact_id, revision",
        (project_id.root, run_id),
    ).fetchall()
    return tuple(
        ArtifactView(
            artifact_id=ArtifactId(str(row[0])),
            revision=int(row[1]),
            kind=str(row[2]),
            status=str(row[3]),
            sha256=str(row[4]),
            raw_text=bytes(row[5]).decode("utf-8", errors="replace"),
        )
        for row in rows
    )


def read_audit_manifests(
    connection: sqlite3.Connection,
    project_id: ProjectId,
    run_id: str,
) -> tuple[AuditManifestView, ...]:
    _require_run_project(connection, project_id, run_id)
    rows = connection.execute(
        "SELECT manifest_id, sha256, manifest_json, created_at FROM audit_manifests WHERE run_id = ? "
        "ORDER BY created_at, manifest_id",
        (run_id,),
    ).fetchall()
    return tuple(
        AuditManifestView(str(row[0]), str(row[1]), str(row[2]), str(row[3])) for row in rows
    )


def read_quality_reports(
    connection: sqlite3.Connection,
    project_id: ProjectId,
    run_id: str,
) -> tuple[QualityReportView, ...]:
    _require_run_project(connection, project_id, run_id)
    rows = connection.execute(
        "SELECT report_id, decision, report_json, created_at FROM quality_reports WHERE run_id = ? "
        "ORDER BY created_at, report_id",
        (run_id,),
    ).fetchall()
    return tuple(QualityReportView(str(row[0]), str(row[1]), str(row[2]), str(row[3])) for row in rows)


def project_audit_manifest(manifest_json: str) -> AuditManifestProjection:
    payload = json.loads(manifest_json)
    if not isinstance(payload, dict):
        return _empty_manifest_projection()
    skills = payload.get("skills")
    skill_versions = (
        tuple(
            str(skill.get("skill_ref"))
            for skill in skills
            if isinstance(skill, dict) and skill.get("skill_ref") is not None
        )
        if isinstance(skills, list)
        else ()
    )
    template_hashes = (
        tuple(
            str(skill.get("template_hash"))
            for skill in skills
            if isinstance(skill, dict) and skill.get("template_hash") is not None
        )
        if isinstance(skills, list)
        else ()
    )
    parameters = payload.get("parameters")
    safe_parameters = (
        {key: value for key, value in parameters.items() if key in _SAFE_PARAMETER_KEYS}
        if isinstance(parameters, dict)
        else {key: payload[key] for key in _SAFE_PARAMETER_KEYS if key in payload}
    )
    adapter_version = _first_text(payload, "adapter_version")
    adapter_and_skills = tuple(item for item in (adapter_version, *skill_versions) if item != "not recorded")
    prompt_hash = _first_text(payload, "prompt_template_sha256", "prompt_sha256", "request_sha256")
    if prompt_hash == "not recorded" and template_hashes:
        prompt_hash = ", ".join(template_hashes)
    return AuditManifestProjection(
        model=_first_text(payload, "model_name", "model"),
        parameters=json.dumps(safe_parameters, ensure_ascii=False, sort_keys=True),
        prompt_template_hash=prompt_hash,
        response_hash=_first_text(payload, "response_sha256", "raw_sha256"),
        adapter_skill_version=", ".join(adapter_and_skills) or "not recorded",
        artifact_revision=_first_text(payload, "artifact_revision", "plan_revision"),
    )


def _first_text(payload: dict, *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return str(value)
    return "not recorded"


def _empty_manifest_projection() -> AuditManifestProjection:
    return AuditManifestProjection(
        model="not recorded",
        parameters="{}",
        prompt_template_hash="not recorded",
        response_hash="not recorded",
        adapter_skill_version="not recorded",
        artifact_revision="not recorded",
    )


def _require_run_project(connection: sqlite3.Connection, project_id: ProjectId, run_id: str) -> None:
    row = connection.execute("SELECT project_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None or str(row[0]) != project_id.root:
        raise RunProjectAuditError(project_id, run_id)


@dataclass(slots=True)  # noqa: MUTABLE_OK
class RunProjectAuditError(Exception):
    project_id: ProjectId
    run_id: str

    def __str__(self) -> str:
        return f"run does not belong to project: {self.project_id.root}/{self.run_id}"


__all__ = (
    "ArtifactNotFoundError",
    "ArtifactView",
    "AuditManifestView",
    "AuditManifestProjection",
    "QualityReportView",
    "RunProjectAuditError",
    "read_artifact",
    "read_audit_manifests",
    "read_quality_reports",
    "project_audit_manifest",
)
