from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from novel_agent.skill_manifests.models import ActiveSkillPin


@dataclass(frozen=True, slots=True)
class PlanningAuditRequest:
    run_id: str
    brief_id: str
    plan_id: str
    plan_revision: int
    artifacts: tuple[tuple[str, str], ...]
    skills: tuple[ActiveSkillPin, ...]
    created_at: datetime
    model_name: str = "not-recorded"
    parameters: tuple[tuple[str, str], ...] = ()
    prompt_sha256: str = ""
    response_sha256: str = ""
    adapter_version: str = "not-recorded"
    artifact_revision: int = 1


def insert_planning_audit_manifest(connection: sqlite3.Connection, request: PlanningAuditRequest) -> None:
    payload = {
        "adapter_version": request.adapter_version,
        "artifacts": [{"artifact_id": artifact_id, "sha256": sha256} for artifact_id, sha256 in request.artifacts],
        "artifact_revision": request.artifact_revision,
        "brief_id": request.brief_id,
        "plan_id": request.plan_id,
        "plan_revision": request.plan_revision,
        "model_name": request.model_name,
        "parameters": dict(request.parameters),
        "prompt_sha256": request.prompt_sha256 or "not-recorded",
        "prompt_template_sha256": ", ".join(
            sorted(pin.template_hash for pin in request.skills)
        ) or "not-recorded",
        "response_sha256": request.response_sha256 or "not-recorded",
        "schema_version": 1,
        "skills": [
            {"manifest_hash": pin.manifest_hash, "skill_ref": pin.manifest.skill_ref, "template_hash": pin.template_hash}
            for pin in request.skills
        ],
    }
    manifest_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(manifest_json.encode()).hexdigest()
    manifest_id = hashlib.sha256(
        f"{request.run_id}:{request.plan_id}:{request.plan_revision}:{digest}".encode()
    ).hexdigest()
    connection.execute(
        "INSERT INTO audit_manifests(manifest_id, run_id, schema_version, manifest_json, sha256, created_at) "
        "VALUES (?, ?, 1, ?, ?, ?)",
        (manifest_id, request.run_id, manifest_json, digest, request.created_at.isoformat()),
    )


__all__ = ("PlanningAuditRequest", "insert_planning_audit_manifest")
