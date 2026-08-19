from __future__ import annotations

import hashlib
import json

from novel_agent.chapter_commit_models import ChapterCommitContext
from novel_agent.chapter_models import ChapterAcceptanceRequest


def insert_chapter_audit_manifest(
    context: ChapterCommitContext,
    request: ChapterAcceptanceRequest,
    result_sha256: str,
) -> None:
    payload = {
        "adapter_version": request.staged.adapter_version,
        "artifact_revision": request.staged.revision,
        "chapter_plan_id": request.chapter_plan.chapter_plan_id,
        "context_digest": request.context_digest,
        "model_name": "not-recorded",
        "parameters": {
            "generation_attempt": request.staged.generation_attempt,
            "parse_route": request.staged.parse_route,
        },
        "prompt_sha256": request.context_digest,
        "prompt_template_sha256": ", ".join(sorted(pin.template_hash for pin in request.skills)) or "not-recorded",
        "quality": request.decision.model_dump(mode="json"),
        "raw_sha256": result_sha256,
        "response_sha256": result_sha256,
        "schema_version": 1,
        "skills": [
            {"manifest_hash": pin.manifest_hash, "skill_ref": pin.manifest.skill_ref, "template_hash": pin.template_hash}
            for pin in request.skills
        ],
    }
    manifest_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(manifest_json.encode()).hexdigest()
    manifest_id = hashlib.sha256(f"{request.run_id}:{request.task_id}:{digest}".encode()).hexdigest()
    context.connection.execute(
        "INSERT INTO audit_manifests(manifest_id, run_id, schema_version, manifest_json, sha256, created_at) "
        "VALUES (?, ?, 1, ?, ?, ?)",
        (manifest_id, request.run_id, manifest_json, digest, context.committed_at.isoformat()),
    )


__all__ = ("insert_chapter_audit_manifest",)
