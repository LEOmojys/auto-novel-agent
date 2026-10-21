from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from novel_agent.artifact_types import (
    ArtifactAuthorityMismatchError,
    ArtifactCommitReceipt,
    ArtifactCommitRequest,
    ArtifactFaultPoint,
    ArtifactRunAuthorityError,
    ConfirmedMemoryWrite,
    EvidenceMismatchError,
    FaultHook,
    MaterializationChecksumError,
    MaterializationReport,
    OutboxTargetMismatchError,
    no_fault,
)
from novel_agent.memory_store import index_memory
from novel_agent.pathing import (
    ArtifactExtension,
    ArtifactId,
    ArtifactName,
    ProjectId,
    ProjectPathResolver,
)
from novel_agent.storage import write_transaction

_M3_LAYER: Final = "m3_confirmed"
_CONFIRMED_STATUS: Final = "confirmed"
_ACCEPTED_STATUS: Final = "accepted"
_PENDING_STATE: Final = "pending"
_MATERIALIZED_STATE: Final = "materialized"

@dataclass(frozen=True, slots=True)
class _OutboxEntry:
    outbox_id: str
    artifact_id: ArtifactId
    revision: int
    project_id: ProjectId
    target_path: Path
    raw_bytes: bytes
    sha256: str
    extension: ArtifactExtension


def enqueue_materialization(
    connection: sqlite3.Connection,
    *,
    project_id: str,
    artifact_id: str,
    revision: int,
    raw_bytes: bytes,
    created_at: object,
    sha256: str | None = None,
    extension: ArtifactExtension = ArtifactExtension.JSON,
) -> str:
    """Enqueue a durable materialization row for one accepted artifact.

    Shared by the planning, chapter and governed commit paths so every
    accepted artifact is eventually written to `.governed/artifacts/` by
    `ArtifactService.materialize_pending`.  Idempotent on the content-addressed
    outbox id.
    """
    digest = sha256 or hashlib.sha256(raw_bytes).hexdigest()
    target_path = ProjectPathResolver.from_config().resolve_export(
        ProjectId(project_id),
        ArtifactName(
            artifact_id=ArtifactId(artifact_id),
            content_hash=digest,
            extension=extension,
        ),
    )
    outbox_id = hashlib.sha256(f"{artifact_id}:{revision}:{digest}".encode()).hexdigest()
    timestamp = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
    connection.execute(
        "INSERT INTO artifact_outbox(outbox_id, artifact_id, artifact_revision, target_path, "
        "state, created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', ?, ?) "
        "ON CONFLICT(outbox_id) DO NOTHING",
        (outbox_id, artifact_id, revision, str(target_path), timestamp, timestamp),
    )
    return outbox_id


@dataclass(frozen=True, slots=True)
class ArtifactService:
    connection: sqlite3.Connection
    path_resolver: ProjectPathResolver

    def commit(
        self,
        request: ArtifactCommitRequest,
        fault_hook: FaultHook = no_fault,
    ) -> ArtifactCommitReceipt:
        digest = hashlib.sha256(request.raw_bytes).hexdigest()
        target_path = self.path_resolver.resolve_export(
            request.project_id,
            ArtifactName(
                artifact_id=request.artifact_id,
                content_hash=digest,
                extension=request.extension,
            ),
        )
        outbox_id = hashlib.sha256(
            f"{request.artifact_id.root}:{request.revision}:{digest}".encode()
        ).hexdigest()
        for memory in request.memories:
            source = request.raw_bytes[memory.source_start : memory.source_end]
            actual = hashlib.sha256(source).hexdigest()
            if memory.source_end > len(request.raw_bytes) or actual != memory.source_sha256:
                raise EvidenceMismatchError(memory.memory_id, memory.source_sha256, actual)

        with write_transaction(self.connection):
            self._require_run_authority(request)
            self.connection.execute(
                "UPDATE artifacts SET is_current = 0 WHERE artifact_id = ? AND is_current = 1",
                (request.artifact_id.root,),
            )
            self.connection.execute(
                "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, "
                "raw_bytes, sha256, schema_version, is_current, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    request.artifact_id.root,
                    request.revision,
                    request.project_id.root,
                    request.run_id,
                    request.kind,
                    _ACCEPTED_STATUS,
                    request.raw_bytes,
                    digest,
                    request.schema_version,
                    request.created_at,
                ),
            )
            for memory in request.memories:
                self.connection.execute(
                    "INSERT INTO memory_records(memory_id, project_id, run_id, plan_revision, "
                    "layer, kind, status, subject_id, predicate, value_json, source_artifact_id, "
                    "source_artifact_revision, source_span, source_hash, confidence, created_by, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        memory.memory_id,
                        request.project_id.root,
                        request.run_id,
                        memory.plan_revision,
                        _M3_LAYER,
                        memory.kind,
                        _CONFIRMED_STATUS,
                        memory.subject_id,
                        memory.predicate,
                        memory.value_json,
                        request.artifact_id.root,
                        request.revision,
                        f"{memory.source_start}:{memory.source_end}",
                        memory.source_sha256,
                        memory.confidence,
                        memory.created_by,
                        request.created_at,
                    ),
                )
                index_memory(self.connection, memory.memory_id, memory.value_json)
            self.connection.execute(
                "INSERT INTO artifact_outbox(outbox_id, artifact_id, artifact_revision, target_path, "
                "state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    outbox_id,
                    request.artifact_id.root,
                    request.revision,
                    str(target_path),
                    _PENDING_STATE,
                    request.created_at,
                    request.created_at,
                ),
            )
            fault_hook(ArtifactFaultPoint.BEFORE_DB_COMMIT)
        fault_hook(ArtifactFaultPoint.AFTER_DB_COMMIT)
        return ArtifactCommitReceipt(request.artifact_id, request.revision, digest, target_path, outbox_id)

    def _require_run_authority(self, request: ArtifactCommitRequest) -> None:
        if request.run_id is None:
            return
        row = self.connection.execute(
            "SELECT project_id FROM runs WHERE run_id = ?",
            (request.run_id,),
        ).fetchone()
        persisted_project_id = None if row is None else str(row[0])
        if persisted_project_id != request.project_id.root:
            raise ArtifactRunAuthorityError(
                run_id=request.run_id,
                request_project_id=request.project_id.root,
                persisted_project_id=persisted_project_id,
            )

    def materialize_pending(
        self,
        fault_hook: FaultHook = no_fault,
    ) -> MaterializationReport:
        rows = self.connection.execute(
            "SELECT o.outbox_id, a.artifact_id, a.revision, a.project_id, o.target_path, "
            "a.raw_bytes, a.sha256, CASE WHEN o.target_path LIKE '%.json' THEN 'json' ELSE 'txt' END "
            "FROM artifact_outbox o JOIN artifacts a ON a.artifact_id = o.artifact_id "
            "AND a.revision = o.artifact_revision WHERE o.state = ? ORDER BY o.created_at, o.outbox_id",
            (_PENDING_STATE,),
        ).fetchall()
        materialized = 0
        already_present = 0
        failed = 0
        for row in rows:
            entry = _OutboxEntry(
                outbox_id=str(row[0]),
                artifact_id=ArtifactId(str(row[1])),
                revision=int(row[2]),
                project_id=ProjectId(str(row[3])),
                target_path=Path(str(row[4])),
                raw_bytes=bytes(row[5]),
                sha256=str(row[6]),
                extension=ArtifactExtension(str(row[7])),
            )
            try:
                was_present = self._materialize_entry(entry, fault_hook)
            except (
                ArtifactAuthorityMismatchError,
                OutboxTargetMismatchError,
                MaterializationChecksumError,
                OSError,
            ) as error:
                failed += 1
                self.connection.execute(
                    "UPDATE artifact_outbox SET attempts = attempts + 1, last_error = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE outbox_id = ? AND state = ?",
                    (str(error), entry.outbox_id, _PENDING_STATE),
                )
                continue
            self.connection.execute(
                "UPDATE artifact_outbox SET state = ?, attempts = attempts + 1, last_error = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE outbox_id = ? AND state = ?",
                (_MATERIALIZED_STATE, entry.outbox_id, _PENDING_STATE),
            )
            already_present += int(was_present)
            materialized += int(not was_present)
        return MaterializationReport(len(rows), materialized, already_present, failed)

    def _materialize_entry(self, entry: _OutboxEntry, fault_hook: FaultHook) -> bool:
        if hashlib.sha256(entry.raw_bytes).hexdigest() != entry.sha256:
            raise ArtifactAuthorityMismatchError(entry.outbox_id)
        expected_path = self.path_resolver.resolve_export(
            entry.project_id,
            ArtifactName(
                artifact_id=entry.artifact_id,
                content_hash=entry.sha256,
                extension=entry.extension,
            ),
        )
        if entry.target_path != expected_path:
            raise OutboxTargetMismatchError(entry.outbox_id)
        if expected_path.is_file() and hashlib.sha256(expected_path.read_bytes()).hexdigest() == entry.sha256:
            return True

        expected_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=expected_path.parent,
                prefix=f".{entry.artifact_id.root}-{entry.sha256}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(entry.raw_bytes)
                temporary.flush()
                os.fsync(temporary.fileno())
            fault_hook(ArtifactFaultPoint.AFTER_TEMP_FSYNC)
            os.replace(temporary_path, expected_path)
            fault_hook(ArtifactFaultPoint.AFTER_REPLACE)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        if hashlib.sha256(expected_path.read_bytes()).hexdigest() != entry.sha256:
            raise MaterializationChecksumError(entry.outbox_id)
        return False


__all__ = (
    "ArtifactCommitReceipt",
    "ArtifactCommitRequest",
    "ArtifactFaultPoint",
    "ArtifactRunAuthorityError",
    "ArtifactService",
    "ConfirmedMemoryWrite",
    "EvidenceMismatchError",
    "MaterializationReport",
)
