from __future__ import annotations

import hashlib
import sqlite3
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

from novel_agent.artifacts import (
    ArtifactCommitRequest,
    ArtifactFaultPoint,
    ArtifactRunAuthorityError,
    ArtifactService,
    ConfirmedMemoryWrite,
    EvidenceMismatchError,
)
from novel_agent.pathing import ArtifactExtension, ArtifactId, ProjectId, ProjectPathResolver
from novel_agent.storage import connect_project_database, migrate


class SimulatedCrash(RuntimeError):
    pass


def _request(raw_bytes: bytes = b"chapter one: exact governed bytes") -> ArtifactCommitRequest:
    span = raw_bytes[0:11]
    return ArtifactCommitRequest(
        artifact_id=ArtifactId("chapter-001"),
        revision=1,
        project_id=ProjectId("novel-one"),
        run_id=None,
        kind="chapter",
        raw_bytes=raw_bytes,
        extension=ArtifactExtension.TXT,
        schema_version=1,
        created_at="2030-01-02T03:04:05+00:00",
        memories=(
            ConfirmedMemoryWrite(
                memory_id="memory-001",
                plan_revision=1,
                kind="event",
                subject_id="character-001",
                predicate="appears",
                value_json='{"chapter":1}',
                source_start=0,
                source_end=11,
                source_sha256=hashlib.sha256(span).hexdigest(),
                confidence=1.0,
                created_by="coordinator",
            ),
        ),
    )


def _service(
    tmp_path: Path,
) -> tuple[ArtifactService, Path, AbstractContextManager[sqlite3.Connection]]:
    projects_root = tmp_path / "projects"
    project_dir = projects_root / "novel-one"
    connection_context = connect_project_database(project_dir)
    connection = connection_context.__enter__()
    migrate(connection)
    service = ArtifactService(connection, ProjectPathResolver(projects_root))
    return service, project_dir, connection_context


def _close(connection_context: AbstractContextManager[sqlite3.Connection]) -> None:
    connection_context.__exit__(None, None, None)


def test_commit_then_materialize(tmp_path: Path) -> None:
    # Given: a migrated project and an accepted revision with exact M3 evidence.
    service, _project_dir, connection_context = _service(tmp_path)
    request = _request()
    try:
        # When: the revision is committed, then the durable outbox is reconciled.
        receipt = service.commit(request)
        absent_before_reconcile = receipt.target_path.exists()
        report = service.materialize_pending()
        artifact_row = service.connection.execute(
            "SELECT raw_bytes, sha256, is_current FROM artifacts"
        ).fetchone()
        memory_row = service.connection.execute(
            "SELECT layer, status, source_artifact_id, source_artifact_revision, source_hash "
            "FROM memory_records"
        ).fetchone()
        outbox_row = service.connection.execute(
            "SELECT state, attempts FROM artifact_outbox"
        ).fetchone()
    finally:
        _close(connection_context)

    # Then: no early file existed and the materialized bytes match committed authority.
    assert absent_before_reconcile is False
    assert report.materialized == 1
    assert artifact_row == (request.raw_bytes, receipt.sha256, 1)
    assert memory_row == (
        "m3_confirmed",
        "confirmed",
        request.artifact_id.root,
        request.revision,
        request.memories[0].source_sha256,
    )
    assert outbox_row == ("materialized", 1)
    assert receipt.target_path.read_bytes() == request.raw_bytes
    assert hashlib.sha256(receipt.target_path.read_bytes()).hexdigest() == receipt.sha256


def test_atomic_commit_rolls_back_artifact_current_m3_and_outbox_on_crash(tmp_path: Path) -> None:
    # Given: a current revision and a fault at the next revision's transaction boundary.
    service, _project_dir, connection_context = _service(tmp_path)
    baseline = _request()
    service.commit(baseline)
    replacement = baseline.model_copy(
        update={
            "revision": 2,
            "memories": (
                baseline.memories[0].model_copy(update={"memory_id": "memory-002"}),
            ),
        }
    )

    def crash(point: ArtifactFaultPoint) -> None:
        if point is ArtifactFaultPoint.BEFORE_DB_COMMIT:
            raise SimulatedCrash("before commit")

    try:
        # When: the service reaches the commit boundary.
        with pytest.raises(SimulatedCrash, match="before commit"):
            service.commit(replacement, fault_hook=crash)
        counts = tuple(
            service.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("artifacts", "memory_records", "artifact_outbox")
        )
        current_revision = service.connection.execute(
            "SELECT revision FROM artifacts WHERE artifact_id = ? AND is_current = 1",
            (baseline.artifact_id.root,),
        ).fetchone()[0]
    finally:
        _close(connection_context)

    # Then: every authoritative row rolls back together and no file is emitted.
    assert counts == (1, 1, 1)
    assert current_revision == 1
    assert tuple(tmp_path.rglob("*.txt")) == ()


def test_crash_after_commit_can_later_materialize(tmp_path: Path) -> None:
    # Given: a revision whose transaction committed before the process stopped.
    service, _project_dir, connection_context = _service(tmp_path)

    def crash(point: ArtifactFaultPoint) -> None:
        if point is ArtifactFaultPoint.AFTER_DB_COMMIT:
            raise SimulatedCrash("after commit")

    try:
        with pytest.raises(SimulatedCrash, match="after commit"):
            service.commit(_request(), fault_hook=crash)
        target_path = Path(
            service.connection.execute("SELECT target_path FROM artifact_outbox").fetchone()[0]
        )

        # When: startup reconciliation runs later.
        report = service.materialize_pending()
    finally:
        _close(connection_context)

    # Then: the durable pending row recreates the exact view.
    assert report.materialized == 1
    assert target_path.read_bytes() == _request().raw_bytes


def test_interrupted_write_before_replace_is_recovered(tmp_path: Path) -> None:
    # Given: a committed row and a crash after its temporary file is durable.
    service, _project_dir, connection_context = _service(tmp_path)
    receipt = service.commit(_request())

    def crash(point: ArtifactFaultPoint) -> None:
        if point is ArtifactFaultPoint.AFTER_TEMP_FSYNC:
            raise SimulatedCrash("interrupted write")

    try:
        # When: the first materialization is interrupted and reconciliation retries.
        with pytest.raises(SimulatedCrash, match="interrupted write"):
            service.materialize_pending(fault_hook=crash)
        absent_after_crash = receipt.target_path.exists()
        report = service.materialize_pending()
    finally:
        _close(connection_context)

    # Then: only the final atomic name appears and contains the committed bytes.
    assert absent_after_crash is False
    assert report.materialized == 1
    assert receipt.target_path.read_bytes() == _request().raw_bytes


def test_duplicate_retry_after_replace_does_not_rewrite(tmp_path: Path) -> None:
    # Given: replacement succeeded but the outbox acknowledgement crashed.
    service, _project_dir, connection_context = _service(tmp_path)
    receipt = service.commit(_request())

    def crash(point: ArtifactFaultPoint) -> None:
        if point is ArtifactFaultPoint.AFTER_REPLACE:
            raise SimulatedCrash("after replace")

    try:
        with pytest.raises(SimulatedCrash, match="after replace"):
            service.materialize_pending(fault_hook=crash)
        first_stat = receipt.target_path.stat()

        # When: reconciliation consumes the still-pending row twice.
        first_retry = service.materialize_pending()
        second_retry = service.materialize_pending()
        second_stat = receipt.target_path.stat()
        outbox_count = service.connection.execute(
            "SELECT COUNT(*) FROM artifact_outbox"
        ).fetchone()[0]
    finally:
        _close(connection_context)

    # Then: the matching file is acknowledged without replacement or duplication.
    assert first_retry.already_present == 1
    assert second_retry.processed == 0
    assert first_stat.st_ino == second_stat.st_ino
    assert outbox_count == 1


def test_checksum_mismatch_is_rebuilt_from_committed_bytes(tmp_path: Path) -> None:
    # Given: a committed row whose governed materialization is corrupted.
    service, _project_dir, connection_context = _service(tmp_path)
    receipt = service.commit(_request())
    receipt.target_path.parent.mkdir(parents=True, exist_ok=True)
    receipt.target_path.write_bytes(b"corrupted")
    try:
        # When: reconciliation verifies the existing checksum.
        report = service.materialize_pending()
    finally:
        _close(connection_context)

    # Then: the corrupt view is atomically rebuilt from SQLite authority.
    assert report.materialized == 1
    assert receipt.target_path.read_bytes() == _request().raw_bytes
    assert hashlib.sha256(receipt.target_path.read_bytes()).hexdigest() == receipt.sha256


def test_json_materialization_uses_committed_extension(tmp_path: Path) -> None:
    # Given: an accepted JSON artifact whose suffix is longer than TXT.
    service, _project_dir, connection_context = _service(tmp_path)
    request = _request().model_copy(update={"extension": ArtifactExtension.JSON})
    try:
        # When: its committed outbox row is reconciled.
        receipt = service.commit(request)
        report = service.materialize_pending()
    finally:
        _close(connection_context)

    # Then: the hash-addressed JSON view is written and acknowledged.
    assert receipt.target_path.suffix == ".json"
    assert report.materialized == 1
    assert receipt.target_path.read_bytes() == request.raw_bytes


def test_malformed_evidence_checksum_rejects_whole_commit(tmp_path: Path) -> None:
    # Given: M3 evidence whose digest does not match its exact artifact span.
    service, _project_dir, connection_context = _service(tmp_path)
    request = _request().model_copy(
        update={
            "memories": (
                _request().memories[0].model_copy(update={"source_sha256": "0" * 64}),
            )
        }
    )
    try:
        # When: the atomic commit validates evidence against authoritative bytes.
        with pytest.raises(EvidenceMismatchError):
            service.commit(request)
        artifact_count = service.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    finally:
        _close(connection_context)

    # Then: malformed evidence admits no partial artifact state.
    assert artifact_count == 0


def test_cross_project_run_rejects_without_persisting_any_artifact_state(tmp_path: Path) -> None:
    # Given: a persisted run owned by project A and a request claiming project B.
    service, _project_dir, connection_context = _service(tmp_path)
    service.connection.execute(
        "INSERT INTO runs(run_id, project_id, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("run-a", "project-a", "running", "2030-01-02T03:04:05+00:00", "2030-01-02T03:04:05+00:00"),
    )
    try:
        # When: the artifact commit crosses the persisted run/project boundary.
        with pytest.raises(ArtifactRunAuthorityError):
            service.commit(_request().model_copy(update={"project_id": ProjectId("project-b"), "run_id": "run-a"}))
        counts = tuple(
            service.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("artifacts", "memory_records", "artifact_outbox")
        )
    finally:
        _close(connection_context)

    # Then: the rejected request leaves every artifact-side table untouched.
    assert counts == (0, 0, 0)


def test_stale_outbox_target_outside_governed_path_is_rejected(tmp_path: Path) -> None:
    # Given: a committed outbox row tampered to point into a legacy source directory.
    service, project_dir, connection_context = _service(tmp_path)
    receipt = service.commit(_request())
    legacy_target = project_dir / "novel" / "01_chapter.txt"
    service.connection.execute(
        "UPDATE artifact_outbox SET target_path = ?",
        (str(legacy_target),),
    )
    try:
        # When: reconciliation derives the expected governed path from database identity.
        report = service.materialize_pending()
        state = service.connection.execute(
            "SELECT state, attempts, last_error FROM artifact_outbox"
        ).fetchone()
    finally:
        _close(connection_context)

    # Then: it records a failed attempt without touching either target.
    assert report.failed == 1
    assert state[0:2] == ("pending", 1)
    assert "outbox_target_mismatch" in state[2]
    assert legacy_target.exists() is False
    assert receipt.target_path.exists() is False


def test_scheduler_drains_committed_outbox(tmp_path, monkeypatch) -> None:
    """The scheduler materializes pending outbox rows into .governed/artifacts."""
    import hashlib

    import config
    from novel_agent.artifacts import enqueue_materialization
    from novel_agent.local_run_scheduler import LocalRunScheduler
    from novel_agent.pathing import (
        ArtifactExtension,
        ArtifactId,
        ArtifactName,
        ProjectId,
        ProjectPathResolver,
    )
    from tests.test_workflow_gateway import FIXED_TIME

    projects_root = tmp_path / "projects"
    project_root = projects_root / "novel-one"
    project_root.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_DIR", str(projects_root))
    connection_context = connect_project_database(project_root)
    connection = connection_context.__enter__()
    try:
        migrate(connection)
        raw = b'{"body":"materialize me"}'
        digest = hashlib.sha256(raw).hexdigest()
        connection.execute(
            "INSERT INTO artifacts(artifact_id, revision, project_id, run_id, kind, status, raw_bytes, "
            "sha256, schema_version, is_current, created_at) VALUES ('a1', 1, 'novel-one', NULL, "
            "'chapter', 'accepted', ?, ?, 1, 1, ?)",
            (raw, digest, FIXED_TIME.isoformat()),
        )
        enqueue_materialization(
            connection,
            project_id="novel-one",
            artifact_id="a1",
            revision=1,
            raw_bytes=raw,
            created_at=FIXED_TIME,
        )
        assert connection.execute(
            "SELECT state FROM artifact_outbox WHERE artifact_id = 'a1'"
        ).fetchone() == ("pending",)

        LocalRunScheduler._sync_exports(connection, ProjectId("novel-one"))

        assert connection.execute(
            "SELECT state FROM artifact_outbox WHERE artifact_id = 'a1'"
        ).fetchone() == ("materialized",)
        target = ProjectPathResolver.from_config().resolve_export(
            ProjectId("novel-one"),
            ArtifactName(
                artifact_id=ArtifactId("a1"),
                content_hash=digest,
                extension=ArtifactExtension.JSON,
            ),
        )
        assert target.read_bytes() == raw
    finally:
        connection_context.__exit__(None, None, None)
