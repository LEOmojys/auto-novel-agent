from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
GOVERNED = ROOT / "novel_agent"


def test_governed_code_has_no_legacy_direct_writer_calls() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in GOVERNED.rglob("*.py")
        if path.relative_to(GOVERNED).as_posix() not in ("adapters/legacy.py", "export.py")
    )
    # export.py is the sanctioned legacy-format compatibility writer: it emits
    # the old file layout (including the plot_history key) from SQLite but
    # never invokes the legacy save functions.
    forbidden = ("writting_novel", "novel_save", "save_character", "plot_history")
    assert not any(name in source for name in forbidden)


def test_canonical_ui_uses_service_boundary() -> None:
    source = (ROOT / "api_v1.py").read_text(encoding="utf-8")
    assert "NovelAgentService" in source
    assert "sqlite3" not in source
    assert "requests" not in source


def test_provider_attempt_audit_has_one_authoritative_writer() -> None:
    writers: list[str] = []
    for path in GOVERNED.rglob("*.py"):
        relative = path.relative_to(GOVERNED).as_posix()
        if relative.startswith("migrations/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "INSERT INTO provider_calls" in node.value
            for node in ast.walk(tree)
        ):
            writers.append(relative)
    assert writers == ["run_coordinator.py"]
