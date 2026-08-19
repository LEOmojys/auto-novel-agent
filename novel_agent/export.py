"""Legacy-format export: materialize governed SQLite output as the old file
layout (worlds/*.txt, characters/*.json, novel/*.json, 00_故事大纲.txt).

The governed store remains authoritative; these files are a compatibility
view that follows the database.  Exports are triggered automatically when a
run reaches a terminal state and on demand through the web console.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from novel_agent.pathing import ProjectId, ProjectPathResolver

_LEGACY_KINDS: Final[tuple[str, ...]] = (
    "world_bible",
    "character_set",
    "plan_contract",
    "chapter",
)


@dataclass(frozen=True, slots=True)
class ExportReport:
    project_id: str
    files: tuple[str, ...] = ()


def export_project(
    connection: sqlite3.Connection,
    project_id: ProjectId,
    resolver: ProjectPathResolver | None = None,
) -> ExportReport:
    """Write legacy-format JSON/TXT files for every accepted artifact.

    Files are overwritten on each call, so exports always follow the latest
    SQLite state.  The legacy directories are created if missing.  The
    project root is derived from the connection itself when no resolver is
    given, so exports never depend on mutable global configuration.
    """
    if resolver is not None:
        project_root = resolver.resolve_project(project_id)
    else:
        row = connection.execute("PRAGMA database_list").fetchone()
        project_root = Path(str(row[2])).resolve().parent
    worlds_dir = project_root / "worlds"
    characters_dir = project_root / "characters"
    novel_dir = project_root / "novel"
    worlds_dir.mkdir(parents=True, exist_ok=True)
    characters_dir.mkdir(parents=True, exist_ok=True)
    novel_dir.mkdir(parents=True, exist_ok=True)

    artifacts = connection.execute(
        "SELECT artifact_id, revision, kind, raw_bytes FROM artifacts "
        "WHERE project_id = ? AND status = 'accepted' AND is_current = 1 "
        "ORDER BY created_at, artifact_id",
        (project_id.root,),
    ).fetchall()
    summaries = _chapter_summaries(connection)
    events_by_character = _character_events(connection)
    chapter_index = _chapter_index(connection)

    written: list[str] = []
    for artifact_id, revision, kind, raw_bytes in artifacts:
        kind = str(kind)
        if kind not in _LEGACY_KINDS:
            continue
        raw = bytes(raw_bytes)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if kind != "chapter":
                continue
            # Legacy plain-text chapter (raw is the verbatim prose, not a
            # JSON envelope): rebuild the envelope so it exports like any
            # other chapter.
            data = _plain_text_chapter_envelope(connection, str(artifact_id), raw)
            if data is None:
                continue
        if kind == "world_bible":
            written.append(_write_world(worlds_dir, data))
        elif kind == "character_set":
            written.extend(_write_characters(characters_dir, data, events_by_character))
        elif kind == "plan_contract":
            written.append(_write_outline(novel_dir, data))
        elif kind == "chapter":
            written.append(_write_chapter(novel_dir, data, summaries, chapter_index))
    return ExportReport(project_id=project_id.root, files=tuple(written))


def _plain_text_chapter_envelope(
    connection: sqlite3.Connection,
    artifact_id: str,
    raw_bytes: bytes,
) -> dict | None:
    """Rebuild a chapter envelope from a legacy plain-text chapter.

    The artifact id encodes ``sha256(run_id:chapter_plan_id)[:24]``; resolve
    the plan id by matching the id suffix against every chapter of the
    artifact's run plan.
    """
    import hashlib

    row = connection.execute(
        "SELECT run_id FROM artifacts WHERE artifact_id = ? LIMIT 1", (artifact_id,)
    ).fetchone()
    if row is None:
        return None
    run_id = str(row[0])
    suffix = artifact_id.removeprefix("chapter-")
    plans = connection.execute(
        "SELECT raw_bytes FROM artifacts WHERE kind = 'plan_contract' AND status = 'accepted' "
        "ORDER BY created_at DESC, revision DESC"
    ).fetchall()
    for (plan_bytes,) in plans:
        try:
            plan = json.loads(bytes(plan_bytes).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for chapter in plan.get("chapters") or []:
            plan_id = str(chapter.get("chapter_plan_id") or "")
            if plan_id and hashlib.sha256(f"{run_id}:{plan_id}".encode()).hexdigest()[:24] == suffix:
                return {
                    "chapter_plan_id": plan_id,
                    "body": raw_bytes.decode("utf-8", errors="replace"),
                }
    return None


# -- writers ----------------------------------------------------------------


def _safe_filename(name: str) -> str:
    """Sanitize a user/LLM-derived name for use as a file name.

    Rejects path separators and traversal, strips Windows-reserved characters,
    and caps the length so a hostile or malformed title (world title, character
    name, chapter title) can never escape the target directory or crash the
    writer with an invalid path.
    """
    cleaned = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "", name)
    cleaned = re.sub(r"\.{2,}", ".", cleaned).strip(" .")
    if not cleaned:
        return "untitled"
    return cleaned[:120]


def _write_world(worlds_dir: Path, world: dict) -> str:
    title = _safe_filename(str(world.get("title") or "world"))
    lines = [str(world.get("premise") or "")]
    rules = world.get("rules") or []
    if rules:
        lines.append("")
        lines.append("硬规则：")
        lines.extend(f"- {rule}" for rule in rules)
    constraints = world.get("constraints") or []
    if constraints:
        lines.append("")
        lines.append("约束：")
        lines.extend(f"- {constraint}" for constraint in constraints)
    target = worlds_dir / f"{title}.txt"
    target.write_text("\n".join(lines), encoding="utf-8")
    return str(target)


def _write_characters(characters_dir: Path, cast: dict, events_by_character: dict[str, list[dict]]) -> list[str]:
    written: list[str] = []
    for character in cast.get("characters") or []:
        name = _safe_filename(str(character.get("name") or "").strip())
        if not name:
            continue
        payload = {
            "name": name,
            "role_introduction": str(character.get("role") or ""),
            "role_experience": str(character.get("goal") or ""),
            "role_personality": "；".join(str(item) for item in (character.get("traits") or ())),
            "plot_history": events_by_character.get(name, []),
        }
        target = characters_dir / f"{name}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")
        written.append(str(target))
    return written


def _write_outline(novel_dir: Path, plan: dict) -> str:
    lines: list[str] = []
    for chapter in plan.get("chapters") or []:
        title = str(chapter.get("title") or f"第{chapter.get('chapter_number')}章")
        lines.append(f"{chapter.get('chapter_number')}. {title}")
        if chapter.get("goal"):
            lines.append(f"   目标：{chapter['goal']}")
        for event in chapter.get("planned_events") or []:
            lines.append(f"   - {event}")
    target = novel_dir / "00_故事大纲.txt"
    target.write_text("\n".join(lines), encoding="utf-8")
    return str(target)


def _write_chapter(
    novel_dir: Path,
    chapter: dict,
    summaries: dict[str, str],
    chapter_index: dict[str, tuple[int, str]],
) -> str:
    chapter_plan_id = str(chapter.get("chapter_plan_id") or "")
    number, title = chapter_index.get(chapter_plan_id, (0, f"第{len(chapter_index) + 1}章"))
    payload = {
        "passage": str(chapter.get("body") or ""),
        "summary": summaries.get(chapter_plan_id, ""),
    }
    target = novel_dir / f"{number} {_safe_filename(title)}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")
    return str(target)


def _chapter_summaries(connection: sqlite3.Connection) -> dict[str, str]:
    """Map chapter_plan_id -> confirmed M3 summary text."""
    rows = connection.execute(
        "SELECT subject_id, value_json FROM memory_records "
        "WHERE layer = 'm3_confirmed' AND predicate = 'summary' AND status = 'confirmed'"
    ).fetchall()
    summaries: dict[str, str] = {}
    for subject_id, value_json in rows:
        try:
            value = json.loads(str(value_json))
            text = value.get("summary", "")
        except (json.JSONDecodeError, AttributeError):
            text = ""
        if text:
            summaries[str(subject_id)] = text
    return summaries


def _character_events(connection: sqlite3.Connection) -> dict[str, list[dict]]:
    """Group confirmed DIRECT_EVENT memories by character name (chapter label -> event)."""
    rows = connection.execute(
        "SELECT subject_id, value_json FROM memory_records "
        "WHERE layer = 'm3_confirmed' AND kind = 'direct_event' AND status = 'confirmed'"
    ).fetchall()
    chapter_titles = _chapter_titles(connection)
    events: dict[str, list[dict]] = {}
    for subject_id, value_json in rows:
        try:
            value = json.loads(str(value_json))
        except (json.JSONDecodeError, AttributeError):
            continue
        chapter_number = int(value.get("chapter") or 0)
        label = chapter_titles.get(chapter_number, f"第{chapter_number}章")
        name = str(subject_id)
        events.setdefault(name, []).append({label: str(value.get("event") or value.get("object") or "")})
    return events


def _chapter_titles(connection: sqlite3.Connection) -> dict[int, str]:
    rows = connection.execute(
        "SELECT raw_bytes FROM artifacts WHERE kind = 'plan_contract' AND status = 'accepted' AND is_current = 1"
    ).fetchall()
    titles: dict[int, str] = {}
    for (raw_bytes,) in rows:
        try:
            plan = json.loads(bytes(raw_bytes).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for chapter in plan.get("chapters") or []:
            titles[int(chapter.get("chapter_number") or 0)] = str(chapter.get("title") or "")
    return titles


def _chapter_index(connection: sqlite3.Connection) -> dict[str, tuple[int, str]]:
    """Map chapter_plan_id -> (chapter_number, title) from the current plan."""
    rows = connection.execute(
        "SELECT raw_bytes FROM artifacts WHERE kind = 'plan_contract' AND status = 'accepted' AND is_current = 1"
    ).fetchall()
    index: dict[str, tuple[int, str]] = {}
    for (raw_bytes,) in rows:
        try:
            plan = json.loads(bytes(raw_bytes).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for chapter in plan.get("chapters") or []:
            plan_id = str(chapter.get("chapter_plan_id") or "")
            number = int(chapter.get("chapter_number") or 0)
            title = str(chapter.get("title") or f"第{number}章")
            index[plan_id] = (number, title)
    return index


__all__ = ("ExportReport", "export_project")
