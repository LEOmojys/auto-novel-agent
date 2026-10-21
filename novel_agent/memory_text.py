"""Render layer memories into model-facing text and estimate prompt budget.

Memory rows store canonical JSON (`value_json`) for auditability, but feeding
raw JSON to the model wastes budget and reads poorly.  This module derives a
compact, human-readable rendering per (layer, kind) and provides the shared
token estimator/truncation helpers used by `ContextBuilder`.

The rendering is a pure function of stored data, so the context digest stays
deterministic and replayable; `value_json` remains the audit source of truth.
"""

from __future__ import annotations

import json
from typing import Final

from novel_agent.contracts.common import MemoryLayer

TOKENIZER_VERSION: Final = "cjk-1"

#: A fragment smaller than this is not worth truncating into the prompt.
MIN_FRAGMENT_TOKENS: Final = 24


def estimate_tokens(text: str) -> int:
    """Rough prompt-token estimate (CJK-aware): 1 per CJK char, 4 ASCII per token."""
    cjk = 0
    for character in text:
        codepoint = ord(character)
        if 0x4E00 <= codepoint <= 0x9FFF or 0x3400 <= codepoint <= 0x4DBF:
            cjk += 1
    ascii_characters = len(text) - cjk
    return max(1, cjk + ascii_characters // 4)


def truncate_text(text: str, max_tokens: int) -> str:
    """Deterministically cut `text` so its estimate (ellipsis included) fits."""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    suffix = "…"
    if estimate_tokens(suffix) > max_tokens:
        return ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle] + suffix) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    if low <= 0:
        return ""
    return text[:low].rstrip() + suffix


def _load(value_json: str) -> dict | None:
    try:
        parsed = json.loads(value_json)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (list, tuple)):
        return "、".join(item for item in (_text(item) for item in value) if item)
    if isinstance(value, dict):
        return "；".join(f"{key}: {_text(item)}" for key, item in value.items() if _text(item))
    return " ".join(str(value).split())


def _render_world(data: dict) -> str:
    title = _text(data.get("title"))
    premise = _text(data.get("premise"))
    rules = _text(data.get("rules"))
    constraints = _text(data.get("constraints"))
    parts: list[str] = []
    if title:
        parts.append(f"世界观《{title}》")
    if premise:
        parts.append(premise)
    if rules:
        parts.append(f"规则：{rules}")
    if constraints:
        parts.append(f"约束：{constraints}")
    return "。".join(part for part in parts if part)


def _render_character(data: dict) -> str:
    name = _text(data.get("name"))
    role = _text(data.get("role"))
    goal = _text(data.get("goal"))
    traits = _text(data.get("traits"))
    parts = [name or _text(data.get("character_id"))]
    if role:
        parts.append(f"身份：{role}")
    if goal:
        parts.append(f"目标：{goal}")
    if traits:
        parts.append(f"性格：{traits}")
    return "｜".join(part for part in parts if part)


def _render_plan(data: dict) -> str:
    number = data.get("chapter_number")
    title = _text(data.get("title"))
    goal = _text(data.get("goal"))
    events = _text(data.get("planned_events"))
    head = f"第{number}章" if isinstance(number, int) else "章节计划"
    if title:
        head += f"《{title}》"
    parts = [head]
    if goal:
        parts.append(f"目标：{goal}")
    if events:
        parts.append(f"关键事件：{events}")
    return "｜".join(part for part in parts if part)


def _render_event(data: dict) -> str:
    event = _text(data.get("event") or data.get("summary") or data.get("description"))
    target = _text(data.get("object") or data.get("target"))
    chapter = data.get("chapter")
    parts: list[str] = []
    if isinstance(chapter, int):
        parts.append(f"第{chapter}章")
    if event:
        parts.append(event)
    if target:
        parts.append(f"对象：{target}")
    return "｜".join(part for part in parts if part)


def _render_summary(data: dict) -> str:
    return _text(data.get("summary") or data.get("chapter_summary") or data.get("text"))


def render_memory_text(layer: str, kind: str, value_json: str) -> str:
    """Render one memory row into compact model-facing text (deterministic)."""
    data = _load(value_json)
    if data is None:
        return " ".join(str(value_json).split())
    normalized_layer = str(layer)
    normalized_kind = str(kind)
    if normalized_layer == MemoryLayer.M1_CANONICAL.value:
        if normalized_kind == "world_rule":
            rendered = _render_world(data)
        elif normalized_kind == "character_identity":
            rendered = _render_character(data)
        else:
            rendered = ""
    elif normalized_layer == MemoryLayer.M2_PLAN.value:
        rendered = _render_plan(data) if normalized_kind == "plan_goal" else ""
    else:  # M3 confirmed
        if normalized_kind == "chapter_summary":
            rendered = _render_summary(data)
        elif normalized_kind in {"direct_event", "state_change", "hook_status"}:
            rendered = _render_event(data)
        else:
            rendered = ""
    if rendered:
        return rendered
    # Unknown shape: keep the JSON compact rather than dropping information.
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


__all__ = (
    "MIN_FRAGMENT_TOKENS",
    "TOKENIZER_VERSION",
    "estimate_tokens",
    "render_memory_text",
    "truncate_text",
)
