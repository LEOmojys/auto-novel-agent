from typing import Final


TOOL_CALL_RESULT_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    "ALTER TABLE tool_calls ADD COLUMN result_payload_json TEXT",
)

__all__ = ("TOOL_CALL_RESULT_SCHEMA_STATEMENTS",)
