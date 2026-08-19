from typing import Final

from novel_agent.migrations.v1_governance import GOVERNANCE_SCHEMA_STATEMENTS
from novel_agent.migrations.v1_operational import OPERATIONAL_SCHEMA_STATEMENTS

INITIAL_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    *OPERATIONAL_SCHEMA_STATEMENTS,
    *GOVERNANCE_SCHEMA_STATEMENTS,
)

__all__ = ("INITIAL_SCHEMA_STATEMENTS",)
