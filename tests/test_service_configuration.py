from pathlib import Path

import pytest

from novel_agent.pathing import ProjectPathResolver
from novel_agent.service import NovelAgentService, StageTimeoutConfigurationError


def test_service_rejects_non_positive_stage_timeout_with_typed_error(tmp_path: Path) -> None:
    with pytest.raises(StageTimeoutConfigurationError) as caught:
        NovelAgentService(
            ProjectPathResolver(tmp_path / "projects"),
            stage_timeout_seconds=0,
        )

    assert caught.value.stage_timeout_seconds == 0
