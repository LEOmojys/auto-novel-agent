from __future__ import annotations

from pathlib import Path

import gradio as gr

import api
import api_v1
import api_v2
from novel_agent.pathing import ProjectPathResolver
from novel_agent.service import NovelAgentService


def test_compatibility_entrypoints_delegate_to_the_canonical_gradio_builder(tmp_path: Path) -> None:
    # Given: the compatibility modules and an isolated service dependency.
    service = NovelAgentService(ProjectPathResolver(tmp_path / "projects"))

    # When: each public entrypoint builds an application.
    canonical = api_v1.create_novel_app(service)
    legacy_one = api.create_novel_app(service)
    legacy_two = api_v2.create_novel_app(service)

    # Then: all exposed applications are Gradio Blocks built through the same canonical service boundary.
    assert isinstance(canonical, gr.Blocks)
    assert isinstance(legacy_one, gr.Blocks)
    assert isinstance(legacy_two, gr.Blocks)
    for app in (canonical, legacy_one, legacy_two):
        app.close()


def test_compatibility_modules_do_not_retain_legacy_writer_imports() -> None:
    # Given: the two compatibility module source files.
    source_paths = (Path(api.__file__), Path(api_v2.__file__))

    # When: their source is inspected for the prohibited direct-write dependencies.
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)

    # Then: compatibility remains a delegating launch surface only.
    assert "tools_writting" not in source
    assert "tools_history_save" not in source
    assert "novel_save" not in source
