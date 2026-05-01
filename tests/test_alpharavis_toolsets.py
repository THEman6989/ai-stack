from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import alpharavis_toolsets as toolsets  # noqa: E402


class FakeTool:
    def __init__(self, name: str):
        self.name = name


def test_resolve_toolset_includes_parents_and_dedupes_tools():
    resolved = toolsets.resolve_toolset("coding/write")

    assert "coding/write" in resolved.resolved_toolsets
    assert "coding/read" in resolved.resolved_toolsets
    assert "write_alpha_ravis_artifact" in resolved.tools
    assert "read_repo_ai_skill" in resolved.tools


def test_cycle_detection_does_not_recurse_forever():
    old_a = toolsets.TOOLSETS.get("cycle/a")
    old_b = toolsets.TOOLSETS.get("cycle/b")
    try:
        toolsets.TOOLSETS["cycle/a"] = toolsets.Toolset("cycle/a", "a", includes=("cycle/b",))
        toolsets.TOOLSETS["cycle/b"] = toolsets.Toolset("cycle/b", "b", includes=("cycle/a",))

        resolved = toolsets.resolve_toolset("cycle/a")

        assert resolved.cycles
        assert "cycle/a" in resolved.resolved_toolsets
    finally:
        if old_a is None:
            toolsets.TOOLSETS.pop("cycle/a", None)
        else:
            toolsets.TOOLSETS["cycle/a"] = old_a
        if old_b is None:
            toolsets.TOOLSETS.pop("cycle/b", None)
        else:
            toolsets.TOOLSETS["cycle/b"] = old_b


def test_mcp_schema_cache_classifies_pixelle_tools():
    cache = toolsets.build_mcp_schema_cache(
        [
            {
                "name": "pixelle",
                "tools": [
                    {"name": "submit_image_job", "description": "Generate an image with ComfyUI"},
                    {"name": "render_video", "description": "Create video frames"},
                ],
            }
        ]
    )

    assert "pixelle" in cache
    assert "image" in cache
    assert "video" in cache


def test_materialize_toolsets_filters_to_available_tools_and_mcp_category():
    available = {
        "read_repo_ai_skill": FakeTool("read_repo_ai_skill"),
        "write_alpha_ravis_artifact": FakeTool("write_alpha_ravis_artifact"),
    }
    mcp = [FakeTool("submit_image_job"), FakeTool("unrelated_tool")]
    cache = toolsets.build_mcp_schema_cache(
        [{"name": "pixelle", "tools": [{"name": "submit_image_job", "description": "image prompt"}]}]
    )

    materialized = toolsets.materialize_toolsets(
        ["coding/write", "media/image"],
        available,
        mcp_tools=mcp,
        mcp_schema_cache=cache,
    )

    assert "read_repo_ai_skill" in materialized.tool_names
    assert "write_alpha_ravis_artifact" in materialized.tool_names
    assert "submit_image_job" in materialized.tool_names
    assert "unrelated_tool" not in materialized.tool_names


def test_infer_toolsets_from_text_prefers_bounded_categories():
    inferred = toolsets.infer_toolsets_from_text("Bitte debugge docker logs und fix die Datei")

    assert "coding/execute" in inferred
    assert "coding/write" in inferred
    assert "system/docker" in inferred


def _run_all() -> None:
    tests = [
        test_resolve_toolset_includes_parents_and_dedupes_tools,
        test_cycle_detection_does_not_recurse_forever,
        test_mcp_schema_cache_classifies_pixelle_tools,
        test_materialize_toolsets_filters_to_available_tools_and_mcp_category,
        test_infer_toolsets_from_text_prefers_bounded_categories,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    _run_all()
