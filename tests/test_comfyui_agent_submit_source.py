from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_GRAPH = ROOT / "langgraph-app" / "agent_graph.py"


def _agent_graph_source() -> str:
    return AGENT_GRAPH.read_text(encoding="utf-8")


def _submit_tool_source() -> str:
    content = _agent_graph_source()
    start = content.index("async def submit_comfyui_workflow")
    end = content.index("\n\n@tool", start + 1)
    return content[start:end]


def test_comfyui_agent_submit_prefers_media_gallery_prompt_route() -> None:
    body = _submit_tool_source()

    assert "ALPHARAVIS_COMFYUI_AGENT_SUBMIT_VIA_MEDIA_GALLERY" in body
    assert '"/comfyui/prompt"' in body or "'/comfyui/prompt'" in body
    assert "MEDIA_GALLERY" in body
    assert "client.submit_workflow" in body  # fallback remains for compatibility
