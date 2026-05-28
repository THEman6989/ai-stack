from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "submodules" / "deep-agents-ui" / "src" / "app" / "components" / "ComfyUIPanel.tsx"


def _panel_source() -> str:
    return PANEL.read_text(encoding="utf-8")


def _submit_workflow_source() -> str:
    content = _panel_source()
    start = content.index("const submitWorkflow = useCallback")
    end = content.index("const sendAgentPrompt", start)
    return content[start:end]


def test_comfyui_panel_live_submit_uses_proxy_preflight_and_prompt_only() -> None:
    body = _submit_workflow_source()

    assert 'postJson(proxy, "/preflight"' in body
    assert 'postJson(proxy, "/prompt"' in body
    assert "directWorkflowPreflight" not in body
    assert 'connection: "proxy"' in body


def test_comfyui_panel_treats_proxy_ok_false_and_blocked_submit_as_failure() -> None:
    content = _panel_source()
    body = _submit_workflow_source()

    assert "ok === false" in content
    assert "result?.blocked" in body or "submitResult?.result?.blocked" in body
    assert "blocked by backend" in body


def test_comfyui_panel_model_mapping_matches_extended_backend_surface() -> None:
    content = _panel_source()

    for folder in ("clip_vision", "upscale_models", "style_models", "diffusion_models"):
        assert folder in content
    for input_name in ("clip_name1", "clip_name2", "clip_name3", "clip_vision_name", "upscale_model_name", "style_model_name", "diffusion_model"):
        assert input_name in content
    assert "NODE_CLASS_MODEL_INPUT_FOLDERS" in content
    assert "CLIPVisionLoader" in content
    assert "UpscaleModelLoader" in content


def test_comfyui_panel_proxy_default_uses_window_location_hostname_not_hardcoded_localhost() -> None:
    content = _panel_source()

    assert "DEFAULT_PROXY_API_BASE" in content
    assert "window.location.hostname" in content
    assert "window.location.protocol" in content
    assert ":8130/comfyui" in content
    # The env-var check and dynamic window-based default must come BEFORE
    # the SSR localhost fallback (which is still present as a safety net).
    env_check_idx = content.index("NEXT_PUBLIC_COMFYUI_PROXY_API_BASE")
    hostname_idx = content.index("window.location.hostname")
    localhost_idx = content.index("localhost:8130/comfyui")
    assert env_check_idx < hostname_idx < localhost_idx
