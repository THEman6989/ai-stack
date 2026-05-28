from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E2E_SPEC = ROOT / "submodules" / "deep-agents-ui" / "e2e" / "comfyui.spec.ts"
PLAYWRIGHT_CONFIG = ROOT / "submodules" / "deep-agents-ui" / "playwright.config.ts"


def _spec_source() -> str:
    return E2E_SPEC.read_text(encoding="utf-8")


def test_comfyui_e2e_spec_exists_and_covers_tab_render() -> None:
    content = _spec_source()

    assert "ComfyUI tab renders" in content
    assert "ComfyUI Control" in content
    assert "reachable" in content
    assert "button:has-text(\"ComfyUI\")" in content


def test_comfyui_e2e_spec_covers_live_submit_disabled_and_proxy_preflight() -> None:
    content = _spec_source()

    assert "Live submit is disabled" in content
    assert "disabled" in content
    assert "Draft Preflight" in content
    assert "/comfyui/preflight" in content


def test_comfyui_e2e_spec_covers_proxy_ok_false_blocked_as_failure() -> None:
    content = _spec_source()

    assert "ok:false" in content or "ok === false" in content or "blocked" in content
    assert "blocked" in content
    assert "/comfyui/prompt" in content


def test_comfyui_e2e_spec_covers_agent_disabled_copy_fallback() -> None:
    content = _spec_source()

    assert "Copy prompt" in content or "agent is disabled" in content
    assert "Pruefe ComfyUI" in content or "agent handoff" in content


def test_comfyui_e2e_spec_has_mocked_proxy_routes() -> None:
    content = _spec_source()

    for route in ("/comfyui/status", "/comfyui/queue", "/comfyui/models", "/comfyui/preflight", "/comfyui/prompt", "/comfyui/view", "/comfyui/history"):
        assert route in content


def test_comfyui_playwright_config_exists() -> None:
    assert PLAYWRIGHT_CONFIG.exists()
    config = PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")

    assert "defineConfig" in config
    assert "testDir" in config
    assert "e2e" in config
