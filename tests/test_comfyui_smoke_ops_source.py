from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
SMOKE_SCRIPT = ROOT / "scripts" / "comfyui_smoke.py"
USAGE_NOTES = ROOT / "docs" / "ALPHARAVIS_USAGE_NOTES.md"
MAKEFILE_README = ROOT / "docs" / "MAKEFILE_README.md"
ENV_EXAMPLE = ROOT / ".env(exaple)"
COMPOSE = ROOT / "docker-compose.yml"


def test_makefile_exposes_comfyui_smoke_and_relay_lifecycle_targets() -> None:
    content = MAKEFILE.read_text(encoding="utf-8")

    assert "comfyui-smoke:" in content
    assert "scripts/comfyui_smoke.py" in content
    assert "comfyui-relay-status:" in content
    assert "comfyui-relay-smoke:" in content
    assert "COMFYUI_RELAY_SOCKET" in content


def test_comfyui_smoke_script_documents_checked_paths_without_secrets() -> None:
    content = SMOKE_SCRIPT.read_text(encoding="utf-8")

    assert "system_stats" in content
    assert "/comfyui/status" in content or '"/status"' in content
    assert "/comfyui/queue" in content or '"/queue"' in content
    assert "blocked" in content
    assert "unix://" in content
    assert "COMFYUI_SMOKE_VIEW_FILENAME" in content


def test_comfyui_docs_include_smoke_and_relay_supervision_examples() -> None:
    usage = USAGE_NOTES.read_text(encoding="utf-8")
    makefile_doc = MAKEFILE_README.read_text(encoding="utf-8")

    assert "make comfyui-smoke" in usage
    assert "systemd --user" in usage
    assert "pm2 start" in usage
    assert "make comfyui-relay-status" in usage
    assert "make comfyui-relay-smoke" in usage
    assert "make comfyui-smoke" in makefile_doc


def test_build_time_comfyui_flags_are_documented_and_env_template_is_unambiguous() -> None:
    usage = USAGE_NOTES.read_text(encoding="utf-8")
    env_example = ENV_EXAMPLE.read_text(encoding="utf-8")
    compose = COMPOSE.read_text(encoding="utf-8")

    assert "NEXT_PUBLIC_COMFYUI_WORKFLOW_SUBMIT_ENABLED" in usage
    assert "build-time" in usage.lower()
    assert "docker compose build deep-agents-ui" in usage
    assert "Container recreate reicht" in usage or "container recreate" in usage.lower()
    assert "Compose mirrors ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT" in env_example
    assert "NEXT_PUBLIC_COMFYUI_WORKFLOW_SUBMIT_ENABLED: ${ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT:-true}" in compose
    assert "ALPHARAVIS_COMFYUI_WORKFLOW_SUBMIT" not in env_example
