from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import comfyui_client  # noqa: E402


def test_resolve_comfyui_base_url_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_COMFYUI_API_BASE", "comfypc.local:8188/system_stats")

    assert comfyui_client.resolve_comfyui_base_url({}) == "http://comfypc.local:8188"


def test_resolve_comfyui_base_url_uses_remote_pc_and_port(monkeypatch):
    monkeypatch.delenv("ALPHARAVIS_COMFYUI_API_BASE", raising=False)
    monkeypatch.delenv("ALPHARAVIS_COMFY_API_BASE", raising=False)
    monkeypatch.delenv("ALPHARAVIS_COMFY_HEALTH_URL", raising=False)
    monkeypatch.setenv("ALPHARAVIS_COMFY_PC", "comfy_server")
    monkeypatch.setenv("ALPHARAVIS_COMFYUI_PORT", "8190")

    base = comfyui_client.resolve_comfyui_base_url({"comfy_server": {"ip": "192.168.1.44"}})

    assert base == "http://192.168.1.44:8190"


def test_submit_workflow_is_blocked_by_default(monkeypatch):
    monkeypatch.delenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT", raising=False)
    client = comfyui_client.ComfyUIClient(base_url="http://comfypc:8188")

    import asyncio

    result = asyncio.run(client.submit_workflow({"1": {"class_type": "CheckpointLoaderSimple"}}))

    assert result["blocked"] is True
