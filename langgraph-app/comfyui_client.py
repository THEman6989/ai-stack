from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx


TRUE_VALUES = {"1", "true", "yes", "on"}


def env_bool(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default)).strip().lower() in TRUE_VALUES


def _normalize_base_url(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlparse(value)
    path = parsed.path.rstrip("/")
    if path.endswith("/system_stats"):
        path = path[: -len("/system_stats")]
    return urlunparse((parsed.scheme or "http", parsed.netloc, path, "", "", "")).rstrip("/")


def resolve_comfyui_base_url(remote_pcs: dict[str, Any] | None = None) -> str:
    """Resolve the ComfyUI REST base URL without hardcoding a LAN address."""

    for name in ("ALPHARAVIS_COMFYUI_API_BASE", "ALPHARAVIS_COMFY_API_BASE"):
        base = _normalize_base_url(os.getenv(name, ""))
        if base:
            return base

    health = _normalize_base_url(os.getenv("ALPHARAVIS_COMFY_HEALTH_URL", ""))
    if health:
        return health

    remote_pcs = remote_pcs or {}
    pc_key = os.getenv("ALPHARAVIS_COMFY_PC", "comfy_server")
    comfy_pc = remote_pcs.get(pc_key) or remote_pcs.get("comfy_server") or {}
    if isinstance(comfy_pc, dict):
        raw_url = comfy_pc.get("comfyui_url") or comfy_pc.get("comfy_url") or comfy_pc.get("url")
        base = _normalize_base_url(str(raw_url or ""))
        if base:
            return base
        ip = str(comfy_pc.get("ip") or "").strip()
        if ip:
            port = str(comfy_pc.get("comfyui_port") or comfy_pc.get("comfy_port") or os.getenv("ALPHARAVIS_COMFYUI_PORT", "8188"))
            return _normalize_base_url(f"http://{ip}:{port}")

    return _normalize_base_url(os.getenv("ALPHARAVIS_COMFYUI_FALLBACK_BASE", "http://127.0.0.1:8188"))


def comfyui_workflow_submit_enabled() -> bool:
    return env_bool("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT", "false")


class ComfyUIClient:
    def __init__(self, base_url: str | None = None, *, timeout: float | None = None):
        self.base_url = _normalize_base_url(base_url or resolve_comfyui_base_url())
        self.timeout = timeout or float(os.getenv("ALPHARAVIS_COMFYUI_TIMEOUT_SECONDS", "30"))

    async def get_json(self, path: str) -> dict[str, Any]:
        path = "/" + path.lstrip("/")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}{path}")
            response.raise_for_status()
            data = response.json()
        return data if isinstance(data, dict) else {"data": data}

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        path = "/" + path.lstrip("/")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}{path}", json=payload)
            response.raise_for_status()
            data = response.json()
        return data if isinstance(data, dict) else {"data": data}

    async def system_stats(self) -> dict[str, Any]:
        return await self.get_json("/system_stats")

    async def queue(self) -> dict[str, Any]:
        return await self.get_json("/queue")

    async def models(self, folder: str = "checkpoints") -> dict[str, Any]:
        safe_folder = (folder or "checkpoints").strip().strip("/") or "checkpoints"
        return await self.get_json(f"/models/{safe_folder}")

    async def history(self, prompt_id: str) -> dict[str, Any]:
        prompt_id = (prompt_id or "").strip()
        if not prompt_id:
            return {"error": "prompt_id is required"}
        return await self.get_json(f"/history/{prompt_id}")

    async def submit_workflow(self, workflow: dict[str, Any], *, client_id: str = "alpharavis") -> dict[str, Any]:
        if not comfyui_workflow_submit_enabled():
            return {
                "ok": False,
                "blocked": True,
                "message": "ComfyUI workflow submit is disabled. Set ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT=true to allow direct workflow execution.",
            }
        if not isinstance(workflow, dict) or not workflow:
            return {"ok": False, "error": "workflow must be a non-empty API-format JSON object"}
        if any(isinstance(v, dict) and "class_type" in v for v in workflow.values()):
            prompt = workflow
        else:
            return {
                "ok": False,
                "error": "workflow must be ComfyUI API format (node id map with class_type entries), not editor format.",
            }
        return await self.post_json("/prompt", {"prompt": prompt, "client_id": client_id or "alpharavis"})


async def comfyui_status(remote_pcs: dict[str, Any] | None = None) -> dict[str, Any]:
    base_url = resolve_comfyui_base_url(remote_pcs)
    client = ComfyUIClient(base_url)
    try:
        stats = await client.system_stats()
    except Exception as exc:
        return {"ok": False, "base_url": base_url, "error": str(exc)}
    return {"ok": True, "base_url": base_url, "system_stats": stats}
