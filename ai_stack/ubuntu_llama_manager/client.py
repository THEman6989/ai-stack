from __future__ import annotations

import os
import time
from typing import Any

import httpx

from ai_stack.context_budget.policies import ensure_kv_unified_in_command
from ai_stack.ubuntu_llama_manager.schemas import UbuntuLlamaInstance


def _clean_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def _host_to_url(host: str, port: str = "8099") -> str:
    host = host.strip()
    if not host:
        return ""
    if host.startswith(("http://", "https://")):
        return _clean_base_url(host)
    if ":" in host and host.count(":") > 1 and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}".rstrip("/")


class UbuntuLlamaManagerClient:
    """Control-plane client for ubuntu-llama-manager.

    Runtime tokenization and completions intentionally do not go through this client.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        timeout_seconds: float = 10,
        runtime_host_override: str = "",
    ) -> None:
        self.base_url = _clean_base_url(base_url)
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.runtime_host_override = runtime_host_override.strip()

    @classmethod
    def from_env(cls) -> "UbuntuLlamaManagerClient | None":
        base_url = os.getenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL", "").strip()
        if not base_url:
            base_url = _host_to_url(
                os.getenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_IP", ""),
                os.getenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_PORT", "8099"),
            )
        if not base_url:
            return None
        return cls(
            base_url,
            api_key=os.getenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_API_KEY", "").strip()
            or os.getenv("ALPHARAVIS_MODEL_MGMT_API_KEY", "").strip(),
            timeout_seconds=float(os.getenv("ALPHARAVIS_LLAMA_MANAGER_TIMEOUT_SECONDS", "10")),
            runtime_host_override=os.getenv("ALPHARAVIS_LLAMA_RUNTIME_HOST_OVERRIDE", "").strip(),
        )

    async def _request(self, method: str, path: str, *, json_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.base_url:
            return {"ok": False, "error": "manager_base_url_not_configured"}
        headers = {"Accept": "application/json"}
        if json_payload is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}{path}"
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.request(method.upper(), url, headers=headers, json=json_payload)
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text
        return {
            "ok": response.status_code < 400,
            "status_code": response.status_code,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "method": method.upper(),
            "url": url,
            "response": body,
        }

    def _instance_from_api(self, data: dict[str, Any]) -> UbuntuLlamaInstance:
        return UbuntuLlamaInstance.from_api(
            data,
            manager_base_url=self.base_url,
            runtime_host_override=self.runtime_host_override,
        )

    async def get_instances(self) -> list[UbuntuLlamaInstance]:
        result = await self._request("GET", "/llama/instances")
        if not result.get("ok"):
            return []
        body = result.get("response")
        if isinstance(body, list):
            items = body
        elif isinstance(body, dict):
            if isinstance(body.get("instances"), list):
                items = body["instances"]
            elif isinstance(body.get("instances"), dict):
                items = list(body["instances"].values())
            elif isinstance(body.get("llama"), dict):
                items = [body["llama"]]
            else:
                items = [value for value in body.values() if isinstance(value, dict) and value.get("id")]
        else:
            items = []
        return [self._instance_from_api(item) for item in items if isinstance(item, dict)]

    async def get_instance(self, instance_id: str) -> UbuntuLlamaInstance | None:
        result = await self._request("GET", f"/llama/instances/{instance_id}")
        body = result.get("response")
        if not result.get("ok") or not isinstance(body, dict):
            return None
        if isinstance(body.get("instance"), dict):
            body = body["instance"]
        return self._instance_from_api(body)

    async def update_instance_config(
        self,
        instance_id: str,
        *,
        model: str | None = None,
        context_size: int | None = None,
        command: str | None = None,
        parallel_slots: int | None = None,
        ensure_kv_unified: bool = False,
        restart: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"restart": bool(restart)}
        if model:
            payload["model"] = model
        if context_size is not None:
            payload["context_size"] = int(context_size)
        if parallel_slots is not None:
            payload["parallel"] = int(parallel_slots)
        selected_command = command
        if ensure_kv_unified:
            if not selected_command:
                instance = await self.get_instance(instance_id)
                selected_command = instance.command if instance else ""
            if not selected_command:
                return {
                    "ok": False,
                    "error": "kv_unified_requires_command",
                    "message": "Manager API can patch model/context/parallel, but adding --kv-unified requires a command to rewrite.",
                }
            selected_command = ensure_kv_unified_in_command(selected_command)
        if selected_command:
            payload["command"] = selected_command
        return await self._request("POST", f"/llama/instances/{instance_id}/config", json_payload=payload)

    async def restart_instance(self, instance_id: str) -> dict[str, Any]:
        return await self._request("POST", self._control_path(instance_id, "restart"))

    async def stop_instance(self, instance_id: str) -> dict[str, Any]:
        return await self._request("POST", self._control_path(instance_id, "stop"))

    async def diagnose_llama(self, *, reason: str = "ai-stack-context-scheduler", probe_timeout_seconds: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"reason": reason}
        if probe_timeout_seconds is not None:
            payload["probe_timeout_seconds"] = int(probe_timeout_seconds)
        return await self._request("POST", "/ai-stack/diagnose-llama", json_payload=payload)

    @staticmethod
    def _control_path(instance_id: str, action: str) -> str:
        normalized = instance_id.strip().lower()
        if normalized in {"secondary", "second", "2", "8001", "llama-secondary"}:
            return f"/llama-secondary/{action}"
        return f"/llama/{action}"
