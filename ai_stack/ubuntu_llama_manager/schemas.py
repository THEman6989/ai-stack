from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

from ai_stack.context_budget.policies import RuntimeConfig, parse_runtime_config_from_command


LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def _manager_host(manager_base_url: str) -> str:
    parsed = urlparse(manager_base_url)
    return parsed.hostname or ""


def _runtime_base_url(
    *,
    raw_base_url: str = "",
    host: str = "",
    port: int | str | None = None,
    manager_base_url: str = "",
    host_override: str = "",
) -> str:
    candidate = (raw_base_url or "").strip().rstrip("/")
    if candidate:
        if candidate.endswith("/v1"):
            candidate = candidate[:-3].rstrip("/")
        parsed = urlparse(candidate)
        runtime_host = parsed.hostname or ""
        if runtime_host in LOCAL_HOSTS and (host_override or manager_base_url):
            replacement = host_override or _manager_host(manager_base_url)
            netloc = replacement
            if parsed.port:
                netloc = f"{replacement}:{parsed.port}"
            return urlunparse((parsed.scheme or "http", netloc, parsed.path.rstrip("/"), "", "", "")).rstrip("/")
        return candidate

    selected_host = (host_override or host or _manager_host(manager_base_url)).strip()
    if selected_host in LOCAL_HOSTS and manager_base_url:
        selected_host = _manager_host(manager_base_url)
    if not selected_host:
        return ""
    selected_port = str(port or "").strip()
    if selected_host.startswith(("http://", "https://")):
        parsed = urlparse(selected_host)
        netloc = parsed.netloc
        if selected_port and parsed.port is None:
            netloc = f"{parsed.hostname}:{selected_port}"
        return urlunparse((parsed.scheme, netloc, "", "", "", "")).rstrip("/")
    if ":" in selected_host and selected_host.count(":") > 1 and not selected_host.startswith("["):
        selected_host = f"[{selected_host}]"
    return f"http://{selected_host}{':' + selected_port if selected_port else ''}".rstrip("/")


@dataclass(frozen=True)
class UbuntuLlamaInstance:
    id: str
    service: str = ""
    configured: bool = False
    active: bool = False
    command_key: str = ""
    command: str = ""
    host: str = ""
    port: int | None = None
    base_url: str = ""
    model: str = ""
    ctx_total: int = 8192
    parallel: int = 1
    kv_unified: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def runtime_config(self) -> RuntimeConfig:
        return RuntimeConfig(
            ctx_total=self.ctx_total,
            parallel=self.parallel,
            kv_unified=self.kv_unified,
            command=self.command,
        )

    @classmethod
    def from_api(
        cls,
        data: dict[str, Any],
        *,
        manager_base_url: str = "",
        runtime_host_override: str = "",
    ) -> "UbuntuLlamaInstance":
        instance_id = str(data.get("id") or data.get("instance_id") or data.get("name") or "").strip()
        command = str(data.get("command") or data.get("start_command") or "")
        runtime = parse_runtime_config_from_command(
            command,
            ctx_total=data.get("ctx_total") or data.get("context_size") or data.get("n_ctx"),
            parallel=data.get("parallel") or data.get("parallel_slots") or data.get("n_parallel"),
            kv_unified=data.get("kv_unified") if "kv_unified" in data else None,
        )
        port_value = data.get("port")
        try:
            port = int(port_value) if port_value not in (None, "") else None
        except (TypeError, ValueError):
            port = None
        return cls(
            id=instance_id,
            service=str(data.get("service") or ""),
            configured=bool(data.get("configured")),
            active=bool(data.get("active") or data.get("running")),
            command_key=str(data.get("command_key") or ""),
            command=command,
            host=str(data.get("host") or ""),
            port=port,
            base_url=_runtime_base_url(
                raw_base_url=str(data.get("base_url") or data.get("llama_base_url") or ""),
                host=str(data.get("host") or ""),
                port=port,
                manager_base_url=manager_base_url,
                host_override=runtime_host_override,
            ),
            model=str(data.get("model") or data.get("model_id") or ""),
            ctx_total=runtime.ctx_total,
            parallel=runtime.parallel,
            kv_unified=runtime.kv_unified,
            raw=dict(data),
        )
