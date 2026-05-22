from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import quote

import httpx

try:
    from vector_memory import queue_stats as _vector_queue_stats
    from vector_memory import run_embedding_jobs as _vector_run_embedding_jobs
except Exception as exc:  # pragma: no cover - optional local module/deps
    _vector_queue_stats = None
    _vector_run_embedding_jobs = None
    VECTOR_QUEUE_IMPORT_ERROR: Exception | None = exc
else:
    VECTOR_QUEUE_IMPORT_ERROR = None


TRUTHY = {"1", "true", "yes", "on"}


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in TRUTHY


def _clean_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def _host_to_http_url(host: str, *, port: int | str | None = None) -> str:
    host = host.strip()
    if not host:
        return ""
    if host.startswith(("http://", "https://")):
        return _clean_base_url(host)
    if ":" in host and not host.startswith("[") and host.count(":") > 1:
        host = f"[{host}]"
    suffix = f":{port}" if port not in (None, "") else ""
    return f"http://{host}{suffix}"


def _strip_openai_v1(value: str) -> str:
    value = _clean_base_url(value)
    return value[:-3] if value.endswith("/v1") else value


def _remote_pc(remote_pcs: dict[str, Any], name: str) -> dict[str, Any]:
    pc = remote_pcs.get(name)
    return pc if isinstance(pc, dict) else {}


def _public_remote_pc(remote_pcs: dict[str, Any], name: str) -> dict[str, Any]:
    pc = dict(_remote_pc(remote_pcs, name))
    for key in ("ssh_pass", "password", "token", "api_key"):
        if pc.get(key):
            pc[key] = "***"
    return pc


def _default_big_llm_probe_url() -> str:
    base = os.getenv("ALPHARAVIS_BIG_LLM_HEALTH_URL", "").strip()
    if base:
        return base
    api_base = os.getenv("BIG_BOSS_API_BASE", "").strip()
    if api_base:
        return f"{_clean_base_url(api_base)}/models" if api_base.rstrip("/").endswith("/v1") else f"{_clean_base_url(api_base)}/v1/models"
    return ""


def _default_comfy_probe_url(remote_pcs: dict[str, Any], comfy_pc: str) -> str:
    configured = os.getenv("ALPHARAVIS_COMFY_HEALTH_URL", "").strip()
    if configured:
        return configured
    ip = _remote_pc(remote_pcs, comfy_pc).get("ip")
    return f"http://{ip}:8188/system_stats" if ip else ""


def _default_ollama_base_url() -> str:
    configured = os.getenv("ALPHARAVIS_OLLAMA_BASE_URL", "").strip()
    if configured:
        return _strip_openai_v1(configured)
    embedding_base = os.getenv("EMBEDDING_API_BASE", "").strip()
    if embedding_base:
        return _strip_openai_v1(embedding_base)
    return "http://192.168.178.140:11434"


def _default_ubuntu_manager_url() -> str:
    configured = os.getenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL", "").strip()
    if configured:
        return _clean_base_url(configured)
    host = os.getenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_IP", "").strip()
    port = os.getenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_PORT", "8099").strip()
    return _host_to_http_url(host, port=port)


def _default_ubuntu_esp_url() -> str:
    configured = os.getenv("ALPHARAVIS_UBUNTU_LLAMA_ESP_URL", "").strip()
    if configured:
        return _clean_base_url(configured)
    host = os.getenv("ALPHARAVIS_UBUNTU_LLAMA_ESP_IP", "").strip()
    port = os.getenv("ALPHARAVIS_UBUNTU_LLAMA_ESP_PORT", "80").strip()
    return _host_to_http_url(host, port="" if port in {"", "80"} else port)


@dataclass(frozen=True)
class ModelManagementConfig:
    enabled: bool
    power_enabled: bool
    allow_actions: bool
    embedding_policy: str
    idle_seconds: float
    big_llm_pc: str
    comfy_pc: str
    big_llm_probe_url: str
    comfy_probe_url: str
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embedding_model: str
    ollama_embedding_fallback_model: str
    action_url: str
    action_api_key: str
    ubuntu_llama_manager_ip: str
    ubuntu_llama_manager_port: int
    ubuntu_llama_manager_url: str
    ubuntu_llama_manager_api_key: str
    ubuntu_llama_esp_ip: str
    ubuntu_llama_esp_port: int
    ubuntu_llama_esp_url: str
    ubuntu_llama_esp_api_key: str
    ubuntu_llama_context_min: int
    ubuntu_llama_context_max: int
    ubuntu_llama_parallel_max: int
    probe_timeout_seconds: float
    comfy_wake_wait_seconds: float


def load_config(remote_pcs: dict[str, Any] | None = None) -> ModelManagementConfig:
    remote_pcs = remote_pcs or {}
    comfy_pc = os.getenv("ALPHARAVIS_COMFY_PC", "comfy_server")
    return ModelManagementConfig(
        enabled=env_bool("ALPHARAVIS_ENABLE_MODEL_MANAGEMENT", "false"),
        power_enabled=env_bool("ALPHARAVIS_ENABLE_POWER_MANAGEMENT", "false"),
        allow_actions=env_bool("ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS", "false"),
        embedding_policy=os.getenv("ALPHARAVIS_EMBEDDING_LOAD_POLICY", "idle_or_big_llm_active").strip().lower(),
        idle_seconds=float(os.getenv("ALPHARAVIS_MODEL_IDLE_SECONDS", "600")),
        big_llm_pc=os.getenv("ALPHARAVIS_BIG_LLM_PC", "main_pc"),
        comfy_pc=comfy_pc,
        big_llm_probe_url=_default_big_llm_probe_url(),
        comfy_probe_url=_default_comfy_probe_url(remote_pcs, comfy_pc),
        ollama_base_url=_default_ollama_base_url(),
        ollama_chat_model=os.getenv("ALPHARAVIS_OLLAMA_CHAT_MODEL", os.getenv("EDGE_GEMMA_LITELLM_MODEL", "openai/gemma4:e2b")).replace("openai/", ""),
        ollama_embedding_model=os.getenv("ALPHARAVIS_OLLAMA_EMBED_MODEL", os.getenv("EMBEDDING_LITELLM_MODEL", "openai/Q78KG/gte-Qwen2-1.5B-instruct")).replace("openai/", ""),
        ollama_embedding_fallback_model=os.getenv("ALPHARAVIS_OLLAMA_EMBED_FALLBACK_MODEL", os.getenv("EMBEDDING_FALLBACK_LITELLM_MODEL", "openai/bge-m3")).replace("openai/", ""),
        action_url=os.getenv("ALPHARAVIS_MODEL_MGMT_ACTION_URL", "").strip(),
        action_api_key=os.getenv("ALPHARAVIS_MODEL_MGMT_API_KEY", "").strip(),
        ubuntu_llama_manager_ip=os.getenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_IP", "").strip(),
        ubuntu_llama_manager_port=int(os.getenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_PORT", "8099")),
        ubuntu_llama_manager_url=_default_ubuntu_manager_url(),
        ubuntu_llama_manager_api_key=os.getenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_API_KEY", "").strip()
        or os.getenv("ALPHARAVIS_MODEL_MGMT_API_KEY", "").strip(),
        ubuntu_llama_esp_ip=os.getenv("ALPHARAVIS_UBUNTU_LLAMA_ESP_IP", "").strip(),
        ubuntu_llama_esp_port=int(os.getenv("ALPHARAVIS_UBUNTU_LLAMA_ESP_PORT", "80")),
        ubuntu_llama_esp_url=_default_ubuntu_esp_url(),
        ubuntu_llama_esp_api_key=os.getenv("ALPHARAVIS_UBUNTU_LLAMA_ESP_API_KEY", "").strip(),
        ubuntu_llama_context_min=int(os.getenv("ALPHARAVIS_UBUNTU_LLAMA_CONTEXT_MIN", "512")),
        ubuntu_llama_context_max=int(os.getenv("ALPHARAVIS_UBUNTU_LLAMA_CONTEXT_MAX", "262144")),
        ubuntu_llama_parallel_max=int(os.getenv("ALPHARAVIS_UBUNTU_LLAMA_PARALLEL_MAX", "2")),
        probe_timeout_seconds=float(os.getenv("ALPHARAVIS_MODEL_MGMT_PROBE_TIMEOUT_SECONDS", "5")),
        comfy_wake_wait_seconds=float(os.getenv("ALPHARAVIS_COMFY_WAKE_WAIT_SECONDS", "0")),
    )


def _public_config(config: ModelManagementConfig) -> dict[str, Any]:
    public_config = asdict(config)
    for key in ("action_api_key", "ubuntu_llama_manager_api_key", "ubuntu_llama_esp_api_key"):
        if public_config.get(key):
            public_config[key] = "***"
    return public_config


async def probe_http(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    if not url:
        return {"ok": False, "url": "", "error": "not_configured"}

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url)
        return {
            "ok": response.status_code < 500,
            "url": url,
            "status_code": response.status_code,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": "" if response.status_code < 500 else response.text[:300],
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": str(exc),
        }


async def _ollama_running_models(config: ModelManagementConfig) -> dict[str, Any]:
    url = f"{config.ollama_base_url}/api/ps"
    probe = await probe_http(url, timeout_seconds=config.probe_timeout_seconds)
    if not probe.get("ok"):
        probe["running_models"] = []
        return probe

    try:
        async with httpx.AsyncClient(timeout=config.probe_timeout_seconds) as client:
            response = await client.get(url)
        data = response.json()
        models = data.get("models", []) if isinstance(data, dict) else []
        running = [str(item.get("name") or item.get("model") or "") for item in models if isinstance(item, dict)]
        probe["running_models"] = [name for name in running if name]
    except Exception as exc:
        probe["ok"] = False
        probe["error"] = str(exc)
        probe["running_models"] = []
    return probe


async def check_ollama_models(remote_pcs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the real Ollama runtime model state used by model-management tools."""

    config = load_config(remote_pcs or {})
    result = await _ollama_running_models(config)
    running_models = result.get("running_models", []) or []
    return {
        "ok": bool(result.get("ok")),
        "ollama_base_url": config.ollama_base_url,
        "running_models": running_models,
        "chat_model": config.ollama_chat_model,
        "embedding_model": config.ollama_embedding_model,
        "fallback_embedding_model": config.ollama_embedding_fallback_model,
        "chat_model_loaded": any(_model_name_matches(name, config.ollama_chat_model) for name in running_models),
        "embedding_model_loaded": any(_model_name_matches(name, config.ollama_embedding_model) for name in running_models),
        "probe": result,
    }


async def load_embedding_model(
    model: str = "",
    *,
    keep_alive: str | None = None,
    remote_pcs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the configured Ollama embedding model by issuing a real keep_alive request."""

    config = load_config(remote_pcs or {})
    selected_model = (model or config.ollama_embedding_model).strip()
    if not selected_model:
        return {"ok": False, "error": "embedding model is not configured"}
    return await _ollama_generate_control(
        config,
        model=selected_model,
        keep_alive=keep_alive or os.getenv("ALPHARAVIS_EMBEDDING_KEEP_ALIVE", "30m"),
    )


async def unload_ollama_model(
    model: str = "",
    *,
    remote_pcs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unload a real Ollama model with keep_alive=0."""

    config = load_config(remote_pcs or {})
    selected_model = (model or config.ollama_embedding_model).strip()
    if not selected_model:
        return {"ok": False, "error": "ollama model is required"}
    return await _ollama_generate_control(config, model=selected_model, keep_alive="0")


async def run_embedding_jobs(job_limit: int | None = None) -> dict[str, Any]:
    """Drain queued pgvector embedding jobs through the real vector-memory queue."""

    if _vector_run_embedding_jobs is None:
        return {"ok": False, "message": f"vector queue unavailable: {VECTOR_QUEUE_IMPORT_ERROR}"}
    limit = job_limit or int(os.getenv("ALPHARAVIS_EMBEDDING_JOB_BATCH_SIZE", "10"))
    return await _vector_run_embedding_jobs(limit=max(1, int(limit)))


async def _embedding_queue_status() -> dict[str, Any]:
    if _vector_queue_stats is None:
        return {"ok": False, "message": f"vector queue unavailable: {VECTOR_QUEUE_IMPORT_ERROR}"}
    try:
        stats = await _vector_queue_stats()
        return {"ok": True, **stats}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


async def inspect_runtime(remote_pcs: dict[str, Any] | None = None) -> dict[str, Any]:
    remote_pcs = remote_pcs or {}
    config = load_config(remote_pcs)
    big_task = probe_http(config.big_llm_probe_url, timeout_seconds=config.probe_timeout_seconds)
    comfy_task = probe_http(config.comfy_probe_url, timeout_seconds=config.probe_timeout_seconds)
    ollama_task = _ollama_running_models(config)
    queue_task = _embedding_queue_status()
    big_llm, comfy, ollama, embedding_queue = await asyncio.gather(big_task, comfy_task, ollama_task, queue_task)
    return {
        "config": _public_config(config),
        "remote_pcs": {
            "big_llm_pc": {"name": config.big_llm_pc, **_public_remote_pc(remote_pcs, config.big_llm_pc)},
            "comfy_pc": {"name": config.comfy_pc, **_public_remote_pc(remote_pcs, config.comfy_pc)},
        },
        "services": {
            "big_llm": big_llm,
            "comfyui": comfy,
            "ollama": ollama,
            "embedding_queue": embedding_queue,
        },
    }


def _model_name_matches(running_name: str, wanted_name: str) -> bool:
    running = running_name.lower()
    wanted = wanted_name.lower()
    return running == wanted or running.endswith(f"/{wanted}") or wanted in running


def embedding_maintenance_decision(runtime: dict[str, Any], *, last_activity_age_seconds: float | None = None) -> dict[str, Any]:
    config_data = runtime.get("config", {})
    config = ModelManagementConfig(**config_data)
    services = runtime.get("services", {})
    big_llm_ok = bool(services.get("big_llm", {}).get("ok"))
    running_models = services.get("ollama", {}).get("running_models", []) or []
    chat_model_loaded = any(_model_name_matches(name, config.ollama_chat_model) for name in running_models)
    embedding_loaded = any(_model_name_matches(name, config.ollama_embedding_model) for name in running_models)
    idle_ok = last_activity_age_seconds is not None and last_activity_age_seconds >= config.idle_seconds

    if not config.enabled:
        return {
            "allowed": False,
            "reason": "model_management_disabled",
            "recommendation": "No model-management action. Set ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true to enable planning.",
        }

    allowed_window = False
    if config.embedding_policy == "idle_only":
        allowed_window = idle_ok
    elif config.embedding_policy == "big_llm_active_only":
        allowed_window = big_llm_ok
    else:
        allowed_window = idle_ok or big_llm_ok

    if embedding_loaded:
        recommendation = "Embedding model already appears loaded on Ollama; run queued embedding jobs."
    elif not allowed_window:
        recommendation = (
            "Do not switch Ollama models yet. Wait for inactivity or for the big llama.cpp server "
            "to be available so chat work does not depend on the small Ollama node."
        )
    elif chat_model_loaded:
        recommendation = (
            "Safe window detected. Unload the Ollama chat/crisis model, load the embedding model, "
            "run embedding jobs, then restore the chat model if needed."
        )
    else:
        recommendation = "Safe window detected. Load the embedding model and run embedding jobs."

    return {
        "allowed": allowed_window or embedding_loaded,
        "reason": {
            "policy": config.embedding_policy,
            "idle_ok": idle_ok,
            "big_llm_ok": big_llm_ok,
            "chat_model_loaded": chat_model_loaded,
            "embedding_loaded": embedding_loaded,
            "last_activity_age_seconds": last_activity_age_seconds,
        },
        "recommendation": recommendation,
        "planned_actions": [
            "drain_or_pause_embedding_queue",
            "unload_ollama_chat_model_if_loaded",
            "load_ollama_embedding_model",
            "run_embedding_jobs",
            "restore_ollama_chat_model_if_configured",
        ]
        if allowed_window and not embedding_loaded
        else ["run_embedding_jobs"] if embedding_loaded else [],
    }


async def call_management_action(
    action: str,
    payload: dict[str, Any] | None = None,
    *,
    remote_pcs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_config(remote_pcs or {})
    payload = payload or {}
    safe_payload = {"action": action, "payload": payload}

    if not config.allow_actions or not config.action_url:
        return {
            "ok": False,
            "dry_run": True,
            "reason": "actions_disabled_or_missing_endpoint",
            "message": (
                "Model/power action was planned but not executed. Populate "
                "ALPHARAVIS_MODEL_MGMT_ACTION_URL and set ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=true "
                "after you wire the curated OpenWebUI/Hermes tools."
            ),
            **safe_payload,
        }

    headers = {"Content-Type": "application/json"}
    if config.action_api_key:
        headers["Authorization"] = f"Bearer {config.action_api_key}"

    try:
        async with httpx.AsyncClient(timeout=config.probe_timeout_seconds) as client:
            response = await client.post(config.action_url, headers=headers, json=safe_payload)
        return {
            "ok": response.status_code < 400,
            "dry_run": False,
            "status_code": response.status_code,
            "response": response.text[:2000],
            **safe_payload,
        }
    except Exception as exc:
        return {"ok": False, "dry_run": False, "error": str(exc), **safe_payload}


def _ubuntu_manager_not_configured(config: ModelManagementConfig) -> dict[str, Any]:
    return {
        "ok": False,
        "reason": "ubuntu_llama_manager_not_configured",
        "message": "Set ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL to the Ubuntu Llama Manager API base URL.",
        "config": _public_config(config),
    }


async def _ubuntu_manager_request(
    config: ModelManagementConfig,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    action_required: bool = False,
) -> dict[str, Any]:
    if not config.ubuntu_llama_manager_url:
        return _ubuntu_manager_not_configured(config)

    url = f"{config.ubuntu_llama_manager_url}{path}"
    safe_result = {"method": method.upper(), "url": url, "payload": payload or {}}
    if action_required and not config.allow_actions:
        return {
            "ok": False,
            "dry_run": True,
            "reason": "actions_disabled",
            "message": "Set ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=true before executing Ubuntu Llama Manager changes.",
            **safe_result,
        }

    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if config.ubuntu_llama_manager_api_key:
        headers["Authorization"] = f"Bearer {config.ubuntu_llama_manager_api_key}"

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=max(config.probe_timeout_seconds, 5)) as client:
            response = await client.request(method.upper(), url, headers=headers, json=payload)
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text[:2000]
        return {
            "ok": response.status_code < 400,
            "dry_run": False,
            "status_code": response.status_code,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "response": body,
            **safe_result,
        }
    except Exception as exc:
        return {
            "ok": False,
            "dry_run": False,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": str(exc),
            **safe_result,
        }


async def _ubuntu_esp_direct_request(
    config: ModelManagementConfig,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    action_required: bool = True,
) -> dict[str, Any]:
    if not config.ubuntu_llama_esp_url:
        return {
            "ok": False,
            "reason": "ubuntu_llama_esp_not_configured",
            "message": "Set ALPHARAVIS_UBUNTU_LLAMA_ESP_URL to use direct ESP power control.",
            "config": _public_config(config),
        }
    if action_required and not config.allow_actions:
        return {
            "ok": False,
            "dry_run": True,
            "reason": "actions_disabled",
            "method": method.upper(),
            "url": f"{config.ubuntu_llama_esp_url}{path}",
            "payload": payload or {},
            "message": "Set ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=true before executing direct ESP actions.",
        }

    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if config.ubuntu_llama_esp_api_key:
        headers["Authorization"] = f"Bearer {config.ubuntu_llama_esp_api_key}"

    url = f"{config.ubuntu_llama_esp_url}{path}"
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=max(config.probe_timeout_seconds, 5)) as client:
            response = await client.request(method.upper(), url, headers=headers, json=payload)
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text[:2000]
        return {
            "ok": response.status_code < 400,
            "dry_run": False,
            "method": method.upper(),
            "url": url,
            "payload": payload or {},
            "status_code": response.status_code,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "response": body,
        }
    except Exception as exc:
        return {
            "ok": False,
            "dry_run": False,
            "method": method.upper(),
            "url": url,
            "payload": payload or {},
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": str(exc),
        }


async def inspect_ubuntu_llama_manager(remote_pcs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect the external Ubuntu Llama Manager API and known llama.cpp instances."""

    config = load_config(remote_pcs or {})
    if not config.ubuntu_llama_manager_url:
        return _ubuntu_manager_not_configured(config)
    health_task = _ubuntu_manager_request(config, "GET", "/health")
    status_task = _ubuntu_manager_request(config, "GET", "/status")
    instances_task = _ubuntu_manager_request(config, "GET", "/llama/instances")
    models_task = _ubuntu_manager_request(config, "GET", "/models")
    health, status, instances, models = await asyncio.gather(health_task, status_task, instances_task, models_task)
    return {
        "ok": bool(health.get("ok")) and bool(instances.get("ok")),
        "config": _public_config(config),
        "health": health,
        "status": status,
        "instances": instances,
        "models": models,
    }


def _validate_llama_instance(instance_id: str) -> str:
    normalized = instance_id.strip().lower()
    aliases = {
        "primary": "primary",
        "main": "primary",
        "1": "primary",
        "llama": "primary",
        "secondary": "secondary",
        "second": "secondary",
        "2": "secondary",
        "8001": "secondary",
        "llama-secondary": "secondary",
    }
    if normalized not in aliases:
        raise ValueError("instance_id must be primary or secondary")
    return aliases[normalized]


def _validate_context_size(config: ModelManagementConfig, context_size: int | str | None) -> int | None:
    if context_size in (None, ""):
        return None
    try:
        value = int(str(context_size).strip())
    except ValueError as exc:
        raise ValueError("context_size must be an integer") from exc
    if value < config.ubuntu_llama_context_min or value > config.ubuntu_llama_context_max:
        raise ValueError(
            f"context_size must be between {config.ubuntu_llama_context_min} and {config.ubuntu_llama_context_max}"
        )
    return value


def _validate_parallel_slots(config: ModelManagementConfig, parallel_slots: int | str | None) -> int | None:
    if parallel_slots in (None, ""):
        return None
    try:
        value = int(str(parallel_slots).strip())
    except ValueError as exc:
        raise ValueError("parallel_slots must be an integer") from exc
    if value < 1 or value > max(1, config.ubuntu_llama_parallel_max):
        raise ValueError(f"parallel_slots must be between 1 and {max(1, config.ubuntu_llama_parallel_max)}")
    return value


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def model_context_policy_plan(
    *,
    reason: str = "",
    requested_context_size: int | str | None = None,
    current_instance: str = "",
    rollback: bool = False,
    remote_pcs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan automatic primary/secondary context sizing without executing it."""

    config = load_config(remote_pcs or {})
    secondary_normal = _env_int("ALPHARAVIS_SECONDARY_CONTEXT_NORMAL", 8192)
    secondary_high = _env_int("ALPHARAVIS_SECONDARY_CONTEXT_HIGH", 16384)
    primary_normal = _env_int("ALPHARAVIS_PRIMARY_CONTEXT_NORMAL", 131072)
    primary_high = _env_int("ALPHARAVIS_PRIMARY_CONTEXT_HIGH", 200000)
    reason_text = str(reason or "").lower()
    requested = _validate_context_size(config, requested_context_size) if requested_context_size not in (None, "") else None

    if rollback:
        instance = _validate_llama_instance(current_instance or "secondary")
        target = primary_normal if instance == "primary" else secondary_normal
        return {
            "ok": True,
            "action": "rollback_context",
            "instance_id": instance,
            "target_context_size": _validate_context_size(config, target),
            "restart": True,
            "reason": reason,
            "rollback_to": "primary_normal" if instance == "primary" else "secondary_normal",
            "policy": {
                "secondary_normal": secondary_normal,
                "secondary_high": secondary_high,
                "primary_normal": primary_normal,
                "primary_high": primary_high,
            },
        }

    large_context_reason = any(
        marker in reason_text
        for marker in ("context_overflow", "payload_too_large", "large", "big-context", "200k", "primary")
    )
    if requested is not None:
        target = requested
        instance = "primary" if requested > secondary_high else "secondary"
    elif large_context_reason:
        instance = "primary"
        target = primary_high
    else:
        instance = "secondary"
        target = secondary_high

    target = _validate_context_size(config, min(target, config.ubuntu_llama_context_max))
    return {
        "ok": True,
        "action": "raise_context",
        "instance_id": instance,
        "target_context_size": target,
        "restart": True,
        "reason": reason,
        "rollback_context_size": primary_normal if instance == "primary" else secondary_normal,
        "rollback_action": {
            "instance_id": instance,
            "context_size": primary_normal if instance == "primary" else secondary_normal,
            "restart": True,
        },
        "policy": {
            "secondary_normal": secondary_normal,
            "secondary_high": secondary_high,
            "primary_normal": primary_normal,
            "primary_high": primary_high,
        },
    }


async def apply_model_context_policy(
    *,
    reason: str = "",
    requested_context_size: int | str | None = None,
    current_instance: str = "",
    rollback: bool = False,
    remote_pcs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply automatic primary/secondary context policy through Ubuntu Llama Manager."""

    plan = model_context_policy_plan(
        reason=reason,
        requested_context_size=requested_context_size,
        current_instance=current_instance,
        rollback=rollback,
        remote_pcs=remote_pcs,
    )
    if not plan.get("ok"):
        return plan
    result = await configure_ubuntu_llama_instance(
        str(plan["instance_id"]),
        context_size=int(plan["target_context_size"]),
        restart=bool(plan.get("restart", True)),
        remote_pcs=remote_pcs,
    )
    return {
        "ok": bool(result.get("ok")),
        "plan": plan,
        "result": result,
        "rollback_action": plan.get("rollback_action"),
    }


async def configure_ubuntu_llama_instance(
    instance_id: str,
    *,
    model: str = "",
    model_flag: str = "auto",
    context_size: int | str | None = None,
    parallel_slots: int | str | None = None,
    command: str = "",
    restart: bool = True,
    remote_pcs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Patch model/context/command for a Ubuntu Llama Manager llama.cpp instance."""

    config = load_config(remote_pcs or {})
    try:
        instance = _validate_llama_instance(instance_id)
        validated_context_size = _validate_context_size(config, context_size)
        validated_parallel_slots = _validate_parallel_slots(config, parallel_slots)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    payload: dict[str, Any] = {"restart": bool(restart)}
    if model.strip():
        payload["model"] = model.strip()
        payload["model_flag"] = (model_flag or "auto").strip()
    if validated_context_size is not None:
        payload["context_size"] = validated_context_size
    if validated_parallel_slots is not None:
        payload["parallel"] = validated_parallel_slots
        payload["parallel_vram_note"] = (
            "Use parallel=2 only during a safe VRAM window. Roll high-context big-boss "
            "or small 2B instances back to parallel=1 when concurrent context-heavy work starts."
        )
    if command.strip():
        payload["command"] = command.strip()
    if set(payload) == {"restart"}:
        return {"ok": False, "error": "send model, context_size, parallel_slots, or command"}

    path = f"/llama/instances/{quote(instance, safe='')}/config"
    return await _ubuntu_manager_request(config, "POST", path, payload=payload, action_required=True)


async def control_ubuntu_llama_service(
    instance_id: str,
    action: str,
    *,
    confirmed: bool = False,
    remote_pcs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start, stop, restart, or force-kill a managed Ubuntu Llama Manager llama.cpp service."""

    config = load_config(remote_pcs or {})
    try:
        instance = _validate_llama_instance(instance_id)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    normalized = action.strip().lower().replace("_", "-")
    if normalized not in {"start", "stop", "restart", "force-kill"}:
        return {"ok": False, "error": "action must be start, stop, restart, or force-kill"}
    if normalized in {"stop", "force-kill"} and not confirmed:
        return {
            "ok": False,
            "needs_confirmation": True,
            "action": normalized,
            "instance_id": instance,
            "message": "Confirm the exact target before stopping or force-killing a llama.cpp service.",
        }
    if normalized == "force-kill" and instance != "primary":
        return {"ok": False, "error": "force-kill is only exposed by Ubuntu Llama Manager for the primary instance"}

    if normalized == "force-kill":
        path = "/llama/force-kill"
    elif instance == "primary":
        path = f"/llama/{normalized}"
    else:
        path = f"/llama-secondary/{normalized}"
    return await _ubuntu_manager_request(config, "POST", path, payload={}, action_required=True)


def _power_payload(
    action: str,
    *,
    reason: str = "",
    hold_seconds: int | None = None,
    wait_seconds: int | None = None,
    delay_before_action_seconds: int | None = None,
    requested_by: str = "alpharavis",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": action,
        "reason": reason or "alpharavis-server-management",
        "requested_by": requested_by,
    }
    if hold_seconds is not None:
        payload["hold_seconds"] = int(hold_seconds)
    if wait_seconds is not None:
        payload["wait_seconds"] = int(wait_seconds)
    if delay_before_action_seconds is not None:
        payload["delay_before_action_seconds"] = int(delay_before_action_seconds)
    return payload


async def request_ubuntu_server_power_action(
    action: str,
    *,
    reason: str = "",
    direct_esp: bool = False,
    confirmed: bool = False,
    hold_seconds: int | None = None,
    wait_seconds: int | None = None,
    delay_before_action_seconds: int | None = None,
    remote_pcs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a gated Ubuntu Llama Manager server or ESP power action."""

    config = load_config(remote_pcs or {})
    normalized = action.strip().lower().replace("_", "-")
    aliases = {
        "power-on": "power-on",
        "on": "power-on",
        "start-pc": "power-on",
        "power-off": "power-off",
        "off": "power-off",
        "power-cycle": "power-cycle",
        "cycle": "power-cycle",
        "reset": "reset",
        "esp-cancel": "esp-cancel",
        "cancel": "esp-cancel",
        "shutdown": "shutdown",
        "power-shutdown": "shutdown",
        "reboot": "reboot-now",
        "reboot-now": "reboot-now",
        "reboot-enable": "reboot-enable",
        "reboot-disable": "reboot-disable",
        "gpu-fault": "gpu-fault",
    }
    mapped = aliases.get(normalized)
    if mapped is None:
        return {
            "ok": False,
            "error": (
                "action must be one of power-on, power-off, power-cycle, reset, "
                "esp-cancel, shutdown, reboot-now, reboot-enable, reboot-disable, gpu-fault"
            ),
        }
    destructive = {"power-off", "power-cycle", "reset", "shutdown", "reboot-now", "gpu-fault"}
    if mapped in destructive and not confirmed:
        target = "direct ESP" if direct_esp and mapped in {"power-off", "power-cycle", "reset"} else "Ubuntu Llama Manager"
        return {
            "ok": False,
            "needs_confirmation": True,
            "action": mapped,
            "target": target,
            "message": (
                "Confirm the exact server/power target before running this destructive action. "
                "Power-on, service start, reboot timer enable/disable, and ESP cancel do not require this extra flag."
            ),
        }

    if mapped in {"power-on", "power-off", "power-cycle", "reset"}:
        payload = _power_payload(
            mapped,
            reason=reason,
            hold_seconds=hold_seconds,
            wait_seconds=wait_seconds,
            delay_before_action_seconds=delay_before_action_seconds,
        )
        if direct_esp:
            return await _ubuntu_esp_direct_request(config, "POST", "/action", payload=payload, action_required=True)
        return await _ubuntu_manager_request(config, "POST", "/esp/action", payload=payload, action_required=True)

    if mapped == "esp-cancel":
        if direct_esp:
            return await _ubuntu_esp_direct_request(config, "POST", "/cancel", payload={}, action_required=True)
        return await _ubuntu_manager_request(config, "POST", "/esp/cancel", payload={}, action_required=True)

    paths = {
        "shutdown": "/power/shutdown",
        "reboot-now": "/reboot/now",
        "reboot-enable": "/reboot/enable",
        "reboot-disable": "/reboot/disable",
        "gpu-fault": "/diagnostics/handle-gpu-fault",
    }
    payload = {"reason": reason or "alpharavis-server-management"} if mapped == "gpu-fault" else {}
    return await _ubuntu_manager_request(config, "POST", paths[mapped], payload=payload, action_required=True)


async def recover_ubuntu_llama_no_response(
    reason: str = "alpharavis-crisis",
    *,
    diagnose_only: bool = True,
    probe_timeout_seconds: int | None = None,
    remote_pcs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ask Ubuntu Llama Manager to diagnose or recover a stuck primary llama.cpp server."""

    config = load_config(remote_pcs or {})
    payload: dict[str, Any] = {"reason": reason}
    if probe_timeout_seconds is not None:
        payload["probe_timeout_seconds"] = int(probe_timeout_seconds)
    path = "/ai-stack/diagnose-llama" if diagnose_only else "/ai-stack/llama-no-response"
    return await _ubuntu_manager_request(config, "POST", path, payload=payload, action_required=not diagnose_only)


async def _ollama_generate_control(
    config: ModelManagementConfig,
    *,
    model: str,
    keep_alive: str,
    prompt: str = "",
) -> dict[str, Any]:
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": keep_alive}
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=max(config.probe_timeout_seconds, 30)) as client:
            response = await client.post(f"{config.ollama_base_url}/api/generate", json=payload)
        return {
            "ok": response.status_code < 400,
            "model": model,
            "keep_alive": keep_alive,
            "status_code": response.status_code,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "response": response.text[:500],
        }
    except Exception as exc:
        return {
            "ok": False,
            "model": model,
            "keep_alive": keep_alive,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": str(exc),
        }


async def run_embedding_lifecycle(
    reason: str = "",
    *,
    remote_pcs: dict[str, Any] | None = None,
    job_limit: int | None = None,
    last_activity_age_seconds: float | None = None,
) -> dict[str, Any]:
    runtime = await inspect_runtime(remote_pcs or {})
    decision = embedding_maintenance_decision(runtime, last_activity_age_seconds=last_activity_age_seconds)
    if not decision.get("allowed"):
        return {"ok": False, "skipped": True, "reason": reason, "runtime": runtime, "decision": decision}

    config = load_config(remote_pcs or {})
    services = runtime.get("services", {})
    running_models = services.get("ollama", {}).get("running_models", []) or []
    chat_model_loaded = any(_model_name_matches(name, config.ollama_chat_model) for name in running_models)
    embedding_loaded = any(_model_name_matches(name, config.ollama_embedding_model) for name in running_models)

    actions: list[dict[str, Any]] = []
    if (
        chat_model_loaded
        and not embedding_loaded
        and not env_bool("ALPHARAVIS_EMBEDDING_UNLOAD_CHAT_MODEL", "false")
        and env_bool("ALPHARAVIS_EMBEDDING_SKIP_IF_CHAT_MODEL_LOADED", "true")
    ):
        return {
            "ok": False,
            "skipped": True,
            "reason": reason,
            "decision": decision,
            "message": (
                "Ollama chat/crisis model is loaded; embedding queue is paused. "
                "Set ALPHARAVIS_EMBEDDING_UNLOAD_CHAT_MODEL=true to let the lifecycle runner unload it."
            ),
            "running_models": running_models,
        }

    should_unload_chat = (
        chat_model_loaded
        and not embedding_loaded
        and env_bool("ALPHARAVIS_EMBEDDING_UNLOAD_CHAT_MODEL", "false")
    )
    if should_unload_chat:
        actions.append(
            {
                "action": "unload_chat_model",
                "result": await _ollama_generate_control(config, model=config.ollama_chat_model, keep_alive="0"),
            }
        )

    if not embedding_loaded:
        actions.append(
            {
                "action": "load_embedding_model",
                "result": await _ollama_generate_control(
                    config,
                    model=config.ollama_embedding_model,
                    keep_alive=os.getenv("ALPHARAVIS_EMBEDDING_KEEP_ALIVE", "30m"),
                ),
            }
        )

    if _vector_run_embedding_jobs is None:
        queue_result = {"ok": False, "message": f"vector queue unavailable: {VECTOR_QUEUE_IMPORT_ERROR}"}
    else:
        queue_result = await _vector_run_embedding_jobs(
            limit=job_limit or int(os.getenv("ALPHARAVIS_EMBEDDING_JOB_BATCH_SIZE", "10"))
        )

    restore_chat = should_unload_chat and env_bool("ALPHARAVIS_EMBEDDING_RESTORE_CHAT_MODEL", "true")
    if restore_chat:
        actions.append(
            {
                "action": "restore_chat_model",
                "result": await _ollama_generate_control(
                    config,
                    model=config.ollama_chat_model,
                    keep_alive=os.getenv("ALPHARAVIS_OLLAMA_CHAT_KEEP_ALIVE", "30m"),
                ),
            }
        )

    return {
        "ok": bool(queue_result.get("ok")),
        "reason": reason,
        "decision": decision,
        "actions": actions,
        "queue_result": queue_result,
    }


async def request_power_action(
    action: str,
    target: str,
    reason: str,
    *,
    remote_pcs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    allowed = {
        "wake_pc",
        "shutdown_pc",
        "check_service",
        "start_service",
        "stop_service",
        "load_embedding_model",
        "unload_ollama_model",
        "run_embedding_jobs",
    }
    normalized = action.strip().lower()
    if normalized not in allowed:
        return {
            "ok": False,
            "dry_run": True,
            "reason": "unsupported_action",
            "supported_actions": sorted(allowed),
        }
    if normalized == "load_embedding_model":
        return await load_embedding_model(target, remote_pcs=remote_pcs)
    if normalized == "unload_ollama_model":
        return await unload_ollama_model(target, remote_pcs=remote_pcs)
    if normalized == "run_embedding_jobs":
        try:
            job_limit = int(target) if str(target or "").strip().isdigit() else None
        except Exception:
            job_limit = None
        return await run_embedding_jobs(job_limit=job_limit)
    return await call_management_action(
        normalized,
        {"target": target, "reason": reason},
        remote_pcs=remote_pcs,
    )


async def prepare_comfy_for_pixelle(remote_pcs: dict[str, Any] | None = None) -> dict[str, Any]:
    remote_pcs = remote_pcs or {}
    config = load_config(remote_pcs)
    if not config.enabled:
        return {
            "ready": True,
            "skipped": True,
            "message": "Custom model management disabled; Pixelle preflight did not run.",
        }

    if not env_bool("ALPHARAVIS_PIXELLE_PREPARE_COMFY", "false"):
        return {"ready": True, "skipped": True, "message": "Pixelle ComfyUI preflight disabled."}

    if not config.comfy_probe_url:
        return {
            "ready": True,
            "skipped": True,
            "message": "No ComfyUI health URL configured; Pixelle preflight did not block the job.",
        }

    initial = await probe_http(config.comfy_probe_url, timeout_seconds=config.probe_timeout_seconds)
    if initial.get("ok"):
        return {
            "ready": True,
            "comfy_probe": initial,
            "woke_for_request": False,
            "message": "ComfyUI is reachable.",
        }

    wake_result: dict[str, Any] | None = None
    if config.power_enabled:
        wake_result = await request_power_action(
            "wake_pc",
            config.comfy_pc,
            "Pixelle image generation requested and ComfyUI was not reachable.",
            remote_pcs=remote_pcs,
        )

    retry = None
    if wake_result and config.comfy_wake_wait_seconds > 0:
        await asyncio.sleep(config.comfy_wake_wait_seconds)
        retry = await probe_http(config.comfy_probe_url, timeout_seconds=config.probe_timeout_seconds)
        if retry.get("ok"):
            return {
                "ready": True,
                "comfy_probe": retry,
                "wake_result": wake_result,
                "woke_for_request": bool(wake_result.get("ok")),
                "message": "ComfyUI became reachable after wake request.",
            }

    return {
        "ready": False,
        "comfy_probe": initial,
        "wake_result": wake_result,
        "woke_for_request": bool(wake_result and wake_result.get("ok")),
        "retry_probe": retry,
        "message": (
            "ComfyUI is not reachable. Pixelle may fail unless the ComfyUI machine is awake. "
            "Power actions are dry-run by default until the curated management endpoint is configured."
        ),
    }
