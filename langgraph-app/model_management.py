from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

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
        probe_timeout_seconds=float(os.getenv("ALPHARAVIS_MODEL_MGMT_PROBE_TIMEOUT_SECONDS", "5")),
        comfy_wake_wait_seconds=float(os.getenv("ALPHARAVIS_COMFY_WAKE_WAIT_SECONDS", "0")),
    )


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
    public_config = asdict(config)
    if public_config.get("action_api_key"):
        public_config["action_api_key"] = "***"
    return {
        "config": public_config,
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
        return {"ready": True, "comfy_probe": initial, "message": "ComfyUI is reachable."}

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
                "message": "ComfyUI became reachable after wake request.",
            }

    return {
        "ready": False,
        "comfy_probe": initial,
        "wake_result": wake_result,
        "retry_probe": retry,
        "message": (
            "ComfyUI is not reachable. Pixelle may fail unless the ComfyUI machine is awake. "
            "Power actions are dry-run by default until the curated management endpoint is configured."
        ),
    }
