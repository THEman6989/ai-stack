from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
EXAMPLE_PATH = ROOT / ".env(exaple)"


IMPORTANT_KEYS = [
    ("POSTGRES_PASSWORD", "Postgres password for vectordb/rag_api"),
    ("LITELLM_MASTER_KEY", "LiteLLM master key"),
    ("BIG_BOSS_API_BASE", "llama.cpp big model OpenAI /v1 URL"),
    ("EDGE_GEMMA_API_BASE", "Ollama fallback OpenAI /v1 URL"),
    ("EMBEDDING_API_BASE", "Ollama embedding OpenAI /v1 URL"),
    ("HERMES_API_BASE", "internal Docker URL for LibreChat/LangGraph to Hermes"),
    ("HERMES_EXTERNAL_API_BASE", "host URL for humans/tools to call Hermes"),
    ("HERMES_API_KEY", "Hermes API bearer token"),
    ("HERMES_MODEL", "Hermes advertised OpenAI model id"),
    ("HERMES_INFERENCE_MODEL", "real LiteLLM model Hermes should use"),
    ("HERMES_OPENAI_BASE_URL", "LiteLLM/OpenAI-compatible URL used by Hermes"),
    ("HERMES_OPENAI_API_KEY", "API key Hermes uses for LiteLLM"),
    ("ALPHARAVIS_ENABLE_HERMES_AGENT", "enable LangGraph -> Hermes sub-agent"),
    ("ALPHARAVIS_ENABLE_MEDIA_GALLERY", "enable Pixelle/upload media gallery registration"),
    ("ALPHARAVIS_MEDIA_PUBLIC_BASE_URL", "host URL for the AlphaRavis media gallery"),
    ("ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY", "enable separate pgvector table for image/video embeddings"),
    ("ALPHARAVIS_VISION_EMBEDDING_MODEL_URL", "direct OpenAI-compatible /v1 URL for an external vision embedding server"),
    ("ALPHARAVIS_VISION_EMBEDDING_MODEL", "LiteLLM model id for vision embeddings"),
    ("LIBRECHAT_OPENAI_API_KEY", "optional generic LibreChat OpenAI bucket key"),
    ("LIBRECHAT_OPENAI_REVERSE_PROXY", "optional generic LibreChat OpenAI reverse proxy"),
    ("OPENWEBUI_PORT", "host port for optional OpenWebUI frontend"),
    ("OPENWEBUI_ENABLE_OPENAI_API_PASSTHROUGH", "enable OpenWebUI OpenAI API passthrough"),
    ("OPENWEBUI_ENABLE_WEB_SEARCH", "enable OpenWebUI web search"),
]


MODEL_MANAGEMENT_KEYS = [
    ("ALPHARAVIS_ENABLE_MODEL_MANAGEMENT", "enable the custom model-management planning layer"),
    ("ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT", "enable power_management_agent and advanced hooks"),
    ("ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS", "enable owner-only SSH/Wake-on-LAN tools"),
    ("ALPHARAVIS_ENABLE_CRISIS_MANAGER", "future crisis-manager routing switch"),
    ("ALPHARAVIS_CRISIS_MANAGER_MODEL", "small Ollama/LiteLLM crisis-manager model"),
    ("ALPHARAVIS_CRISIS_MAX_ATTEMPTS", "future maximum automatic recovery attempts"),
    ("ALPHARAVIS_CRISIS_TIMEOUT_SECONDS", "future crisis-manager wall-clock timeout"),
    ("ALPHARAVIS_CRISIS_AUTO_ACTIONS", "future auto-approved recovery action names"),
    ("ALPHARAVIS_CRISIS_HITL_ACTIONS", "future human-approval action names"),
    ("ALPHARAVIS_POWER_MANAGER_MODEL", "small model used by power_management_agent"),
    ("ALPHARAVIS_POWER_MANAGER_TIMEOUT_SECONDS", "power-management agent timeout"),
    ("ALPHARAVIS_OWNER_SSH_PASS", "private owner SSH password fallback; do not commit real values"),
    ("ALPHARAVIS_HARD_CONTEXT_TOKEN_LIMIT", "hard LangGraph context cutoff"),
    ("ALPHARAVIS_LLM_API_MODE", "responses or chat_completions for direct no-tool calls"),
    ("ALPHARAVIS_LLM_STREAMING", "enable ChatLiteLLM streaming for Chat Completions/fallback calls"),
    ("ALPHARAVIS_RESPONSES_API_BASE", "OpenAI-compatible /v1 base for Responses calls"),
    ("ALPHARAVIS_RESPONSES_MODEL", "model id used for native Responses direct calls"),
    ("ALPHARAVIS_RESPONSES_REQUIRE_NATIVE", "fail instead of falling back to ChatLiteLLM"),
    ("ALPHARAVIS_DEEPAGENTS_API_MODE", "responses or chat_completions for DeepAgents tool workers"),
    ("ALPHARAVIS_DEEPAGENTS_RESPONSES_API_BASE", "OpenAI-compatible /v1 base for DeepAgents Responses tool calls"),
    ("ALPHARAVIS_DEEPAGENTS_RESPONSES_MODEL", "model id for DeepAgents Responses mode"),
    ("ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING", "enable DeepAgents Responses token streaming"),
    ("ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING", "false, true, or tool_calling"),
    ("ALPHARAVIS_DEEPAGENTS_REQUIRE_RESPONSES", "fail startup instead of falling back for DeepAgents"),
    ("BRIDGE_PREFERRED_API_MODE", "responses or chat_completions"),
    ("BRIDGE_HARD_INPUT_TOKEN_LIMIT", "optional hard bridge request cutoff"),
    ("BRIDGE_HARD_INPUT_HTTP_ERROR", "return HTTP 413 instead of visible message"),
    ("ALPHARAVIS_ENABLE_POWER_MANAGEMENT", "allow power-management intent handling"),
    ("ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS", "allow real external model/power actions"),
    ("ALPHARAVIS_MODEL_MGMT_ACTION_URL", "curated action endpoint URL"),
    ("ALPHARAVIS_MODEL_MGMT_API_KEY", "curated action endpoint bearer token"),
    ("ALPHARAVIS_PIXELLE_PREPARE_COMFY", "check ComfyUI before Pixelle jobs"),
    ("ALPHARAVIS_COMFY_HEALTH_URL", "ComfyUI health URL"),
    ("ALPHARAVIS_PIXELLE_BLOCK_IF_COMFY_OFFLINE", "block Pixelle when ComfyUI is offline"),
    ("ALPHARAVIS_PIXELLE_OWNER_WAKE_COMFY", "owner-tool ComfyUI wake during Pixelle preflight"),
    ("ALPHARAVIS_MODEL_IDLE_SECONDS", "idle seconds before embedding maintenance is allowed"),
    ("ALPHARAVIS_EMBEDDING_LOAD_POLICY", "idle_only, big_llm_active_only, or idle_or_big_llm_active"),
    ("ALPHARAVIS_PGVECTOR_INDEX_MODE", "queue, background, or inline"),
    ("ALPHARAVIS_EMBEDDING_JOB_BATCH_SIZE", "embedding queue batch size"),
    ("ALPHARAVIS_ENABLE_EMBEDDING_SCHEDULER", "run embedding queue scheduler inside LangGraph"),
    ("ALPHARAVIS_ENABLE_VECTOR_BACKFILL_DAEMON", "run bounded vector backfill daemon"),
    ("ALPHARAVIS_VECTOR_BACKFILL_QUERY", "query required for bounded vector backfill"),
    ("ALPHARAVIS_OLLAMA_BASE_URL", "Ollama management-node base URL"),
    ("ALPHARAVIS_OLLAMA_CHAT_MODEL", "small Ollama chat/crisis model"),
    ("ALPHARAVIS_OLLAMA_EMBED_MODEL", "Ollama embedding model"),
    ("ALPHARAVIS_OLLAMA_EMBED_FALLBACK_MODEL", "fallback embedding model"),
    ("ALPHARAVIS_ENABLE_MEDIA_GALLERY", "enable media gallery service"),
    ("ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY", "enable separate media pgvector table"),
    ("ALPHARAVIS_VISION_EMBEDDING_MODEL_URL", "direct OpenAI-compatible /v1 URL for external vision embeddings"),
    ("ALPHARAVIS_VISION_EMBEDDING_BASE_URL", "OpenAI-compatible vision embedding base URL"),
    ("ALPHARAVIS_VISION_EMBEDDING_MODEL", "primary vision embedding model"),
    ("OPENWEBUI_ENABLE_OPENAI_API_PASSTHROUGH", "OpenWebUI passthrough switch"),
]


ENV_INTERPOLATION_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


SERVICE_URLS = [
    ("Service Dashboard", "http://localhost:${ALPHARAVIS_SERVICE_DASHBOARD_PORT:-8090}"),
    ("LibreChat", "http://localhost:3080"),
    ("LangGraph API", "http://localhost:2024"),
    ("LangGraph Studio", "https://smith.langchain.com/studio/?baseUrl=http://localhost:2024"),
    ("OpenAI Bridge", "http://localhost:8123/v1"),
    ("Bridge Test UI", "http://localhost:${ALPHARAVIS_TEST_UI_PORT:-8140}"),
    ("Hermes API", "HERMES_EXTERNAL_API_BASE"),
    ("LiteLLM", "http://localhost:4000/v1"),
    ("RAG API", "http://localhost:8000"),
    ("Media Gallery", "http://localhost:${ALPHARAVIS_MEDIA_PORT:-8130}/gallery"),
    ("OpenWebUI", "http://localhost:${OPENWEBUI_PORT:-3090}"),
    ("DeepAgents UI", "http://localhost:3000"),
    ("Agent Custom UI", "http://localhost:3001"),
    ("Pixelle MCP", "http://localhost:9004"),
]

STREAMING_MODE_DESCRIPTIONS = {
    "responses-hybrid": "Responses API; stream no-tool calls, run tool-bound calls non-streaming",
    "responses-full": "Responses API; experimental full streaming for tool-bound DeepAgents calls",
    "responses-nonstreaming": "Responses API; internal model streaming disabled",
    "chat-full": "Chat Completions API; ChatLiteLLM streaming enabled",
    "chat-nonstreaming": "Chat Completions API; ChatLiteLLM streaming disabled",
}

STREAMING_MODE_ALIASES = {
    "": "prompt",
    "ask": "prompt",
    "default": "responses-hybrid",
    "stable": "responses-hybrid",
    "hybrid": "responses-hybrid",
    "responses": "responses-hybrid",
    "response": "responses-hybrid",
    "responses-hybrid": "responses-hybrid",
    "tool_calling": "responses-hybrid",
    "tool-calling": "responses-hybrid",
    "full": "responses-full",
    "fullstreaming": "responses-full",
    "full-streaming": "responses-full",
    "responses-full": "responses-full",
    "responses-fullstreaming": "responses-full",
    "nonstreaming": "responses-nonstreaming",
    "non-streaming": "responses-nonstreaming",
    "responses-nonstreaming": "responses-nonstreaming",
    "false": "responses-nonstreaming",
    "off": "responses-nonstreaming",
    "none": "responses-nonstreaming",
    "no": "responses-nonstreaming",
    "chat": "chat-full",
    "chat-full": "chat-full",
    "chat-fullstreaming": "chat-full",
    "chat-completions": "chat-full",
    "chat_completions": "chat-full",
    "chat-completions-full": "chat-full",
    "chat_completions_full": "chat-full",
    "legacy": "chat-full",
    "chat-nonstreaming": "chat-nonstreaming",
    "chat-non-streaming": "chat-nonstreaming",
    "chat-completions-nonstreaming": "chat-nonstreaming",
    "chat_completions_nonstreaming": "chat-nonstreaming",
}

STREAMING_MODE_VALUES = {
    "responses-hybrid": {
        "ALPHARAVIS_LLM_API_MODE": "responses",
        "ALPHARAVIS_DEEPAGENTS_API_MODE": "responses",
        "ALPHARAVIS_LLM_STREAMING": "true",
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING": "true",
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING": "tool_calling",
        "ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING": "false",
        "BRIDGE_ENABLE_RESPONSES_API": "true",
        "BRIDGE_PREFERRED_API_MODE": "responses",
    },
    "responses-full": {
        "ALPHARAVIS_LLM_API_MODE": "responses",
        "ALPHARAVIS_DEEPAGENTS_API_MODE": "responses",
        "ALPHARAVIS_LLM_STREAMING": "true",
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING": "true",
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING": "false",
        "ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING": "true",
        "BRIDGE_ENABLE_RESPONSES_API": "true",
        "BRIDGE_PREFERRED_API_MODE": "responses",
    },
    "responses-nonstreaming": {
        "ALPHARAVIS_LLM_API_MODE": "responses",
        "ALPHARAVIS_DEEPAGENTS_API_MODE": "responses",
        "ALPHARAVIS_LLM_STREAMING": "false",
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING": "false",
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING": "true",
        "ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING": "false",
        "BRIDGE_ENABLE_RESPONSES_API": "true",
        "BRIDGE_PREFERRED_API_MODE": "responses",
    },
    "chat-full": {
        "ALPHARAVIS_LLM_API_MODE": "chat_completions",
        "ALPHARAVIS_DEEPAGENTS_API_MODE": "chat_completions",
        "ALPHARAVIS_LLM_STREAMING": "true",
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING": "false",
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING": "true",
        "ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING": "false",
        "BRIDGE_ENABLE_RESPONSES_API": "true",
        "BRIDGE_PREFERRED_API_MODE": "chat_completions",
    },
    "chat-nonstreaming": {
        "ALPHARAVIS_LLM_API_MODE": "chat_completions",
        "ALPHARAVIS_DEEPAGENTS_API_MODE": "chat_completions",
        "ALPHARAVIS_LLM_STREAMING": "false",
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING": "false",
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING": "true",
        "ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING": "false",
        "BRIDGE_ENABLE_RESPONSES_API": "true",
        "BRIDGE_PREFERRED_API_MODE": "chat_completions",
    },
}


NETWORK_MODE_ALIASES = {
    "": "tailscale",
    "auto": "tailscale",
    "apply": "tailscale",
    "tailscale": "tailscale",
    "tailnet": "tailscale",
    "https": "tailscale",
    "on": "tailscale",
    "true": "tailscale",
    "yes": "tailscale",
    "1": "tailscale",
    "lan": "lan",
    "local-lan": "lan",
    "disable": "lan",
    "disabled": "lan",
    "off": "lan",
    "false": "lan",
    "no": "lan",
    "0": "lan",
}

NETWORK_MODE_VALUES = {
    "tailscale": {
        "ALPHARAVIS_DOCKER_HOST_BIND": "127.0.0.1",
    },
    "lan": {
        "ALPHARAVIS_DOCKER_HOST_BIND": "0.0.0.0",
    },
}


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, text=True, check=check)


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def ensure_env() -> None:
    if not ENV_PATH.exists():
        shutil.copyfile(EXAMPLE_PATH, ENV_PATH)
        print(f"created {ENV_PATH.name} from {EXAMPLE_PATH.name}")

    current = read_env(ENV_PATH)
    example = read_env(EXAMPLE_PATH)
    missing = [(key, value) for key, value in example.items() if key not in current]
    if not missing:
        return

    with ENV_PATH.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n\n# Added by make install/update from .env(exaple)\n")
        for key, value in missing:
            fh.write(f"{key}={value}\n")
    print(f"added {len(missing)} missing env defaults to .env")


def update_env_value(key: str, value: str) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    found = False
    out: list[str] = []
    for line in lines:
        if line.strip().startswith("#") or "=" not in line:
            out.append(line)
            continue
        current_key = line.split("=", 1)[0].strip()
        if current_key == key:
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def configure() -> None:
    ensure_env()
    values = read_env(ENV_PATH)
    print("Press Enter to keep the current/default value.")
    for key, description in IMPORTANT_KEYS:
        current = values.get(key, "")
        answer = input(f"{key} [{current}] - {description}: ").strip()
        if answer:
            update_env_value(key, answer)
            values[key] = answer
    print(".env updated")


def set_many(values: dict[str, str]) -> None:
    for key, value in values.items():
        update_env_value(key, value)


def normalize_network_mode(mode: str) -> str:
    normalized = (mode or "tailscale").strip().lower().replace("_", "-")
    normalized = NETWORK_MODE_ALIASES.get(normalized, normalized)
    if normalized in NETWORK_MODE_VALUES:
        return normalized
    valid = ", ".join(NETWORK_MODE_VALUES)
    raise ValueError(f"Unsupported network mode {mode!r}. Use one of: {valid}")


def apply_network_mode(mode: str) -> str:
    ensure_env()
    resolved = normalize_network_mode(mode)
    set_many(NETWORK_MODE_VALUES[resolved])
    bind = NETWORK_MODE_VALUES[resolved]["ALPHARAVIS_DOCKER_HOST_BIND"]
    if resolved == "tailscale":
        print(f"Network mode: tailscale HTTPS via Tailscale Serve; Docker host bind set to {bind}")
    else:
        print(f"Network mode: LAN HTTP; Docker host bind set to {bind}")
    return resolved


def normalize_streaming_mode(mode: str) -> str:
    normalized = (mode or "prompt").strip().lower()
    normalized = STREAMING_MODE_ALIASES.get(normalized, normalized)
    if normalized in {"prompt", "keep"} or normalized in STREAMING_MODE_VALUES:
        return normalized
    valid = ", ".join(["prompt", "keep", *STREAMING_MODE_VALUES])
    raise ValueError(f"Unsupported streaming mode {mode!r}. Use one of: {valid}")


def current_streaming_mode(values: dict[str, str]) -> str:
    llm_mode = values.get("ALPHARAVIS_LLM_API_MODE", "").lower()
    deepagents_mode = values.get("ALPHARAVIS_DEEPAGENTS_API_MODE", "").lower()
    chat_streaming = values.get("ALPHARAVIS_LLM_STREAMING", "true").lower()
    streaming = values.get("ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING", "").lower()
    disable_streaming = values.get("ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING", "").lower()
    experimental = values.get("ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING", "").lower()
    if llm_mode == "chat_completions" or deepagents_mode == "chat_completions":
        return "chat-full" if chat_streaming != "false" else "chat-nonstreaming"
    if streaming == "false" or disable_streaming == "true":
        return "responses-nonstreaming"
    if disable_streaming == "false":
        return "responses-full" if experimental == "true" else "responses-full"
    return "responses-hybrid"


def format_profile_env(mode: str) -> list[str]:
    values = STREAMING_MODE_VALUES[mode]
    keys = [
        "ALPHARAVIS_LLM_API_MODE",
        "ALPHARAVIS_LLM_STREAMING",
        "ALPHARAVIS_DEEPAGENTS_API_MODE",
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING",
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING",
        "ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING",
        "BRIDGE_PREFERRED_API_MODE",
    ]
    return [f"{key}={values[key]}" for key in keys if key in values]


def print_streaming_profiles() -> None:
    print("\nAvailable AlphaRavis runtime profiles")
    for mode, description in STREAMING_MODE_DESCRIPTIONS.items():
        print(f"\n[{mode}] {description}")
        for line in format_profile_env(mode):
            print(f"  {line}")


def choose_streaming_mode(values: dict[str, str]) -> str:
    current = current_streaming_mode(values)
    print("\nResponses / DeepAgents streaming mode")
    for index, mode in enumerate(STREAMING_MODE_VALUES, start=1):
        marker = " (current)" if mode == current else ""
        print(f"  {index}. {mode}{marker}: {STREAMING_MODE_DESCRIPTIONS[mode]}")
    print("  i. show env values for all profiles")
    print("  k. keep current .env streaming settings")
    try:
        answer = input(f"Select streaming mode [{current}]: ").strip().lower()
    except EOFError:
        return current
    if not answer:
        return current
    if answer in {str(index) for index in range(1, len(STREAMING_MODE_VALUES) + 1)}:
        return list(STREAMING_MODE_VALUES)[int(answer) - 1]
    if answer in {"i", "info", "help", "?"}:
        print_streaming_profiles()
        return choose_streaming_mode(values)
    if answer in {"k", "keep"}:
        return "keep"
    return normalize_streaming_mode(answer)


def apply_streaming_mode(mode: str) -> str:
    ensure_env()
    resolved = normalize_streaming_mode(mode)
    if resolved == "prompt":
        resolved = choose_streaming_mode(read_env(ENV_PATH))
    if resolved == "keep":
        print("Streaming mode unchanged")
        return current_streaming_mode(read_env(ENV_PATH))
    set_many(STREAMING_MODE_VALUES[resolved])
    print(f"Streaming mode set to {resolved}: {STREAMING_MODE_DESCRIPTIONS[resolved]}")
    return resolved


def configure_streaming(mode: str = "prompt") -> None:
    apply_streaming_mode(mode)


def configure_model_management() -> None:
    ensure_env()
    values = read_env(ENV_PATH)
    print("Custom model/power management is off by default.")
    print("Press Enter to keep the shown value.")

    enable = ask_yes_no(
        "Enable custom model-management planning",
        default=values.get("ALPHARAVIS_ENABLE_MODEL_MANAGEMENT", "false").lower() in {"1", "true", "yes"},
    )
    if not enable:
        set_many(
            {
                "ALPHARAVIS_ENABLE_MODEL_MANAGEMENT": "false",
                "ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT": "false",
                "ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS": "false",
                "ALPHARAVIS_ENABLE_CRISIS_MANAGER": "false",
                "ALPHARAVIS_ENABLE_POWER_MANAGEMENT": "false",
                "ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS": "false",
                "ALPHARAVIS_PIXELLE_PREPARE_COMFY": "false",
            }
        )
        print("Custom model/power management disabled in .env")
        return

    update_env_value("ALPHARAVIS_ENABLE_MODEL_MANAGEMENT", "true")
    advanced = ask_yes_no(
        "Enable advanced hooks (power_management_agent, Pixelle preflight, future crisis manager)",
        default=values.get("ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT", "false").lower() in {"1", "true", "yes"},
    )
    update_env_value("ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT", "true" if advanced else "false")

    if not advanced:
        set_many(
            {
                "ALPHARAVIS_ENABLE_CRISIS_MANAGER": "false",
                "ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS": "false",
                "ALPHARAVIS_ENABLE_POWER_MANAGEMENT": "false",
                "ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS": "false",
                "ALPHARAVIS_PIXELLE_PREPARE_COMFY": "false",
            }
        )
        print("Basic model-management planning enabled; advanced hooks disabled.")
        return

    values = read_env(ENV_PATH)
    prompts = [
        ("ALPHARAVIS_ENABLE_POWER_MANAGEMENT", "Enable power-management intents", False),
        ("ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS", "Enable owner-only SSH/Wake-on-LAN tools", False),
        ("ALPHARAVIS_PIXELLE_PREPARE_COMFY", "Check ComfyUI before Pixelle jobs", False),
        ("ALPHARAVIS_PIXELLE_BLOCK_IF_COMFY_OFFLINE", "Block Pixelle if ComfyUI is offline", False),
        ("ALPHARAVIS_ENABLE_EMBEDDING_SCHEDULER", "Run embedding queue scheduler in LangGraph", False),
        ("ALPHARAVIS_ENABLE_VECTOR_BACKFILL_DAEMON", "Run bounded vector backfill daemon", False),
        ("ALPHARAVIS_ENABLE_CRISIS_MANAGER", "Enable future crisis-manager routing", False),
        ("ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS", "Allow real external model/power actions", False),
    ]
    for key, prompt, default in prompts:
        current = values.get(key, "true" if default else "false").lower() in {"1", "true", "yes"}
        update_env_value(key, "true" if ask_yes_no(prompt, default=current) else "false")

    values = read_env(ENV_PATH)
    print("Press Enter to keep text values.")
    for key, description in MODEL_MANAGEMENT_KEYS:
        if key.startswith("ALPHARAVIS_ENABLE_") or key in {
            "ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS",
            "ALPHARAVIS_PIXELLE_PREPARE_COMFY",
            "ALPHARAVIS_PIXELLE_BLOCK_IF_COMFY_OFFLINE",
            "ALPHARAVIS_PIXELLE_OWNER_WAKE_COMFY",
            "ALPHARAVIS_RESPONSES_REQUIRE_NATIVE",
            "ALPHARAVIS_DEEPAGENTS_REQUIRE_RESPONSES",
            "ALPHARAVIS_ENABLE_EMBEDDING_SCHEDULER",
            "ALPHARAVIS_ENABLE_VECTOR_BACKFILL_DAEMON",
            "BRIDGE_HARD_INPUT_HTTP_ERROR",
        }:
            continue
        current = values.get(key, "")
        answer = input(f"{key} [{current}] - {description}: ").strip()
        if answer:
            update_env_value(key, answer)
    print("Model-management .env settings updated")


def _bool_env_value(value: str) -> str:
    return "true" if value.strip().lower() in {"1", "true", "yes", "on"} else "false"


def _media_vision_args_present(
    *,
    vision_enabled: str = "",
    vision_url: str = "",
    vision_base_url: str = "",
    vision_model: str = "",
    vision_fallback: str = "",
) -> bool:
    return any(
        str(value or "").strip()
        for value in (vision_enabled, vision_url, vision_base_url, vision_model, vision_fallback)
    )


def configure_media_vision(
    *,
    vision_enabled: str = "",
    vision_url: str = "",
    vision_base_url: str = "",
    vision_model: str = "",
    vision_fallback: str = "",
    interactive: bool = True,
) -> None:
    ensure_env()
    values = read_env(ENV_PATH)
    if vision_enabled and vision_enabled.strip().lower() != "keep":
        update_env_value("ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY", _bool_env_value(vision_enabled))
    if vision_url:
        update_env_value("ALPHARAVIS_VISION_EMBEDDING_MODEL_URL", vision_url.strip())
    if vision_base_url:
        update_env_value("ALPHARAVIS_VISION_EMBEDDING_BASE_URL", vision_base_url.strip())
    if vision_model:
        update_env_value("ALPHARAVIS_VISION_EMBEDDING_MODEL", vision_model.strip())
    if vision_fallback:
        update_env_value("ALPHARAVIS_VISION_EMBEDDING_FALLBACK_MODEL", vision_fallback.strip())
    if not interactive:
        print("Media/vision .env settings updated")
        return

    media_enabled = ask_yes_no(
        "Enable AlphaRavis media gallery registration",
        default=values.get("ALPHARAVIS_ENABLE_MEDIA_GALLERY", "true").lower() in {"1", "true", "yes"},
    )
    update_env_value("ALPHARAVIS_ENABLE_MEDIA_GALLERY", "true" if media_enabled else "false")
    vision_enabled = ask_yes_no(
        "Enable separate vision/media pgvector table",
        default=values.get("ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY", "false").lower() in {"1", "true", "yes"},
    )
    update_env_value("ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY", "true" if vision_enabled else "false")

    values = read_env(ENV_PATH)
    prompts = [
        ("ALPHARAVIS_MEDIA_PUBLIC_BASE_URL", "host-visible media gallery base URL"),
        ("ALPHARAVIS_MEDIA_PORT", "host port for media gallery"),
        ("ALPHARAVIS_VIDEO_ANALYSIS_ENABLED", "enable explicit video download/frame analysis tools"),
        ("ALPHARAVIS_VIDEO_ANALYSIS_FPS", "default video analysis FPS"),
        ("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES", "maximum sampled frames per video"),
        ("ALPHARAVIS_VIDEO_ANALYSIS_CACHE_ROOT", "video analysis cache root"),
        ("ALPHARAVIS_VIDEO_ANALYSIS_MODEL_CARD_PATH", "video analysis model-card JSON path"),
        ("ALPHARAVIS_MEDIA_AUTO_INDEX_ENABLED", "master switch for automatic media indexing queues"),
        ("ALPHARAVIS_MEDIA_AUTO_INDEX_USER_UPLOADS", "auto-index user-uploaded/input videos"),
        ("ALPHARAVIS_MEDIA_AUTO_INDEX_PIXELLE_MCP_OUTPUTS", "auto-index Pixelle MCP / ComfyUI processed video outputs"),
        ("ALPHARAVIS_MEDIA_AUTO_INDEX_LINK_REFERENCES", "auto-index pasted gallery/link references"),
        ("ALPHARAVIS_MEDIA_INDEX_VERSION", "media index version used for dedupe"),
        ("ALPHARAVIS_MEDIA_VISION_EMBEDDING_MODEL_CARD", "vision embedding model-card id"),
        ("ALPHARAVIS_VISION_EMBEDDING_MODEL_URL", "direct external OpenAI-compatible /v1 URL for vision embeddings"),
        ("ALPHARAVIS_VISION_EMBEDDING_BASE_URL", "OpenAI-compatible /v1 base for vision embeddings"),
        ("ALPHARAVIS_VISION_EMBEDDING_MODEL", "primary vision embedding LiteLLM model"),
        ("ALPHARAVIS_VISION_EMBEDDING_FALLBACK_MODEL", "fallback vision embedding LiteLLM model"),
        ("VISION_EMBEDDING_LITELLM_MODEL", "LiteLLM backend model name for vision-embed"),
        ("VISION_EMBEDDING_API_BASE", "backend OpenAI/Ollama /v1 URL for vision embeddings"),
    ]
    print("Press Enter to keep text values.")
    for key, description in prompts:
        current = values.get(key, "")
        answer = input(f"{key} [{current}] - {description}: ").strip()
        if answer:
            update_env_value(key, answer)
    print("Media/vision .env settings updated")


def configure_video_analysis(*, enabled: str, fps: str, max_frames: str) -> None:
    ensure_env()
    if enabled.lower() not in {"keep", ""}:
        update_env_value("ALPHARAVIS_VIDEO_ANALYSIS_ENABLED", "true" if enabled.lower() in {"1", "true", "yes", "on"} else "false")
    if fps:
        update_env_value("ALPHARAVIS_VIDEO_ANALYSIS_FPS", fps)
        update_env_value("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FPS", fps)
    if max_frames:
        update_env_value("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES", max_frames)
    print("Video-analysis .env settings updated")


def configure_openwebui() -> None:
    ensure_env()
    values = read_env(ENV_PATH)
    prompts_bool = [
        ("OPENWEBUI_ENABLE_OPENAI_API_PASSTHROUGH", "Enable OpenWebUI passthrough to AlphaRavis Bridge", True),
        ("OPENWEBUI_ENABLE_WEB_SEARCH", "Enable OpenWebUI web search", False),
    ]
    for key, prompt, default in prompts_bool:
        current = values.get(key, "true" if default else "false").lower() in {"1", "true", "yes"}
        update_env_value(key, "true" if ask_yes_no(prompt, default=current) else "false")

    values = read_env(ENV_PATH)
    prompts = [
        ("OPENWEBUI_PORT", "host port for OpenWebUI"),
        ("OPENWEBUI_DEFAULT_MODELS", "default model id shown in OpenWebUI"),
        ("OPENWEBUI_OPENAI_API_KEY", "API key OpenWebUI sends to AlphaRavis Bridge"),
        ("OPENWEBUI_RAG_WEB_SEARCH_ENGINE", "OpenWebUI web search engine"),
        ("OPENWEBUI_SEARXNG_QUERY_URL", "SearXNG query URL if web search is enabled"),
    ]
    print("Press Enter to keep text values.")
    for key, description in prompts:
        current = values.get(key, "")
        answer = input(f"{key} [{current}] - {description}: ").strip()
        if answer:
            update_env_value(key, answer)
    print("OpenWebUI .env settings updated")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    try:
        answer = input(f"{prompt} [{suffix}]: ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in {"y", "yes", "j", "ja", "true", "1"}


def normalize_yes_no_option(value: str) -> str:
    normalized = (value or "prompt").strip().lower()
    if normalized in {"", "prompt", "ask"}:
        return "prompt"
    if normalized in {"1", "true", "yes", "y", "ja", "j", "on"}:
        return "yes"
    if normalized in {"0", "false", "no", "n", "nein", "off"}:
        return "no"
    raise ValueError(f"Unsupported yes/no option {value!r}; use prompt, yes, or no")


def should_run_option(value: str, prompt: str, *, default: bool) -> bool:
    normalized = normalize_yes_no_option(value)
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    return ask_yes_no(prompt, default=default)


def split_profiles(value: str) -> list[str]:
    cleaned = (value or "").strip()
    if cleaned.lower() in {"", "none", "off", "false", "no", "-"}:
        return []
    return [item.strip() for item in cleaned.replace(";", ",").split(",") if item.strip()]


def normalize_profiles(value: str) -> str:
    return ",".join(split_profiles(value))


def configure_compose_profiles(value: str = "prompt") -> str:
    ensure_env()
    values = read_env(ENV_PATH)
    current = values.get("COMPOSE_PROFILES", "")
    requested = (value or "prompt").strip()
    if requested.lower() in {"prompt", "ask"}:
        print("\nOptional Docker Compose profiles")
        print("- openwebui: start the optional OpenWebUI frontend")
        print("- hermes-dashboard: start the optional Hermes dashboard")
        print("Use comma-separated values, or none for the base stack only.")
        try:
            answer = input(f"COMPOSE_PROFILES [{current or 'none'}]: ").strip()
        except EOFError:
            answer = current
        requested = current if not answer else answer
    elif requested.lower() == "keep":
        requested = current
    profiles = normalize_profiles(requested)
    update_env_value("COMPOSE_PROFILES", profiles)
    print(f"Compose profiles set to {profiles or 'none'}")
    return profiles


def compose_command(profiles: str = "") -> list[str]:
    cmd = ["docker", "compose"]
    for profile in split_profiles(profiles):
        cmd.extend(["--profile", profile])
    return cmd


def install(
    *,
    streaming_mode: str = "prompt",
    submodules: str = "prompt",
    build: str = "prompt",
    start: str = "prompt",
    profiles: str = "prompt",
    vision_enabled: str = "",
    vision_url: str = "",
    vision_base_url: str = "",
    vision_model: str = "",
    vision_fallback: str = "",
) -> None:
    ensure_env()
    configure_streaming(streaming_mode)
    if ask_yes_no("Edit important .env values now", default=False):
        configure()
    if ask_yes_no("Configure custom model/power management now", default=False):
        configure_model_management()
    has_media_args = _media_vision_args_present(
        vision_enabled=vision_enabled,
        vision_url=vision_url,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        vision_fallback=vision_fallback,
    )
    if has_media_args:
        configure_media_vision(
            vision_enabled=vision_enabled,
            vision_url=vision_url,
            vision_base_url=vision_base_url,
            vision_model=vision_model,
            vision_fallback=vision_fallback,
            interactive=False,
        )
    elif ask_yes_no("Configure media gallery / vision embeddings now", default=False):
        configure_media_vision()
    if ask_yes_no("Configure OpenWebUI frontend now", default=False):
        configure_openwebui()
    selected_profiles = configure_compose_profiles(profiles)
    if should_run_option(submodules, "Initialize/update submodules now", default=True):
        run(["git", "submodule", "update", "--init", "--recursive"])
    should_start = should_run_option(start, "Build images and start the stack now", default=True)
    should_build = False if should_start else should_run_option(build, "Build Docker images now", default=True)
    if should_start:
        run([*compose_command(selected_profiles), "up", "-d", "--build"])
    elif should_build:
        run([*compose_command(selected_profiles), "build"])
    print_status()
    if should_start:
        print("Stack start requested. Use make logs or make status to inspect it.")
    else:
        print("Next: make up")


def update(
    *,
    streaming_mode: str = "prompt",
    submodules: str = "prompt",
    build: str = "yes",
    start: str = "yes",
    profiles: str = "prompt",
    vision_enabled: str = "",
    vision_url: str = "",
    vision_base_url: str = "",
    vision_model: str = "",
    vision_fallback: str = "",
) -> None:
    ensure_env()
    run(["git", "pull", "--ff-only"])
    configure_streaming(streaming_mode)
    selected_profiles = configure_compose_profiles(profiles)
    if should_run_option(submodules, "Update submodules to their configured remote branches", default=True):
        run(["git", "submodule", "update", "--init", "--recursive", "--remote"])
    if ask_yes_no("Edit important .env values after update", default=False):
        configure()
    if ask_yes_no("Configure custom model/power management after update", default=False):
        configure_model_management()
    has_media_args = _media_vision_args_present(
        vision_enabled=vision_enabled,
        vision_url=vision_url,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        vision_fallback=vision_fallback,
    )
    if has_media_args:
        configure_media_vision(
            vision_enabled=vision_enabled,
            vision_url=vision_url,
            vision_base_url=vision_base_url,
            vision_model=vision_model,
            vision_fallback=vision_fallback,
            interactive=False,
        )
    elif ask_yes_no("Configure media gallery / vision embeddings after update", default=False):
        configure_media_vision()
    if ask_yes_no("Configure OpenWebUI after update", default=False):
        configure_openwebui()
    should_start = should_run_option(start, "Build images and start/recreate the stack after update", default=True)
    should_build = False if should_start else should_run_option(build, "Build Docker images after update", default=True)
    if should_start:
        run([*compose_command(selected_profiles), "up", "-d", "--build"])
    elif should_build:
        run([*compose_command(selected_profiles), "build"])
    print_status()


def docker_ps() -> None:
    try:
        result = subprocess.run(
            ["docker", "compose", "ps"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        print("docker compose not found on PATH")
        return
    if result.returncode == 0:
        print(result.stdout.rstrip())
    else:
        detail = (result.stderr or result.stdout).strip()
        print(f"docker compose is not reachable right now: {detail}")


def resolve_url(value: str, env: dict[str, str]) -> str:
    if value.isupper():
        return env.get(value, "")
    return ENV_INTERPOLATION_RE.sub(lambda match: env.get(match.group(1), match.group(2) or ""), value)


def print_status() -> None:
    env = read_env(ENV_PATH if ENV_PATH.exists() else EXAMPLE_PATH)
    print("\nAlphaRavis service URLs")
    for label, value in SERVICE_URLS:
        print(f"- {label}: {resolve_url(value, env)}")
    print("\nLibreChat model picker")
    print("- LangGraph Agent: custom endpoint -> api-bridge:8123/v1")
    print("- Hermes Agent: custom endpoint -> hermes-agent:8642/v1")
    print("- OpenAI: only appears if LIBRECHAT_OPENAI_API_KEY/REVERSE_PROXY are set")
    print("\nCustom model management")
    print(f"- Model management: {env.get('ALPHARAVIS_ENABLE_MODEL_MANAGEMENT', 'false')}")
    print(f"- Advanced hooks: {env.get('ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT', 'false')}")
    print(f"- Owner power tools: {env.get('ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS', 'false')}")
    print(f"- Crisis manager: {env.get('ALPHARAVIS_ENABLE_CRISIS_MANAGER', 'false')}")
    print(f"- Real action endpoint: {'configured' if env.get('ALPHARAVIS_MODEL_MGMT_ACTION_URL') else 'not configured'}")
    print("\nResponses / streaming")
    print(f"- Install profile: {current_streaming_mode(env)}")
    print(f"- Direct calls API mode: {env.get('ALPHARAVIS_LLM_API_MODE', 'responses')}")
    print(f"- ChatLiteLLM streaming: {env.get('ALPHARAVIS_LLM_STREAMING', 'true')}")
    print(f"- DeepAgents API mode: {env.get('ALPHARAVIS_DEEPAGENTS_API_MODE', 'responses')}")
    print(f"- DeepAgents streaming: {env.get('ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING', 'true')}")
    print(
        "- DeepAgents disable_streaming: "
        f"{env.get('ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING', 'tool_calling')}"
    )
    print(f"- Experimental tool-stream patch: {env.get('ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING', 'false')}")
    print(f"- Bridge preferred API: {env.get('BRIDGE_PREFERRED_API_MODE', 'responses')}")
    print(f"- Compose profiles: {env.get('COMPOSE_PROFILES', '') or 'none'}")
    bind = env.get("ALPHARAVIS_DOCKER_HOST_BIND", "0.0.0.0")
    mode = "Tailscale Serve HTTPS" if bind == "127.0.0.1" else "LAN HTTP"
    print("\nNetwork exposure")
    print(f"- Mode: {mode}")
    print(f"- Docker host bind: {bind}")
    print("\nMedia / vision")
    print(f"- Media gallery: {env.get('ALPHARAVIS_ENABLE_MEDIA_GALLERY', 'true')}")
    print(f"- Vision vector memory: {env.get('ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY', 'false')}")
    print(f"- Raw media to bridge context: {env.get('BRIDGE_ALLOW_RAW_MEDIA_CONTEXT', 'false')}")
    print(f"- Bridge auto-register incoming videos: {env.get('BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_VIDEOS', 'true')}")
    print(f"- Video analysis: {env.get('ALPHARAVIS_VIDEO_ANALYSIS_ENABLED', 'true')}")
    print(f"- Video analysis FPS/max frames: {env.get('ALPHARAVIS_VIDEO_ANALYSIS_FPS', '1')} / {env.get('ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES', '100')}")
    print(f"- Video analysis cache: {env.get('ALPHARAVIS_VIDEO_ANALYSIS_CACHE_ROOT', '/workspace/media-data/analysis-cache')}")
    print("\nOpenWebUI")
    print(f"- Profile: docker compose --profile openwebui up -d openwebui")
    print(f"- Passthrough: {env.get('OPENWEBUI_ENABLE_OPENAI_API_PASSTHROUGH', 'true')}")
    print("- Native tool calling: enable per model in OpenWebUI UI when the model supports it")
    print("\nDocker status")
    docker_ps()


def http_json(url: str, *, api_key: str = "", payload: dict | None = None, timeout: int = 15) -> str:
    data = None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return json.dumps(
            {
                "ok": False,
                "url": url,
                "status": exc.code,
                "reason": exc.reason,
                "body": body[:2000],
            },
            indent=2,
            ensure_ascii=False,
        )
    except error.URLError as exc:
        return json.dumps(
            {
                "ok": False,
                "url": url,
                "error": str(exc.reason),
            },
            indent=2,
            ensure_ascii=False,
        )


def bridge_smoke() -> None:
    env = read_env(ENV_PATH)
    base = env.get("BRIDGE_EXTERNAL_API_BASE", "http://localhost:8123/v1").rstrip("/")
    body = {
        "model": env.get("OPENAI_MODEL_NAME", "my-agent"),
        "messages": [{"role": "user", "content": "Antworte nur mit OK."}],
        "stream": False,
    }
    print(http_json(f"{base}/chat/completions", api_key="sk-1234", payload=body, timeout=30))


def hermes_smoke() -> None:
    env = read_env(ENV_PATH)
    base = env.get("HERMES_EXTERNAL_API_BASE", "http://localhost:8642/v1").rstrip("/")
    body = {
        "model": env.get("HERMES_MODEL", "hermes-agent"),
        "messages": [{"role": "user", "content": "Antworte nur mit OK."}],
        "stream": False,
    }
    print(http_json(f"{base}/chat/completions", api_key=env.get("HERMES_API_KEY", ""), payload=body, timeout=60))


def media_smoke() -> None:
    env = read_env(ENV_PATH)
    base = env.get("ALPHARAVIS_MEDIA_PUBLIC_BASE_URL", "http://localhost:8130").rstrip("/")
    print(http_json(f"{base}/health", timeout=15))


def openwebui_smoke() -> None:
    env = read_env(ENV_PATH)
    port = env.get("OPENWEBUI_PORT", "3090")
    print(http_json(f"http://localhost:{port}/", timeout=15)[:1000])


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AlphaRavis setup helper")
    parser.add_argument(
        "command",
        choices=[
            "install",
            "configure",
            "model-management",
            "streaming",
            "profiles",
            "media-vision",
            "video-analysis",
            "openwebui",
            "network-mode",
            "update",
            "status",
            "bridge-smoke",
            "hermes-smoke",
            "media-smoke",
            "openwebui-smoke",
        ],
    )
    parser.add_argument(
        "--streaming-mode",
        default=os.getenv("STREAMING", os.getenv("ALPHARAVIS_INSTALL_STREAMING_MODE", "prompt")),
        help=(
            "Runtime profile: prompt, keep, responses-hybrid, responses-full, "
            "responses-nonstreaming, chat-full, or chat-nonstreaming. "
            "Aliases include hybrid, full, nonstreaming, and chat."
        ),
    )
    parser.add_argument(
        "--submodules",
        default=os.getenv("SUBMODULES", os.getenv("ALPHARAVIS_INSTALL_SUBMODULES", "prompt")),
        help="For install: prompt, yes, or no.",
    )
    parser.add_argument(
        "--build",
        default=os.getenv("BUILD", os.getenv("ALPHARAVIS_INSTALL_BUILD", "prompt")),
        help="For install: prompt, yes, or no. Ignored when start is yes.",
    )
    parser.add_argument(
        "--start",
        default=os.getenv("START", os.getenv("ALPHARAVIS_INSTALL_START", "prompt")),
        help="For install: prompt, yes, or no.",
    )
    parser.add_argument(
        "--profiles",
        default=os.getenv("PROFILES", os.getenv("COMPOSE_PROFILES", "prompt")),
        help="Compose profiles to store/use: prompt, keep, none, openwebui, hermes-dashboard, or comma-separated.",
    )
    parser.add_argument("--enabled", default=os.getenv("ENABLED", "keep"), help="For video-analysis: true, false, or keep.")
    parser.add_argument("--fps", default=os.getenv("FPS", ""), help="For video-analysis: sample FPS and max FPS.")
    parser.add_argument("--max-frames", default=os.getenv("MAX_FRAMES", ""), help="For video-analysis: maximum sampled frames.")
    parser.add_argument("--vision-enabled", default=os.getenv("VISION_ENABLED", ""), help="For media-vision/install/update: true, false, or keep.")
    parser.add_argument("--vision-url", default=os.getenv("VISION_URL", ""), help="Direct external OpenAI-compatible /v1 URL for vision embeddings.")
    parser.add_argument("--vision-base-url", default=os.getenv("VISION_BASE_URL", ""), help="LiteLLM/OpenAI-compatible fallback /v1 URL for vision embeddings.")
    parser.add_argument("--vision-model", default=os.getenv("VISION_MODEL", ""), help="Primary vision embedding model id.")
    parser.add_argument("--vision-fallback", default=os.getenv("VISION_FALLBACK", ""), help="Fallback vision embedding model id.")
    parser.add_argument(
        "--mode",
        default=os.getenv("NETWORK_MODE", os.getenv("ALPHARAVIS_NETWORK_MODE", "tailscale")),
        help="For network-mode: tailscale for localhost Docker binds, or lan for 0.0.0.0 LAN HTTP binds.",
    )
    args = parser.parse_args(argv)
    if args.command == "install":
        install(
            streaming_mode=args.streaming_mode,
            submodules=args.submodules,
            build=args.build,
            start=args.start,
            profiles=args.profiles,
            vision_enabled=args.vision_enabled,
            vision_url=args.vision_url,
            vision_base_url=args.vision_base_url,
            vision_model=args.vision_model,
            vision_fallback=args.vision_fallback,
        )
    elif args.command == "configure":
        configure()
    elif args.command == "model-management":
        configure_model_management()
    elif args.command == "streaming":
        configure_streaming(args.streaming_mode)
    elif args.command == "profiles":
        print_streaming_profiles()
    elif args.command == "media-vision":
        configure_media_vision(
            vision_enabled=args.vision_enabled,
            vision_url=args.vision_url,
            vision_base_url=args.vision_base_url,
            vision_model=args.vision_model,
            vision_fallback=args.vision_fallback,
            interactive=not _media_vision_args_present(
                vision_enabled=args.vision_enabled,
                vision_url=args.vision_url,
                vision_base_url=args.vision_base_url,
                vision_model=args.vision_model,
                vision_fallback=args.vision_fallback,
            ),
        )
    elif args.command == "video-analysis":
        configure_video_analysis(enabled=args.enabled, fps=args.fps, max_frames=args.max_frames)
    elif args.command == "openwebui":
        configure_openwebui()
    elif args.command == "network-mode":
        apply_network_mode(args.mode)
    elif args.command == "update":
        update(
            streaming_mode=args.streaming_mode,
            submodules=args.submodules,
            build=args.build,
            start=args.start,
            profiles=args.profiles,
            vision_enabled=args.vision_enabled,
            vision_url=args.vision_url,
            vision_base_url=args.vision_base_url,
            vision_model=args.vision_model,
            vision_fallback=args.vision_fallback,
        )
    elif args.command == "status":
        print_status()
    elif args.command == "bridge-smoke":
        bridge_smoke()
    elif args.command == "hermes-smoke":
        hermes_smoke()
    elif args.command == "media-smoke":
        media_smoke()
    elif args.command == "openwebui-smoke":
        openwebui_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
