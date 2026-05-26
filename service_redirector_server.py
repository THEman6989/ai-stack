from __future__ import annotations

import html
import json
import os
import re
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8090
DEFAULT_TAILSCALE_URLS_PATH = "/app/service-dashboard-data/tailscale_service_urls.json"
DEFAULT_ENV_PATH = "/app/.env"
DEFAULT_EXAMPLE_PATH = "/app/.env(exaple)"
DEFAULT_RUNTIME_SETTINGS_PATH = "/app/service-dashboard-data/runtime_settings.json"


def env_port(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


SERVICE_DASHBOARD_PUBLIC_PORT = env_port("ALPHARAVIS_SERVICE_DASHBOARD_PUBLIC_PORT", DEFAULT_PORT)
MEDIA_PORT = env_port("ALPHARAVIS_MEDIA_PORT", 8130)
TEST_UI_PORT = env_port("ALPHARAVIS_TEST_UI_PORT", 8140)
OPENWEBUI_PORT = env_port("OPENWEBUI_PORT", 3090)
TAILSCALE_URLS_PATH = Path(os.getenv("ALPHARAVIS_TAILSCALE_URLS_FILE", DEFAULT_TAILSCALE_URLS_PATH))
ENV_PATH = Path(os.getenv("ALPHARAVIS_SETTINGS_ENV_PATH", DEFAULT_ENV_PATH))
EXAMPLE_PATH = Path(os.getenv("ALPHARAVIS_SETTINGS_EXAMPLE_PATH", DEFAULT_EXAMPLE_PATH))
RUNTIME_SETTINGS_PATH = Path(os.getenv("ALPHARAVIS_RUNTIME_SETTINGS_FILE", DEFAULT_RUNTIME_SETTINGS_PATH))
URL_MODE = os.getenv("ALPHARAVIS_SERVICE_DASHBOARD_URL_MODE", "auto").strip().lower()
BOOLEAN_VALUES = {"true", "false"}
SECRET_MARKERS = ("API_KEY", "PASSWORD", "SECRET", "AUTH_TOKEN", "ACCESS_TOKEN", "BEARER_TOKEN", "SSH_PASS", "CREDS")
URL_MARKERS = ("URL", "URI", "API_BASE", "BASE_URL", "HOST")


FAVICON_SVG = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#0b1018"/><path d="M16 41V23l16-9 16 9v18l-16 9-16-9Z" fill="#47d7ac"/><path d="M24 37V27l8-5 8 5v10l-8 5-8-5Z" fill="#101620"/></svg>"""


SERVICES: list[dict[str, Any]] = [
    {
        "name": "AlphaRavis Dashboard",
        "service": "service-dashboard",
        "kind": "Navigation",
        "category": "web",
        "icon": "AR",
        "description": "Landing page for every local AlphaRavis service.",
        "host_url": f"http://localhost:{SERVICE_DASHBOARD_PUBLIC_PORT}",
        "docker_url": "http://service-dashboard:8090",
        "port": SERVICE_DASHBOARD_PUBLIC_PORT,
        "accent": "#47d7ac",
    },
    {
        "name": "LibreChat",
        "service": "librechat",
        "kind": "Main UI",
        "category": "web",
        "icon": "LC",
        "description": "Primary chat interface for AlphaRavis and Hermes.",
        "host_url": "http://localhost:3080",
        "docker_url": "http://librechat:3080",
        "port": 3080,
        "accent": "#7c5cff",
    },
    {
        "name": "Settings",
        "service": "service-dashboard-settings",
        "kind": "Runtime Config",
        "category": "web",
        "icon": "ST",
        "description": "Mobile-friendly runtime and .env settings UI generated from .env(exaple).",
        "host_url": f"http://localhost:{SERVICE_DASHBOARD_PUBLIC_PORT}/settings",
        "tailscale_public_path": "/settings",
        "docker_url": "http://service-dashboard:8090/settings",
        "port": SERVICE_DASHBOARD_PUBLIC_PORT,
        "accent": "#7dd3fc",
    },
    {
        "name": "LangGraph API",
        "service": "langgraph-api",
        "kind": "Brain",
        "category": "api",
        "icon": "LG",
        "description": "Runs the AlphaRavis LangGraph graph alpha_ravis.",
        "host_url": "http://localhost:2024",
        "docker_url": "http://langgraph-api:2024",
        "port": 2024,
        "accent": "#45a3ff",
    },
    {
        "name": "LangGraph Studio",
        "service": "langgraph-api",
        "kind": "External Studio",
        "category": "web",
        "icon": "LS",
        "description": "LangSmith Studio pointed at the local LangGraph API.",
        "host_url": "https://smith.langchain.com/studio/?baseUrl=http://localhost:2024",
        "docker_url": "http://langgraph-api:2024",
        "port": 2024,
        "accent": "#5ec8ff",
    },
    {
        "name": "AlphaRavis Bridge",
        "service": "api-bridge",
        "kind": "OpenAI API",
        "category": "api",
        "icon": "AB",
        "description": "OpenAI-compatible bridge used by LibreChat and OpenWebUI.",
        "host_url": "http://localhost:8123",
        "docker_url": "http://api-bridge:8123",
        "port": 8123,
        "accent": "#ffbd4a",
    },
    {
        "name": "Bridge Test UI",
        "service": "bridge-test-ui",
        "kind": "Diagnostics",
        "category": "web",
        "icon": "BT",
        "description": "Minimal UI for streaming and protocol debugging.",
        "host_url": f"http://localhost:{TEST_UI_PORT}",
        "docker_url": "http://bridge-test-ui:8140",
        "port": TEST_UI_PORT,
        "accent": "#f46d8d",
    },
    {
        "name": "Hermes Agent",
        "service": "hermes-agent",
        "kind": "Coding Agent API",
        "category": "api",
        "icon": "HA",
        "description": "OpenAI-compatible coding and system specialist.",
        "host_url": "http://localhost:8642",
        "docker_url": "http://hermes-agent:8642",
        "port": 8642,
        "accent": "#d68cff",
    },
    {
        "name": "Hermes Dashboard",
        "service": "hermes-dashboard",
        "kind": "Optional UI",
        "category": "web",
        "icon": "HD",
        "description": "Optional Hermes dashboard, enabled by Compose profile.",
        "host_url": "http://localhost:9119",
        "docker_url": "http://hermes-dashboard:9119",
        "port": 9119,
        "profile": "hermes-dashboard",
        "accent": "#bb7cff",
    },
    {
        "name": "Hermes Web UI",
        "service": "hermes-webui",
        "kind": "Coding UI",
        "category": "web",
        "icon": "HW",
        "description": "Browser interface for Hermes Agent — sessions, workspace, coding.",
        "host_url": "http://localhost:8643",
        "docker_url": "http://hermes-webui:8643",
        "port": 8643,
        "accent": "#c084fc",
    },
    {
        "name": "LiteLLM",
        "service": "litellm",
        "kind": "Model Gateway UI",
        "category": "web",
        "icon": "LM",
        "description": "LiteLLM browser surface for model gateway inspection and admin tasks.",
        "host_url": "http://localhost:4000",
        "docker_url": "http://litellm:4000",
        "port": 4000,
        "accent": "#4bd2ff",
    },
    {
        "name": "LiteLLM API",
        "service": "litellm-api",
        "kind": "Model Gateway API",
        "category": "api",
        "icon": "LA",
        "description": "OpenAI-compatible LiteLLM proxy API used by AlphaRavis and Hermes.",
        "host_url": "http://localhost:4000/v1",
        "tailscale_public_path": "/v1",
        "docker_url": "http://litellm:4000/v1",
        "port": 4000,
        "accent": "#4bd2ff",
    },
    {
        "name": "RAG API",
        "service": "rag_api",
        "kind": "Retrieval API",
        "category": "api",
        "icon": "RG",
        "description": "Local document retrieval and embedding backend.",
        "host_url": "http://localhost:8000",
        "docker_url": "http://rag_api:8000",
        "port": 8000,
        "accent": "#75db6f",
    },
    {
        "name": "Media Gallery",
        "service": "media-gallery",
        "kind": "Media UI/API",
        "category": "web",
        "icon": "MG",
        "description": "Serves registered media, galleries, and analysis assets.",
        "host_url": f"http://localhost:{MEDIA_PORT}/gallery",
        "tailscale_public_path": "/gallery",
        "docker_url": "http://media-gallery:8130",
        "port": MEDIA_PORT,
        "accent": "#ff8a58",
    },
    {
        "name": "Deep Agents UI",
        "service": "deep-agents-ui",
        "kind": "Inspection UI",
        "category": "web",
        "icon": "DA",
        "description": "LangGraph/DeepAgents inspection frontend.",
        "host_url": "http://localhost:3000",
        "docker_url": "http://deep-agents-ui:3000",
        "port": 3000,
        "accent": "#6ee7f9",
    },
    {
        "name": "Agent Custom UI",
        "service": "agent-custom-ui",
        "kind": "Agent UI",
        "category": "web",
        "icon": "AU",
        "description": "Custom AlphaRavis frontend wired to alpha_ravis.",
        "host_url": "http://localhost:3001",
        "docker_url": "http://agent-custom-ui:3000",
        "port": 3001,
        "accent": "#a7f36d",
    },
    {
        "name": "OpenWebUI",
        "service": "openwebui",
        "kind": "Optional UI",
        "category": "web",
        "icon": "OW",
        "description": "Optional second frontend through the AlphaRavis bridge.",
        "host_url": f"http://localhost:{OPENWEBUI_PORT}",
        "docker_url": "http://openwebui:8080",
        "port": OPENWEBUI_PORT,
        "profile": "openwebui",
        "accent": "#f8df72",
    },
    {
        "name": "Pixelle",
        "service": "pixelle",
        "kind": "Media Tool UI",
        "category": "web",
        "icon": "PX",
        "description": "Pixelle image/video tool web surface.",
        "host_url": "http://localhost:9004",
        "docker_url": "http://pixelle:9004",
        "port": 9004,
        "accent": "#ff65b3",
    },
    {
        "name": "Pixelle MCP",
        "service": "pixelle-mcp",
        "kind": "Streamable HTTP MCP",
        "category": "api",
        "icon": "PM",
        "description": "Streamable HTTP MCP endpoint for Pixelle tool integration.",
        "host_url": "http://localhost:9004/pixelle/mcp",
        "tailscale_public_path": "/pixelle/mcp",
        "docker_url": "http://pixelle:9004/pixelle/mcp",
        "port": 9004,
        "accent": "#ff65b3",
    },
    {
        "name": "LangGraph Research UI",
        "service": "langgraph-api",
        "kind": "Agent Visual Port",
        "category": "infra",
        "icon": "LR",
        "description": "Experimental Research specialist port exposed by langgraph-api; use LangGraph Studio for normal graph inspection.",
        "host_url": "http://localhost:8760",
        "docker_url": "http://langgraph-api:8760",
        "port": 8760,
        "accent": "#7ad7ff",
        "non_http": True,
    },
    {
        "name": "LangGraph General UI",
        "service": "langgraph-api",
        "kind": "Agent Visual Port",
        "category": "infra",
        "icon": "LG",
        "description": "Experimental General specialist port exposed by langgraph-api; not guaranteed to serve a standalone browser UI.",
        "host_url": "http://localhost:8762",
        "docker_url": "http://langgraph-api:8762",
        "port": 8762,
        "accent": "#7ad7ff",
        "non_http": True,
    },
    {
        "name": "LangGraph Computer UI",
        "service": "langgraph-api",
        "kind": "Agent Visual Port",
        "category": "infra",
        "icon": "LC",
        "description": "Experimental Computer/CUA specialist port exposed by langgraph-api; not guaranteed to serve a standalone browser UI.",
        "host_url": "http://localhost:8764",
        "docker_url": "http://langgraph-api:8764",
        "port": 8764,
        "accent": "#7ad7ff",
        "non_http": True,
    },
    {
        "name": "LangGraph Debugger UI",
        "service": "langgraph-api",
        "kind": "Agent Visual Port",
        "category": "infra",
        "icon": "LD",
        "description": "Experimental Debugger specialist port exposed by langgraph-api; not guaranteed to serve a standalone browser UI.",
        "host_url": "http://localhost:8766",
        "docker_url": "http://langgraph-api:8766",
        "port": 8766,
        "accent": "#7ad7ff",
        "non_http": True,
    },
    {
        "name": "LangGraph Supervisor UI",
        "service": "langgraph-api",
        "kind": "Agent Visual Port",
        "category": "infra",
        "icon": "LV",
        "description": "Experimental Supervisor specialist port exposed by langgraph-api; not guaranteed to serve a standalone browser UI.",
        "host_url": "http://localhost:8768",
        "docker_url": "http://langgraph-api:8768",
        "port": 8768,
        "accent": "#7ad7ff",
        "non_http": True,
    },
    {
        "name": "LangGraph VNC",
        "service": "langgraph-api",
        "kind": "Remote Desktop",
        "category": "infra",
        "icon": "VN",
        "description": "VNC access to the LangGraph sandbox display.",
        "host_url": "vnc://localhost:5900",
        "docker_url": "langgraph-api:5900",
        "port": 5900,
        "accent": "#a2b2ff",
    },
    {
        "name": "MongoDB",
        "service": "mongodb",
        "kind": "Database",
        "category": "infra",
        "icon": "DB",
        "description": "LibreChat storage plus LangGraph checkpoint/state backing.",
        "host_url": "mongodb://localhost:27017",
        "docker_url": "mongodb://mongodb:27017",
        "port": 27017,
        "accent": "#5fd77b",
        "non_http": True,
    },
    {
        "name": "Redis",
        "service": "redis",
        "kind": "Cache",
        "category": "infra",
        "icon": "RD",
        "description": "Redis sidecar for cache and service coordination.",
        "host_url": "redis://localhost:6379",
        "docker_url": "redis://redis:6379",
        "port": 6379,
        "accent": "#ff6f68",
        "non_http": True,
    },
    {
        "name": "pgvector",
        "service": "vectordb",
        "kind": "Vector DB",
        "category": "infra",
        "icon": "PG",
        "description": "Postgres with pgvector for RAG and semantic memory.",
        "host_url": "postgresql://localhost:5432/rag_api",
        "docker_url": "postgresql://postgres@vectordb:5432/rag_api",
        "port": 5432,
        "accent": "#67b7ff",
        "non_http": True,
    },
]


def load_tailscale_payload() -> dict[str, Any]:
    if URL_MODE == "local" or not TAILSCALE_URLS_PATH.exists() or TAILSCALE_URLS_PATH.is_dir():
        return {}
    try:
        return json.loads(TAILSCALE_URLS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read Tailscale URL overrides from {TAILSCALE_URLS_PATH}: {exc}", flush=True)
        return {}


def effective_services() -> list[dict[str, Any]]:
    payload = load_tailscale_payload()
    host_overrides = payload.get("host_url_overrides", {})
    service_overrides = payload.get("redirector_overrides", {})
    tailscale_host = str(payload.get("tailscale_host") or "").strip().rstrip(".")
    route_ports = {
        int(route.get("port"))
        for route in payload.get("routes", [])
        if isinstance(route, dict) and str(route.get("port") or "").isdigit()
    }
    use_tailscale = URL_MODE in {"auto", "tailscale", "https"}
    services: list[dict[str, Any]] = []
    for original in SERVICES:
        service = dict(original)
        local_url = str(service.get("host_url", ""))
        tailscale_url = ""
        if use_tailscale and not service.get("non_http"):
            tailscale_url = str(host_overrides.get(local_url) or service_overrides.get(str(service.get("service", ""))) or "")
        service["local_url"] = local_url
        service["https_url"] = tailscale_url
        service["tailnet_http_url"] = ""
        parsed = urlparse(local_url)
        if tailscale_host and parsed.scheme == "http" and parsed.port is not None and not service.get("non_http"):
            service["tailnet_http_url"] = f"http://{tailscale_host}:{parsed.port}{parsed.path or ''}"
            if use_tailscale and not tailscale_url and parsed.port in route_ports:
                tailscale_url = f"https://{tailscale_host}:{parsed.port}{parsed.path or ''}"
        if tailscale_url:
            service["host_url"] = tailscale_url
            service["tailscale_url"] = tailscale_url
            service["url_mode"] = "tailscale"
        else:
            service["url_mode"] = "local"
        services.append(service)
    return services


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


def update_env_value(key: str, value: str) -> None:
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines() if ENV_PATH.exists() else []
    out: list[str] = []
    found = False
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
    tmp = ENV_PATH.with_name(f".{ENV_PATH.name}.tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.replace(tmp, ENV_PATH)


def _clean_comment(line: str) -> str:
    return line.strip()[1:].strip()


def _is_section_title(text: str) -> bool:
    return bool(text) and text.upper() == text and not text.startswith("=")


def parse_env_template(path: Path = EXAMPLE_PATH) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_section = "General"
    comments: list[str] = []
    awaiting_section_title = False
    defaults = read_env(path)

    def section_bucket(title: str) -> dict[str, Any]:
        for section in sections:
            if section["title"] == title:
                return section
        section = {"title": title, "entries": []}
        sections.append(section)
        return section

    if not path.exists():
        return []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("# ====="):
            awaiting_section_title = True
            comments = []
            continue
        if stripped.startswith("#"):
            text = _clean_comment(stripped)
            if awaiting_section_title and _is_section_title(text):
                current_section = text
                section_bucket(current_section)
                awaiting_section_title = False
                comments = []
                continue
            if text and not text.startswith("="):
                comments.append(text)
            continue
        awaiting_section_title = False
        if not stripped or "=" not in stripped:
            comments = []
            continue
        key = stripped.split("=", 1)[0].strip()
        section_bucket(current_section)["entries"].append(
            {
                "key": key,
                "default": defaults.get(key, ""),
                "description": " ".join(comments[-7:]),
            }
        )
        comments = []
    return [section for section in sections if section["entries"]]


def setting_category(key: str, section: str, description: str) -> str:
    key_lower = key.lower()
    text = f"{key} {section} {description}".lower()
    if "reviewer" in key_lower or "async_review" in key_lower:
        return "features"
    if "background_task" in key_lower or "background_context" in key_lower:
        return "features"
    if "run_state" in key_lower or "runtime_settings" in key_lower:
        return "runtime"
    if "service_dashboard" in key_lower or "tailscale" in key_lower:
        return "network"
    if "ubuntu_llama" in key_lower or "server_model_manager" in key_lower:
        return "model"
    if is_secret_key(key):
        return "security"
    mapping = [
        ("model", ("model", "llm", "litellm", "llama", "ollama", "embedding", "comfy")),
        ("streaming", ("stream", "responses", "deepagents")),
        ("memory", ("memory", "rag", "archive", "pgvector", "vector")),
        ("media", ("media", "vision", "video", "pixelle")),
        ("network", ("tailscale", "network", "port", "host", "url", "dashboard")),
        ("bridge", ("bridge", "librechat", "openwebui", "api-bridge")),
        ("runtime", ("runtime", "timeout", "seconds", "limit", "auto", "enable")),
    ]
    for category, needles in mapping:
        if any(needle in text for needle in needles):
            return category
    return "general"


def setting_tags(key: str, section: str, description: str, category: str) -> list[str]:
    key_lower = key.lower()
    text = f"{key} {section} {description}".lower()
    tags: list[str] = [category]
    tag_rules = [
        ("run-state", ("run_state", "resume", "checkpoint")),
        ("reviewer", ("reviewer", "async_review", "review after run")),
        ("background-tasks", ("background_task", "background_context", "latency-hiding")),
        ("runtime", ("runtime", "temporary", "resume", "timeout")),
        ("dashboard", ("service_dashboard", "dashboard")),
        ("tailscale", ("tailscale", "tailnet")),
        ("server-manager", ("server_model_manager", "server model manager")),
        ("ubuntu-llama", ("ubuntu_llama", "llama manager", "llama.cpp")),
        ("crisis", ("crisis", "recovery")),
        ("bridge", ("bridge", "librechat", "responses api", "openai_model_name")),
        ("provider", ("provider", "api_base", "api mode", "fallback_mode")),
        ("security", ("api_key", "secret", "password", "auth_token", "access_token", "ssh_pass")),
        ("model", ("model", "llm", "embedding", "temperature", "context")),
        ("network", ("url", "uri", "host", "port")),
        ("storage", ("mongodb", "_db", "_collection", "file")),
    ]
    for tag, needles in tag_rules:
        if any(needle in text or needle in key_lower for needle in needles):
            tags.append(tag)
    if is_secret_key(key):
        tags.append("security")
    if any(needle in key_lower for needle in ("power", "ubuntu_llama_esp", "reboot", "shutdown", "wake")):
        tags.append("power")
    fallback = fallback_info(key, description)
    if fallback["isFallback"]:
        tags.append("fallback")
    if fallback["deprecated"]:
        tags.append("legacy")
    return _dedupe_values(tags)[:5]


def fallback_info(key: str, description: str) -> dict[str, Any]:
    key_lower = key.lower()
    desc_lower = description.lower()
    is_fallback = "fallback" in key_lower or "fallback" in desc_lower
    deprecated = any(marker in desc_lower for marker in ("legacy", "older deployments", "deprecated", "veraltet"))
    fallback_for = ""
    if key in {"ALPHARAVIS_ACTIVE_TOKEN_LIMIT", "ALPHARAVIS_HANDOFF_CONTEXT_TOKEN_LIMIT"}:
        is_fallback = True
        deprecated = True
        fallback_for = "Fallback fuer Prozent-Kontextlimits, wenn ALPHARAVIS_ENABLE_PERCENT_CONTEXT_LIMITS=false ist oder automatische Kontextberechnung nicht genutzt wird."
    elif "PGVECTOR_FALLBACK_EMBEDDING_MODEL" in key:
        fallback_for = "Fallback-Embedding-Modell fuer pgvector Memory, wenn die primaere Embedding-Route nicht genutzt werden kann."
    elif "VISION_EMBEDDING_FALLBACK_MODEL" in key:
        fallback_for = "Fallback-Modell fuer Vision-Embeddings, wenn die primaere Vision-Embedding-Route nicht verfuegbar ist."
    elif "VISION_EMBEDDING_FALLBACK_TEXT" in key:
        fallback_for = "Fallback auf textbasierte Embeddings, wenn echte Vision-Embeddings nicht verfuegbar sind."
    elif "SERVER_MODEL_MANAGER_FALLBACK" in key:
        fallback_for = "Fallback fuer den Server Model Manager, wenn das primaere BigBoss/Provider-Modell ausfaellt."
    elif "FAST_PATH_FALLBACK" in key:
        fallback_for = "Fallback fuer Fast-Path-Antworten, wenn BigBoss nicht schnell genug antwortet."
    elif "fallback" in key_lower:
        fallback_for = "Fallback-Wert fuer den zugehoerigen primaeren Pfad."
    elif is_fallback:
        fallback_for = "Kompatibilitaets- oder Ersatzpfad laut .env(exaple)-Beschreibung."
    return {"isFallback": is_fallback, "deprecated": deprecated, "fallbackFor": fallback_for}


def setting_importance(key: str, section: str, description: str) -> int:
    text = f"{key} {section} {description}".lower()
    high = (
        "alpharavis_model",
        "api_base",
        "api_key",
        "enable",
        "streaming",
        "timeout",
        "tailscale",
        "port",
        "model_management",
        "server_model_manager",
        "ubuntu_llama",
        "big_llm",
        "comfy",
        "pixelle",
        "run_state",
        "reviewer",
        "async_review",
        "background_task",
        "background_context",
        "runtime_settings",
        "crisis",
        "curated_memory_auto_accept",
    )
    medium = ("url", "mode", "limit", "seconds", "memory", "rag", "media", "bridge", "context")
    if any(item in text for item in high):
        return 90
    if any(item in text for item in medium):
        return 60
    return 25


def inferred_allowed_values(key: str, description: str) -> list[str]:
    text = description or ""
    lowered_key = key.lower()
    match = re.search(r"Allowed values:\s*([^\.]+)", text, flags=re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
        keyed = [
            value
            for value in re.findall(r"(?:^|\s)([A-Za-z0-9_.-]+)\s*=", raw)
            if not ("_" in value and value.upper() == value)
        ]
        if keyed:
            return _dedupe_values(keyed)
        values = [part.strip().strip("`") for part in re.split(r"[,/|]", raw) if part.strip()]
        cleaned = [
            value
            for value in values
            if value
            and len(value) <= 64
            and " " not in value
            and not value.endswith(":")
        ]
        if cleaned:
            return _dedupe_values(cleaned)
    keyed = re.findall(r"(?:^|\s)([A-Za-z0-9_.-]+)\s*=", text)
    keyed = [value for value in keyed if value.lower() not in {"if", "default", "defaults"}]
    keyed = [value for value in keyed if not ("_" in value and value.upper() == value)]
    if len(keyed) >= 2:
        return _dedupe_values(keyed)
    if lowered_key.endswith("_omit_temperature_mode"):
        return ["auto", "always", "never"]
    if lowered_key.endswith("_token_limit_param_mode"):
        return ["auto", "max_completion_tokens", "max_tokens", "none"]
    if lowered_key.endswith("_provider_profile"):
        return ["auto", "local_litellm", "kimi_moonshot", "openai_reasoning", "responses_required"]
    if lowered_key.endswith("_api_mode"):
        return ["responses", "chat_completions"]
    if lowered_key.endswith("_url_mode"):
        return ["auto", "local", "tailscale"]
    if lowered_key.endswith("_sudo_mode"):
        return ["auto", "always", "never"]
    if lowered_key.endswith("_fallback_mode") or lowered_key.endswith("_require_responses_mode"):
        return ["auto", "always", "never"]
    if lowered_key.endswith("_splitter"):
        return ["auto", "langchain", "alpharavis"]
    if lowered_key.endswith("_log_level"):
        return ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    return []


def _dedupe_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip().strip("`")
        marker = cleaned.lower()
        if not cleaned or marker in seen:
            continue
        seen.add(marker)
        result.append(cleaned)
    return result[:12]


def fallback_description(key: str, description: str) -> str:
    if description.strip():
        text = description.strip()
        fallback = fallback_info(key, text)
        if fallback["fallbackFor"] and fallback["fallbackFor"] not in text:
            return f"{text} Fallback: {fallback['fallbackFor']}"
        return text
    lower = key.lower()
    if "run_state_manager_enabled" in lower:
        return "Aktiviert den durablen Run-State Manager, der Snapshots von Agent-Runs speichert."
    if "run_state_auto_resume" in lower:
        return "Wenn true, setzt AlphaRavis gespeicherte unterbrochene Agent-Jobs automatisch fort, statt im Thread nachzufragen."
    if "run_state_resume_prompt_timeout" in lower:
        return "Zeitlimit in Sekunden fuer die Bestaetigung der Fortsetzung eines unterbrochenen Runs."
    if "run_state_db" in lower:
        return "Mongo-Datenbank fuer durable Run-State-Checkpoints und Resume-Metadaten."
    if "run_state_collection" in lower:
        return "Mongo-Collection fuer den jeweils neuesten wiederaufnehmbaren Job-Checkpoint pro Thread."
    if "runtime_settings_file" in lower:
        return "JSON-Datei fuer temporaere Settings aus der Dashboard-WebUI; LangGraph laedt sie vor neuen Runs."
    if "async_reviewer_enabled" in lower:
        return "Aktiviert einen optionalen Hintergrund-Reviewer nach Agent-Runs. Der Reviewer schreibt nur Hinweise und korrigiert nichts automatisch."
    if "async_reviewer_model" in lower:
        return "Modell fuer den optionalen Hintergrund-Reviewer. Leer nutzt das primaere AlphaRavis Modell."
    if "async_reviewer_timeout" in lower:
        return "Zeitlimit in Sekunden fuer den Hintergrund-Reviewer nach einem Run."
    if "async_reviewer_min_output_chars" in lower:
        return "Mindestlaenge der finalen Antwort, ab der der optionale Hintergrund-Reviewer startet."
    if "async_review_store_path" in lower:
        return "JSON-Speicher fuer ausstehende Hintergrund-Review-Hinweise pro Thread."
    if "background_tasks_enabled" in lower:
        return "Aktiviert die parallele Background-Lane fuer kleine read-only Nebenaufgaben; LLM-Nebenjobs brauchen weiterhin Context-Leases."
    if "background_read_only_max_concurrency" in lower:
        return "Maximale Parallelitaet fuer ungefaehrliche read-only Background-Tools."
    if "background_small_llm_max_concurrency" in lower:
        return "Maximale Parallelitaet fuer kleine Background-LLM-Jobs nach Context-Lease-Pruefung."
    if "background_context_max_utilization" in lower:
        return "Maximaler Kontextanteil fuer Background-LLM-Leases, damit der Main-Agent priorisiert bleibt."
    if "background_task_timeout_seconds" in lower:
        return "Timeout in Sekunden fuer einzelne Background-Tasks."
    if "background_cancel_on_context_pressure" in lower:
        return "Wenn true, duerfen niedrige oder spekulative Background-Tasks bei Kontextdruck abgebrochen werden."
    if "code_window" in lower or "code_windows" in lower:
        return "Steuert Markdown-Codefenster-Unterstuetzung fuer Bridge- und LangGraph-Ausgaben."
    if "ubuntu_llama_manager_ip" in lower:
        return "IP-Adresse des Hosts, auf dem der Ubuntu Llama Manager laeuft."
    if "ubuntu_llama_manager_port" in lower:
        return "Port der Ubuntu Llama Manager API (Standard 8099)."
    if "ubuntu_llama_manager_url" in lower:
        return "Base-URL der externen Ubuntu Llama Manager API fuer die Steuerung von llama.cpp Instanzen."
    if "ubuntu_llama_manager_api_key" in lower:
        return "API-Key fuer den Zugriff auf die Ubuntu Llama Manager API (Bearer Auth)."
    if "ubuntu_llama_esp_url" in lower:
        return "Direkte URL zum ESP-Modul fuer Power-Aktionen, falls der Ubuntu-Host offline ist."
    if "ubuntu_llama_context_min" in lower:
        return "Untergrenze fuer Kontextfenster-Aenderungen am Ubuntu Llama Manager."
    if "ubuntu_llama_context_max" in lower:
        return "Obergrenze fuer Kontextfenster-Aenderungen am Ubuntu Llama Manager."
    if "ubuntu_llama_parallel_max" in lower:
        return "Obergrenze fuer llama.cpp --parallel Slots ueber den Ubuntu Llama Manager; parallel=2 nur bei sicherem VRAM-Fenster nutzen."
    if "pixelle_auto_shutdown_comfy_after_job" in lower:
        return "Wenn true, plant AlphaRavis nach einem Pixelle-Job ein ComfyUI-Shutdown, aber nur wenn ComfyUI fuer diesen Job geweckt wurde."
    if "pixelle_auto_shutdown_delay_seconds" in lower:
        return "Wartezeit in Sekunden nach abgeschlossenem Pixelle-Job, bevor ein zuvor geweckter ComfyUI-Server heruntergefahren wird."
    if "big_llm_auto_shutdown_after_managed_run" in lower:
        return "Policy-Schalter fuer den Power Manager: Einen nur fuer diesen Run eingeschalteten BigBoss/Ubuntu-Llama-Host nach dem Run wieder herunterfahren."
    if "big_llm_auto_shutdown_delay_seconds" in lower:
        return "Idle-Wartezeit in Sekunden, bevor ein nur fuer diesen Run eingeschalteter BigBoss/Ubuntu-Llama-Host wieder ausgeschaltet werden darf."
    if "server_model_manager_model_name" in lower:
        return "Modellname, unter dem der Server Model Manager fuer LibreChat sichtbar ist."
    if "server_model_manager_fallback_timeout" in lower:
        return "Timeout in Sekunden fuer den Fallback-Modellaufruf des Server Model Managers."
    if "server_model_manager_timeout" in lower:
        return "Timeout in Sekunden fuer den primaeren Server Model Manager Modellaufruf."
    if "server_model_manager_temperature" in lower:
        return "Sampling-Temperatur fuer den Server Model Manager; 0 haelt Diagnose- und Steuerantworten deterministisch."
    if "server_model_manager_fallback_model" in lower:
        return "Fallback-Modell fuer den Server Model Manager, falls BigBoss oder der primaere Provider ausfaellt."
    if "server_model_manager_model" in lower:
        return "Primaeres Modell fuer den dedizierten Server Model Manager Agent."
    if "context_min" in lower:
        return "Kleinster erlaubter Kontextwert fuer automatische oder manuelle Modellmanager-Aenderungen."
    if "context_max" in lower:
        return "Groesster erlaubter Kontextwert fuer automatische oder manuelle Modellmanager-Aenderungen."
    if "temperature" in lower:
        return "Sampling-Temperatur fuer diesen Modellpfad."
    if "timeout" in lower or lower.endswith("_seconds"):
        return "Zeitlimit in Sekunden fuer diesen Ablauf."
    if "api_base" in lower or lower.endswith("_base_url") or lower.endswith("_url"):
        return "Endpoint-URL, die dieser Dienst oder Provider fuer Requests verwendet."
    if "api_key" in lower or "secret" in lower or "token" in lower:
        return "Geheimer Zugriffswert fuer den zugehoerigen Dienst. Nur aendern, wenn der Zielservice denselben Wert erwartet."
    if "model" in lower:
        return "Modellname oder Modellprofil fuer diesen Pfad."
    if "stream" in lower:
        return "Steuert Streaming-Verhalten fuer diesen Pfad."
    if "port" in lower:
        return "Lokaler Port fuer diesen Dienst."
    if "enable" in lower or lower.startswith("enable_"):
        return "Aktiviert oder deaktiviert diese Funktion."
    if lower.endswith("_mode") or lower.endswith("_profile"):
        return "Betriebsmodus fuer diese Funktion. Auto waehlt die konservative Standardlogik."
    return "Konfigurationswert aus .env(exaple)."


def infer_kind(key: str, value: str, description: str, allowed: list[str] | None = None) -> str:
    lowered = value.strip().lower()
    desc_lower = description.lower()
    allowed = allowed if allowed is not None else inferred_allowed_values(key, description)
    if lowered in BOOLEAN_VALUES or "allowed values: true, false" in desc_lower:
        return "bool"
    if len(allowed) > 1 and len(allowed) <= 12:
        return "select"
    if value.startswith(("http://", "https://")) or any(marker in key for marker in URL_MARKERS):
        return "url"
    if key.endswith(("_PORT", "_SECONDS", "_LIMIT", "_CHARS", "_COUNT", "_MAX", "_MIN", "_TEMPERATURE", "_RATIO")):
        return "number"
    return "text"


def is_secret_key(key: str) -> bool:
    upper = key.upper()
    non_secret_markers = ("TOKEN_LIMIT", "TOKEN_COUNT", "TOKEN_BUDGET", "TOKEN_WINDOW", "TOKEN_RATIO")
    if any(marker in upper for marker in non_secret_markers):
        return False
    return any(marker in upper for marker in SECRET_MARKERS)


def load_runtime_settings() -> dict[str, Any]:
    if not RUNTIME_SETTINGS_PATH.exists():
        return {"values": {}, "updated_at": 0}
    try:
        payload = json.loads(RUNTIME_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"values": {}, "updated_at": 0, "error": "invalid runtime settings JSON"}
    values = payload.get("values") if isinstance(payload, dict) else {}
    return {
        "values": values if isinstance(values, dict) else {},
        "updated_at": payload.get("updated_at", 0) if isinstance(payload, dict) else 0,
    }


def write_runtime_settings(values: dict[str, str]) -> None:
    RUNTIME_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": time.time(), "values": values}
    tmp = RUNTIME_SETTINGS_PATH.with_name(f".{RUNTIME_SETTINGS_PATH.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, RUNTIME_SETTINGS_PATH)


def settings_model() -> dict[str, Any]:
    template = parse_env_template(EXAMPLE_PATH)
    current = read_env(ENV_PATH)
    runtime = load_runtime_settings()
    runtime_values = runtime.get("values") if isinstance(runtime.get("values"), dict) else {}
    sections: list[dict[str, Any]] = []
    for section in template:
        section_title = str(section["title"])
        entries: list[dict[str, Any]] = []
        for raw_entry in section["entries"]:
            order = sum(len(existing["entries"]) for existing in sections) + len(entries)
            key = str(raw_entry["key"])
            default = str(raw_entry.get("default", ""))
            env_value = current.get(key, default)
            runtime_value = str(runtime_values.get(key, "")) if key in runtime_values else ""
            effective = runtime_value if key in runtime_values else env_value
            raw_description = str(raw_entry.get("description", ""))
            description = fallback_description(key, raw_description)
            category = setting_category(key, section_title, description)
            importance = setting_importance(key, section_title, description)
            allowed = inferred_allowed_values(key, description)
            tags = setting_tags(key, section_title, description, category)
            fallback = fallback_info(key, description)
            entries.append(
                {
                    "key": key,
                    "value": effective,
                    "envValue": env_value,
                    "runtimeValue": runtime_value,
                    "hasRuntime": key in runtime_values,
                    "default": default,
                    "description": description,
                    "section": section_title,
                    "category": category,
                    "tags": tags,
                    "fallback": fallback["isFallback"],
                    "deprecated": fallback["deprecated"],
                    "fallbackFor": fallback["fallbackFor"],
                    "importance": importance,
                    "kind": infer_kind(key, effective or env_value or default, description, allowed),
                    "allowedValues": allowed,
                    "secret": is_secret_key(key),
                    "changed": env_value != default,
                    "envOrder": order,
                }
            )
        sections.append({"title": section_title, "entries": entries})
    return {
        "envPath": str(ENV_PATH),
        "examplePath": str(EXAMPLE_PATH),
        "runtimePath": str(RUNTIME_SETTINGS_PATH),
        "runtimeUpdatedAt": runtime.get("updated_at", 0),
        "sections": sections,
    }


def _template_keys() -> set[str]:
    return {
        str(entry["key"])
        for section in parse_env_template(EXAMPLE_PATH)
        for entry in section["entries"]
    }


def _clean_settings_values(raw: Any) -> dict[str, str]:
    allowed = _template_keys()
    if not isinstance(raw, dict):
        return {}
    values: dict[str, str] = {}
    for key, value in raw.items():
        text_key = str(key)
        if text_key not in allowed:
            continue
        values[text_key] = "" if value is None else str(value).replace("\r", "").replace("\n", " ")
    return values


def apply_runtime_settings(values: dict[str, str]) -> int:
    existing = load_runtime_settings().get("values", {})
    merged = dict(existing if isinstance(existing, dict) else {})
    merged.update(values)
    write_runtime_settings({str(key): str(value) for key, value in merged.items()})
    return len(values)


def save_permanent_settings(values: dict[str, str]) -> int:
    updated = 0
    for key, value in values.items():
        update_env_value(key, value)
        updated += 1
    return updated


def render_index() -> bytes:
    services = effective_services()
    service_json = json.dumps(services, ensure_ascii=True)
    web_services = [service for service in services if service.get("category") == "web"]
    api_services = [service for service in services if service.get("category") == "api"]
    infra_services = [service for service in services if service.get("category") == "infra"]
    web_cards = "\n".join(render_card(service) for service in web_services)
    api_cards = "\n".join(render_card(service, address_picker=True) for service in api_services)
    infra_cards = "\n".join(render_card(service, address_picker=True) for service in infra_services)
    mode_label = "Tailscale HTTPS" if any(service.get("url_mode") == "tailscale" for service in services) else "Localhost"
    body = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlphaRavis Service Dashboard</title>
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <style>
    :root {{
      color-scheme: dark;
      --bg: #080b10;
      --panel: #101620;
      --panel-2: #151d29;
      --line: #253144;
      --text: #eef4ff;
      --muted: #94a3b8;
      --soft: #c7d2e4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 15% 0%, rgba(71, 215, 172, .13), transparent 34rem),
        radial-gradient(circle at 88% 12%, rgba(124, 92, 255, .16), transparent 30rem),
        linear-gradient(145deg, #080b10 0%, #0d1118 48%, #10151d 100%);
      color: var(--text);
      font: 15px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1480px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 34px 0 48px;
    }}
    header {{
      display: grid;
      gap: 14px;
      margin-bottom: 28px;
    }}
    .eyebrow {{
      color: #47d7ac;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      max-width: 960px;
      font-size: clamp(28px, 6vw, 72px);
      line-height: .98;
      letter-spacing: -0.02em;
    }}
    .subhead {{
      max-width: 860px;
      margin: 0;
      color: var(--soft);
      font-size: 16px;
      line-height: 1.4;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      margin: 20px 0 16px;
    }}
    .search {{
      width: min(420px, 100%);
      min-height: 44px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(16, 22, 32, .86);
      color: var(--text);
      padding: 0 14px;
      outline: none;
      font-size: 16px; /* Prevents iOS zoom */
    }}
    .search:focus {{ border-color: #47d7ac; box-shadow: 0 0 0 3px rgba(71, 215, 172, .16); }}
    .count {{
      color: var(--muted);
      font-size: 13px;
    }}
    .mode {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      border: 1px solid rgba(71, 215, 172, .36);
      border-radius: 999px;
      padding: 0 12px;
      color: #d8fff2;
      background: rgba(71, 215, 172, .08);
      font-size: 12px;
      font-weight: 800;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 12px;
    }}
    .section {{
      margin-top: 14px;
    }}
    .section-head {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 12px;
      margin: 26px 0 10px;
    }}
    .section-head h2 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .section-head p {{
      margin: 0;
      max-width: 680px;
      color: var(--muted);
      font-size: 13px;
    }}
    details.section {{
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }}
    details.section > summary {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      cursor: pointer;
      list-style: none;
      margin-bottom: 12px;
    }}
    details.section > summary::-webkit-details-marker {{ display: none; }}
    details.section > summary h2 {{
      margin: 0;
      font-size: 18px;
    }}
    .summary-note {{
      color: var(--muted);
      font-size: 13px;
    }}
    .card {{
      min-height: 200px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      position: relative;
      overflow: hidden;
      text-decoration: none;
      color: inherit;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: linear-gradient(180deg, rgba(21, 29, 41, .96), rgba(12, 17, 25, .97));
      padding: 16px;
      cursor: pointer;
      transition: transform .16s ease, border-color .16s ease, background .16s ease;
    }}
    .card::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 4px;
      background: var(--accent, #47d7ac);
    }}
    .card:hover {{
      transform: translateY(-2px);
      border-color: color-mix(in srgb, var(--accent, #47d7ac) 70%, white 8%);
      background: linear-gradient(180deg, rgba(25, 35, 50, .98), rgba(13, 19, 28, .99));
    }}
    .card:active {{
      transform: translateY(0);
      background: var(--panel-2);
    }}
    .card[aria-disabled="true"] {{
      cursor: default;
      opacity: 0.8;
    }}
    .topline {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
    }}
    .identity {{
      display: flex;
      align-items: flex-start;
      gap: 10px;
      min-width: 0;
    }}
    .logo {{
      flex: none;
      display: grid;
      place-items: center;
      width: 42px;
      height: 42px;
      border: 1px solid color-mix(in srgb, var(--accent, #47d7ac) 46%, white 8%);
      border-radius: 8px;
      color: #081018;
      background: var(--accent, #47d7ac);
      font-weight: 900;
      font-size: 13px;
      letter-spacing: 0;
    }}
    .name {{
      margin: 0;
      font-size: 19px;
      line-height: 1.15;
      letter-spacing: -0.01em;
      font-weight: 700;
    }}
    .kind {{
      flex: none;
      max-width: 130px;
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 999px;
      padding: 3px 9px;
      color: var(--soft);
      font-size: 11px;
      text-align: center;
      overflow-wrap: anywhere;
      background: rgba(255,255,255,0.03);
    }}
    .description {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .meta {{
      display: grid;
      gap: 6px;
      margin-top: auto;
      color: var(--soft);
      font-size: 12px;
    }}
    .row {{
      display: grid;
      grid-template-columns: 52px minmax(0, 1fr);
      gap: 8px;
      align-items: baseline;
    }}
    .label {{
      color: var(--muted);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: .06em;
      text-transform: uppercase;
    }}
    code {{
      color: #f8fbff;
      font: 12px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
      background: rgba(255,255,255,0.05);
      padding: 1px 4px;
      border-radius: 4px;
    }}
    .actions {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-top: 4px;
      padding-top: 10px;
      border-top: 1px solid rgba(255,255,255,0.05);
    }}
    .open {{
      color: var(--accent, #47d7ac);
      font-weight: 800;
      font-size: 13px;
      text-decoration: none;
    }}
    .profile {{
      color: #f8df72;
      font-size: 11px;
      opacity: 0.9;
    }}
    .url-mode {{
      color: #47d7ac;
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    .address-list {{
      display: grid;
      gap: 6px;
      margin-top: 4px;
    }}
    .address {{
      display: grid;
      grid-template-columns: 80px minmax(0, 1fr) auto;
      gap: 7px;
      align-items: center;
    }}
    .address a {{
      color: #f8fbff;
      text-decoration: none;
      overflow-wrap: anywhere;
    }}
    .mini {{
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 6px;
      color: var(--soft);
      padding: 3px 6px;
      font-size: 11px;
      text-decoration: none;
      background: rgba(255,255,255,.04);
    }}
    @media (max-width: 780px) {{
      main {{ width: calc(100% - 32px); padding: 24px 0; }}
      h1 {{ font-size: clamp(26px, 10vw, 48px); }}
      .grid {{ grid-template-columns: 1fr; }}
      .card {{ min-height: auto; }}
      .description {{ -webkit-line-clamp: 4; }}
      .section-head {{ align-items: flex-start; flex-direction: column; }}
    }}
    @media (max-width: 480px) {{
      main {{ width: calc(100% - 24px); }}
      .toolbar {{ flex-direction: column; align-items: stretch; }}
      .mode {{ width: fit-content; }}
      .count {{ text-align: right; margin-top: -30px; }}
      .address {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">AlphaRavis Local Stack</div>
      <h1>Service Dashboard</h1>
      <p class="subhead">Alle wichtigen Docker-Compose-Services, öffentliche URLs und interne Docker-Adressen an einem Ort. Kacheln mit Web- oder API-URL öffnen direkt den jeweiligen Dienst.</p>
    </header>
    <section class="toolbar" aria-label="Dashboard filter">
      <input id="filter" class="search" type="search" placeholder="Service suchen" autocomplete="off">
      <span class="mode">{html.escape(mode_label)}</span>
      <div class="count"><span id="visible-count">{len(services)}</span> Services</div>
    </section>
    <section class="section" aria-label="Web Interfaces">
      <div class="section-head">
        <h2>Web Interfaces</h2>
        <p>Primäre Oberflächen fuer Chat, Beobachtung, Media und Agent-Inspektion. In Tailscale-Modus oeffnen die Karten bevorzugt HTTPS.</p>
      </div>
      <div class="grid">
        {web_cards}
      </div>
    </section>
    <details class="section" open>
      <summary>
        <h2>APIs</h2>
        <span class="summary-note">HTTP lokal, HTTP Tailnet und Tailscale HTTPS sichtbar nebeneinander.</span>
      </summary>
      <div class="grid">
        {api_cards}
      </div>
    </details>
    <details class="section">
      <summary>
        <h2>Infrastructure</h2>
        <span class="summary-note">Datenbanken, VNC und interne Protokolle.</span>
      </summary>
      <div class="grid">
        {infra_cards}
      </div>
    </details>
  </main>
  <script>
    window.ALPHARAVIS_SERVICES = {service_json};
    const input = document.getElementById("filter");
    const cards = Array.from(document.querySelectorAll("[data-card]"));
    const count = document.getElementById("visible-count");
    let pointerDownAt = 0;
    let suppressCardOpen = false;
    input.addEventListener("input", () => {{
      const q = input.value.trim().toLowerCase();
      let visible = 0;
      for (const card of cards) {{
        const haystack = card.dataset.search || "";
        const show = !q || haystack.includes(q);
        card.hidden = !show;
        if (show) visible += 1;
      }}
      count.textContent = String(visible);
    }});
    document.addEventListener("pointerdown", () => {{
      pointerDownAt = Date.now();
      suppressCardOpen = false;
    }}, {{ passive: true }});
    document.addEventListener("pointerup", () => {{
      const selected = window.getSelection ? String(window.getSelection()).trim() : "";
      if (Date.now() - pointerDownAt > 450 || selected) {{
        suppressCardOpen = true;
      }}
    }}, {{ passive: true }});
    document.addEventListener("click", (event) => {{
      const button = event.target.closest("[data-copy-url]");
      if (button) {{
        event.preventDefault();
        event.stopPropagation();
        if (!navigator.clipboard) return;
        navigator.clipboard.writeText(button.dataset.copyUrl || "");
        button.textContent = "Kopiert";
        window.setTimeout(() => {{ button.textContent = "Copy"; }}, 900);
        return;
      }}
      const selected = window.getSelection ? String(window.getSelection()).trim() : "";
      if (suppressCardOpen || selected) {{
        suppressCardOpen = false;
        return;
      }}
      if (event.target.closest("a, button, input, select, textarea, summary, [data-no-card-open]")) return;
      const card = event.target.closest("[data-open-url]");
      const url = card?.dataset.openUrl || "";
      if (url) window.open(url, "_blank", "noreferrer");
    }});
  </script>
</body>
</html>
"""
    return body.encode("utf-8")


def render_settings() -> bytes:
    model_json = json.dumps(settings_model(), ensure_ascii=True)
    body = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-title" content="AR Settings">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="theme-color" content="#081018">
  <link rel="manifest" href="/settings.webmanifest">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <title>AlphaRavis Settings</title>
  <style>
    :root {{ color-scheme: dark; --bg:#091012; --panel:rgba(16,23,28,.86); --panel2:rgba(22,30,36,.92); --line:rgba(226,238,242,.12); --text:#f4fafb; --muted:#a3b2b7; --soft:#d8e5e8; --accent:#82e8cc; --accent-dim:rgba(130,232,204,.16); --accent2:#9db7ef; --warn:#dfba6c; --danger:#e58b98; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; min-height:100vh; color:var(--text); font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:linear-gradient(145deg,#091012 0%,#10161b 48%,#11141b 100%); padding-bottom:calc(118px + env(safe-area-inset-bottom)); }}
    .shell {{ width:min(1500px,calc(100vw - 32px)); margin:0 auto; }}
    header {{ position:sticky; top:0; z-index:5; backdrop-filter:blur(22px); -webkit-backdrop-filter:blur(22px); background:rgba(9,16,18,.86); border-bottom:1px solid var(--line); padding:calc(14px + env(safe-area-inset-top)) 0 14px; }}
    .top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:14px; }}
    .brand {{ display:flex; align-items:center; gap:12px; min-width:0; }}
    .mark {{ width:42px; height:42px; display:grid; place-items:center; border-radius:10px; background:linear-gradient(135deg,#d8fff3,var(--accent)); color:#071111; font-weight:950; }}
    .eyebrow {{ color:var(--accent); font-size:11px; font-weight:900; letter-spacing:.08em; text-transform:uppercase; }}
    h1 {{ margin:0; font-size:clamp(26px,4.4vw,54px); line-height:1; letter-spacing:0; }}
    .paths {{ color:var(--muted); font-size:12px; overflow-wrap:anywhere; margin-top:5px; }}
    .hero-actions {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
    a.back, button.ghost {{ min-height:38px; border:1px solid var(--line); border-radius:8px; padding:9px 12px; color:var(--soft); background:rgba(255,255,255,.045); text-decoration:none; font:inherit; cursor:pointer; }}
    .controls {{ display:grid; grid-template-columns:minmax(220px,1.2fr) repeat(4,minmax(130px,.5fr)); gap:9px; margin-top:14px; }}
    input, select {{ width:100%; min-height:42px; border:1px solid var(--line); border-radius:9px; padding:9px 11px; background:rgba(7,12,16,.72); color:var(--text); font:inherit; font-size:16px; }}
    input:focus, select:focus, button:focus-visible {{ outline:2px solid rgba(121,242,208,.7); outline-offset:2px; }}
    .chips {{ display:flex; gap:8px; overflow-x:auto; padding:12px 0 2px; scrollbar-width:none; }}
    .chips::-webkit-scrollbar {{ display:none; }}
    .chip {{ flex:none; min-height:34px; border:1px solid var(--line); border-radius:999px; padding:7px 11px; color:var(--soft); background:rgba(255,255,255,.04); cursor:pointer; font:inherit; }}
    .chip.active {{ background:var(--accent-dim); color:#effffc; border-color:rgba(130,232,204,.38); font-weight:850; }}
    main {{ padding:18px 0 24px; }}
    .stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:4px 0 16px; }}
    .stat {{ border:1px solid var(--line); border-radius:10px; padding:12px; background:rgba(255,255,255,.035); }}
    .stat strong {{ display:block; font-size:22px; }}
    .stat span {{ color:var(--muted); font-size:12px; }}
    .section {{ margin:18px 0 24px; }}
    .section h2 {{ margin:0 0 10px; font-size:18px; letter-spacing:0; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(390px,1fr)); gap:8px; }}
    .card {{ border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:12px; box-shadow:0 12px 28px rgba(0,0,0,.18); display:grid; grid-template-columns:minmax(0,1fr) minmax(190px,34%); gap:14px; align-items:start; }}
    .card.changed {{ border-color:rgba(223,186,108,.38); }}
    .card.runtime {{ border-color:rgba(130,232,204,.34); }}
    .rowtop {{ display:flex; gap:9px; align-items:flex-start; }}
    .favorite {{ flex:none; width:30px; height:30px; border:1px solid var(--line); border-radius:8px; background:rgba(255,255,255,.035); color:var(--muted); cursor:pointer; font-size:16px; line-height:1; }}
    .favorite.on {{ color:#f3d584; border-color:rgba(243,213,132,.42); background:rgba(243,213,132,.09); }}
    .key {{ margin:0; font:800 14px/1.28 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; overflow-wrap:anywhere; }}
    .badges {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:7px; }}
    .badge {{ border:1px solid rgba(255,255,255,.12); border-radius:999px; padding:2px 7px; color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.04em; }}
    .badge.hot {{ color:#071111; background:var(--accent); border-color:var(--accent); font-weight:900; }}
    .badge.fallback {{ color:#f6e8be; border-color:rgba(223,186,108,.45); background:rgba(223,186,108,.09); }}
    .desc {{ color:var(--muted); font-size:13px; margin:7px 0 0; max-width:72ch; }}
    .field {{ margin-top:0; display:grid; gap:6px; }}
    .fieldlabel {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
    .secret-wrap {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:6px; }}
    .reveal {{ min-height:42px; border:1px solid var(--line); border-radius:9px; padding:0 10px; color:var(--soft); background:rgba(255,255,255,.045); cursor:pointer; font:inherit; }}
    .toggle {{ display:grid; grid-template-columns:1fr 1fr; border:1px solid var(--line); border-radius:8px; overflow:hidden; min-height:38px; }}
    .toggle button {{ border:0; background:rgba(255,255,255,.035); color:var(--soft); font:inherit; cursor:pointer; }}
    .toggle button.on {{ background:var(--accent-dim); color:#eafff7; font-weight:900; box-shadow:inset 0 0 0 1px rgba(130,232,204,.25); }}
    .meta {{ display:grid; gap:5px; margin-top:10px; color:var(--muted); font-size:11px; }}
    .meta code {{ color:var(--soft); overflow-wrap:anywhere; }}
    .bottom {{ position:fixed; left:max(12px,env(safe-area-inset-left)); right:max(12px,env(safe-area-inset-right)); bottom:max(12px,env(safe-area-inset-bottom)); z-index:10; display:grid; grid-template-columns:minmax(0,1fr) auto; gap:10px; align-items:end; pointer-events:none; }}
    .status {{ color:var(--muted); min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .actions {{ pointer-events:auto; border:1px solid var(--line); border-radius:14px; background:rgba(9,14,18,.9); backdrop-filter:blur(22px); -webkit-backdrop-filter:blur(22px); padding:10px; display:grid; grid-template-columns:minmax(190px,230px) minmax(190px,230px); gap:9px; align-items:start; }}
    .action-stack {{ display:grid; grid-template-rows:42px 18px; gap:6px; }}
    .primary, .secondary, .danger {{ width:100%; min-height:42px; border-radius:10px; border:1px solid var(--line); padding:10px 13px; font:inherit; font-weight:900; cursor:pointer; }}
    .secondary {{ background:rgba(138,180,255,.14); color:#e6efff; border-color:rgba(138,180,255,.35); }}
    .primary {{ background:var(--accent); color:#071111; border-color:var(--accent); }}
    .danger {{ background:rgba(255,143,155,.12); color:#ffe4e8; border-color:rgba(255,143,155,.35); }}
    .confirm {{ display:flex; align-items:center; justify-content:center; gap:6px; color:var(--muted); font-size:12px; }}
    .confirm input {{ width:auto; min-height:auto; }}
    @media(max-width:900px) {{ .controls {{ grid-template-columns:1fr 1fr; }} .stats {{ grid-template-columns:1fr 1fr; }} .grid {{ grid-template-columns:1fr; }} .card {{ grid-template-columns:1fr; }} .bottom {{ grid-template-columns:1fr; }} .actions {{ width:100%; grid-template-columns:1fr 1fr; }} .hero-actions {{ justify-content:flex-start; }} }}
    @media(max-width:540px) {{ .shell {{ width:calc(100vw - 20px); }} .top {{ display:grid; }} .controls {{ grid-template-columns:1fr; }} .stats {{ grid-template-columns:1fr 1fr; gap:8px; }} .card {{ padding:11px; }} .actions {{ grid-template-columns:1fr; }} h1 {{ font-size:30px; }} }}
  </style>
</head>
<body>
  <header><div class="shell">
    <div class="top">
      <div class="brand"><div class="mark">ST</div><div><div class="eyebrow">AlphaRavis Control</div><h1>Settings</h1><div class="paths" id="paths"></div></div></div>
      <div class="hero-actions"><a class="back" href="/">Dashboard</a><button class="ghost" id="resetFilters">Reset Filter</button></div>
    </div>
    <div class="controls">
      <input id="search" type="search" placeholder="Setting suchen">
      <select id="importance"><option value="all">Alle Wichtigkeiten</option><option value="important">Nur wichtige</option><option value="normal">Nur normale</option><option value="low">Nur unwichtige</option><option value="changed">Geaendert</option><option value="runtime">Runtime aktiv</option><option value="favorite">Favoriten</option><option value="fallback">Nur Fallback</option><option value="legacy">Nur Legacy</option><option value="fallbackLegacy">Fallback + Legacy</option></select>
      <select id="category"></select>
      <select id="exclude"><option value="none">Nichts ausblenden</option><option value="fallback">Fallback ausblenden</option><option value="legacy">Legacy ausblenden</option><option value="fallbackLegacy">Fallback + Legacy ausblenden</option></select>
      <select id="sort"><option value="importance">Nach Wichtigkeit</option><option value="envOrder">Nach .env Reihenfolge</option><option value="alpha">Alphabetisch</option><option value="section">Nach Bereich</option><option value="changed">Geaendert zuerst</option></select>
    </div>
    <div class="chips" id="chips"></div>
  </div></header>
  <main class="shell">
    <div class="stats"><div class="stat"><strong id="visible">0</strong><span>sichtbar</span></div><div class="stat"><strong id="changed">0</strong><span>geaendert</span></div><div class="stat"><strong id="runtime">0</strong><span>runtime</span></div><div class="stat"><strong id="total">0</strong><span>gesamt</span></div></div>
    <div id="settings"></div>
  </main>
  <div class="bottom">
    <div class="status" id="status">Bereit.</div>
    <div class="actions">
      <div class="action-stack"><button class="secondary" id="applyRuntime">Temporary anwenden</button><span></span></div>
      <div class="action-stack"><button class="primary" id="savePermanent">Permanent speichern</button><label class="confirm"><input id="dontAsk" type="checkbox"> nicht mehr fragen</label></div>
    </div>
  </div>
  <script>
    let model = {model_json};
    let entries = [];
    let draft = {{}};
    let activeCat = "all";
    let favorites = new Set(JSON.parse(localStorage.getItem("settingsFavorites") || "[]"));
    const $ = id => document.getElementById(id);
    function flatten() {{ entries = model.sections.flatMap(s => s.entries.map(e => ({{...e, section:s.title}}))); for (const e of entries) draft[e.key] = e.value ?? ""; }}
    function categories() {{ return ["all", ...Array.from(new Set(entries.map(e => e.category))).sort()]; }}
    function labelCat(c) {{ return c === "all" ? "Alle" : c[0].toUpperCase()+c.slice(1); }}
    function setStatus(t) {{ $("status").textContent = t; }}
    function changedKeys() {{ return entries.filter(e => String(draft[e.key] ?? "") !== String(e.value ?? "")).map(e => e.key); }}
    function inputFor(e) {{
      const val = String(draft[e.key] ?? "");
      if (e.kind === "bool") {{
        return `<label class="fieldlabel">Schalter</label><div class="toggle"><button type="button" class="${{val.toLowerCase()==='true'?'on':''}}" data-set="${{e.key}}" data-val="true">True</button><button type="button" class="${{val.toLowerCase()==='false'?'on':''}}" data-set="${{e.key}}" data-val="false">False</button></div>`;
      }}
      if (e.kind === "select") {{
        const opts = (e.allowedValues || []).map(v => `<option value="${{esc(v)}}" ${{v===val?'selected':''}}>${{esc(v)}}</option>`).join("");
        const custom = !(e.allowedValues||[]).includes(val) ? `<option value="${{esc(val)}}" selected>${{val ? esc(val) : "leer"}}</option>` : "";
        return `<label class="fieldlabel">Auswahl</label><select data-key="${{e.key}}">${{opts}}${{custom}}</select>`;
      }}
      const type = e.secret ? "password" : (e.kind === "number" ? "number" : (e.kind === "url" ? "url" : "text"));
      const input = `<input data-key="${{e.key}}" type="${{type}}" value="${{esc(val)}}" autocomplete="off" data-secret="${{e.secret ? '1' : ''}}">`;
      return `<label class="fieldlabel">Wert</label>${{e.secret ? `<div class="secret-wrap">${{input}}<button class="reveal" type="button" data-reveal>${{type === "password" ? "zeigen" : "verbergen"}}</button></div>` : input}}`;
    }}
    function esc(v) {{ return String(v ?? "").replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
    function filtered() {{
      const q = $("search").value.trim().toLowerCase();
      const imp = $("importance").value;
      let list = entries.filter(e => {{
        const hay = `${{e.key}} ${{e.description}} ${{e.section}} ${{e.category}} ${{(e.tags||[]).join(" ")}}`.toLowerCase();
        if (q && !hay.includes(q)) return false;
        if (activeCat !== "all" && e.category !== activeCat) return false;
        const exclude = $("exclude").value;
        if ((exclude === "fallback" || exclude === "fallbackLegacy") && e.fallback) return false;
        if ((exclude === "legacy" || exclude === "fallbackLegacy") && e.deprecated) return false;
        if (imp === "important" && e.importance < 80) return false;
        if (imp === "normal" && (e.importance < 50 || e.importance >= 80)) return false;
        if (imp === "low" && e.importance >= 50) return false;
        if (imp === "changed" && !e.changed) return false;
        if (imp === "runtime" && !e.hasRuntime) return false;
        if (imp === "favorite" && !favorites.has(e.key)) return false;
        if (imp === "fallback" && !e.fallback) return false;
        if (imp === "legacy" && !e.deprecated) return false;
        if (imp === "fallbackLegacy" && !(e.fallback || e.deprecated)) return false;
        return true;
      }});
      const sort = $("sort").value;
      list.sort((a,b) => sort === "alpha" ? a.key.localeCompare(b.key) : sort === "envOrder" ? (a.envOrder ?? 999999)-(b.envOrder ?? 999999) : sort === "section" ? (a.section+a.key).localeCompare(b.section+b.key) : sort === "changed" ? Number(b.changed)-Number(a.changed)||b.importance-a.importance : b.importance-a.importance||a.key.localeCompare(b.key));
      return list;
    }}
    function render() {{
      const list = filtered();
      $("visible").textContent = list.length; $("total").textContent = entries.length; $("changed").textContent = entries.filter(e=>e.changed).length; $("runtime").textContent = entries.filter(e=>e.hasRuntime).length;
      const grouped = new Map();
      for (const e of list) {{ const group = ["section","envOrder"].includes($("sort").value) ? e.section : labelCat(e.category); if (!grouped.has(group)) grouped.set(group, []); grouped.get(group).push(e); }}
      $("settings").innerHTML = Array.from(grouped.entries()).map(([group, items]) => `<section class="section"><h2>${{esc(group)}} · ${{items.length}}</h2><div class="grid">${{items.map(card).join("")}}</div></section>`).join("") || `<section class="section"><h2>Nichts gefunden</h2></section>`;
    }}
    function card(e) {{
      const tagBadges = (e.tags || []).filter(t => t !== e.category).slice(0, 4).map(t => `<span class="badge ${{t === 'fallback' ? 'fallback' : ''}}">${{esc(t)}}</span>`).join("");
      const fallbackText = e.fallbackFor ? `<p class="desc"><strong>Fallback fuer:</strong> ${{esc(e.fallbackFor)}}</p>` : "";
      return `<article class="card ${{e.changed?'changed':''}} ${{e.hasRuntime?'runtime':''}}"><div><div class="rowtop"><button class="favorite ${{favorites.has(e.key)?'on':''}}" type="button" data-fav="${{e.key}}" aria-label="Favorit">★</button><div><h3 class="key">${{esc(e.key)}}</h3><div class="badges"><span class="badge ${{e.importance>=80?'hot':''}}">${{e.importance>=80?'wichtig':e.importance>=50?'normal':'low'}}</span><span class="badge">${{esc(e.category)}}</span>${{tagBadges}}${{e.hasRuntime?'<span class="badge hot">runtime</span>':''}}</div><p class="desc">${{esc(e.description)}}</p>${{fallbackText}}</div></div><div class="meta"><div>ENV: <code>${{esc(e.envValue)}}</code></div><div>Default: <code>${{esc(e.default)}}</code></div></div></div><div class="field">${{inputFor(e)}}</div></article>`;
    }}
    function renderFilters() {{
      $("paths").textContent = `${{model.envPath}} · ${{model.examplePath}}`;
      $("category").innerHTML = categories().map(c => `<option value="${{c}}">${{labelCat(c)}}</option>`).join("");
      $("chips").innerHTML = categories().map(c => `<button class="chip ${{c===activeCat?'active':''}}" data-cat="${{c}}">${{labelCat(c)}}</button>`).join("");
    }}
    async function refresh() {{ model = await (await fetch("/api/settings")).json(); flatten(); renderFilters(); render(); }}
    async function post(url, values) {{ const r = await fetch(url, {{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify({{values}})}}); if (!r.ok) throw new Error(await r.text()); return r.json(); }}
    function changedPayload() {{ const out = {{}}; for (const key of changedKeys()) out[key] = draft[key]; return out; }}
    document.addEventListener("input", e => {{ if(e.target.dataset.key) {{ draft[e.target.dataset.key]=e.target.value; render(); }} if(["search","importance","sort","exclude"].includes(e.target.id)) render(); }});
    document.addEventListener("click", async e => {{
      const set = e.target.closest("[data-set]"); if (set) {{ draft[set.dataset.set]=set.dataset.val; render(); return; }}
      const reveal = e.target.closest("[data-reveal]"); if (reveal) {{ const input = reveal.parentElement.querySelector("input"); if (input.type === "password") {{ input.type = "text"; reveal.textContent = "verbergen"; }} else {{ input.type = "password"; reveal.textContent = "zeigen"; }} return; }}
      const fav = e.target.closest("[data-fav]"); if (fav) {{ favorites.has(fav.dataset.fav) ? favorites.delete(fav.dataset.fav) : favorites.add(fav.dataset.fav); localStorage.setItem("settingsFavorites", JSON.stringify(Array.from(favorites))); render(); return; }}
      const cat = e.target.closest("[data-cat]"); if (cat) {{ activeCat=cat.dataset.cat; $("category").value=activeCat; renderFilters(); render(); return; }}
    }});
    $("category").addEventListener("change", e => {{ activeCat=e.target.value; renderFilters(); render(); }});
    $("resetFilters").onclick = () => {{ $("search").value=""; $("importance").value="all"; $("exclude").value="none"; $("sort").value="importance"; activeCat="all"; $("category").value="all"; renderFilters(); render(); }};
    $("applyRuntime").onclick = async () => {{ const payload = changedPayload(); await post("/api/settings/runtime", payload); setStatus(`Temporary angewendet: ${{Object.keys(payload).length}} Werte. Neue Chat-Turns laden sie sofort.`); await refresh(); }};
    $("savePermanent").onclick = async () => {{ const payload = changedPayload(); if (!$("dontAsk").checked && !localStorage.getItem("settingsNoPermanentConfirm")) {{ if(!confirm("Wirklich permanent in .env speichern?")) return; }} if ($("dontAsk").checked) localStorage.setItem("settingsNoPermanentConfirm","1"); await post("/api/settings/permanent", payload); setStatus(`Permanent gespeichert: ${{Object.keys(payload).length}} Werte.`); await refresh(); }};
    flatten(); renderFilters(); render();
  </script>
</body>
</html>"""
    return body.encode("utf-8")


def settings_manifest() -> dict[str, Any]:
    return {
        "name": "AlphaRavis Settings",
        "short_name": "AR Settings",
        "start_url": "/settings",
        "display": "standalone",
        "background_color": "#071013",
        "theme_color": "#071013",
        "icons": [{"src": "/favicon.svg", "sizes": "64x64", "type": "image/svg+xml"}],
    }


def render_card(service: dict[str, Any], *, address_picker: bool = False) -> str:
    host_url = str(service["host_url"])
    local_url = str(service.get("local_url", host_url))
    https_url = str(service.get("https_url") or service.get("tailscale_url") or "")
    tailnet_http_url = str(service.get("tailnet_http_url") or "")
    disabled = bool(service.get("non_http"))
    tag = "div"
    search = " ".join(
        str(service.get(key, ""))
        for key in ("name", "service", "kind", "description", "host_url", "local_url", "tailscale_url", "docker_url", "profile")
    ).lower()
    profile = service.get("profile")
    open_label = "TCP endpoint" if disabled else "Öffnen"
    profile_html = f'<span class="profile">profile: {html.escape(str(profile))}</span>' if profile else "<span></span>"
    mode = str(service.get("url_mode", "local"))
    mode_html = f'<span class="url-mode">{html.escape(mode)}</span>'
    local_row = ""
    if host_url != local_url:
        local_row = f'<div class="row" data-no-card-open><span class="label">Local</span><code>{html.escape(local_url)}</code></div>'
    open_href = "" if disabled else f'<a class="open" href="{html.escape(host_url, quote=True)}" target="_blank" rel="noreferrer">{open_label}</a>'
    if disabled:
        open_href = f'<span class="open">{open_label}</span>'
    addresses = ""
    if address_picker and not disabled:
        address_rows = []
        for label, url, preferred in [
            ("HTTPS", https_url, mode == "tailscale"),
            ("Local", local_url, mode != "tailscale"),
            ("Tailnet", tailnet_http_url, False),
        ]:
            if not url:
                continue
            badge = " *" if preferred else ""
            address_rows.append(
                "<div class=\"address\" data-no-card-open>"
                f"<span class=\"label\">{html.escape(label + badge)}</span>"
                f"<a href=\"{html.escape(url, quote=True)}\" target=\"_blank\" rel=\"noreferrer\">{html.escape(url)}</a>"
                f"<button class=\"mini\" type=\"button\" data-copy-url=\"{html.escape(url, quote=True)}\">Copy</button>"
                "</div>"
            )
        if address_rows:
            addresses = f"<div class=\"address-list\">{''.join(address_rows)}</div>"
    return f"""
      <{tag} class="card" style="--accent: {html.escape(str(service["accent"]), quote=True)};" data-card data-open-url="{html.escape(host_url if not disabled else '', quote=True)}" data-search="{html.escape(search, quote=True)}" aria-disabled="{str(disabled).lower()}">
        <div class="topline">
          <div class="identity">
            <span class="logo">{html.escape(str(service.get("icon") or str(service["name"])[:2]).upper()[:3])}</span>
            <h2 class="name">{html.escape(str(service["name"]))}</h2>
          </div>
          <span class="kind">{html.escape(str(service["kind"]))}</span>
        </div>
        <p class="description">{html.escape(str(service["description"]))}</p>
        <div class="meta">
          <div class="row" data-no-card-open><span class="label">URL</span><code>{html.escape(host_url)}</code></div>
          {local_row}
          <div class="row" data-no-card-open><span class="label">Docker</span><code>{html.escape(str(service["docker_url"]))}</code></div>
          <div class="row" data-no-card-open><span class="label">Port</span><code>{html.escape(str(service["port"]))}</code></div>
          {addresses}
        </div>
        <div class="actions">
          {open_href}
          {mode_html}
          {profile_html}
        </div>
      </{tag}>"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "AlphaRavisServiceDashboard/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_bytes(render_index(), "text/html; charset=utf-8")
            return
        if parsed.path == "/settings":
            self.send_bytes(render_settings(), "text/html; charset=utf-8")
            return
        if parsed.path == "/settings.webmanifest":
            self.send_json(settings_manifest())
            return
        if parsed.path == "/api/settings":
            self.send_json(settings_model())
            return
        if parsed.path == "/favicon.svg":
            self.send_bytes(FAVICON_SVG, "image/svg+xml")
            return
        if parsed.path == "/services.json":
            self.send_json({"services": effective_services(), "url_mode": URL_MODE})
            return
        if parsed.path == "/health":
            self.send_json({"ok": True, "service": "service-dashboard"})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/settings/runtime", "/api/settings/permanent"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        try:
            payload = json.loads(self.rfile.read(max(0, min(length, 2_000_000))).decode("utf-8") or "{}")
        except Exception:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid JSON")
            return
        values = _clean_settings_values(payload.get("values") if isinstance(payload, dict) else {})
        if parsed.path == "/api/settings/runtime":
            updated = apply_runtime_settings(values)
            self.send_json({"ok": True, "mode": "runtime", "updated": updated, "runtimePath": str(RUNTIME_SETTINGS_PATH)})
            return
        updated = save_permanent_settings(values)
        self.send_json({"ok": True, "mode": "permanent", "updated": updated, "envPath": str(ENV_PATH)})

    def send_json(self, payload: dict[str, Any]) -> None:
        self.send_bytes(json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8"), "application/json")

    def send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    host = os.getenv("ALPHARAVIS_SERVICE_DASHBOARD_HOST", DEFAULT_HOST)
    port = env_port("ALPHARAVIS_SERVICE_DASHBOARD_PORT", DEFAULT_PORT)
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"AlphaRavis service dashboard listening on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
