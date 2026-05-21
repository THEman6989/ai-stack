from __future__ import annotations

import html
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8090
DEFAULT_TAILSCALE_URLS_PATH = "/app/service-dashboard-data/tailscale_service_urls.json"


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
URL_MODE = os.getenv("ALPHARAVIS_SERVICE_DASHBOARD_URL_MODE", "auto").strip().lower()


SERVICES: list[dict[str, Any]] = [
    {
        "name": "AlphaRavis Dashboard",
        "service": "service-dashboard",
        "kind": "Navigation",
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
        "description": "Primary chat interface for AlphaRavis and Hermes.",
        "host_url": "http://localhost:3080",
        "docker_url": "http://librechat:3080",
        "port": 3080,
        "accent": "#7c5cff",
    },
    {
        "name": "LangGraph API",
        "service": "langgraph-api",
        "kind": "Brain",
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
        "description": "Optional Hermes dashboard, enabled by Compose profile.",
        "host_url": "http://localhost:9119",
        "docker_url": "http://hermes-dashboard:9119",
        "port": 9119,
        "profile": "hermes-dashboard",
        "accent": "#bb7cff",
    },
    {
        "name": "LiteLLM",
        "service": "litellm",
        "kind": "Model Gateway",
        "description": "Central model router for local and external model backends.",
        "host_url": "http://localhost:4000",
        "docker_url": "http://litellm:4000",
        "port": 4000,
        "accent": "#4bd2ff",
    },
    {
        "name": "RAG API",
        "service": "rag_api",
        "kind": "Retrieval API",
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
        "description": "Serves registered media, galleries, and analysis assets.",
        "host_url": f"http://localhost:{MEDIA_PORT}",
        "docker_url": "http://media-gallery:8130",
        "port": MEDIA_PORT,
        "accent": "#ff8a58",
    },
    {
        "name": "Deep Agents UI",
        "service": "deep-agents-ui",
        "kind": "Inspection UI",
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
        "description": "Optional second frontend through the AlphaRavis bridge.",
        "host_url": f"http://localhost:{OPENWEBUI_PORT}",
        "docker_url": "http://openwebui:8080",
        "port": OPENWEBUI_PORT,
        "profile": "openwebui",
        "accent": "#f8df72",
    },
    {
        "name": "Pixelle MCP",
        "service": "pixelle",
        "kind": "Media Tool",
        "description": "Pixelle image/video tool service.",
        "host_url": "http://localhost:9004",
        "docker_url": "http://pixelle:9004",
        "port": 9004,
        "accent": "#ff65b3",
    },
    {
        "name": "LangGraph Research UI",
        "service": "langgraph-api",
        "kind": "LangGraphics UI",
        "description": "Research specialist visual port exposed by langgraph-api.",
        "host_url": "http://localhost:8760",
        "docker_url": "http://langgraph-api:8760",
        "port": 8760,
        "accent": "#7ad7ff",
    },
    {
        "name": "LangGraph General UI",
        "service": "langgraph-api",
        "kind": "LangGraphics UI",
        "description": "General specialist visual port exposed by langgraph-api.",
        "host_url": "http://localhost:8762",
        "docker_url": "http://langgraph-api:8762",
        "port": 8762,
        "accent": "#7ad7ff",
    },
    {
        "name": "LangGraph Computer UI",
        "service": "langgraph-api",
        "kind": "LangGraphics UI",
        "description": "Computer/CUA specialist visual port exposed by langgraph-api.",
        "host_url": "http://localhost:8764",
        "docker_url": "http://langgraph-api:8764",
        "port": 8764,
        "accent": "#7ad7ff",
    },
    {
        "name": "LangGraph Debugger UI",
        "service": "langgraph-api",
        "kind": "LangGraphics UI",
        "description": "Debugger specialist visual port exposed by langgraph-api.",
        "host_url": "http://localhost:8766",
        "docker_url": "http://langgraph-api:8766",
        "port": 8766,
        "accent": "#7ad7ff",
    },
    {
        "name": "LangGraph Supervisor UI",
        "service": "langgraph-api",
        "kind": "LangGraphics UI",
        "description": "Main supervisor visual port exposed by langgraph-api.",
        "host_url": "http://localhost:8768",
        "docker_url": "http://langgraph-api:8768",
        "port": 8768,
        "accent": "#7ad7ff",
    },
    {
        "name": "LangGraph VNC",
        "service": "langgraph-api",
        "kind": "Remote Desktop",
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
    use_tailscale = URL_MODE in {"auto", "tailscale", "https"}
    services: list[dict[str, Any]] = []
    for original in SERVICES:
        service = dict(original)
        local_url = str(service.get("host_url", ""))
        tailscale_url = ""
        if use_tailscale and not service.get("non_http"):
            tailscale_url = str(host_overrides.get(local_url) or service_overrides.get(str(service.get("service", ""))) or "")
        service["local_url"] = local_url
        if tailscale_url:
            service["host_url"] = tailscale_url
            service["tailscale_url"] = tailscale_url
            service["url_mode"] = "tailscale"
        else:
            service["url_mode"] = "local"
        services.append(service)
    return services


def render_index() -> bytes:
    services = effective_services()
    service_json = json.dumps(services, ensure_ascii=True)
    cards = "\n".join(render_card(service) for service in services)
    mode_label = "Tailscale HTTPS" if any(service.get("url_mode") == "tailscale" for service in services) else "Localhost"
    body = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlphaRavis Service Dashboard</title>
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
    @media (max-width: 780px) {{
      main {{ width: calc(100% - 32px); padding: 24px 0; }}
      h1 {{ font-size: clamp(26px, 10vw, 48px); }}
      .grid {{ grid-template-columns: 1fr; }}
      .card {{ min-height: auto; }}
      .description {{ -webkit-line-clamp: 4; }}
    }}
    @media (max-width: 480px) {{
      main {{ width: calc(100% - 24px); }}
      .toolbar {{ flex-direction: column; align-items: stretch; }}
      .mode {{ width: fit-content; }}
      .count {{ text-align: right; margin-top: -30px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">AlphaRavis Local Stack</div>
      <h1>Service Dashboard</h1>
      <p class="subhead">Alle wichtigen Docker-Compose-Services, oeffentliche URLs und interne Docker-Adressen an einem Ort. Kacheln mit Web- oder API-URL oeffnen direkt den jeweiligen Dienst.</p>
    </header>
    <section class="toolbar" aria-label="Dashboard filter">
      <input id="filter" class="search" type="search" placeholder="Service suchen" autocomplete="off">
      <span class="mode">{html.escape(mode_label)}</span>
      <div class="count"><span id="visible-count">{len(services)}</span> Services</div>
    </section>
    <section id="grid" class="grid">
      {cards}
    </section>
  </main>
  <script>
    window.ALPHARAVIS_SERVICES = {service_json};
    const input = document.getElementById("filter");
    const cards = Array.from(document.querySelectorAll("[data-card]"));
    const count = document.getElementById("visible-count");
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
  </script>
</body>
</html>
"""
    return body.encode("utf-8")


def render_card(service: dict[str, Any]) -> str:
    host_url = str(service["host_url"])
    local_url = str(service.get("local_url", host_url))
    disabled = bool(service.get("non_http"))
    tag = "div" if disabled else "a"
    href = "" if disabled else f' href="{html.escape(host_url, quote=True)}"'
    target = "" if disabled else ' target="_blank" rel="noreferrer"'
    search = " ".join(
        str(service.get(key, ""))
        for key in ("name", "service", "kind", "description", "host_url", "docker_url", "profile")
    ).lower()
    profile = service.get("profile")
    open_label = "TCP endpoint" if disabled else "Oeffnen"
    profile_html = f'<span class="profile">profile: {html.escape(str(profile))}</span>' if profile else "<span></span>"
    mode = str(service.get("url_mode", "local"))
    mode_html = f'<span class="url-mode">{html.escape(mode)}</span>'
    local_row = ""
    if host_url != local_url:
        local_row = f'<div class="row"><span class="label">Local</span><code>{html.escape(local_url)}</code></div>'
    return f"""
      <{tag} class="card" style="--accent: {html.escape(str(service["accent"]), quote=True)};" data-card data-search="{html.escape(search, quote=True)}"{href}{target} aria-disabled="{str(disabled).lower()}">
        <div class="topline">
          <h2 class="name">{html.escape(str(service["name"]))}</h2>
          <span class="kind">{html.escape(str(service["kind"]))}</span>
        </div>
        <p class="description">{html.escape(str(service["description"]))}</p>
        <div class="meta">
          <div class="row"><span class="label">URL</span><code>{html.escape(host_url)}</code></div>
          {local_row}
          <div class="row"><span class="label">Docker</span><code>{html.escape(str(service["docker_url"]))}</code></div>
          <div class="row"><span class="label">Port</span><code>{html.escape(str(service["port"]))}</code></div>
        </div>
        <div class="actions">
          <span class="open">{open_label}</span>
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
        if parsed.path == "/services.json":
            self.send_json({"services": effective_services(), "url_mode": URL_MODE})
            return
        if parsed.path == "/health":
            self.send_json({"ok": True, "service": "service-dashboard"})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

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
