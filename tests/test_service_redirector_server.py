from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import service_redirector_server as dashboard  # noqa: E402


def test_dashboard_separates_web_apis_and_infrastructure() -> None:
    html = dashboard.render_index().decode("utf-8")

    assert "Web Interfaces" in html
    assert "<h2>APIs</h2>" in html
    assert "Infrastructure" in html
    assert "data-copy-url" in html
    assert "data-no-card-open" in html
    assert "suppressCardOpen" in html
    assert "Local" in html
    assert "Tailnet" in html
    assert "/favicon.svg" in html
    assert ">MG<" in html
    assert "Öffnen" in html
    assert "data-open-url=" in html
    assert "Pixelle" in html
    assert "Pixelle MCP" in html
    assert "LiteLLM API" in html
    assert "http://localhost:9004/pixelle/mcp" in html
    assert "TCP endpoint" in html


def test_effective_services_expose_https_local_and_tailnet_http_for_same_port_paths(monkeypatch, tmp_path: Path) -> None:
    payload_path = tmp_path / "tailscale_service_urls.json"
    payload_path.write_text(
        json.dumps(
            {
                "tailscale_host": "node.tailnet.ts.net",
                "routes": [{"service": "pixelle", "port": 9004, "tailscale_url": "https://node.tailnet.ts.net:9004"}],
                "host_url_overrides": {
                    f"http://localhost:{dashboard.MEDIA_PORT}/gallery": "https://node.tailnet.ts.net:8130/gallery"
                },
                "redirector_overrides": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "TAILSCALE_URLS_PATH", payload_path)
    monkeypatch.setattr(dashboard, "URL_MODE", "auto")

    media = next(service for service in dashboard.effective_services() if service["service"] == "media-gallery")

    assert media["host_url"] == "https://node.tailnet.ts.net:8130/gallery"
    assert media["local_url"] == f"http://localhost:{dashboard.MEDIA_PORT}/gallery"
    assert media["https_url"] == "https://node.tailnet.ts.net:8130/gallery"
    assert media["tailnet_http_url"] == f"http://node.tailnet.ts.net:{dashboard.MEDIA_PORT}/gallery"
    assert media["url_mode"] == "tailscale"

    pixelle_mcp = next(service for service in dashboard.effective_services() if service["service"] == "pixelle-mcp")
    assert pixelle_mcp["host_url"] == "https://node.tailnet.ts.net:9004/pixelle/mcp"
    assert pixelle_mcp["local_url"] == "http://localhost:9004/pixelle/mcp"
    assert pixelle_mcp["tailnet_http_url"] == "http://node.tailnet.ts.net:9004/pixelle/mcp"
