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
    assert "Settings" in html
    assert "/settings" in html
    assert "http://localhost:9004/pixelle/mcp" in html
    assert "TCP endpoint" in html


def test_settings_page_exposes_positive_and_negative_fallback_filters() -> None:
    html = dashboard.render_settings().decode("utf-8")

    assert "Nur Fallback" in html
    assert "Nur Legacy" in html
    assert "Fallback + Legacy" in html
    assert "Fallback ausblenden" in html
    assert "Legacy ausblenden" in html
    assert "Nach .env Reihenfolge" in html
    assert "envOrder" in html


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


def test_settings_model_reads_template_runtime_and_current_env(monkeypatch, tmp_path: Path) -> None:
    example = tmp_path / ".env(exaple)"
    env = tmp_path / ".env"
    runtime = tmp_path / "runtime_settings.json"
    example.write_text(
        "\n".join(
            [
                "# =====",
                "# MODEL MANAGEMENT",
                "# Allowed values: true, false",
                "ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=false",
                "# Main model endpoint.",
                "OPENAI_API_BASE=http://litellm:4000/v1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env.write_text("ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true\n", encoding="utf-8")
    runtime.write_text(
        json.dumps({"updated_at": 1, "values": {"OPENAI_API_BASE": "http://runtime:4000/v1"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "EXAMPLE_PATH", example)
    monkeypatch.setattr(dashboard, "ENV_PATH", env)
    monkeypatch.setattr(dashboard, "RUNTIME_SETTINGS_PATH", runtime)

    model = dashboard.settings_model()
    entries = {entry["key"]: entry for section in model["sections"] for entry in section["entries"]}

    assert entries["ALPHARAVIS_ENABLE_MODEL_MANAGEMENT"]["kind"] == "bool"
    assert entries["ALPHARAVIS_ENABLE_MODEL_MANAGEMENT"]["value"] == "true"
    assert entries["OPENAI_API_BASE"]["value"] == "http://runtime:4000/v1"
    assert entries["OPENAI_API_BASE"]["hasRuntime"] is True
    assert entries["OPENAI_API_BASE"]["category"] == "model"
    assert "provider" in entries["OPENAI_API_BASE"]["tags"]


def test_settings_model_infers_dropdowns_and_fallback_descriptions(monkeypatch, tmp_path: Path) -> None:
    example = tmp_path / ".env(exaple)"
    env = tmp_path / ".env"
    runtime = tmp_path / "runtime_settings.json"
    example.write_text(
        "\n".join(
            [
                "# Provider hardening.",
                "# ALPHARAVIS_RESPONSES_OMIT_TEMPERATURE_MODE:",
                "#   auto   = omit temperature for managed providers",
                "#   always = never send temperature",
                "#   never  = send temperature when requested",
                "ALPHARAVIS_RESPONSES_OMIT_TEMPERATURE_MODE=auto",
                "ALPHARAVIS_RESPONSES_API_BASE=http://litellm:4000/v1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env.write_text("", encoding="utf-8")
    monkeypatch.setattr(dashboard, "EXAMPLE_PATH", example)
    monkeypatch.setattr(dashboard, "ENV_PATH", env)
    monkeypatch.setattr(dashboard, "RUNTIME_SETTINGS_PATH", runtime)

    model = dashboard.settings_model()
    entries = {entry["key"]: entry for section in model["sections"] for entry in section["entries"]}

    omit = entries["ALPHARAVIS_RESPONSES_OMIT_TEMPERATURE_MODE"]
    assert omit["kind"] == "select"
    assert omit["allowedValues"] == ["auto", "always", "never"]
    assert "temperature" in omit["description"].lower()
    assert entries["ALPHARAVIS_RESPONSES_API_BASE"]["description"].startswith("Endpoint-URL")


def test_settings_model_tags_new_runtime_and_model_manager_keys(monkeypatch, tmp_path: Path) -> None:
    example = tmp_path / ".env(exaple)"
    env = tmp_path / ".env"
    runtime = tmp_path / "runtime_settings.json"
    example.write_text(
        "\n".join(
            [
                "ALPHARAVIS_RUN_STATE_AUTO_RESUME=false",
                "ALPHARAVIS_RUN_STATE_DB=alpharavis_state",
                "ALPHARAVIS_RUNTIME_SETTINGS_FILE=/workspace/service-dashboard-data/runtime_settings.json",
                "ALPHARAVIS_ASYNC_REVIEWER_ENABLED=false",
                    "ALPHARAVIS_ASYNC_REVIEW_STORE_PATH=/tmp/alpharavis_run_reviews.json",
                    "ALPHARAVIS_BACKGROUND_TASKS_ENABLED=true",
                    "ALPHARAVIS_BACKGROUND_CONTEXT_MAX_UTILIZATION=0.70",
                    "ALPHARAVIS_SERVER_MODEL_MANAGER_TEMPERATURE=0",
                    "ALPHARAVIS_UBUNTU_LLAMA_MANAGER_API_KEY=",
                    "ALPHARAVIS_UBUNTU_LLAMA_CONTEXT_MAX=262144",
                    "ALPHARAVIS_PIXELLE_AUTO_SHUTDOWN_COMFY_AFTER_JOB=false",
                    "ALPHARAVIS_BIG_LLM_AUTO_SHUTDOWN_AFTER_MANAGED_RUN=false",
                ]
            )
        + "\n",
        encoding="utf-8",
    )
    env.write_text("", encoding="utf-8")
    monkeypatch.setattr(dashboard, "EXAMPLE_PATH", example)
    monkeypatch.setattr(dashboard, "ENV_PATH", env)
    monkeypatch.setattr(dashboard, "RUNTIME_SETTINGS_PATH", runtime)

    model = dashboard.settings_model()
    entries = {entry["key"]: entry for section in model["sections"] for entry in section["entries"]}

    run_auto = entries["ALPHARAVIS_RUN_STATE_AUTO_RESUME"]
    assert run_auto["category"] == "runtime"
    assert "run-state" in run_auto["tags"]
    assert "automatisch fort" in run_auto["description"]

    assert entries["ALPHARAVIS_RUN_STATE_DB"]["category"] == "runtime"
    assert "Mongo-Datenbank" in entries["ALPHARAVIS_RUN_STATE_DB"]["description"]
    assert "storage" in entries["ALPHARAVIS_RUNTIME_SETTINGS_FILE"]["tags"]
    assert entries["ALPHARAVIS_ASYNC_REVIEWER_ENABLED"]["category"] == "features"
    assert "reviewer" in entries["ALPHARAVIS_ASYNC_REVIEWER_ENABLED"]["tags"]
    assert "korrigiert nichts automatisch" in entries["ALPHARAVIS_ASYNC_REVIEWER_ENABLED"]["description"]
    assert entries["ALPHARAVIS_ASYNC_REVIEW_STORE_PATH"]["envOrder"] > entries["ALPHARAVIS_ASYNC_REVIEWER_ENABLED"]["envOrder"]
    assert entries["ALPHARAVIS_BACKGROUND_TASKS_ENABLED"]["category"] == "features"
    assert "background-tasks" in entries["ALPHARAVIS_BACKGROUND_TASKS_ENABLED"]["tags"]
    assert entries["ALPHARAVIS_BACKGROUND_TASKS_ENABLED"]["importance"] == 90
    assert "Context-Leases" in entries["ALPHARAVIS_BACKGROUND_TASKS_ENABLED"]["description"]
    assert entries["ALPHARAVIS_BACKGROUND_CONTEXT_MAX_UTILIZATION"]["category"] == "features"
    assert "Main-Agent" in entries["ALPHARAVIS_BACKGROUND_CONTEXT_MAX_UTILIZATION"]["description"]
    assert entries["ALPHARAVIS_UBUNTU_LLAMA_MANAGER_API_KEY"]["category"] == "model"
    assert entries["ALPHARAVIS_UBUNTU_LLAMA_MANAGER_API_KEY"]["secret"] is True
    assert entries["ALPHARAVIS_PIXELLE_AUTO_SHUTDOWN_COMFY_AFTER_JOB"]["importance"] == 90
    assert "ComfyUI-Shutdown" in entries["ALPHARAVIS_PIXELLE_AUTO_SHUTDOWN_COMFY_AFTER_JOB"]["description"]
    assert entries["ALPHARAVIS_BIG_LLM_AUTO_SHUTDOWN_AFTER_MANAGED_RUN"]["importance"] == 90
    assert "BigBoss" in entries["ALPHARAVIS_BIG_LLM_AUTO_SHUTDOWN_AFTER_MANAGED_RUN"]["description"]
    assert "Sampling-Temperatur" in entries["ALPHARAVIS_SERVER_MODEL_MANAGER_TEMPERATURE"]["description"]
    assert "ubuntu-llama" in entries["ALPHARAVIS_UBUNTU_LLAMA_CONTEXT_MAX"]["tags"]
    assert "Obergrenze" in entries["ALPHARAVIS_UBUNTU_LLAMA_CONTEXT_MAX"]["description"]


def test_settings_model_marks_fallbacks_and_limits_are_not_secret(monkeypatch, tmp_path: Path) -> None:
    example = tmp_path / ".env(exaple)"
    env = tmp_path / ".env"
    runtime = tmp_path / "runtime_settings.json"
    example.write_text(
        "\n".join(
            [
                "# Fixed fallback token limits used when percent context limits are disabled.",
                "# With ALPHARAVIS_ENABLE_PERCENT_CONTEXT_LIMITS=true these remain only fallback",
                "# documentation for older deployments.",
                "ALPHARAVIS_ACTIVE_TOKEN_LIMIT=30000",
                "ALPHARAVIS_RESPONSES_API_KEY=sk-local-dev",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env.write_text("", encoding="utf-8")
    monkeypatch.setattr(dashboard, "EXAMPLE_PATH", example)
    monkeypatch.setattr(dashboard, "ENV_PATH", env)
    monkeypatch.setattr(dashboard, "RUNTIME_SETTINGS_PATH", runtime)

    model = dashboard.settings_model()
    entries = {entry["key"]: entry for section in model["sections"] for entry in section["entries"]}

    active_limit = entries["ALPHARAVIS_ACTIVE_TOKEN_LIMIT"]
    assert active_limit["kind"] == "number"
    assert active_limit["secret"] is False
    assert active_limit["category"] != "security"
    assert "security" not in active_limit["tags"]
    assert active_limit["fallback"] is True
    assert active_limit["deprecated"] is True
    assert "fallback" in active_limit["tags"]
    assert "Prozent-Kontextlimits" in active_limit["fallbackFor"]
    assert entries["ALPHARAVIS_RESPONSES_API_KEY"]["secret"] is True


def test_runtime_and_permanent_settings_write_only_template_keys(monkeypatch, tmp_path: Path) -> None:
    example = tmp_path / ".env(exaple)"
    env = tmp_path / ".env"
    runtime = tmp_path / "runtime_settings.json"
    example.write_text("KNOWN=old\n", encoding="utf-8")
    env.write_text("KNOWN=old\n", encoding="utf-8")
    monkeypatch.setattr(dashboard, "EXAMPLE_PATH", example)
    monkeypatch.setattr(dashboard, "ENV_PATH", env)
    monkeypatch.setattr(dashboard, "RUNTIME_SETTINGS_PATH", runtime)

    cleaned = dashboard._clean_settings_values({"KNOWN": "new", "UNKNOWN": "nope"})

    assert cleaned == {"KNOWN": "new"}
    assert dashboard.apply_runtime_settings(cleaned) == 1
    assert json.loads(runtime.read_text(encoding="utf-8"))["values"] == {"KNOWN": "new"}
    assert dashboard.save_permanent_settings(cleaned) == 1
    assert dashboard.read_env(env)["KNOWN"] == "new"
