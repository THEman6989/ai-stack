import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import model_management  # noqa: E402


def test_configure_ubuntu_llama_instance_is_dry_run_when_actions_disabled(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL", "http://manager.local:8099")
    monkeypatch.setenv("ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS", "false")

    result = asyncio.run(
        model_management.configure_ubuntu_llama_instance(
            "secondary",
            context_size=16384,
            restart=True,
        )
    )

    assert result["ok"] is False
    assert result["dry_run"] is True
    assert result["reason"] == "actions_disabled"
    assert result["url"] == "http://manager.local:8099/llama/instances/secondary/config"
    assert result["payload"] == {"restart": True, "context_size": 16384}


def test_ubuntu_manager_url_can_be_derived_from_ip_and_port(monkeypatch):
    monkeypatch.delenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL", raising=False)
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_IP", "192.168.178.113")
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_PORT", "8099")
    monkeypatch.setenv("ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS", "false")

    result = asyncio.run(model_management.control_ubuntu_llama_service("primary", "restart"))

    assert result["dry_run"] is True
    assert result["url"] == "http://192.168.178.113:8099/llama/restart"


def test_direct_esp_url_can_be_derived_from_ip(monkeypatch):
    monkeypatch.delenv("ALPHARAVIS_UBUNTU_LLAMA_ESP_URL", raising=False)
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_ESP_IP", "192.168.178.114")
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_ESP_PORT", "80")
    monkeypatch.setenv("ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS", "false")

    result = asyncio.run(
        model_management.request_ubuntu_server_power_action(
            "power-on",
            direct_esp=True,
            hold_seconds=1,
        )
    )

    assert result["dry_run"] is True
    assert result["url"] == "http://192.168.178.114/action"


def test_configure_ubuntu_llama_instance_rejects_out_of_range_context(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL", "http://manager.local:8099")
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_CONTEXT_MAX", "32768")

    result = asyncio.run(
        model_management.configure_ubuntu_llama_instance(
            "primary",
            context_size=200000,
        )
    )

    assert result["ok"] is False
    assert "context_size must be between" in result["error"]


def test_recover_ubuntu_llama_no_response_uses_diagnose_endpoint_without_action_gate(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 202
        text = '{"ok": true}'

        def json(self):
            return {"ok": True, "decision": "llama-hung"}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def request(self, method, url, headers=None, json=None):
            calls.append({"method": method, "url": url, "headers": headers, "json": json})
            return FakeResponse()

    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL", "http://manager.local:8099/")
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_API_KEY", "secret-token")
    monkeypatch.setenv("ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS", "false")
    monkeypatch.setattr(model_management.httpx, "AsyncClient", FakeClient)

    result = asyncio.run(
        model_management.recover_ubuntu_llama_no_response(
            reason="timeout",
            diagnose_only=True,
            probe_timeout_seconds=20,
        )
    )

    assert result["ok"] is True
    assert calls == [
        {
            "method": "POST",
            "url": "http://manager.local:8099/ai-stack/diagnose-llama",
            "headers": {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer secret-token",
            },
            "json": {"reason": "timeout", "probe_timeout_seconds": 20},
        }
    ]


def test_control_ubuntu_llama_service_maps_secondary_restart_and_is_gated(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL", "http://manager.local:8099")
    monkeypatch.setenv("ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS", "false")

    result = asyncio.run(model_management.control_ubuntu_llama_service("secondary", "restart"))

    assert result["ok"] is False
    assert result["dry_run"] is True
    assert result["url"] == "http://manager.local:8099/llama-secondary/restart"


def test_request_ubuntu_server_power_action_maps_manager_esp_payload_and_is_gated(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL", "http://manager.local:8099")
    monkeypatch.setenv("ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS", "false")

    result = asyncio.run(
        model_management.request_ubuntu_server_power_action(
            "power-cycle",
            reason="gpu-reset",
            confirmed=True,
            hold_seconds=8,
            wait_seconds=20,
        )
    )

    assert result["ok"] is False
    assert result["dry_run"] is True
    assert result["url"] == "http://manager.local:8099/esp/action"
    assert result["payload"] == {
        "action": "power-cycle",
        "reason": "gpu-reset",
        "requested_by": "alpharavis",
        "hold_seconds": 8,
        "wait_seconds": 20,
    }


def test_request_ubuntu_server_power_action_can_use_direct_esp(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_ESP_URL", "http://esp.local")
    monkeypatch.setenv("ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS", "false")

    result = asyncio.run(
        model_management.request_ubuntu_server_power_action(
            "power-on",
            reason="host-offline",
            direct_esp=True,
            hold_seconds=1,
        )
    )

    assert result["ok"] is False
    assert result["dry_run"] is True
    assert result["url"] == "http://esp.local/action"
    assert result["payload"]["action"] == "power-on"


def test_request_ubuntu_server_power_action_requires_confirmation_for_power_off(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL", "http://manager.local:8099")
    monkeypatch.setenv("ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS", "true")

    result = asyncio.run(
        model_management.request_ubuntu_server_power_action(
            "power-off",
            reason="operator-request",
        )
    )

    assert result["ok"] is False
    assert result["needs_confirmation"] is True
    assert result["action"] == "power-off"
