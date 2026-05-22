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


def test_configure_ubuntu_llama_instance_supports_bounded_parallel_slots(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL", "http://manager.local:8099")
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_PARALLEL_MAX", "2")

    result = asyncio.run(
        model_management.configure_ubuntu_llama_instance(
            "primary",
            parallel_slots=2,
        )
    )

    assert result["ok"] is False
    assert result["reason"] == "actions_disabled"
    assert result["payload"]["parallel"] == 2
    assert "parallel=2" in result["payload"]["parallel_vram_note"]

    rejected = asyncio.run(model_management.configure_ubuntu_llama_instance("primary", parallel_slots=3))

    assert rejected["ok"] is False
    assert "parallel_slots must be between 1 and 2" in rejected["error"]


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


def test_prepare_comfy_marks_woke_for_request(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_ENABLE_MODEL_MANAGEMENT", "true")
    monkeypatch.setenv("ALPHARAVIS_ENABLE_POWER_MANAGEMENT", "true")
    monkeypatch.setenv("ALPHARAVIS_PIXELLE_PREPARE_COMFY", "true")
    monkeypatch.setenv("ALPHARAVIS_COMFY_WAKE_WAIT_SECONDS", "1")
    calls = {"probe": 0}

    async def fake_probe(url, *, timeout_seconds):
        calls["probe"] += 1
        return {"ok": calls["probe"] > 1, "url": url}

    async def fake_request_power_action(action, target, reason, *, remote_pcs=None):
        return {"ok": True, "action": action, "target": target, "reason": reason}

    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(model_management, "probe_http", fake_probe)
    monkeypatch.setattr(model_management, "request_power_action", fake_request_power_action)
    monkeypatch.setattr(model_management.asyncio, "sleep", fake_sleep)

    result = asyncio.run(
        model_management.prepare_comfy_for_pixelle(
            {"comfy_server": {"ip": "comfy.local", "mac": "AA:BB:CC:DD:EE:FF"}}
        )
    )

    assert result["ready"] is True
    assert result["woke_for_request"] is True
    assert result["wake_result"]["action"] == "wake_pc"


def test_check_ollama_models_reads_real_runtime_models(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = '{"models": []}'

        def json(self):
            return {"models": [{"name": "qwen3-embedding:0.6b"}, {"model": "edge-gemma"}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            return FakeResponse()

    monkeypatch.setenv("ALPHARAVIS_OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("ALPHARAVIS_OLLAMA_CHAT_MODEL", "edge-gemma")
    monkeypatch.setattr(model_management.httpx, "AsyncClient", FakeClient)

    result = asyncio.run(model_management.check_ollama_models())

    assert result["ok"] is True
    assert result["embedding_model_loaded"] is True
    assert result["chat_model_loaded"] is True


def test_load_and_unload_ollama_models_use_keep_alive(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200
        text = '{"done": true}'

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None):
            calls.append({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setenv("ALPHARAVIS_OLLAMA_EMBED_MODEL", "embedder")
    monkeypatch.setattr(model_management.httpx, "AsyncClient", FakeClient)

    loaded = asyncio.run(model_management.load_embedding_model(keep_alive="10m"))
    unloaded = asyncio.run(model_management.unload_ollama_model("embedder"))

    assert loaded["ok"] is True
    assert unloaded["ok"] is True
    assert calls[0]["json"]["model"] == "embedder"
    assert calls[0]["json"]["keep_alive"] == "10m"
    assert calls[1]["json"]["keep_alive"] == "0"


def test_request_power_action_dispatches_embedding_jobs_locally(monkeypatch):
    async def fake_run_embedding_jobs(limit):
        return {"ok": True, "limit": limit, "processed": 2}

    monkeypatch.setattr(model_management, "_vector_run_embedding_jobs", fake_run_embedding_jobs)

    result = asyncio.run(model_management.request_power_action("run_embedding_jobs", "7", "operator"))

    assert result == {"ok": True, "limit": 7, "processed": 2}


def test_model_context_policy_routes_large_context_to_primary(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_UBUNTU_LLAMA_CONTEXT_MAX", "262144")
    monkeypatch.setenv("ALPHARAVIS_PRIMARY_CONTEXT_HIGH", "200000")

    plan = model_management.model_context_policy_plan(reason="context_overflow")

    assert plan["ok"] is True
    assert plan["action"] == "raise_context"
    assert plan["instance_id"] == "primary"
    assert plan["target_context_size"] == 200000
    assert plan["rollback_action"]["context_size"] == 131072


def test_model_context_policy_rolls_secondary_back_to_normal(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_SECONDARY_CONTEXT_NORMAL", "8192")

    plan = model_management.model_context_policy_plan(current_instance="secondary", rollback=True)

    assert plan["action"] == "rollback_context"
    assert plan["instance_id"] == "secondary"
    assert plan["target_context_size"] == 8192


def test_apply_model_context_policy_uses_configure_action(monkeypatch):
    calls = []

    async def fake_configure(instance_id, **kwargs):
        calls.append({"instance_id": instance_id, **kwargs})
        return {"ok": True, "payload": {"context_size": kwargs["context_size"]}}

    monkeypatch.setattr(model_management, "configure_ubuntu_llama_instance", fake_configure)

    result = asyncio.run(model_management.apply_model_context_policy(requested_context_size=16384))

    assert result["ok"] is True
    assert calls[0]["instance_id"] == "secondary"
    assert calls[0]["context_size"] == 16384
    assert result["rollback_action"]["context_size"] == 8192
