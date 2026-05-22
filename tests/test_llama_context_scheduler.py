from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_stack.context_budget.leases import LeaseStore
from ai_stack.context_budget.policies import ensure_kv_unified_in_command, parse_runtime_config_from_command
from ai_stack.context_budget.scheduler import ContextScheduler
from ai_stack.llama_runtime.client import LlamaCppRuntimeClient
from ai_stack.ubuntu_llama_manager.client import UbuntuLlamaManagerClient
from ai_stack.ubuntu_llama_manager.schemas import UbuntuLlamaInstance


def test_parse_runtime_config_from_llama_command():
    config = parse_runtime_config_from_command(
        "./llama-server -hf model --ctx-size 262144 --parallel 2 --kv-unified"
    )

    assert config.ctx_total == 262144
    assert config.parallel == 2
    assert config.kv_unified is True
    assert config.conservative_ctx_per_slot == 131072


def test_ensure_kv_unified_in_command_is_idempotent():
    command = "./llama-server -c 8192 --no-kv-unified"

    patched = ensure_kv_unified_in_command(command)

    assert "--kv-unified" in patched
    assert "--no-kv-unified" not in patched
    assert ensure_kv_unified_in_command(patched) == patched


def test_instance_derives_runtime_base_url_from_manager_host_for_localhost():
    instance = UbuntuLlamaInstance.from_api(
        {
            "id": "primary",
            "host": "127.0.0.1",
            "port": 8033,
            "command": "./llama-server -c 4096 -np 2",
        },
        manager_base_url="http://llama-box:8099",
    )

    assert instance.base_url == "http://llama-box:8033"
    assert instance.ctx_total == 4096
    assert instance.parallel == 2
    assert instance.kv_unified is False


def test_manager_client_maps_stop_restart_to_documented_endpoints():
    assert UbuntuLlamaManagerClient._control_path("primary", "restart") == "/llama/restart"
    assert UbuntuLlamaManagerClient._control_path("secondary", "stop") == "/llama-secondary/stop"


def test_runtime_chat_count_uses_apply_template_then_tokenize(monkeypatch):
    calls = []

    async def fake_request(self, method, path, *, json_payload=None):
        calls.append((method, path, json_payload))
        if path == "/apply-template":
            return {"prompt": "<s>hello</s>"}
        if path == "/tokenize":
            return {"tokens": [1, 2, 3]}
        raise AssertionError(path)

    monkeypatch.setattr(LlamaCppRuntimeClient, "_request", fake_request)

    count = asyncio.run(LlamaCppRuntimeClient("http://llama:8033").count_tokens_chat([{"role": "user", "content": "hello"}]))

    assert count == 3
    assert calls[0][1] == "/apply-template"
    assert calls[1][1] == "/tokenize"
    assert calls[1][2] == {"content": "<s>hello</s>"}


def test_scheduler_rejects_when_active_leases_exceed_capacity(monkeypatch):
    scheduler = ContextScheduler(lease_store=LeaseStore(), safety_factor=0.9)
    instance = UbuntuLlamaInstance.from_api(
        {
            "id": "primary",
            "host": "llama-box",
            "port": 8033,
            "command": "./llama-server -c 1000 --parallel 1 --kv-unified",
        }
    )
    scheduler.instances = {"primary": instance}

    async def fake_count(self, messages):
        return 400

    monkeypatch.setattr(LlamaCppRuntimeClient, "count_tokens_chat", fake_count)

    first, first_admission = asyncio.run(
        scheduler.estimate_and_reserve(
            messages=[{"role": "user", "content": "one"}],
            max_output_tokens=300,
            safety_margin=50,
            preferred_instance_id="primary",
        )
    )
    second, second_admission = asyncio.run(
        scheduler.estimate_and_reserve(
            messages=[{"role": "user", "content": "two"}],
            max_output_tokens=300,
            safety_margin=50,
            preferred_instance_id="primary",
        )
    )

    assert first is not None
    assert first_admission["ok"] is True
    assert second is None
    assert second_admission["reason"] == "insufficient_context"


def test_scheduler_without_kv_unified_uses_conservative_slot_capacity(monkeypatch):
    scheduler = ContextScheduler(lease_store=LeaseStore(), safety_factor=1.0)
    scheduler.instances = {
        "primary": UbuntuLlamaInstance.from_api(
            {
                "id": "primary",
                "host": "llama-box",
                "port": 8033,
                "command": "./llama-server -c 1000 --parallel 2",
            }
        )
    }

    async def fake_count(self, messages):
        return 400

    monkeypatch.setattr(LlamaCppRuntimeClient, "count_tokens_chat", fake_count)

    lease, admission = asyncio.run(
        scheduler.estimate_and_reserve(
            messages=[{"role": "user", "content": "one"}],
            max_output_tokens=80,
            safety_margin=30,
            preferred_instance_id="primary",
        )
    )

    assert lease is None
    assert admission["reason"] == "insufficient_context"
    assert admission["capacity_tokens"] == 500
