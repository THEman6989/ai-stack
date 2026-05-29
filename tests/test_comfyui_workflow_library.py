from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import comfyui_workflow_library as library  # noqa: E402


class FakeState:
    def __init__(self):
        self.records: dict[str, dict] = {}

    def save_workflow_record(self, *, namespace: str, workflow_id: str, record: dict):
        stored = {**record, "namespace": namespace, "workflow_id": workflow_id, "_id": f"{namespace}:{workflow_id}"}
        self.records[f"{namespace}:{workflow_id}"] = stored
        return {"saved": True, "record": stored}

    def load_workflow_record(self, namespace: str, workflow_id: str):
        return self.records.get(f"{namespace}:{workflow_id}")

    def list_workflow_records(self, *, namespace: str, status: str = "", file: str = "", limit: int = 50):
        return [record for record in self.records.values() if record.get("namespace") == namespace][:limit]


def _workflow():
    return {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "old prompt"}},
        "2": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20}},
    }


def test_save_workflow_record_validates_and_persists_metadata(monkeypatch):
    fake = FakeState()
    monkeypatch.setenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_LIBRARY", "true")
    monkeypatch.setattr(library, "run_state_manager", fake)

    result = library.save_comfyui_workflow_record(
        workflow_name="wan_animate",
        workflow=_workflow(),
        description="Wan animation workflow",
        aliases=["wan animate", "wananimate"],
        parameter_map={"prompt": "1.inputs.text", "seed": "2.inputs.seed"},
        workflow_type="video",
        overwrite=False,
    )

    assert result["saved"] is True
    record = result["record"]
    assert record["workflow_id"] == "wan_animate"
    assert record["aliases"] == ["wan animate", "wananimate"]
    assert record["parameter_map"] == {"prompt": "1.inputs.text", "seed": "2.inputs.seed"}
    assert record["node_classes"] == ["CLIPTextEncode", "KSampler"]
    assert record["workflow"]["1"]["inputs"]["text"] == "old prompt"


def test_save_workflow_record_is_blocked_when_library_flag_off(monkeypatch):
    fake = FakeState()
    monkeypatch.delenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_LIBRARY", raising=False)
    monkeypatch.setattr(library, "run_state_manager", fake)

    result = library.save_comfyui_workflow_record(workflow_name="wan_animate", workflow=_workflow())

    assert result["blocked"] is True
    assert fake.records == {}


def test_resolve_saved_workflow_finds_alias(monkeypatch):
    fake = FakeState()
    monkeypatch.setenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_LIBRARY", "true")
    monkeypatch.setattr(library, "run_state_manager", fake)
    library.save_comfyui_workflow_record(workflow_name="wan_animate", workflow=_workflow(), aliases=["wan animate"])

    result = library.get_comfyui_workflow_record("wan animate")

    assert result["found"] is True
    assert result["record"]["workflow_id"] == "wan_animate"


def test_apply_workflow_parameters_uses_parameter_map_and_unique_inputs():
    patched, report = library.apply_workflow_parameters(
        _workflow(),
        {"prompt": "new prompt", "steps": 32, "missing": "x"},
        parameter_map={"prompt": "1.inputs.text"},
    )

    assert patched["1"]["inputs"]["text"] == "new prompt"
    assert patched["2"]["inputs"]["steps"] == 32
    assert report["applied"] == {"prompt": "1.inputs.text", "steps": "2.inputs.steps"}
    assert report["unresolved"] == ["missing"]


def test_submit_saved_workflow_loads_applies_and_calls_client(monkeypatch):
    fake = FakeState()
    monkeypatch.setenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_LIBRARY", "true")
    monkeypatch.setattr(library, "run_state_manager", fake)
    library.save_comfyui_workflow_record(
        workflow_name="wan_animate",
        workflow=_workflow(),
        parameter_map={"prompt": "1.inputs.text"},
    )

    calls = []

    class FakeClient:
        async def submit_workflow(self, workflow, *, client_id="alpharavis"):
            calls.append((workflow, client_id))
            return {"ok": True, "prompt_id": "abc123"}

    result = asyncio.run(
        library.submit_saved_comfyui_workflow_record(
            "wan_animate",
            {"prompt": "new prompt"},
            client=FakeClient(),
            client_id="test-client",
        )
    )

    assert result["ok"] is True
    assert result["workflow_name"] == "wan_animate"
    assert result["submit_result"] == {"ok": True, "prompt_id": "abc123"}
    assert calls[0][0]["1"]["inputs"]["text"] == "new prompt"
    assert calls[0][1] == "test-client"
