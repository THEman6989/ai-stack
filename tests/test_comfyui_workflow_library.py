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
        "2": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 7.5, "denoise": 0.8}},
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


# ---- Structured parameter schema tests ----


def test_infer_workflow_parameters_detects_types_and_skips_connections(monkeypatch):
    """Structured params: auto-detect types, descriptions, skip node references."""
    monkeypatch.setenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_LIBRARY", "true")
    workflow = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["2", 0]}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen.safetensors", "device": "default"}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20, "cfg": 7.5, "denoise": 1.0,
               "model": ["4", 0], "positive": ["1", 0], "negative": ["1", 0], "latent_image": ["5", 0],
               "sampler_name": "euler", "scheduler": "normal"}},
        "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image.safetensors", "weight_dtype": "default"}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 768, "batch_size": 1}},
    }

    params = library.infer_workflow_parameters(workflow)

    names = {p["name"] for p in params}
    # Node references (model, positive, negative, latent_image, clip) should be skipped
    assert "clip" not in names
    assert "model" not in names
    assert "positive" not in names
    assert "negative" not in names
    assert "latent_image" not in names

    # User-facing params should be detected
    assert "text" in names
    assert "seed" in names
    assert "steps" in names
    assert "cfg" in names
    assert "denoise" in names
    assert "width" in names
    assert "height" in names
    assert "batch_size" in names

    # Type inference
    by_name = {p["name"]: p for p in params}
    assert by_name["text"]["type"] == "str"
    assert by_name["seed"]["type"] == "int"
    assert by_name["steps"]["type"] == "int"
    assert by_name["cfg"]["type"] == "float"
    assert by_name["denoise"]["type"] == "float"
    assert by_name["width"]["type"] == "int"
    assert by_name["height"]["type"] == "int"
    assert by_name["batch_size"]["type"] == "int"

    # Descriptions
    assert by_name["text"]["description"] == "Text prompt for generation"
    assert by_name["seed"]["description"] == "Random seed for reproducibility"
    assert by_name["steps"]["description"] == "Number of sampling steps"

    # field_path back-fill
    assert by_name["text"]["field_path"] != ""
    assert by_name["seed"]["field_path"] != ""


def test_infer_workflow_outputs_detects_known_output_nodes(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_LIBRARY", "true")
    workflow = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "test"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
        "3": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0]}},
        "4": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["4", 0]}},
    }

    outputs = library.infer_workflow_outputs(workflow)

    assert len(outputs) == 2
    types = {o["node_id"]: o["output_type"] for o in outputs}
    assert types.get("2") == "images"
    assert types.get("4") == "videos"


def test_validate_parameter_schema_rejects_invalid(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_LIBRARY", "true")

    ok, _ = library.validate_parameter_schema([
        {"name": "prompt", "type": "str", "required": True},
        {"name": "seed", "type": "int", "default": 42},
    ])
    assert ok is True

    ok, err = library.validate_parameter_schema([
        {"name": "prompt", "type": "str"},
        {"name": "prompt", "type": "int"},  # duplicate
    ])
    assert ok is False
    assert "duplicate" in err.lower()

    ok, err = library.validate_parameter_schema([
        {"type": "str"},  # missing name
    ])
    assert ok is False
    assert "name" in err.lower()

    ok, err = library.validate_parameter_schema("not a list")  # type: ignore
    assert ok is False


def test_save_workflow_record_with_structured_params(monkeypatch):
    fake = FakeState()
    monkeypatch.setenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_LIBRARY", "true")
    monkeypatch.setattr(library, "run_state_manager", fake)

    # Explicit structured params
    result = library.save_comfyui_workflow_record(
        workflow_name="test_params",
        workflow=_workflow(),
        parameters=[
            {"name": "prompt", "type": "str", "required": True, "description": "Text prompt"},
        ],
        auto_infer_parameters=False,
    )

    assert result["saved"] is True
    params = result["record"]["parameters"]
    assert len(params) == 1
    assert params[0]["name"] == "prompt"
    assert params[0]["type"] == "str"
    assert params[0]["required"] is True


def test_save_workflow_record_auto_infers_params(monkeypatch):
    fake = FakeState()
    monkeypatch.setenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_LIBRARY", "true")
    monkeypatch.setattr(library, "run_state_manager", fake)

    result = library.save_comfyui_workflow_record(
        workflow_name="auto_infer_test",
        workflow=_workflow(),
        auto_infer_parameters=True,
        overwrite=True,
    )

    assert result["saved"] is True
    params = result["record"]["parameters"]
    # _workflow has: CLIPTextEncode.text (str) + KSampler.seed (int) + KSampler.steps (int)
    names = {p["name"] for p in params}
    assert "text" in names
    assert "seed" in names
    assert "steps" in names

    outputs = result["record"]["outputs"]
    # _workflow has no output nodes → empty
    assert outputs == []


def test_apply_workflow_parameters_with_type_coercion():
    schema = [
        {"name": "seed", "type": "int"},
        {"name": "steps", "type": "int"},
        {"name": "cfg", "type": "float"},
    ]

    patched, report = library.apply_workflow_parameters(
        _workflow(),
        {"seed": "999", "steps": "50", "cfg": "2.5", "prompt": "hello"},
        parameter_map={"prompt": "1.inputs.text"},
        parameter_schema=schema,
    )

    # Type coercion from schema
    assert isinstance(patched["2"]["inputs"]["seed"], int)
    assert patched["2"]["inputs"]["seed"] == 999
    assert isinstance(patched["2"]["inputs"]["steps"], int)
    assert patched["2"]["inputs"]["steps"] == 50
    assert isinstance(patched["2"]["inputs"]["cfg"], float)
    assert patched["2"]["inputs"]["cfg"] == 2.5
    assert patched["1"]["inputs"]["text"] == "hello"

    assert "seed" in report["coerced"]
    assert "steps" in report["coerced"]
    assert "cfg" in report["coerced"]
