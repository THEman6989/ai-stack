from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import comfyui_client  # noqa: E402


def test_resolve_comfyui_base_url_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_COMFYUI_API_BASE", "comfypc.local:8188/system_stats")

    assert comfyui_client.resolve_comfyui_base_url({}) == "http://comfypc.local:8188"


def test_resolve_comfyui_base_url_preserves_unix_socket_env(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_COMFYUI_API_BASE", "unix:///workspace/runtime/comfyui.sock")

    assert comfyui_client.resolve_comfyui_base_url({}) == "unix:///workspace/runtime/comfyui.sock"


def test_unix_socket_client_uses_public_view_urls(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_COMFYUI_PUBLIC_BASE_URL", "http://localhost:8188")
    client = comfyui_client.ComfyUIClient(base_url="unix:///workspace/runtime/comfyui.sock")

    assert client.base_url == "unix:///workspace/runtime/comfyui.sock"
    assert client.public_base_url == "http://localhost:8188"
    assert client.view_url("ComfyUI_00001_.png") == "http://localhost:8188/view?filename=ComfyUI_00001_.png&subfolder=&type=output"


def test_unix_socket_client_fetches_view_bytes_through_internal_transport(monkeypatch):
    calls: list[str] = []

    class FakeResponse:
        content = b"PNGDATA"
        headers = {"content-type": "image/png"}

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str):
            calls.append(url)
            return FakeResponse()

    class FakeClient(comfyui_client.ComfyUIClient):
        def _async_client(self):
            return FakeAsyncClient()

    monkeypatch.setenv("ALPHARAVIS_COMFYUI_PUBLIC_BASE_URL", "http://localhost:8188")
    client = FakeClient(base_url="unix:///workspace/runtime/comfyui.sock")

    result = asyncio.run(client.view_bytes("ComfyUI_00001_.png", subfolder="smoke", file_type="output"))

    assert calls == ["/view?filename=ComfyUI_00001_.png&subfolder=smoke&type=output"]
    assert result["content"] == b"PNGDATA"
    assert result["content_type"] == "image/png"


def test_resolve_comfyui_base_url_uses_remote_pc_and_port(monkeypatch):
    monkeypatch.delenv("ALPHARAVIS_COMFYUI_API_BASE", raising=False)
    monkeypatch.delenv("ALPHARAVIS_COMFY_API_BASE", raising=False)
    monkeypatch.delenv("ALPHARAVIS_COMFY_HEALTH_URL", raising=False)
    monkeypatch.setenv("ALPHARAVIS_COMFY_PC", "comfy_server")
    monkeypatch.setenv("ALPHARAVIS_COMFYUI_PORT", "8190")

    base = comfyui_client.resolve_comfyui_base_url({"comfy_server": {"ip": "192.168.1.44"}})

    assert base == "http://192.168.1.44:8190"


def test_models_rejects_unknown_folder_before_network(monkeypatch):
    client = comfyui_client.ComfyUIClient(base_url="http://comfypc:8188")

    async def run():
        try:
            await client.models("../secret")
        except ValueError as exc:
            return str(exc)
        return ""

    assert "Unsupported ComfyUI model folder" in asyncio.run(run())


def test_preflight_rejects_editor_format():
    client = comfyui_client.ComfyUIClient(base_url="http://comfypc:8188")

    result = asyncio.run(client.preflight_workflow({"nodes": [], "links": []}, check_server=False))

    assert result["ready"] is False
    assert result["format"] == "editor"


def test_preflight_extracts_node_classes_and_model_requirements_without_server():
    client = comfyui_client.ComfyUIClient(base_url="http://comfypc:8188")
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
        "2": {"class_type": "LoraLoader", "inputs": {"lora_name": "style.safetensors"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "embedding:easynegative portrait"}},
    }

    result = asyncio.run(client.preflight_workflow(workflow, check_server=False))

    assert result["ready"] is True
    assert result["node_count"] == 3
    assert result["node_classes"] == ["CheckpointLoaderSimple", "LoraLoader", "CLIPTextEncode"]
    assert result["model_requirements"] == {
        "checkpoints": ["model.safetensors"],
        "loras": ["style.safetensors"],
        "embeddings": ["easynegative"],
    }


def test_preflight_extracts_extended_comfyui_model_inputs_without_server():
    client = comfyui_client.ComfyUIClient(base_url="http://comfypc:8188")
    workflow = {
        "1": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8.safetensors"}},
        "2": {"class_type": "TripleCLIPLoader", "inputs": {"clip_name3": "clip_g.safetensors"}},
        "3": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "4": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x-ultrasharp.pth"}},
        "5": {"class_type": "StyleModelLoader", "inputs": {"style_model_name": "style.safetensors"}},
        "6": {"class_type": "FluxGuidance", "inputs": {"diffusion_model": "flux1-dev-fp8.safetensors"}},
        "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors"}},
        "8": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors"}},
    }

    result = asyncio.run(client.preflight_workflow(workflow, check_server=False))

    assert result["model_requirements"] == {
        "clip": ["clip_g.safetensors", "clip_l.safetensors", "t5xxl_fp8.safetensors"],
        "clip_vision": ["clip_vision_h.safetensors"],
        "diffusion_models": ["flux1-dev-fp8.safetensors", "z_image_turbo_bf16.safetensors"],
        "text_encoders": ["qwen_3_4b.safetensors"],
        "style_models": ["style.safetensors"],
        "upscale_models": ["4x-ultrasharp.pth"],
    }


def test_preflight_checks_missing_nodes_and_models(monkeypatch):
    class FakeClient(comfyui_client.ComfyUIClient):
        async def object_info(self, class_name: str = ""):
            return {"CheckpointLoaderSimple": {}}

        async def models(self, folder: str = "checkpoints"):
            if folder == "checkpoints":
                return {"data": ["present.safetensors"]}
            return {"data": []}

    client = FakeClient(base_url="http://comfypc:8188")
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "missing.safetensors"}},
        "2": {"class_type": "UnknownNode", "inputs": {}},
    }

    result = asyncio.run(client.preflight_workflow(workflow, check_server=True))

    assert result["ready"] is False
    assert result["missing_node_classes"] == ["UnknownNode"]
    assert result["missing_models"] == {"checkpoints": ["missing.safetensors"]}


def test_history_outputs_extracts_view_urls():
    history = {
        "abc": {
            "outputs": {
                "9": {
                    "images": [
                        {"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"},
                    ]
                }
            }
        }
    }

    outputs = comfyui_client.extract_history_outputs(history, "abc", base_url="http://comfypc:8188")

    assert outputs == [
        {
            "node_id": "9",
            "output_type": "images",
            "filename": "ComfyUI_00001_.png",
            "subfolder": "",
            "type": "output",
            "url": "http://comfypc:8188/view?filename=ComfyUI_00001_.png&subfolder=&type=output",
        }
    ]


def test_submit_workflow_is_blocked_by_default(monkeypatch):
    monkeypatch.delenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT", raising=False)
    monkeypatch.delenv("ALPHARAVIS_COMFYUI_WORKFLOW_SUBMIT", raising=False)
    client = comfyui_client.ComfyUIClient(base_url="http://comfypc:8188")

    result = asyncio.run(client.submit_workflow({"1": {"class_type": "CheckpointLoaderSimple"}}))

    assert result["blocked"] is True


def test_submit_workflow_accepts_legacy_alias_only_when_canonical_unset(monkeypatch):
    monkeypatch.delenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT", raising=False)
    monkeypatch.setenv("ALPHARAVIS_COMFYUI_WORKFLOW_SUBMIT", "true")

    assert comfyui_client.comfyui_workflow_submit_enabled() is True

    monkeypatch.setenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT", "false")

    assert comfyui_client.comfyui_workflow_submit_enabled() is False


def test_submit_workflow_runs_preflight_when_enabled(monkeypatch):
    monkeypatch.setenv("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT", "true")

    class FakeClient(comfyui_client.ComfyUIClient):
        async def preflight_workflow(self, workflow, *, check_server=True):
            return {"ok": True, "ready": True, "format": "api"}

        async def post_json(self, path, payload):
            return {"prompt_id": "abc", "path": path, "client_id": payload["client_id"]}

    client = FakeClient(base_url="http://comfypc:8188")
    result = asyncio.run(client.submit_workflow({"1": {"class_type": "CheckpointLoaderSimple"}}, client_id="test-client"))

    assert result == {"prompt_id": "abc", "path": "/prompt", "client_id": "test-client"}
