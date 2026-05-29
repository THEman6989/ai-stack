from __future__ import annotations

import os
import re
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, urlencode, urlparse, urlunparse

import httpx
from env_utils import env_bool


ALLOWED_MODEL_FOLDERS = {
    "checkpoints",
    "vae",
    "loras",
    "controlnet",
    "clip",
    "clip_vision",
    "text_encoders",
    "unet",
    "embeddings",
    "diffusion_models",
    "style_models",
    "upscale_models",
}
MODEL_INPUT_FOLDERS = {
    "ckpt_name": "checkpoints",
    "checkpoint": "checkpoints",
    "checkpoint_name": "checkpoints",
    "lora_name": "loras",
    "lora": "loras",
    "vae_name": "vae",
    "vae": "vae",
    "control_net_name": "controlnet",
    "controlnet_name": "controlnet",
    "control_net": "controlnet",
    "clip_name": "clip",
    "clip_name1": "clip",
    "clip_name2": "clip",
    "clip_name3": "clip",
    "clip_vision_name": "clip_vision",
    "clip": "clip",
    "unet_name": "unet",
    "unet": "unet",
    "diffusion_model": "diffusion_models",
    "diffusion_model_name": "diffusion_models",
    "style_model_name": "style_models",
    "upscale_model_name": "upscale_models",
}
NODE_CLASS_MODEL_INPUT_FOLDERS = {
    ("UNETLoader", "unet_name"): "diffusion_models",
    ("UNETLoader", "unet"): "diffusion_models",
    ("CLIPLoader", "clip_name"): "text_encoders",
    ("CLIPLoader", "clip"): "text_encoders",
    ("CLIPVisionLoader", "clip_name"): "clip_vision",
    ("CLIPVisionLoader", "clip_vision_name"): "clip_vision",
    ("UpscaleModelLoader", "model_name"): "upscale_models",
    ("StyleModelLoader", "model_name"): "style_models",
    ("StyleModelLoader", "style_model_name"): "style_models",
}


def _normalize_base_url(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if value.startswith("unix://"):
        parsed = urlparse(value)
        socket_path = parsed.path or parsed.netloc
        if not socket_path.startswith("/"):
            socket_path = f"/{socket_path}"
        return f"unix://{socket_path}"
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlparse(value)
    path = parsed.path.rstrip("/")
    for suffix in ("/system_stats", "/queue", "/object_info"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
    return urlunparse((parsed.scheme or "http", parsed.netloc, path, "", "", "")).rstrip("/")


def _safe_model_folder(folder: str = "checkpoints") -> str:
    safe_folder = (folder or "checkpoints").strip().strip("/") or "checkpoints"
    if safe_folder not in ALLOWED_MODEL_FOLDERS:
        allowed = ", ".join(sorted(ALLOWED_MODEL_FOLDERS))
        raise ValueError(f"Unsupported ComfyUI model folder: {safe_folder!r}. Allowed: {allowed}")
    return safe_folder


def _is_comfyui_api_workflow(workflow: dict[str, Any]) -> bool:
    return bool(workflow) and all(isinstance(v, dict) and "class_type" in v for v in workflow.values())


def _looks_like_editor_workflow(workflow: dict[str, Any]) -> bool:
    return isinstance(workflow.get("nodes"), list) and isinstance(workflow.get("links"), list)


def _workflow_node_classes(workflow: dict[str, Any]) -> list[str]:
    classes: list[str] = []
    for node in workflow.values():
        if isinstance(node, dict):
            class_type = node.get("class_type")
            if isinstance(class_type, str) and class_type and class_type not in classes:
                classes.append(class_type)
    return classes


def _iter_node_inputs(workflow: dict[str, Any]) -> list[tuple[str, str, Any]]:
    pairs: list[tuple[str, str, Any]] = []
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            pairs.extend((class_type, key, value) for key, value in inputs.items())
    return pairs


def _extract_model_requirements(workflow: dict[str, Any]) -> dict[str, list[str]]:
    required: dict[str, set[str]] = {folder: set() for folder in ALLOWED_MODEL_FOLDERS}
    for class_type, key, value in _iter_node_inputs(workflow):
        input_key = str(key)
        folder = NODE_CLASS_MODEL_INPUT_FOLDERS.get((class_type, input_key)) or MODEL_INPUT_FOLDERS.get(input_key)
        if folder and isinstance(value, str) and value.strip() and value.lower() not in {"none", "default"}:
            required[folder].add(value.strip())
        if isinstance(value, str):
            for match in re.findall(r"embedding:([A-Za-z0-9_.\-/]+)", value):
                required["embeddings"].add(match.strip())
    return {folder: sorted(values) for folder, values in required.items() if values}


def _model_names_from_payload(payload: Any) -> list[str]:
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        for key in ("models", "data", "files"):
            value = payload.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
            if isinstance(value, dict):
                values = list(value.values())
                if all(isinstance(item, str) for item in values):
                    return [str(item) for item in values]
    return []


def _model_present(required: str, available: list[str]) -> bool:
    req = required.strip().lower()
    req_name = PurePosixPath(req).name
    req_stem = req_name.rsplit(".", 1)[0]
    for item in available:
        cand = item.strip().lower()
        cand_name = PurePosixPath(cand).name
        cand_stem = cand_name.rsplit(".", 1)[0]
        if req in {cand, cand_name} or req_name in {cand, cand_name} or req_stem == cand_stem:
            return True
    return False


def extract_history_outputs(history: dict[str, Any], prompt_id: str = "", *, base_url: str = "") -> list[dict[str, Any]]:
    """Extract image/video/audio output metadata from a ComfyUI /history payload."""

    if not isinstance(history, dict):
        return []
    prompt_payload: Any = history
    if prompt_id and isinstance(history.get(prompt_id), dict):
        prompt_payload = history[prompt_id]
    elif len(history) == 1:
        only_value = next(iter(history.values()))
        if isinstance(only_value, dict) and "outputs" in only_value:
            prompt_payload = only_value
    outputs = prompt_payload.get("outputs") if isinstance(prompt_payload, dict) else None
    if not isinstance(outputs, dict):
        return []

    extracted: list[dict[str, Any]] = []
    for node_id, node_outputs in outputs.items():
        if not isinstance(node_outputs, dict):
            continue
        for output_type in ("images", "videos", "gifs", "audio"):
            items = node_outputs.get(output_type)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                filename = str(item.get("filename") or "").strip()
                if not filename:
                    continue
                subfolder = str(item.get("subfolder") or "")
                file_type = str(item.get("type") or "output")
                record = {
                    "node_id": str(node_id),
                    "output_type": output_type,
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": file_type,
                }
                if base_url:
                    query = urlencode({"filename": filename, "subfolder": subfolder, "type": file_type})
                    record["url"] = f"{base_url.rstrip('/')}/view?{query}"
                extracted.append(record)
    return extracted


def resolve_comfyui_base_url(remote_pcs: dict[str, Any] | None = None) -> str:
    """Resolve the ComfyUI REST base URL without hardcoding a LAN address."""

    for name in ("ALPHARAVIS_COMFYUI_API_BASE", "ALPHARAVIS_COMFY_API_BASE"):
        base = _normalize_base_url(os.getenv(name, ""))
        if base:
            return base

    health = _normalize_base_url(os.getenv("ALPHARAVIS_COMFY_HEALTH_URL", ""))
    if health:
        return health

    remote_pcs = remote_pcs or {}
    pc_key = os.getenv("ALPHARAVIS_COMFY_PC", "comfy_server")
    comfy_pc = remote_pcs.get(pc_key) or remote_pcs.get("comfy_server") or {}
    if isinstance(comfy_pc, dict):
        raw_url = comfy_pc.get("comfyui_url") or comfy_pc.get("comfy_url") or comfy_pc.get("url")
        base = _normalize_base_url(str(raw_url or ""))
        if base:
            return base
        ip = str(comfy_pc.get("ip") or "").strip()
        if ip:
            port = str(comfy_pc.get("comfyui_port") or comfy_pc.get("comfy_port") or os.getenv("ALPHARAVIS_COMFYUI_PORT", "8188"))
            return _normalize_base_url(f"http://{ip}:{port}")

    return _normalize_base_url(os.getenv("ALPHARAVIS_COMFYUI_FALLBACK_BASE", "http://127.0.0.1:8188"))


def comfyui_workflow_submit_enabled() -> bool:
    """Return the canonical ComfyUI submit gate, with a legacy alias for operators.

    `ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT` remains canonical. The older
    `ALPHARAVIS_COMFYUI_WORKFLOW_SUBMIT` alias is accepted fail-closed/backwards
    compatible when the canonical variable is unset.
    """

    if "ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT" in os.environ:
        return env_bool("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT", "false")
    return env_bool("ALPHARAVIS_COMFYUI_WORKFLOW_SUBMIT", "false")


class ComfyUIClient:
    def __init__(self, base_url: str | None = None, *, timeout: float | None = None):
        self.base_url = _normalize_base_url(base_url or resolve_comfyui_base_url())
        self.timeout = timeout or float(os.getenv("ALPHARAVIS_COMFYUI_TIMEOUT_SECONDS", "30"))
        parsed = urlparse(self.base_url)
        self._uds_path = parsed.path if parsed.scheme == "unix" else ""
        self._http_base_url = "http://comfyui" if self._uds_path else self.base_url
        public_base = _normalize_base_url(os.getenv("ALPHARAVIS_COMFYUI_PUBLIC_BASE_URL", ""))
        self.public_base_url = public_base or ("http://localhost:8188" if self._uds_path else self.base_url)

    def _async_client(self) -> httpx.AsyncClient:
        if self._uds_path:
            return httpx.AsyncClient(
                timeout=self.timeout,
                transport=httpx.AsyncHTTPTransport(uds=self._uds_path),
                base_url=self._http_base_url,
            )
        return httpx.AsyncClient(timeout=self.timeout)

    def _url(self, path: str) -> str:
        path = "/" + path.lstrip("/")
        if self._uds_path:
            return path
        return f"{self._http_base_url}{path}"

    async def get_json(self, path: str) -> dict[str, Any]:
        async with self._async_client() as client:
            response = await client.get(self._url(path))
            response.raise_for_status()
            data = response.json()
        return data if isinstance(data, dict) else {"data": data}

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._async_client() as client:
            response = await client.post(self._url(path), json=payload)
            response.raise_for_status()
            data = response.json() if response.content else {}
        return data if isinstance(data, dict) else {"data": data}

    async def system_stats(self) -> dict[str, Any]:
        return await self.get_json("/system_stats")

    async def queue(self) -> dict[str, Any]:
        return await self.get_json("/queue")

    async def object_info(self, class_name: str = "") -> dict[str, Any]:
        class_name = (class_name or "").strip().strip("/")
        if class_name:
            return await self.get_json(f"/object_info/{quote(class_name, safe='')}")
        return await self.get_json("/object_info")

    async def models(self, folder: str = "checkpoints") -> dict[str, Any]:
        safe_folder = _safe_model_folder(folder)
        return await self.get_json(f"/models/{safe_folder}")

    async def history(self, prompt_id: str) -> dict[str, Any]:
        prompt_id = (prompt_id or "").strip()
        if not prompt_id:
            return {"error": "prompt_id is required"}
        return await self.get_json(f"/history/{quote(prompt_id, safe='')}")

    async def history_outputs(self, prompt_id: str) -> dict[str, Any]:
        history = await self.history(prompt_id)
        return {
            "prompt_id": prompt_id,
            "history": history,
            "outputs": extract_history_outputs(history, prompt_id, base_url=self.public_base_url),
        }

    async def clear_queue(self) -> dict[str, Any]:
        return await self.post_json("/queue", {"clear": True})

    async def interrupt(self) -> dict[str, Any]:
        return await self.post_json("/interrupt", {})

    async def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> dict[str, Any]:
        return await self.post_json("/free", {"unload_models": unload_models, "free_memory": free_memory})

    def _view_query_path(self, filename: str, *, subfolder: str = "", file_type: str = "output") -> str:
        filename = (filename or "").strip()
        if not filename or "/" in filename or "\\" in filename or filename in {".", ".."}:
            raise ValueError("filename must be a plain ComfyUI output filename")
        query = urlencode({"filename": filename, "subfolder": subfolder or "", "type": file_type or "output"})
        return f"/view?{query}"

    def view_url(self, filename: str, *, subfolder: str = "", file_type: str = "output") -> str:
        return f"{self.public_base_url}{self._view_query_path(filename, subfolder=subfolder, file_type=file_type)}"

    async def view_bytes(self, filename: str, *, subfolder: str = "", file_type: str = "output") -> dict[str, Any]:
        """Fetch a ComfyUI /view output through the configured internal transport."""

        async with self._async_client() as client:
            response = await client.get(self._url(self._view_query_path(filename, subfolder=subfolder, file_type=file_type)))
            response.raise_for_status()
        return {
            "content": response.content,
            "content_type": response.headers.get("content-type") or "application/octet-stream",
            "headers": dict(response.headers),
        }

    async def upload_image_bytes(
        self,
        content: bytes,
        *,
        filename: str,
        image_type: str = "input",
        overwrite: bool = True,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        filename = (filename or "").strip()
        if not filename or "/" in filename or "\\" in filename:
            return {"ok": False, "error": "filename must be a plain filename"}
        async with self._async_client() as client:
            response = await client.post(
                self._url("/upload/image"),
                data={"type": image_type, "overwrite": str(bool(overwrite)).lower()},
                files={"image": (filename, content, content_type)},
            )
            response.raise_for_status()
            data = response.json()
        return data if isinstance(data, dict) else {"data": data}

    async def preflight_workflow(self, workflow: dict[str, Any], *, check_server: bool = True) -> dict[str, Any]:
        if not isinstance(workflow, dict) or not workflow:
            return {"ok": False, "ready": False, "error": "workflow must be a non-empty JSON object"}
        if _looks_like_editor_workflow(workflow):
            return {
                "ok": False,
                "ready": False,
                "format": "editor",
                "error": "workflow is editor format (top-level nodes/links); export ComfyUI API format first.",
            }
        if not _is_comfyui_api_workflow(workflow):
            return {
                "ok": False,
                "ready": False,
                "format": "unknown",
                "error": "workflow must be ComfyUI API format: node-id map where every node has class_type.",
            }

        node_classes = _workflow_node_classes(workflow)
        model_requirements = _extract_model_requirements(workflow)
        report: dict[str, Any] = {
            "ok": True,
            "ready": True,
            "format": "api",
            "node_count": len(workflow),
            "node_classes": node_classes,
            "model_requirements": model_requirements,
            "missing_node_classes": [],
            "missing_models": {},
            "server_checked": False,
        }
        if not check_server:
            report["ready"] = True
            return report

        try:
            object_info = await self.object_info()
            report["server_checked"] = True
            known_classes = set(object_info.keys()) if isinstance(object_info, dict) else set()
            report["missing_node_classes"] = [class_name for class_name in node_classes if class_name not in known_classes]
        except Exception as exc:
            report["ready"] = False
            report["object_info_error"] = str(exc)

        missing_models: dict[str, list[str]] = {}
        available_models: dict[str, list[str]] = {}
        for folder, required_names in model_requirements.items():
            try:
                models_payload = await self.models(folder)
                available = _model_names_from_payload(models_payload)
                available_models[folder] = available
                missing = [name for name in required_names if not _model_present(name, available)]
                if missing:
                    missing_models[folder] = missing
            except Exception as exc:
                missing_models[folder] = list(required_names)
                report.setdefault("model_check_errors", {})[folder] = str(exc)
        report["available_model_counts"] = {folder: len(names) for folder, names in available_models.items()}
        report["missing_models"] = missing_models
        report["ready"] = not report["missing_node_classes"] and not missing_models and not report.get("object_info_error")
        return report

    async def submit_workflow(self, workflow: dict[str, Any], *, client_id: str = "alpharavis") -> dict[str, Any]:
        if not comfyui_workflow_submit_enabled():
            return {
                "ok": False,
                "blocked": True,
                "message": "ComfyUI workflow submit is disabled. Set ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT=true to allow direct workflow execution.",
            }
        preflight = await self.preflight_workflow(workflow, check_server=True)
        if not preflight.get("ready"):
            return {"ok": False, "blocked": True, "preflight": preflight, "message": "ComfyUI workflow preflight failed; not submitting."}
        return await self.post_json("/prompt", {"prompt": workflow, "client_id": client_id or "alpharavis"})


async def comfyui_status(remote_pcs: dict[str, Any] | None = None) -> dict[str, Any]:
    base_url = resolve_comfyui_base_url(remote_pcs)
    client = ComfyUIClient(base_url)
    try:
        stats = await client.system_stats()
    except Exception as exc:
        return {"ok": False, "base_url": base_url, "error": str(exc)}
    return {"ok": True, "base_url": base_url, "system_stats": stats}
