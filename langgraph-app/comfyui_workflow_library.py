from __future__ import annotations

import copy
import json
import os
import re
import time
from typing import Any

try:  # Optional in unit tests and lightweight containers.
    import run_state_manager  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - guarded runtime fallback
    run_state_manager = None  # type: ignore[assignment]

from comfyui_client import (
    ComfyUIClient,
    _extract_model_requirements,
    _is_comfyui_api_workflow,
    _iter_node_inputs,
    _looks_like_editor_workflow,
    _workflow_node_classes,
)

TRUE_VALUES = {"1", "true", "yes", "on"}
WORKFLOW_LIBRARY_NAMESPACE = "comfyui_workflows"
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

# Max description lengths (matching Pixelle approach: descriptions in the schema)
MAX_PARAM_DESCRIPTION_CHARS = 500
MAX_OUTPUT_DESCRIPTION_CHARS = 500

# Output node classes that produce media (auto-detected like Pixelle does)
_OUTPUT_NODE_TYPES: dict[str, str] = {
    "SaveImage": "images",
    "PreviewImage": "images",
    "SaveVideo": "videos",
    "VHS_SaveVideo": "videos",
    "VHS_VideoCombine": "videos",
    "SaveAnimatedWEBP": "gifs",
    "SaveAnimatedPNG": "gifs",
    "SaveAudio": "audios",
    "VHS_SaveAudio": "audios",
    "SaveGIF": "gifs",
}

# Map of input key names to human-friendly descriptions (for auto-inference)
_PARAM_DESCRIPTIONS: dict[str, str] = {
    "text": "Text prompt for generation",
    "prompt": "Text prompt for generation",
    "positive": "Positive conditioning prompt",
    "negative": "Negative conditioning prompt",
    "seed": "Random seed for reproducibility",
    "steps": "Number of sampling steps",
    "cfg": "CFG scale (classifier-free guidance)",
    "denoise": "Denoising strength (0.0-1.0)",
    "width": "Output width in pixels",
    "height": "Output height in pixels",
    "batch_size": "Batch size (number of images per run)",
    "sampler_name": "Sampler algorithm name",
    "scheduler": "Scheduler type",
    "filename_prefix": "Output filename prefix",
    "frame_rate": "Video frame rate",
    "duration": "Duration in seconds",
    "fps": "Frames per second",
    "length": "Video length in frames",
    "frame_count": "Number of frames to generate",
    "shift": "Model shift parameter",
}


def env_bool(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default)).strip().lower() in TRUE_VALUES


def comfyui_workflow_library_enabled() -> bool:
    return env_bool("ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_LIBRARY", "false")


def _blocked() -> dict[str, Any]:
    return {
        "ok": False,
        "blocked": True,
        "message": (
            "ComfyUI workflow library is disabled. "
            "Set ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_LIBRARY=true to save and run named workflows."
        ),
    }


def _state_unavailable() -> dict[str, Any]:
    return {"ok": False, "error": "run_state_manager is unavailable; cannot persist ComfyUI workflows."}


def normalize_workflow_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "").strip()).strip("_")
    if normalized and normalized[0].isdigit():
        normalized = f"w_{normalized}"
    if not normalized or not NAME_RE.match(normalized):
        raise ValueError(
            "workflow_name must be a valid tool-style name: "
            "start with letter/underscore, then letters/digits/underscore, max 64 chars"
        )
    return normalized


# ---------------------------------------------------------------------------
# Structured parameter schema (inspired by Pixelle's DSL, stored in our record)
# ---------------------------------------------------------------------------

_PARAM_TYPES = {"str", "int", "float", "bool"}


def _infer_parameter_type(value: Any) -> str:
    """Infer parameter type from its default value (same heuristic as Pixelle)."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _is_node_reference(value: Any) -> bool:
    """ComfyUI node references are lists like ['node_id', slot_index]."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], (int, float)):
        return True
    return False


def _clean_parameter_name(key: str) -> str:
    """Normalize input key to a clean tool-style parameter name."""
    name = re.sub(r"[^A-Za-z0-9_]+", "_", str(key).strip()).strip("_").lower()
    if not name or name[0].isdigit():
        name = f"p_{name}"
    return name[:50]


def infer_workflow_parameters(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Auto-detect structured parameters from a ComfyUI API workflow.

    Scans all leaf inputs, skips node references (connections), infers types
    from default values, and generates human-readable descriptions.
    This is the AlphaRavis equivalent of Pixelle's title DSL — the schema
    is derived automatically from the workflow JSON without needing to
    annotate node titles in the ComfyUI editor.
    """
    result: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for class_type, key, value in _iter_node_inputs(workflow):
        if _is_node_reference(value):
            continue  # connected to another node → not a user-facing parameter

        name = _clean_parameter_name(key)
        param_type = _infer_parameter_type(value)

        # Disambiguate duplicate names (e.g. two "text" inputs from different nodes)
        base = name
        counter = 2
        while name in seen_names:
            name = f"{base}_{counter}"
            counter += 1
        seen_names.add(name)

        description = _PARAM_DESCRIPTIONS.get(str(key).strip().lower(), "")
        if not description and str(key).strip().lower() in {"image", "audio", "video"}:
            description = f"Upload {str(key).strip().lower()} file"

        param: dict[str, Any] = {
            "name": name,
            "field_path": "",  # filled after discovery if unique
            "type": param_type,
            "required": False,
            "default": value,
            "description": description,
        }
        result.append(param)

    # Back-fill field_path for inputs that appear uniquely
    for param in result:
        raw_key = str(param["name"]).replace("_", " ")
        for class_type, key, value in _iter_node_inputs(workflow):
            if (raw_key == str(key).strip().lower()
                    or _clean_parameter_name(key) == param["name"]):
                if _is_node_reference(value):
                    continue
                # Find node_id
                for node_id, node in workflow.items():
                    if not isinstance(node, dict):
                        continue
                    if node.get("class_type") == class_type:
                        param["field_path"] = f"{node_id}.inputs.{key}"
                        break
                break

    return result


def infer_workflow_outputs(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Auto-detect output nodes (like Pixelle does for SaveImage, SaveVideo, etc.)."""
    outputs: list[dict[str, Any]] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        output_type = _OUTPUT_NODE_TYPES.get(class_type)
        if output_type:
            outputs.append({
                "node_id": node_id,
                "output_type": output_type,
                "class_type": class_type,
                "description": f"{class_type} output node",
            })
    return outputs


def validate_parameter_schema(parameters: list[dict[str, Any]]) -> tuple[bool, str]:
    """Validate a user-provided parameter schema list."""
    if not isinstance(parameters, list):
        return False, "parameters must be a JSON array"
    seen: set[str] = set()
    for idx, param in enumerate(parameters):
        if not isinstance(param, dict):
            return False, f"parameters[{idx}] must be an object"
        name = str(param.get("name", "")).strip()
        if not name:
            return False, f"parameters[{idx}]: name is required"
        param_type = str(param.get("type", "str")).strip().lower()
        if param_type not in _PARAM_TYPES:
            return False, f"parameters[{idx}] ({name}): type must be one of {', '.join(sorted(_PARAM_TYPES))}"
        if name in seen:
            return False, f"parameters[{idx}] ({name}): duplicate parameter name"
        seen.add(name)
        field_path = str(param.get("field_path", "")).strip()
        if field_path and not re.match(r"^\d+[:\-]?\d*\.inputs\.[A-Za-z_][A-Za-z0-9_]*$", field_path):
            # lenient — just warn in description if present
            pass
    return True, ""


def _parse_json_object(value: str | dict[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return parsed


def _parse_json_list(value: str | list[Any] | None, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return parsed


def _normalize_string_list(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values or []:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _validate_api_workflow(workflow: dict[str, Any]) -> dict[str, Any] | None:
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
    return None


def _safe_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in dict(record).items() if k != "_id"}


def _public_record(record: dict[str, Any], *, include_workflow: bool = False) -> dict[str, Any]:
    public = _safe_record(record)
    if not include_workflow:
        public.pop("workflow", None)
    return public


def _truncate(text: str, max_chars: int) -> str:
    return (text or "")[:max_chars]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_comfyui_workflow_record(
    *,
    workflow_name: str,
    workflow: dict[str, Any],
    description: str = "",
    aliases: list[Any] | tuple[Any, ...] | None = None,
    parameter_map: dict[str, Any] | None = None,
    parameters: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
    auto_infer_parameters: bool = True,
    tags: list[Any] | tuple[Any, ...] | None = None,
    workflow_type: str = "",
    source: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    if not comfyui_workflow_library_enabled():
        return _blocked()
    if run_state_manager is None:
        return _state_unavailable()

    try:
        name = normalize_workflow_name(workflow_name)
    except ValueError as exc:
        return {"ok": False, "saved": False, "error": str(exc)}

    validation = _validate_api_workflow(workflow)
    if validation:
        return {"ok": False, "saved": False, "preflight": validation, "error": validation.get("error", "invalid workflow")}

    existing = run_state_manager.load_workflow_record(WORKFLOW_LIBRARY_NAMESPACE, name)
    if existing and not overwrite:
        return {
            "ok": False,
            "saved": False,
            "exists": True,
            "workflow_name": name,
            "message": "Workflow already exists. Pass overwrite=true to replace it.",
        }

    now = time.time()

    # Resolve structured parameters
    final_parameters: list[dict[str, Any]] = []
    if parameters is not None:
        ok, err = validate_parameter_schema(parameters)
        if not ok:
            return {"ok": False, "saved": False, "error": err}
        final_parameters = copy.deepcopy(parameters)
    elif auto_infer_parameters:
        final_parameters = infer_workflow_parameters(workflow)

    # Resolve outputs
    final_outputs: list[dict[str, Any]] = []
    if outputs is not None:
        final_outputs = copy.deepcopy(outputs)
    else:
        final_outputs = infer_workflow_outputs(workflow)

    record = {
        "status": "active",
        "workflow_id": name,
        "name": name,
        "description": _truncate(str(description or ""), 4000),
        "aliases": _normalize_string_list(aliases),
        "tags": _normalize_string_list(tags),
        "workflow_type": _truncate(str(workflow_type or ""), 200),
        "source": _truncate(str(source or ""), 4000),
        "parameter_map": {
            str(k): str(v) for k, v in (parameter_map or {}).items() if str(k).strip() and str(v).strip()
        },
        "parameters": final_parameters,
        "outputs": final_outputs,
        "workflow": copy.deepcopy(workflow),
        "node_count": len(workflow),
        "node_classes": _workflow_node_classes(workflow),
        "model_requirements": _extract_model_requirements(workflow),
        "created_by": "alpharavis_comfyui_agent",
        "updated_at": now,
    }
    saved = run_state_manager.save_workflow_record(
        namespace=WORKFLOW_LIBRARY_NAMESPACE, workflow_id=name, record=record
    )
    if not saved.get("saved"):
        return {"ok": False, "saved": False, **saved}
    return {
        "ok": True,
        "saved": True,
        "workflow_name": name,
        "record": _public_record(saved.get("record", record), include_workflow=True),
    }


def list_comfyui_workflow_records(*, limit: int = 50, include_workflow: bool = False) -> dict[str, Any]:
    if not comfyui_workflow_library_enabled():
        return _blocked()
    if run_state_manager is None:
        return _state_unavailable()
    records = run_state_manager.list_workflow_records(
        namespace=WORKFLOW_LIBRARY_NAMESPACE, limit=max(1, min(int(limit), 200))
    )
    workflows = [
        _public_record(record, include_workflow=include_workflow)
        for record in records
        if isinstance(record, dict)
    ]
    workflows.sort(key=lambda item: str(item.get("name") or item.get("workflow_id") or ""))
    return {"ok": True, "count": len(workflows), "workflows": workflows}


def get_comfyui_workflow_record(workflow_name: str, *, include_workflow: bool = True) -> dict[str, Any]:
    if not comfyui_workflow_library_enabled():
        return _blocked()
    if run_state_manager is None:
        return _state_unavailable()

    query = str(workflow_name or "").strip()
    if not query:
        return {"ok": False, "found": False, "error": "workflow_name is required"}

    candidate_names: list[str] = []
    try:
        candidate_names.append(normalize_workflow_name(query))
    except ValueError:
        pass
    for name in candidate_names:
        record = run_state_manager.load_workflow_record(WORKFLOW_LIBRARY_NAMESPACE, name)
        if isinstance(record, dict):
            return {
                "ok": True,
                "found": True,
                "workflow_name": record.get("name") or name,
                "record": _public_record(record, include_workflow=include_workflow),
            }

    listed = list_comfyui_workflow_records(limit=200, include_workflow=True)
    for record in listed.get("workflows", []) if isinstance(listed, dict) else []:
        aliases = [
            str(alias).strip().lower()
            for alias in record.get("aliases", [])
            if str(alias).strip()
        ]
        names = {
            str(record.get("name") or "").lower(),
            str(record.get("workflow_id") or "").lower(),
            *aliases,
        }
        if query.lower() in names:
            return {
                "ok": True,
                "found": True,
                "workflow_name": record.get("name") or record.get("workflow_id"),
                "record": _public_record(record, include_workflow=include_workflow),
            }

    return {
        "ok": True,
        "found": False,
        "workflow_name": query,
        "message": "No saved ComfyUI workflow matched that name or alias.",
    }


def _set_path(workflow: dict[str, Any], path: str, value: Any) -> str | None:
    parts = [part for part in str(path or "").split(".") if part]
    if len(parts) == 2:
        parts = [parts[0], "inputs", parts[1]]
    if len(parts) != 3 or parts[1] != "inputs":
        return None
    node_id, _, input_key = parts
    node = workflow.get(node_id)
    if not isinstance(node, dict):
        return None
    inputs = node.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        return None
    inputs[input_key] = value
    return f"{node_id}.inputs.{input_key}"


def _find_unique_input_path(workflow: dict[str, Any], input_key: str) -> str | None:
    matches: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and input_key in inputs:
            matches.append(f"{node_id}.inputs.{input_key}")
    if len(matches) == 1:
        return matches[0]
    return None


def apply_workflow_parameters(
    workflow: dict[str, Any],
    parameters: dict[str, Any] | None,
    *,
    parameter_map: dict[str, Any] | None = None,
    parameter_schema: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply user parameters to a workflow, with optional type coercion from schema."""
    patched = copy.deepcopy(workflow)
    report: dict[str, Any] = {"applied": {}, "unresolved": [], "ambiguous": {}, "coerced": {}}
    params = parameters or {}
    param_map = {str(k): str(v) for k, v in (parameter_map or {}).items() if str(k).strip() and str(v).strip()}

    # Build type coercion map from schema
    type_map: dict[str, str] = {}
    for entry in (parameter_schema or []):
        pname = str(entry.get("name", "")).strip()
        ptype = str(entry.get("type", "str")).strip().lower()
        if pname and ptype in _PARAM_TYPES:
            type_map[pname] = ptype

    for key, value in params.items():
        param = str(key)

        # Type coercion from schema
        coerced = value
        wanted_type = type_map.get(param)
        if wanted_type and not isinstance(value, (int, float)) and str(value).lstrip("-").replace(".", "", 1).isdigit():
            try:
                if wanted_type == "int":
                    coerced = int(value)
                elif wanted_type == "float":
                    coerced = float(value)
                elif wanted_type == "bool":
                    coerced = str(value).strip().lower() in TRUE_VALUES
                    if param == "cfg" and isinstance(coerced, bool):
                        continue  # strings not meant to be bool — skip coercion
                if coerced != value and wanted_type != "str":
                    report["coerced"][param] = f"{type(value).__name__} → {wanted_type}"
            except (ValueError, TypeError):
                coerced = value

        target_path = param_map.get(param)
        if not target_path and "." in param:
            target_path = param
        if not target_path:
            target_path = _find_unique_input_path(patched, param)
        if target_path:
            applied_path = _set_path(patched, target_path, coerced)
            if applied_path:
                report["applied"][param] = applied_path
                continue
        report["unresolved"].append(param)
    return patched, report


async def submit_saved_comfyui_workflow_record(
    workflow_name: str,
    parameters: dict[str, Any] | None = None,
    *,
    client: Any | None = None,
    client_id: str = "alpharavis",
    allow_unresolved_parameters: bool = False,
) -> dict[str, Any]:
    loaded = get_comfyui_workflow_record(workflow_name, include_workflow=True)
    if not loaded.get("ok") or not loaded.get("found"):
        return {"ok": False, **loaded}
    record = loaded.get("record") or {}
    workflow = record.get("workflow")
    if not isinstance(workflow, dict):
        return {"ok": False, "workflow_name": workflow_name, "error": "Saved workflow record has no workflow JSON."}
    patched, parameter_report = apply_workflow_parameters(
        workflow,
        parameters or {},
        parameter_map=record.get("parameter_map") or {},
        parameter_schema=record.get("parameters") or [],
    )
    if parameter_report.get("unresolved") and not allow_unresolved_parameters:
        return {
            "ok": False,
            "blocked": True,
            "workflow_name": record.get("name") or workflow_name,
            "parameter_report": parameter_report,
            "message": (
                "Some parameters could not be mapped into the workflow. "
                "Add them to parameter_map or pass allow_unresolved_parameters=true."
            ),
        }
    submit_client = client or ComfyUIClient()
    submit_result = await submit_client.submit_workflow(patched, client_id=client_id or "alpharavis")
    return {
        "ok": not bool(submit_result.get("blocked")) and not bool(submit_result.get("error")),
        "workflow_name": record.get("name") or workflow_name,
        "parameter_report": parameter_report,
        "submit_result": submit_result,
    }


__all__ = [
    "WORKFLOW_LIBRARY_NAMESPACE",
    "apply_workflow_parameters",
    "comfyui_workflow_library_enabled",
    "get_comfyui_workflow_record",
    "infer_workflow_outputs",
    "infer_workflow_parameters",
    "list_comfyui_workflow_records",
    "normalize_workflow_name",
    "save_comfyui_workflow_record",
    "submit_saved_comfyui_workflow_record",
    "validate_parameter_schema",
]
