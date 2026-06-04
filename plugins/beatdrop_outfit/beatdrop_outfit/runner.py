from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_plan(plan_json_or_path: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(plan_json_or_path, dict):
        return plan_json_or_path
    raw = str(plan_json_or_path or "").strip()
    if not raw:
        raise ValueError("plan_json_or_path is required")
    path = Path(raw).expanduser()
    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("drop plan must be a JSON object")
    return data


def _select_drop(plan: dict[str, Any], drop_id: str) -> dict[str, Any]:
    drops = plan.get("drops") or []
    if not isinstance(drops, list) or not drops:
        raise ValueError("drop plan contains no drops")
    wanted = str(drop_id or "").strip()
    if not wanted:
        return drops[0]
    for drop in drops:
        if isinstance(drop, dict) and str(drop.get("drop_id") or "") == wanted:
            return drop
    raise ValueError(f"drop_id not found: {wanted}")


def _outfit_url(plan: dict[str, Any], selected_id: str) -> str:
    for outfit in plan.get("outfit_images") or []:
        if isinstance(outfit, dict) and str(outfit.get("id") or "") == selected_id:
            return str(outfit.get("url") or outfit.get("path") or "")
    return selected_id


def build_video_outfit_drop_parameters(plan: dict[str, Any], drop: dict[str, Any], extra_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = str(drop.get("selected_outfit_image") or "")
    params: dict[str, Any] = {
        "source_video": str(plan.get("source_video") or ""),
        "reference_image": _outfit_url(plan, selected),
        "outfit_image": _outfit_url(plan, selected),
        "drop_id": str(drop.get("drop_id") or ""),
        "beat_frame": int(drop.get("beat_frame") or 0),
        "target_frame": int(drop.get("first_new_outfit_frame") or drop.get("visual_change_frame") or drop.get("beat_frame") or 0),
        "first_new_outfit_frame": int(drop.get("first_new_outfit_frame") or 0),
        "start_frame": int(drop.get("window_start_frame") or 0),
        "end_frame": int(drop.get("window_end_frame") or 0),
        "insert_black_frame": bool(drop.get("insert_black_frame")),
        "black_frame_count": int(drop.get("black_frame_count") or 0),
    }
    params.update(extra_parameters or {})
    return params


def run_video_outfit_drop(
    plan_json_or_path: str | dict[str, Any],
    drop_id: str = "",
    *,
    workflow_name: str = "outfit_change_beatdrop",
    dry_run: bool = True,
    extra_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = _load_plan(plan_json_or_path)
    drop = _select_drop(plan, drop_id)
    parameters = build_video_outfit_drop_parameters(plan, drop, extra_parameters)
    resolved_workflow = workflow_name or str(plan.get("workflow_name") or "outfit_change_beatdrop")
    result = {
        "ok": True,
        "dry_run": bool(dry_run),
        "workflow_name": resolved_workflow,
        "drop_id": parameters["drop_id"],
        "parameters": parameters,
    }
    if dry_run:
        return result
    return {
        **result,
        "ok": False,
        "blocked": True,
        "message": "Live ComfyUI submit is not wired in the extension runner yet; run with dry_run=true.",
    }
