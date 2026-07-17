from __future__ import annotations

import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any

from .render_contract import normalize_analysis_plan


_WORKFLOW_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _normalize_workflow_name(name: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "").strip()).strip("_")
    if normalized and normalized[0].isdigit():
        normalized = f"w_{normalized}"
    if not normalized or not _WORKFLOW_NAME_RE.fullmatch(normalized):
        raise ValueError(
            "workflow_name must start with a letter/underscore, contain only "
            "letters/digits/underscores after normalization, and be at most 64 characters"
        )
    return normalized


def _env_bool(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolve_allowed_plan_path(path: Path) -> Path:
    if not _env_bool("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS", "false"):
        raise ValueError(
            "Local plan paths are disabled for BeatDrop Outfit. Pass inline JSON "
            "or set ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS=true."
        )
    roots = [
        Path(value.strip()).expanduser().resolve()
        for value in os.getenv(
            "ALPHARAVIS_BEATDROP_OUTFIT_ALLOWED_ROOTS", ""
        ).split(os.pathsep)
        if value.strip()
    ]
    if not roots:
        raise ValueError(
            "No allowed root is configured in ALPHARAVIS_BEATDROP_OUTFIT_ALLOWED_ROOTS."
        )
    resolved = path.expanduser().resolve()
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError("plan path must be under a configured allowed root")
    return resolved


def _extract_authoritative_outputs(entry: Any) -> list[dict[str, Any]]:
    outputs = entry.get("outputs") if isinstance(entry, dict) else None
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
                extracted.append(
                    {
                        **item,
                        "node_id": str(node_id),
                        "output_type": output_type,
                        "filename": str(item.get("filename") or ""),
                        "subfolder": str(item.get("subfolder") or ""),
                        "type": str(item.get("type") or "output"),
                    }
                )
    return extracted


def _output_matches_invocation(output: dict[str, Any], output_key: str) -> bool:
    expected_parts = output_key.split("/")
    if len(expected_parts) < 2 or any(not part for part in expected_parts):
        return False
    expected_subfolder = "/".join(expected_parts[:-1])
    expected_filename_prefix = expected_parts[-1]
    actual_subfolder = str(output.get("subfolder") or "").strip("/\\").replace("\\", "/")
    actual_filename = str(output.get("filename") or "").strip()
    return actual_subfolder == expected_subfolder and (
        actual_filename == expected_filename_prefix
        or actual_filename.startswith(expected_filename_prefix + "_")
        or actual_filename.startswith(expected_filename_prefix + ".")
    )


def _load_plan(plan_json_or_path: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(plan_json_or_path, dict):
        return plan_json_or_path
    raw = str(plan_json_or_path or "").strip()
    if not raw:
        raise ValueError("plan_json_or_path is required")
    path = Path(raw).expanduser()
    looks_like_path = not raw.startswith("{") and (
        path.suffix.lower() == ".json"
        or "/" in raw
        or "\\" in raw
        or raw.startswith((".", "~"))
    )
    if looks_like_path:
        path = _resolve_allowed_plan_path(path)
        if not path.exists():
            raise FileNotFoundError(f"plan path not found: {path}")
        if not path.is_file():
            raise ValueError(f"plan path must be a file, not a directory: {path}")
        data = json.loads(path.read_text())
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("drop plan must be a JSON object")
    return data


def build_beatdrop_render_schedule(
    plan_json_or_path: str | dict[str, Any],
) -> dict[str, Any]:
    return normalize_analysis_plan(_load_plan(plan_json_or_path))


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


async def run_beatdrop_outfit_sequence(
    plan_json_or_path: str | Path | dict[str, Any],
    *,
    workflow_name: str = "beatdrop_outfit_sequence",
    dry_run: bool = True,
    extra_parameters: dict[str, Any] | None = None,
    submitter: Any = None,
    history_loader: Any = None,
    attempt_store: Any = None,
    client_id: str = "alpharavis-beatdrop",
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    """Build or execute one saved-workflow invocation per BeatDrop schedule item."""
    from .render_contract import _canonical_json, new_render_attempt

    loaded = _load_plan(plan_json_or_path)  # type: ignore[arg-type]
    if all(field in loaded for field in ("items", "plan_hash", "run_id")):
        schedule = new_render_attempt(loaded)
    else:
        schedule = normalize_analysis_plan(loaded)

    resolved_workflow = _normalize_workflow_name(workflow_name)
    schedule_json = _canonical_json(schedule)
    invocations: list[dict[str, Any]] = []
    for item in schedule["items"]:
        iteration = item["iteration"]
        output_key = (
            f"beatdrop/{schedule['run_id']}/{schedule['attempt_id']}/"
            f"iteration_{iteration:04d}"
        )
        parameters = dict(extra_parameters or {})
        parameters.update(
            {
                "schedule_json": schedule_json,
                "run_id": schedule["run_id"],
                "attempt_id": schedule["attempt_id"],
                "plan_hash": schedule["plan_hash"],
                "iteration": iteration,
                "source_video": schedule["source_video"],
                "beat_frame": item["beat_frame"],
                "source_frame_index": item["source_frame_index"],
                "outfit_image": item["outfit_path"],
                "reference_image": item["outfit_path"],
                "outfit_path": item["outfit_path"],
                "output_key": output_key,
                "filename_prefix": output_key,
            }
        )
        invocations.append(
            {
                "workflow_name": resolved_workflow,
                "iteration": iteration,
                "output_key": output_key,
                "parameters": parameters,
            }
        )

    result = {
        "ok": True,
        "dry_run": bool(dry_run),
        "workflow_name": resolved_workflow,
        "schedule": schedule,
        "invocations": invocations,
        "completed_records": [],
        "counts": {"total": len(invocations), "completed": 0, "failed": 0},
    }
    if dry_run:
        return result

    try:
        timeout_value = float(timeout_seconds)
        poll_interval_value = float(poll_interval_seconds)
    except (TypeError, ValueError) as exc:
        return {
            **result,
            "ok": False,
            "dry_run": False,
            "blocked": True,
            "error": f"timeout_seconds and poll_interval_seconds must be finite non-negative numbers: {exc}",
        }
    invalid_timing_fields = [
        name
        for name, value in (
            ("timeout_seconds", timeout_value),
            ("poll_interval_seconds", poll_interval_value),
        )
        if not math.isfinite(value) or value < 0
    ]
    if invalid_timing_fields:
        return {
            **result,
            "ok": False,
            "dry_run": False,
            "blocked": True,
            "error": (
                f"{', '.join(invalid_timing_fields)} must be finite non-negative numbers"
            ),
        }

    import asyncio
    import importlib
    import inspect
    import time

    dependency_errors: list[str] = []
    if submitter is None:
        try:
            workflow_library = importlib.import_module("comfyui_workflow_library")
            submitter = workflow_library.submit_saved_comfyui_workflow_record
        except Exception as exc:
            dependency_errors.append(f"submitter unavailable: {exc}")
    if history_loader is None:
        try:
            comfyui_client_module = importlib.import_module("comfyui_client")
            history_loader = comfyui_client_module.ComfyUIClient().history_outputs
        except Exception as exc:
            dependency_errors.append(f"history loader unavailable: {exc}")
    elif not callable(history_loader) and callable(
        getattr(history_loader, "history_outputs", None)
    ):
        history_loader = history_loader.history_outputs
    if attempt_store is None:
        try:
            state_manager = importlib.import_module("run_state_manager")
            attempt_store = state_manager.save_workflow_record
        except Exception as exc:
            dependency_errors.append(f"attempt store unavailable: {exc}")
    elif not callable(attempt_store) and callable(
        getattr(attempt_store, "save_workflow_record", None)
    ):
        attempt_store = attempt_store.save_workflow_record
    for dependency_name, dependency in (
        ("submitter", submitter),
        ("history loader", history_loader),
        ("attempt store", attempt_store),
    ):
        if not callable(dependency):
            dependency_errors.append(f"{dependency_name} must be callable")
    if dependency_errors:
        return {
            **result,
            "ok": False,
            "dry_run": False,
            "blocked": True,
            "error": ("live dependencies unavailable: " + "; ".join(dependency_errors))[:1000],
        }

    async def call_dependency(function: Any, *args: Any, **kwargs: Any) -> Any:
        is_async_callable = inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(
            getattr(function, "__call__", None)
        )
        if is_async_callable:
            value = function(*args, **kwargs)
        else:
            loop = asyncio.get_running_loop()
            completed = loop.create_future()

            def resolve_success(value: Any) -> None:
                if not completed.done():
                    completed.set_result(value)

            def resolve_error(exc: Exception) -> None:
                if not completed.done():
                    completed.set_exception(exc)

            def invoke() -> None:
                try:
                    value = function(*args, **kwargs)
                except Exception as exc:
                    callback = lambda exc=exc: resolve_error(exc)
                else:
                    callback = lambda value=value: resolve_success(value)
                try:
                    loop.call_soon_threadsafe(callback)
                except RuntimeError:
                    pass

            threading.Thread(target=invoke, daemon=True).start()
            value = await completed
        return await value if inspect.isawaitable(value) else value

    async def call_dependency_bounded(
        function: Any, *args: Any, **kwargs: Any
    ) -> Any:
        dependency_timeout = timeout_value if timeout_value > 0 else 0.05
        return await asyncio.wait_for(
            call_dependency(function, *args, **kwargs),
            timeout=dependency_timeout,
        )

    async def call_submitter_reconciled(
        function: Any, *args: Any, **kwargs: Any
    ) -> tuple[Any, bool]:
        """Never abandon a synchronous submitter that may create a remote prompt."""
        is_async_callable = inspect.iscoroutinefunction(
            function
        ) or inspect.iscoroutinefunction(getattr(function, "__call__", None))
        dependency_timeout = timeout_value if timeout_value > 0 else 0.05
        if is_async_callable:
            value = await asyncio.wait_for(
                call_dependency(function, *args, **kwargs),
                timeout=dependency_timeout,
            )
            return value, False

        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        timed_out = False
        try:
            value = await asyncio.wait_for(
                asyncio.shield(task), timeout=dependency_timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            value = await task
        return (await value if inspect.isawaitable(value) else value), timed_out

    def history_execution_error(history: dict[str, Any], prompt_id: str) -> str:
        raw_history = history.get("history")
        entry = raw_history.get(prompt_id) if isinstance(raw_history, dict) else None
        status = entry.get("status") if isinstance(entry, dict) else history.get("status")
        if not isinstance(status, dict):
            return ""
        status_str = str(status.get("status_str") or status.get("status") or "").lower()
        messages = status.get("messages") or []
        error_details: list[str] = []
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, (list, tuple)) or len(message) < 2:
                    continue
                if str(message[0]).lower() not in {"execution_error", "error", "failed"}:
                    continue
                payload = message[1]
                if isinstance(payload, dict):
                    detail = (
                        payload.get("exception_message")
                        or payload.get("error")
                        or payload.get("message")
                    )
                else:
                    detail = payload
                if detail:
                    error_details.append(str(detail))
        if status_str in {"error", "failed", "failure"} or error_details:
            detail = "; ".join(error_details) or status_str or "unknown execution error"
            return f"ComfyUI execution failed: {detail}"[:500]
        return ""

    async def persist(workflow_id: str, record: dict[str, Any]) -> tuple[bool, str]:
        try:
            saved = await call_dependency_bounded(
                attempt_store,
                namespace="beatdrop_outfit_render_attempts",
                workflow_id=workflow_id,
                record=record,
            )
        except asyncio.TimeoutError:
            return False, "persistence failed: timeout"
        except Exception as exc:
            return False, f"persistence failed: {exc}"
        if saved is True or (isinstance(saved, dict) and saved.get("saved") is True):
            return True, ""
        if isinstance(saved, dict):
            detail = saved.get("error") or (
                "persistence disabled" if saved.get("disabled") else "record was not saved"
            )
        else:
            detail = "record was not saved"
        return False, f"persistence failed: {detail}"

    async def persist_with_retry(
        workflow_id: str,
        record: dict[str, Any],
        *,
        attempts: int = 3,
    ) -> tuple[bool, str]:
        last_error = "persistence failed"
        for _ in range(attempts):
            saved, error = await persist(workflow_id, record)
            if saved:
                return True, ""
            last_error = error
        return False, last_error

    completed_records: list[dict[str, Any]] = []

    async def fail(
        workflow_id: str, record: dict[str, Any], error: str
    ) -> dict[str, Any]:
        concise_error = str(error or "sequence execution failed")[:500]
        failed = {**record, "status": "failed", "error": concise_error}
        failed_saved, failed_save_error = await persist_with_retry(workflow_id, failed)
        return {
            **result,
            "ok": False,
            "dry_run": False,
            "error": failed_save_error if not failed_saved else concise_error,
            "counts": {
                "total": len(invocations),
                "completed": len(completed_records),
                "failed": 1,
            },
            "completed_records": completed_records,
            "failed_record": failed,
        }

    for invocation in invocations:
        iteration = invocation["iteration"]
        output_key = invocation["output_key"]
        workflow_id = (
            f"run_id={schedule['run_id']}:attempt_id={schedule['attempt_id']}:"
            f"plan_hash={schedule['plan_hash']}:iteration={iteration}:"
            f"output_key={output_key}:workflow_name={resolved_workflow}"
        )
        record = {
            "run_id": schedule["run_id"],
            "attempt_id": schedule["attempt_id"],
            "plan_hash": schedule["plan_hash"],
            "iteration": iteration,
            "output_key": output_key,
            "prompt_id": "",
            "status": "prepared",
            "workflow_name": resolved_workflow,
        }
        persisted, persistence_error = await persist_with_retry(workflow_id, record)
        if not persisted:
            return {
                **result,
                "ok": False,
                "dry_run": False,
                "error": persistence_error,
                "counts": {
                    "total": len(invocations),
                    "completed": len(completed_records),
                    "failed": 1,
                },
                "completed_records": completed_records,
                "failed_record": record,
            }

        submit_reconciled_after_timeout = False
        try:
            submit_result, submit_reconciled_after_timeout = await call_submitter_reconciled(
                submitter,
                resolved_workflow,
                invocation["parameters"],
                client_id=client_id,
            )
        except asyncio.TimeoutError:
            submit_result = {"error": "submit timeout"}
        except Exception as exc:
            submit_result = {"error": f"submit failed: {exc}"}
        nested_submit_value = (
            submit_result.get("submit_result")
            if isinstance(submit_result, dict)
            else None
        )
        nested_submit: dict[str, Any] = (
            nested_submit_value if isinstance(nested_submit_value, dict) else {}
        )
        submit_failed = isinstance(submit_result, dict) and (
            bool(submit_result.get("blocked"))
            or bool(submit_result.get("error"))
            or submit_result.get("ok") is False
            or bool(nested_submit.get("blocked"))
            or bool(nested_submit.get("error"))
            or nested_submit.get("ok") is False
        )
        if submit_failed:
            submit_error = str(
                submit_result.get("error")
                or submit_result.get("message")
                or nested_submit.get("error")
                or nested_submit.get("message")
                or "submit failed"
            )[:500]
            failed = {**record, "status": "failed", "error": submit_error}
            failed_saved, failed_save_error = await persist_with_retry(
                workflow_id, failed
            )
            return {
                **result,
                "ok": False,
                "dry_run": False,
                "error": failed_save_error if not failed_saved else submit_error,
                "counts": {"total": len(invocations), "completed": len(completed_records), "failed": 1},
                "completed_records": completed_records,
                "failed_record": failed,
            }
        prompt_id = ""
        if isinstance(submit_result, str):
            prompt_id = submit_result.strip()
        elif isinstance(submit_result, dict):
            prompt_id = str(
                submit_result.get("prompt_id")
                or (
                    nested_submit.get("prompt_id")
                    if isinstance(nested_submit, dict)
                    else ""
                )
                or ""
            ).strip()
        if not prompt_id:
            failed = {**record, "status": "failed", "error": "submit returned no prompt_id"}
            failed_saved, failed_save_error = await persist_with_retry(
                workflow_id, failed
            )
            return {
                **result,
                "ok": False,
                "dry_run": False,
                "error": failed_save_error if not failed_saved else failed["error"],
                "counts": {"total": len(invocations), "completed": len(completed_records), "failed": 1},
                "completed_records": completed_records,
                "failed_record": failed,
            }
        submitted = {
            **record,
            "prompt_id": prompt_id,
            "status": "submitted",
            **(
                {"submission_reconciled_after_timeout": True}
                if submit_reconciled_after_timeout
                else {}
            ),
        }
        persisted, persistence_error = await persist_with_retry(
            workflow_id, submitted
        )
        if not persisted:
            return {
                **result,
                "ok": False,
                "dry_run": False,
                "blocked": True,
                "requires_reconciliation": True,
                "do_not_retry": True,
                "orphaned_prompt_id": prompt_id,
                "error": persistence_error,
                "counts": {"total": len(invocations), "completed": len(completed_records), "failed": 1},
                "completed_records": completed_records,
                "failed_record": submitted,
            }

        deadline = time.monotonic() + timeout_value
        first_poll = True
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 and not first_poll:
                return await fail(workflow_id, submitted, "history timeout")
            call_timeout = remaining if remaining > 0 else 0.05
            try:
                history = await asyncio.wait_for(
                    call_dependency(history_loader, prompt_id),
                    timeout=call_timeout,
                )
            except asyncio.TimeoutError:
                return await fail(workflow_id, submitted, "history timeout")
            except Exception as exc:
                return await fail(workflow_id, submitted, f"history failed: {exc}")
            first_poll = False
            if not isinstance(history, dict):
                return await fail(
                    workflow_id, submitted, "history response must be an object"
                )
            if history.get("error"):
                return await fail(
                    workflow_id, submitted, f"history failed: {history['error']}"
                )
            response_prompt_id = str(history.get("prompt_id") or "").strip()
            if not response_prompt_id:
                return await fail(
                    workflow_id,
                    submitted,
                    f"history response missing prompt_id for expected {prompt_id}",
                )
            if response_prompt_id != prompt_id:
                return await fail(
                    workflow_id,
                    submitted,
                    f"history prompt_id mismatch: expected {prompt_id}, got {response_prompt_id}",
                )
            raw_history = history.get("history")
            reported_outputs = history.get("outputs") or []
            if not isinstance(reported_outputs, list):
                return await fail(
                    workflow_id, submitted, "history outputs must be a list"
                )
            if not isinstance(raw_history, dict) or prompt_id not in raw_history:
                if reported_outputs:
                    return await fail(
                        workflow_id,
                        submitted,
                        f"history key mismatch: expected authoritative raw history entry {prompt_id}",
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return await fail(workflow_id, submitted, "history timeout")
                await asyncio.sleep(min(poll_interval_value, remaining))
                continue
            execution_error = history_execution_error(history, prompt_id)
            if execution_error:
                return await fail(workflow_id, submitted, execution_error)
            raw_entry = raw_history[prompt_id]
            outputs = _extract_authoritative_outputs(raw_entry)
            if reported_outputs and not outputs:
                return await fail(
                    workflow_id,
                    submitted,
                    "history outputs are not backed by the authoritative raw history entry",
                )
            if outputs:
                if len(outputs) != 1:
                    return await fail(
                        workflow_id,
                        submitted,
                        "history must contain exactly one image output",
                    )
                output = outputs[0]
                output_type = (
                    str(output.get("output_type") or "").lower()
                    if isinstance(output, dict)
                    else ""
                )
                output_kind = (
                    str(output.get("kind") or "").lower()
                    if isinstance(output, dict)
                    else ""
                )
                output_mime = (
                    str(output.get("mime_type") or output.get("mime") or "").lower()
                    if isinstance(output, dict)
                    else ""
                )
                if not (
                    output_type in {"image", "images"}
                    or output_kind == "image"
                    or output_mime.startswith("image/")
                ):
                    return await fail(
                        workflow_id, submitted, "history output must be an image"
                    )
                filename = str(output.get("filename") or "").strip()
                if (
                    not filename
                    or filename in {".", ".."}
                    or "/" in filename
                    or "\\" in filename
                ):
                    return await fail(
                        workflow_id,
                        submitted,
                        "history image output must contain a plain nonblank filename",
                    )
                output = {**output, "filename": filename}
                if not _output_matches_invocation(output, invocation["output_key"]):
                    return await fail(
                        workflow_id,
                        submitted,
                        "history image output does not match the invocation output_key/filename_prefix",
                    )
                completed = {
                    **submitted,
                    "status": "completed",
                    "output": output,
                }
                persisted, persistence_error = await persist_with_retry(
                    workflow_id, completed
                )
                if not persisted:
                    return {
                        **result,
                        "ok": False,
                        "dry_run": False,
                        "error": persistence_error,
                        "counts": {"total": len(invocations), "completed": len(completed_records), "failed": 1},
                        "completed_records": completed_records,
                        "failed_record": completed,
                    }
                completed_records.append(completed)
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return await fail(workflow_id, submitted, "history timeout")
            await asyncio.sleep(min(poll_interval_value, remaining))

    return {
        **result,
        "dry_run": False,
        "completed_records": completed_records,
        "counts": {"total": len(invocations), "completed": len(completed_records), "failed": 0},
    }
