from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from typing import Any


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not valid canonical JSON") from exc


def _validate_canonical_uuid(value: Any, field: str) -> None:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{field} must be a canonical UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{field} must be a canonical UUID string") from exc
    if value != str(parsed):
        raise ValueError(f"{field} must be a canonical UUID string")


def _resolve_alias(
    value: dict[str, Any],
    canonical_name: str,
    legacy_name: str,
    *,
    strip_strings: bool = False,
) -> Any:
    canonical_present = canonical_name in value
    legacy_present = legacy_name in value
    if canonical_present and legacy_present:
        canonical_value = value[canonical_name]
        legacy_value = value[legacy_name]
        if strip_strings:
            if isinstance(canonical_value, str):
                canonical_value = canonical_value.strip()
            if isinstance(legacy_value, str):
                legacy_value = legacy_value.strip()
        if type(canonical_value) is not type(legacy_value) or canonical_value != legacy_value:
            raise ValueError(f"{canonical_name} conflicts with {legacy_name}")
    if canonical_present:
        return value[canonical_name]
    return value.get(legacy_name)


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_render_items(items: list[Any]) -> None:
    required_fields = (
        "iteration",
        "transition_index",
        "beat_index",
        "source_frame_index",
        "beat_frame",
        "time_seconds",
        "outfit_state_before",
        "outfit_state_after",
        "outfit_state",
        "outfit_state_index",
        "candidate_type",
        "candidate_frame",
        "outfit_batch_index",
        "source_identity",
        "source_path",
        "outfit_path",
    )
    seen_identities: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"items[{index}] must be an object")
        for field in required_fields:
            if field not in item:
                raise ValueError(f"items[{index}].{field} is required")

        if not _is_nonnegative_int(item["iteration"]) or item["iteration"] != index:
            raise ValueError(f"items[{index}].iteration must equal its list index")
        if (
            not _is_nonnegative_int(item["transition_index"])
            or item["transition_index"] != index
        ):
            raise ValueError(
                f"items[{index}].transition_index must be contiguous and equal its list index"
            )
        if (
            not _is_nonnegative_int(item["beat_index"])
            or item["beat_index"] != item["transition_index"]
        ):
            raise ValueError(f"items[{index}].beat_index must equal transition_index")

        if not _is_nonnegative_int(item["source_frame_index"]):
            raise ValueError(
                f"items[{index}].source_frame_index must be a nonnegative integer"
            )
        if (
            not _is_nonnegative_int(item["beat_frame"])
            or item["beat_frame"] != item["source_frame_index"]
        ):
            raise ValueError(
                f"items[{index}].beat_frame must equal source_frame_index"
            )

        time_seconds = item["time_seconds"]
        if (
            not isinstance(time_seconds, float)
            or not math.isfinite(time_seconds)
            or time_seconds < 0
        ):
            raise ValueError(
                f"items[{index}].time_seconds must be a finite nonnegative float"
            )

        if (
            not _is_nonnegative_int(item["outfit_state_before"])
            or item["outfit_state_before"] != index
        ):
            raise ValueError(f"items[{index}].outfit_state_before must equal {index}")
        expected_after = index + 1
        if (
            not _is_nonnegative_int(item["outfit_state_after"])
            or item["outfit_state_after"] != expected_after
        ):
            raise ValueError(
                f"items[{index}].outfit_state_after must equal {expected_after}"
            )
        for alias in ("outfit_state", "outfit_state_index"):
            if (
                not _is_nonnegative_int(item[alias])
                or item[alias] != item["outfit_state_after"]
            ):
                raise ValueError(
                    f"items[{index}].{alias} must equal outfit_state_after"
                )

        if item["candidate_type"] != "outfit_image":
            raise ValueError(f"items[{index}].candidate_type must be outfit_image")
        if not _is_nonnegative_int(item["candidate_frame"]):
            raise ValueError(
                f"items[{index}].candidate_frame must be a nonnegative integer"
            )
        if (
            not _is_nonnegative_int(item["outfit_batch_index"])
            or item["outfit_batch_index"] != item["candidate_frame"]
        ):
            raise ValueError(
                f"items[{index}].outfit_batch_index must equal candidate_frame"
            )

        source_identity = item["source_identity"]
        if (
            not isinstance(source_identity, str)
            or not source_identity.strip()
            or source_identity != source_identity.strip()
            or source_identity.startswith("frame:")
        ):
            raise ValueError(
                f"items[{index}].source_identity must be a canonical nonblank identity"
            )
        if source_identity in seen_identities:
            raise ValueError(f"duplicate source_identity: {source_identity}")
        seen_identities.add(source_identity)
        outfit_path = item["outfit_path"]
        if (
            not isinstance(outfit_path, str)
            or not outfit_path.strip()
            or outfit_path != outfit_path.strip()
        ):
            raise ValueError(f"items[{index}].outfit_path must be a nonblank string")
        source_path = item["source_path"]
        if source_path is not None and (
            not isinstance(source_path, str)
            or not source_path.strip()
            or source_path != source_path.strip()
        ):
            raise ValueError(
                f"items[{index}].source_path must be a canonical nonblank string or null"
            )


def new_render_attempt(schedule: dict[str, Any]) -> dict[str, Any]:
    """Copy a render schedule for another attempt within the same run."""
    if not isinstance(schedule, dict):
        raise ValueError("render schedule must be an object")
    for field in ("plan_id", "source_video", "run_id", "plan_hash", "items"):
        if field not in schedule:
            raise ValueError(f"{field} is required")
    for field in ("plan_id", "source_video"):
        value = schedule[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonblank string")
        if value != value.strip():
            raise ValueError(f"{field} must be a canonical nonblank string")
    _validate_canonical_uuid(schedule["run_id"], "run_id")
    plan_hash = schedule["plan_hash"]
    if (
        not isinstance(plan_hash, str)
        or len(plan_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in plan_hash)
    ):
        raise ValueError("plan_hash must be exactly 64 hexadecimal characters")
    if not isinstance(schedule["items"], list):
        raise ValueError("items must be a list")
    _validate_render_items(schedule["items"])
    plan_core = {
        "plan_id": schedule["plan_id"],
        "source_video": schedule["source_video"],
        "items": schedule["items"],
    }
    expected_plan_hash = hashlib.sha256(
        _canonical_json(plan_core).encode("utf-8")
    ).hexdigest()
    if plan_hash.lower() != expected_plan_hash:
        raise ValueError("plan_hash integrity check failed")
    if "attempt_id" in schedule:
        _validate_canonical_uuid(schedule["attempt_id"], "attempt_id")
    attempt = copy.deepcopy(schedule)
    attempt["attempt_id"] = str(uuid.uuid4())
    return attempt


def normalize_analysis_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a ComfyUI PlanWriter analysis plan for rendering."""
    if not isinstance(plan, dict):
        raise ValueError("analysis plan must be an object")
    plan_id = plan.get("plan_id")
    if not isinstance(plan_id, str) or not plan_id.strip():
        raise ValueError("plan_id must be a nonblank string")
    source_video = plan.get("source_video")
    if not isinstance(source_video, str) or not source_video.strip():
        raise ValueError("source_video must be a nonblank string")
    beat_decisions = plan.get("beat_decisions")
    if not isinstance(beat_decisions, list):
        raise ValueError("beat_decisions must be a list")
    outfit_state_plan = plan.get("outfit_state_plan")
    if not isinstance(outfit_state_plan, list):
        raise ValueError("outfit_state_plan must be a list")
    if len(outfit_state_plan) != len(beat_decisions) + 1:
        raise ValueError("N beat_decisions require exactly N+1 outfit states")

    seen_identities: set[str] = set()
    states: dict[int, dict[str, Any]] = {}
    state_indices: list[Any] = []
    for position, state in enumerate(outfit_state_plan):
        if not isinstance(state, dict):
            raise ValueError(f"outfit_state_plan[{position}] must be an object")
        candidate_frame = _resolve_alias(state, "candidate_frame", "outfit_batch_index")
        if (
            not isinstance(candidate_frame, int)
            or isinstance(candidate_frame, bool)
            or candidate_frame < 0
        ):
            raise ValueError(f"outfit_state_plan[{position}].candidate_frame must be a nonnegative integer")
        outfit_path = _resolve_alias(
            state, "candidate_path", "outfit_path", strip_strings=True
        )
        if not isinstance(outfit_path, str) or not outfit_path.strip():
            raise ValueError(f"outfit_state_plan[{position}].outfit_path is required")
        source_identity = state.get("source_identity")
        if not isinstance(source_identity, str) or not source_identity.strip():
            raise ValueError(f"outfit_state_plan[{position}].source_identity is required")
        source_identity = source_identity.strip()
        if source_identity.startswith("frame:"):
            raise ValueError(f"outfit_state_plan[{position}] uses forbidden source video frame identity")
        if source_identity in seen_identities:
            raise ValueError(f"duplicate source_identity: {source_identity}")
        seen_identities.add(source_identity)
        source_path = state.get("source_path")
        if source_path is not None and not isinstance(source_path, str):
            raise ValueError(
                f"outfit_state_plan[{position}].source_path must be a string or null"
            )
        source_path = source_path.strip() or None if source_path is not None else None
        state_index = _resolve_alias(state, "outfit_state", "state_index")
        if (
            not isinstance(state_index, int)
            or isinstance(state_index, bool)
            or state_index < 0
        ):
            raise ValueError(
                f"outfit_state_plan[{position}].state_index must be a nonnegative integer"
            )
        state_indices.append(state_index)
        states[state_index] = {
            **state,
            "candidate_frame": candidate_frame,
            "outfit_batch_index": candidate_frame,
            "source_identity": source_identity,
            "source_path": source_path,
            "outfit_path": outfit_path.strip(),
        }
    if (
        any(not isinstance(index, int) or isinstance(index, bool) for index in state_indices)
        or sorted(state_indices) != list(range(len(state_indices)))
    ):
        raise ValueError("state_index values must be unique and contiguous from 0")

    schedule = []
    for position, decision in enumerate(beat_decisions):
        if not isinstance(decision, dict):
            raise ValueError(f"beat_decisions[{position}] must be an object")
        candidate_type = decision.get("candidate_type")
        if candidate_type == "source_video_frame":
            raise ValueError(f"beat_decisions[{position}] uses forbidden source_video_frame candidate")
        if candidate_type not in (None, "outfit_image"):
            raise ValueError(f"beat_decisions[{position}].candidate_type must be outfit_image")
        transition_index = _resolve_alias(decision, "transition_index", "beat_index")
        if (
            not isinstance(transition_index, int)
            or isinstance(transition_index, bool)
            or transition_index < 0
        ):
            raise ValueError(f"beat_decisions[{position}].transition_index must be a nonnegative integer")
        outfit_state_before = decision.get("outfit_state_before")
        if not isinstance(outfit_state_before, int) or isinstance(outfit_state_before, bool):
            raise ValueError(f"beat_decisions[{position}].outfit_state_before must be an integer")
        outfit_state_after = _resolve_alias(
            decision, "outfit_state_after", "outfit_state_index"
        )
        if not isinstance(outfit_state_after, int) or isinstance(outfit_state_after, bool):
            raise ValueError(f"beat_decisions[{position}].outfit_state_after must be an integer")
        if outfit_state_before not in states:
            raise ValueError(f"beat_decisions[{position}].outfit_state_before is invalid")
        if outfit_state_after not in states:
            raise ValueError(
                f"beat_decisions[{position}].outfit_state_after/outfit_state_index is invalid"
            )
        state = states[outfit_state_after]
        source_frame_index = _resolve_alias(
            decision, "source_frame_index", "beat_frame"
        )
        if (
            not isinstance(source_frame_index, int)
            or isinstance(source_frame_index, bool)
            or source_frame_index < 0
        ):
            raise ValueError(f"beat_decisions[{position}].source_frame_index must be a nonnegative integer")
        time_seconds = decision.get("time_seconds")
        if not isinstance(time_seconds, (int, float)) or isinstance(time_seconds, bool):
            raise ValueError(
                f"beat_decisions[{position}].time_seconds must be a finite nonnegative number"
            )
        try:
            time_seconds = float(time_seconds)
        except OverflowError as exc:
            raise ValueError(
                f"beat_decisions[{position}].time_seconds must be a finite nonnegative number"
            ) from exc
        if not math.isfinite(time_seconds) or time_seconds < 0:
            raise ValueError(
                f"beat_decisions[{position}].time_seconds must be a finite nonnegative number"
            )
        if time_seconds == 0.0:
            time_seconds = 0.0
        schedule.append(
            {
                "transition_index": transition_index,
                "beat_index": transition_index,
                "source_frame_index": source_frame_index,
                "beat_frame": source_frame_index,
                "time_seconds": time_seconds,
                "outfit_state_before": outfit_state_before,
                "outfit_state_after": outfit_state_after,
                "outfit_state": outfit_state_after,
                "outfit_state_index": outfit_state_after,
                "candidate_type": (
                    candidate_type if candidate_type is not None else "outfit_image"
                ),
                "candidate_frame": state.get("candidate_frame"),
                "outfit_batch_index": state.get("candidate_frame"),
                "source_identity": state["source_identity"],
                "source_path": state.get("source_path"),
                "outfit_path": state["outfit_path"],
            }
        )
    if sorted(item["transition_index"] for item in schedule) != list(range(len(schedule))):
        raise ValueError("transition_index values must be unique and contiguous from 0")
    schedule.sort(key=lambda item: (item["transition_index"], item["source_frame_index"]))
    for iteration, item in enumerate(schedule):
        transition_index = item["transition_index"]
        if item["outfit_state_before"] != transition_index:
            raise ValueError(
                f"beat_decisions transition {transition_index}.outfit_state_before "
                "must equal transition_index"
            )
        if item["outfit_state_after"] != transition_index + 1:
            raise ValueError(
                f"beat_decisions transition {transition_index}.outfit_state_after "
                "must equal transition_index + 1"
            )
        item["iteration"] = iteration
    canonical_plan = {
        "plan_id": plan_id.strip(),
        "source_video": source_video.strip(),
        "items": schedule,
    }
    return {
        **canonical_plan,
        "plan_hash": hashlib.sha256(_canonical_json(canonical_plan).encode("utf-8")).hexdigest(),
        "run_id": str(uuid.uuid4()),
        "attempt_id": str(uuid.uuid4()),
    }
