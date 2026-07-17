from __future__ import annotations

import copy
import asyncio
import hashlib
import json
import math
import sys
import threading
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugins" / "beatdrop_outfit"))

import beatdrop_outfit.runner as beatdrop_runner  # noqa: E402
from beatdrop_outfit.render_contract import (  # noqa: E402
    _canonical_json,
    new_render_attempt,
    normalize_analysis_plan,
)


def _current_planwriter_plan() -> dict:
    return {
        "schema_version": "2.0",
        "plan_id": "plan-42",
        "source_video": "/media/source.mp4",
        "beat_decisions": [
            {
                "transition_index": 1,
                "outfit_state_before": 1,
                "outfit_state_after": 2,
                "source": "audio_beat",
                "source_frame_index": 90,
                "time_seconds": 3.0,
                "needs_generated_outfit_drop": True,
            },
            {
                "transition_index": 0,
                "outfit_state_before": 0,
                "outfit_state_after": 1,
                "source": "audio_beat",
                "source_frame_index": 30,
                "time_seconds": 1.0,
                "needs_generated_outfit_drop": True,
            },
        ],
        "outfit_state_plan": [
            {
                "outfit_state": 0,
                "candidate_frame": 2,
                "source_identity": "source-look",
                "candidate_path": "/media/source-look.png",
                "source_path": "/library/source-look.png",
            },
            {
                "outfit_state": 1,
                "candidate_frame": 3,
                "source_identity": "red-look",
                "candidate_path": "/media/red.png",
                "source_path": "/library/red.png",
            },
            {
                "outfit_state": 2,
                "candidate_frame": 4,
                "source_identity": "blue-look",
                "candidate_path": "/media/blue.png",
                "source_path": "/library/blue.png",
            },
        ],
    }


def _recompute_schedule_plan_hash(schedule: dict) -> None:
    core = {
        "plan_id": schedule["plan_id"],
        "source_video": schedule["source_video"],
        "items": schedule["items"],
    }
    schedule["plan_hash"] = hashlib.sha256(
        _canonical_json(core).encode("utf-8")
    ).hexdigest()


def _authoritative_history(prompt_id: str, outputs: list[dict]) -> dict:
    raw_outputs: dict[str, dict[str, list[dict]]] = {"save": {}}
    for output in outputs:
        raw = copy.deepcopy(output)
        output_type = raw.pop("output_type", "images")
        raw_outputs["save"].setdefault(output_type, []).append(raw)
    return {
        "prompt_id": prompt_id,
        "history": {prompt_id: {"outputs": raw_outputs}},
        "outputs": copy.deepcopy(outputs),
    }


def _image_output_for_key(output_key: str) -> dict:
    subfolder, filename_prefix = output_key.rsplit("/", 1)
    return {
        "output_type": "images",
        "filename": f"{filename_prefix}_00001_.png",
        "subfolder": subfolder,
        "type": "output",
    }


def test_normalize_analysis_plan_builds_deterministic_sorted_render_schedule() -> None:
    plan = _current_planwriter_plan()

    first = normalize_analysis_plan(copy.deepcopy(plan))
    second = normalize_analysis_plan(copy.deepcopy(plan))

    assert [item["transition_index"] for item in first["items"]] == [0, 1]
    assert [item["outfit_state_after"] for item in first["items"]] == [1, 2]
    assert [item["time_seconds"] for item in first["items"]] == [1.0, 3.0]
    assert [item["outfit_batch_index"] for item in first["items"]] == [3, 4]
    assert [item["outfit_path"] for item in first["items"]] == [
        "/media/red.png",
        "/media/blue.png",
    ]
    assert len(first["items"]) == len(plan["beat_decisions"])
    assert first["plan_hash"] == second["plan_hash"]
    assert len(first["plan_hash"]) == 64
    assert first["run_id"] != second["run_id"]
    assert first["attempt_id"] != second["attempt_id"]


def test_new_render_attempt_preserves_run_and_plan_without_mutating_schedule() -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    original = copy.deepcopy(schedule)

    attempt = new_render_attempt(schedule)

    assert attempt is not schedule
    assert attempt["items"] is not schedule["items"]
    for field in ("plan_id", "source_video", "items", "plan_hash", "run_id"):
        assert attempt[field] == schedule[field]
    assert attempt["attempt_id"] != schedule["attempt_id"]
    assert schedule == original


@pytest.mark.parametrize("uppercase", [False, True])
def test_new_render_attempt_rejects_stale_plan_hash_after_item_mutation(
    uppercase: bool,
) -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    if uppercase:
        schedule["plan_hash"] = schedule["plan_hash"].upper()
    schedule["items"][0]["outfit_path"] = "/media/tampered.png"

    with pytest.raises(ValueError, match="plan_hash|integrity"):
        new_render_attempt(schedule)


def test_new_render_attempt_rejects_malformed_item_with_matching_hash() -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    schedule["items"] = [{"garbage": True}]
    core = {
        "plan_id": schedule["plan_id"],
        "source_video": schedule["source_video"],
        "items": schedule["items"],
    }
    schedule["plan_hash"] = hashlib.sha256(
        _canonical_json(core).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="items|item"):
        new_render_attempt(schedule)


def test_new_render_attempt_rejects_duplicate_source_identities_with_matching_hash() -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    schedule["items"][1]["source_identity"] = schedule["items"][0][
        "source_identity"
    ]
    _recompute_schedule_plan_hash(schedule)

    with pytest.raises(ValueError, match="duplicate source_identity"):
        new_render_attempt(schedule)


@pytest.mark.parametrize("invalid_identity", ["frame:10", " red-look "])
def test_new_render_attempt_rejects_noncanonical_source_identity_with_matching_hash(
    invalid_identity: str,
) -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    schedule["items"][0]["source_identity"] = invalid_identity
    _recompute_schedule_plan_hash(schedule)

    with pytest.raises(ValueError, match=r"items\[0\].*source_identity"):
        new_render_attempt(schedule)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("outfit_path", " /media/red.png "),
        ("source_path", ""),
        ("source_path", " /library/red.png "),
    ],
)
def test_new_render_attempt_rejects_noncanonical_item_paths_with_matching_hash(
    field: str, invalid_value: str
) -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    schedule["items"][0][field] = invalid_value
    _recompute_schedule_plan_hash(schedule)

    with pytest.raises(ValueError, match=rf"items\[0\].*{field}"):
        new_render_attempt(schedule)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("iteration", 1),
        ("transition_index", 1),
        ("beat_index", 1),
        ("beat_index", False),
        ("source_frame_index", -1),
        ("source_frame_index", True),
        ("beat_frame", 31),
        ("time_seconds", -0.1),
        ("time_seconds", "1.0"),
        ("time_seconds", 1),
        ("outfit_state_before", 1),
        ("outfit_state_before", False),
        ("outfit_state_after", 2),
        ("outfit_state_after", True),
        ("outfit_state", 2),
        ("outfit_state", True),
        ("outfit_state_index", 2),
        ("outfit_state_index", True),
        ("candidate_type", "generated_outfit"),
        ("candidate_frame", -1),
        ("candidate_frame", True),
        ("outfit_batch_index", 99),
        ("source_identity", "   "),
        ("source_identity", 42),
        ("source_path", []),
        ("outfit_path", ""),
        ("outfit_path", None),
    ],
)
def test_new_render_attempt_rejects_noncanonical_item_fields_with_matching_hash(
    field: str, invalid_value: object
) -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    schedule["items"][0][field] = invalid_value
    core = {
        "plan_id": schedule["plan_id"],
        "source_video": schedule["source_video"],
        "items": schedule["items"],
    }
    schedule["plan_hash"] = hashlib.sha256(
        _canonical_json(core).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match=rf"items\[0\].*{field}"):
        new_render_attempt(schedule)


def test_new_render_attempt_rejects_non_object_schedule() -> None:
    with pytest.raises(ValueError, match="render schedule must be an object"):
        new_render_attempt([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field", ["plan_id", "source_video", "run_id", "plan_hash", "items"]
)
def test_new_render_attempt_rejects_missing_required_schedule_fields(field: str) -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    del schedule[field]

    with pytest.raises(ValueError, match=field):
        new_render_attempt(schedule)


@pytest.mark.parametrize("field", ["plan_id", "source_video", "run_id"])
@pytest.mark.parametrize("invalid_value", [None, "", "   ", 123])
def test_new_render_attempt_rejects_invalid_required_string_fields(
    field: str, invalid_value: object
) -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    schedule[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        new_render_attempt(schedule)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("plan_id", " plan-42 "), ("source_video", " /media/source.mp4 ")],
)
def test_new_render_attempt_rejects_noncanonical_root_strings_with_matching_hash(
    field: str, invalid_value: str
) -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    schedule[field] = invalid_value
    _recompute_schedule_plan_hash(schedule)

    with pytest.raises(ValueError, match=field):
        new_render_attempt(schedule)


@pytest.mark.parametrize(
    "invalid_plan_hash",
    [None, "", "a" * 63, "a" * 65, "a" * 63 + "g", "A" * 63 + "-", 123],
)
def test_new_render_attempt_rejects_invalid_plan_hash(invalid_plan_hash: object) -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    schedule["plan_hash"] = invalid_plan_hash

    with pytest.raises(ValueError, match="plan_hash"):
        new_render_attempt(schedule)


@pytest.mark.parametrize("invalid_items", [None, {}, (), "not-a-list"])
def test_new_render_attempt_rejects_invalid_items(invalid_items: object) -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    schedule["items"] = invalid_items

    with pytest.raises(ValueError, match="items"):
        new_render_attempt(schedule)


@pytest.mark.parametrize(
    "invalid_run_id",
    ["not-a-uuid", "../escape", "run/id", "00000000-0000-0000-0000-00000000000g"],
)
def test_new_render_attempt_rejects_non_uuid_run_id(invalid_run_id: str) -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    schedule["run_id"] = invalid_run_id

    with pytest.raises(ValueError, match="run_id"):
        new_render_attempt(schedule)


@pytest.mark.parametrize(
    "invalid_attempt_id",
    [None, "", "   ", 123, "not-a-uuid", "../escape", "attempt/id"],
)
def test_new_render_attempt_rejects_invalid_existing_attempt_id(
    invalid_attempt_id: object,
) -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    schedule["attempt_id"] = invalid_attempt_id

    with pytest.raises(ValueError, match="attempt_id"):
        new_render_attempt(schedule)


def test_new_render_attempt_accepts_uppercase_hash_and_missing_attempt_id() -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    schedule["plan_hash"] = schedule["plan_hash"].upper()
    del schedule["attempt_id"]
    original = copy.deepcopy(schedule)

    attempt = new_render_attempt(schedule)

    assert attempt["plan_hash"] == original["plan_hash"]
    assert attempt["attempt_id"]
    assert schedule == original


def test_new_render_attempt_accepts_empty_items_with_matching_hash() -> None:
    schedule = normalize_analysis_plan(_current_planwriter_plan())
    schedule["items"] = []
    core = {
        "plan_id": schedule["plan_id"],
        "source_video": schedule["source_video"],
        "items": schedule["items"],
    }
    schedule["plan_hash"] = hashlib.sha256(
        _canonical_json(core).encode("utf-8")
    ).hexdigest()

    attempt = new_render_attempt(schedule)

    assert attempt["items"] == []
    assert attempt["attempt_id"] != schedule["attempt_id"]


def test_normalize_analysis_plan_numbers_items_after_canonical_sort() -> None:
    result = normalize_analysis_plan(_current_planwriter_plan())

    assert [item["iteration"] for item in result["items"]] == [0, 1]
    assert [item["outfit_state"] for item in result["items"]] == [1, 2]


def test_normalize_analysis_plan_hash_ignores_semantic_input_order() -> None:
    original = _current_planwriter_plan()
    reordered = copy.deepcopy(original)
    reordered["beat_decisions"].reverse()
    reordered["outfit_state_plan"].reverse()

    first = normalize_analysis_plan(original)
    second = normalize_analysis_plan(reordered)

    assert first["items"] == second["items"]
    assert first["plan_hash"] == second["plan_hash"]


def test_canonical_json_wraps_serialization_errors() -> None:
    invalid_values = [{"value": object()}, {"value": float("nan")}]

    for value in invalid_values:
        with pytest.raises(ValueError, match="^value is not valid canonical JSON$"):
            _canonical_json(value)


@pytest.mark.parametrize("source_path", [[], {}, float("nan")])
def test_normalize_analysis_plan_rejects_invalid_source_path(source_path: object) -> None:
    plan = _current_planwriter_plan()
    plan["outfit_state_plan"][1]["source_path"] = source_path

    with pytest.raises(ValueError, match=r"outfit_state_plan\[1\]\.source_path"):
        normalize_analysis_plan(plan)


def test_normalize_analysis_plan_canonicalizes_optional_source_paths() -> None:
    plan = _current_planwriter_plan()
    plan["outfit_state_plan"][1]["source_path"] = " /library/red.png "
    plan["outfit_state_plan"][2]["source_path"] = "   "

    result = normalize_analysis_plan(plan)

    assert [item["source_path"] for item in result["items"]] == [
        "/library/red.png",
        None,
    ]


def test_build_beatdrop_render_schedule_supports_dict_json_and_opted_in_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS", "true")
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOWED_ROOTS", str(tmp_path))
    plan = _current_planwriter_plan()
    plan_path = tmp_path / "analysis-plan.json"
    plan_path.write_text(json.dumps(plan))

    schedules = [
        beatdrop_runner.build_beatdrop_render_schedule(copy.deepcopy(plan)),
        beatdrop_runner.build_beatdrop_render_schedule(json.dumps(plan)),
        beatdrop_runner.build_beatdrop_render_schedule(str(plan_path)),
    ]

    assert all(schedule["items"] == schedules[0]["items"] for schedule in schedules)
    assert all(schedule["plan_hash"] == schedules[0]["plan_hash"] for schedule in schedules)


def test_build_beatdrop_render_schedule_rejects_local_path_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS", raising=False)
    plan_path = tmp_path / "analysis-plan.json"
    plan_path.write_text(json.dumps(_current_planwriter_plan()))

    with pytest.raises(ValueError, match=r"(?i)local.*path.*disabled"):
        beatdrop_runner.build_beatdrop_render_schedule(str(plan_path))


def test_build_beatdrop_render_schedule_rejects_missing_json_path(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "definitely-missing-analysis-plan.json"

    with pytest.raises(
        (FileNotFoundError, ValueError), match=r"(?i)path|not found"
    ):
        beatdrop_runner.build_beatdrop_render_schedule(str(missing_path))


def test_build_beatdrop_render_schedule_rejects_directory_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS", "true")
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOWED_ROOTS", str(tmp_path))
    plan_directory = tmp_path / "analysis-plan.json"
    plan_directory.mkdir()

    with pytest.raises(ValueError, match=r"(?i)file|directory"):
        beatdrop_runner.build_beatdrop_render_schedule(str(plan_directory))


def test_build_beatdrop_render_schedule_rejects_path_outside_allowed_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    plan_path = outside / "analysis-plan.json"
    plan_path.write_text(json.dumps(_current_planwriter_plan()))
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS", "true")
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOWED_ROOTS", str(allowed))

    with pytest.raises(ValueError, match=r"(?i)allowed root"):
        beatdrop_runner.build_beatdrop_render_schedule(str(plan_path))


def test_legacy_run_video_outfit_drop_dry_run_mapping_is_unchanged() -> None:
    plan = {
        "source_video": "/legacy/source.mp4",
        "outfit_images": [{"id": "legacy-look", "url": "/legacy/look.png"}],
        "drops": [
            {
                "drop_id": "legacy-drop",
                "beat_frame": 120,
                "first_new_outfit_frame": 121,
                "selected_outfit_image": "legacy-look",
            }
        ],
    }

    result = beatdrop_runner.run_video_outfit_drop(
        plan, "legacy-drop", dry_run=True
    )

    assert {
        key: result["parameters"][key]
        for key in ("source_video", "reference_image", "outfit_image", "target_frame")
    } == {
        "source_video": "/legacy/source.mp4",
        "reference_image": "/legacy/look.png",
        "outfit_image": "/legacy/look.png",
        "target_frame": 121,
    }


@pytest.mark.parametrize("invalid_plan", [None, [], "not-an-object"])
def test_normalize_analysis_plan_rejects_non_object_root(invalid_plan: object) -> None:
    with pytest.raises(ValueError, match="analysis plan must be an object"):
        normalize_analysis_plan(invalid_plan)  # type: ignore[arg-type]


def test_sequence_dry_run_builds_exact_correlated_invocations_without_side_effects() -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not call injected dependencies")

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            workflow_name="amin_future_beatdrop_graph",
            dry_run=True,
            submitter=forbidden,
            history_loader=forbidden,
            attempt_store=forbidden,
        )
    )

    schedule = result["schedule"]
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["counts"] == {"total": 2, "completed": 0, "failed": 0}
    assert result["completed_records"] == []
    assert len(result["invocations"]) == 2
    for iteration, invocation in enumerate(result["invocations"]):
        item = schedule["items"][iteration]
        output_key = (
            f"beatdrop/{schedule['run_id']}/{schedule['attempt_id']}/"
            f"iteration_{iteration:04d}"
        )
        assert invocation == {
            "workflow_name": "amin_future_beatdrop_graph",
            "iteration": iteration,
            "output_key": output_key,
            "parameters": {
                "schedule_json": _canonical_json(schedule),
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
            },
        }


def test_sequence_reserved_parameters_cannot_be_overridden() -> None:
    forged = {
        "schedule_json": "forged",
        "run_id": "forged",
        "attempt_id": "forged",
        "plan_hash": "forged",
        "iteration": 999,
        "source_video": "/forged/video.mp4",
        "beat_frame": 999,
        "source_frame_index": 999,
        "outfit_image": "/forged/outfit.png",
        "reference_image": "/forged/outfit.png",
        "outfit_path": "/forged/outfit.png",
        "output_key": "forged",
        "filename_prefix": "forged",
        "custom_strength": 0.75,
    }

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(), extra_parameters=forged
        )
    )

    invocation = result["invocations"][0]
    schedule = result["schedule"]
    item = schedule["items"][0]
    parameters = invocation["parameters"]
    assert parameters["custom_strength"] == 0.75
    assert parameters["schedule_json"] == _canonical_json(schedule)
    assert parameters["run_id"] == schedule["run_id"]
    assert parameters["attempt_id"] == schedule["attempt_id"]
    assert parameters["plan_hash"] == schedule["plan_hash"]
    assert parameters["iteration"] == item["iteration"]
    assert parameters["source_video"] == schedule["source_video"]
    assert parameters["beat_frame"] == item["beat_frame"]
    assert parameters["source_frame_index"] == item["source_frame_index"]
    for field in ("outfit_image", "reference_image", "outfit_path"):
        assert parameters[field] == item["outfit_path"]
    assert parameters["output_key"] == invocation["output_key"]
    assert parameters["filename_prefix"] == invocation["output_key"]


def test_sequence_canonical_retry_preserves_run_plan_items_and_input() -> None:
    canonical = normalize_analysis_plan(_current_planwriter_plan())
    original = copy.deepcopy(canonical)

    retry = asyncio.run(beatdrop_runner.run_beatdrop_outfit_sequence(canonical))

    schedule = retry["schedule"]
    assert canonical == original
    assert schedule is not canonical
    assert schedule["items"] is not canonical["items"]
    for field in ("run_id", "plan_hash", "plan_id", "source_video", "items"):
        assert schedule[field] == canonical[field]
    assert schedule["attempt_id"] != canonical["attempt_id"]
    assert all(
        canonical["attempt_id"] not in invocation["output_key"]
        for invocation in retry["invocations"]
    )
    assert all(
        f"/{schedule['attempt_id']}/" in invocation["output_key"]
        for invocation in retry["invocations"]
    )


def test_sequence_live_happy_path_is_sequential_and_persists_lifecycle() -> None:
    events: list[tuple] = []
    persisted: list[tuple[str, str, dict]] = []
    output_keys: dict[str, str] = {}

    async def submit(workflow_name: str, parameters: dict, *, client_id: str):
        iteration = parameters["iteration"]
        events.append(("submit", iteration, workflow_name, client_id))
        prompt_id = f"prompt-{iteration}"
        output_keys[prompt_id] = parameters["output_key"]
        return {"prompt_id": prompt_id}

    async def history(prompt_id: str):
        iteration = int(prompt_id.rsplit("-", 1)[1])
        events.append(("history", iteration, prompt_id))
        return _authoritative_history(
            prompt_id,
            [_image_output_for_key(output_keys[prompt_id])],
        )

    def store(*, namespace: str, workflow_id: str, record: dict):
        snapshot = copy.deepcopy(record)
        persisted.append((namespace, workflow_id, snapshot))
        events.append(("store", snapshot["iteration"], snapshot["status"]))
        return {"saved": True, "record": snapshot}

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            workflow_name="amin_future_beatdrop_graph",
            dry_run=False,
            submitter=submit,
            history_loader=history,
            attempt_store=store,
            client_id="test-client",
            poll_interval_seconds=0,
        )
    )

    assert result["ok"] is True
    assert result["dry_run"] is False
    assert result["counts"] == {"total": 2, "completed": 2, "failed": 0}
    assert events == [
        ("store", 0, "prepared"),
        ("submit", 0, "amin_future_beatdrop_graph", "test-client"),
        ("store", 0, "submitted"),
        ("history", 0, "prompt-0"),
        ("store", 0, "completed"),
        ("store", 1, "prepared"),
        ("submit", 1, "amin_future_beatdrop_graph", "test-client"),
        ("store", 1, "submitted"),
        ("history", 1, "prompt-1"),
        ("store", 1, "completed"),
    ]
    assert [record["output"]["filename"] for record in result["completed_records"]] == [
        "iteration_0000_00001_.png",
        "iteration_0001_00001_.png",
    ]
    assert [record[2]["status"] for record in persisted] == [
        "prepared",
        "submitted",
        "completed",
        "prepared",
        "submitted",
        "completed",
    ]
    for namespace, workflow_id, record in persisted:
        assert namespace == "beatdrop_outfit_render_attempts"
        for field in (
            "run_id",
            "attempt_id",
            "plan_hash",
            "iteration",
            "output_key",
            "prompt_id",
            "status",
            "workflow_name",
        ):
            assert field in record
        for value in (
            record["run_id"],
            record["attempt_id"],
            record["plan_hash"],
            str(record["iteration"]),
            record["output_key"],
            record["workflow_name"],
        ):
            assert value in workflow_id


def test_sequence_accepts_nested_prompt_ids_with_sync_dependencies() -> None:
    saved: list[dict] = []
    output_keys: dict[str, str] = {}

    def submit(workflow_name: str, parameters: dict, *, client_id: str):
        prompt_id = f"nested-{parameters['iteration']}"
        output_keys[prompt_id] = parameters["output_key"]
        return {
            "ok": True,
            "submit_result": {"prompt_id": prompt_id},
        }

    def history(prompt_id: str):
        return _authoritative_history(
            prompt_id,
            [_image_output_for_key(output_keys[prompt_id])],
        )

    def store(*, namespace: str, workflow_id: str, record: dict):
        saved.append(copy.deepcopy(record))
        return {"saved": True}

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=submit,
            history_loader=history,
            attempt_store=store,
            poll_interval_seconds=0,
        )
    )

    assert result["ok"] is True
    assert [record["prompt_id"] for record in result["completed_records"]] == [
        "nested-0",
        "nested-1",
    ]
    assert [record["prompt_id"] for record in saved if record["status"] == "submitted"] == [
        "nested-0",
        "nested-1",
    ]


@pytest.mark.parametrize(
    ("submit_result", "expected_error"),
    [
        ({"blocked": True, "message": "submit disabled", "prompt_id": "ignore"}, "submit disabled"),
        ({"error": "submit exploded", "prompt_id": "ignore"}, "submit exploded"),
        ({}, "prompt_id"),
    ],
)
def test_sequence_submit_failures_are_persisted_and_fail_fast(
    submit_result: dict, expected_error: str
) -> None:
    saved: list[dict] = []
    submit_calls = 0

    def submit(workflow_name: str, parameters: dict, *, client_id: str):
        nonlocal submit_calls
        submit_calls += 1
        return submit_result

    def forbidden_history(prompt_id: str):
        raise AssertionError("history must not run after submit failure")

    def store(*, namespace: str, workflow_id: str, record: dict):
        saved.append(copy.deepcopy(record))
        return {"saved": True}

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=submit,
            history_loader=forbidden_history,
            attempt_store=store,
        )
    )

    assert result["ok"] is False
    assert expected_error in result["error"]
    assert result["counts"] == {"total": 2, "completed": 0, "failed": 1}
    assert submit_calls == 1
    assert [record["status"] for record in saved] == ["prepared", "failed"]
    assert saved[-1]["prompt_id"] == ""
    assert expected_error in saved[-1]["error"]


@pytest.mark.parametrize(
    ("history_mode", "expected_error"),
    [
        ("mismatch", "mismatch"),
        ("timeout", "timeout"),
        ("multiple", "exactly one"),
        ("non_image", "image"),
        ("exception", "history"),
    ],
)
def test_sequence_history_failures_are_persisted_and_fail_fast(
    history_mode: str, expected_error: str
) -> None:
    saved: list[dict] = []
    history_calls: list[str] = []

    def submit(workflow_name: str, parameters: dict, *, client_id: str):
        return {"prompt_id": "exact-prompt"}

    def history(prompt_id: str):
        history_calls.append(prompt_id)
        image = {"output_type": "images", "filename": "one.png"}
        if history_mode == "mismatch":
            return {"prompt_id": "different-prompt", "outputs": [image]}
        if history_mode == "timeout":
            return {"prompt_id": prompt_id, "outputs": []}
        if history_mode == "multiple":
            return _authoritative_history(
                prompt_id, [image, {**image, "filename": "two.png"}]
            )
        if history_mode == "non_image":
            return _authoritative_history(
                prompt_id,
                [{"output_type": "videos", "filename": "clip.mp4"}],
            )
        raise RuntimeError("history endpoint unavailable")

    def store(*, namespace: str, workflow_id: str, record: dict):
        saved.append(copy.deepcopy(record))
        return {"saved": True}

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=submit,
            history_loader=history,
            attempt_store=store,
            timeout_seconds=0,
            poll_interval_seconds=0,
        )
    )

    assert result["ok"] is False
    assert expected_error in result["error"].lower()
    assert result["counts"] == {"total": 2, "completed": 0, "failed": 1}
    assert history_calls == ["exact-prompt"]
    assert [record["status"] for record in saved] == [
        "prepared",
        "submitted",
        "failed",
    ]
    assert saved[-1]["prompt_id"] == "exact-prompt"
    assert expected_error in saved[-1]["error"].lower()


@pytest.mark.parametrize(
    ("fail_status", "expected_submit_calls", "expected_history_calls"),
    [
        ("prepared", 0, 0),
        ("submitted", 1, 0),
        ("completed", 1, 1),
    ],
)
def test_sequence_persistence_failure_blocks_unsafe_progression(
    fail_status: str, expected_submit_calls: int, expected_history_calls: int
) -> None:
    persisted_statuses: list[str] = []
    submit_calls = 0
    history_calls = 0
    output_keys: dict[str, str] = {}

    def submit(workflow_name: str, parameters: dict, *, client_id: str):
        nonlocal submit_calls
        submit_calls += 1
        output_keys["prompt-0"] = parameters["output_key"]
        return {"prompt_id": "prompt-0"}

    def history(prompt_id: str):
        nonlocal history_calls
        history_calls += 1
        return _authoritative_history(
            prompt_id,
            [_image_output_for_key(output_keys[prompt_id])],
        )

    def store(*, namespace: str, workflow_id: str, record: dict):
        persisted_statuses.append(record["status"])
        if record["status"] == fail_status:
            return {"saved": False, "error": f"cannot save {fail_status}"}
        return {"saved": True}

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=submit,
            history_loader=history,
            attempt_store=store,
            poll_interval_seconds=0,
        )
    )

    assert result["ok"] is False
    assert f"cannot save {fail_status}" in result["error"]
    assert submit_calls == expected_submit_calls
    assert history_calls == expected_history_calls
    assert persisted_statuses[-1] == fail_status
    assert "prepared" in persisted_statuses
    assert result["completed_records"] == []


def test_submitted_persistence_retries_then_requires_reconciliation() -> None:
    persisted_statuses: list[str] = []
    history_calls = 0

    def submit(workflow_name: str, parameters: dict, *, client_id: str):
        return {"prompt_id": "orphan-risk-prompt"}

    def history(prompt_id: str):
        nonlocal history_calls
        history_calls += 1
        return {"prompt_id": prompt_id, "outputs": []}

    def store(*, namespace: str, workflow_id: str, record: dict):
        persisted_statuses.append(record["status"])
        if record["status"] == "submitted":
            return {"saved": False, "error": "temporary state-store outage"}
        return {"saved": True}

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=submit,
            history_loader=history,
            attempt_store=store,
            timeout_seconds=1,
            poll_interval_seconds=0,
        )
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["requires_reconciliation"] is True
    assert result["do_not_retry"] is True
    assert result["orphaned_prompt_id"] == "orphan-risk-prompt"
    assert persisted_statuses == ["prepared", "submitted", "submitted", "submitted"]
    assert history_calls == 0


def test_sequence_unavailable_default_dependencies_are_controlled_and_pre_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module_name in (
        "comfyui_workflow_library",
        "comfyui_client",
        "run_state_manager",
    ):
        monkeypatch.setitem(sys.modules, module_name, None)

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(), dry_run=False
        )
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "unavailable" in result["error"].lower()
    assert result["completed_records"] == []
    assert result["counts"] == {"total": 2, "completed": 0, "failed": 0}


@pytest.mark.parametrize(
    ("timeout_seconds", "poll_interval_seconds"),
    [
        (-1, 0),
        (float("nan"), 0),
        (float("inf"), 0),
        (1, -1),
        (1, float("nan")),
        (1, float("inf")),
    ],
)
def test_sequence_rejects_invalid_poll_controls_before_side_effects(
    timeout_seconds: float, poll_interval_seconds: float
) -> None:
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("called")
        raise AssertionError("invalid timing must fail before dependencies")

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=forbidden,
            history_loader=forbidden,
            attempt_store=forbidden,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "timeout_seconds" in result["error"] or "poll_interval_seconds" in result["error"]
    assert calls == []


def test_sequence_rejects_nested_submit_error_even_when_prompt_id_exists() -> None:
    saved: list[dict] = []

    def submit(workflow_name: str, parameters: dict, *, client_id: str):
        return {
            "ok": True,
            "submit_result": {
                "ok": False,
                "message": "provider rejected",
                "prompt_id": "must-not-run",
            },
        }

    def forbidden_history(prompt_id: str):
        raise AssertionError("history must not run after nested submit failure")

    def store(*, namespace: str, workflow_id: str, record: dict):
        saved.append(copy.deepcopy(record))
        return {"saved": True}

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=submit,
            history_loader=forbidden_history,
            attempt_store=store,
        )
    )

    assert result["ok"] is False
    assert "provider rejected" in result["error"]
    assert [record["status"] for record in saved] == ["prepared", "failed"]
    assert saved[-1]["prompt_id"] == ""


@pytest.mark.parametrize(
    "history_mode",
    ["missing_prompt_id", "wrong_raw_key", "missing_raw_history", "empty_raw_history"],
)
def test_sequence_requires_exact_history_prompt_correlation(history_mode: str) -> None:
    saved: list[dict] = []

    def submit(workflow_name: str, parameters: dict, *, client_id: str):
        return {"prompt_id": "expected-prompt"}

    def history(prompt_id: str):
        image = {"output_type": "images", "filename": "wrong.png"}
        if history_mode == "missing_prompt_id":
            return {"outputs": [image]}
        if history_mode == "missing_raw_history":
            return {"prompt_id": prompt_id, "outputs": [image]}
        if history_mode == "empty_raw_history":
            return {"prompt_id": prompt_id, "history": {}, "outputs": [image]}
        return {
            "prompt_id": prompt_id,
            "history": {"different-prompt": {"outputs": {}}},
            "outputs": [image],
        }

    def store(*, namespace: str, workflow_id: str, record: dict):
        saved.append(copy.deepcopy(record))
        return {"saved": True}

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=submit,
            history_loader=history,
            attempt_store=store,
            timeout_seconds=0.1,
            poll_interval_seconds=0,
        )
    )

    assert result["ok"] is False
    assert "prompt_id" in result["error"] or "history key" in result["error"]
    assert [record["status"] for record in saved] == [
        "prepared",
        "submitted",
        "failed",
    ]


def test_sequence_rejects_authoritative_image_without_filename() -> None:
    saved: list[dict] = []

    def submit(workflow_name: str, parameters: dict, *, client_id: str):
        return {"prompt_id": "blank-image-prompt"}

    def history(prompt_id: str):
        return _authoritative_history(
            prompt_id,
            [{"output_type": "images", "filename": "   ", "type": "output"}],
        )

    def store(*, namespace: str, workflow_id: str, record: dict):
        saved.append(copy.deepcopy(record))
        return {"saved": True}

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=submit,
            history_loader=history,
            attempt_store=store,
            timeout_seconds=0.1,
            poll_interval_seconds=0,
        )
    )

    assert result["ok"] is False
    assert "filename" in result["error"]
    assert saved[-1]["status"] == "failed"


def test_sequence_fails_immediately_on_terminal_comfyui_execution_error() -> None:
    saved: list[dict] = []
    history_calls = 0

    def submit(workflow_name: str, parameters: dict, *, client_id: str):
        return {"prompt_id": "prompt-oom"}

    def history(prompt_id: str):
        nonlocal history_calls
        history_calls += 1
        return {
            "prompt_id": prompt_id,
            "history": {
                prompt_id: {
                    "status": {
                        "status_str": "error",
                        "completed": False,
                        "messages": [
                            [
                                "execution_error",
                                {"exception_message": "CUDA OOM"},
                            ]
                        ],
                    }
                }
            },
            "outputs": [],
        }

    def store(*, namespace: str, workflow_id: str, record: dict):
        saved.append(copy.deepcopy(record))
        return {"saved": True}

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=submit,
            history_loader=history,
            attempt_store=store,
            timeout_seconds=1,
            poll_interval_seconds=0,
        )
    )

    assert result["ok"] is False
    assert "CUDA OOM" in result["error"]
    assert history_calls == 1
    assert saved[-1]["status"] == "failed"


def test_sequence_rejects_output_not_attributed_to_invocation_prefix_or_node() -> None:
    def submit(workflow_name: str, parameters: dict, *, client_id: str):
        return {"prompt_id": f"prompt-{parameters['iteration']}"}

    def history(prompt_id: str):
        return _authoritative_history(
            prompt_id,
            [{"output_type": "images", "filename": "unrelated.png"}],
        )

    def store(*, namespace: str, workflow_id: str, record: dict):
        return {"saved": True}

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=submit,
            history_loader=history,
            attempt_store=store,
            poll_interval_seconds=0,
        )
    )

    assert result["ok"] is False
    assert "output_key" in result["error"]
    assert result["failed_record"]["status"] == "failed"


def test_sequence_deadline_bounds_slow_sync_history_and_long_poll_interval() -> None:
    def submit(workflow_name: str, parameters: dict, *, client_id: str):
        return {"prompt_id": "slow-prompt"}

    def slow_history(prompt_id: str):
        time.sleep(1)
        return {"prompt_id": prompt_id, "outputs": []}

    def store(*, namespace: str, workflow_id: str, record: dict):
        return {"saved": True}

    started = time.monotonic()
    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=submit,
            history_loader=slow_history,
            attempt_store=store,
            timeout_seconds=0.02,
            poll_interval_seconds=10,
        )
    )
    elapsed = time.monotonic() - started

    assert result["ok"] is False
    assert "timeout" in result["error"]
    assert elapsed < 0.25


def test_sequence_reconciles_slow_sync_submitter_instead_of_false_timeout() -> None:
    saved: list[dict] = []
    output_keys: dict[str, str] = {}

    def slow_submit(workflow_name: str, parameters: dict, *, client_id: str):
        time.sleep(0.06)
        prompt_id = f"reconciled-{parameters['iteration']}"
        output_keys[prompt_id] = parameters["output_key"]
        return {"prompt_id": prompt_id}

    def history(prompt_id: str):
        return _authoritative_history(
            prompt_id,
            [_image_output_for_key(output_keys[prompt_id])],
        )

    def store(*, namespace: str, workflow_id: str, record: dict):
        saved.append(copy.deepcopy(record))
        return {"saved": True}

    started = time.monotonic()
    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=slow_submit,
            history_loader=history,
            attempt_store=store,
            timeout_seconds=0.01,
            poll_interval_seconds=0,
        )
    )
    elapsed = time.monotonic() - started

    assert result["ok"] is True
    assert elapsed >= 0.1
    submitted = [record for record in saved if record["status"] == "submitted"]
    assert len(submitted) == 2
    assert all(record["submission_reconciled_after_timeout"] is True for record in submitted)


def test_sequence_does_not_return_while_timed_out_sync_submitter_can_submit_later() -> None:
    submit_finished = threading.Event()
    output_keys: dict[str, str] = {}

    def slow_submit(workflow_name: str, parameters: dict, *, client_id: str):
        time.sleep(0.06)
        prompt_id = f"tracked-{parameters['iteration']}"
        output_keys[prompt_id] = parameters["output_key"]
        submit_finished.set()
        return {"prompt_id": prompt_id}

    def history(prompt_id: str):
        return _authoritative_history(
            prompt_id,
            [_image_output_for_key(output_keys[prompt_id])],
        )

    def store(*, namespace: str, workflow_id: str, record: dict):
        return {"saved": True}

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=slow_submit,
            history_loader=history,
            attempt_store=store,
            timeout_seconds=0.01,
            poll_interval_seconds=0,
        )
    )

    assert submit_finished.is_set()
    assert result["ok"] is True
    assert result["counts"]["completed"] == 2


def test_sequence_rejects_non_callable_dependencies_before_side_effects() -> None:
    submit_calls = 0

    def submit(workflow_name: str, parameters: dict, *, client_id: str):
        nonlocal submit_calls
        submit_calls += 1
        return {"prompt_id": "must-not-submit"}

    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            dry_run=False,
            submitter=submit,
            history_loader=object(),
            attempt_store=object(),
        )
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "callable" in result["error"]
    assert submit_calls == 0


def test_sequence_normalizes_workflow_name_before_building_persistent_ids() -> None:
    result = asyncio.run(
        beatdrop_runner.run_beatdrop_outfit_sequence(
            _current_planwriter_plan(),
            workflow_name="  Amin's canvas / v1  ",
            dry_run=True,
        )
    )

    assert result["workflow_name"] == "Amin_s_canvas_v1"
    assert {item["workflow_name"] for item in result["invocations"]} == {
        "Amin_s_canvas_v1"
    }


@pytest.mark.parametrize("workflow_name", ["", "   ", "x" * 65])
def test_sequence_rejects_invalid_normalized_workflow_name(workflow_name: str) -> None:
    with pytest.raises(ValueError, match="workflow_name"):
        asyncio.run(
            beatdrop_runner.run_beatdrop_outfit_sequence(
                _current_planwriter_plan(),
                workflow_name=workflow_name,
                dry_run=True,
            )
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("plan_id", "   "),
        ("source_video", None),
        ("beat_decisions", {}),
        ("outfit_state_plan", ()),
    ],
)
def test_normalize_analysis_plan_rejects_invalid_planwriter_root_fields(
    field: str, invalid_value: object
) -> None:
    plan = _current_planwriter_plan()
    plan[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        normalize_analysis_plan(plan)


def test_normalize_analysis_plan_rejects_non_object_outfit_state() -> None:
    plan = _current_planwriter_plan()
    plan["outfit_state_plan"][1] = "not-an-object"

    with pytest.raises(ValueError, match=r"outfit_state_plan\[1\].*object"):
        normalize_analysis_plan(plan)


@pytest.mark.parametrize("candidate_frame", [None, True, -1, 1.5])
def test_normalize_analysis_plan_rejects_invalid_candidate_frame(candidate_frame: object) -> None:
    plan = _current_planwriter_plan()
    plan["outfit_state_plan"][1]["candidate_frame"] = candidate_frame

    with pytest.raises(ValueError, match="candidate_frame"):
        normalize_analysis_plan(plan)


@pytest.mark.parametrize("missing_value", [None, "", "   "])
def test_normalize_analysis_plan_rejects_missing_outfit_paths(missing_value: str | None) -> None:
    plan = _current_planwriter_plan()
    if missing_value is None:
        del plan["outfit_state_plan"][1]["candidate_path"]
    else:
        plan["outfit_state_plan"][1]["candidate_path"] = missing_value

    with pytest.raises(ValueError, match="outfit_path"):
        normalize_analysis_plan(plan)


def test_normalize_analysis_plan_rejects_duplicate_source_identities() -> None:
    plan = _current_planwriter_plan()
    plan["outfit_state_plan"][2]["source_identity"] = "red-look"

    with pytest.raises(ValueError, match="duplicate source_identity"):
        normalize_analysis_plan(plan)


def test_normalize_analysis_plan_rejects_padded_duplicate_source_identities() -> None:
    plan = _current_planwriter_plan()
    plan["outfit_state_plan"][2]["source_identity"] = " red-look "

    with pytest.raises(ValueError, match="duplicate source_identity"):
        normalize_analysis_plan(plan)


@pytest.mark.parametrize(
    ("collection", "index", "canonical_name", "legacy_name", "legacy_value"),
    [
        ("outfit_state_plan", 1, "candidate_frame", "outfit_batch_index", 4),
        ("outfit_state_plan", 1, "candidate_path", "outfit_path", "/media/blue.png"),
        ("outfit_state_plan", 1, "outfit_state", "state_index", 2),
        ("beat_decisions", 0, "transition_index", "beat_index", 0),
        ("beat_decisions", 0, "outfit_state_after", "outfit_state_index", 1),
        ("beat_decisions", 0, "source_frame_index", "beat_frame", 91),
    ],
)
def test_normalize_analysis_plan_rejects_conflicting_aliases(
    collection: str,
    index: int,
    canonical_name: str,
    legacy_name: str,
    legacy_value: object,
) -> None:
    plan = _current_planwriter_plan()
    plan[collection][index][legacy_name] = legacy_value

    with pytest.raises(ValueError) as exc_info:
        normalize_analysis_plan(plan)

    message = str(exc_info.value)
    assert "conflict" in message
    assert canonical_name in message
    assert legacy_name in message


@pytest.mark.parametrize(
    (
        "collection",
        "index",
        "canonical_name",
        "legacy_name",
        "canonical_value",
        "legacy_value",
    ),
    [
        ("outfit_state_plan", 0, "candidate_frame", "outfit_batch_index", 1, True),
        ("outfit_state_plan", 1, "outfit_state", "state_index", 1, True),
        ("beat_decisions", 0, "transition_index", "beat_index", 1, True),
        (
            "beat_decisions",
            1,
            "outfit_state_after",
            "outfit_state_index",
            1,
            True,
        ),
        ("beat_decisions", 1, "source_frame_index", "beat_frame", 1, True),
        ("outfit_state_plan", 1, "candidate_frame", "outfit_batch_index", 3, 3.0),
        ("outfit_state_plan", 1, "outfit_state", "state_index", 1, 1.0),
        ("beat_decisions", 0, "transition_index", "beat_index", 1, 1.0),
        (
            "beat_decisions",
            1,
            "outfit_state_after",
            "outfit_state_index",
            1,
            1.0,
        ),
        ("beat_decisions", 0, "source_frame_index", "beat_frame", 90, 90.0),
    ],
)
def test_normalize_analysis_plan_rejects_type_confused_integer_aliases(
    collection: str,
    index: int,
    canonical_name: str,
    legacy_name: str,
    canonical_value: int,
    legacy_value: bool | float,
) -> None:
    plan = _current_planwriter_plan()
    plan[collection][index][canonical_name] = canonical_value
    plan[collection][index][legacy_name] = legacy_value

    with pytest.raises(ValueError) as exc_info:
        normalize_analysis_plan(plan)

    message = str(exc_info.value)
    assert "conflict" in message
    assert canonical_name in message
    assert legacy_name in message


@pytest.mark.parametrize(
    ("collection", "index", "legacy_name", "legacy_value"),
    [
        ("outfit_state_plan", 1, "outfit_batch_index", 3),
        ("outfit_state_plan", 1, "outfit_path", " /media/red.png "),
        ("outfit_state_plan", 1, "state_index", 1),
        ("beat_decisions", 0, "beat_index", 1),
        ("beat_decisions", 0, "outfit_state_index", 2),
        ("beat_decisions", 0, "beat_frame", 90),
    ],
)
def test_normalize_analysis_plan_accepts_equal_aliases(
    collection: str,
    index: int,
    legacy_name: str,
    legacy_value: object,
) -> None:
    baseline_plan = _current_planwriter_plan()
    aliased_plan = copy.deepcopy(baseline_plan)
    aliased_plan[collection][index][legacy_name] = legacy_value

    baseline = normalize_analysis_plan(baseline_plan)
    aliased = normalize_analysis_plan(aliased_plan)

    assert aliased["items"] == baseline["items"]
    assert aliased["plan_hash"] == baseline["plan_hash"]


def test_normalize_analysis_plan_requires_one_more_state_than_decisions() -> None:
    plan = _current_planwriter_plan()
    plan["outfit_state_plan"].append(
        {
            "outfit_state": 3,
            "candidate_frame": 5,
            "source_identity": "green-look",
            "candidate_path": "/media/green.png",
            "source_path": "/library/green.png",
        }
    )

    with pytest.raises(ValueError, match=r"N\+1.*states"):
        normalize_analysis_plan(plan)


@pytest.mark.parametrize("state_index", [None, True, -1, 1.5, [], {}])
def test_normalize_analysis_plan_rejects_invalid_state_index(state_index: object) -> None:
    plan = _current_planwriter_plan()
    plan["outfit_state_plan"][1]["outfit_state"] = state_index

    with pytest.raises(ValueError, match="state_index"):
        normalize_analysis_plan(plan)


@pytest.mark.parametrize("state_indices", [[0, 2, 3], [0, 1, 1], [1, 2, 3]])
def test_normalize_analysis_plan_rejects_non_contiguous_state_indices(state_indices: list[int]) -> None:
    plan = _current_planwriter_plan()
    for state, state_index in zip(plan["outfit_state_plan"], state_indices, strict=True):
        state["outfit_state"] = state_index

    with pytest.raises(ValueError, match="state_index.*contiguous"):
        normalize_analysis_plan(plan)


def test_normalize_analysis_plan_treats_null_candidate_type_as_default() -> None:
    omitted_plan = _current_planwriter_plan()
    explicit_null_plan = copy.deepcopy(omitted_plan)
    explicit_null_plan["beat_decisions"][0]["candidate_type"] = None

    omitted = normalize_analysis_plan(omitted_plan)
    explicit_null = normalize_analysis_plan(explicit_null_plan)

    assert all(item["candidate_type"] == "outfit_image" for item in omitted["items"])
    assert all(item["candidate_type"] == "outfit_image" for item in explicit_null["items"])
    assert explicit_null["items"] == omitted["items"]
    assert explicit_null["plan_hash"] == omitted["plan_hash"]


def test_normalize_analysis_plan_rejects_source_video_frame_candidates() -> None:
    plan = _current_planwriter_plan()
    plan["beat_decisions"][0]["candidate_type"] = "source_video_frame"

    with pytest.raises(ValueError, match="source_video_frame"):
        normalize_analysis_plan(plan)


def test_normalize_analysis_plan_rejects_unknown_candidate_type() -> None:
    plan = _current_planwriter_plan()
    plan["beat_decisions"][0]["candidate_type"] = "generated_outfit"

    with pytest.raises(ValueError, match="candidate_type"):
        normalize_analysis_plan(plan)


def test_normalize_analysis_plan_rejects_planwriter_frame_fallback_identity() -> None:
    plan = _current_planwriter_plan()
    plan["outfit_state_plan"][2]["source_identity"] = "frame:4"

    with pytest.raises(ValueError, match="source.video.frame"):
        normalize_analysis_plan(plan)


def test_normalize_analysis_plan_rejects_padded_frame_fallback_identity() -> None:
    plan = _current_planwriter_plan()
    plan["outfit_state_plan"][2]["source_identity"] = " frame:4 "

    with pytest.raises(ValueError, match="source.video.frame"):
        normalize_analysis_plan(plan)


def test_normalize_analysis_plan_rejects_non_object_beat_decision() -> None:
    plan = _current_planwriter_plan()
    plan["beat_decisions"][0] = "not-an-object"

    with pytest.raises(ValueError, match=r"beat_decisions\[0\].*object"):
        normalize_analysis_plan(plan)


@pytest.mark.parametrize("transition_index", [None, True, -1, 0.5])
def test_normalize_analysis_plan_rejects_invalid_transition_index(
    transition_index: object,
) -> None:
    plan = _current_planwriter_plan()
    plan["beat_decisions"][0]["transition_index"] = transition_index

    with pytest.raises(ValueError, match="transition_index"):
        normalize_analysis_plan(plan)


@pytest.mark.parametrize("source_frame_index", [None, True, -1, 0.5])
def test_normalize_analysis_plan_rejects_invalid_source_frame_index(
    source_frame_index: object,
) -> None:
    plan = _current_planwriter_plan()
    plan["beat_decisions"][0]["source_frame_index"] = source_frame_index

    with pytest.raises(ValueError, match="source_frame_index"):
        normalize_analysis_plan(plan)


@pytest.mark.parametrize(
    "time_seconds", [None, True, -0.1, float("nan"), float("inf"), float("-inf")]
)
def test_normalize_analysis_plan_rejects_invalid_time_seconds(time_seconds: object) -> None:
    plan = _current_planwriter_plan()
    plan["beat_decisions"][0]["time_seconds"] = time_seconds

    with pytest.raises(ValueError, match="time_seconds"):
        normalize_analysis_plan(plan)


def test_normalize_analysis_plan_rejects_overflowing_time_seconds() -> None:
    plan = _current_planwriter_plan()
    plan["beat_decisions"][0]["time_seconds"] = 10**400

    with pytest.raises(ValueError, match="time_seconds"):
        normalize_analysis_plan(plan)


@pytest.mark.parametrize(
    ("first_value", "second_value", "expected"),
    [(1, 1.0, 1.0), (-0.0, 0.0, 0.0)],
)
def test_normalize_analysis_plan_canonicalizes_numeric_timestamps(
    first_value: int | float,
    second_value: float,
    expected: float,
) -> None:
    first_plan = _current_planwriter_plan()
    second_plan = copy.deepcopy(first_plan)
    first_plan["beat_decisions"][1]["time_seconds"] = first_value
    second_plan["beat_decisions"][1]["time_seconds"] = second_value

    first = normalize_analysis_plan(first_plan)
    second = normalize_analysis_plan(second_plan)
    first_time = first["items"][0]["time_seconds"]
    second_time = second["items"][0]["time_seconds"]

    assert isinstance(first_time, float)
    assert isinstance(second_time, float)
    assert first_time == second_time == expected
    if expected == 0.0:
        assert math.copysign(1.0, first_time) == 1.0
        assert math.copysign(1.0, second_time) == 1.0
    assert first["items"] == second["items"]
    assert first["plan_hash"] == second["plan_hash"]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("outfit_state_before", None),
        ("outfit_state_before", True),
        ("outfit_state_before", 0.5),
        ("outfit_state_after", None),
        ("outfit_state_after", True),
        ("outfit_state_after", 0.5),
    ],
)
def test_normalize_analysis_plan_requires_integer_decision_states(
    field: str, invalid_value: object
) -> None:
    plan = _current_planwriter_plan()
    if invalid_value is None:
        del plan["beat_decisions"][0][field]
    else:
        plan["beat_decisions"][0][field] = invalid_value

    with pytest.raises(ValueError, match=field):
        normalize_analysis_plan(plan)


@pytest.mark.parametrize("field", ["outfit_state_before", "outfit_state_after"])
def test_normalize_analysis_plan_requires_decision_states_to_exist(field: str) -> None:
    plan = _current_planwriter_plan()
    plan["beat_decisions"][0][field] = 99

    with pytest.raises(ValueError, match=field):
        normalize_analysis_plan(plan)


@pytest.mark.parametrize(
    ("field", "wrong_state"),
    [("outfit_state_before", 0), ("outfit_state_after", 1)],
)
def test_normalize_analysis_plan_requires_canonical_decision_state_sequence(
    field: str, wrong_state: int
) -> None:
    plan = _current_planwriter_plan()
    plan["beat_decisions"][0][field] = wrong_state

    with pytest.raises(ValueError, match=field):
        normalize_analysis_plan(plan)


@pytest.mark.parametrize("transition_indices", [[0, 0], [0, 2]])
def test_normalize_analysis_plan_rejects_non_contiguous_transition_indices(
    transition_indices: list[int],
) -> None:
    plan = _current_planwriter_plan()
    for decision, transition_index in zip(
        plan["beat_decisions"], transition_indices, strict=True
    ):
        decision["transition_index"] = transition_index

    with pytest.raises(ValueError, match="transition_index.*contiguous"):
        normalize_analysis_plan(plan)


def test_normalize_analysis_plan_rejects_beat_decision_with_unknown_outfit_state() -> None:
    plan = _current_planwriter_plan()
    plan["beat_decisions"][0]["outfit_state_after"] = 99

    with pytest.raises(ValueError, match="outfit_state_index"):
        normalize_analysis_plan(plan)
