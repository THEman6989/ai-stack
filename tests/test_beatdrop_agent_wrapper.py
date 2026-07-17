from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_GRAPH = ROOT / "langgraph-app" / "agent_graph.py"


def _agent_source() -> str:
    return AGENT_GRAPH.read_text(encoding="utf-8")


def _load_sequence_wrapper(**globals_overrides):
    source = _agent_source()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "run_beatdrop_outfit_sequence"
    )
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "json": json,
        "_json_tool_result": lambda value: json.dumps(value, sort_keys=True),
        "_run_beatdrop_outfit_sequence": None,
        "VIDEO_DROP_PLANNER_IMPORT_ERROR": None,
        **globals_overrides,
    }
    exec(compile(module, str(AGENT_GRAPH), "exec"), namespace)
    return namespace["run_beatdrop_outfit_sequence"]


def test_agent_graph_imports_and_exposes_sequence_runner_tool():
    source = _agent_source()

    assert "from plugin_loader import PLUGINS_DIR as _PLUGINS_DIR" in source
    assert "_BEATDROP_PLUGIN_ROOT = Path(_PLUGINS_DIR) / \"beatdrop_outfit\"" in source
    assert "sys.path.insert(0, str(_BEATDROP_PLUGIN_ROOT))" in source
    assert "_run_beatdrop_outfit_sequence = beatdrop_runner.run_beatdrop_outfit_sequence" in source
    assert "async def run_beatdrop_outfit_sequence(" in source
    assert (
        "beatdrop_outfit_tools = [plan_video_outfit_drops, run_video_outfit_drop, "
        "run_beatdrop_outfit_sequence]"
    ) in source


def test_beatdrop_import_and_sys_path_mutation_are_inside_enable_guard():
    source = _agent_source()
    tree = ast.parse(source)
    guarded_block = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id == "_beatdrop_plugin_config_enabled"
        ),
        None,
    )

    assert guarded_block is not None
    guarded_source = ast.get_source_segment(source, guarded_block) or ""
    assert "sys.path.insert(0, str(_BEATDROP_PLUGIN_ROOT))" in guarded_source
    assert 'importlib.import_module("beatdrop_outfit.runner")' in guarded_source


def test_beatdrop_plugin_gate_reads_pluginenv_from_resolved_plugin_root():
    source = _agent_source()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_beatdrop_plugin_config_enabled"
    )
    function_source = ast.get_source_segment(source, function) or ""

    assert "_BEATDROP_PLUGIN_ROOT / \".pluginenv\"" in function_source
    assert "_WORKSPACE_ROOT / \"plugins\"" not in function_source


def test_agent_wrapper_parses_extra_parameters_and_awaits_async_runner():
    calls = []

    async def fake_runner(plan_json_or_path, **kwargs):
        calls.append((plan_json_or_path, kwargs))
        await asyncio.sleep(0)
        return {"ok": True, "counts": {"total": 2}}

    wrapper = _load_sequence_wrapper(_run_beatdrop_outfit_sequence=fake_runner)
    result = json.loads(
        asyncio.run(
            wrapper(
                "plan.json",
                workflow_name="amins_canvas_workflow",
                dry_run=True,
                extra_parameters_json='{"cfg": 4.5}',
                client_id="client-a",
                timeout_seconds=9.0,
                poll_interval_seconds=0.2,
            )
        )
    )

    assert result == {"counts": {"total": 2}, "ok": True}
    assert calls == [
        (
            "plan.json",
            {
                "workflow_name": "amins_canvas_workflow",
                "dry_run": True,
                "extra_parameters": {"cfg": 4.5},
                "client_id": "client-a",
                "timeout_seconds": 9.0,
                "poll_interval_seconds": 0.2,
            },
        )
    ]


def test_agent_wrapper_rejects_non_object_extra_parameters_before_runner_call():
    calls = []

    async def fake_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return {"ok": True}

    wrapper = _load_sequence_wrapper(_run_beatdrop_outfit_sequence=fake_runner)
    result = json.loads(
        asyncio.run(wrapper("plan.json", extra_parameters_json="[]"))
    )

    assert result == {
        "error": "extra_parameters_json must decode to a JSON object.",
        "ok": False,
    }
    assert calls == []
