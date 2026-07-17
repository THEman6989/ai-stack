"""Tests for the plugin loader module."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "langgraph-app"))

from plugin_loader import (  # noqa: E402
    PluginManifest,
    _env_bool,
    _resolve_plugins_dir,
    _plugin_is_enabled,
    _load_manifest,
    load_plugins,
    load_plugin_python_tools,
    merge_mcp_servers,
    merge_toolsets,
    get_merged_mcp_config,
)


# ── helpers ────────────────────────────────────────────────────────


def test_resolve_plugins_dir_falls_back_to_workspace_mount(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPHARAVIS_PLUGINS_DIR", raising=False)
    module_file = tmp_path / "app" / "plugin_loader.py"
    workspace_root = tmp_path / "workspace"
    expected = workspace_root / "plugins"
    expected.mkdir(parents=True)

    assert (
        _resolve_plugins_dir(
            module_file=module_file,
            workspace_root=workspace_root,
        )
        == expected
    )


def test_resolve_plugins_dir_prefers_explicit_environment(tmp_path, monkeypatch):
    explicit = tmp_path / "custom-plugins"
    monkeypatch.setenv("ALPHARAVIS_PLUGINS_DIR", str(explicit))

    assert _resolve_plugins_dir() == explicit

def _make_plugin(base: Path, name: str, yaml_content: str, *, enabled: bool = False) -> Path:
    """Create a plugin directory with plugin.yaml and .pluginenv."""
    plugin_dir = base / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(yaml_content)
    (plugin_dir / ".pluginenv").write_text(f"ENABLED={'true' if enabled else 'false'}\n")
    return plugin_dir


def test_langgraph_container_exposes_safe_plugin_system_gate():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert (
        "ALPHARAVIS_ENABLE_PLUGIN_SYSTEM=${ALPHARAVIS_ENABLE_PLUGIN_SYSTEM:-false}"
        in compose
    )


# ── _env_bool ──────────────────────────────────────────────────────

def test_env_bool_true_values():
    for val in ("1", "true", "yes", "on", "TRUE", "YES", "ON"):
        with patch.dict(os.environ, {"TEST_KEY": val}):
            assert _env_bool("TEST_KEY") is True


def test_env_bool_false_values():
    for val in ("0", "false", "no", "off", "", "garbage"):
        with patch.dict(os.environ, {"TEST_KEY": val}):
            assert _env_bool("TEST_KEY") is False


def test_env_bool_missing_uses_default():
    with patch.dict(os.environ, {}, clear=True):
        assert _env_bool("MISSING", True) is True
        assert _env_bool("MISSING", False) is False
        assert _env_bool("MISSING") is False  # default default


# ── _plugin_is_enabled ─────────────────────────────────────────────

def test_pluginenv_missing_disabled():
    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp) / "p"
        plugin_dir.mkdir()
        assert _plugin_is_enabled(plugin_dir) is False


def test_pluginenv_enabled_false():
    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp) / "p"
        plugin_dir.mkdir()
        (plugin_dir / ".pluginenv").write_text("ENABLED=false\n")
        assert _plugin_is_enabled(plugin_dir) is False


def test_pluginenv_enabled_true():
    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp) / "p"
        plugin_dir.mkdir()
        (plugin_dir / ".pluginenv").write_text("ENABLED=true\n")
        assert _plugin_is_enabled(plugin_dir) is True


def test_pluginenv_enabled_1():
    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp) / "p"
        plugin_dir.mkdir()
        (plugin_dir / ".pluginenv").write_text("ENABLED=1\n")
        assert _plugin_is_enabled(plugin_dir) is True


def test_pluginenv_with_comments_and_whitespace():
    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp) / "p"
        plugin_dir.mkdir()
        (plugin_dir / ".pluginenv").write_text(
            "# Plugin config\nENABLED = true\n# end\n"
        )
        assert _plugin_is_enabled(plugin_dir) is True


def test_pluginenv_case_insensitive():
    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp) / "p"
        plugin_dir.mkdir()
        (plugin_dir / ".pluginenv").write_text("enabled=TRUE\n")
        assert _plugin_is_enabled(plugin_dir) is True


def test_pluginenv_corrupt_file_returns_false():
    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp) / "p"
        plugin_dir.mkdir()
        (plugin_dir / ".pluginenv").write_text("garbage\nno equals sign\n")
        assert _plugin_is_enabled(plugin_dir) is False


# ── _load_manifest ─────────────────────────────────────────────────

def test_load_valid_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_plugin(Path(tmp), "test-plugin", """
name: test-plugin
version: 1.0.0
description: A test plugin
mcp_servers:
  test-srv:
    type: sse
    url: http://localhost:9999/sse
toolsets:
  test/hello:
    description: hello toolset
    tools:
      - say_hello
    mcp_categories:
      - test
""")
        manifest = _load_manifest(d)

    assert manifest is not None
    assert manifest.name == "test-plugin"
    assert manifest.version == "1.0.0"
    assert manifest.description == "A test plugin"
    assert "test-srv" in manifest.mcp_servers
    assert "test/hello" in manifest.toolsets


def test_load_manifest_with_all_fields():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_plugin(Path(tmp), "full-plugin", """
name: full-plugin
version: 2.0.0
description: Full featured
author: test-author
depends_on:
  - base-plugin
mcp_servers:
  srv1:
    type: sse
    url: http://srv/sse
toolsets:
  full/tools:
    description: tools
    tools: [a, b]
    mcp_categories: [cat1]
docker_compose:
  services:
    srv1:
      image: alpine
skills:
  - path: skills/s1.md
env_defaults:
  KEY: val
""")
        manifest = _load_manifest(d)

    assert manifest is not None
    assert manifest.name == "full-plugin"
    assert manifest.version == "2.0.0"
    assert manifest.author == "test-author"
    assert manifest.depends_on == ["base-plugin"]
    assert len(manifest.mcp_servers) == 1
    assert len(manifest.toolsets) == 1
    assert manifest.docker_compose == {"services": {"srv1": {"image": "alpine"}}}
    assert manifest.skills == [{"path": "skills/s1.md"}]
    assert manifest.env_defaults == {"KEY": "val"}
    # No enabled_env / enabled_default — removed


def test_missing_yaml_manifest_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _load_manifest(Path(tmp))
    assert manifest is None


def test_missing_name_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_plugin(Path(tmp), "bad-plugin", "version: 1.0.0")
        manifest = _load_manifest(d)
    assert manifest is None


def test_missing_version_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_plugin(Path(tmp), "bad-plugin", "name: bad-plugin")
        manifest = _load_manifest(d)
    assert manifest is None


def test_empty_yaml_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_plugin(Path(tmp), "empty", "")
        manifest = _load_manifest(d)
    assert manifest is None


def test_minimal_manifest_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_plugin(Path(tmp), "minimal", "name: minimal\nversion: 0.1.0")
        manifest = _load_manifest(d)

    assert manifest is not None
    assert manifest.description == ""
    assert manifest.author == ""
    assert manifest.depends_on == []
    assert manifest.mcp_servers == {}
    assert manifest.toolsets == {}
    assert manifest.docker_compose == {}
    assert manifest.skills == []
    assert manifest.env_defaults == {}
    assert manifest.python_tools == {}


def test_load_manifest_parses_python_tools():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_plugin(Path(tmp), "pt",
            "name: pt\n"
            "version: 1.0\n"
            "python_tools:\n"
            "  pt.tools:\n"
            "    import_names:\n"
            "      - tool_a\n"
            "      - tool_b\n"
        )
        manifest = _load_manifest(d)

    assert manifest is not None
    assert manifest.python_tools == {"pt.tools": {"import_names": ["tool_a", "tool_b"]}}


def test_preinstalled_beatdrop_outfit_plugin_is_disabled_by_default():
    plugin_dir = ROOT / "plugins" / "beatdrop_outfit"

    manifest = _load_manifest(plugin_dir)

    assert manifest is not None
    assert manifest.name == "beatdrop-outfit"
    assert manifest.toolsets["media/beatdrop-outfit"]["tools"] == [
        "plan_video_outfit_drops",
        "run_video_outfit_drop",
        "run_beatdrop_outfit_sequence",
    ]
    assert manifest.python_tools["beatdrop_outfit.tools"]["import_names"] == [
        "sort_outfits",
        "plan_video_outfit_drops",
        "run_video_outfit_drop",
        "run_beatdrop_outfit_sequence",
    ]
    assert _plugin_is_enabled(plugin_dir) is False


# ── load_plugins ───────────────────────────────────────────────────

def test_no_plugins_when_system_flag_off():
    with tempfile.TemporaryDirectory() as tmp:
        _make_plugin(Path(tmp), "p1", "name: p1\nversion: 1.0", enabled=True)
        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "false"}):
            plugins = load_plugins(plugins_dir=str(tmp))
    assert plugins == []


def test_no_plugins_when_system_flag_missing():
    with tempfile.TemporaryDirectory() as tmp:
        _make_plugin(Path(tmp), "p1", "name: p1\nversion: 1.0", enabled=True)
        with patch.dict(os.environ, {}, clear=True):
            plugins = load_plugins(plugins_dir=str(tmp))
    assert plugins == []


def test_disabled_plugin_not_loaded():
    with tempfile.TemporaryDirectory() as tmp:
        _make_plugin(Path(tmp), "p1", "name: p1\nversion: 1.0", enabled=False)
        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "true"}):
            plugins = load_plugins(plugins_dir=str(tmp))
    assert plugins == []


def test_loads_enabled_plugins():
    with tempfile.TemporaryDirectory() as tmp:
        _make_plugin(Path(tmp), "p1", "name: p1\nversion: 1.0", enabled=True)
        _make_plugin(Path(tmp), "p2", "name: p2\nversion: 1.0", enabled=True)
        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "true"}):
            plugins = load_plugins(plugins_dir=str(tmp))
    assert len(plugins) == 2
    names = {p.name for p in plugins}
    assert names == {"p1", "p2"}


def test_missing_plugins_dir_returns_empty():
    with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "true"}):
        plugins = load_plugins(plugins_dir="/nonexistent/path")
    assert plugins == []


def test_skips_dot_prefix_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        _make_plugin(Path(tmp), ".hidden", "name: hidden\nversion: 1.0", enabled=True)
        _make_plugin(Path(tmp), "visible", "name: visible\nversion: 1.0", enabled=True)
        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "true"}):
            plugins = load_plugins(plugins_dir=str(tmp))
    assert len(plugins) == 1
    assert plugins[0].name == "visible"


def test_skips_underscore_prefix_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        _make_plugin(Path(tmp), "_example", "name: example\nversion: 1.0", enabled=True)
        _make_plugin(Path(tmp), "real", "name: real\nversion: 1.0", enabled=True)
        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "true"}):
            plugins = load_plugins(plugins_dir=str(tmp))
    assert len(plugins) == 1
    assert plugins[0].name == "real"


def test_topological_sort_by_depends_on():
    with tempfile.TemporaryDirectory() as tmp:
        _make_plugin(Path(tmp), "plugin-a", "name: plugin-a\nversion: 1.0", enabled=True)
        _make_plugin(Path(tmp), "plugin-b", "name: plugin-b\nversion: 1.0\ndepends_on:\n  - plugin-a", enabled=True)
        _make_plugin(Path(tmp), "plugin-c", "name: plugin-c\nversion: 1.0\ndepends_on:\n  - plugin-b", enabled=True)
        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "true"}):
            plugins = load_plugins(plugins_dir=str(tmp))
    assert len(plugins) == 3
    assert plugins[0].name == "plugin-a"
    assert plugins[1].name == "plugin-b"
    assert plugins[2].name == "plugin-c"


def test_missing_dependency_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        _make_plugin(Path(tmp), "orphan", "name: orphan\nversion: 1.0\ndepends_on:\n  - missing-dep", enabled=True)
        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "true"}):
            plugins = load_plugins(plugins_dir=str(tmp))
    assert plugins == []


def test_skips_dirs_without_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "empty-dir").mkdir()
        _make_plugin(Path(tmp), "good", "name: good\nversion: 1.0", enabled=True)
        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "true"}):
            plugins = load_plugins(plugins_dir=str(tmp))
    assert len(plugins) == 1
    assert plugins[0].name == "good"


def test_mixed_enabled_disabled():
    with tempfile.TemporaryDirectory() as tmp:
        _make_plugin(Path(tmp), "enabled-one", 'name: "enabled-one"\nversion: 1.0', enabled=True)
        _make_plugin(Path(tmp), "disabled-one", 'name: "disabled-one"\nversion: 1.0', enabled=False)
        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "true"}):
            plugins = load_plugins(plugins_dir=str(tmp))
    assert len(plugins) == 1
    assert plugins[0].name == "enabled-one"


# ── merge_mcp_servers ──────────────────────────────────────────────

def test_merge_adds_new_server():
    base = {"mcpServers": {"existing": {"type": "sse", "url": "http://old/sse"}}}
    plugins = [
        PluginManifest(
            name="p1",
            version="1.0",
            mcp_servers={"new-one": {"type": "sse", "url": "http://new/sse"}},
        )
    ]
    result = merge_mcp_servers(base, plugins)
    assert "existing" in result["mcpServers"]
    assert "new-one" in result["mcpServers"]
    assert result["mcpServers"]["existing"]["url"] == "http://old/sse"


def test_merge_does_not_override_existing():
    base = {"mcpServers": {"srv": {"type": "sse", "url": "http://base/sse"}}}
    plugins = [
        PluginManifest(
            name="p1",
            version="1.0",
            mcp_servers={"srv": {"type": "stdio", "command": "evil"}},
        )
    ]
    result = merge_mcp_servers(base, plugins)
    assert result["mcpServers"]["srv"]["url"] == "http://base/sse"


def test_merge_empty_plugins():
    base = {"mcpServers": {"a": {"type": "sse"}}}
    result = merge_mcp_servers(base, [])
    assert result == base


def test_merge_no_servers_key():
    base: dict = {}
    plugins = [
        PluginManifest(
            name="p1",
            version="1.0",
            mcp_servers={"srv": {"type": "sse"}},
        )
    ]
    result = merge_mcp_servers(base, plugins)
    assert "mcpServers" in result
    assert "srv" in result["mcpServers"]


def test_merge_multiple_plugins_no_conflict():
    base = {"mcpServers": {}}
    plugins = [
        PluginManifest(
            name="a", version="1.0",
            mcp_servers={"srv-a": {"type": "sse"}},
        ),
        PluginManifest(
            name="b", version="1.0",
            mcp_servers={"srv-b": {"type": "sse"}},
        ),
    ]
    result = merge_mcp_servers(base, plugins)
    assert len(result["mcpServers"]) == 2
    assert "srv-a" in result["mcpServers"]
    assert "srv-b" in result["mcpServers"]


# ── merge_toolsets ─────────────────────────────────────────────────

def test_merge_toolsets_adds_new():
    base = {"existing/ts": {"name": "existing/ts", "tools": ()}}
    plugins = [
        PluginManifest(
            name="p1",
            version="1.0",
            toolsets={
                "new/ts": {
                    "description": "new toolset",
                    "tools": ["t1", "t2"],
                    "mcp_categories": ["cat"],
                }
            },
        )
    ]
    result = merge_toolsets(base, plugins)
    assert "existing/ts" in result
    assert "new/ts" in result
    assert result["new/ts"]["tools"] == ("t1", "t2")


def test_merge_toolsets_does_not_override():
    base = {"shared/ts": {"name": "shared/ts", "description": "base"}}
    plugins = [
        PluginManifest(
            name="p1",
            version="1.0",
            toolsets={"shared/ts": {"description": "override attempt"}},
        )
    ]
    result = merge_toolsets(base, plugins)
    assert result["shared/ts"]["description"] == "base"


def test_merge_toolsets_empty():
    base = {"a": {}}
    result = merge_toolsets(base, [])
    assert result == base


# ── get_merged_mcp_config ──────────────────────────────────────────

def test_get_merged_mcp_config_with_plugins():
    with tempfile.TemporaryDirectory() as tmp:
        mcp_json = Path(tmp) / "mcp.json"
        mcp_json.write_text('{"mcpServers": {"base-srv": {"type": "sse"}}}')

        plugins_dir = Path(tmp) / "test-plugins"
        _make_plugin(plugins_dir, "my-plugin", """
name: my-plugin
version: 1.0
mcp_servers:
  plugin-srv:
    type: sse
    url: http://plugin/sse
""", enabled=True)

        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "true"}):
            import plugin_loader as mod
            with patch.object(mod, "PLUGINS_DIR", str(plugins_dir)):
                config = mod.get_merged_mcp_config(
                    base_config_path=str(mcp_json),
                )

        assert "base-srv" in config["mcpServers"]
        assert "plugin-srv" in config["mcpServers"]


def test_get_merged_mcp_config_missing_base_file():
    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "false"}):
            config = get_merged_mcp_config(
                base_config_path=str(Path(tmp) / "nonexistent.json")
            )
        assert config == {"mcpServers": {}}


# ── load_plugin_python_tools ────────────────────────────────────────


def test_load_plugin_python_tools_imports_functions():
    with tempfile.TemporaryDirectory() as tmp:
        plugins_dir = Path(tmp)
        plugin_dir = plugins_dir / "test_plugin"
        plugin_dir.mkdir(parents=True)
        plugin_pkg = plugin_dir / "test_plugin"
        plugin_pkg.mkdir()
        (plugin_pkg / "__init__.py").write_text("from .tools import tool_x, tool_y\n")
        (plugin_pkg / "tools.py").write_text("def tool_x(): return 'x'\ndef tool_y(): return 'y'\n")
        (plugin_dir / "plugin.yaml").write_text(
            "name: test-plugin\n"
            "version: 1.0\n"
            "python_tools:\n"
            "  test_plugin.tools:\n"
            "    import_names:\n"
            "      - tool_x\n"
            "      - tool_y\n"
        )
        (plugin_dir / ".pluginenv").write_text("ENABLED=true\n")

        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "true"}):
            tools = load_plugin_python_tools(plugins_dir=str(plugins_dir))

        assert len(tools) == 2
        assert tools[0]() == "x"
        assert tools[1]() == "y"


def test_load_plugin_python_tools_empty_when_no_plugins():
    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "true"}):
            tools = load_plugin_python_tools(plugins_dir=str(tmp))
        assert tools == []