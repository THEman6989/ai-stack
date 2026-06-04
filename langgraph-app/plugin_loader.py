"""
Plugin loader for AlphaRavis.
Scans plugins/ directory, validates manifests, returns loaded plugin data.

Feature-flagged via ALPHARAVIS_ENABLE_PLUGIN_SYSTEM (default: false).

Each plugin ships a .pluginenv file:
  ENABLED=false   ← safe default, user changes to true to activate
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PLUGINS_DIR = os.path.join(os.path.dirname(__file__), "..", "plugins")


@dataclass
class PluginManifest:
    name: str
    version: str
    description: str = ""
    author: str = ""
    depends_on: list[str] = field(default_factory=list)
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    toolsets: dict[str, dict[str, Any]] = field(default_factory=dict)
    python_tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    docker_compose: dict[str, Any] = field(default_factory=dict)
    skills: list[dict[str, str]] = field(default_factory=list)
    env_defaults: dict[str, str] = field(default_factory=dict)


from env_utils import env_bool as _env_bool


def _load_manifest(plugin_dir: Path) -> PluginManifest | None:
    """Load and validate a single plugin.yaml."""
    try:
        import yaml
    except ImportError:
        return None

    manifest_path = plugin_dir / "plugin.yaml"
    if not manifest_path.exists():
        return None

    with open(manifest_path) as f:
        raw = yaml.safe_load(f)

    if not raw or not isinstance(raw, dict):
        return None

    name = raw.get("name")
    version = raw.get("version")
    if not name or not version:
        return None

    return PluginManifest(
        name=str(name),
        version=str(version),
        description=str(raw.get("description", "")),
        author=str(raw.get("author", "")),
        depends_on=list(raw.get("depends_on", [])),
        mcp_servers=dict(raw.get("mcp_servers", {})),
        toolsets=dict(raw.get("toolsets", {})),
        python_tools=dict(raw.get("python_tools", {})),
        docker_compose=dict(raw.get("docker_compose", {})),
        skills=list(raw.get("skills", [])),
        env_defaults=dict(raw.get("env_defaults", {})),
    )


def _plugin_is_enabled(plugin_dir: Path) -> bool:
    """Check plugins/<name>/.pluginenv for ENABLED=true.

    Safe default: if .pluginenv is missing or says ENABLED=false, plugin is off.
    Plugin authors ship .pluginenv with ENABLED=false — user changes to true.
    """
    pluginenv = plugin_dir / ".pluginenv"
    if not pluginenv.exists():
        return False

    try:
        for line in pluginenv.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip().upper() == "ENABLED":
                return val.strip().lower() in ("true", "1", "yes", "on")
    except Exception:
        return False

    return False


def load_plugins(plugins_dir: str | None = None) -> list[PluginManifest]:
    """Scan plugins/ and return list of enabled, validated manifests."""
    directory = Path(plugins_dir or PLUGINS_DIR)

    if not _env_bool("ALPHARAVIS_ENABLE_PLUGIN_SYSTEM", "false"):
        return []

    if not directory.exists():
        return []

    plugins: list[PluginManifest] = []

    for plugin_path in sorted(directory.iterdir()):
        if not plugin_path.is_dir():
            continue
        if plugin_path.name.startswith("_") or plugin_path.name.startswith("."):
            continue

        if not _plugin_is_enabled(plugin_path):
            continue

        manifest = _load_manifest(plugin_path)
        if manifest:
            plugins.append(manifest)

    # Topological sort by depends_on — skip plugins with missing dependencies
    name_to_plugin = {p.name: p for p in plugins}
    loaded_names: set[str] = set()
    ordered: list[PluginManifest] = []

    def visit(name: str, path: set[str]) -> bool:
        if name in loaded_names:
            return True
        if name in path:
            return False  # cycle — skip
        if name not in name_to_plugin:
            return False  # missing dependency — skip
        path.add(name)
        for dep in name_to_plugin[name].depends_on:
            if not visit(dep, path):
                return False  # dependency failed — skip this plugin too
        loaded_names.add(name)
        ordered.append(name_to_plugin[name])
        return True

    for p in plugins:
        visit(p.name, set())

    return ordered


def merge_mcp_servers(
    base_config: dict[str, Any], plugins: list[PluginManifest]
) -> dict[str, Any]:
    """Merge plugin MCP servers into the base mcp.json config.

    Base config takes priority — plugins cannot override existing servers.
    """
    result = dict(base_config)
    existing = set(result.get("mcpServers", {}).keys())

    for plugin in plugins:
        for server_name, server_def in plugin.mcp_servers.items():
            if server_name not in existing:
                result.setdefault("mcpServers", {})[server_name] = server_def
                existing.add(server_name)

    return result


def get_merged_mcp_config(base_config_path: str | None = None) -> dict[str, Any]:
    """Load base mcp.json, merge plugin servers, return merged config."""
    import json

    if base_config_path is None:
        base_config_path = os.path.join(os.path.dirname(__file__), "mcp.json")

    if os.path.exists(base_config_path):
        with open(base_config_path) as f:
            base = json.load(f)
    else:
        base = {"mcpServers": {}}

    plugins = load_plugins()
    return merge_mcp_servers(base, plugins)


def merge_toolsets(
    base_toolsets: dict[str, Any], plugins: list[PluginManifest]
) -> dict[str, Any]:
    """Merge plugin toolsets into the base TOOLSETS dict.

    Base toolsets take priority — plugins cannot override existing names.
    Returns a new dict with plugin toolsets added.
    Does NOT import Toolset class — caller handles that.
    """
    result = dict(base_toolsets)

    for plugin in plugins:
        for name, ts_def in plugin.toolsets.items():
            if name not in result:
                result[name] = {
                    "name": name,
                    "description": ts_def.get(
                        "description", f"Plugin {plugin.name}"
                    ),
                    "tools": tuple(ts_def.get("tools", [])),
                    "includes": tuple(ts_def.get("includes", [])),
                    "mcp_categories": tuple(ts_def.get("mcp_categories", [])),
                }

    return result


def load_plugin_python_tools(
    plugins: list[PluginManifest] | None = None,
    plugins_dir: str | None = None,
) -> list[Any]:
    """Import callable Python tools from enabled plugins' python_tools manifests.

    Ensures the plugins base directory is on sys.path so `plugin_name.module`
    imports resolve correctly.
    """

    import sys

    base = Path(plugins_dir or PLUGINS_DIR)
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

    if plugins is None:
        plugins = load_plugins(plugins_dir=plugins_dir)

    tools: list[Any] = []

    for plugin in plugins:
        plugin_dir = base / plugin.name.replace("-", "_")
        if plugin_dir.is_dir() and str(plugin_dir) not in sys.path:
            sys.path.insert(0, str(plugin_dir))
        for module_name, config in plugin.python_tools.items():
            names = list(config.get("import_names", [])) if isinstance(config, dict) else []
            if not names:
                continue
            try:
                mod = __import__(module_name, fromlist=names)
            except Exception:
                continue
            for name in names:
                obj = getattr(mod, name, None)
                if callable(obj):
                    tools.append(obj)

    return tools
