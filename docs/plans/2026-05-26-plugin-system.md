# Plugin-System (Custom Nodes) für den AI Stack — Konzept & Implementierungsplan

> **Für Hermes:** Nutze subagent-driven-development Skill, um diesen Plan Task für Task umzusetzen.

**Ziel:** Ein ComfyUI-ähnliches Plugin-System, mit dem Drittanbieter neue Funktionen (Tools, MCP-Server, Agenten, Skills) als installierbare Pakete bereitstellen können.

**Architektur:** Ein Plugin ist ein Verzeichnis mit `plugin.yaml`-Manifest + Payload. Ein Plugin-Loader liest beim Start alle aktiven Plugins, merged deren MCP-Server-Definitionen, Toolset-Definitionen, Docker-Fragmente und Skills in die bestehende Infrastruktur ein.

**Tech Stack:** Python (Loader), YAML (Manifest), bestehende MCP/Toolset/Docker-Compose-Infrastruktur. Feature-flagged via `ALPHARAVIS_ENABLE_PLUGIN_SYSTEM=false`.

---

## Design-Entscheidungen

1. **Keine Breaking Changes** — bestehende `mcp.json`, `alpharavis_toolsets.py`, `docker-compose.yml` bleiben unverändert. Plugins sind additiv.
2. **Feature-Flag Default OFF** — `ALPHARAVIS_ENABLE_PLUGIN_SYSTEM=false`. Aktivierung ändert nichts an bestehenden Workflows, nur neue Plugin-Funktionen kommen dazu.
3. **Plugin = Verzeichnis** — `plugins/<name>/plugin.yaml` + optionale Dateien. Kein komplexer Package-Manager für V1.
4. **Manifest-basiert** — `plugin.yaml` deklariert, was das Plugin bereitstellt. Der Loader validiert und merged.
5. **Git-basierte Installation** — `git clone` in `plugins/` reicht. Optional später: `make plugin-install <url>`.

---

## Plugin-Manifest-Format (`plugin.yaml`)

```yaml
# plugins/advanced-image-gen/plugin.yaml
name: advanced-image-gen
version: 1.0.0
description: "Erweiterte Bildgenerierung mit ControlNet und IP-Adapter"
author: "amin"
license: MIT
homepage: "https://github.com/example/advanced-image-gen"

# Feature-Gate: Plugin wird nur geladen, wenn diese Env-Variable truthy ist
# (leer = immer laden, wenn Plugin-System aktiv)
enabled_env: ALPHARAVIS_ENABLE_PLUGIN_ADVANCED_IMAGE_GEN
# Default-Wert, wenn die Variable nicht gesetzt ist
enabled_default: true

# Dependencies: andere Plugins, die zuerst geladen werden müssen
depends_on: []

# --- MCP-Server (wird in mcp.json gemerged) ---
mcp_servers:
  advanced-image-gen:
    type: sse
    url: "http://advanced-image-gen:9010/mcp/sse"
    timeout: 300
    connect_timeout: 30

# --- Toolsets (werden in Toolset-Registry gemerged) ---
toolsets:
  media/advanced-image:
    description: "Advanced image generation via ControlNet + IP-Adapter"
    tools:
      - start_advanced_image_job
      - check_advanced_image_job
      - list_controlnet_models
    mcp_categories:
      - advanced-image
    includes: []  # optional: andere Toolsets, die inkludiert werden

# --- Docker Compose Fragment (wird in docker-compose.yml gemerged) ---
# Nur wenn das Plugin einen eigenen Service braucht
docker_compose:
  services:
    advanced-image-gen:
      build:
        context: ./plugins/advanced-image-gen
        dockerfile: Dockerfile
      ports:
        - "9010:9010"
      environment:
        - COMFY_URL=${COMFY_URL:-http://comfyui:8188}
      profiles:
        - plugins
        - advanced-image-gen

# --- Skills (werden nach Hermes-Skills kopiert/verlinkt) ---
skills:
  - path: skills/controlnet-workflows.md
  - path: skills/ip-adapter-guide.md

# --- Env-Variablen (Default-Werte, die in .env gemerged werden) ---
env_defaults:
  ADVANCED_IMAGE_GEN_MODEL: sdxl
  ADVANCED_IMAGE_GEN_MAX_STEPS: "50"
```

---

## Task 1: Plugin-Loader-Modul erstellen

**Ziel:** Ein Python-Modul, das `plugins/` scannt, `plugin.yaml` parst, validiert und geladene Plugin-Daten als dicts zurückgibt.

**Files:**
- Create: `langgraph-app/plugin_loader.py`
- Create: `tests/test_plugin_loader.py`

**Step 1: Modulstruktur anlegen**

```python
# langgraph-app/plugin_loader.py
"""
Plugin loader for AlphaRavis.
Scans plugins/ directory, validates manifests, returns loaded plugin data.

Feature-flagged via ALPHARAVIS_ENABLE_PLUGIN_SYSTEM (default: false).
"""

from __future__ import annotations

import os
import yaml
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
    enabled_env: str = ""
    enabled_default: bool = True
    depends_on: list[str] = field(default_factory=list)
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    toolsets: dict[str, dict[str, Any]] = field(default_factory=dict)
    docker_compose: dict[str, Any] = field(default_factory=dict)
    skills: list[dict[str, str]] = field(default_factory=list)
    env_defaults: dict[str, str] = field(default_factory=dict)


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _load_manifest(plugin_dir: Path) -> PluginManifest | None:
    """Load and validate a single plugin.yaml."""
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
        enabled_env=str(raw.get("enabled_env", "")),
        enabled_default=bool(raw.get("enabled_default", True)),
        depends_on=list(raw.get("depends_on", [])),
        mcp_servers=dict(raw.get("mcp_servers", {})),
        toolsets=dict(raw.get("toolsets", {})),
        docker_compose=dict(raw.get("docker_compose", {})),
        skills=list(raw.get("skills", [])),
        env_defaults=dict(raw.get("env_defaults", {})),
    )


def _is_plugin_enabled(manifest: PluginManifest) -> bool:
    """Check if plugin is enabled via env var or default."""
    if manifest.enabled_env:
        return _env_bool(manifest.enabled_env, manifest.enabled_default)
    return manifest.enabled_default


def load_plugins(plugins_dir: str | None = None) -> list[PluginManifest]:
    """Scan plugins/ and return list of enabled, validated manifests."""
    directory = Path(plugins_dir or PLUGINS_DIR)
    
    if not _env_bool("ALPHARAVIS_ENABLE_PLUGIN_SYSTEM", False):
        return []
    
    if not directory.exists():
        return []
    
    plugins: list[PluginManifest] = []
    
    for plugin_path in sorted(directory.iterdir()):
        if not plugin_path.is_dir():
            continue
        manifest = _load_manifest(plugin_path)
        if manifest and _is_plugin_enabled(manifest):
            plugins.append(manifest)
    
    # Topological sort by depends_on
    name_to_plugin = {p.name: p for p in plugins}
    loaded_names: set[str] = set()
    ordered: list[PluginManifest] = []
    
    def visit(name: str, path: set[str]) -> None:
        if name in loaded_names:
            return
        if name in path:
            return  # cycle — skip
        if name not in name_to_plugin:
            return  # missing dependency — skip
        path.add(name)
        for dep in name_to_plugin[name].depends_on:
            visit(dep, path)
        loaded_names.add(name)
        ordered.append(name_to_plugin[name])
    
    for p in plugins:
        visit(p.name, set())
    
    return ordered
```

**Step 2: Tests schreiben**

```python
# tests/test_plugin_loader.py
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from langgraph_app.plugin_loader import (
    _load_manifest,
    load_plugins,
    _is_plugin_enabled,
    PluginManifest,
)


def test_load_valid_manifest():
    yaml_content = """
name: test-plugin
version: 1.0.0
description: A test plugin
mcp_servers:
  test-srv:
    type: sse
    url: http://localhost:9999/sse
"""
    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp) / "test-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(yaml_content)
        manifest = _load_manifest(plugin_dir)
    
    assert manifest is not None
    assert manifest.name == "test-plugin"
    assert manifest.version == "1.0.0"
    assert "test-srv" in manifest.mcp_servers


def test_missing_manifest_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _load_manifest(Path(tmp))
    assert manifest is None


def test_missing_name_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp) / "bad-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text("version: 1.0.0")
        manifest = _load_manifest(plugin_dir)
    assert manifest is None


def test_plugin_disabled_when_system_flag_off():
    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp) / "my-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text("name: my-plugin\nversion: 1.0.0")
        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "false"}):
            plugins = load_plugins(plugins_dir=str(tmp))
    assert plugins == []


def test_plugin_enabled_when_system_flag_on():
    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp) / "my-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text("name: my-plugin\nversion: 1.0.0")
        with patch.dict(os.environ, {"ALPHARAVIS_ENABLE_PLUGIN_SYSTEM": "true"}):
            plugins = load_plugins(plugins_dir=str(tmp))
    assert len(plugins) == 1
    assert plugins[0].name == "my-plugin"


def test_plugin_disabled_by_own_env_flag():
    manifest = PluginManifest(
        name="opt-in-plugin",
        version="1.0.0",
        enabled_env="ENABLE_OPT_IN",
        enabled_default=False,
    )
    with patch.dict(os.environ, {}, clear=True):
        assert _is_plugin_enabled(manifest) is False
    with patch.dict(os.environ, {"ENABLE_OPT_IN": "true"}):
        assert _is_plugin_enabled(manifest) is True
```

**Step 3: Tests ausführen (sollten fehlschlagen — Modul existiert noch nicht)**

```bash
cd langgraph-app && python -m pytest ../tests/test_plugin_loader.py -v
# Expected: ImportError — plugin_loader module not found
```

**Step 4: Modul erstellen und Tests grün machen**

```bash
# Datei anlegen:
# langgraph-app/plugin_loader.py mit dem Code aus Step 1

cd langgraph-app && python -m pytest ../tests/test_plugin_loader.py -v
# Expected: ALL PASS
```

**Step 5: Commit**

```bash
git add langgraph-app/plugin_loader.py tests/test_plugin_loader.py
git commit -m "feat: add plugin loader module with manifest parsing and env gating"
```

---

## Task 2: MCP-Merging in den Loader einbauen

**Ziel:** Der Plugin-Loader merged MCP-Server-Definitionen aus Plugins in die bestehende `mcp.json`.

**Files:**
- Modify: `langgraph-app/plugin_loader.py`
- Modify: `langgraph-app/mcp_client.py` (Loader beim Start aufrufen)
- Create: `tests/test_plugin_mcp_merge.py`

**Step 1: Merge-Funktion schreiben**

```python
# In plugin_loader.py hinzufügen:

def merge_mcp_servers(base_config: dict, plugins: list[PluginManifest]) -> dict:
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


def get_merged_mcp_config(base_config_path: str | None = None) -> dict:
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
```

**Step 2: Tests für Merge-Logik**

```python
# tests/test_plugin_mcp_merge.py
from langgraph_app.plugin_loader import merge_mcp_servers, PluginManifest


def test_merge_adds_new_server():
    base = {"mcpServers": {"existing": {"type": "sse", "url": "http://old/sse"}}}
    plugins = [
        PluginManifest(
            name="p1", version="1.0",
            mcp_servers={"new-one": {"type": "sse", "url": "http://new/sse"}},
        )
    ]
    result = merge_mcp_servers(base, plugins)
    assert "existing" in result["mcpServers"]
    assert "new-one" in result["mcpServers"]


def test_merge_does_not_override_existing():
    base = {"mcpServers": {"srv": {"type": "sse", "url": "http://base/sse"}}}
    plugins = [
        PluginManifest(
            name="p1", version="1.0",
            mcp_servers={"srv": {"type": "stdio", "command": "evil"}},
        )
    ]
    result = merge_mcp_servers(base, plugins)
    # Base wins — plugin cannot override
    assert result["mcpServers"]["srv"]["url"] == "http://base/sse"


def test_merge_empty_plugins():
    base = {"mcpServers": {"a": {}}}
    result = merge_mcp_servers(base, [])
    assert result == base
```

**Step 3: Im mcp_client.py den Loader aufrufen**

In `langgraph-app/mcp_client.py` die Stelle finden, wo `mcp.json` geladen wird, und durch den gemergten Config-Aufruf ersetzen:

```python
# Vorher (sinngemäß):
# with open("mcp.json") as f:
#     config = json.load(f)

# Nachher:
from plugin_loader import get_merged_mcp_config

def _load_mcp_config() -> dict:
    """Load MCP config, merging in any enabled plugins."""
    return get_merged_mcp_config()
```

**Step 4: Tests grün machen und commiten**

```bash
cd langgraph-app && python -m pytest ../tests/test_plugin_mcp_merge.py -v
# Expected: ALL PASS

git add langgraph-app/plugin_loader.py langgraph-app/mcp_client.py tests/test_plugin_mcp_merge.py
git commit -m "feat: merge plugin MCP servers into mcp.json at load time"
```

---

## Task 3: Toolset-Merging

**Ziel:** Plugin-definierte Toolsets werden in die bestehende `TOOLSETS`-Registry gemerged.

**Files:**
- Modify: `langgraph-app/plugin_loader.py`
- Modify: `langgraph-app/alpharavis_toolsets.py`
- Create: `tests/test_plugin_toolset_merge.py`

**Step 1: Merge-Funktion für Toolsets**

```python
# In plugin_loader.py:

from alpharavis_toolsets import Toolset  # guarded import

def merge_toolsets(base_toolsets: dict[str, Any], plugins: list[PluginManifest]) -> dict[str, Any]:
    """Merge plugin toolsets into the base TOOLSETS dict.
    
    Base toolsets take priority — plugins cannot override existing names.
    """
    result = dict(base_toolsets)
    
    for plugin in plugins:
        for name, ts_def in plugin.toolsets.items():
            if name not in result:
                result[name] = Toolset(
                    name=name,
                    description=ts_def.get("description", f"Plugin {plugin.name}"),
                    tools=tuple(ts_def.get("tools", [])),
                    includes=tuple(ts_def.get("includes", [])),
                    mcp_categories=tuple(ts_def.get("mcp_categories", [])),
                )
    
    return result
```

**Step 2: In alpharavis_toolsets.py den Loader aufrufen**

Am Ende von `alpharavis_toolsets.py`, nach der `TOOLSETS`-Definition:

```python
# Merge plugin toolsets if plugin system is enabled
def _maybe_merge_plugin_toolsets() -> None:
    """Merge plugin-defined toolsets into TOOLSETS at import time."""
    if not os.getenv("ALPHARAVIS_ENABLE_PLUGIN_SYSTEM", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    try:
        from plugin_loader import load_plugins
        plugins = load_plugins()
        for plugin in plugins:
            for name, ts_def in plugin.toolsets.items():
                if name not in TOOLSETS:
                    TOOLSETS[name] = Toolset(
                        name=name,
                        description=ts_def.get("description", f"Plugin {plugin.name}"),
                        tools=tuple(ts_def.get("tools", [])),
                        includes=tuple(ts_def.get("includes", [])),
                        mcp_categories=tuple(ts_def.get("mcp_categories", [])),
                    )
    except ImportError:
        pass  # plugin_loader not available

_maybe_merge_plugin_toolsets()
```

**Step 3: Tests und Commit**

```bash
cd langgraph-app && python -m pytest ../tests/ -v -k plugin
# Expected: ALL PLUGIN TESTS PASS

git add langgraph-app/plugin_loader.py langgraph-app/alpharavis_toolsets.py tests/
git commit -m "feat: merge plugin toolsets into TOOLSETS registry"
```

---

## Task 4: Docker-Compose-Merging

**Ziel:** Ein Script merged Plugin-Docker-Fragmente in `docker-compose.yml` oder erzeugt ein Override-File.

**Files:**
- Create: `scripts/merge_plugin_compose.py`
- Modify: `Makefile` (optionaler Target)
- Create: `tests/test_plugin_compose_merge.py`

**Step 1: Merge-Script**

```python
# scripts/merge_plugin_compose.py
"""
Merge plugin docker-compose fragments into docker-compose.plugins.yml.
This override file is loaded by docker compose alongside the main file.
"""

import os
import sys
import yaml
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGINS_DIR = REPO_ROOT / "plugins"
OUTPUT = REPO_ROOT / "docker-compose.plugins.yml"


def main() -> None:
    if not os.getenv("ALPHARAVIS_ENABLE_PLUGIN_SYSTEM", "").strip().lower() in ("1", "true", "yes", "on"):
        print("Plugin system disabled — removing override file if present.")
        if OUTPUT.exists():
            OUTPUT.unlink()
        return

    merged: dict[str, dict] = {"services": {}}
    
    for plugin_dir in sorted(PLUGINS_DIR.iterdir()):
        if not plugin_dir.is_dir():
            continue
        manifest_path = plugin_dir / "plugin.yaml"
        if not manifest_path.exists():
            continue
        
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        compose = manifest.get("docker_compose", {})
        services = compose.get("services", {})
        
        for svc_name, svc_def in services.items():
            if svc_name not in merged["services"]:
                merged["services"][svc_name] = svc_def
    
    if merged["services"]:
        with open(OUTPUT, "w") as f:
            f.write("# Auto-generated by merge_plugin_compose.py — do not edit manually.\n")
            f.write(f"# Generated from plugins/ manifests.\n")
            yaml.dump(merged, f, default_flow_style=False)
        print(f"Wrote {len(merged['services'])} plugin service(s) to {OUTPUT}")
    else:
        print("No plugin services to merge.")
        if OUTPUT.exists():
            OUTPUT.unlink()


if __name__ == "__main__":
    main()
```

**Step 2: Makefile-Target**

```makefile
# In Makefile:
plugins-compose:
	python3 scripts/merge_plugin_compose.py

# Bei docker compose up automatisch einbinden, wenn das Override existiert:
# docker compose -f docker-compose.yml -f docker-compose.plugins.yml up
```

**Step 3: Tests und Commit**

```bash
python3 scripts/merge_plugin_compose.py
# Expected: No plugin services (no plugins/ dir with manifests yet)

git add scripts/merge_plugin_compose.py
git commit -m "feat: plugin docker-compose fragment merger"
```

---

## Task 5: Beispiel-Plugin + Dokumentation

**Ziel:** Ein Minimal-Plugin, das zeigt, wie alles zusammenhängt. Plus Dokumentation.

**Files:**
- Create: `plugins/_example/plugin.yaml`
- Create: `plugins/_example/README.md`
- Create: `docs/ALPHARAVIS_PLUGIN_SYSTEM.md`
- Create: `plugins/.gitkeep`

**Step 1: Beispiel-Plugin**

```yaml
# plugins/_example/plugin.yaml
name: example-hello-world
version: 0.1.0
description: "Minimal example plugin showing the manifest format"
author: "AlphaRavis"
enabled_env: ALPHARAVIS_ENABLE_EXAMPLE_PLUGIN
enabled_default: false  # disabled by default

mcp_servers:
  example-hello:
    type: sse
    url: "http://example-hello:9999/mcp/sse"
    timeout: 30

toolsets:
  example/greeting:
    description: "Example greeting toolset"
    tools:
      - example_say_hello
      - example_say_goodbye
    mcp_categories:
      - example

docker_compose:
  services:
    example-hello:
      image: hello-world
      profiles:
        - plugins

skills:
  - path: skills/example-skill.md

env_defaults:
  EXAMPLE_GREETING: "Hello from plugin!"
```

**Step 2: Dokumentation**

```markdown
# AlphaRavis Plugin System

## Overview

Das Plugin-System erlaubt es, neue Funktionen als installierbare Pakete
bereitzustellen — ähnlich wie ComfyUI Custom Nodes.

Ein Plugin ist ein Verzeichnis unter `plugins/` mit einer `plugin.yaml`-Manifest-Datei.

## Quick Start

### Installation

```bash
git clone https://github.com/user/my-plugin.git plugins/my-plugin
make plugins-compose
```

### Aktivierung

```bash
# Systemweit aktivieren:
# In .env:
ALPHARAVIS_ENABLE_PLUGIN_SYSTEM=true

# Plugin-spezifisch (wenn plugin.yaml ein enabled_env hat):
ALPHARAVIS_ENABLE_MY_PLUGIN=true
```

## Manifest-Format

Siehe `plugins/_example/plugin.yaml` für ein vollständiges Beispiel.

| Feld | Typ | Beschreibung |
|------|-----|-------------|
| `name` | string | Eindeutiger Plugin-Name (required) |
| `version` | string | SemVer (required) |
| `enabled_env` | string | Env-Variable zum Aktivieren (optional) |
| `enabled_default` | bool | Default wenn `enabled_env` nicht gesetzt |
| `depends_on` | list | Plugins, die zuerst geladen werden müssen |
| `mcp_servers` | dict | MCP-Server-Definitionen → merged in `mcp.json` |
| `toolsets` | dict | Toolset-Definitionen → merged in Toolset-Registry |
| `docker_compose` | dict | Docker-Compose-Fragment → merged in Override-File |
| `skills` | list | Pfade zu Skill-Dateien |
| `env_defaults` | dict | Default-Werte für Env-Variablen |

## Was ein Plugin NICHT kann

- Bestehende MCP-Server oder Toolsets überschreiben (Base hat Priorität)
- Ohne `ALPHARAVIS_ENABLE_PLUGIN_SYSTEM=true` laden
- Bestehende Swarm-Agenten ersetzen (Agent-Plugins sind V2)

## Architektur

```
plugins/                          # Plugin-Verzeichnisse
  my-plugin/
    plugin.yaml                   # Manifest
    skills/                       # Hermes Skills
    Dockerfile                    # Docker-Image
    ...

langgraph-app/
  plugin_loader.py                # Lädt + merged Plugins
  alpharavis_toolsets.py          # Merged Toolsets beim Import
  mcp_client.py                   # Merged MCP-Config beim Start

scripts/
  merge_plugin_compose.py         # Merged Docker-Fragmente → docker-compose.plugins.yml
```
```

**Step 4: Commit**

```bash
git add plugins/ docs/ALPHARAVIS_PLUGIN_SYSTEM.md
git commit -m "docs: add example plugin and plugin system documentation"
```

---

## Zusammenfassung der Architektur

```
┌─────────────────────────────────────────────────────────┐
│ plugin.yaml                                             │
│ ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│ │ mcp_servers │  │  toolsets    │  │ docker_compose  │ │
│ └──────┬──────┘  └──────┬───────┘  └───────┬─────────┘ │
└────────┼────────────────┼──────────────────┼───────────┘
         │                │                  │
         ▼                ▼                  ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────────────┐
│ plugin_loader   │ │ toolsets.py  │ │ merge_plugin_compose │
│ → mcp.json      │ │ → TOOLSETS   │ │ → docker-compose.    │
│   (merged)      │ │   dict       │ │   plugins.yml        │
└─────────────────┘ └──────────────┘ └──────────────────────┘
         │                │                  │
         ▼                ▼                  ▼
    MCP Client      Agent Tools       docker compose up
```

## Was kommt NICHT in V1

- Plugin-Registry mit automatischen Updates (manuelles `git pull` reicht)
- Plugin-Abhängigkeitsauflösung mit Version-Constraints (nur Namens-Check)
- Swarm-Agent-Plugins (komplexer, braucht Graph-Modifikation)
- GUI/PWA Plugin-Manager (CLI + Makefile reichen für V1)

## Verifikation nach Implementierung

```bash
# 1. Plugin-System deaktiviert:
ALPHARAVIS_ENABLE_PLUGIN_SYSTEM=false make status
# → Keine Plugin-Toolsets, keine Plugin-MCP-Server

# 2. Plugin-System aktiviert, Beispiel-Plugin deaktiviert:
ALPHARAVIS_ENABLE_PLUGIN_SYSTEM=true make status
# → Keine Plugin-Toolsets (Beispiel-Plugin ist default=false)

# 3. Beides aktiviert:
ALPHARAVIS_ENABLE_PLUGIN_SYSTEM=true ALPHARAVIS_ENABLE_EXAMPLE_PLUGIN=true make status
# → example/greeting Toolset erscheint, example-hello MCP-Server registriert

# 4. Unit-Tests:
pytest tests/test_plugin_*.py -v
```
