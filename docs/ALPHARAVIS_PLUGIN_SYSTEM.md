# AlphaRavis Plugin System

Plugin-System für den AI Stack — ähnlich wie ComfyUI Custom Nodes.
Plugins erweitern den Stack um neue MCP-Server, Toolsets, Docker-Services und Skills.

## Quick Start

```bash
# Plugin installieren
git clone https://github.com/user/my-plugin.git plugins/my-plugin

# Aktivieren: .pluginenv editieren
echo "ENABLED=true" > plugins/my-plugin/.pluginenv

# Systemweit einschalten (in .env):
ALPHARAVIS_ENABLE_PLUGIN_SYSTEM=true

# Docker-Services mergen
make plugins-compose
```

## Plugin aktivieren / deaktivieren

Jedes Plugin hat eine `.pluginenv`-Datei:

```
# plugins/my-plugin/.pluginenv
ENABLED=false   ← false = deaktiviert (safe default)
```

Zum Aktivieren einfach auf `ENABLED=true` ändern. Kein Eintrag in der globalen `.env` nötig.
Das Plugin wird beim nächsten Start von LangGraph automatisch geladen.

## Architektur

```
plugins/
  _example/                       # Referenz (_-Prefix = ignoriert)
  my-plugin/
    plugin.yaml                   # Manifest
    .pluginenv                    # ENABLED=true|false
    skills/                       # Hermes Skills (optional)
    Dockerfile                    # Docker-Image (optional)
```

## Manifest-Format (`plugin.yaml`)

```yaml
name: my-plugin           # Eindeutiger Name (required)
version: 1.0.0            # SemVer (required)
description: "..."        # Beschreibung
author: "..."             # Autor
depends_on:               # Abhängigkeiten zu anderen Plugins
  - base-plugin
mcp_servers:              # MCP-Server → merged in mcp.json
  my-server:
    type: sse
    url: "http://my-server:9010/mcp/sse"
toolsets:                 # Toolsets → merged in Toolset-Registry
  my/tools:
    description: "My tools"
    tools: [tool_a, tool_b]
    mcp_categories: [my-cat]
docker_compose:           # Docker-Fragment → merged in docker-compose.plugins.yml
  services:
    my-server:
      image: my-image
      profiles: [plugins]
skills:                   # Skill-Pfade (relativ zum Plugin-Verzeichnis)
  - path: skills/my-skill.md
env_defaults:             # Default-Werte für Env-Variablen
  MY_SETTING: "default"
```

## Wie es funktioniert

### Plugin-Loader (`plugin_loader.py`)

- Scannt `plugins/` beim Import
- Ignoriert Verzeichnisse mit `_` oder `.` Prefix
- Prüft `.pluginenv`: `ENABLED=true` → laden, sonst überspringen
- Parst und validiert `plugin.yaml`
- Topologische Sortierung nach `depends_on`

### MCP-Merging (`mcp_client.py`)

`load_mcp_config()` merged Plugin-MCP-Server am Ende. Base hat Priorität.

### Toolset-Merging (`alpharavis_toolsets.py`)

`_maybe_merge_plugin_toolsets()` läuft beim Import. Plugin-Toolsets werden als `Toolset`-Objekte in `TOOLSETS` eingefügt.

### Docker-Compose (`merge_plugin_compose.py`)

`make plugins-compose` erzeugt `docker-compose.plugins.yml` aus Plugin-Fragmenten.

## Feature-Flags

| Flag | Default | Beschreibung |
|------|---------|-------------|
| `ALPHARAVIS_ENABLE_PLUGIN_SYSTEM` | `false` | Systemweite Aktivierung |

Einzelne Plugins werden NICHT über globale Env-Variablen gesteuert, sondern
ausschließlich über ihre `.pluginenv`-Datei.

## Limits (V1)

- Keine Überschreibung bestehender MCP-Server oder Toolsets
- Keine Swarm-Agent-Plugins
- Keine automatischen Updates
