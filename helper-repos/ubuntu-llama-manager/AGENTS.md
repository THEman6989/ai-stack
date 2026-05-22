# Agent Instructions

This repository is **Ubuntu Llama Manager**, a standalone Ubuntu/ROCm
management service for a `llama.cpp` server. It is not part of the user's AI
Stack repository. Treat it as an independent project.

## Project Purpose

The manager is responsible for:

- running and supervising a `llama.cpp` server with systemd
- controlled interval reboots for GPU stability
- optional AMD MI50/MI60 power limits through `rocm-smi`
- optional ROCm performance/clock tuning through `rocm-smi`
- autonomous ROCm/PCIe/RAS/kernel health checks
- recovery decisions when Llama stops responding
- HTTP/JSON API endpoints for a future AI Stack
- prepared ESP8266 / NodeMCU power-control integration
- local HuggingFace/GGUF model scanning

The interval reboot and GPU health shutdown behavior are intentional stability
features, not bugs.

## Safety Rules

Do not run dangerous commands unless the user explicitly asks for it in the
current turn.

Dangerous commands include:

```bash
sudo ./install.sh apply
sudo ./install.sh restart
sudo ./install.sh disable
sudo systemctl restart ubuntu-llama.service
sudo systemctl poweroff
sudo systemctl reboot
./bin/reboot-now.sh
```

Why: applying systemd units may restart Llama, enable timers, change GPU power
limits/clocks, or activate shutdown behavior.

When testing, prefer non-destructive checks:

```bash
./install.sh check
python3 -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python3 -m ubuntu_manager.monitor --config ubuntu-llama.conf --once
```

## Runtime Config

Runtime config files are local and ignored by Git:

```text
ubuntu-llama.conf
rakam-llama.conf
.env
logs/
state/
```

Do not commit real config, API tokens, passwords, logs, or state files.
Change `.env.example` when adding documented config keys.

`rakam-llama.conf` exists only as legacy compatibility. New code and docs
should use `ubuntu-llama.conf`, `UBUNTU_CONFIG`, and `ubuntu-*` service names.

## Naming

Use "Ubuntu", not "Rakam", for user-facing names.

Preferred names:

```text
Ubuntu Llama Manager
ubuntu-llama-manager
ubuntu_manager
ubuntu-llama.service
ubuntu-manager-api.service
ubuntu-gpu-power.service
ubuntu-gpu-health.service
ubuntu-reboot-watch.service
```

Legacy `rakam-*` names should only appear where needed to stop or migrate old
services.

## Code Layout

```text
ubuntu_manager/api.py           HTTP/JSON API
ubuntu_manager/config.py        .env-style config loader
ubuntu_manager/gpu.py           ROCm/PCIe/RAS/kernel diagnostics
ubuntu_manager/monitor.py       autonomous GPU health monitor loop
ubuntu_manager/recovery.py      recovery actions and ESP notification
ubuntu_manager/services.py      systemd/Llama/reboot orchestration
ubuntu_manager/models.py        local model scanning
ubuntu_manager/llama_config.py  LLAMA_COMMAND model switching
bin/*.sh                        systemd entrypoints and operator helpers
install.sh                      systemd unit generator/installer
docs/                           operator documentation
tests/                          dependency-free unittest tests
```

## Systemd Model

The installer generates these units:

```text
ubuntu-gpu-power.service
ubuntu-gpu-health.service
ubuntu-llama.service
ubuntu-manager-api.service
llama-reboot.service
llama-reboot.timer
ubuntu-reboot-watch.service
```

`llama-reboot.timer` is the preferred interval reboot mechanism.
`ubuntu-reboot-watch.service` is the fallback for `REBOOT_TIMER_MODE="llama-start"`.

The GPU health monitor should normally run as root through systemd, so it can
read kernel logs, call ROCm tools, and perform configured shutdown/reboot
actions without storing a sudo password.

## API Safety

Dangerous API endpoints require Bearer auth:

```http
Authorization: Bearer <API_TOKEN>
```

Dangerous endpoints include Llama start/stop/restart, force-kill, model switch,
reboot controls, GPU fault handling, recovery, and ESP power requests.

Public endpoints like `/health`, `/status`, `/models`, `/llama/status`,
`/reboot/status`, `/diagnostics/gpu`, and `/esp/status` may be queried for
debugging.

The API is intended to bind to `0.0.0.0` for LAN access. Do not remove auth from
dangerous endpoints. Mention firewall/API-token implications when changing API
network exposure.

## Recovery Semantics

When the AI Stack reports that Llama is not responding, the intended flow is:

1. Run GPU diagnostics.
2. If GPU/ROCm/PCIe is critical, prepare ESP power-cycle request and execute
   `GPU_HEALTH_CRITICAL_ACTION`.
3. If GPU looks healthy, treat it as a Llama hang: force-kill Llama and restart
   `ubuntu-llama.service`.

Do not replace this with a blind reboot unless the user asks.

## Model Switching

`POST /llama/switch-model` should preserve existing Llama flags such as Jinja,
context size, port, batch settings, GPU layer settings, and cache options. Only
the model argument should change.

The implementation lives in `ubuntu_manager/llama_config.py`.

## Validation Checklist

Before committing changes, run:

```bash
python3 - <<'PY'
import ast
from pathlib import Path
for path in list(Path("ubuntu_manager").glob("*.py")) + list(Path("tests").glob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("AST syntax ok")
PY
python3 -m unittest discover -s tests -v
bash -n install.sh status.sh bin/*.sh
./install.sh check
```

Use `PYTHONDONTWRITEBYTECODE=1` when running modules manually to avoid root-owned
`__pycache__` files if systemd has touched the repo.

## GitHub

Remote repository:

```text
https://github.com/THEman6989/ubuntu-llama-manager
```

Keep commits focused and do not commit ignored runtime files. Push to `main`
when requested or when the user has asked for repository updates.
