from __future__ import annotations

import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .gpu import diagnose_gpu, vram_summary
from .llama_config import patch_llama_command, update_llama_command, update_llama_model
from .recovery import force_kill_llama, handle_gpu_fault, probe_llama_generation


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_command(args: list[str], timeout: int = 15) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "command": args,
        }
    except FileNotFoundError as exc:
        return {"ok": False, "returncode": 127, "stdout": "", "stderr": str(exc), "command": args}
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "returncode": 124, "stdout": exc.stdout or "", "stderr": "timeout", "command": args}


def systemctl(*args: str, timeout: int = 15) -> dict[str, Any]:
    return run_command(["systemctl", *args], timeout=timeout)


def service_active(name: str) -> bool:
    return systemctl("is-active", "--quiet", name)["ok"]


def service_enabled(name: str) -> bool:
    return systemctl("is-enabled", "--quiet", name)["ok"]


def check_port(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pgrep(pattern: str, user: str = "") -> dict[str, Any]:
    args = ["pgrep", "-af", "--", pattern]
    if user:
        args = ["pgrep", "-u", user, "-af", "--", pattern]
    return run_command(args, timeout=5)


class Manager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def reload_settings(self) -> None:
        self.settings = Settings.load(self.settings.config_path)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "ubuntu-llama-manager",
            "time": now_iso(),
            "config_loaded": self.settings.config_path.exists(),
        }

    def _llama_status(
        self,
        *,
        service_name: str,
        pattern_key: str,
        pattern_default: str,
        host_key: str,
        host_default: str,
        port_key: str,
        port_default: int,
        log_key: str,
        log_default: str,
        configured_key: str,
        configured_default: bool,
        workdir_key: str,
        command_key: str,
        instance_id: str,
    ) -> dict[str, Any]:
        pattern = self.settings.get(pattern_key, pattern_default)
        user = self.settings.get("RUN_AS_USER", "")
        process = pgrep(pattern, user=user)
        host = self.settings.get(host_key, host_default)
        port = self.settings.int(port_key, port_default)
        port_open = check_port(host, port)
        log_file = Path(self.settings.get(log_key, log_default)).expanduser()
        return {
            "service": service_name,
            "id": instance_id,
            "configured": self.settings.bool(configured_key, configured_default),
            "active": service_active(service_name),
            "enabled": service_enabled(service_name),
            "workdir": self.settings.get(workdir_key, ""),
            "command_key": command_key,
            "command": self.settings.get(command_key, ""),
            "process_running": process["ok"],
            "processes": process["stdout"].splitlines() if process["stdout"] else [],
            "host": host,
            "port": port,
            "port_open": port_open,
            "log_file": str(log_file),
            "log_exists": log_file.exists(),
        }

    def llama_status(self) -> dict[str, Any]:
        return self._llama_status(
            service_name="ubuntu-llama.service",
            pattern_key="LLAMA_PROCESS_PATTERN",
            pattern_default="llama-server",
            host_key="LLAMA_HOST",
            host_default=self.settings.llama_host,
            port_key="LLAMA_PORT",
            port_default=self.settings.llama_port,
            log_key="LLAMA_LOG_FILE",
            log_default=str(self.settings.llama_log_file),
            configured_key="ENABLE_LLAMA_SERVICE",
            configured_default=True,
            workdir_key="LLAMA_WORKDIR",
            command_key="LLAMA_COMMAND",
            instance_id="primary",
        )

    def llama_secondary_status(self) -> dict[str, Any]:
        return self._llama_status(
            service_name="ubuntu-llama-8001.service",
            pattern_key="LLAMA_SECONDARY_PROCESS_PATTERN",
            pattern_default="llama-server.*--port 8001",
            host_key="LLAMA_SECONDARY_HOST",
            host_default=self.settings.llama_host,
            port_key="LLAMA_SECONDARY_PORT",
            port_default=8001,
            log_key="LLAMA_SECONDARY_LOG_FILE",
            log_default=str(self.settings.project_root / "logs" / "llama-8001.log"),
            configured_key="ENABLE_LLAMA_SECONDARY_SERVICE",
            configured_default=False,
            workdir_key="LLAMA_SECONDARY_WORKDIR",
            command_key="LLAMA_SECONDARY_COMMAND",
            instance_id="secondary",
        )

    def llama_instances(self) -> dict[str, Any]:
        primary = self.llama_status()
        secondary = self.llama_secondary_status()
        return {
            "ok": True,
            "instances": [primary, secondary],
            "by_id": {"primary": primary, "secondary": secondary},
        }

    def _instance_spec(self, instance_id: str) -> dict[str, str]:
        normalized = instance_id.strip().lower()
        if normalized in {"primary", "main", "1", "llama"}:
            return {
                "id": "primary",
                "service": "ubuntu-llama.service",
                "command_key": "LLAMA_COMMAND",
                "status": "primary",
            }
        if normalized in {"secondary", "second", "2", "8001", "llama-secondary"}:
            return {
                "id": "secondary",
                "service": "ubuntu-llama-8001.service",
                "command_key": "LLAMA_SECONDARY_COMMAND",
                "status": "secondary",
            }
        raise ValueError(f"unknown llama instance: {instance_id}")

    def llama_instance_status(self, instance_id: str) -> dict[str, Any]:
        spec = self._instance_spec(instance_id)
        if spec["status"] == "primary":
            return self.llama_status()
        return self.llama_secondary_status()

    def llama_instance_action(self, instance_id: str, action: str) -> dict[str, Any]:
        spec = self._instance_spec(instance_id)
        if action not in {"start", "stop", "restart"}:
            return {"ok": False, "error": f"Unsupported llama action: {action}"}
        result = systemctl(action, spec["service"], timeout=60)
        result["status"] = self.llama_instance_status(spec["id"])
        return result

    def llama_instance_reload(self, instance_id: str) -> dict[str, Any]:
        spec = self._instance_spec(instance_id)
        stop = systemctl("stop", spec["service"], timeout=120)
        start = systemctl("start", spec["service"], timeout=120)
        return {
            "ok": bool(stop.get("ok")) and bool(start.get("ok")),
            "note": "Explicit stop/start so the old model is unloaded before the new command starts.",
            "stop": stop,
            "start": start,
            "status": self.llama_instance_status(spec["id"]),
        }

    def llama_instance_configure(self, instance_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            spec = self._instance_spec(instance_id)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        command_key = spec["command_key"]
        current_command = self.settings.get(command_key, "")
        restart = str(payload.get("restart", "true")).lower() in {"1", "true", "yes", "on"}
        command = str(payload.get("command", "")).strip()
        model = str(payload.get("model", "")).strip()
        model_flag = str(payload.get("model_flag", "auto")).strip() or "auto"
        context_size = payload.get("context_size", payload.get("ctx_size", ""))

        try:
            if command:
                updated = update_llama_command(self.settings.config_path, command_key, current_command, command)
            else:
                updated = patch_llama_command(
                    self.settings.config_path,
                    command_key,
                    current_command,
                    model=model,
                    model_flag=model_flag,
                    context_size=context_size,
                )
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "instance": spec["id"], "command_key": command_key}

        self.reload_settings()
        result: dict[str, Any] = {
            "ok": True,
            "instance": spec["id"],
            "updated": updated,
            "status": self.llama_instance_status(spec["id"]),
        }
        if restart:
            result["reload"] = self.llama_instance_reload(spec["id"])
        return result

    def llama_action(self, action: str) -> dict[str, Any]:
        if action not in {"start", "stop", "restart"}:
            return {"ok": False, "error": f"Unsupported llama action: {action}"}
        result = systemctl(action, "ubuntu-llama.service", timeout=60)
        result["status"] = self.llama_status()
        return result

    def llama_secondary_action(self, action: str) -> dict[str, Any]:
        if action not in {"start", "stop", "restart"}:
            return {"ok": False, "error": f"Unsupported secondary llama action: {action}"}
        result = systemctl(action, "ubuntu-llama-8001.service", timeout=60)
        result["status"] = self.llama_secondary_status()
        return result

    def llama_force_kill(self) -> dict[str, Any]:
        result = force_kill_llama(self.settings)
        result["status"] = self.llama_status()
        result["vram"] = vram_summary(self.settings)
        return result

    def llama_switch_model(self, model: str, model_flag: str = "auto", restart: bool = True) -> dict[str, Any]:
        if not model:
            return {"ok": False, "error": "Missing model"}
        updated = update_llama_model(self.settings.config_path, self.settings.get("LLAMA_COMMAND", ""), model, model_flag)
        self.reload_settings()
        result: dict[str, Any] = {"ok": True, "updated": updated}
        if restart:
            result["reload"] = self.llama_instance_reload("primary")
        return result

    def gpu_diagnostics(self) -> dict[str, Any]:
        return diagnose_gpu(self.settings)

    def handle_gpu_fault(self, reason: str = "api-request") -> dict[str, Any]:
        diagnostics = diagnose_gpu(self.settings)
        return handle_gpu_fault(self.settings, self, diagnostics, reason=reason)

    def recover_llama_no_response(self, payload: dict[str, Any] | str | None = None) -> dict[str, Any]:
        if isinstance(payload, str):
            request_payload: dict[str, Any] = {"reason": payload}
        else:
            request_payload = payload or {}
        reason = str(request_payload.get("reason", "ai-stack-no-response"))
        diagnose_only = str(request_payload.get("diagnose_only", "false")).lower() in {"1", "true", "yes", "on"}
        probe = probe_llama_generation(self.settings, request_payload)
        if probe.get("ok"):
            result = {
                "ok": True,
                "decision": "llama-responsive",
                "reason": reason,
                "probe": probe,
                "action": {"executed": False, "note": "Llama generated content during probe."},
                "status": self.llama_status(),
            }
            self.write_json_state("last-ai-stack-recovery.json", result)
            return result

        diagnostics = diagnose_gpu(self.settings)
        if diagnose_only:
            decision = "gpu-critical" if diagnostics.get("critical") else "llama-hung"
            result = {
                "ok": True,
                "decision": decision,
                "reason": reason,
                "probe": probe,
                "gpu": {
                    "critical": diagnostics.get("critical", False),
                    "decision": diagnostics.get("decision", "unknown"),
                    "command_failures": diagnostics.get("command_failures", []),
                    "matches": diagnostics.get("matches", []),
                },
                "action": {"executed": False, "note": "diagnose_only=true"},
                "status": self.llama_status(),
            }
            self.write_json_state("last-ai-stack-recovery.json", result)
            return result

        if diagnostics.get("critical"):
            result = {
                "ok": True,
                "decision": "gpu-critical",
                "reason": reason,
                "probe": probe,
                "recovery": handle_gpu_fault(self.settings, self, diagnostics, reason=reason),
            }
            self.write_json_state("last-ai-stack-recovery.json", result)
            return result

        killed = force_kill_llama(self.settings)
        restart = systemctl("restart", "ubuntu-llama.service", timeout=60)
        result = {
            "ok": True,
            "decision": "llama-hung",
            "reason": reason,
            "probe": probe,
            "gpu": {
                "critical": False,
                "command_failures": diagnostics.get("command_failures", []),
                "matches": diagnostics.get("matches", []),
            },
            "kill": killed,
            "restart": restart,
            "status": self.llama_status(),
        }
        self.write_json_state("last-ai-stack-recovery.json", result)
        return result

    def reboot_status(self) -> dict[str, Any]:
        backend = self.settings.get("REBOOT_BACKEND", "timer")
        return {
            "auto_reboot_enabled": self.settings.bool("ENABLE_AUTO_REBOOT", self.settings.bool("ENABLE_REBOOT_TIMER", True)),
            "backend": backend,
            "timer_active": service_active("llama-reboot.timer"),
            "timer_enabled": service_enabled("llama-reboot.timer"),
            "watchdog_active": service_active("ubuntu-reboot-watch.service"),
            "watchdog_enabled": service_enabled("ubuntu-reboot-watch.service"),
            "reboot_interval_seconds": self.settings.reboot_interval_seconds,
            "reboot_interval_hours": self.settings.reboot_interval_hours,
        }

    def reboot_enable(self) -> dict[str, Any]:
        backend = self.settings.get("REBOOT_BACKEND", "timer")
        if backend == "watchdog" or self.settings.get("REBOOT_TIMER_MODE", "boot") == "llama-start":
            result = systemctl("enable", "--now", "ubuntu-reboot-watch.service", timeout=30)
        else:
            result = systemctl("enable", "--now", "llama-reboot.timer", timeout=30)
        result["status"] = self.reboot_status()
        return result

    def reboot_disable(self) -> dict[str, Any]:
        timer = systemctl("disable", "--now", "llama-reboot.timer", timeout=30)
        watchdog = systemctl("disable", "--now", "ubuntu-reboot-watch.service", timeout=30)
        return {"ok": timer["ok"] or watchdog["ok"], "timer": timer, "watchdog": watchdog, "status": self.reboot_status()}

    def reboot_now(self) -> dict[str, Any]:
        return run_command([str(self.settings.project_root / "bin" / "reboot-now.sh"), "api"], timeout=5)

    def shutdown_now(self) -> dict[str, Any]:
        return run_command(["systemctl", "poweroff"], timeout=5)

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "ubuntu-llama-manager",
            "time": now_iso(),
            "llama": self.llama_status(),
            "llama_secondary": self.llama_secondary_status(),
            "reboot": self.reboot_status(),
            "gpu_power_service": {
                "active": service_active("ubuntu-gpu-power.service"),
                "enabled": service_enabled("ubuntu-gpu-power.service"),
                "configured": self.settings.bool("ENABLE_GPU_POWER_LIMIT", False),
                "watts": self.settings.get("POWER_LIMIT_WATTS", ""),
            },
            "gpu_health_monitor": {
                "active": service_active("ubuntu-gpu-health.service"),
                "enabled": service_enabled("ubuntu-gpu-health.service"),
                "configured": self.settings.bool("ENABLE_GPU_HEALTH_MONITOR", False),
                "critical_action": self.settings.get("GPU_HEALTH_CRITICAL_ACTION", "none"),
                "poll_seconds": self.settings.int("GPU_HEALTH_POLL_SECONDS", 30),
            },
            "api": {
                "host": self.settings.api_host,
                "port": self.settings.api_port,
                "auth_required_for_dangerous_endpoints": True,
            },
        }

    def read_json_state(self, name: str, default: dict[str, Any]) -> dict[str, Any]:
        path = self.settings.state_dir / name
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def write_json_state(self, name: str, payload: dict[str, Any]) -> Path:
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.state_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
