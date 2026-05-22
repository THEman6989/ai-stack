from __future__ import annotations

import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from typing import Any
from urllib.parse import urljoin

from .config import Settings
from .esp import request_action

if TYPE_CHECKING:
    from .services import Manager


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


def llama_base_url(settings: Settings) -> str:
    explicit = settings.get("LLAMA_PROBE_BASE_URL", "")
    if explicit:
        return explicit.rstrip("/")

    host = settings.llama_host
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{settings.llama_port}"


def parse_llama_probe_content(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("content", "response", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    choices = payload.get("choices")
    if isinstance(choices, list):
        texts: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            text = choice.get("text")
            if isinstance(text, str):
                texts.append(text)
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                texts.append(message["content"])
        return "".join(texts)

    return ""


def probe_llama_generation(settings: Settings, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    base_url = llama_base_url(settings)
    path = str(payload.get("probe_path") or settings.get("LLAMA_PROBE_PATH", "/completion"))
    prompt = str(
        payload.get("probe_prompt")
        or settings.get("LLAMA_PROBE_PROMPT", "Reply with exactly: ok")
    )
    max_tokens = int(payload.get("probe_max_tokens") or settings.int("LLAMA_PROBE_MAX_TOKENS", 8))
    timeout = int(payload.get("probe_timeout_seconds") or settings.int("LLAMA_PROBE_TIMEOUT_SECONDS", 20))
    require_content = settings.bool("LLAMA_PROBE_REQUIRE_CONTENT", True)

    url = urljoin(f"{base_url}/", path.lstrip("/"))
    body = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "stream": False,
        "temperature": 0,
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})

    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(65536).decode("utf-8", errors="replace")
            elapsed_ms = int((time.monotonic() - started) * 1000)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            content = parse_llama_probe_content(parsed)
            ok = 200 <= response.status < 300 and (bool(content.strip()) or not require_content)
            return {
                "ok": ok,
                "url": url,
                "status_code": response.status,
                "elapsed_ms": elapsed_ms,
                "content_received": bool(content.strip()),
                "content_preview": content[:200],
                "error": "" if ok else "no generated content received",
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read(4096).decode("utf-8", errors="replace")
        return {
            "ok": False,
            "url": url,
            "status_code": exc.code,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "content_received": False,
            "error": raw or str(exc),
        }
    except (urllib.error.URLError, TimeoutError) as exc:
        return {
            "ok": False,
            "url": url,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "content_received": False,
            "error": str(exc),
        }


def notify_esp(settings: Settings, manager: "Manager", reason: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
    action = settings.get("ESP_POWER_ACTION_ON_GPU_FAULT", "power-cycle")
    payload = {
        "reason": reason,
        "requested_at": now_iso(),
        "hold_seconds": settings.int("ESP_POWER_HOLD_SECONDS", 12),
        "wait_seconds": settings.int("ESP_POWER_WAIT_SECONDS", 20),
        "delay_before_action_seconds": settings.int("ESP_POWER_DELAY_BEFORE_ACTION_SECONDS", 30),
        "diagnostics_summary": {
            "critical": diagnostics.get("critical", False),
            "matches": diagnostics.get("matches", [])[:10],
            "command_failures": diagnostics.get("command_failures", []),
        },
    }
    queued = request_action(manager, action, payload)
    webhook_url = settings.get("ESP_WEBHOOK_URL", "")
    if not webhook_url:
        queued["webhook"] = {"ok": False, "skipped": True, "reason": "ESP_WEBHOOK_URL empty"}
        return queued

    data = json.dumps({"action": action, **payload}).encode("utf-8")
    token = settings.get("ESP_WEBHOOK_TOKEN", "")
    timeout = max(1, settings.int("ESP_ACTION_TIMEOUT_SECONDS", 5))
    retries = max(1, settings.int("GPU_FAULT_ESP_RETRIES", 3))
    retry_sleep = max(0, settings.int("GPU_FAULT_ESP_RETRY_SECONDS", 2))
    attempts: list[dict[str, Any]] = []

    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            webhook_url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if token:
            request.add_header("Authorization", f"Bearer {token}")

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                result = {"ok": True, "attempt": attempt, "status": response.status, "body": body}
                attempts.append(result)
                queued["webhook"] = {**result, "attempts": attempts}
                return queued
        except urllib.error.HTTPError as exc:
            body = exc.read(4096).decode("utf-8", errors="replace")
            attempts.append({"ok": False, "attempt": attempt, "status": exc.code, "error": body or str(exc)})
        except (urllib.error.URLError, TimeoutError) as exc:
            attempts.append({"ok": False, "attempt": attempt, "error": str(exc)})

        if attempt < retries and retry_sleep:
            time.sleep(retry_sleep)

    queued["webhook"] = {"ok": False, "attempts": attempts, "error": "ESP webhook did not confirm request"}
    return queued


def force_kill_llama(settings: Settings) -> dict[str, Any]:
    pattern = settings.get("LLAMA_PROCESS_PATTERN", "llama-server")
    user = settings.get("RUN_AS_USER", "")
    args = ["pkill", "-9", "-f", "--", pattern]
    if user:
        args = ["pkill", "-9", "-u", user, "-f", "--", pattern]
    result = run_command(args, timeout=15)
    stop = systemctl("stop", "ubuntu-llama.service", timeout=30)
    return {"ok": result["ok"] or result["returncode"] == 1, "pkill": result, "service_stop": stop}


def handle_gpu_fault(settings: Settings, manager: "Manager", diagnostics: dict[str, Any], reason: str = "gpu-health-critical") -> dict[str, Any]:
    event = {
        "reason": reason,
        "time": now_iso(),
        "diagnostics": {
            "critical": diagnostics.get("critical", False),
            "matches": diagnostics.get("matches", [])[:25],
            "command_failures": diagnostics.get("command_failures", []),
        },
    }
    manager.write_json_state("last-gpu-fault.json", event)
    esp = notify_esp(settings, manager, reason, diagnostics)
    action = settings.get("GPU_HEALTH_CRITICAL_ACTION", "none").lower()

    action_result: dict[str, Any]
    if action in {"none", "log"}:
        action_result = {"ok": True, "action": action, "executed": False}
    elif action == "shutdown":
        if settings.bool("GPU_FAULT_REQUIRE_ESP_WEBHOOK", True) and not esp.get("webhook", {}).get("ok"):
            action_result = {
                "ok": False,
                "action": "shutdown",
                "executed": False,
                "error": "ESP webhook was not confirmed; refusing local shutdown for GPU fault.",
            }
            return {"ok": False, "event": event, "esp": esp, "action": action_result}
        if not settings.bool("GPU_FAULT_LOCAL_SHUTDOWN_AFTER_ESP", True):
            action_result = {
                "ok": True,
                "action": "shutdown",
                "executed": False,
                "note": "GPU_FAULT_LOCAL_SHUTDOWN_AFTER_ESP=false; ESP is responsible for the next power action.",
            }
            return {"ok": True, "event": event, "esp": esp, "action": action_result}
        settle_seconds = max(0, settings.int("ESP_NOTIFY_SETTLE_SECONDS", 2))
        if settle_seconds:
            time.sleep(settle_seconds)
        shutdown_command = settings.get("GPU_FAULT_SHUTDOWN_COMMAND", "/usr/bin/systemctl poweroff")
        action_result = run_command(shlex.split(shutdown_command), timeout=5)
    elif action == "reboot":
        action_result = run_command([str(settings.project_root / "bin" / "reboot-now.sh"), "gpu-health"], timeout=5)
    elif action == "force-kill-llama":
        action_result = force_kill_llama(settings)
    else:
        action_result = {"ok": False, "error": f"Unknown GPU_HEALTH_CRITICAL_ACTION: {action}"}

    return {"ok": True, "event": event, "esp": esp, "action": action_result}
