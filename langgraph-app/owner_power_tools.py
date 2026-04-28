from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _owner_password() -> str:
    return _env("ALPHARAVIS_OWNER_SSH_PASS", _env("SSH_PASS_DEFAULT", ""))


OWNER_POWER_CONFIG: dict[str, Any] = {
    # Owner-only defaults derived from the exported OpenWebUI tools.
    # Keep passwords out of git. Set ALPHARAVIS_OWNER_SSH_PASS or SSH_PASS_DEFAULT.
    "wake_host": {
        "ip": "100.105.33.53",
        "user": "amin",
        "password": "",
    },
    "llama": {
        "ip": "100.71.57.22",
        "mac": "b4:2e:99:89:35:14",
        "user": "amin",
        "password": "",
        "ssh_port": 22,
        "api_port": 8033,
        "log_path": "/home/amin/llama.log",
        "start_command": (
            "sh -c 'nohup /home/amin/experi/llama.cpp-gfx906/build/bin/llama-server "
            "-hf ggml-org/gpt-oss-120b-GGUF --jinja -c 32768 --host 0.0.0.0 "
            "--port 8033 -fa on -ngl 999 -sm layer --no-mmap -b 2048 -ub 1800 "
            "-ctk q8_0 -ctv q8_0 --repeat-penalty 1.1 > /home/amin/llama.log "
            "2>&1 < /dev/null &'"
        ),
    },
    "comfy": {
        "ip": "100.100.176.89",
        "mac": "10:7c:61:47:07:d9",
        "user": "amin",
        "password": "",
        "ssh_port": 22,
        "api_port": 8188,
    },
}


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    action: str
    message: str
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    elapsed_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "message": self.message,
            "stdout": self.stdout[-4000:],
            "stderr": self.stderr[-4000:],
            "returncode": self.returncode,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _pc(name: str) -> dict[str, Any]:
    return dict(OWNER_POWER_CONFIG[name])


def _password(pc: dict[str, Any]) -> str:
    return str(pc.get("password") or _owner_password())


async def _run_exec(
    args: list[str],
    *,
    timeout: float = 20,
    stdin_text: str | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    started = time.perf_counter()
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE if stdin_text is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(stdin_text.encode("utf-8") if stdin_text is not None else None),
            timeout=timeout,
        )
        return CommandResult(
            ok=proc.returncode == 0,
            action=args[0] if args else "command",
            message="command completed" if proc.returncode == 0 else "command failed",
            stdout=stdout.decode("utf-8", errors="replace").strip(),
            stderr=stderr.decode("utf-8", errors="replace").strip(),
            returncode=proc.returncode,
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
    except asyncio.TimeoutError:
        return CommandResult(
            ok=False,
            action=args[0] if args else "command",
            message=f"command timed out after {timeout:g}s",
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
    except FileNotFoundError as exc:
        return CommandResult(ok=False, action=args[0] if args else "command", message=f"missing executable: {exc}")
    except Exception as exc:
        return CommandResult(ok=False, action=args[0] if args else "command", message=str(exc))


async def _tcp_open(host: str, port: int, *, timeout: float = 1.5) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def _ssh(pc: dict[str, Any], remote_command: str, *, timeout: float = 20, stdin_text: str | None = None) -> CommandResult:
    password = _password(pc)
    if not password:
        return CommandResult(ok=False, action="ssh", message="missing ALPHARAVIS_OWNER_SSH_PASS or SSH_PASS_DEFAULT")

    args = [
        "sshpass",
        "-e",
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=8",
        "-p",
        str(pc.get("ssh_port", 22)),
        f"{pc['user']}@{pc['ip']}",
        remote_command,
    ]
    ssh_env = dict(os.environ)
    ssh_env["SSHPASS"] = password
    return await _run_exec(args, timeout=timeout, stdin_text=stdin_text, env=ssh_env)


async def _wake(mac: str) -> CommandResult:
    wake_host = _pc("wake_host")
    wake_host["password"] = wake_host.get("password") or _owner_password()
    return await _ssh(wake_host, f"wakeonlan {mac}", timeout=15)


async def owner_check_llama_server() -> dict[str, Any]:
    llama = _pc("llama")
    api_open, ssh_open = await asyncio.gather(
        _tcp_open(llama["ip"], int(llama["api_port"])),
        _tcp_open(llama["ip"], int(llama["ssh_port"])),
    )
    process = None
    if ssh_open:
        process = (await _ssh(llama, "pgrep -f llama-server", timeout=10)).as_dict()
    return {
        "ok": api_open,
        "api_open": api_open,
        "ssh_open": ssh_open,
        "process_check": process,
        "host": llama["ip"],
        "api_port": llama["api_port"],
    }


async def owner_get_llama_logs(lines: int = 80) -> dict[str, Any]:
    llama = _pc("llama")
    safe_lines = max(1, min(int(lines), 500))
    result = await _ssh(llama, f"tail -n {safe_lines} {llama['log_path']}", timeout=15)
    return result.as_dict()


async def owner_start_llama_server(wait_seconds: int = 90) -> dict[str, Any]:
    llama = _pc("llama")
    if await _tcp_open(llama["ip"], int(llama["api_port"])):
        return {"ok": True, "message": "llama server already reachable", "host": llama["ip"]}

    if not await _tcp_open(llama["ip"], int(llama["ssh_port"])):
        wake = await _wake(llama["mac"])
        if not wake.ok:
            return {"ok": False, "message": "failed to send wake signal", "wake": wake.as_dict()}
        boot_deadline = time.monotonic() + min(max(wait_seconds, 10), 240)
        while time.monotonic() < boot_deadline:
            if await _tcp_open(llama["ip"], int(llama["ssh_port"]), timeout=2):
                break
            await asyncio.sleep(3)

    start = await _ssh(llama, str(llama["start_command"]), timeout=5)
    if not start.ok and "timed out" not in start.message:
        return {"ok": False, "message": "failed to send llama start command", "start": start.as_dict()}

    deadline = time.monotonic() + min(max(wait_seconds, 10), 240)
    while time.monotonic() < deadline:
        if await _tcp_open(llama["ip"], int(llama["api_port"]), timeout=2):
            return {"ok": True, "message": "llama server reachable", "start": start.as_dict()}
        await asyncio.sleep(3)

    logs = await owner_get_llama_logs(80)
    return {"ok": False, "message": "llama start timed out; logs attached", "start": start.as_dict(), "logs": logs}


async def owner_restart_llama_server(wait_seconds: int = 90) -> dict[str, Any]:
    llama = _pc("llama")
    stop = await _ssh(llama, "pkill -f llama-server || true", timeout=15)
    await asyncio.sleep(2)
    start = await owner_start_llama_server(wait_seconds=wait_seconds)
    return {"ok": bool(start.get("ok")), "message": "restart attempted", "stop": stop.as_dict(), "start": start}


async def owner_shutdown_llama_server() -> dict[str, Any]:
    llama = _pc("llama")
    password = _password(llama)
    if not password:
        return {"ok": False, "message": "missing ALPHARAVIS_OWNER_SSH_PASS or SSH_PASS_DEFAULT"}
    result = await _ssh(llama, "sudo -S -p '' shutdown -h now", timeout=10, stdin_text=f"{password}\n")
    if result.returncode == 255 and "closed" in result.stderr.lower():
        data = result.as_dict()
        data["ok"] = True
        data["message"] = "shutdown command sent; SSH connection closed"
        return data
    return result.as_dict()


async def owner_check_comfyui_server() -> dict[str, Any]:
    comfy = _pc("comfy")
    api_open, ssh_open = await asyncio.gather(
        _tcp_open(comfy["ip"], int(comfy["api_port"])),
        _tcp_open(comfy["ip"], int(comfy["ssh_port"])),
    )
    return {
        "ok": api_open or ssh_open,
        "api_open": api_open,
        "ssh_open": ssh_open,
        "host": comfy["ip"],
        "api_port": comfy["api_port"],
    }


async def owner_start_comfyui_server() -> dict[str, Any]:
    comfy = _pc("comfy")
    if await _tcp_open(comfy["ip"], int(comfy["api_port"])):
        return {"ok": True, "message": "comfyui already reachable", "host": comfy["ip"]}
    wake = await _wake(comfy["mac"])
    return {"ok": wake.ok, "message": "comfyui wake signal sent" if wake.ok else "comfyui wake failed", "wake": wake.as_dict()}


async def owner_shutdown_comfyui_server() -> dict[str, Any]:
    comfy = _pc("comfy")
    password = _password(comfy)
    if not password:
        return {"ok": False, "message": "missing ALPHARAVIS_OWNER_SSH_PASS or SSH_PASS_DEFAULT"}
    result = await _ssh(comfy, "sudo -S -p '' shutdown -h now", timeout=10, stdin_text=f"{password}\n")
    if result.returncode == 255 and "closed" in result.stderr.lower():
        data = result.as_dict()
        data["ok"] = True
        data["message"] = "shutdown command sent; SSH connection closed"
        return data
    return result.as_dict()


async def owner_start_all_model_services(wait_seconds: int = 90) -> dict[str, Any]:
    comfy_task = owner_start_comfyui_server()
    llama_task = owner_start_llama_server(wait_seconds=wait_seconds)
    comfy, llama = await asyncio.gather(comfy_task, llama_task)
    return {"ok": bool(llama.get("ok")), "comfy": comfy, "llama": llama}


async def owner_get_pixelle_logs(lines: int = 80) -> dict[str, Any]:
    safe_lines = max(1, min(int(lines), 500))
    try:
        result = subprocess.run(
            ["docker", "logs", "pixelle", "--tail", str(safe_lines)],
            text=True,
            capture_output=True,
            timeout=20,
        )
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout[-6000:],
            "stderr": result.stderr[-3000:],
            "returncode": result.returncode,
        }
    except Exception as exc:
        return {"ok": False, "message": str(exc)}
