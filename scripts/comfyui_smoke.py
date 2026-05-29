#!/usr/bin/env python3
"""ComfyUI integration smoke checks for AlphaRavis.

Checks the host-direct ComfyUI API, optional Unix-socket relay file, media-gallery
proxy status/queue (`/comfyui/status`, `/comfyui/queue`), the fail-closed
`/comfyui/prompt` gate, and an optional `/comfyui/view` output fetch.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import stat
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _env(name: str, default: str = "", env_file: dict[str, str] | None = None) -> str:
    if name in os.environ:
        return os.environ[name]
    if env_file and name in env_file:
        return env_file[name]
    return default


def _normalize_base(value: str, default: str) -> str:
    value = (value or default).strip().rstrip("/")
    if value and "://" not in value and not value.startswith("unix://"):
        value = f"http://{value}"
    return value.rstrip("/")


def _http_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 5.0) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _http_bytes(url: str, *, timeout: float = 5.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read(32)


def _socket_is_alive(path: Path, *, timeout: float = 2.0) -> bool:
    if not path.exists():
        return False
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    if not stat.S_ISSOCK(mode):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(path))
        return True
    except OSError:
        return False


def _step(name: str, fn) -> dict[str, Any]:
    try:
        detail = fn()
        return {"name": name, "ok": True, "detail": detail}
    except Exception as exc:  # pragma: no cover - exercised by operator smokes
        return {"name": name, "ok": False, "error": str(exc)}


def _result_ok(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("ok") is not False and not payload.get("error")


def _submit_blocked(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    raw_result = payload.get("result")
    result: dict[str, Any] = raw_result if isinstance(raw_result, dict) else {}
    return bool(payload.get("ok") is False or payload.get("blocked") is True or result.get("blocked") is True or result.get("error"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AlphaRavis ComfyUI integration smoke checks.")
    env_file = _load_env_file(ROOT / ".env")
    direct_default = _env(
        "NEXT_PUBLIC_COMFYUI_PANEL_API_BASE",
        _env("ALPHARAVIS_COMFYUI_PUBLIC_BASE_URL", "http://127.0.0.1:8188", env_file),
        env_file,
    )
    proxy_default = _env("NEXT_PUBLIC_COMFYUI_PROXY_API_BASE", "http://127.0.0.1:8130/comfyui", env_file)
    api_base_default = _env("ALPHARAVIS_COMFYUI_API_BASE", "", env_file)
    parser.set_defaults(env_file=env_file, api_base=api_base_default)
    parser.add_argument("--direct-base", default=direct_default, help="Browser/host direct ComfyUI base URL (default from env or http://127.0.0.1:8188).")
    parser.add_argument("--proxy-base", default=proxy_default, help="Media-gallery ComfyUI proxy base URL (default from env or http://127.0.0.1:8130/comfyui).")
    parser.add_argument("--socket", default=_env("COMFYUI_RELAY_SOCKET", "runtime/comfyui.sock", env_file), help="Host path to the Unix relay socket.")
    parser.add_argument("--require-socket", choices=("auto", "true", "false"), default=_env("COMFYUI_SMOKE_REQUIRE_SOCKET", "auto", env_file))
    parser.add_argument("--view-filename", default=_env("COMFYUI_SMOKE_VIEW_FILENAME", "", env_file), help="Optional ComfyUI output filename for /comfyui/view smoke.")
    parser.add_argument("--view-subfolder", default=_env("COMFYUI_SMOKE_VIEW_SUBFOLDER", "", env_file))
    parser.add_argument("--view-type", default=_env("COMFYUI_SMOKE_VIEW_TYPE", "output", env_file))
    parser.add_argument("--timeout", type=float, default=float(_env("COMFYUI_SMOKE_TIMEOUT_SECONDS", "5", env_file)))
    parser.add_argument("--json", action="store_true", help="Print compact JSON only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    direct_base = _normalize_base(args.direct_base, "http://127.0.0.1:8188")
    proxy_base = _normalize_base(args.proxy_base, "http://127.0.0.1:8130/comfyui")
    socket_path = Path(args.socket)
    api_base = str(args.api_base or "")
    require_socket = args.require_socket == "true" or (args.require_socket == "auto" and (api_base.startswith("unix://") or socket_path.exists()))

    steps: list[dict[str, Any]] = []
    steps.append(_step("host_direct_system_stats", lambda: {"base_url": direct_base, "system_stats": _http_json(f"{direct_base}/system_stats", timeout=args.timeout)}))

    if require_socket:
        steps.append(_step("unix_socket_relay", lambda: {"socket": str(socket_path), "alive": _socket_is_alive(socket_path)}))
        if steps[-1]["ok"] and not steps[-1]["detail"].get("alive"):
            steps[-1] = {"name": "unix_socket_relay", "ok": False, "error": f"Unix socket is missing or not accepting connections: {socket_path}"}
    else:
        steps.append({"name": "unix_socket_relay", "ok": True, "skipped": True, "detail": {"socket": str(socket_path), "reason": "not configured/required"}})

    def proxy_status() -> dict[str, Any]:
        payload = _http_json(f"{proxy_base}/status", timeout=args.timeout)
        if not _result_ok(payload):
            raise RuntimeError(json.dumps(payload, ensure_ascii=False)[:500])
        return {"base_url": proxy_base, "payload_ok": payload.get("ok")}

    def proxy_queue() -> dict[str, Any]:
        payload = _http_json(f"{proxy_base}/queue", timeout=args.timeout)
        if not _result_ok(payload):
            raise RuntimeError(json.dumps(payload, ensure_ascii=False)[:500])
        queue = payload.get("queue") if isinstance(payload, dict) else {}
        pending = queue.get("queue_pending") if isinstance(queue, dict) else []
        running = queue.get("queue_running") if isinstance(queue, dict) else []
        return {"base_url": proxy_base, "pending": len(pending) if isinstance(pending, list) else 0, "running": len(running) if isinstance(running, list) else 0}

    def proxy_submit_blocked() -> dict[str, Any]:
        workflow = {"1": {"class_type": "AlphaRavisSmokeMissingNode", "inputs": {}}}
        payload = _http_json(f"{proxy_base}/prompt", method="POST", payload={"workflow": workflow, "client_id": "alpharavis-smoke"}, timeout=args.timeout)
        if not _submit_blocked(payload):
            raise RuntimeError(f"Expected fail-closed blocked submit response, got: {json.dumps(payload, ensure_ascii=False)[:500]}")
        return {"base_url": proxy_base, "blocked": True}

    steps.append(_step("proxy_status", proxy_status))
    steps.append(_step("proxy_queue", proxy_queue))
    steps.append(_step("proxy_prompt_fail_closed", proxy_submit_blocked))

    if args.view_filename:
        def proxy_view() -> dict[str, Any]:
            query = urllib.parse.urlencode({"filename": args.view_filename, "subfolder": args.view_subfolder, "type": args.view_type})
            head = _http_bytes(f"{proxy_base}/view?{query}", timeout=args.timeout)
            if not head:
                raise RuntimeError("empty /comfyui/view response")
            with tempfile.NamedTemporaryFile(prefix="alpharavis-comfyui-view-", suffix=".bin", delete=False) as fh:
                fh.write(head)
                sample_path = fh.name
            return {"base_url": proxy_base, "filename": args.view_filename, "sample_path": sample_path, "first_bytes": head[:8].hex()}
        steps.append(_step("proxy_view", proxy_view))
    else:
        steps.append({"name": "proxy_view", "ok": True, "skipped": True, "detail": {"reason": "set COMFYUI_SMOKE_VIEW_FILENAME or --view-filename to enable"}})

    ok = all(step.get("ok") for step in steps)
    result = {"ok": ok, "direct_base": direct_base, "proxy_base": proxy_base, "steps": steps}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
