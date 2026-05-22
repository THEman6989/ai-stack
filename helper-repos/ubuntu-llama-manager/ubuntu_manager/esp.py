from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from .config import Settings

if TYPE_CHECKING:
    from .services import Manager


HEARTBEAT_FILE = "esp-heartbeat.json"
REQUEST_FILE = "esp-request.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def heartbeat(manager: Manager, payload: dict[str, Any]) -> dict[str, Any]:
    body = {
        "device_id": str(payload.get("device_id", "unknown")),
        "status": str(payload.get("status", "online")),
        "uptime_seconds": int(payload.get("uptime_seconds", 0) or 0),
        "received_at": now_iso(),
        "raw": payload,
    }
    path = manager.write_json_state(HEARTBEAT_FILE, body)
    return {"ok": True, "saved_to": str(path), "heartbeat": body}


def status_url_from_webhook(webhook_url: str) -> str:
    if not webhook_url:
        return ""
    parsed = urlparse(webhook_url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    path = parsed.path.rstrip("/")
    if path.endswith("/action"):
        path = path.removesuffix("/action")
    status_path = f"{path}/status" if path else "/status"
    return urlunparse((parsed.scheme, parsed.netloc, status_path, "", "", ""))


def sibling_url_from_webhook(webhook_url: str, sibling: str) -> str:
    if not webhook_url:
        return ""
    parsed = urlparse(webhook_url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    path = parsed.path.rstrip("/")
    if path.endswith("/action"):
        path = path.removesuffix("/action")
    sibling_path = f"{path}/{sibling.lstrip('/')}" if path else f"/{sibling.lstrip('/')}"
    return urlunparse((parsed.scheme, parsed.netloc, sibling_path, "", "", ""))


def direct_status(settings: Settings) -> dict[str, Any]:
    url = status_url_from_webhook(settings.get("ESP_WEBHOOK_URL", ""))
    if not url:
        return {"ok": False, "skipped": True, "reason": "ESP_WEBHOOK_URL empty"}

    timeout = settings.float("ESP_STATUS_TIMEOUT_SECONDS", 2.0)
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(32768).decode("utf-8")
            body = json.loads(raw)
            return {"ok": True, "url": url, "status_code": response.status, "body": body}
    except json.JSONDecodeError as exc:
        return {"ok": False, "url": url, "error": f"invalid json: {exc}"}
    except HTTPError as exc:
        return {"ok": False, "url": url, "status_code": exc.code, "error": str(exc)}
    except URLError as exc:
        return {"ok": False, "url": url, "error": str(exc.reason)}
    except TimeoutError:
        return {"ok": False, "url": url, "error": "timeout"}


def post_esp(settings: Settings, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not url:
        return {"ok": False, "error": "ESP_WEBHOOK_URL empty"}
    timeout = settings.float("ESP_ACTION_TIMEOUT_SECONDS", 5.0)
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    token = settings.get("ESP_WEBHOOK_TOKEN", "")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(32768).decode("utf-8", errors="replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = raw
            return {"ok": 200 <= response.status < 300, "url": url, "status_code": response.status, "body": body}
    except HTTPError as exc:
        raw = exc.read(32768).decode("utf-8", errors="replace")
        return {"ok": False, "url": url, "status_code": exc.code, "error": raw or str(exc)}
    except URLError as exc:
        return {"ok": False, "url": url, "error": str(exc.reason)}
    except TimeoutError:
        return {"ok": False, "url": url, "error": "timeout"}


def direct_action(settings: Settings, manager: Manager, payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action", "")).strip()
    allowed = {"power-on", "power-off", "power-cycle", "reset"}
    if action not in allowed:
        return {"ok": False, "error": f"Unsupported ESP action: {action}", "allowed": sorted(allowed)}

    body = {
        "action": action,
        "reason": str(payload.get("reason", "manager-web-control")),
        "hold_seconds": int(payload.get("hold_seconds", 1) or 1),
        "wait_seconds": int(payload.get("wait_seconds", 20) or 20),
        "delay_before_action_seconds": int(payload.get("delay_before_action_seconds", 0) or 0),
        "requested_by": str(payload.get("requested_by", "ubuntu-llama-manager")),
    }
    queued = request_action(manager, action, body)
    dry_run = str(payload.get("dry_run", "false")).lower() in {"1", "true", "yes", "on"}
    if dry_run:
        return {"ok": True, "dry_run": True, "queued": queued, "payload": body}
    webhook_url = settings.get("ESP_WEBHOOK_URL", "")
    sent = post_esp(settings, webhook_url, body)
    return {"ok": bool(sent.get("ok")), "queued": queued, "sent": sent, "payload": body}


def direct_cancel(settings: Settings) -> dict[str, Any]:
    cancel_url = sibling_url_from_webhook(settings.get("ESP_WEBHOOK_URL", ""), "cancel")
    return post_esp(settings, cancel_url, {})


def direct_pin_test(settings: Settings, payload: dict[str, Any]) -> dict[str, Any]:
    pin = str(payload.get("pin", "power")).strip().lower()
    level = str(payload.get("level", "float")).strip().lower()
    if pin not in {"power", "reset"}:
        return {"ok": False, "error": "Unsupported pin", "allowed_pins": ["power", "reset"]}
    if level not in {"high", "low", "float"}:
        return {"ok": False, "error": "Unsupported level", "allowed_levels": ["high", "low", "float"]}

    body = {
        "pin": pin,
        "level": level,
        "hold_seconds": int(payload.get("hold_seconds", 5) or 5),
        "reason": str(payload.get("reason", "manager-pin-test")),
    }
    url = sibling_url_from_webhook(settings.get("ESP_WEBHOOK_URL", ""), "pin-test")
    sent = post_esp(settings, url, body)
    return {"ok": bool(sent.get("ok")), "sent": sent, "payload": body}


def status(settings: Settings, manager: Manager) -> dict[str, Any]:
    heartbeat_state = manager.read_json_state(HEARTBEAT_FILE, {})
    request_state = manager.read_json_state(REQUEST_FILE, {})
    last_heartbeat = heartbeat_state.get("received_at")
    stale_after = settings.int("ESP_STALE_AFTER_SECONDS", 90)
    last_dt = parse_time(last_heartbeat) if last_heartbeat else None
    online = False
    if last_dt:
        age = (datetime.now(timezone.utc) - last_dt).total_seconds()
        online = age <= stale_after
    direct = direct_status(settings)
    direct_body = direct.get("body") if direct.get("ok") else {}
    direct_online = bool(direct.get("ok") and direct_body.get("wifi_connected", True))

    return {
        "esp_online": online or direct_online,
        "last_heartbeat": last_heartbeat,
        "device_id": heartbeat_state.get("device_id") or direct_body.get("device_id"),
        "status": heartbeat_state.get("status") or ("online" if direct_online else None),
        "stale_after_seconds": stale_after,
        "direct_status": direct,
        "pending_request": request_state or None,
    }


def request_action(manager: Manager, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = {
        "ok": True,
        "action": action,
        "requested_at": now_iso(),
        "status": "queued",
        "note": "Prepared interface only. No hardware action is executed by this server yet.",
        "payload": payload or {},
    }
    path = manager.write_json_state(REQUEST_FILE, body)
    body["saved_to"] = str(path)
    return body
