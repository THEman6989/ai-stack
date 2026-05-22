from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlparse

from .config import Settings
from .esp import direct_action as esp_direct_action
from .esp import direct_cancel as esp_direct_cancel
from .esp import direct_pin_test as esp_direct_pin_test
from .esp import heartbeat as esp_heartbeat
from .esp import request_action as esp_request_action
from .esp import status as esp_status
from .models import get_model, scan_models
from .services import Manager
from .web import ESP_CONTROL_HTML


PUBLIC_GET = {"/health", "/status", "/models"}
PUBLIC_POST = {"/esp/heartbeat"}
DANGEROUS_POST = {
    "/llama/start",
    "/llama/stop",
    "/llama/restart",
    "/llama/config",
    "/llama/force-kill",
    "/llama/switch-model",
    "/llama-secondary/start",
    "/llama-secondary/stop",
    "/llama-secondary/restart",
    "/llama-secondary/config",
    "/reboot/enable",
    "/reboot/disable",
    "/reboot/now",
    "/power/shutdown",
    "/diagnostics/handle-gpu-fault",
    "/ai-stack/diagnose-llama",
    "/ai-stack/llama-no-response",
    "/recovery/llama-no-response",
    "/esp/action",
    "/esp/cancel",
    "/esp/pin-test",
    "/esp/request-power-cycle",
    "/esp/request-power-on",
    "/esp/request-power-off",
}


def is_dangerous_post(path: str) -> bool:
    if path in DANGEROUS_POST:
        return True
    return path.startswith("/llama/instances/") and path.endswith("/config")


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "UbuntuLlamaManager/0.2"

    @property
    def settings(self) -> Settings:
        return self.server.settings  # type: ignore[attr-defined]

    @property
    def manager(self) -> Manager:
        return self.server.manager  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def authorized(self) -> bool:
        expected = self.settings.api_token
        supplied = self.headers.get("Authorization", "")
        return bool(expected) and supplied == f"Bearer {expected}"

    def require_auth(self) -> bool:
        if self.authorized():
            return True
        self.send_json({"ok": False, "error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED)
        return False

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/esp/control", "/esp/test"}:
            self.send_html(ESP_CONTROL_HTML)
            return
        if path == "/health":
            self.send_json(self.manager.health())
            return
        if path == "/status":
            self.send_json(self.manager.status())
            return
        if path == "/models":
            self.send_json({"ok": True, "models": scan_models(self.settings)})
            return
        if path.startswith("/models/"):
            model_id = unquote(path.removeprefix("/models/"))
            model = get_model(self.settings, model_id)
            if model:
                self.send_json({"ok": True, "model": model})
            else:
                self.send_json({"ok": False, "error": "Model not found"}, HTTPStatus.NOT_FOUND)
            return
        if path == "/llama/status":
            self.send_json({"ok": True, "llama": self.manager.llama_status()})
            return
        if path == "/llama/config":
            self.send_json({"ok": True, "instance": self.manager.llama_status()})
            return
        if path == "/llama-secondary/status":
            self.send_json({"ok": True, "llama_secondary": self.manager.llama_secondary_status()})
            return
        if path == "/llama-secondary/config":
            self.send_json({"ok": True, "instance": self.manager.llama_secondary_status()})
            return
        if path == "/llama/instances":
            self.send_json(self.manager.llama_instances())
            return
        if path.startswith("/llama/instances/"):
            instance_id = unquote(path.removeprefix("/llama/instances/")).strip("/")
            try:
                self.send_json({"ok": True, "instance": self.manager.llama_instance_status(instance_id)})
            except ValueError as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        if path == "/reboot/status":
            self.send_json({"ok": True, "reboot": self.manager.reboot_status()})
            return
        if path == "/esp/status":
            self.send_json({"ok": True, "esp": esp_status(self.settings, self.manager)})
            return
        if path == "/diagnostics/gpu":
            self.send_json({"ok": True, "gpu": self.manager.gpu_diagnostics()})
            return
        self.send_json({"ok": False, "error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if is_dangerous_post(path) and not self.require_auth():
            return
        if path == "/esp/heartbeat" and self.settings.bool("ESP_HEARTBEAT_AUTH_REQUIRED", False) and not self.require_auth():
            return

        try:
            payload = self.read_json()
        except json.JSONDecodeError:
            self.send_json({"ok": False, "error": "Invalid JSON"}, HTTPStatus.BAD_REQUEST)
            return

        if path == "/llama/start":
            self.send_json(self.manager.llama_action("start"))
            return
        if path == "/llama/stop":
            self.send_json(self.manager.llama_action("stop"))
            return
        if path == "/llama/restart":
            self.send_json(self.manager.llama_action("restart"))
            return
        if path == "/llama/config":
            self.send_json(self.manager.llama_instance_configure("primary", payload))
            return
        if path == "/llama-secondary/start":
            self.send_json(self.manager.llama_secondary_action("start"))
            return
        if path == "/llama-secondary/stop":
            self.send_json(self.manager.llama_secondary_action("stop"))
            return
        if path == "/llama-secondary/restart":
            self.send_json(self.manager.llama_secondary_action("restart"))
            return
        if path == "/llama-secondary/config":
            self.send_json(self.manager.llama_instance_configure("secondary", payload))
            return
        if path.startswith("/llama/instances/") and path.endswith("/config"):
            instance_id = unquote(path.removeprefix("/llama/instances/").removesuffix("/config")).strip("/")
            self.send_json(self.manager.llama_instance_configure(instance_id, payload))
            return
        if path == "/llama/force-kill":
            self.send_json(self.manager.llama_force_kill())
            return
        if path == "/llama/switch-model":
            self.send_json(
                self.manager.llama_switch_model(
                    str(payload.get("model", "")),
                    str(payload.get("model_flag", "auto")),
                    bool(payload.get("restart", True)),
                )
            )
            return
        if path == "/reboot/status":
            self.send_json({"ok": True, "reboot": self.manager.reboot_status()})
            return
        if path == "/reboot/enable":
            self.send_json(self.manager.reboot_enable())
            return
        if path == "/reboot/disable":
            self.send_json(self.manager.reboot_disable())
            return
        if path == "/reboot/now":
            self.send_json(self.manager.reboot_now(), HTTPStatus.ACCEPTED)
            return
        if path == "/power/shutdown":
            self.send_json(self.manager.shutdown_now(), HTTPStatus.ACCEPTED)
            return
        if path == "/diagnostics/handle-gpu-fault":
            self.send_json(self.manager.handle_gpu_fault(str(payload.get("reason", "api-request"))), HTTPStatus.ACCEPTED)
            return
        if path == "/ai-stack/diagnose-llama":
            payload["diagnose_only"] = True
            self.send_json(self.manager.recover_llama_no_response(payload), HTTPStatus.ACCEPTED)
            return
        if path in {"/recovery/llama-no-response", "/ai-stack/llama-no-response"}:
            self.send_json(self.manager.recover_llama_no_response(payload), HTTPStatus.ACCEPTED)
            return
        if path == "/esp/heartbeat":
            self.send_json(esp_heartbeat(self.manager, payload))
            return
        if path == "/esp/action":
            self.send_json(esp_direct_action(self.settings, self.manager, payload), HTTPStatus.ACCEPTED)
            return
        if path == "/esp/cancel":
            self.send_json(esp_direct_cancel(self.settings), HTTPStatus.ACCEPTED)
            return
        if path == "/esp/pin-test":
            self.send_json(esp_direct_pin_test(self.settings, payload), HTTPStatus.ACCEPTED)
            return
        if path == "/esp/request-power-cycle":
            self.send_json(esp_request_action(self.manager, "power-cycle", payload), HTTPStatus.ACCEPTED)
            return
        if path == "/esp/request-power-on":
            self.send_json(esp_request_action(self.manager, "power-on", payload), HTTPStatus.ACCEPTED)
            return
        if path == "/esp/request-power-off":
            self.send_json(esp_request_action(self.manager, "power-off", payload), HTTPStatus.ACCEPTED)
            return

        self.send_json({"ok": False, "error": "Not found"}, HTTPStatus.NOT_FOUND)


class ManagerServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], settings: Settings):
        super().__init__(address, handler)
        self.settings = settings
        self.manager = Manager(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ubuntu llama.cpp manager API")
    parser.add_argument("--config", default=None, help="Path to ubuntu-llama.conf/.env")
    args = parser.parse_args()

    settings = Settings.load(args.config)
    server = ManagerServer((settings.api_host, settings.api_port), ApiHandler, settings)
    print(f"API listening on http://{settings.api_host}:{settings.api_port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
