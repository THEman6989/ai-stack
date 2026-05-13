from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse, urlunparse


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = ROOT / "service-dashboard-data" / "tailscale_service_urls.json"
LOCAL_PROXY_HOST = "127.0.0.1"


@dataclass(frozen=True)
class ServiceRoute:
    name: str
    service: str
    kind: str
    host_url: str
    local_target: str
    tailscale_url: str
    port: int
    path: str


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    sudo_password: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if sudo_password is not None:
        cmd = ["sudo", "-S", *cmd]
    kwargs: dict[str, Any] = {
        "cwd": ROOT,
        "text": True,
        "check": check,
    }
    if capture:
        kwargs["capture_output"] = True
    if sudo_password is not None:
        kwargs["input"] = sudo_password + "\n"
    return subprocess.run(cmd, **kwargs)


def load_redirector_services() -> list[dict[str, Any]]:
    try:
        import service_redirector_server
    except Exception as exc:
        raise SystemExit(
            "Could not import service_redirector_server.py. Run this script from the ai-stack root."
        ) from exc
    return list(service_redirector_server.SERVICES)


def is_local_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"} and parsed.port is not None


def normalize_local_target(host_url: str) -> tuple[str, int, str]:
    parsed = urlparse(host_url)
    if parsed.port is None:
        raise ValueError(f"Local URL has no port: {host_url}")
    path = parsed.path or ""
    target = urlunparse(("http", f"{LOCAL_PROXY_HOST}:{parsed.port}", path, "", "", ""))
    return target, parsed.port, path


def discover_tailscale_dns_name(*, sudo_password: str | None = None) -> str:
    result = run(["tailscale", "status", "--json"], sudo_password=sudo_password)
    data = json.loads(result.stdout)
    dns_name = str(data.get("Self", {}).get("DNSName", "")).rstrip(".")
    if not dns_name:
        raise SystemExit("Tailscale DNS name not found. Is `tailscale up` already connected?")
    return dns_name


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "service"


def build_routes(
    services: Iterable[dict[str, Any]],
    *,
    tailscale_host: str,
    include_self: bool,
) -> list[ServiceRoute]:
    routes: list[ServiceRoute] = []
    seen_ports: set[int] = set()
    for service in services:
        host_url = str(service.get("host_url", ""))
        if not is_local_http_url(host_url):
            continue
        service_name = str(service.get("service", ""))
        if not include_self and service_name == "service-dashboard":
            continue
        target, port, path = normalize_local_target(host_url)
        if port in seen_ports:
            continue
        seen_ports.add(port)
        parsed = urlparse(host_url)
        public_url = urlunparse(("https", f"{tailscale_host}:{port}", parsed.path or "", "", "", ""))
        routes.append(
            ServiceRoute(
                name=str(service.get("name", service_name or f"Port {port}")),
                service=service_name or slug(str(service.get("name", f"port-{port}"))),
                kind=str(service.get("kind", "")),
                host_url=host_url,
                local_target=target,
                tailscale_url=public_url,
                port=port,
                path=path,
            )
        )
    return routes


def serve_command(route: ServiceRoute, *, yes: bool = True) -> list[str]:
    cmd = ["tailscale", "serve", "--bg"]
    if yes:
        cmd.append("--yes")
    cmd.append(f"--https={route.port}")
    cmd.append(route.local_target)
    return cmd


def disable_command(route: ServiceRoute, *, yes: bool = True) -> list[str]:
    cmd = ["tailscale", "serve"]
    if yes:
        cmd.append("--yes")
    cmd.append(f"--https={route.port}")
    cmd.append("off")
    return cmd


def route_payload(routes: list[ServiceRoute], *, tailscale_host: str) -> dict[str, Any]:
    return {
        "tailscale_host": tailscale_host,
        "mode": "tailscale-serve-https",
        "routes": [
            {
                "name": route.name,
                "service": route.service,
                "kind": route.kind,
                "host_url": route.host_url,
                "tailscale_url": route.tailscale_url,
                "local_target": route.local_target,
                "port": route.port,
                "path": route.path,
            }
            for route in routes
        ],
        "redirector_overrides": {
            route.service: route.tailscale_url
            for route in routes
            if route.service
        },
        "host_url_overrides": {
            route.host_url: route.tailscale_url
            for route in routes
        },
    }


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def print_plan(routes: list[ServiceRoute], *, tailscale_host: str) -> None:
    print(f"Tailscale host: {tailscale_host}")
    print(f"HTTPS routes: {len(routes)}")
    for route in routes:
        print(f"- {route.name}: {route.host_url} -> {route.tailscale_url}")
        print("  " + " ".join(serve_command(route)))


def apply_routes(routes: list[ServiceRoute], *, sudo_password: str | None) -> None:
    for route in routes:
        cmd = serve_command(route)
        print("+ " + " ".join(cmd))
        run(cmd, capture=False, sudo_password=sudo_password)


def disable_routes(routes: list[ServiceRoute], *, sudo_password: str | None) -> None:
    for route in routes:
        cmd = disable_command(route)
        print("+ " + " ".join(cmd))
        run(cmd, capture=False, sudo_password=sudo_password)


def maybe_prompt_sudo_password(enabled: bool) -> str | None:
    if not enabled:
        return None
    return getpass.getpass("sudo password for tailscale commands: ")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Tailscale Serve HTTPS routes for AlphaRavis local services. "
            "This uses Tailscale Serve inside your tailnet only; it never runs Tailscale Funnel."
        )
    )
    parser.add_argument(
        "command",
        choices=["plan", "apply", "disable", "status", "write-overrides"],
        help="plan prints commands, apply runs tailscale serve, disable turns these ports off, status prints serve status.",
    )
    parser.add_argument(
        "--tailscale-host",
        default=os.getenv("ALPHARAVIS_TAILSCALE_HOST", ""),
        help="MagicDNS name, for example my-node.tailnet.ts.net. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--include-dashboard",
        action="store_true",
        help=(
            "Also add a Tailscale Serve route for the service-dashboard itself. "
            "Disabled by default to avoid a self-referential route."
        ),
    )
    parser.add_argument(
        "--output",
        default=os.getenv("ALPHARAVIS_TAILSCALE_URLS_FILE", str(DEFAULT_OUTPUT_PATH)),
        help="JSON file for redirector URL overrides.",
    )
    parser.add_argument(
        "--sudo",
        action="store_true",
        help="Run tailscale commands through sudo -S and prompt for the password interactively.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write the redirector override JSON after apply/write-overrides.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    sudo_password = maybe_prompt_sudo_password(args.sudo)
    tailscale_host = args.tailscale_host.rstrip(".") or discover_tailscale_dns_name(sudo_password=sudo_password)
    routes = build_routes(
        load_redirector_services(),
        tailscale_host=tailscale_host,
        include_self=args.include_dashboard,
    )
    payload = route_payload(routes, tailscale_host=tailscale_host)
    output_path = Path(args.output)

    if args.command == "plan":
        print_plan(routes, tailscale_host=tailscale_host)
    elif args.command == "apply":
        print_plan(routes, tailscale_host=tailscale_host)
        apply_routes(routes, sudo_password=sudo_password)
        if not args.no_write:
            write_payload(output_path, payload)
    elif args.command == "disable":
        disable_routes(routes, sudo_password=sudo_password)
    elif args.command == "status":
        result = run(["tailscale", "serve", "status"], sudo_password=sudo_password)
        print(result.stdout.rstrip())
    elif args.command == "write-overrides":
        write_payload(output_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
