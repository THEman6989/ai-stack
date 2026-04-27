from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable
from urllib import request


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
EXAMPLE_PATH = ROOT / ".env(exaple)"


IMPORTANT_KEYS = [
    ("POSTGRES_PASSWORD", "Postgres password for vectordb/rag_api"),
    ("LITELLM_MASTER_KEY", "LiteLLM master key"),
    ("BIG_BOSS_API_BASE", "llama.cpp big model OpenAI /v1 URL"),
    ("EDGE_GEMMA_API_BASE", "Ollama fallback OpenAI /v1 URL"),
    ("EMBEDDING_API_BASE", "Ollama embedding OpenAI /v1 URL"),
    ("HERMES_API_BASE", "internal Docker URL for LibreChat/LangGraph to Hermes"),
    ("HERMES_EXTERNAL_API_BASE", "host URL for humans/tools to call Hermes"),
    ("HERMES_API_KEY", "Hermes API bearer token"),
    ("HERMES_MODEL", "Hermes advertised OpenAI model id"),
    ("HERMES_INFERENCE_MODEL", "real LiteLLM model Hermes should use"),
    ("HERMES_OPENAI_BASE_URL", "LiteLLM/OpenAI-compatible URL used by Hermes"),
    ("HERMES_OPENAI_API_KEY", "API key Hermes uses for LiteLLM"),
    ("ALPHARAVIS_ENABLE_HERMES_AGENT", "enable LangGraph -> Hermes sub-agent"),
    ("LIBRECHAT_OPENAI_API_KEY", "optional generic LibreChat OpenAI bucket key"),
    ("LIBRECHAT_OPENAI_REVERSE_PROXY", "optional generic LibreChat OpenAI reverse proxy"),
]


SERVICE_URLS = [
    ("LibreChat", "http://localhost:3080"),
    ("LangGraph API", "http://localhost:2024"),
    ("LangGraph Studio", "https://smith.langchain.com/studio/?baseUrl=http://localhost:2024"),
    ("OpenAI Bridge", "http://localhost:8123/v1"),
    ("Hermes API", "HERMES_EXTERNAL_API_BASE"),
    ("LiteLLM", "http://localhost:4000/v1"),
    ("RAG API", "http://localhost:8000"),
    ("DeepAgents UI", "http://localhost:3000"),
    ("Agent Custom UI", "http://localhost:3001"),
    ("Pixelle MCP", "http://localhost:9004"),
]


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, text=True, check=check)


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def ensure_env() -> None:
    if not ENV_PATH.exists():
        shutil.copyfile(EXAMPLE_PATH, ENV_PATH)
        print(f"created {ENV_PATH.name} from {EXAMPLE_PATH.name}")

    current = read_env(ENV_PATH)
    example = read_env(EXAMPLE_PATH)
    missing = [(key, value) for key, value in example.items() if key not in current]
    if not missing:
        return

    with ENV_PATH.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n\n# Added by make install/update from .env(exaple)\n")
        for key, value in missing:
            fh.write(f"{key}={value}\n")
    print(f"added {len(missing)} missing env defaults to .env")


def update_env_value(key: str, value: str) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    found = False
    out: list[str] = []
    for line in lines:
        if line.strip().startswith("#") or "=" not in line:
            out.append(line)
            continue
        current_key = line.split("=", 1)[0].strip()
        if current_key == key:
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def configure() -> None:
    ensure_env()
    values = read_env(ENV_PATH)
    print("Press Enter to keep the current/default value.")
    for key, description in IMPORTANT_KEYS:
        current = values.get(key, "")
        answer = input(f"{key} [{current}] - {description}: ").strip()
        if answer:
            update_env_value(key, answer)
            values[key] = answer
    print(".env updated")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    answer = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes", "j", "ja", "true", "1"}


def install() -> None:
    ensure_env()
    if ask_yes_no("Edit important .env values now", default=False):
        configure()
    if ask_yes_no("Initialize/update submodules now", default=True):
        run(["git", "submodule", "update", "--init", "--recursive"])
    print_status()
    print("Next: make up")


def update() -> None:
    ensure_env()
    run(["git", "pull", "--ff-only"])
    if ask_yes_no("Update submodules to their configured remote branches", default=True):
        run(["git", "submodule", "update", "--init", "--recursive", "--remote"])
    if ask_yes_no("Edit important .env values after update", default=False):
        configure()
    print_status()


def docker_ps() -> None:
    try:
        result = subprocess.run(
            ["docker", "compose", "ps"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        print("docker compose not found on PATH")
        return
    if result.returncode == 0:
        print(result.stdout.rstrip())
    else:
        detail = (result.stderr or result.stdout).strip()
        print(f"docker compose is not reachable right now: {detail}")


def resolve_url(value: str, env: dict[str, str]) -> str:
    if value.isupper():
        return env.get(value, "")
    return value


def print_status() -> None:
    env = read_env(ENV_PATH if ENV_PATH.exists() else EXAMPLE_PATH)
    print("\nAlphaRavis service URLs")
    for label, value in SERVICE_URLS:
        print(f"- {label}: {resolve_url(value, env)}")
    print("\nLibreChat model picker")
    print("- LangGraph Agent: custom endpoint -> api-bridge:8123/v1")
    print("- Hermes Agent: custom endpoint -> hermes-agent:8642/v1")
    print("- OpenAI: only appears if LIBRECHAT_OPENAI_API_KEY/REVERSE_PROXY are set")
    print("\nDocker status")
    docker_ps()


def http_json(url: str, *, api_key: str = "", payload: dict | None = None, timeout: int = 15) -> str:
    data = None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    req = request.Request(url, data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def bridge_smoke() -> None:
    env = read_env(ENV_PATH)
    base = env.get("BRIDGE_EXTERNAL_API_BASE", "http://localhost:8123/v1").rstrip("/")
    body = {
        "model": env.get("OPENAI_MODEL_NAME", "my-agent"),
        "messages": [{"role": "user", "content": "Antworte nur mit OK."}],
        "stream": False,
    }
    print(http_json(f"{base}/chat/completions", api_key="sk-1234", payload=body, timeout=30))


def hermes_smoke() -> None:
    env = read_env(ENV_PATH)
    base = env.get("HERMES_EXTERNAL_API_BASE", "http://localhost:8642/v1").rstrip("/")
    body = {
        "model": env.get("HERMES_MODEL", "hermes-agent"),
        "messages": [{"role": "user", "content": "Antworte nur mit OK."}],
        "stream": False,
    }
    print(http_json(f"{base}/chat/completions", api_key=env.get("HERMES_API_KEY", ""), payload=body, timeout=60))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AlphaRavis setup helper")
    parser.add_argument("command", choices=["install", "configure", "update", "status", "bridge-smoke", "hermes-smoke"])
    args = parser.parse_args(argv)
    if args.command == "install":
        install()
    elif args.command == "configure":
        configure()
    elif args.command == "update":
        update()
    elif args.command == "status":
        print_status()
    elif args.command == "bridge-smoke":
        bridge_smoke()
    elif args.command == "hermes-smoke":
        hermes_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
