from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        try:
            parsed = shlex.split(value, comments=False, posix=True)
            values[key] = parsed[0] if parsed else ""
        except ValueError:
            values[key] = value.strip("\"'")
    return values


def first_non_placeholder(*values: str) -> str:
    placeholders = {"", "change-me", "change-me-esp-token"}
    for value in values:
        if value not in placeholders:
            return value
    return values[0] if values else ""


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}


def as_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    config_path: Path
    project_root: Path
    raw: dict[str, str]

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Settings":
        path = Path(
            config_path
            or os.environ.get("UBUNTU_CONFIG")
            or os.environ.get("RAKAM_CONFIG")
            or PROJECT_ROOT / "ubuntu-llama.conf"
        ).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()

        values = parse_env_file(path)

        esp_env_raw = values.get("ESP_ENV_FILE", "")
        esp_env_path = Path(esp_env_raw).expanduser() if esp_env_raw else PROJECT_ROOT / "firmware" / "nodemcu-v3" / ".env"
        if not esp_env_path.is_absolute():
            esp_env_path = (PROJECT_ROOT / esp_env_path).resolve()
        esp_values = parse_env_file(esp_env_path)
        values.update(esp_values)

        if "MANAGER_API_TOKEN" in values:
            values["API_TOKEN"] = first_non_placeholder(values.get("API_TOKEN", ""), values["MANAGER_API_TOKEN"])
        if "ESP_AUTH_TOKEN" in values:
            values["ESP_WEBHOOK_TOKEN"] = first_non_placeholder(values.get("ESP_WEBHOOK_TOKEN", ""), values["ESP_AUTH_TOKEN"])

        for key, value in os.environ.items():
            if key.startswith(("UBUNTU_", "RAKAM_", "LLAMA_", "API_", "REBOOT_", "ENABLE_", "HF_", "MODEL_", "ESP_", "POWER_", "GPU_", "RUN_AS_", "STOP_")):
                values[key] = value

        return cls(config_path=path, project_root=PROJECT_ROOT, raw=values)

    def get(self, key: str, default: str = "") -> str:
        return self.raw.get(key, default)

    def bool(self, key: str, default: bool = False) -> bool:
        return as_bool(self.raw.get(key), default)

    def int(self, key: str, default: int) -> int:
        return as_int(self.raw.get(key), default)

    def float(self, key: str, default: float) -> float:
        return as_float(self.raw.get(key), default)

    @property
    def api_host(self) -> str:
        return self.get("API_HOST", "0.0.0.0")

    @property
    def api_port(self) -> int:
        return self.int("API_PORT", 8099)

    @property
    def api_token(self) -> str:
        return self.get("API_TOKEN", "change-me")

    @property
    def llama_port(self) -> int:
        return self.int("LLAMA_PORT", 8033)

    @property
    def llama_host(self) -> str:
        return self.get("LLAMA_HOST", "127.0.0.1")

    @property
    def llama_log_file(self) -> Path:
        return Path(self.get("LLAMA_LOG_FILE", str(self.project_root / "logs" / "llama.log"))).expanduser()

    @property
    def state_dir(self) -> Path:
        return Path(self.get("UBUNTU_STATE_DIR", self.get("RAKAM_STATE_DIR", str(self.project_root / "state")))).expanduser()

    @property
    def reboot_interval_seconds(self) -> int:
        explicit = self.get("REBOOT_INTERVAL_SECONDS", "")
        if explicit:
            return max(60, self.int("REBOOT_INTERVAL_SECONDS", 10800))

        hours = self.float("REBOOT_INTERVAL_HOURS", 3.0)
        if self.get("REBOOT_INTERVAL_HOURS", ""):
            return max(60, int(hours * 3600))

        legacy = self.get("REBOOT_AFTER_SECONDS", "")
        if legacy:
            return max(60, self.int("REBOOT_AFTER_SECONDS", 10800))

        return 10800

    @property
    def reboot_interval_hours(self) -> float:
        return round(self.reboot_interval_seconds / 3600, 4)

    @property
    def model_scan_dirs(self) -> list[Path]:
        raw = self.get("MODEL_SCAN_DIRS", "")
        if not raw:
            raw = self.get("HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface"))
        return [Path(item).expanduser() for item in raw.split(":") if item.strip()]
