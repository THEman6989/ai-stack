from __future__ import annotations

import re
import subprocess
from typing import Any

from .config import Settings


DEFAULT_CRITICAL_PATTERNS = (
    r"GPU reset",
    r"amdgpu.*ring.*timeout",
    r"amdgpu.*GPU fault",
    r"RAS.*uncorrectable",
    r"ENABLED\s+[0-9]+\s+[1-9][0-9]*",
    r"PCIe.*AER",
    r"AER:.*error",
    r"pcieport.*error",
    r"GPU has fallen off",
    r"not responding",
)


def critical_patterns(settings: Settings) -> list[str]:
    raw = settings.get("GPU_HEALTH_CRITICAL_PATTERNS", "")
    if not raw:
        return list(DEFAULT_CRITICAL_PATTERNS)
    return [item for item in raw.split("|") if item.strip()]


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


def run_rocm(args: list[str]) -> dict[str, Any]:
    return run_command(["rocm-smi", *args], timeout=20)


def run_dmesg(settings: Settings) -> dict[str, Any]:
    since_seconds = settings.int("GPU_HEALTH_DMESG_SINCE_SECONDS", 300)
    return run_command(["journalctl", "-k", "--since", f"-{since_seconds} seconds", "--no-pager"], timeout=20)


def find_matches(text: str, patterns: list[str]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for line in text.splitlines():
        if line.startswith("===") or "Not supported on the given system" in line:
            continue
        for pattern in patterns:
            if re.search(pattern, line, flags=re.IGNORECASE):
                matches.append({"pattern": pattern, "line": line})
                break
    return matches


def parse_pcie_replay_counts(text: str, threshold: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    counts: list[dict[str, Any]] = []
    matches: list[dict[str, str]] = []
    for line in text.splitlines():
        if "Not supported on the given system" in line:
            continue
        gpu_match = re.search(r"GPU\[(\d+)\]", line)
        if not gpu_match or "Replay" not in line:
            continue
        value_match = re.search(r"([0-9]+)(?:\s*$|\s*[A-Za-z])", line)
        if not value_match:
            continue
        value = int(value_match.group(1))
        gpu = int(gpu_match.group(1))
        counts.append({"gpu": gpu, "replay_count": value})
        if value > threshold:
            matches.append(
                {
                    "pattern": "pcie_replay_count",
                    "line": line,
                    "gpu": str(gpu),
                    "value": str(value),
                    "threshold": str(threshold),
                }
            )
    return counts, matches


def diagnose_gpu(settings: Settings) -> dict[str, Any]:
    checks = {
        "rocm_smi_id": run_rocm(["--showid"]),
        "rocm_smi_power": run_rocm(["--showpower", "--showmaxpower"]),
        "rocm_smi_mem": run_rocm(["--showmemuse"]),
        "rocm_smi_replay": run_rocm(["--showreplaycount"]),
        "rocm_smi_ras": run_rocm(["--showrasinfo"]),
        "rocm_smi_xgmi": run_rocm(["--showxgmierr"]),
        "kernel": run_dmesg(settings),
    }
    patterns = critical_patterns(settings)
    combined = "\n".join(
        f"## {name}\n{result.get('stdout', '')}\n{result.get('stderr', '')}"
        for name, result in checks.items()
    )
    matches = find_matches(combined, patterns)
    replay_counts, replay_matches = parse_pcie_replay_counts(
        f"{checks['rocm_smi_replay'].get('stdout', '')}\n{checks['rocm_smi_replay'].get('stderr', '')}",
        settings.int("GPU_HEALTH_PCIE_REPLAY_THRESHOLD", 0),
    )
    matches.extend(replay_matches)
    command_failures = [name for name, result in checks.items() if not result.get("ok")]
    decision = "gpu-critical" if matches else "healthy"
    if command_failures and not matches:
        decision = "diagnostic-command-failure"
    return {
        "ok": len(matches) == 0 and len(command_failures) == 0,
        "critical": len(matches) > 0,
        "decision": decision,
        "pcie_replay_counts": replay_counts,
        "matches": matches,
        "command_failures": command_failures,
        "checks": checks,
    }


def vram_summary(settings: Settings) -> dict[str, Any]:
    result = run_rocm(["--showmemuse"])
    text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    used_values = []
    for line in text.splitlines():
        match = re.search(r"GPU\[(\d+)\].*?([0-9.]+)%", line)
        if match:
            used_values.append({"gpu": int(match.group(1)), "used_percent": float(match.group(2))})
    return {"ok": result["ok"], "raw": text.strip(), "gpus": used_values}
