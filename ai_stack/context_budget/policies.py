from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any


CTX_FLAGS = {"-c", "--ctx-size", "--ctx_size", "--context", "--context-size"}
PARALLEL_FLAGS = {"--parallel", "-np", "--parallel-slots"}
KV_UNIFIED_FLAGS = {"--kv-unified", "--kv_unified"}
KV_UNIFIED_FALSE_FLAGS = {"--no-kv-unified", "--no-kv_unified"}


@dataclass(frozen=True)
class RuntimeConfig:
    ctx_total: int
    parallel: int
    kv_unified: bool
    command: str = ""

    @property
    def conservative_ctx_per_slot(self) -> int:
        return max(1, self.ctx_total // max(1, self.parallel))


def _int_or_default(value: Any, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def parse_runtime_config_from_command(
    command: str | None,
    *,
    ctx_total: int | str | None = None,
    parallel: int | str | None = None,
    kv_unified: bool | None = None,
) -> RuntimeConfig:
    """Parse llama-server context flags from a saved start command."""

    tokens: list[str]
    try:
        tokens = shlex.split(command or "")
    except ValueError:
        tokens = str(command or "").split()

    parsed_ctx = _int_or_default(ctx_total, 0)
    parsed_parallel = _int_or_default(parallel, 1)
    parsed_kv_unified = bool(kv_unified) if kv_unified is not None else False

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in CTX_FLAGS and index + 1 < len(tokens):
            parsed_ctx = _int_or_default(tokens[index + 1], parsed_ctx)
            index += 2
            continue
        if token in PARALLEL_FLAGS and index + 1 < len(tokens):
            parsed_parallel = _int_or_default(tokens[index + 1], parsed_parallel)
            index += 2
            continue
        if token in KV_UNIFIED_FLAGS:
            parsed_kv_unified = True
        elif token in KV_UNIFIED_FALSE_FLAGS:
            parsed_kv_unified = False
        elif token.startswith("--ctx-size=") or token.startswith("--ctx_size=") or token.startswith("--context-size="):
            parsed_ctx = _int_or_default(token.split("=", 1)[1], parsed_ctx)
        elif token.startswith("--parallel=") or token.startswith("--parallel-slots="):
            parsed_parallel = _int_or_default(token.split("=", 1)[1], parsed_parallel)
        index += 1

    return RuntimeConfig(
        ctx_total=parsed_ctx if parsed_ctx > 0 else 8192,
        parallel=max(1, parsed_parallel),
        kv_unified=parsed_kv_unified,
        command=command or "",
    )


def ensure_kv_unified_in_command(command: str) -> str:
    if not command.strip():
        return command
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command if "--kv-unified" in command else f"{command.rstrip()} --kv-unified"
    if any(token in KV_UNIFIED_FLAGS for token in tokens):
        return command
    tokens = [token for token in tokens if token not in KV_UNIFIED_FALSE_FLAGS]
    tokens.append("--kv-unified")
    return " ".join(shlex.quote(token) for token in tokens)


def capacity_limit(ctx_total: int, safety_factor: float) -> int:
    factor = min(1.0, max(0.1, float(safety_factor)))
    return max(1, int(ctx_total * factor))

