from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any


MODEL_FLAGS = {"-hf", "--hf", "--hf-repo", "-m", "--model", "--model-url"}
HF_FLAGS = {"-hf", "--hf", "--hf-repo"}
LOCAL_MODEL_FLAGS = {"-m", "--model"}
CONTEXT_FLAGS = {"-c", "--ctx-size", "--ctx_size", "--context", "--context-size"}
PARALLEL_FLAGS = {"--parallel", "-np", "--parallel-slots"}


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def replace_config_value(config_path: Path, key: str, value: str) -> None:
    lines = config_path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            output.append(f"{key}={shell_quote(value)}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(f"{key}={shell_quote(value)}")
    config_path.write_text("\n".join(output) + "\n", encoding="utf-8")


def switch_model_in_command(command: str, model: str, model_flag: str = "auto") -> str:
    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("LLAMA_COMMAND is empty")

    if model_flag == "auto":
        target_flags = MODEL_FLAGS
    elif model_flag == "hf":
        target_flags = HF_FLAGS
    elif model_flag == "local":
        target_flags = LOCAL_MODEL_FLAGS
    else:
        target_flags = {model_flag}

    for index, token in enumerate(tokens):
        if token in target_flags and index + 1 < len(tokens):
            tokens[index + 1] = model
            return shlex.join(tokens)
        if any(token.startswith(f"{flag}=") for flag in target_flags):
            flag = token.split("=", 1)[0]
            tokens[index] = f"{flag}={model}"
            return shlex.join(tokens)

    default_flag = "-hf" if model_flag in {"auto", "hf"} else "-m"
    tokens.extend([default_flag, model])
    return shlex.join(tokens)


def replace_flag_value(command: str, flags: set[str], value: str) -> str:
    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("command is empty")

    for index, token in enumerate(tokens):
        if token in flags and index + 1 < len(tokens):
            tokens[index + 1] = value
            return shlex.join(tokens)
        for flag in flags:
            if token.startswith(f"{flag}="):
                tokens[index] = f"{flag}={value}"
                return shlex.join(tokens)

    default_flag = "-c" if "-c" in flags else sorted(flags)[0]
    tokens.extend([default_flag, value])
    return shlex.join(tokens)


def switch_context_in_command(command: str, context_size: int | str) -> str:
    value = str(context_size).strip()
    if not value.isdigit() or int(value) < 1:
        raise ValueError("context_size must be a positive integer")
    return replace_flag_value(command, CONTEXT_FLAGS, value)


def switch_parallel_in_command(command: str, parallel: int | str) -> str:
    value = str(parallel).strip()
    if not value.isdigit() or int(value) < 1:
        raise ValueError("parallel must be a positive integer")
    return replace_flag_value(command, PARALLEL_FLAGS, value)


def update_llama_command(config_path: Path, key: str, current_command: str, new_command: str) -> dict[str, Any]:
    if not new_command.strip():
        raise ValueError("command is empty")
    replace_config_value(config_path, key, new_command)
    return {"key": key, "old_command": current_command, "new_command": new_command}


def patch_llama_command(
    config_path: Path,
    key: str,
    current_command: str,
    *,
    model: str = "",
    model_flag: str = "auto",
    context_size: int | str | None = None,
    parallel: int | str | None = None,
) -> dict[str, Any]:
    new_command = current_command
    changed: dict[str, Any] = {}
    if model:
        new_command = switch_model_in_command(new_command, model, model_flag)
        changed["model"] = model
        changed["model_flag"] = model_flag
    if context_size not in (None, ""):
        new_command = switch_context_in_command(new_command, context_size)
        changed["context_size"] = int(str(context_size))
    if parallel not in (None, ""):
        new_command = switch_parallel_in_command(new_command, parallel)
        changed["parallel"] = int(str(parallel))
    if not changed:
        raise ValueError("nothing to update; send command, model, context_size, or parallel")
    replace_config_value(config_path, key, new_command)
    return {"key": key, "old_command": current_command, "new_command": new_command, "changed": changed}


def update_llama_model(config_path: Path, current_command: str, model: str, model_flag: str = "auto") -> dict[str, Any]:
    new_command = switch_model_in_command(current_command, model, model_flag)
    replace_config_value(config_path, "LLAMA_COMMAND", new_command)
    return {"old_command": current_command, "new_command": new_command, "model": model, "model_flag": model_flag}
