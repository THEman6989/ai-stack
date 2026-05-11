from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP_PATH = ROOT / "scripts" / "alpharavis_setup.py"
SETUP_SPEC = importlib.util.spec_from_file_location("alpharavis_setup", SETUP_PATH)
alpharavis_setup = importlib.util.module_from_spec(SETUP_SPEC)
assert SETUP_SPEC and SETUP_SPEC.loader
SETUP_SPEC.loader.exec_module(alpharavis_setup)


def test_full_streaming_mode_sets_required_env_values() -> None:
    values = alpharavis_setup.STREAMING_MODE_VALUES["responses-full"]

    assert values["ALPHARAVIS_LLM_API_MODE"] == "responses"
    assert values["ALPHARAVIS_DEEPAGENTS_API_MODE"] == "responses"
    assert values["ALPHARAVIS_LLM_STREAMING"] == "true"
    assert values["ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING"] == "true"
    assert values["ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING"] == "false"
    assert values["ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING"] == "true"
    assert values["BRIDGE_PREFERRED_API_MODE"] == "responses"


def test_chat_full_streaming_mode_sets_chat_completions_values() -> None:
    values = alpharavis_setup.STREAMING_MODE_VALUES["chat-full"]

    assert values["ALPHARAVIS_LLM_API_MODE"] == "chat_completions"
    assert values["ALPHARAVIS_DEEPAGENTS_API_MODE"] == "chat_completions"
    assert values["ALPHARAVIS_LLM_STREAMING"] == "true"
    assert values["BRIDGE_PREFERRED_API_MODE"] == "chat_completions"


def test_apply_streaming_mode_updates_existing_env(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env(exaple)"
    example_path.write_text("ALPHARAVIS_LLM_API_MODE=responses\n", encoding="utf-8")
    env_path.write_text(
        "\n".join(
            [
                "ALPHARAVIS_LLM_API_MODE=responses",
                "ALPHARAVIS_LLM_STREAMING=true",
                "ALPHARAVIS_DEEPAGENTS_API_MODE=responses",
                "ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true",
                "ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling",
                "ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=false",
                "BRIDGE_PREFERRED_API_MODE=responses",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(alpharavis_setup, "ENV_PATH", env_path)
    monkeypatch.setattr(alpharavis_setup, "EXAMPLE_PATH", example_path)

    mode = alpharavis_setup.apply_streaming_mode("fullstreaming")
    values = alpharavis_setup.read_env(env_path)

    assert mode == "responses-full"
    assert values["ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING"] == "false"
    assert values["ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING"] == "true"


def test_nonstreaming_and_chat_modes_are_detected() -> None:
    assert (
        alpharavis_setup.current_streaming_mode(
            {
                "ALPHARAVIS_LLM_API_MODE": "responses",
                "ALPHARAVIS_DEEPAGENTS_API_MODE": "responses",
                "ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING": "false",
                "ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING": "true",
            }
        )
        == "responses-nonstreaming"
    )
    assert (
        alpharavis_setup.current_streaming_mode(
            {
                "ALPHARAVIS_LLM_API_MODE": "chat_completions",
                "ALPHARAVIS_DEEPAGENTS_API_MODE": "chat_completions",
                "ALPHARAVIS_LLM_STREAMING": "true",
            }
        )
        == "chat-full"
    )
    assert (
        alpharavis_setup.current_streaming_mode(
            {
                "ALPHARAVIS_LLM_API_MODE": "chat_completions",
                "ALPHARAVIS_DEEPAGENTS_API_MODE": "chat_completions",
                "ALPHARAVIS_LLM_STREAMING": "false",
            }
        )
        == "chat-nonstreaming"
    )


def test_compose_profiles_are_normalized() -> None:
    assert alpharavis_setup.normalize_profiles(" openwebui ; hermes-dashboard ") == "openwebui,hermes-dashboard"
    assert alpharavis_setup.normalize_profiles("none") == ""


def test_streaming_false_alias_means_nonstreaming() -> None:
    assert alpharavis_setup.normalize_streaming_mode("false") == "responses-nonstreaming"


def test_chat_aliases_select_chat_full_streaming() -> None:
    assert alpharavis_setup.normalize_streaming_mode("chat") == "chat-full"
    assert alpharavis_setup.normalize_streaming_mode("chat-completions-full") == "chat-full"


def test_all_runtime_profiles_expose_operator_env_reference() -> None:
    required = {
        "ALPHARAVIS_LLM_API_MODE",
        "ALPHARAVIS_LLM_STREAMING",
        "ALPHARAVIS_DEEPAGENTS_API_MODE",
        "BRIDGE_PREFERRED_API_MODE",
    }
    for values in alpharavis_setup.STREAMING_MODE_VALUES.values():
        assert required.issubset(values)
