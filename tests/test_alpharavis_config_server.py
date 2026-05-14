from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "scripts" / "alpharavis_config_server.py"
CONFIG_SPEC = importlib.util.spec_from_file_location("alpharavis_config_server", CONFIG_PATH)
alpharavis_config_server = importlib.util.module_from_spec(CONFIG_SPEC)
assert CONFIG_SPEC and CONFIG_SPEC.loader
CONFIG_SPEC.loader.exec_module(alpharavis_config_server)


def test_parse_env_template_groups_keys_by_documented_sections(tmp_path: Path) -> None:
    example = tmp_path / ".env(exaple)"
    example.write_text(
        "\n".join(
            [
                "# =====================================================================",
                "# MUST-HAVE - FILL THESE FIRST",
                "# =====================================================================",
                "",
                "# Allowed values: true, false",
                "ALLOW_REGISTRATION=true",
                "",
                "# =====================================================================",
                "# CORE DOCKER URLS - USUALLY DO NOT CHANGE",
                "# =====================================================================",
                "",
                "# Internal URL.",
                "LANGGRAPH_API_URL=http://langgraph-api:2024",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    sections = alpharavis_config_server.parse_env_template(example)

    assert [section["title"] for section in sections] == [
        "MUST-HAVE - FILL THESE FIRST",
        "CORE DOCKER URLS - USUALLY DO NOT CHANGE",
    ]
    assert sections[0]["entries"][0]["key"] == "ALLOW_REGISTRATION"
    assert sections[1]["entries"][0]["key"] == "LANGGRAPH_API_URL"


def test_build_config_model_marks_bool_url_secret_and_changed(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env(exaple)"
    example_path.write_text(
        "\n".join(
            [
                "# =====================================================================",
                "# MUST-HAVE",
                "# =====================================================================",
                "# Allowed values: true, false",
                "ALLOW_REGISTRATION=true",
                "# Secret value",
                "OPENAI_API_KEY=sk-local-dev",
                "# Service URL",
                "OPENAI_API_BASE=http://litellm:4000/v1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_path.write_text(
        "ALLOW_REGISTRATION=false\nOPENAI_API_KEY=sk-real\nOPENAI_API_BASE=http://example/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(alpharavis_config_server, "ENV_PATH", env_path)
    monkeypatch.setattr(alpharavis_config_server, "EXAMPLE_PATH", example_path)

    model = alpharavis_config_server.build_config_model()
    entries = {
        entry["key"]: entry
        for section in model["sections"]
        for entry in section["entries"]
    }

    assert entries["ALLOW_REGISTRATION"]["kind"] == "bool"
    assert entries["ALLOW_REGISTRATION"]["changed"] is True
    assert entries["OPENAI_API_KEY"]["secret"] is True
    assert entries["OPENAI_API_BASE"]["kind"] == "url"


def test_apply_config_updates_only_writes_template_keys(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env(exaple)"
    example_path.write_text(
        "\n".join(
            [
                "# =====================================================================",
                "# MUST-HAVE",
                "# =====================================================================",
                "ALPHARAVIS_MODEL=openai/big-boss",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_path.write_text("ALPHARAVIS_MODEL=openai/big-boss\n", encoding="utf-8")
    monkeypatch.setattr(alpharavis_config_server, "ENV_PATH", env_path)
    monkeypatch.setattr(alpharavis_config_server, "EXAMPLE_PATH", example_path)

    updated = alpharavis_config_server.apply_config_updates(
        {
            "ALPHARAVIS_MODEL": "openai/new-boss",
            "UNKNOWN_KEY": "ignored",
        }
    )

    values = alpharavis_config_server.read_env(env_path)
    assert updated == 1
    assert values["ALPHARAVIS_MODEL"] == "openai/new-boss"
    assert "UNKNOWN_KEY" not in values
