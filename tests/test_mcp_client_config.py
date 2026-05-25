from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import mcp_client  # noqa: E402


def test_mcp_server_enabled_env_defaults_to_disabled(tmp_path, monkeypatch):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        """
        {
          "mcpServers": {
            "officecli": {
              "type": "stdio",
              "command": "officecli",
              "args": ["mcp", "start"],
              "enabled_env": "ALPHARAVIS_ENABLE_OFFICECLI_MCP"
            }
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_client, "_mcp_config_candidate_paths", lambda: [config_path])
    monkeypatch.setenv("ALPHARAVIS_MCP_ALLOW_STDIO", "true")
    monkeypatch.delenv("ALPHARAVIS_ENABLE_OFFICECLI_MCP", raising=False)

    config, _, warnings = mcp_client.load_mcp_config()

    assert config["mcpServers"] == {}
    assert warnings == []


def test_mcp_server_enabled_env_allows_server_when_true(tmp_path, monkeypatch):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        """
        {
          "mcpServers": {
            "officecli": {
              "type": "stdio",
              "command": "officecli",
              "args": ["mcp", "start"],
              "enabled_env": "ALPHARAVIS_ENABLE_OFFICECLI_MCP"
            }
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_client, "_mcp_config_candidate_paths", lambda: [config_path])
    monkeypatch.setenv("ALPHARAVIS_MCP_ALLOW_STDIO", "true")
    monkeypatch.setenv("ALPHARAVIS_ENABLE_OFFICECLI_MCP", "true")

    config, _, warnings = mcp_client.load_mcp_config()

    assert "officecli" in config["mcpServers"]
    assert config["mcpServers"]["officecli"]["command"] == "officecli"
    assert warnings == []
