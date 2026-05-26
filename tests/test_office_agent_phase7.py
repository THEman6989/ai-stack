from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_GRAPH = ROOT / "langgraph-app" / "agent_graph.py"
UI_ROOT = ROOT / "submodules" / "deep-agents-ui"


def _agent_graph_source() -> str:
    return AGENT_GRAPH.read_text(encoding="utf-8")


def _office_panel_source() -> str:
    return (UI_ROOT / "src" / "app" / "components" / "OfficePanel.tsx").read_text(encoding="utf-8")


def _chat_hook_source() -> str:
    return (UI_ROOT / "src" / "app" / "hooks" / "useChat.ts").read_text(encoding="utf-8")


def test_phase7_graph_gates_office_agent_with_feature_flag():
    content = _agent_graph_source()

    assert 'ALPHARAVIS_ENABLE_OFFICE_AGENT' in content
    assert '_office_agent_enabled()' in content
    assert 'office_agent_enabled =' in content
    assert 'office_agent' in content
    assert 'agent/office' in content
    assert 'agent_name="office_agent"' in content
    assert 'swarm_workers.append(office_worker)' in content


def test_phase7_graph_peer_agents_can_handoff_to_office_agent():
    content = _agent_graph_source()

    assert 'transfer_to_office = create_handoff_tool' in content
    assert 'Office document workflows' in content
    assert 'transfer_to_office' in content
    assert 'office_agent' in content
    assert 'inspect before modifying' in content.lower()
    assert 'copy-first' in content.lower()
    assert 'run_state_manager' in content


def test_phase7_office_tab_submits_directly_to_office_agent_state():
    panel = _office_panel_source()
    chat_hook = _chat_hook_source()

    assert 'NEXT_PUBLIC_OFFICE_AGENT_ENABLED' in panel
    assert 'NEXT_PUBLIC_OFFICE_AGENT_NAME' in panel
    assert 'OFFICE_AGENT_NAME' in panel
    assert 'sendOfficeAgentMessage' in panel
    assert 'activeAgent: OFFICE_AGENT_NAME' in panel
    assert 'Office-Agent' in panel
    assert 'activeAgent?: string' in chat_hook
    assert 'active_agent: options?.activeAgent' in chat_hook
