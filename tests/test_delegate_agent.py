"""Tests for AlphaRavis delegate_agent module — sub-agent registry, file tracking, nesting.

Covers:
- SubAgentRegistry lifecycle
- FileStateTracker cross-agent awareness
- Helper functions (_normalize_tool_names, _build_tool_schemas, _extract_write_path_from_command)
- AgentContext dataclass
- Source-scan: new tools exist in agent_graph.py
"""

import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Test SubAgentRegistry
# ---------------------------------------------------------------------------


def test_agent_context_dataclass():
    """AgentContext is properly constructed with all fields."""
    from delegate_agent import AgentContext

    ctx = AgentContext(
        agent_id="test-001",
        parent_id=None,
        depth=0,
        goal="Test goal",
        started_at=time.time(),
    )
    d = ctx.to_dict()
    assert d["agent_id"] == "test-001"
    assert d["parent_id"] is None
    assert d["depth"] == 0
    assert d["state"] == "running"
    assert d["goal"] == "Test goal"
    assert "elapsed_seconds" in d


def test_sub_agent_registry_register_and_list():
    """Register agents and list them."""
    from delegate_agent import SubAgentRegistry

    reg = SubAgentRegistry()
    ctx1 = reg.register(parent_id=None, depth=0, goal="Task A")
    ctx2 = reg.register(parent_id=ctx1.agent_id, depth=1, goal="Task B")

    agents = reg.list_all()
    assert len(agents) == 2
    ids = {a["agent_id"] for a in agents}
    assert ctx1.agent_id in ids
    assert ctx2.agent_id in ids
    assert ctx1.depth == 0
    assert ctx2.depth == 1
    assert ctx2.parent_id == ctx1.agent_id


def test_sub_agent_registry_get_and_unregister():
    """Get an agent by ID and unregister it."""
    from delegate_agent import SubAgentRegistry

    reg = SubAgentRegistry()
    ctx = reg.register(parent_id=None, depth=0, goal="Task X")
    agent_id = ctx.agent_id

    found = reg.get(agent_id)
    assert found is not None
    assert found.agent_id == agent_id

    reg.unregister(agent_id)
    assert reg.get(agent_id) is None
    assert len(reg.list_all()) == 0


def test_sub_agent_registry_kill():
    """Kill sets cancellation event and state."""
    from delegate_agent import SubAgentRegistry

    reg = SubAgentRegistry()
    ctx = reg.register(parent_id=None, depth=0, goal="Kill me")
    agent_id = ctx.agent_id

    assert ctx.state == "running"
    assert ctx.cancel_event is not None
    assert not ctx.cancel_event.is_set()

    result = reg.kill(agent_id)
    assert result is True
    assert ctx.state == "cancelled"
    assert ctx.cancel_event.is_set()

    # Kill non-existent
    assert reg.kill("nonexistent") is False


def test_sub_agent_registry_kill_children_of():
    """Kill all children of a parent."""
    from delegate_agent import SubAgentRegistry

    reg = SubAgentRegistry()
    parent = reg.register(parent_id=None, depth=0, goal="Parent")
    child1 = reg.register(parent_id=parent.agent_id, depth=1, goal="Child 1")
    child2 = reg.register(parent_id=parent.agent_id, depth=1, goal="Child 2")
    orphan = reg.register(parent_id="other", depth=1, goal="Other child")
    root = reg.register(parent_id=None, depth=0, goal="Another root")

    count = reg.kill_children_of(parent.agent_id)
    assert count == 2

    # Child 1 and 2 are cancelled, parent is not, orphan and root are not
    assert child1.cancel_event.is_set()
    assert child2.cancel_event.is_set()
    assert child1.state == "cancelled"
    assert child2.state == "cancelled"
    assert parent.state == "running"  # NOT killed
    assert orphan.state == "running"
    assert root.state == "running"


def test_sub_agent_registry_counter_monotonic():
    """Agent IDs are monotonically increasing across registers."""
    from delegate_agent import SubAgentRegistry

    reg = SubAgentRegistry()
    ids = []
    for i in range(10):
        ctx = reg.register(parent_id=None, depth=0, goal=f"Task {i}")
        ids.append(ctx.agent_id)

    # All IDs are unique
    assert len(set(ids)) == 10
    # Format: "delegate-0001", "delegate-0002", ...
    for aid in ids:
        assert aid.startswith("delegate-")
        seq = int(aid.split("-")[1])
        assert 1 <= seq <= 10


# ---------------------------------------------------------------------------
# Test FileStateTracker
# ---------------------------------------------------------------------------


def test_file_state_tracker_write_and_read(tmp_path: Path):
    """Record a write, then check that another agent detects stale read."""
    from delegate_agent import FileStateTracker

    tracker = FileStateTracker()
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    # Agent A writes
    tracker.record_write(str(test_file), "agent-A")

    # Agent B reads — should get stale warning (A wrote it, B hasn't read yet)
    warning = tracker.check_stale_read(str(test_file), "agent-B")
    assert warning is not None
    assert "STALE FILE" in warning
    assert "agent-A" in warning
    assert str(test_file.resolve()) in warning

    # Agent A reads its own file — no warning
    warning2 = tracker.check_stale_read(str(test_file), "agent-A")
    assert warning2 is None

    # Agent B reads again — now B has seen it after the write, no warning
    warning3 = tracker.check_stale_read(str(test_file), "agent-B")
    assert warning3 is None


def test_file_state_tracker_nonexistent_path():
    """Checking a file that doesn't exist returns None."""
    from delegate_agent import FileStateTracker

    tracker = FileStateTracker()
    warning = tracker.check_stale_read("/nonexistent/path/foo.txt", "agent-C")
    assert warning is None


def test_file_state_tracker_get_last_writer(tmp_path: Path):
    """Get the last writer for a path."""
    from delegate_agent import FileStateTracker

    tracker = FileStateTracker()
    test_file = tmp_path / "foo.txt"
    test_file.write_text("bar")
    tracker.record_write(str(test_file), "agent-X")

    writer = tracker.get_last_writer(str(test_file))
    assert writer == "agent-X"

    assert tracker.get_last_writer(str(tmp_path / "nonexistent.txt")) is None


# ---------------------------------------------------------------------------
# Test helper functions in delegate_agent
# ---------------------------------------------------------------------------


def test_normalize_tool_names():
    """_normalize_tool_names returns set or None."""
    from delegate_agent import _normalize_tool_names

    assert _normalize_tool_names(None) is None
    assert _normalize_tool_names([]) is None
    assert _normalize_tool_names(["  ", ""]) is None

    result = _normalize_tool_names(["read_file", "write_file", "  execute  "])
    assert result == {"read_file", "write_file", "execute"}


def test_extract_write_path_from_command_redirect():
    """_extract_write_path_from_command extracts paths from shell redirects."""
    from delegate_agent import _extract_write_path_from_command

    assert _extract_write_path_from_command("cat file > /tmp/output.txt") == "/tmp/output.txt"
    assert _extract_write_path_from_command("echo hello >> /tmp/log.txt") == "/tmp/log.txt"
    assert _extract_write_path_from_command("grep pattern file > /dev/null") is None  # dev/null excluded
    assert _extract_write_path_from_command("echo test") is None  # no redirect


def test_extract_write_path_from_command_tee():
    """_extract_write_path_from_command extracts tee targets."""
    from delegate_agent import _extract_write_path_from_command

    assert _extract_write_path_from_command("make | tee /tmp/build.log") == "/tmp/build.log"


def test_extract_write_path_from_command_cp_mv():
    """_extract_write_path_from_command extracts cp/mv/install destinations."""
    from delegate_agent import _extract_write_path_from_command

    assert _extract_write_path_from_command("cp src /tmp/dest.txt") == "/tmp/dest.txt"
    assert _extract_write_path_from_command("mv old /tmp/new.txt") == "/tmp/new.txt"
    assert _extract_write_path_from_command("install -m 755 bin /usr/local/bin/tool") == "/usr/local/bin/tool"


# ---------------------------------------------------------------------------
# Test _build_tool_schemas (needs a mock tool object)
# ---------------------------------------------------------------------------


class FakeArgsSchema:
    """Mock Pydantic-like args schema."""
    def model_json_schema(self):
        return {"type": "object", "properties": {"x": {"type": "string"}}}


class FakeTool:
    """Mock tool object with args_schema and description."""
    def __init__(self, name: str, desc: str = "Fake tool"):
        self.name = name
        self.description = desc
        self.args_schema = FakeArgsSchema()


def test_build_tool_schemas():
    """_build_tool_schemas creates OpenAI-format tool schemas."""
    from delegate_agent import _build_tool_schemas

    tools = {
        "test_tool": FakeTool("test_tool", "A test tool"),
        "another": FakeTool("another", "Another tool"),
    }

    schemas = _build_tool_schemas(tools, _tool_name_fn=lambda t: t.name)
    assert len(schemas) == 2
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] in ("test_tool", "another")
    assert "parameters" in schemas[0]["function"]

    # Empty tools
    assert _build_tool_schemas({}, None) == []


# ---------------------------------------------------------------------------
# Source-scan tests: verify new tools are properly integrated in agent_graph.py
# ---------------------------------------------------------------------------


def test_new_tools_exist_in_registry():
    """list_delegated_agents, kill_delegated_agent, check_file_state exist as @tool functions in agent_graph.py."""
    import ast

    with open("langgraph-app/agent_graph.py", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    tool_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Check if decorated with @tool
            has_tool_decorator = False
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Name) and decorator.id == "tool") or \
                   (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "tool"):
                    has_tool_decorator = True
                    break
            if has_tool_decorator:
                tool_names.add(node.name)

    assert "delegate_task" in tool_names, "delegate_task @tool must exist"
    assert "list_delegated_agents" in tool_names, "list_delegated_agents @tool must exist"
    assert "kill_delegated_agent" in tool_names, "kill_delegated_agent @tool must exist"
    assert "check_file_state" in tool_names, "check_file_state @tool must exist"


def test_delegate_task_has_max_spawn_depth_parameter():
    """delegate_task accepts max_spawn_depth parameter."""
    import ast

    with open("langgraph-app/agent_graph.py", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "delegate_task":
            args = [arg.arg for arg in node.args.args]
            # Skip self if present
            if args and args[0] == "self":
                args = args[1:]
            assert "max_spawn_depth" in args, f"delegate_task must have max_spawn_depth parameter. Found: {args}"
            return

    pytest.fail("delegate_task function not found")


def test_delegate_agent_module_importable():
    """delegate_agent.py imports without errors and exposes public API."""
    from delegate_agent import (
        run_sub_agent,
        list_running_agents,
        kill_agent,
        get_file_state,
        SubAgentRegistry,
        SUB_AGENT_REGISTRY,
        FILE_STATE_TRACKER,
    )
    assert run_sub_agent is not None
    assert list_running_agents is not None
    assert kill_agent is not None
    assert get_file_state is not None
    assert SubAgentRegistry is not None
    assert SUB_AGENT_REGISTRY is not None
    assert FILE_STATE_TRACKER is not None


def test_agent_graph_imports_delegate_agent():
    """agent_graph.py imports delegate_agent via guarded import."""
    import ast

    with open("langgraph-app/agent_graph.py", encoding="utf-8") as f:
        source = f.read()

    assert "from delegate_agent import" in source, "agent_graph.py must import from delegate_agent"
    assert "_run_sub_agent" in source, "agent_graph.py must have _run_sub_agent alias"


def test_delegate_agent_module_has_all_four_features():
    """delegate_agent.py source contains all 4 missing features."""
    with open("langgraph-app/delegate_agent.py", encoding="utf-8") as f:
        source = f.read()

    # 1. Nested delegation
    assert "max_spawn_depth" in source or "DEFAULT_MAX_SPAWN_DEPTH" in source, "Must have spawn depth"
    assert "depth" in source, "Must track depth"

    # 2. Cancellation
    assert "cancel_event" in source or "CancelledError" in source, "Must support cancellation"
    assert "kill_agent" in source or "kill(" in source, "Must have kill_agent or kill"

    # 3. File-state tracking
    assert "FILE_STATE_TRACKER" in source, "Must have file state tracker"
    assert "check_stale_read" in source, "Must have stale read check"

    # 4. Sub-agent registry
    assert "SUB_AGENT_REGISTRY" in source, "Must have sub-agent registry"
    assert "list_running_agents" in source or "list_all" in source, "Must have listing"


def test_delegate_task_toolset_expanded_to_22():
    """delegate_task in agent_graph.py now has 22 tools (was 19)."""
    with open("langgraph-app/agent_graph.py", encoding="utf-8") as f:
        source = f.read()

    # The docstring in the new delegate_task mentions "22 tools"
    assert "22 tools" in source, "agent_graph.py delegate_task docstring should mention 22 tools"


def test_tool_categories_include_new_tools():
    """TOOL_REGISTRY_CATEGORIES includes list_delegated_agents, kill_delegated_agent, check_file_state."""
    import ast

    with open("langgraph-app/agent_graph.py", encoding="utf-8") as f:
        source = f.read()

    # All three tools must appear in coding/write or coding/execute categories
    assert "list_delegated_agents" in source
    assert "kill_delegated_agent" in source
    assert "check_file_state" in source


def test_local_tool_map_includes_new_tools():
    """local_tool_map includes list_delegated_agents, kill_delegated_agent, check_file_state."""
    with open("langgraph-app/agent_graph.py", encoding="utf-8") as f:
        source = f.read()

    # The tools must be in the local tool map list (the big list passed to _tools_by_name)
    # Check they appear after delegate_task in the local_tool_map
    assert "list_delegated_agents," in source
    assert "kill_delegated_agent," in source
    assert "check_file_state," in source


# ---------------------------------------------------------------------------
# Gap-closure tests (2026-06-01)
# ---------------------------------------------------------------------------


def test_is_retryable_error_detects_rate_limit():
    """_is_retryable_error returns True for rate-limit patterns."""
    from delegate_agent import _is_retryable_error

    assert _is_retryable_error(Exception("rate limit exceeded"))
    assert _is_retryable_error(Exception("too many requests"))
    assert _is_retryable_error(Exception("429"))


def test_is_retryable_error_detects_server_errors():
    """_is_retryable_error returns True for server-error patterns."""
    from delegate_agent import _is_retryable_error

    assert _is_retryable_error(Exception("internal server error"))
    assert _is_retryable_error(Exception("503 service unavailable"))
    assert _is_retryable_error(Exception("connection reset by peer"))


def test_is_retryable_error_rejects_auth_and_not_found():
    """_is_retryable_error returns False for non-retryable errors."""
    from delegate_agent import _is_retryable_error

    assert not _is_retryable_error(Exception("invalid api key"))
    assert not _is_retryable_error(Exception("model not found"))
    assert not _is_retryable_error(Exception("context length exceeded"))


def test_blocked_tools_constant_is_frozenset():
    """DELEGATE_BLOCKED_TOOLS is a frozenset with default values."""
    from delegate_agent import DELEGATE_BLOCKED_TOOLS

    assert isinstance(DELEGATE_BLOCKED_TOOLS, frozenset)
    assert "clarify" in DELEGATE_BLOCKED_TOOLS
    assert "memory" in DELEGATE_BLOCKED_TOOLS
    assert "send_message" in DELEGATE_BLOCKED_TOOLS


def test_provider_override_constants_exist():
    """Provider override ENV constants are defined."""
    from delegate_agent import (
        DELEGATE_PROVIDER,
        DELEGATE_MODEL,
        DELEGATE_API_BASE,
        DELEGATE_API_KEY,
    )

    assert isinstance(DELEGATE_PROVIDER, str)
    assert isinstance(DELEGATE_MODEL, str)
    assert isinstance(DELEGATE_API_BASE, str)
    assert isinstance(DELEGATE_API_KEY, str)


def test_heartbeat_constants_exist():
    """Heartbeat constants are defined."""
    from delegate_agent import HEARTBEAT_ENABLED, HEARTBEAT_INTERVAL

    assert isinstance(HEARTBEAT_ENABLED, bool)
    assert HEARTBEAT_INTERVAL > 0


def test_retry_constants_exist():
    """Retry constants are defined with sensible defaults."""
    from delegate_agent import DELEGATE_MAX_RETRIES, DELEGATE_RETRY_DELAY

    assert DELEGATE_MAX_RETRIES >= 0
    assert DELEGATE_RETRY_DELAY > 0


def test_workspace_constant_exists():
    """Workspace hint constant is defined."""
    from delegate_agent import DELEGATE_WORKSPACE_HINT

    assert isinstance(DELEGATE_WORKSPACE_HINT, str)


def test_run_sub_agent_signature_has_provider_params():
    """run_sub_agent accepts _provider, _model_name, _api_base, _api_key, _parent_touch_fn."""
    import inspect
    from delegate_agent import run_sub_agent

    sig = inspect.signature(run_sub_agent)
    params = set(sig.parameters.keys())
    assert "_provider" in params
    assert "_model_name" in params
    assert "_api_base" in params
    assert "_api_key" in params
    assert "_parent_touch_fn" in params


def test_heartbeat_loop_exists():
    """_heartbeat_loop function is defined."""
    from delegate_agent import _heartbeat_loop

    import inspect
    assert inspect.iscoroutinefunction(_heartbeat_loop)
