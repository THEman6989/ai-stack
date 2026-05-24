"""
Robust MCP tool loader for LangGraph — Hermes-style resilience.

Adds reconnect with exponential backoff, circuit breaker, per-server
timeout configuration, and error classification on top of
``langchain_mcp_adapters``.

Design principles (borrowed from Hermes' ``tools/mcp_tool.py``):
- Reconnect: automatic retry with 1s→2s→4s→… backoff on connection loss.
- Circuit breaker: 3-state (closed/open/half-open). After 3 consecutive
  failures, short-circuits for 60s to prevent iteration-burn loops.
- Timeout: per-server ``timeout`` and ``connect_timeout`` in mcp.json.
- Error classification: auth errors, session expiry, transient vs permanent.

Usage (in agent_graph.py)::

    from mcp_client import load_robust_mcp_tools

    mcp_tools = await load_robust_mcp_tools(stack)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Module-level state (mirrors agent_graph.py globals)
MCP_LOAD_WARNINGS: List[str] = []
MCP_SERVER_INFOS: List[Dict[str, Any]] = []
MCP_CONFIG_PATHS: List[str] = []

# ---------------------------------------------------------------------------
# Constants (mirroring Hermes' mcp_tool.py)
# ---------------------------------------------------------------------------

_DEFAULT_TOOL_TIMEOUT = 120       # seconds for tool calls
_DEFAULT_CONNECT_TIMEOUT = 60     # seconds for initial connection
_MAX_RECONNECT_RETRIES = 5
_MAX_BACKOFF_SECONDS = 60
_CIRCUIT_BREAKER_THRESHOLD = 3
_CIRCUIT_BREAKER_COOLDOWN_SEC = 60.0

# Substrings that indicate a transport-level failure (connection lost,
# session expired) — distinct from application-level tool errors.
_SESSION_EXPIRED_MARKERS: Tuple[str, ...] = (
    "invalid or expired session",
    "expired session",
    "session expired",
    "session not found",
    "unknown session",
    "session terminated",
    "closedresourceerror",
    "closed resource",
    "transport is closed",
    "connection closed",
    "connection refused",
    "broken pipe",
    "end of file",
    "connection reset",
    "remote end closed",
    "server disconnected",
    "eof received",
)

# HTTP status codes that indicate the server is temporarily unavailable.
_RECOVERABLE_HTTP_STATUSES: frozenset = frozenset({502, 503, 504})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _exc_str(exc: BaseException) -> str:
    """Non-empty human-readable string for *exc*."""
    text = str(exc).strip()
    return text if text else repr(exc)


def _is_transient_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like a transient transport failure."""
    msg = str(exc).lower()
    if not msg:
        return False
    for marker in _SESSION_EXPIRED_MARKERS:
        if marker in msg:
            return True
    # Check for HTTP status codes in the message
    for code in _RECOVERABLE_HTTP_STATUSES:
        if str(code) in msg:
            return True
    return False


def _is_auth_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like an authentication failure."""
    msg = str(exc).lower()
    if not msg:
        return False
    auth_markers = ("401", "unauthorized", "unauthenticated", "forbidden",
                    "403", "token expired", "invalid token", "access denied")
    return any(m in msg for m in auth_markers)


# ---------------------------------------------------------------------------
# Circuit breaker state
# ---------------------------------------------------------------------------

class _CircuitBreaker:
    """3-state circuit breaker for a single MCP server.

    closed   → failures < threshold, all calls go through.
    open     → threshold reached, short-circuits for cooldown.
    half-open → cooldown elapsed, next call is a probe.
    """

    def __init__(self, threshold: int = _CIRCUIT_BREAKER_THRESHOLD,
                 cooldown: float = _CIRCUIT_BREAKER_COOLDOWN_SEC):
        self._threshold = threshold
        self._cooldown = cooldown
        self._failures = 0
        self._opened_at: float = 0.0

    @property
    def is_open(self) -> bool:
        if self._failures < self._threshold:
            return False
        age = time.monotonic() - self._opened_at
        return age < self._cooldown

    @property
    def remaining_cooldown(self) -> float:
        if self._failures < self._threshold:
            return 0.0
        return max(0.0, self._cooldown - (time.monotonic() - self._opened_at))

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()

    def record_success(self) -> None:
        self._failures = 0

    def reset(self) -> None:
        self._failures = 0


# ---------------------------------------------------------------------------
# Robust MCP Server Manager
# ---------------------------------------------------------------------------

class RobustMCPServerManager:
    """Manages one MCP server connection with reconnect and circuit breaker.

    Wraps ``langchain_mcp_adapters.MultiServerMCPClient``, adding:
    - Automatic reconnection with exponential backoff on transport failures.
    - Circuit breaker to prevent retry-loop burn on persistent failures.
    - Configurable per-server timeouts.
    - Tool wrapping with error classification.
    """

    def __init__(self, server_name: str, connection: Dict[str, Any],
                 server_config: Dict[str, Any], tool_prefix: bool = True):
        self.name = server_name
        self._connection = connection
        self._server_config = server_config
        self._tool_prefix = tool_prefix
        self._tool_timeout = float(
            server_config.get("timeout", _DEFAULT_TOOL_TIMEOUT)
        )
        self._connect_timeout = float(
            server_config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
        )
        self._breaker = _CircuitBreaker()
        self._session: Any = None
        self._tools: List[Any] = []
        self._backoff = 1.0
        self._client: Any = None
        self._stack: Optional[contextlib.AsyncExitStack] = None

    async def connect(self, stack: contextlib.AsyncExitStack) -> List[Any]:
        """Connect to the server and load tools.

        Uses ``langchain_mcp_adapters`` for protocol handling.
        Tools are wrapped with circuit breaker and error handling.
        """
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        self._client = MultiServerMCPClient({self.name: self._connection})

        try:
            session = await asyncio.wait_for(
                stack.enter_async_context(self._client.session(self.name)),
                timeout=self._connect_timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"MCP server '{self.name}' connection timed out "
                f"after {self._connect_timeout}s"
            )

        self._session = session
        self._stack = stack

        try:
            tools = await load_mcp_tools(
                session,
                server_name=self.name,
                tool_name_prefix=self._tool_prefix,
            )
        except TypeError:
            # Older langchain_mcp_adapters without server_name/tool_name_prefix
            tools = await load_mcp_tools(session)

        tool_list = list(tools)
        # Wrap each tool with circuit breaker + error handling
        wrapped = [self._wrap_tool(t) for t in tool_list]
        self._tools = wrapped
        self._backoff = 1.0  # reset backoff on successful connect
        return wrapped

    async def _reconnect(self, stack: contextlib.AsyncExitStack) -> List[Any]:
        """Reconnect after a transport failure, with backoff."""
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        # Close old session if any
        self._session = None

        backoff = self._backoff
        for attempt in range(1, _MAX_RECONNECT_RETRIES + 1):
            logger.info(
                "MCP server '%s': reconnecting (attempt %d/%d, %.0fs backoff)",
                self.name, attempt, _MAX_RECONNECT_RETRIES, backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)

            try:
                self._client = MultiServerMCPClient(
                    {self.name: self._connection}
                )
                session = await asyncio.wait_for(
                    stack.enter_async_context(
                        self._client.session(self.name)
                    ),
                    timeout=self._connect_timeout,
                )
                self._session = session
                tools = await load_mcp_tools(
                    session,
                    server_name=self.name,
                    tool_name_prefix=self._tool_prefix,
                )
                tool_list = list(tools)
                wrapped = [self._wrap_tool(t) for t in tool_list]
                self._tools = wrapped
                self._backoff = 1.0  # reset on success
                logger.info(
                    "MCP server '%s': reconnected successfully", self.name,
                )
                return wrapped
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP server '%s': reconnect timed out", self.name,
                )
            except Exception as exc:
                logger.warning(
                    "MCP server '%s': reconnect failed: %s",
                    self.name, _exc_str(exc),
                )

        self._backoff = min(self._backoff * 2, _MAX_BACKOFF_SECONDS)
        raise ConnectionError(
            f"MCP server '{self.name}' failed to reconnect after "
            f"{_MAX_RECONNECT_RETRIES} attempts"
        )

    def _wrap_tool(self, tool: Any) -> Any:
        """Wrap a LangChain tool with circuit breaker and error handling.

        The wrapper intercepts ``_arun`` to:
        1. Check circuit breaker — short-circuit if open.
        2. Enforce per-server timeout.
        3. Classify errors: auth, transient, permanent.
        4. Bump/reset circuit breaker state accordingly.
        """
        # Store original methods
        _original_arun = tool._arun
        _original_run = tool._run
        _manager = self

        async def _wrapped_arun(*args, **kwargs):
            # Check circuit breaker
            breaker = _manager._breaker
            if breaker.is_open:
                remaining = int(breaker.remaining_cooldown)
                return json.dumps({
                    "error": (
                        f"MCP server '{_manager.name}' is unreachable after "
                        f"{breaker._failures} consecutive failures. "
                        f"Auto-retry available in ~{remaining}s. "
                        f"Do NOT retry — use alternative approaches."
                    )
                })

            try:
                result = await asyncio.wait_for(
                    _original_arun(*args, **kwargs),
                    timeout=_manager._tool_timeout,
                )
                breaker.record_success()
                return result
            except asyncio.TimeoutError:
                breaker.record_failure()
                return json.dumps({
                    "error": (
                        f"MCP tool '{tool.name}' on server '{_manager.name}' "
                        f"timed out after {_manager._tool_timeout}s"
                    )
                })
            except Exception as exc:
                exc_str = _exc_str(exc)
                if _is_auth_error(exc):
                    breaker.record_failure()
                    return json.dumps({
                        "error": (
                            f"MCP server '{_manager.name}' requires "
                            f"re-authentication: {exc_str}"
                        ),
                        "needs_reauth": True,
                    })
                if _is_transient_error(exc):
                    # Trigger reconnect and retry once
                    breaker.record_failure()
                    stack = _manager._stack
                    if stack is not None:
                        try:
                            await _manager._reconnect(stack)
                            # Retry after successful reconnect
                            result = await asyncio.wait_for(
                                _original_arun(*args, **kwargs),
                                timeout=_manager._tool_timeout,
                            )
                            breaker.record_success()
                            return result
                        except Exception as retry_exc:
                            logger.warning(
                                "MCP server '%s': retry after reconnect failed: %s",
                                _manager.name, _exc_str(retry_exc),
                            )
                    return json.dumps({
                        "error": (
                            f"MCP server '{_manager.name}' connection lost: "
                            f"{exc_str}. Reconnection attempted."
                        )
                    })
                # Permanent / application error
                return json.dumps({
                    "error": f"MCP tool '{tool.name}' failed: {exc_str}"
                })

        def _wrapped_run(*args, **kwargs):
            # Sync fallback: run the async version
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, caller should use _arun
                    raise RuntimeError(
                        "Cannot call sync _run in async context"
                    )
                return asyncio.run(_wrapped_arun(*args, **kwargs))
            except RuntimeError:
                # No event loop running
                return asyncio.run(_wrapped_arun(*args, **kwargs))

        tool._arun = _wrapped_arun
        tool._run = _wrapped_run

        # Attach server info for introspection
        tool._mcp_server_name = self.name
        tool._mcp_manager = self

        return tool


# ---------------------------------------------------------------------------
# Config loading (replaces agent_graph.py helpers)
# ---------------------------------------------------------------------------

PIXELLE_URL = os.getenv("PIXELLE_URL", "http://localhost:9004")


def _workspace_root() -> str:
    return os.getenv(
        "ALPHARAVIS_WORKSPACE_ROOT",
        os.getenv("LANGGRAPH_WORKSPACE_ROOT", os.getcwd()),
    )


def _resolve_mcp_path(value: str) -> Path:
    expanded = os.path.expandvars(value.strip())
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return path
    return Path(_workspace_root()) / path


def _mcp_config_candidate_paths() -> List[Path]:
    paths: List[Path] = [
        Path.home() / ".deepagents" / ".mcp.json",
        Path(_workspace_root()) / ".deepagents" / ".mcp.json",
        Path(_workspace_root()) / ".mcp.json",
        Path(__file__).resolve().parent / "mcp.json",
    ]

    extra_paths = os.getenv("ALPHARAVIS_MCP_CONFIG_PATHS", "")
    for value in extra_paths.split("|"):
        if value.strip():
            paths.append(_resolve_mcp_path(value))

    explicit_path = os.getenv("ALPHARAVIS_MCP_CONFIG_PATH", "")
    if explicit_path.strip():
        paths.append(_resolve_mcp_path(explicit_path))

    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _expand_mcp_config_value(value: Any) -> Any:
    """Resolve env vars in config values, with PIXELLE_URL fallback."""
    if isinstance(value, str):
        # Ensure PIXELLE_URL is available for ${PIXELLE_URL} references
        with _pixelle_url_context():
            return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_mcp_config_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_mcp_config_value(item)
                for key, item in value.items()}
    return value


@contextlib.contextmanager
def _pixelle_url_context():
    """Temporarily set PIXELLE_URL if not already in the environment."""
    had_key = "PIXELLE_URL" in os.environ
    old_value = os.environ.get("PIXELLE_URL")
    if not had_key:
        os.environ["PIXELLE_URL"] = PIXELLE_URL
    try:
        yield
    finally:
        if not had_key:
            del os.environ["PIXELLE_URL"]
        else:
            os.environ["PIXELLE_URL"] = old_value


def _mcp_transport(server_config: Dict[str, Any]) -> str:
    return str(
        server_config.get("type", server_config.get("transport", "stdio"))
    ).lower()


def load_mcp_config() -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Load MCP config from candidate file paths.

    Returns:
        (config_dict, config_paths_found, warnings)
    """
    allow_stdio = _env_bool("ALPHARAVIS_MCP_ALLOW_STDIO", "false")
    servers: Dict[str, Dict[str, Any]] = {}
    config_paths: List[str] = []
    warnings: List[str] = []

    for path in _mcp_config_candidate_paths():
        if not path.is_file():
            continue
        config_paths.append(str(path))
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"{path}: could not parse MCP config: {exc}")
            continue

        raw_servers = raw.get("mcpServers", {})
        if not isinstance(raw_servers, dict):
            warnings.append(
                f"{path}: MCP config must contain object field `mcpServers`."
            )
            continue

        for name, server_config in raw_servers.items():
            if not isinstance(server_config, dict):
                warnings.append(
                    f"{path}: MCP server `{name}` config must be an object."
                )
                continue
            server_config = _expand_mcp_config_value(server_config)
            transport = _mcp_transport(server_config)
            if transport == "stdio" and not allow_stdio:
                warnings.append(
                    f"{path}: skipped stdio MCP server `{name}`. "
                    "Set ALPHARAVIS_MCP_ALLOW_STDIO=true only for trusted "
                    "configs."
                )
                continue
            if transport in {"http", "sse"} and not server_config.get("url"):
                warnings.append(
                    f"{path}: MCP server `{name}` missing `url`."
                )
                continue
            if transport == "stdio" and not server_config.get("command"):
                warnings.append(
                    f"{path}: MCP server `{name}` missing `command`."
                )
                continue
            if transport not in {"http", "sse", "stdio", "streamable_http"}:
                warnings.append(
                    f"{path}: MCP server `{name}` has unsupported "
                    f"transport `{transport}`."
                )
                continue
            servers[str(name)] = server_config

    return {"mcpServers": servers}, config_paths, warnings


def _mcp_connection_from_config(
    server_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Build a connection dict for langchain_mcp_adapters."""
    transport = _mcp_transport(server_config)
    if transport == "http":
        transport = "streamable_http"

    if transport in {"sse", "streamable_http"}:
        connection = {"transport": transport, "url": server_config["url"]}
        if server_config.get("headers"):
            connection["headers"] = server_config["headers"]
        return connection

    return {
        "transport": "stdio",
        "command": server_config["command"],
        "args": server_config.get("args", []),
        "env": server_config.get("env") or None,
    }


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for agent_graph._load_configured_mcp_tools
# ---------------------------------------------------------------------------

async def load_robust_mcp_tools(
    stack: contextlib.AsyncExitStack,
) -> List[Any]:
    """Load MCP tools with reconnect and circuit breaker.

    Replaces ``_load_configured_mcp_tools()`` in ``agent_graph.py``.

    Features (from Hermes):
    - Reconnect with exponential backoff on connection loss.
    - Circuit breaker: 3 failures → 60s cooldown.
    - Per-server ``timeout`` and ``connect_timeout`` in mcp.json.
    - Error classification: auth, transient, permanent.
    - Pixelle SSE support via ``${PIXELLE_URL}`` env var.

    Args:
        stack: ``contextlib.AsyncExitStack`` for session lifecycle.

    Returns:
        List of wrapped LangChain tools.
    """
    # Use module-level globals (also exported for agent_graph.py)
    global MCP_LOAD_WARNINGS, MCP_SERVER_INFOS, MCP_CONFIG_PATHS
    MCP_LOAD_WARNINGS = []
    MCP_SERVER_INFOS = []
    MCP_CONFIG_PATHS = []

    strict = _env_bool("ALPHARAVIS_MCP_STRICT", "false")
    tool_prefix = _env_bool("ALPHARAVIS_MCP_TOOL_PREFIX", "true")

    config, config_paths, warnings = load_mcp_config()
    MCP_LOAD_WARNINGS = list(warnings)
    MCP_CONFIG_PATHS = list(config_paths)
    if strict and warnings:
        raise RuntimeError("Invalid MCP config:\n" + "\n".join(warnings))

    servers = config.get("mcpServers", {})
    if not servers:
        return []

    all_tools = []
    server_infos = []

    for server_name in sorted(servers):
        server_config = servers[server_name]
        connection = _mcp_connection_from_config(server_config)
        manager = RobustMCPServerManager(
            server_name, connection, server_config,
            tool_prefix=tool_prefix,
        )

        try:
            tools = await manager.connect(stack)
            all_tools.extend(tools)
            server_infos.append({
                "name": server_name,
                "transport": _mcp_transport(server_config),
                "tools": [
                    {
                        "name": getattr(t, "name", "unknown"),
                        "description": getattr(t, "description", "") or "",
                    }
                    for t in tools
                ],
            })
        except Exception as exc:
            message = (
                f"MCP server `{server_name}` could not be loaded: "
                f"{_exc_str(exc)}"
            )
            warnings.append(message)
            if strict:
                raise RuntimeError(message) from exc

    # Expose server infos
    MCP_SERVER_INFOS = server_infos

    if server_infos:
        total_tools = sum(
            len(info["tools"]) for info in server_infos
        )
        print(
            f"Loaded {total_tools} MCP tools from "
            f"{len(server_infos)} server(s) with reconnect + circuit breaker."
        )

    # Tools carry _mcp_manager reference for introspection
    return all_tools
