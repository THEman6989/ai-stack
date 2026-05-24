from __future__ import annotations

import re
import shlex

# ---------- command word / segment helpers ----------


def first_shell_word(command: str) -> str:
    try:
        parts = shlex.split(command, posix=True)
    except ValueError:
        return ""
    return parts[0] if parts else ""


def command_segments(command: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"\s*(?:&&|\|\||;|\|)\s*", command) if segment.strip()]


# ---------- read-only command classifier ----------

# Commands that are never dangerous by themselves (subcommands gated below)
_SAFE_COMMAND_ROOTS: set[str] = {
    "awk",
    "cat",
    "curl",
    "date",
    "df",
    "docker",
    "du",
    "echo",
    "file",
    "find",
    "free",
    "git",
    "grep",
    "head",
    "hostname",
    "id",
    "journalctl",
    "less",
    "ls",
    "netstat",
    "pm2",
    "ps",
    "pwd",
    "rg",
    "service",
    "ss",
    "stat",
    "systemctl",
    "tail",
    "top",
    "uname",
    "uptime",
    "which",
    "whoami",
}

# Permitted subcommands for commands that are safe only with specific subcommands
_ALLOWED_SUBCOMMANDS: dict[str, set[str]] = {
    "docker": {"ps", "logs", "inspect", "version", "info", "stats", "compose"},
    "git": {"status", "diff", "log", "show", "branch", "remote", "rev-parse"},
    "pm2": {"list", "status", "logs", "show", "describe", "monit"},
    "service": {"status"},
    "systemctl": {"status", "is-active", "is-enabled", "list-units", "list-timers"},
}

# Dangerous patterns that make a command NOT read-only
_DANGEROUS_PATTERNS: list[str] = [
    r"\b(rm|rmdir|mv|cp|chmod|chown|dd|mkfs|fdisk|parted|mount|umount|truncate|tee)\b",
    r"\b(kill|pkill|killall|reboot|shutdown|poweroff)\b",
    r"\b(apt|apt-get|apk|yum|dnf|pip|pip3|npm|pnpm|yarn)\s+(install|remove|uninstall|upgrade|update|add)\b",
    r"\b(git)\s+(push|commit|merge|rebase|reset|clean|checkout|switch|restore)\b",
    r"\b(docker)\s+(restart|stop|start|kill|rm|rmi|compose\s+(up|down|restart|stop|start|pull|build)|system\s+prune)\b",
    r"\b(systemctl|service)\s+(restart|stop|start|enable|disable|reload)\b",
    r"\b(pm2)\s+(restart|stop|start|delete|reload|save)\b",
    r"\bsed\s+-i\b",
    r"(^|[^<])>(?!>)|>>",
]


def is_read_only_command(command: str) -> bool:
    """Return True if the command contains only safe/read-only operations."""
    command = command.strip()
    if not command:
        return False

    lowered = command.lower()
    if any(re.search(pattern, lowered) for pattern in _DANGEROUS_PATTERNS):
        return False

    for segment in command_segments(command):
        root = first_shell_word(segment)
        if root not in _SAFE_COMMAND_ROOTS:
            return False
        if root in _ALLOWED_SUBCOMMANDS:
            parts = shlex.split(segment, posix=True)
            subcommand = parts[1] if len(parts) > 1 else ""
            if root == "docker" and subcommand == "compose":
                compose_cmd = parts[2] if len(parts) > 2 else ""
                if compose_cmd not in {"ps", "logs", "config"}:
                    return False
            elif subcommand not in _ALLOWED_SUBCOMMANDS[root]:
                return False

    return True
