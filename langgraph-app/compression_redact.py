from __future__ import annotations

import os
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SENSITIVE_FIELD_NAMES = (
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "token",
    "password",
    "passwd",
    "pwd",
    "secret",
    "authorization",
    "bearer",
    "creds_key",
    "creds_iv",
)

SECRET_VALUE_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"),
    re.compile(r"\btvly-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\b[A-Za-z0-9_\-]{24,}\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
]

SECRET_ASSIGNMENT_PATTERNS = [
    re.compile(
        r"(?i)\b("
        + "|".join(re.escape(name) for name in SENSITIVE_FIELD_NAMES)
        + r")\b\s*[:=]\s*([^\s'\"`,}]+)"
    ),
    re.compile(
        r'(?i)("(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|passwd|pwd|secret|authorization|bearer|creds_key|creds_iv)"\s*:\s*)"([^"]*)"'
    ),
    re.compile(
        r"(?i)('(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|passwd|pwd|secret|authorization|bearer|creds_key|creds_iv)'\s*:\s*)'([^']*)'"
    ),
    re.compile(r"(?i)\bAuthorization\s*:\s*Bearer\s+[A-Za-z0-9._\-~+/=]{12,}"),
]

CONNECTION_STRING_PASSWORD = re.compile(
    r"(?i)\b((?:mongodb|postgres(?:ql)?|mysql|redis)://[^:\s/@]+:)([^@\s]+)(@)"
)

URL_WITH_SECRET_QUERY = re.compile(r"https?://[^\s'\"<>]+")


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def mask_secret(value: str) -> str:
    value = str(value or "")
    if len(value) <= 6:
        return "<redacted>"
    return f"{value[:2]}...<redacted>...{value[-2:]}"


def _redact_url(match: re.Match[str]) -> str:
    raw_url = match.group(0)
    try:
        split = urlsplit(raw_url)
    except Exception:
        return raw_url

    netloc = split.netloc
    if "@" in netloc:
        userinfo, host = netloc.rsplit("@", 1)
        if ":" in userinfo:
            user, _password = userinfo.split(":", 1)
            netloc = f"{user}:<redacted>@{host}"

    query_pairs = []
    changed = False
    for key, value in parse_qsl(split.query, keep_blank_values=True):
        if any(name in key.lower().replace("-", "_") for name in SENSITIVE_FIELD_NAMES):
            query_pairs.append((key, "<redacted>"))
            changed = True
        else:
            query_pairs.append((key, value))

    if not changed and netloc == split.netloc:
        return raw_url
    return urlunsplit((split.scheme, netloc, split.path, urlencode(query_pairs), split.fragment))


def redact_sensitive_text(text: str) -> str:
    if not _env_bool("ALPHARAVIS_COMPRESSION_REDACT_SECRETS", "true"):
        return str(text or "")

    redacted = str(text or "")
    redacted = CONNECTION_STRING_PASSWORD.sub(r"\1<redacted>\3", redacted)
    redacted = URL_WITH_SECRET_QUERY.sub(_redact_url, redacted)

    for pattern in SECRET_ASSIGNMENT_PATTERNS:
        def _replace(match: re.Match[str]) -> str:
            if len(match.groups()) >= 2:
                prefix = match.group(1)
                if prefix.rstrip().endswith((':', '"', "'")):
                    return f"{prefix}<redacted>"
                return f"{prefix}=<redacted>"
            return "<redacted-secret>"

        redacted = pattern.sub(_replace, redacted)

    for pattern in SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("<redacted-secret>", redacted)

    return redacted
