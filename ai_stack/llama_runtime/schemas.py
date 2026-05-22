from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenizeResult:
    tokens: list[Any]
    raw: Any

    @property
    def count(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    raw: Any


@dataclass(frozen=True)
class RuntimeCallResult:
    ok: bool
    status_code: int
    response: Any

