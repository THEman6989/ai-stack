from __future__ import annotations

import time
from typing import Any

import httpx

from ai_stack.llama_runtime.schemas import RenderedPrompt, TokenizeResult
from ai_stack.ubuntu_llama_manager.schemas import UbuntuLlamaInstance


class LlamaCppRuntimeClient:
    """Data-plane client that talks directly to a selected llama-server."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_instance(cls, instance: UbuntuLlamaInstance, *, timeout_seconds: float = 30) -> "LlamaCppRuntimeClient":
        return cls(instance.base_url, timeout_seconds=timeout_seconds)

    async def _request(self, method: str, path: str, *, json_payload: dict[str, Any] | None = None) -> Any:
        if not self.base_url:
            raise ValueError("llama runtime base_url is empty")
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.request(method.upper(), url, json=json_payload)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return response.text

    async def get_models(self) -> Any:
        return await self._request("GET", "/v1/models")

    async def get_slots(self) -> Any:
        return await self._request("GET", "/slots")

    async def apply_template(self, messages: list[dict[str, Any]] | list[Any]) -> RenderedPrompt:
        payload = {"messages": [_message_to_dict(message) for message in messages]}
        data = await self._request("POST", "/apply-template", json_payload=payload)
        if isinstance(data, str):
            return RenderedPrompt(text=data, raw=data)
        if isinstance(data, dict):
            for key in ("prompt", "content", "text", "rendered", "template"):
                value = data.get(key)
                if isinstance(value, str):
                    return RenderedPrompt(text=value, raw=data)
        return RenderedPrompt(text=str(data), raw=data)

    async def tokenize(self, text: str) -> TokenizeResult:
        data = await self._request("POST", "/tokenize", json_payload={"content": text})
        if isinstance(data, dict):
            tokens = data.get("tokens")
            if isinstance(tokens, list):
                return TokenizeResult(tokens=tokens, raw=data)
            if isinstance(data.get("token_ids"), list):
                return TokenizeResult(tokens=data["token_ids"], raw=data)
        if isinstance(data, list):
            return TokenizeResult(tokens=data, raw=data)
        return TokenizeResult(tokens=[], raw=data)

    async def count_tokens_text(self, text: str) -> int:
        return (await self.tokenize(text)).count

    async def count_tokens_chat(self, messages: list[dict[str, Any]] | list[Any]) -> int:
        rendered = await self.apply_template(messages)
        return await self.count_tokens_text(rendered.text)

    async def completion(self, **payload: Any) -> Any:
        return await self._request("POST", "/completion", json_payload=payload)

    async def chat_completion(self, **payload: Any) -> Any:
        return await self._request("POST", "/v1/chat/completions", json_payload=payload)


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    role = getattr(message, "role", None) or getattr(message, "type", None) or "user"
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"
    elif role == "system":
        role = "system"
    return {"role": role, "content": getattr(message, "content", message)}

