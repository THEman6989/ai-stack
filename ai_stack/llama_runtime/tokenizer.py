from __future__ import annotations

from typing import Any

from ai_stack.llama_runtime.client import LlamaCppRuntimeClient


async def count_text_tokens(runtime_client: LlamaCppRuntimeClient, text: str) -> int:
    """Count tokens with the selected llama-server's own tokenizer."""

    return await runtime_client.count_tokens_text(text)


async def count_chat_tokens(runtime_client: LlamaCppRuntimeClient, messages: list[dict[str, Any]] | list[Any]) -> int:
    """Render chat with /apply-template, then count with /tokenize on the same server."""

    return await runtime_client.count_tokens_chat(messages)

