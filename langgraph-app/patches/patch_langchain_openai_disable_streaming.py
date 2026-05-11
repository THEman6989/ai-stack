"""Patch langchain-openai issue #35436 until an upstream release includes it.

The bug: when ChatOpenAI is configured with streaming=True and
disable_streaming="tool_calling", LangChain routes tool-bound calls through the
non-streaming _generate/_agenerate methods. langchain-openai still forwards
stream=True in the request payload, so the OpenAI client returns a Stream /
AsyncStream object and result construction crashes.

This applies the essence of langchain-ai/langchain PR #35457:
force payload["stream"] = False in the non-streaming OpenAI code paths.
"""

from __future__ import annotations

from pathlib import Path


PATCH_MARKER = "# AlphaRavis patch for langchain-ai/langchain#35436"
SYNC_NEEDLE = (
    "        payload = self._get_request_payload(messages, stop=stop, **kwargs)\n"
    "        generation_info = None\n"
)
ASYNC_NEEDLE = SYNC_NEEDLE
INSERT = (
    "        "
    + PATCH_MARKER
    + "\n"
    "        payload[\"stream\"] = False\n"
)


def main() -> None:
    import langchain_openai.chat_models.base as base

    path = Path(base.__file__)
    text = path.read_text(encoding="utf-8")

    if PATCH_MARKER in text:
        print(f"AlphaRavis LangChain patch already present: {path}")
        return

    occurrences = text.count(SYNC_NEEDLE)
    if occurrences < 2:
        raise RuntimeError(
            "Could not find both ChatOpenAI _generate/_agenerate payload sites "
            f"in {path}; found {occurrences}."
        )

    patched = text.replace(SYNC_NEEDLE, SYNC_NEEDLE.replace("        generation_info", INSERT + "        generation_info"), 2)
    path.write_text(patched, encoding="utf-8")
    print(f"Applied AlphaRavis LangChain patch for #35436: {path}")


if __name__ == "__main__":
    main()
