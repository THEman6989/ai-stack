"""Experimental LangChain Responses tool-streaming patch for AlphaRavis.

Enable with:

    ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true

The local LiteLLM/llama.cpp Responses stream can expose reasoning and
function-call output items in a shape that currently trips langchain-openai
stream aggregation:

- reasoning `output_item.added` starts with empty `content`
- reasoning `output_item.done` carries the final reasoning content but is not
  emitted back into the aggregated message
- function-call items can reuse the prior output index and merge into the
  reasoning block
- partial function-call argument deltas are parsed as invalid tool calls before
  the final JSON exists

After a tool executes, the next Responses request can therefore contain a
malformed reasoning item with empty content and function-call fields, which
providers reject as:

    item['content'] is empty

This patch keeps function calls on their own LangChain content index when the
provider reuses the previous output index. It is intentionally gated because
full tool streaming remains experimental.
"""

from __future__ import annotations

import os
from pathlib import Path


OLD_FUNCTION_CALL_PATCH_MARKER = "# AlphaRavis experimental patch for Responses function_call streaming"
FUNCTION_CALL_PATCH_MARKER = "# AlphaRavis experimental patch for Responses function_call streaming v2"
OLD_REASONING_DONE_PATCH_MARKER = "# AlphaRavis experimental patch for Responses reasoning done streaming"
REASONING_DONE_PATCH_MARKER = "# AlphaRavis experimental patch for Responses reasoning done streaming v2"
FUNCTION_CALL_DELTA_PATCH_MARKER = "# AlphaRavis experimental patch for Responses function_call delta buffering"
FUNCTION_CALL_DONE_PATCH_MARKER = "# AlphaRavis experimental patch for Responses function_call done buffering"
FUNCTION_CALL_NEEDLE = (
    "    elif (\n"
    "        chunk.type == \"response.output_item.added\"\n"
    "        and chunk.item.type == \"function_call\"\n"
    "    ):\n"
    "        _advance(chunk.output_index)\n"
    "        tool_call_chunks.append(\n"
    "            {\n"
    "                \"type\": \"tool_call_chunk\",\n"
    "                \"name\": chunk.item.name,\n"
    "                \"args\": chunk.item.arguments,\n"
    "                \"id\": chunk.item.call_id,\n"
    "                \"index\": current_index,\n"
    "            }\n"
    "        )\n"
    "        function_call_content: dict = {\n"
    "            \"type\": \"function_call\",\n"
    "            \"name\": chunk.item.name,\n"
    "            \"arguments\": chunk.item.arguments,\n"
    "            \"call_id\": chunk.item.call_id,\n"
    "            \"id\": chunk.item.id,\n"
    "            \"index\": current_index,\n"
    "        }\n"
    "        if getattr(chunk.item, \"namespace\", None) is not None:\n"
    "            function_call_content[\"namespace\"] = chunk.item.namespace\n"
    "        content.append(function_call_content)\n"
)
OLD_FUNCTION_CALL_REPLACEMENT = (
    "    elif (\n"
    "        chunk.type == \"response.output_item.added\"\n"
    "        and chunk.item.type == \"function_call\"\n"
    "    ):\n"
    f"        {OLD_FUNCTION_CALL_PATCH_MARKER}\n"
    "        if current_output_index == chunk.output_index and current_index >= 0:\n"
    "            current_index += 1\n"
    "            current_output_index = chunk.output_index\n"
    "            current_sub_index = -1\n"
    "        else:\n"
    "            _advance(chunk.output_index)\n"
    "        tool_call_chunks.append(\n"
    "            {\n"
    "                \"type\": \"tool_call_chunk\",\n"
    "                \"name\": chunk.item.name,\n"
    "                \"args\": chunk.item.arguments,\n"
    "                \"id\": chunk.item.call_id,\n"
    "                \"index\": current_index,\n"
    "            }\n"
    "        )\n"
    "        function_call_content: dict = {\n"
    "            \"type\": \"function_call\",\n"
    "            \"name\": chunk.item.name,\n"
    "            \"arguments\": chunk.item.arguments,\n"
    "            \"call_id\": chunk.item.call_id,\n"
    "            \"id\": chunk.item.id,\n"
    "            \"index\": current_index,\n"
    "        }\n"
    "        if getattr(chunk.item, \"namespace\", None) is not None:\n"
    "            function_call_content[\"namespace\"] = chunk.item.namespace\n"
    "        content.append(function_call_content)\n"
)
FUNCTION_CALL_REPLACEMENT = (
    "    elif (\n"
    "        chunk.type == \"response.output_item.added\"\n"
    "        and chunk.item.type == \"function_call\"\n"
    "    ):\n"
    f"        {FUNCTION_CALL_PATCH_MARKER}\n"
    "        if current_output_index == chunk.output_index and current_index >= 0:\n"
    "            current_index += 1\n"
    "            current_output_index = chunk.output_index\n"
    "            current_sub_index = -1\n"
    "        else:\n"
    "            _advance(chunk.output_index)\n"
    "        # Buffer function-call metadata until response.output_item.done,\n"
    "        # where the provider exposes complete name, id, and JSON args.\n"
)
REASONING_DONE_NEEDLE = (
    "    elif chunk.type == \"response.refusal.done\":\n"
    "        content.append({\"type\": \"refusal\", \"refusal\": chunk.refusal})\n"
)
FUNCTION_CALL_DELTA_NEEDLE = (
    "    elif chunk.type == \"response.function_call_arguments.delta\":\n"
    "        _advance(chunk.output_index)\n"
    "        tool_call_chunks.append(\n"
    "            {\"type\": \"tool_call_chunk\", \"args\": chunk.delta, \"index\": current_index}\n"
    "        )\n"
    "        content.append(\n"
    "            {\"type\": \"function_call\", \"arguments\": chunk.delta, \"index\": current_index}\n"
    "        )\n"
)
FUNCTION_CALL_DELTA_REPLACEMENT = (
    "    elif chunk.type == \"response.function_call_arguments.delta\":\n"
    f"        {FUNCTION_CALL_DELTA_PATCH_MARKER}\n"
    "        if current_output_index != chunk.output_index:\n"
    "            _advance(chunk.output_index)\n"
    "        # Partial arguments are not valid JSON. Emit the complete tool call\n"
    "        # when response.output_item.done arrives.\n"
    "        pass\n"
)
FUNCTION_CALL_DONE_NEEDLE = (
    "    elif (\n"
    "        chunk.type == \"response.output_item.done\"\n"
    "        and chunk.item.type == \"custom_tool_call\"\n"
    "    ):\n"
)
FUNCTION_CALL_DONE_REPLACEMENT = (
    "    elif chunk.type == \"response.output_item.done\" and chunk.item.type == \"function_call\":\n"
    f"        {FUNCTION_CALL_DONE_PATCH_MARKER}\n"
    "        if current_output_index != chunk.output_index:\n"
    "            _advance(chunk.output_index)\n"
    "        tool_call_chunks.append(\n"
    "            {\n"
    "                \"type\": \"tool_call_chunk\",\n"
    "                \"name\": chunk.item.name,\n"
    "                \"args\": chunk.item.arguments,\n"
    "                \"id\": chunk.item.call_id,\n"
    "                \"index\": current_index,\n"
    "            }\n"
    "        )\n"
    "        function_call_content: dict = {\n"
    "            \"type\": \"function_call\",\n"
    "            \"name\": chunk.item.name,\n"
    "            \"arguments\": chunk.item.arguments,\n"
    "            \"call_id\": chunk.item.call_id,\n"
    "            \"id\": getattr(chunk.item, \"id\", None),\n"
    "            \"index\": current_index,\n"
    "        }\n"
    "        if getattr(chunk.item, \"namespace\", None) is not None:\n"
    "            function_call_content[\"namespace\"] = chunk.item.namespace\n"
    "        content.append(function_call_content)\n"
    "    elif (\n"
    "        chunk.type == \"response.output_item.done\"\n"
    "        and chunk.item.type == \"custom_tool_call\"\n"
    "    ):\n"
)
REASONING_DONE_REPLACEMENT = (
    "    elif chunk.type == \"response.output_item.done\" and chunk.item.type == \"reasoning\":\n"
    f"        {REASONING_DONE_PATCH_MARKER}\n"
    "        if current_output_index == chunk.output_index and current_index > 0:\n"
    "            reasoning_index = current_index - 1\n"
    "        else:\n"
    "            _advance(chunk.output_index)\n"
    "            reasoning_index = current_index\n"
    "        current_sub_index = 0\n"
    "        reasoning = chunk.item.model_dump(exclude_none=True, mode=\"json\")\n"
    "        reasoning[\"index\"] = reasoning_index\n"
    "        content.append(reasoning)\n"
    "    elif chunk.type == \"response.refusal.done\":\n"
    "        content.append({\"type\": \"refusal\", \"refusal\": chunk.refusal})\n"
)
OLD_REASONING_DONE_REPLACEMENT = (
    "    elif chunk.type == \"response.output_item.done\" and chunk.item.type == \"reasoning\":\n"
    f"        {OLD_REASONING_DONE_PATCH_MARKER}\n"
    "        _advance(chunk.output_index)\n"
    "        current_sub_index = 0\n"
    "        reasoning = chunk.item.model_dump(exclude_none=True, mode=\"json\")\n"
    "        reasoning[\"index\"] = current_index\n"
    "        content.append(reasoning)\n"
    "    elif chunk.type == \"response.refusal.done\":\n"
    "        content.append({\"type\": \"refusal\", \"refusal\": chunk.refusal})\n"
)


def _enabled() -> bool:
    return os.getenv("ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def patch_source(text: str) -> str:
    patched = text
    if FUNCTION_CALL_PATCH_MARKER not in patched:
        if OLD_FUNCTION_CALL_REPLACEMENT in patched:
            patched = patched.replace(OLD_FUNCTION_CALL_REPLACEMENT, FUNCTION_CALL_REPLACEMENT, 1)
        elif FUNCTION_CALL_NEEDLE in patched:
            patched = patched.replace(FUNCTION_CALL_NEEDLE, FUNCTION_CALL_REPLACEMENT, 1)
        else:
            raise RuntimeError("Could not find Responses function_call streaming conversion site.")
    if REASONING_DONE_PATCH_MARKER not in patched:
        if OLD_REASONING_DONE_REPLACEMENT in patched:
            patched = patched.replace(OLD_REASONING_DONE_REPLACEMENT, REASONING_DONE_REPLACEMENT, 1)
        elif REASONING_DONE_NEEDLE in patched:
            patched = patched.replace(REASONING_DONE_NEEDLE, REASONING_DONE_REPLACEMENT, 1)
        else:
            raise RuntimeError("Could not find Responses reasoning done streaming conversion site.")
    if FUNCTION_CALL_DELTA_PATCH_MARKER not in patched:
        if FUNCTION_CALL_DELTA_NEEDLE not in patched:
            raise RuntimeError("Could not find Responses function_call delta conversion site.")
        patched = patched.replace(FUNCTION_CALL_DELTA_NEEDLE, FUNCTION_CALL_DELTA_REPLACEMENT, 1)
    if FUNCTION_CALL_DONE_PATCH_MARKER not in patched:
        if FUNCTION_CALL_DONE_NEEDLE not in patched:
            raise RuntimeError("Could not find Responses function_call done conversion site.")
        patched = patched.replace(FUNCTION_CALL_DONE_NEEDLE, FUNCTION_CALL_DONE_REPLACEMENT, 1)
    return patched


def main() -> None:
    if not _enabled():
        print("AlphaRavis experimental Responses tool-streaming patch disabled.")
        return

    import langchain_openai.chat_models.base as base

    path = Path(base.__file__)
    text = path.read_text(encoding="utf-8")
    patched = patch_source(text)
    if patched == text:
        print(f"AlphaRavis experimental Responses tool-streaming patch already present: {path}")
        return
    path.write_text(patched, encoding="utf-8")
    print(f"Applied AlphaRavis experimental Responses tool-streaming patch: {path}")


if __name__ == "__main__":
    main()
