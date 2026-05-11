#!/usr/bin/env python
"""Probe AlphaRavis Responses streaming with tool-capable calls.

This script is intentionally diagnostic. It does not patch LangChain or change
runtime defaults. It records raw provider events and LangChain chunks so the
failure point can be classified before any full-streaming patch is attempted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator

try:
    import httpx
except Exception as exc:  # pragma: no cover - exercised in deployed envs
    httpx = None  # type: ignore[assignment]
    HTTPX_IMPORT_ERROR = exc
else:
    HTTPX_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts/alpharavis/responses_streaming_probe"
DEFAULT_PROMPT = (
    "Call marker_tool exactly once with value alpha_probe. After the tool "
    "returns, answer with the exact tool result and no extra explanation."
)


@dataclass
class JsonlWriter:
    path: Path
    count: int = 0

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self.count += 1
        self._fh.write(json.dumps(to_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


@dataclass
class ProbeResult:
    name: str
    ok: bool = False
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def default_base_url() -> str:
    return os.getenv(
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_API_BASE",
        os.getenv("ALPHARAVIS_RESPONSES_API_BASE", os.getenv("OPENAI_API_BASE", "http://litellm:4000/v1")),
    ).rstrip("/")


def default_model() -> str:
    model = os.getenv(
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_MODEL",
        os.getenv("ALPHARAVIS_RESPONSES_MODEL", os.getenv("ALPHARAVIS_MODEL", "openai/big-boss")),
    ).strip()
    return model.removeprefix("openai/")


def default_api_key() -> str:
    return os.getenv(
        "ALPHARAVIS_DEEPAGENTS_RESPONSES_API_KEY",
        os.getenv("ALPHARAVIS_RESPONSES_API_KEY", os.getenv("OPENAI_API_KEY", "sk-local-dev")),
    )


def to_jsonable(value: Any, *, max_string: int = 12000) -> Any:
    """Convert arbitrary SDK/LangChain objects into bounded JSON values."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > max_string:
            return value[:max_string] + f"...[truncated {len(value) - max_string} chars]"
        return value
    if isinstance(value, bytes):
        return to_jsonable(value.decode("utf-8", errors="replace"), max_string=max_string)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item, max_string=max_string) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item, max_string=max_string) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return to_jsonable(model_dump(), max_string=max_string)
        except Exception:
            pass

    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        try:
            return to_jsonable(to_dict(), max_string=max_string)
        except Exception:
            pass

    asdict = getattr(value, "__dict__", None)
    if isinstance(asdict, dict) and asdict:
        return {
            "type": type(value).__name__,
            "value": to_jsonable(asdict, max_string=max_string),
        }

    return {"type": type(value).__name__, "repr": to_jsonable(repr(value), max_string=max_string)}


def message_chunk_snapshot(chunk: Any) -> dict[str, Any]:
    return {
        "type": type(chunk).__name__,
        "content": to_jsonable(getattr(chunk, "content", "")),
        "tool_call_chunks": to_jsonable(getattr(chunk, "tool_call_chunks", None)),
        "tool_calls": to_jsonable(getattr(chunk, "tool_calls", None)),
        "invalid_tool_calls": to_jsonable(getattr(chunk, "invalid_tool_calls", None)),
        "response_metadata": to_jsonable(getattr(chunk, "response_metadata", None)),
        "additional_kwargs": to_jsonable(getattr(chunk, "additional_kwargs", None)),
        "usage_metadata": to_jsonable(getattr(chunk, "usage_metadata", None)),
    }


def content_text_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if not isinstance(content, list):
        return 0
    total = 0
    for item in content:
        if isinstance(item, str):
            total += len(item)
        elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
            text = item.get("text")
            if isinstance(text, str):
                total += len(text)
    return total


def marker_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "marker_tool",
        "description": "Return a marker value.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "Marker value to echo with a TOOL_RETURN_ prefix.",
                }
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    }


def low_level_payload(model: str, prompt: str, *, force_tool_choice: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "stream": True,
        "store": False,
        "parallel_tool_calls": False,
        "max_output_tokens": 256,
        "tools": [marker_tool_schema()],
    }
    if force_tool_choice:
        payload["tool_choice"] = {"type": "function", "name": "marker_tool"}
    return payload


def _sse_flush(
    *,
    event_name: str,
    data_lines: list[str],
    writer: JsonlWriter,
    sequence: int,
) -> dict[str, Any]:
    raw_data = "\n".join(data_lines)
    parsed: Any = None
    if raw_data and raw_data != "[DONE]":
        try:
            parsed = json.loads(raw_data)
        except Exception:
            parsed = raw_data
    record = {
        "ts": utc_now(),
        "sequence": sequence,
        "event": event_name or (parsed.get("type") if isinstance(parsed, dict) else "message"),
        "data": parsed if parsed is not None else raw_data,
        "raw_data": raw_data,
    }
    writer.write(record)
    return record


async def iter_sse_response(
    response: Any,
    writer: JsonlWriter,
) -> AsyncIterator[dict[str, Any]]:
    event_name = ""
    data_lines: list[str] = []
    sequence = 0
    async for line in response.aiter_lines():
        if not line:
            if data_lines:
                record = _sse_flush(
                    event_name=event_name,
                    data_lines=data_lines,
                    writer=writer,
                    sequence=sequence,
                )
                sequence += 1
                yield record
            event_name = ""
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())

    if data_lines:
        yield _sse_flush(event_name=event_name, data_lines=data_lines, writer=writer, sequence=sequence)


def _event_type(record: dict[str, Any]) -> str:
    data = record.get("data")
    if isinstance(data, dict) and isinstance(data.get("type"), str):
        return data["type"]
    if isinstance(record.get("event"), str):
        return record["event"]
    return ""


async def probe_low_level_responses(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    timeout: float,
    writer: JsonlWriter,
    force_tool_choice: bool = False,
) -> ProbeResult:
    if httpx is None:
        return ProbeResult(
            "low_level_responses_sse",
            ok=False,
            error=f"httpx import failed: {HTTPX_IMPORT_ERROR}",
        )

    payload = low_level_payload(model, prompt, force_tool_choice=force_tool_choice)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    result = ProbeResult("low_level_responses_sse")
    event_types: Counter[str] = Counter()
    text_delta_chars = 0
    function_arg_delta_chars = 0
    function_call_items = 0
    errors: list[Any] = []
    started = time.perf_counter()
    writer.write(
        {
            "ts": utc_now(),
            "event": "probe.request",
            "url": f"{base_url}/responses",
            "payload": payload,
        }
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{base_url}/responses", headers=headers, json=payload) as response:
                writer.write(
                    {
                        "ts": utc_now(),
                        "event": "probe.http_response",
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                    }
                )
                if response.status_code >= 400:
                    body = await response.aread()
                    error_text = body.decode("utf-8", errors="replace")
                    writer.write(
                        {
                            "ts": utc_now(),
                            "event": "probe.http_error_body",
                            "status_code": response.status_code,
                            "body": error_text,
                        }
                    )
                    result.error = f"HTTP {response.status_code}: {error_text[:500]}"
                    result.details = {"status_code": response.status_code}
                    return result

                async for record in iter_sse_response(response, writer):
                    event_type = _event_type(record)
                    event_types[event_type] += 1
                    data = record.get("data")
                    if isinstance(data, dict):
                        if isinstance(data.get("delta"), str):
                            if event_type == "response.output_text.delta":
                                text_delta_chars += len(data["delta"])
                            elif event_type == "response.function_call_arguments.delta":
                                function_arg_delta_chars += len(data["delta"])
                        item = data.get("item")
                        if isinstance(item, dict) and item.get("type") == "function_call":
                            function_call_items += 1
                        if "error" in data and data["error"]:
                            errors.append(data["error"])
    except Exception as exc:
        tb = traceback.format_exc()
        writer.write(
            {
                "ts": utc_now(),
                "event": "probe.exception",
                "error": repr(exc),
                "traceback": tb,
            }
        )
        result.error = f"{type(exc).__name__}: {exc}"
        result.details = {"traceback": tb}
        return result

    tool_call_observed = function_call_items > 0 or function_arg_delta_chars > 0
    result.ok = not errors
    if errors:
        result.error = str(errors[0])
    result.details = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "event_types": dict(event_types),
        "text_delta_chars": text_delta_chars,
        "function_arg_delta_chars": function_arg_delta_chars,
        "function_call_items": function_call_items,
        "tool_call_observed": tool_call_observed,
        "errors": errors,
    }
    return result


def make_chat_openai(*, base_url: str, model: str, api_key: str, timeout: float) -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        streaming=True,
        disable_streaming=False,
        use_responses_api=True,
        max_retries=0,
        timeout=timeout,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


async def probe_langchain_no_tool_stream(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    writer: JsonlWriter,
) -> ProbeResult:
    result = ProbeResult("langchain_no_tool_astream")
    chunk_count = 0
    text_chars = 0
    started = time.perf_counter()
    try:
        llm = make_chat_openai(base_url=base_url, model=model, api_key=api_key, timeout=timeout)
        async for chunk in llm.astream("Reply exactly: NO_TOOL_STREAM_OK"):
            snapshot = message_chunk_snapshot(chunk)
            chunk_count += 1
            text_chars += content_text_chars(snapshot.get("content"))
            writer.write(
                {
                    "ts": utc_now(),
                    "event": "langchain.no_tool.chunk",
                    "sequence": chunk_count - 1,
                    "chunk": snapshot,
                }
            )
    except Exception as exc:
        tb = traceback.format_exc()
        writer.write(
            {
                "ts": utc_now(),
                "event": "langchain.no_tool.exception",
                "error": repr(exc),
                "traceback": tb,
            }
        )
        result.error = f"{type(exc).__name__}: {exc}"
        result.details = {"traceback": tb}
        return result

    result.ok = chunk_count > 0 and text_chars > 0
    if not result.ok:
        result.error = f"chunks={chunk_count}, text_chars={text_chars}"
    result.details = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "chunk_count": chunk_count,
        "text_chars": text_chars,
    }
    return result


async def probe_langchain_react_agent(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    timeout: float,
    writer: JsonlWriter,
) -> ProbeResult:
    result = ProbeResult("langchain_react_agent_astream_events")
    event_types: Counter[str] = Counter()
    chunk_count = 0
    text_chunks = 0
    tool_call_chunks = 0
    tool_calls = 0
    invalid_tool_calls = 0
    tool_starts = 0
    tool_ends = 0
    started = time.perf_counter()

    try:
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent

        @tool
        def marker_tool(value: str) -> str:
            """Return a marker value."""

            return "TOOL_RETURN_" + value

        llm = make_chat_openai(base_url=base_url, model=model, api_key=api_key, timeout=timeout)
        agent = create_react_agent(llm, [marker_tool])
        inputs = {"messages": [HumanMessage(content=prompt)]}

        async for event in agent.astream_events(inputs, version="v2"):
            event_name = str(event.get("event") or "")
            event_types[event_name] += 1
            if event_name == "on_tool_start" and event.get("name") == "marker_tool":
                tool_starts += 1
            if event_name == "on_tool_end" and event.get("name") == "marker_tool":
                tool_ends += 1
            record: dict[str, Any] = {
                "ts": utc_now(),
                "event": event_name,
                "name": event.get("name"),
                "run_id": event.get("run_id"),
                "parent_ids": event.get("parent_ids"),
                "tags": event.get("tags"),
                "metadata": event.get("metadata"),
            }

            data = event.get("data")
            if isinstance(data, dict) and "chunk" in data:
                chunk = data["chunk"]
                snapshot = message_chunk_snapshot(chunk)
                record["chunk"] = snapshot
                chunk_count += 1
                if content_text_chars(snapshot.get("content")):
                    text_chunks += 1
                if snapshot.get("tool_call_chunks"):
                    tool_call_chunks += len(snapshot["tool_call_chunks"])
                if snapshot.get("tool_calls"):
                    tool_calls += len(snapshot["tool_calls"])
                if snapshot.get("invalid_tool_calls"):
                    invalid_tool_calls += len(snapshot["invalid_tool_calls"])
            else:
                record["data"] = to_jsonable(data)
            writer.write(record)
    except Exception as exc:
        tb = traceback.format_exc()
        writer.write(
            {
                "ts": utc_now(),
                "event": "langchain.react_agent.exception",
                "error": repr(exc),
                "traceback": tb,
            }
        )
        result.error = f"{type(exc).__name__}: {exc}"
        result.details = {"traceback": tb}
        return result

    result.ok = invalid_tool_calls == 0 and tool_ends == 1
    if invalid_tool_calls:
        result.error = f"invalid_tool_calls={invalid_tool_calls}"
    elif tool_ends != 1:
        result.error = f"marker_tool executions={tool_ends}, expected=1"
    result.details = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "event_types": dict(event_types),
        "chunk_count": chunk_count,
        "text_chunks": text_chunks,
        "tool_call_chunks": tool_call_chunks,
        "tool_calls": tool_calls,
        "invalid_tool_calls": invalid_tool_calls,
        "marker_tool_starts": tool_starts,
        "marker_tool_ends": tool_ends,
    }
    return result


def classify_probe_results(results: list[ProbeResult]) -> dict[str, Any]:
    by_name = {result.name: result for result in results}
    low = by_name.get("low_level_responses_sse")
    no_tool = by_name.get("langchain_no_tool_astream")
    agent = by_name.get("langchain_react_agent_astream_events")

    evidence: list[str] = []
    bucket = "not_reproduced"
    recommendation = "Full streaming probe did not reproduce a failure. Review artifacts before changing defaults."
    agent_error = (agent.error if agent else "").replace("\\'", "'")

    if low and not low.ok:
        bucket = "provider_litellm_or_openai_sdk"
        evidence.append(f"low_level_responses_sse failed: {low.error}")
        recommendation = "Keep hybrid mode. Fix or bypass provider/LiteLLM Responses stream before patching LangChain."
    elif agent and "item['content'] is empty" in agent_error:
        bucket = "langchain_openai_conversion"
        evidence.append(agent.error)
        recommendation = "Inspect langchain-openai Responses chunk conversion before considering an AlphaRavis patch."
    elif agent and "invalid_tool_calls" in agent.error:
        bucket = "ai_message_chunk_aggregation"
        evidence.append(agent.error)
        recommendation = "Buffer or suppress partial tool-call parsing only behind an experimental env gate."
    elif no_tool and not no_tool.ok:
        bucket = "langchain_no_tool_streaming"
        evidence.append(f"langchain_no_tool_astream failed: {no_tool.error}")
        recommendation = "Fix no-tool Responses streaming first; tool streaming is not the first failure."
    elif agent and not agent.ok:
        bucket = "langgraph_or_deepagents_agent_loop"
        evidence.append(f"langchain_react_agent_astream_events failed: {agent.error}")
        recommendation = "Avoid global LangChain patching. Prefer two-phase final-answer streaming or an agent wrapper."

    if low and low.ok:
        details = low.details
        if details.get("function_call_items") or details.get("function_arg_delta_chars"):
            evidence.append("raw Responses stream included function-call events")
        else:
            evidence.append("raw Responses stream completed without function-call events")
        if details.get("errors"):
            evidence.append(f"raw Responses errors: {details['errors']}")
    if agent and agent.ok and agent.details.get("tool_call_chunks"):
        evidence.append("LangChain emitted tool_call_chunks without invalid_tool_calls")

    return {
        "bucket": bucket,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=default_base_url(), help="OpenAI-compatible base URL.")
    parser.add_argument("--model", default=default_model(), help="Model name for Responses calls.")
    parser.add_argument("--api-key", default=default_api_key(), help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for tool-call probes.")
    parser.add_argument("--timeout", type=float, default=float(os.getenv("ALPHARAVIS_LLM_TIMEOUT_SECONDS", "120")))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT), help="Directory for JSONL artifacts.")
    parser.add_argument("--run-id", default="", help="Optional stable run id for artifact filenames.")
    parser.add_argument("--skip-low-level", action="store_true", help="Skip direct /v1/responses SSE probe.")
    parser.add_argument("--skip-langchain", action="store_true", help="Skip LangChain probes.")
    parser.add_argument("--force-tool-choice", action="store_true", help="Force marker_tool in low-level payload.")
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(args.artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    writers: list[JsonlWriter] = []
    results: list[ProbeResult] = []
    config = {
        "run_id": run_id,
        "base_url": args.base_url,
        "model": args.model,
        "prompt": args.prompt,
        "timeout": args.timeout,
        "artifact_dir": str(run_dir),
        "streaming_env": {
            "ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING": os.getenv("ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING"),
            "ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING": os.getenv(
                "ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING"
            ),
        },
    }

    started = time.perf_counter()
    try:
        if not args.skip_low_level:
            writer = JsonlWriter(run_dir / "low_level_responses_sse.jsonl")
            writers.append(writer)
            results.append(
                await probe_low_level_responses(
                    base_url=args.base_url,
                    model=args.model,
                    api_key=args.api_key,
                    prompt=args.prompt,
                    timeout=args.timeout,
                    writer=writer,
                    force_tool_choice=args.force_tool_choice,
                )
            )

        if not args.skip_langchain:
            writer = JsonlWriter(run_dir / "langchain_no_tool_astream.jsonl")
            writers.append(writer)
            results.append(
                await probe_langchain_no_tool_stream(
                    base_url=args.base_url,
                    model=args.model,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    writer=writer,
                )
            )

            writer = JsonlWriter(run_dir / "langchain_react_agent_astream_events.jsonl")
            writers.append(writer)
            results.append(
                await probe_langchain_react_agent(
                    base_url=args.base_url,
                    model=args.model,
                    api_key=args.api_key,
                    prompt=args.prompt,
                    timeout=args.timeout,
                    writer=writer,
                )
            )
    finally:
        for writer in writers:
            writer.close()

    summary = {
        "config": config,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": [result.__dict__ for result in results],
        "classification": classify_probe_results(results),
    }
    write_summary(run_dir / "summary.json", summary)
    print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(result.ok for result in results) else 2


def main() -> None:
    try:
        raise SystemExit(asyncio.run(async_main()))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
