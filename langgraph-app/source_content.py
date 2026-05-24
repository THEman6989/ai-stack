from __future__ import annotations

import json
import os
import re
from typing import Any

# ---------- source content analysis ----------

_SOURCE_STOPWORDS: set[str] = {
    "about",
    "after",
    "also",
    "and",
    "aus",
    "bei",
    "but",
    "das",
    "der",
    "die",
    "dies",
    "diese",
    "ein",
    "eine",
    "for",
    "from",
    "hier",
    "mit",
    "nicht",
    "oder",
    "sich",
    "that",
    "the",
    "und",
    "von",
    "was",
    "wenn",
    "with",
}


def detect_source_content_type(text: str, *, title: str = "", metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata if isinstance(metadata, dict) else {}
    explicit = str(metadata.get("content_type") or metadata.get("source_content_type") or "").strip().lower()
    if explicit in {"code", "log", "config", "prose", "table", "mixed"}:
        return explicit

    pathish = " ".join(
        str(value or "")
        for value in (
            title,
            metadata.get("filename"),
            metadata.get("file_name"),
            metadata.get("path"),
            metadata.get("file_path"),
            metadata.get("source_path"),
            metadata.get("source_key"),
        )
    ).strip().lower()
    if any(pathish.endswith(ext) or ext in pathish for ext in (".json", ".yaml", ".yml", ".toml", ".ini", ".env", ".cfg", ".conf")):
        return "config"
    if any(pathish.endswith(ext) or ext in pathish for ext in (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".sh", ".sql", ".html", ".css")):
        return "code"
    if any(pathish.endswith(ext) for ext in (".log", ".trace", ".out", ".err")):
        return "log"
    if any(pathish.endswith(ext) or ext in pathish for ext in (".csv", ".tsv")):
        return "table"

    sample = str(text or "")[:12000]
    lines = [line for line in sample.splitlines() if line.strip()]
    if not lines:
        return "prose"
    code_lines = sum(
        1
        for line in lines
        if re.search(r"^\s*(async\s+def|def|class|function|import|from|const|let|var|SELECT|CREATE TABLE|if\s+.+:|for\s+.+:)\b", line)
        or re.search(r"[{};]\s*$", line)
    )
    log_lines = sum(
        1
        for line in lines
        if re.search(r"^\s*(\d{4}-\d{2}-\d{2}[T\s]|\[[^\]]+\]\s*)?(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|Traceback|Exception)\b", line)
    )
    config_lines = sum(1 for line in lines if re.search(r"^\s*[\w.-]+\s*[:=]\s*[^=].*$", line))
    table_lines = sum(
        1
        for line in lines
        if ("\t" in line and len(line.split("\t")) >= 3)
        or (line.count("|") >= 2)
        or (line.count(",") >= 3 and len(line) < 500)
    )
    total = max(1, len(lines))
    strong = [
        name
        for name, count, threshold in (
            ("code", code_lines, 0.12),
            ("log", log_lines, 0.12),
            ("config", config_lines, 0.25),
            ("table", table_lines, 0.25),
        )
        if count / total >= threshold and count >= 3
    ]
    if len(strong) > 1:
        return "mixed"
    if strong:
        return strong[0]
    if "```" in sample:
        return "code"
    return "prose"


def extract_source_keywords(text: str, *, limit: int = 12) -> list[str]:
    words = re.findall(r"(?u)\b[\w][\w.-]{3,}\b", str(text or "")[:20000])
    counts: dict[str, int] = {}
    display: dict[str, str] = {}
    for word in words:
        lowered = word.strip("._-").lower()
        if len(lowered) < 4 or lowered in _SOURCE_STOPWORDS or lowered.isdigit():
            continue
        counts[lowered] = counts.get(lowered, 0) + 1
        display.setdefault(lowered, word.strip("._-"))
    ranked = sorted(counts, key=lambda item: (-counts[item], item))
    return [display[item] for item in ranked[:limit]]


def extract_source_entities(text: str, *, limit: int = 12) -> list[str]:
    sample = str(text or "")[:20000]
    candidates = re.findall(r"\b(?:[A-ZÄÖÜ][\wÄÖÜäöüß.-]{2,}|[A-Z0-9_]{3,})(?:[/.-][A-Z0-9_][\w.-]*)*\b", sample)
    seen: set[str] = set()
    entities: list[str] = []
    for candidate in candidates:
        cleaned = candidate.strip(".,:;()[]{}")
        lowered = cleaned.lower()
        if lowered in seen or lowered in _SOURCE_STOPWORDS or cleaned.isdigit():
            continue
        seen.add(lowered)
        entities.append(cleaned)
        if len(entities) >= limit:
            break
    return entities


def extract_source_symbols(text: str, *, limit: int = 20) -> list[str]:
    sample = str(text or "")[:30000]
    patterns = [
        r"(?m)^\s*(?:async\s+def|def|class)\s+([A-Za-z_][\w]*)",
        r"(?m)^\s*(?:function|const|let|var)\s+([A-Za-z_$][\w$]*)",
        r"(?m)^\s*(?:export\s+)?(?:class|interface|type)\s+([A-Za-z_$][\w$]*)",
        r"(?m)^\s*([A-Z][A-Z0-9_]{2,})\s*=",
        r"\b([\w.-]+\.(?:py|js|ts|tsx|jsx|json|ya?ml|toml|log|md|txt|csv|sql))\b",
    ]
    seen: set[str] = set()
    symbols: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, sample):
            symbol = str(match.group(1)).strip()
            key = symbol.lower()
            if key in seen:
                continue
            seen.add(key)
            symbols.append(symbol)
            if len(symbols) >= limit:
                return symbols
    return symbols


def source_title_from_text(text: str, *, fallback: str) -> str:
    for line in str(text or "").splitlines()[:80]:
        stripped = line.strip().strip("#").strip()
        if not stripped or len(stripped) > 140:
            continue
        if re.match(r"(?i)^(document|source|context|data|instructions?|rules?|logs?)\s*[:#-]?$", stripped):
            continue
        return stripped[:120]
    return fallback[:120]


def source_metadata_summary(
    content: str,
    *,
    title: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata if isinstance(metadata, dict) else {}
    content_type = detect_source_content_type(content, title=title, metadata=metadata)
    source_title = str(metadata.get("source_title") or "").strip()
    if not source_title:
        source_title = source_title_from_text(
            content,
            fallback=str(metadata.get("filename") or metadata.get("file_name") or title or "Untitled source"),
        )
    return {
        "content_type": content_type,
        "source_title": source_title,
        "source_keywords": extract_source_keywords(content),
        "source_entities": extract_source_entities(content),
        "source_symbols": extract_source_symbols(content),
    }


# ---------- line range / text parsing utilities ----------


def line_ranges_from_text(text: str) -> tuple[list[str], list[tuple[int, str]]]:
    lines = str(text or "").splitlines()
    numbered = [(index + 1, line) for index, line in enumerate(lines)]
    return lines, numbered


def normalize_line_ranges(value: Any) -> list[list[int]]:
    ranges: list[list[int]] = []
    if not isinstance(value, list):
        return ranges
    for item in value:
        if isinstance(item, list) and len(item) >= 2:
            start, end = item[0], item[1]
        elif isinstance(item, dict):
            start, end = item.get("start"), item.get("end")
        else:
            continue
        try:
            start_i = max(1, int(start))
            end_i = max(start_i, int(end))
        except (TypeError, ValueError):
            continue
        ranges.append([start_i, end_i])
    return ranges[:40]


def line_range_indexes(line_count: int, ranges: Any) -> set[int]:
    indexes: set[int] = set()
    if line_count <= 0:
        return indexes
    for start, end in normalize_line_ranges(ranges):
        start_i = max(1, min(line_count, start))
        end_i = max(start_i, min(line_count, end))
        indexes.update(range(start_i, end_i + 1))
    return indexes


def text_from_line_ranges(text: str, ranges: Any, *, max_chars: int | None = None) -> str:
    lines = str(text or "").splitlines()
    indexes = line_range_indexes(len(lines), ranges)
    if not indexes:
        return ""
    selected = [line for index, line in enumerate(lines, start=1) if index in indexes]
    rendered = "\n".join(selected).strip()
    if max_chars is not None and len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip() + "\n[Line range text truncated.]"
    return rendered


def strip_line_ranges_from_text(text: str, ranges: Any) -> str:
    lines = str(text or "").splitlines()
    indexes = line_range_indexes(len(lines), ranges)
    if not indexes:
        return str(text or "")
    kept = [line for index, line in enumerate(lines, start=1) if index not in indexes]
    return "\n".join(kept).strip()


def parse_classifier_json(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        raise ValueError("empty classifier response")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    elif "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        recovered: dict[str, Any] = {}
        decoder = json.JSONDecoder()
        for key in ("intent", "retrieval_query", "instruction_lines", "document_lines", "question_lines", "confidence", "reason"):
            match = re.search(rf'"{re.escape(key)}"\s*:\s*', text)
            if not match:
                continue
            try:
                value, _ = decoder.raw_decode(text[match.end() :].lstrip())
            except json.JSONDecodeError:
                continue
            recovered[key] = value
        if recovered:
            payload = recovered
        else:
            raise
    if not isinstance(payload, dict):
        raise ValueError("classifier JSON was not an object")
    return payload


def tail_question_line_ranges(text: str) -> list[list[int]]:
    lines = str(text or "").splitlines()
    max_tail = max(1, int(os.getenv("ALPHARAVIS_LARGE_PASTE_QUESTION_TAIL_LINES", "60")))
    question_re = re.compile(r"(?i)(\?|^(was|wie|warum|wann|wo|welche|welcher|welches|wieso|how|what|why|when|where|who|which)\b)")
    candidate: list[list[int]] = []
    tail_start = max(1, len(lines) - max_tail + 1)
    for index in range(tail_start, len(lines) + 1):
        line = lines[index - 1]
        if question_re.search(line):
            candidate.append([index, index])
    return candidate


def bounded_text_window(text: str, *, start: int = 0, max_chars: int = 12000, search: str = "") -> dict[str, Any]:
    raw = str(text or "")
    total_chars = len(raw)
    if total_chars <= max_chars:
        return {"text": raw, "start": 0, "end": total_chars, "total_chars": total_chars}
    if search and search.strip():
        lowered = raw.lower()
        pos = lowered.find(search.strip().lower())
        if pos >= 0:
            window_start = max(0, pos - max_chars // 4)
            window_end = min(total_chars, window_start + max_chars)
            if window_end - window_start > max_chars:
                window_end = min(total_chars, pos + max_chars // 4)
                window_start = max(0, window_end - max_chars)
            return {"text": raw[window_start:window_end], "start": window_start, "end": window_end, "total_chars": total_chars, "search_match": pos}
    start_pos = max(0, min(total_chars - 1, start))
    end_pos = min(total_chars, start_pos + max_chars)
    return {"text": raw[start_pos:end_pos], "start": start_pos, "end": end_pos, "total_chars": total_chars}


# ---------- classifier / retrieval query helpers ----------


def classifier_window_text(text: str) -> str:
    lines, numbered = line_ranges_from_text(text)
    if len(text) <= int(os.getenv("ALPHARAVIS_RAG_CLASSIFIER_FULL_TEXT_MAX_CHARS", "12000")):
        return "\n".join(f"{index}: {line}" for index, line in numbered)

    marker_re = re.compile(
        r"(?i)(^|\b)(/rag|/rake|/index|/ingest|/big-context|/big_context|<big-context|<big_context|task|instructions?|rules?|document|source|context|question|frage|aufgabe|anweisung|quelle)\b"
    )
    selected: dict[int, str] = {}
    head_lines = int(os.getenv("ALPHARAVIS_RAG_CLASSIFIER_HEAD_LINES", "80"))
    tail_lines = int(os.getenv("ALPHARAVIS_RAG_CLASSIFIER_TAIL_LINES", "100"))
    radius = int(os.getenv("ALPHARAVIS_RAG_CLASSIFIER_MARKER_RADIUS_LINES", "8"))
    for index, line in numbered[:head_lines]:
        selected[index] = line
    for index, line in numbered[-tail_lines:]:
        selected[index] = line
    for index, line in numbered:
        if marker_re.search(line):
            start_idx = max(1, index - radius)
            end_idx = min(len(lines), index + radius)
            for nearby in range(start_idx, end_idx + 1):
                selected[nearby] = lines[nearby - 1]
    rendered = "\n".join(f"{index}: {selected[index]}" for index in sorted(selected))
    max_chars = int(os.getenv("ALPHARAVIS_RAG_CLASSIFIER_WINDOW_MAX_CHARS", "24000"))
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip() + "\n[Classifier window truncated.]"
    return rendered


def local_retrieval_query(text: str) -> str:
    raw = str(text or "").strip()
    max_chars = int(os.getenv("ALPHARAVIS_RETRIEVAL_QUERY_MAX_CHARS", "1200"))
    direct_max = int(os.getenv("ALPHARAVIS_RETRIEVAL_QUERY_DIRECT_MAX_CHARS", "600"))
    if len(raw) <= direct_max:
        return raw[:max_chars].strip()

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    question_re = re.compile(
        r"(?i)(\?|^(was|wie|warum|wann|wo|wer|welche|welcher|welches|wieso|how|what|why|when|where|who|which)\b|"
        r"\b(find|search|suche|such|erklär|erklaer|zeige|tell me|look up|nachschauen|nachschau)\b)"
    )
    selected: list[str] = []
    for line in reversed(lines[-120:]):
        if question_re.search(line):
            selected.insert(0, line)
        if sum(len(item) + 1 for item in selected) >= max_chars:
            break
    if not selected:
        selected = lines[-20:]
    query = "\n".join(selected).strip()
    if len(query) > max_chars:
        query = query[-max_chars:].strip()
    return query or raw[:max_chars].strip()
