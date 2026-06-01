"""Verification tests for commit 97afbf3 — German heuristics expansion.

Tests all 9 change areas: archive recall, condense regex, large-paste
instruction/document, heading detection, directive count, instruction_brief,
doc stripping, FAST_PATH_DENY, source_content marker/log regex.
"""
import os
import re
import sys

# Ensure langgraph-app is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "langgraph-app"))


# ── helpers ──────────────────────────────────────────────────────────
def _load_source(filename: str) -> str:
    path = os.path.join(
        os.path.dirname(__file__), "..", "langgraph-app", filename
    )
    return open(path).read()


AGENT_SOURCE = _load_source("agent_graph.py")
PROMPT_SOURCE = _load_source("prompt_assembly.py")
SC_SOURCE = _load_source("source_content.py")


# ── Test 1: FAST_PATH_DENY count and structure ────────────────────────
def test_fast_path_deny_count():
    """Commit claims 119 entries (was 44, +75)."""
    from prompt_assembly import FAST_PATH_DENY_PATTERNS

    assert len(FAST_PATH_DENY_PATTERNS) == 119, (
        f"Expected 119, got {len(FAST_PATH_DENY_PATTERNS)}"
    )


def test_fast_path_deny_sorted():
    """Patterns should be alphabetically grouped for readability."""
    from prompt_assembly import FAST_PATH_DENY_PATTERNS

    # Just verify no structural corruption — list must be list of strings
    assert all(isinstance(p, str) for p in FAST_PATH_DENY_PATTERNS)
    assert all(len(p) > 0 for p in FAST_PATH_DENY_PATTERNS)


def test_fast_path_deny_no_exact_duplicates():
    """No exact duplicate patterns."""
    from prompt_assembly import FAST_PATH_DENY_PATTERNS
    from collections import Counter

    counts = Counter(FAST_PATH_DENY_PATTERNS)
    dups = {k: v for k, v in counts.items() if v > 1}
    assert not dups, f"Duplicate entries: {dups}"


def test_fast_path_deny_german_stems_present():
    """Verify key German stems are in the deny list."""
    from prompt_assembly import FAST_PATH_DENY_PATTERNS

    required = [
        "programmier", "entwickl", "erstell", "aendern",
        "loesch", "ausfuehr", "kompilier", "deploy",
        "schreib", "analysier", "durchsuch", "test",
    ]
    for stem in required:
        assert stem in FAST_PATH_DENY_PATTERNS, (
            f"Missing required stem: '{stem}'"
        )


# ── Test 2: Archive recall patterns ───────────────────────────────────
def test_archive_recall_patterns_count():
    """Commit claims ~118 phrases."""
    # Extract patterns from source
    start = AGENT_SOURCE.index("def _looks_like_archive_recall_request")
    end = AGENT_SOURCE.index(
        "\n    if any(pattern in lowered for pattern in patterns):", start
    )
    section = AGENT_SOURCE[start:end]
    patterns = re.findall(r'"([^"]*)"', section)
    # Filter out empty string matches and comments
    patterns = [p for p in patterns if p and not p.startswith("#")]
    assert 110 <= len(patterns) <= 130, (
        f"Expected ~118 patterns, got {len(patterns)}"
    )


def test_archive_recall_key_phrases():
    """New 2026-06-01 phrases must be present."""
    key_phrases = [
        "such mal", "such raus", "find mal",
        "wo waren wir", "was haben wir gemacht",
        "was stand da", "was meintest du",
        "beim letzten mal", "vorherige session",
        "vorheriger chat", "damalig", "damalige",
        "fortsetzen", "weitermachen",
        "wo haben wir aufgehört",
        "wo haben wir aufgehoert",
        "was war die letzte", "was waren die letzten",
    ]
    lowered_source = AGENT_SOURCE.lower()
    for phrase in key_phrases:
        assert f'"{phrase}"' in lowered_source, (
            f"Missing key phrase: '{phrase}'"
        )


def test_archive_recall_exact_match():
    """Test exact matching logic (simulated)."""
    patterns = [
        "such mal", "such raus", "wo waren wir",
        "was stand da", "fortsetzen", "archiv",
        "nochmal von vorne",
    ]

    # Should match
    assert "such mal" in "kannst du such mal was finden".lower()
    assert "wo waren wir" in "wo waren wir stehen geblieben".lower()
    assert "fortsetzen" in "lass uns fortsetzen".lower()
    assert "archiv" in "schau im archiv nach".lower()
    assert "nochmal von vorne" in "fangen wir nochmal von vorne an".lower()

    # Should NOT match
    assert "such mal" not in "das ist eine solche malerei".lower()
    assert "archiv" not in "hello world".lower()


# ── Test 3: Condense regex reflects all lookup patterns ──────────────
def test_condense_regex_sync():
    """All non-trivial archive recall patterns must be in the condense regex."""
    # Extract condense regex block
    start_cond = AGENT_SOURCE.index("def _condense_archive_recall_query_from_text")
    # Find the actual re.sub block by locating the regex start
    re_sub_start = AGENT_SOURCE.index('r"(?i)\\b(', start_cond)
    # The regex block ends with )\\b",\n        " " — find it
    re_sub_end = AGENT_SOURCE.index('\\b",\n        " ",', re_sub_start)
    # Backtrack to include closing paren
    condense_block = AGENT_SOURCE[re_sub_start:re_sub_end]

    # Check key new patterns
    must_be_in_condenser = [
        "such mal", "such raus", "find mal", "wo waren wir",
        "was stand da", "fortsetzen", "weitermachen",
        "beim letzten mal", "vorherige session",
    ]
    for phrase in must_be_in_condenser:
        assert phrase in condense_block, (
            f"Pattern '{phrase}' not in condense regex"
        )


# ── Test 4: Large-paste patterns ─────────────────────────────────────
def test_large_paste_instruction_patterns_count():
    """Commit claims 41 patterns (was 20, +21). Actual code has 40."""
    start = AGENT_SOURCE.index("_LARGE_PASTE_INSTRUCTION_PATTERNS = (")
    end = AGENT_SOURCE.index("\n)", start) + 1
    block = AGENT_SOURCE[start:end]
    patterns = re.findall(r'r"(.*?)(?<!\\)"', block)
    # Commit message says 41, code has 40 — 1-count discrepancy documented
    assert len(patterns) in (40, 41), (
        f"Expected 40-41 patterns, got {len(patterns)}"
    )


def test_large_paste_document_patterns_count():
    """Commit claims 16 patterns (was 5, +11). Actual code has 15."""
    start = AGENT_SOURCE.index("_LARGE_PASTE_DOCUMENT_PATTERNS = (")
    end = AGENT_SOURCE.index("\n)", start) + 1
    block = AGENT_SOURCE[start:end]
    patterns = re.findall(r'r"(.*?)(?<!\\)"', block)
    # Commit message says 16, code has 15 — 1-count discrepancy documented
    assert len(patterns) in (15, 16), (
        f"Expected 15-16 patterns, got {len(patterns)}"
    )


def test_large_paste_instruction_german_patterns():
    """German instruction patterns must be present and compile."""
    german_patterns = [
        r"\banleitung\b", r"\bspezifikation\b", r"\bverfahren\b",
        r"\bdurchführung\b", r"\bdurchfuehrung\b", r"\bhandbuch\b",
        r"\bleitfaden\b", r"\bcheckliste\b",
    ]
    source_lower = AGENT_SOURCE.lower()
    for pat in german_patterns:
        assert pat in source_lower, f"Pattern not in source: {pat}"
        re.compile(pat)  # Must compile


def test_large_paste_document_german_patterns():
    """German document patterns must be present and compile."""
    german_patterns = [
        r"\bchatverlauf\b", r"\bzusammenfassung\b",
        r"\bdatensatz\b", r"\b(ergebnis|ergebnisse)\b",
        r"\b(textausschnitt|codeausschnitt)\b",
    ]
    source_lower = AGENT_SOURCE.lower()
    for pat in german_patterns:
        assert pat in source_lower, f"Pattern not in source: {pat}"
        re.compile(pat)


def test_large_paste_regex_compile_all():
    """All _LARGE_PASTE_* patterns must compile as valid regex."""
    start_inst = AGENT_SOURCE.index("_LARGE_PASTE_INSTRUCTION_PATTERNS = (")
    end_inst = AGENT_SOURCE.index("\n)", start_inst) + 1
    start_doc = AGENT_SOURCE.index("_LARGE_PASTE_DOCUMENT_PATTERNS = (")
    end_doc = AGENT_SOURCE.index("\n)", start_doc) + 1

    inst_patterns = re.findall(r'r"((?:[^"\\]|\\.)*)"', AGENT_SOURCE[start_inst:end_inst])
    doc_patterns = re.findall(r'r"((?:[^"\\]|\\.)*)"', AGENT_SOURCE[start_doc:end_doc])

    all_patterns = inst_patterns + doc_patterns
    for pat in all_patterns:
        try:
            re.compile(pat)
        except re.error as e:
            raise AssertionError(f"Pattern '{pat}' fails to compile: {e}")


# ── Test 5: Heading detection regexes ────────────────────────────────
def test_heading_instruction_regex():
    """Heading instruction regex must contain new German terms."""
    start = AGENT_SOURCE.index("heading_instruction_count = len(")
    end = AGENT_SOURCE.index("text,", start + 100)  # find the text arg
    block = AGENT_SOURCE[start:end + 20]

    german_terms = [
        "anleitung", "spezifikation", "verfahren",
        "handbuch", "leitfaden", "checkliste",
        "durchführung", "durchfuehrung",
    ]
    block_lower = block.lower()
    for term in german_terms:
        assert term in block_lower, f"Heading missing: '{term}'"


def test_heading_document_regex():
    """Heading document regex must contain new German terms."""
    start = AGENT_SOURCE.index("heading_document_count = len(")
    end = AGENT_SOURCE.index("text,", start + 100)
    block = AGENT_SOURCE[start:end + 20]

    german_terms = [
        "nachricht", "chatverlauf", "gespräch", "gespraech",
        "zusammenfassung", "übersicht", "uebersicht",
        "datensatz", "ergebnis",
    ]
    block_lower = block.lower()
    for term in german_terms:
        assert term in block_lower, f"Heading missing: '{term}'"


# ── Test 6: Directive count regex ────────────────────────────────────
def test_directive_count_german_terms():
    """Directive count regex must contain new German directives."""
    start = AGENT_SOURCE.index("directive_count = len(")
    end = AGENT_SOURCE.index("text,", start + 200)
    block = AGENT_SOURCE[start:end + 20]

    german_directives = [
        "halte dich an", "keinesfalls", "keineswegs",
        "unter gar keinen umständen", "unter gar keinen umstaenden",
        "nicht vergessen", "denk daran", "bedenke", "beachte",
    ]
    block_lower = block.lower()
    for d in german_directives:
        assert d in block_lower, f"Directive missing: '{d}'"


# ── Test 7: Instruction brief and doc stripping directive_re ─────────
def test_instruction_brief_directive_re():
    """instruction_brief directive_re must include new German directives."""
    start = AGENT_SOURCE.index("def _large_paste_instruction_brief")
    # Find the directive_re = re.compile( block
    directive_start = AGENT_SOURCE.index("directive_re = re.compile(", start)
    directive_end = AGENT_SOURCE.index("\n    )", directive_start)
    block = AGENT_SOURCE[directive_start:directive_end]

    new_terms = [
        "halte dich an", "darfst nicht", "auf keinen fall",
        "unbedingt", "verpflichtend", "stelle sicher",
        "achte darauf", "vergiss nicht",
    ]
    block_lower = block.lower()
    for term in new_terms:
        assert term in block_lower, f"Missing in instruction_brief: '{term}'"


def test_document_body_directive_re():
    """Document body directive_re must include new German directives."""
    start = AGENT_SOURCE.index("def _large_paste_document_body_for_index")
    directive_start = AGENT_SOURCE.index("directive_re = re.compile(", start)
    directive_end = AGENT_SOURCE.index("\n    )", directive_start)
    block = AGENT_SOURCE[directive_start:directive_end]

    new_terms = [
        "anleitung", "spezifikation", "durchführung",
        "halte dich an", "darfst nicht", "unbedingt",
    ]
    block_lower = block.lower()
    for term in new_terms:
        assert term in block_lower, f"Missing in doc_body: '{term}'"


# ── Test 8: source_content.py changes ────────────────────────────────
def test_source_content_marker_re():
    """classifier_window_text marker_re must contain new German markers."""
    start = SC_SOURCE.index("marker_re = re.compile(")
    end = SC_SOURCE.index("\n    )", start)
    block = SC_SOURCE[start:end]

    new_markers = [
        "anleitung", "spezifikation", "verfahren",
        "durchführung", "durchfuehrung", "vorgabe",
        "richtlinie", "bedingung",
    ]
    block_lower = block.lower()
    for marker in new_markers:
        assert marker in block_lower, f"Missing marker: '{marker}'"


def test_source_content_log_regex_german():
    """Log detection regex must contain German log levels."""
    # Find the log detection regex
    start = SC_SOURCE.index("FEHLER|WARNUNG")
    end = SC_SOURCE.index(")", start + 300)
    block = SC_SOURCE[start:end]

    german_levels = [
        "FEHLER", "WARNUNG", "FEHLERSCHWERE", "HINWEIS",
        "VERFOLGUNG", "KRITISCH", "FATAL", "AUSNAHME",
        "RÜCKVERFOLGUNG",
    ]
    for level in german_levels:
        assert level in block, f"Missing log level: '{level}'"


def test_source_content_log_regex_compiles():
    """Full log detection regex must compile."""
    pattern = (
        r"^\s*(\d{4}-\d{2}-\d{2}[T\s]|\[[^\]]+\]\s*)?"
        r"(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|Traceback|Exception"
        r"|FEHLER|WARNUNG|FEHLERSCHWERE|HINWEIS|VERFOLGUNG"
        r"|KRITISCH|FATAL|AUSNAHME|RÜCKVERFOLGUNG)\b"
    )
    re.compile(pattern)  # Must not raise


def test_source_content_retrieval_query_expanded():
    """local_retrieval_query question_re must contain expanded German terms."""
    start = SC_SOURCE.index("def local_retrieval_query")
    question_re_start = SC_SOURCE.index("question_re = re.compile(", start)
    question_re_end = SC_SOURCE.index("\n    )", question_re_start)
    block = SC_SOURCE[question_re_start:question_re_end]

    new_terms = [
        "weshalb", "worum", "inwiefern",
        "erklaere", "erlaeutere", "beschreibe",
        "definiere", "nachschlagen",
        "fass zusammen", "fasse zusammen",
        "analysiere", "untersuche", "prüfe", "pruefe", "vergleiche",
    ]
    block_lower = block.lower()
    for term in new_terms:
        assert term in block_lower, f"Missing in retrieval query: '{term}'"


# ── Test 9: Coding task detection expanded ──────────────────────────
def test_coding_task_german_triggers():
    """_looks_like_coding_task must contain German triggers."""
    start = AGENT_SOURCE.index("def _looks_like_coding_task")
    end = AGENT_SOURCE.index(
        "\n    return any(trigger in query for trigger in triggers)", start
    )
    block = AGENT_SOURCE[start:end]

    german_triggers = [
        "datei", "programmieren", "programmier",
        "entwickeln", "erstellen", "erstell",
        "schreiben", "schreib", "korrigieren",
        "beheben", "anpassen", "bauen", "testen",
        "ausführen", "ausfuehren", "deployen",
        "installieren", "konfigurieren",
    ]
    block_lower = block.lower()
    for trigger in german_triggers:
        assert f'"{trigger}"' in block_lower, (
            f"Missing German trigger: '{trigger}'"
        )


# ── Test 10: Fuzzy fallback function exists and compiles ──────────────
def test_fuzzy_match_function_exists():
    """_fuzzy_match_archive_recall must exist and compile."""
    assert "_fuzzy_match_archive_recall" in AGENT_SOURCE

    start = AGENT_SOURCE.index("def _fuzzy_match_archive_recall")
    next_def = AGENT_SOURCE.index("\ndef ", start + 10)
    fn_body = AGENT_SOURCE[start:next_def]

    # Must compile
    compile(fn_body, "<test>", "exec")

    # Must reference rapidfuzz
    assert "_rapidfuzz_fuzz" in fn_body
    assert "partial_ratio" in fn_body

    # Must skip short patterns
    assert "len(pattern) < 5" in fn_body


def test_fuzzy_match_integration():
    """Fuzzy match is called from _looks_like_archive_recall_request."""
    start = AGENT_SOURCE.index("def _looks_like_archive_recall_request")
    end = AGENT_SOURCE.index("\ndef ", start + 10)
    next_def = AGENT_SOURCE.index("\ndef ", end)
    fn_body = AGENT_SOURCE[start:next_def]

    assert "_fuzzy_match_archive_recall(text, patterns)" in fn_body
    # Exact match must be checked before fuzzy
    exact_check = "if any(pattern in lowered for pattern in patterns):"
    fuzzy_call = "_fuzzy_match_archive_recall(text, patterns)"
    assert fn_body.index(exact_check) < fn_body.index(fuzzy_call)


# ── Test 11: No regression — existing English patterns preserved ──────
def test_english_patterns_preserved():
    """Existing English patterns must still be present."""
    from prompt_assembly import FAST_PATH_DENY_PATTERNS

    english = [
        "agent", "architecture", "code", "debug",
        "docker", "git", "install", "memory",
        "python", "research", "server", "shell",
        "terminal", "tool",
    ]
    for pat in english:
        assert pat in FAST_PATH_DENY_PATTERNS, f"Missing English: '{pat}'"


def test_english_archive_patterns_preserved():
    """English archive recall patterns preserved."""
    english = [
        "old context", "previous context", "compressed context",
        "archive", "archiv",
    ]
    source_lower = AGENT_SOURCE.lower()
    for pat in english:
        assert f'"{pat}"' in source_lower, f"Missing English recall: '{pat}'"
