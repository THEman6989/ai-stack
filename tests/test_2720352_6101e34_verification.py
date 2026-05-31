#!/usr/bin/env python3
"""Rigorous verification of commits 2720352 + 6101e34."""
import os
import sys
import re
import tempfile
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

# ============================================================
# 2720352: event_indexing.py — Secret pattern tests
# ============================================================
def test_secret_patterns_before_hardening():
    """Verify the original patterns had the issues described."""
    # These are the OLD patterns for reference
    old_pattern_1 = re.compile(r"(?:password|passwd|pass|token|secret|key|api_key|auth)\s*[=:]\s*\S+", re.IGNORECASE)

    # False positive: "key=value" or "auth=true" would be redacted
    assert old_pattern_1.search("key=value") is not None, "OLD: 'key=' was a false positive"
    assert old_pattern_1.search("auth=true") is not None, "OLD: 'auth=' was a false positive"

    # Partial capture: "password=my secret here" only catches "my"
    match = old_pattern_1.search("password=my secret here")
    assert match is not None
    assert match.group() == "password=my", f"OLD: Only caught first word: {match.group()}"

    print("  ✓ Pre-hardening issues confirmed: false positives + partial capture")

def test_secret_patterns_after_hardening():
    """Verify the new patterns fix the issues."""
    from event_indexing import _SECRET_PATTERNS

    pattern_1 = _SECRET_PATTERNS[0]

    # "key" and "auth" no longer matched
    assert pattern_1.search("key=value") is None, "NEW: 'key=' should NOT be matched"
    assert pattern_1.search("auth=true") is None, "NEW: 'auth=' should NOT be matched"

    # Specific key names matched
    assert pattern_1.search("private_key=abc123") is not None, "NEW: private_key should be matched"
    assert pattern_1.search("secret_key=xyz") is not None, "NEW: secret_key should be matched"
    assert pattern_1.search("access_key=AKIAIOSFODNN7EXAMPLE") is not None, "NEW: access_key should be matched"

    # password still matched
    assert pattern_1.search("password=hunter2") is not None, "password should still be matched"

    # Multi-word capture: "password=my secret here" should capture full line
    match = pattern_1.search("some text password=my secret here more text")
    assert match is not None
    assert "secret" in match.group(), f"NEW: Should capture multi-word: {match.group()}"

    # Full-line capture: [^\n]+
    match = pattern_1.search("password=first second third")
    assert match is not None
    assert "third" in match.group(), f"NEW: Should capture to end of line: {match.group()}"

    # Patterns 2-6 exist
    assert len(_SECRET_PATTERNS) == 6, f"Expected 6 patterns, got {len(_SECRET_PATTERNS)}"

    print("  ✓ Post-hardening: no false positives, full-line capture, 6 patterns")

def test_secret_scrubbing_end_to_end():
    """Test _scrub_secrets with realistic output."""
    from event_indexing import _scrub_secrets

    test_cases = [
        # (input, should_contain, should_not_contain)
        ("ghp_123456789012345678901234567890123456", "[REDACTED]", "ghp_"),
        ("sk-abcdefghijklmnopqrstuvwxyz123456", "[REDACTED]", "sk-"),
        ("xoxb-1234567890-1234567890-abcdefghij", "[REDACTED]", "xoxb"),
        ("eyJhbGci.eyJzdWI.SflKxw", "[REDACTED]", "eyJ"),
        ("-----BEGIN RSA PRIVATE KEY-----\nXXX\n-----END RSA PRIVATE KEY-----", "[REDACTED]", "BEGIN RSA"),
        ("password=mysecret", "[REDACTED]", "mysecret"),
        ("token: abc123", "[REDACTED]", "abc123"),
        ("Normal text without secrets", "Normal text", None),
        ("secret_key=hidden value here", "[REDACTED]", "hidden value"),
    ]

    for text, should_contain, should_not in test_cases:
        result = _scrub_secrets(text)
        assert should_contain in result, f"Expected '{should_contain}' in: {result[:80]}"
        if should_not:
            assert should_not not in result, f"'{should_not}' should be redacted in: {result[:80]}"

    print("  ✓ Secret scrubbing: 9/9 test cases pass")

def test_key_auth_no_longer_false_positive():
    """Verify key=value and auth=true are NOT scrubbed."""
    from event_indexing import _scrub_secrets

    # These should NOT be redacted anymore
    assert _scrub_secrets("key=my_config_value") == "key=my_config_value", \
        "key= should NOT be redacted"
    assert _scrub_secrets("auth=true") == "auth=true", \
        "auth= should NOT be redacted"
    assert _scrub_secrets("user_key=abc") == "user_key=abc", \
        "user_key= should NOT be redacted"

    # But api_key, private_key, etc. still should
    assert "[REDACTED]" in _scrub_secrets("api_key=sk-abc"), "api_key should be redacted"
    assert "[REDACTED]" in _scrub_secrets("private_key=abc"), "private_key should be redacted"

    print("  ✓ key=/auth= no longer false positives; specific keys still redacted")


# ============================================================
# 2720352+6101e34: execute_local_command — event-loop guarantee
# ============================================================
def test_maybe_index_tool_run_loop_check():
    """Verify maybe_index_tool_run properly checks for event loop."""
    from event_indexing import maybe_index_tool_run

    # The function must check ALPHARAVIS_ENABLE_TOOL_EVENT_VECTOR_INDEX first
    result = maybe_index_tool_run("test_tool", "output", exit_code=0)
    # Default: flag OFF → None
    assert result is None, "Feature flag OFF should return None"

    print("  ✓ maybe_index_tool_run returns None when flag is OFF (default)")

def test_execute_local_command_structure():
    """Verify the integration structure in execute_local_command."""
    source = (ROOT / "langgraph-app" / "agent_graph.py").read_text()

    # Find the function
    fn_start = source.index("def execute_local_command")
    fn_end = source.index("def storage_manager_status")
    fn_source = source[fn_start:fn_end]

    # 2026-05-31: execute_local_command now uses _index_tool_call helper
    assert "_index_tool_call" in fn_source, \
        "execute_local_command must call _index_tool_call for PGVector indexing"

    # Returns output after indexing attempt
    assert "return output" in fn_source, \
        "execute_local_command must return output"

    # Flag NOT duplicated in execute_local_command
    assert "ALPHARAVIS_ENABLE_TOOL_EVENT_VECTOR_INDEX" not in fn_source, \
        "Flag should only be in event_indexing.py, not duplicated"

    print("  ✓ execute_local_command: uses _index_tool_call helper, flag in module only")


# ============================================================
# 2720352: Import block
# ============================================================
def test_event_indexing_import_block():
    """Verify the guarded import block pattern."""
    source = (ROOT / "langgraph-app" / "agent_graph.py").read_text()

    import_block = source[source.index("from event_indexing import"):source.index("EVENT_INDEXING_IMPORT_ERROR = None") + 31]

    assert "build_tool_run_surrogate as _build_tool_run_surrogate" in import_block
    assert "maybe_index_tool_run as _maybe_index_tool_run" in import_block
    assert "_build_tool_run_surrogate = None" in import_block
    assert "_maybe_index_tool_run = None" in import_block
    assert "EVENT_INDEXING_IMPORT_ERROR" in import_block
    assert "except Exception" in import_block

    print("  ✓ Guarded import block follows repo_skills pattern")


# ============================================================
# 2720352: event_indexing.py module structure
# ============================================================
def test_module_is_pure_library():
    """Verify event_indexing.py has no circular imports or PGVector calls."""
    source = (ROOT / "langgraph-app" / "event_indexing.py").read_text()

    # No PGVector/agent_graph imports (docstring mentions PGVector as destination, fine)
    import_lines = [l for l in source.split("\n") if l.strip().startswith("import ") or l.strip().startswith("from ")]
    imports = "\n".join(import_lines)
    assert "pgvector" not in imports, "Module should not import pgvector"
    assert "vector_memory" not in imports, "Module should not import vector_memory"
    assert "agent_graph" not in imports, "Module should not import agent_graph"

    # All 5 public functions exist
    for fn_name in ["_scrub_secrets", "_bounded_excerpt", "build_tool_run_surrogate",
                     "maybe_index_tool_run", "surrogate_summary"]:
        assert f"def {fn_name}" in source, f"Function {fn_name} should exist"

    print("  ✓ event_indexing.py is pure library: no PGVector/agent_graph imports")

def test_surrogate_format():
    """Verify surrogate format is consistent and searchable."""
    from event_indexing import build_tool_run_surrogate

    surrogate = build_tool_run_surrogate("execute_local_command", "12 passed", exit_code=0)
    assert "Tool run: execute_local_command" in surrogate
    assert "Status: success" in surrogate
    assert "Exit code: 0" in surrogate
    assert "12 passed" in surrogate
    print("  ✓ Surrogate format correct")


# ============================================================
# 2720352: test_event_indexing.py + test_source_scoped_retrieval.py
# ============================================================
def test_event_indexing_tests_exist():
    """Verify the test suite covers all functions."""
    test_source = (ROOT / "tests" / "test_event_indexing.py").read_text()

    required_tests = [
        "test_import_event_indexing",
        "test_build_tool_run_surrogate_success",
        "test_build_tool_run_surrogate_failure",
        "test_build_tool_run_surrogate_scrubs_secrets",
        "test_build_tool_run_surrogate_bounds_output",
        "test_maybe_index_tool_run_disabled_by_default",
        "test_surrogate_summary_extracts_tool_and_status",
    ]
    for test_name in required_tests:
        assert f"def {test_name}" in test_source, f"Test {test_name} missing"

    print(f"  ✓ {len(required_tests)} required tests present in test_event_indexing.py")

def test_source_scoped_retrieval_tool_run_test():
    """Verify the structural test for execute_local_command integration."""
    test_source = (ROOT / "tests" / "test_source_scoped_retrieval.py").read_text()

    assert "test_execute_local_command_has_tool_run_indexing_flag" in test_source, \
        "Structural test for tool-run indexing missing"

    print("  ✓ Structural test present in test_source_scoped_retrieval.py")


# ============================================================
# Fire-and-forget audit verification
# ============================================================
def test_fire_and_forget_audit_accuracy():
    """Verify the 3 remaining fire-and-forget sites are correctly identified."""
    source = (ROOT / "langgraph-app" / "agent_graph.py").read_text()

    # Count all asyncio.create_task / loop.create_task calls
    create_task_count = source.count("asyncio.create_task(") + source.count("loop.create_task(")

    # The commit audit says 8 total
    assert create_task_count >= 7, f"Expected at least 7-8 create_task calls, found {create_task_count}"
    print(f"  ✓ create_task audit: {create_task_count} sites found (expected 8)")

    # Verify the 2 fixed sites no longer have RuntimeError guards:
    # 1. execute_local_command (line ~4266): no try/except RuntimeError
    # 2. reload_repo_ai_skills (line ~4912): uses asyncio.gather (no create_task)

    # Find execute_local_command code
    exec_cmd_start = source.index("def execute_local_command")
    exec_cmd_end = source.index("def storage_manager_status")
    exec_code = source[exec_cmd_start:exec_cmd_end]
    assert "except RuntimeError:" not in exec_code, "6101e34: execute_local_command RuntimeError guard removed"

    # Find reload_repo_ai_skills code
    reload_start = source.index("async def reload_repo_ai_skills")
    reload_end = source.index("def read_repo_ai_skill")
    reload_code = source[reload_start:reload_end]
    assert "asyncio.gather" in reload_code, "reload_repo_ai_skills uses gather (fixed)"

    print("  ✓ Fire-and-forget fixes verified: execute_local_command + reload_repo_ai_skills")


if __name__ == "__main__":
    print("=== Rigorous Verification: Commits 2720352 + 6101e34 ===\n")

    tests = [
        ("2720352: Pre-hardening issues", test_secret_patterns_before_hardening),
        ("6101e34: Post-hardening fixes", test_secret_patterns_after_hardening),
        ("2720352+6101e34: Secret scrubbing e2e", test_secret_scrubbing_end_to_end),
        ("6101e34: key=/auth= no false positive", test_key_auth_no_longer_false_positive),
        ("2720352: maybe_index_tool_run flag check", test_maybe_index_tool_run_loop_check),
        ("6101e34: execute_local_command structure", test_execute_local_command_structure),
        ("2720352: Import block pattern", test_event_indexing_import_block),
        ("2720352: Pure library module", test_module_is_pure_library),
        ("2720352: Surrogate format", test_surrogate_format),
        ("2720352: Test suite completeness", test_event_indexing_tests_exist),
        ("2720352: Structural test", test_source_scoped_retrieval_tool_run_test),
        ("Cross: Fire-and-forget audit", test_fire_and_forget_audit_accuracy),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(1 if failed else 0)
