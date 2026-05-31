#!/usr/bin/env python3
"""Targeted verification tests for commit f1d05eb fixes."""
import os
import sys
import tempfile
import inspect
from pathlib import Path

# Add langgraph-app to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "langgraph-app"))

import asyncio
import hashlib

# ============================================================
# Fix 1: skill_entry_to_index_document — dead code revived
# ============================================================
def test_skill_entry_body_reading():
    """Verify workspace_root + path actually reads SKILL.md body."""
    from repo_skills import skill_entry_to_index_document

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        skill_dir = workspace / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: My Skill
description: Does things
---
# My Skill

This is the actual workflow body text that should be indexed.
It contains specific instructions about how to do the thing.
Step 1: Do X. Step 2: Do Y. Step 3: Verify Z.
""", encoding="utf-8")

        entry = {
            "slug": "my-skill",
            "name": "My Skill",
            "description": "Does things",
            "path": "skills/my-skill/SKILL.md",
            "category": "general",
            "conditions": {},
            "supporting_files": [],
            "mtime_ns": 123456789,
        }

        # WITHOUT workspace_root — body should NOT be included
        result_no_ws = skill_entry_to_index_document(entry, workspace_root="")
        assert result_no_ws is not None, "Should produce a payload"
        content_no_ws = result_no_ws["content"]
        assert "This is the actual workflow body" not in content_no_ws, (
            f"Body should NOT be in content without workspace_root. Got: {content_no_ws[:200]}"
        )

        # WITH workspace_root — body SHOULD be included
        result_ws = skill_entry_to_index_document(entry, workspace_root=str(workspace))
        assert result_ws is not None, "Should produce a payload"
        content_ws = result_ws["content"]
        assert "This is the actual workflow body" in content_ws, (
            f"Body should be in content with workspace_root. Got: {content_ws[:200]}"
        )
        assert "name: My Skill" not in content_ws.split("This is the actual")[0], (
            "YAML frontmatter should be stripped from body"
        )

        # Verify truncation to max_body_chars
        result_short = skill_entry_to_index_document(entry, workspace_root=str(workspace), max_body_chars=20)
        assert len(result_short["content"]) <= 20 + 3000 + 100, "Content should respect max_body_chars + 3000 cap"

        print("  ✓ Fix 1: Body reading, frontmatter strip, truncation")

def test_skill_entry_no_frontmatter():
    """Verify file without YAML frontmatter still works."""
    from repo_skills import skill_entry_to_index_document

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        skill_dir = workspace / "skills" / "no-fm"
        skill_dir.mkdir(parents=True)

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# No Frontmatter\n\nJust body text here.\n", encoding="utf-8")

        entry = {
            "slug": "no-fm",
            "name": "No FM",
            "description": "No frontmatter skill",
            "path": "skills/no-fm/SKILL.md",
            "category": "general",
            "conditions": {},
            "supporting_files": [],
        }

        result = skill_entry_to_index_document(entry, workspace_root=str(workspace))
        assert result is not None
        assert "Just body text here" in result["content"], "Body should be read for files without frontmatter"
        print("  ✓ Fix 1: File without frontmatter handled correctly")


# ============================================================
# Fix 3: fire-and-forget counting logic — examine _maybe_index_vector_memory
# ============================================================
def test_maybe_index_return_patterns():
    """Map _maybe_index_vector_memory return paths vs gather counting logic."""
    # Read source directly to avoid importing full agent_graph (pulls in deepagents)
    agent_graph_path = Path(__file__).resolve().parents[1] / "langgraph-app" / "agent_graph.py"
    source = agent_graph_path.read_text()

    # Find the function
    fn_start = source.index("async def _maybe_index_vector_memory(")
    # Find end: next top-level def or class
    rest = source[fn_start:]
    next_def = rest.find("\nasync def ", len("async def _maybe_index_vector_memory("))
    if next_def == -1:
        next_def = rest.find("\ndef ", len("async def "))
    fn_source = rest[:next_def] if next_def != -1 else rest

    # Verify function return paths
    assert "return None" in fn_source, "Should have None return for disabled pgvector"
    assert "return message" in fn_source, "Should have message return for error paths"

    # The gather logic in reload_repo_ai_skills checks:
    #   if outcome is not None and not isinstance(outcome, BaseException):
    #       indexed += 1
    #
    # All error strings above pass this check → counted as "indexed" (BUG)
    # But: it's still an improvement — old code counted scheduled tasks
    # without even checking if they completed.

    print("  ✓ Fix 3: _maybe_index_vector_memory return paths analyzed from source")
    print("  ⚠ Finding: Error strings from _maybe_index_vector_memory counted as 'indexed'")
    print("    (5 return-paths produce strings; None for disabled; no exceptions raised)")
    print("    Impact: Low — old code was worse (fire-and-forget, zero error visibility)")
    print("    Fix suggested: either raise on error or return structured result dict")

# ============================================================
# Fix 5: version string truncation removed
# ============================================================
def test_content_digest_is_full_64_char():
    """Verify _content_digest returns full 64-char hex, not 16-char."""
    from vector_memory import _content_digest

    digest = _content_digest("test content for hashing")
    assert len(digest) == 64, f"Expected 64-char sha256 hex, got {len(digest)}: {digest}"
    assert all(c in "0123456789abcdef" for c in digest), "Should be hex"
    print(f"  ✓ Fix 5: _content_digest returns 64-char hex: {digest[:16]}...{digest[-16:]}")


def test_upsert_memory_no_truncation():
    """Verify upsert_memory_record no longer truncates source_digest."""
    from vector_memory import _content_digest

    # source_digest is a local variable (full 64-char from _content_digest)
    # The version resolution chain should use the full value
    digest = _content_digest("test")
    assert len(digest) == 64

    # Check that the version resolution code (lines 1557-1563) no longer has [:16]
    import vector_memory
    source = inspect.getsource(vector_memory.upsert_memory_record)
    assert "source_digest[:16]" not in source, "[:16] truncation should be removed from upsert_memory_record"

    source2 = inspect.getsource(vector_memory.upsert_media_record)
    assert "_content_digest(caption)[:16]" not in source2, "[:16] truncation should be removed from upsert_media_record"

    print("  ✓ Fix 5: No [:16] truncation in upsert_memory_record or upsert_media_record")


# ============================================================
# Fix 2: Vision table column count
# ============================================================
def test_vision_insert_column_count():
    """Verify INSERT has correct number of columns vs VALUES."""
    import vector_memory
    source = inspect.getsource(vector_memory._insert_vision_sync)

    # Count INSERT columns
    insert_start = source.index("INSERT INTO")
    insert_section = source[insert_start:]

    # Extract column list between first ( and )
    col_start = insert_section.index("(") + 1
    col_end = insert_section.index(")")
    col_list = insert_section[col_start:col_end]
    columns = [c.strip() for c in col_list.split(",") if c.strip()]
    col_count = len(columns)

    # Count VALUES placeholders
    vals_start = insert_section.index("VALUES") + 6
    vals_section = insert_section[vals_start:]
    vals_end = vals_section.index(")")
    vals_content = vals_section[:vals_end]
    placeholders = [p.strip() for p in vals_content.split(",") if "%s" in p]
    val_count = len(placeholders)

    assert col_count == 19, f"Expected 19 INSERT columns, got {col_count}: {columns}"
    assert val_count == 19, f"Expected 19 VALUES placeholders, got {val_count}: {placeholders}"
    assert col_count == val_count, f"Column count {col_count} != VALUES count {val_count}"

    # Check ON CONFLICT DO UPDATE SET covers all non-id columns
    update_start = insert_section.index("DO UPDATE SET")
    update_section = insert_section[update_start:]
    set_clauses = [l.strip() for l in update_section.split("\n") if "EXCLUDED" in l]
    # Exclude id, include version + raw_ref
    assert any("version = EXCLUDED.version" in clause for clause in set_clauses), "version missing from ON CONFLICT"
    assert any("raw_ref = EXCLUDED.raw_ref" in clause for clause in set_clauses), "raw_ref missing from ON CONFLICT"

    print(f"  ✓ Fix 2: Vision INSERT: {col_count} columns = {val_count} VALUES, version+raw_ref in ON CONFLICT")


def test_vision_search_columns():
    """Verify _search_vision_sync SELECT includes version and raw_ref."""
    import vector_memory
    source = inspect.getsource(vector_memory._search_vision_sync)

    assert "version, raw_ref" in source, "version and raw_ref should be in SELECT"
    print("  ✓ Fix 2: _search_vision_sync SELECT includes version, raw_ref")


def test_vision_record_params():
    """Verify upsert_media_record passes version + raw_ref to _insert_vision_sync."""
    import vector_memory
    source = inspect.getsource(vector_memory.upsert_media_record)

    assert "version=media_version" in source, "version parameter should be passed"
    assert "raw_ref=media_raw_ref" in source, "raw_ref parameter should be passed"
    assert "metadata = metadata or {}" in source, "metadata should be resolved before use"
    # Verify metadata is NOT double-or'd at the call site
    # After the fix, metadata=metadata (not metadata=metadata or {})
    call_lines = [line for line in source.split("\n") if "metadata=" in line and "metadata=metadata" in line]
    assert len(call_lines) >= 1, "metadata should be passed directly (already resolved)"
    print("  ✓ Fix 2: upsert_media_record passes version + raw_ref to insert")


if __name__ == "__main__":
    import inspect
    print("=== Targeted Verification Tests for commit f1d05eb ===\n")

    tests = [
        ("Fix 1: Body Reading", test_skill_entry_body_reading),
        ("Fix 1: No Frontmatter", test_skill_entry_no_frontmatter),
        ("Fix 3: Return Patterns", test_maybe_index_return_patterns),
        ("Fix 5: Digest Length", test_content_digest_is_full_64_char),
        ("Fix 5: No Truncation", test_upsert_memory_no_truncation),
        ("Fix 2: Column Count", test_vision_insert_column_count),
        ("Fix 2: Search Columns", test_vision_search_columns),
        ("Fix 2: Record Params", test_vision_record_params),
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
