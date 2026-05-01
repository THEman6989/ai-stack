from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

from repo_skills import (  # noqa: E402
    reload_repo_skill_manifest,
    render_skill_draft_from_candidate,
    resolve_skill_file_path,
    scan_repo_skills,
    slugify_skill_name,
)


def _write_skill(root: Path, slug: str, description: str = "Do the useful workflow.") -> Path:
    skill_dir = root / "ai-skills" / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "\n".join(
            [
                "---",
                f"name: {slug}",
                f"description: {description}",
                "---",
                "",
                f"# {slug}",
                "",
                "Use this skill for tests.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return skill_md


def test_slugify_skill_name_is_command_safe() -> None:
    assert slugify_skill_name("Deep Agents + MCP / Builder") == "deep-agents-mcp-builder"


def test_scan_repo_skills_writes_and_reuses_manifest_cache() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        skills_dir = workspace / "ai-skills"
        cache_path = workspace / ".cache" / "alpharavis" / "skills.json"
        _write_skill(workspace, "alpha-test")

        first = scan_repo_skills(skills_dir, workspace_root=workspace, cache_path=cache_path)
        second = scan_repo_skills(skills_dir, workspace_root=workspace, cache_path=cache_path)

    assert first["cache_status"] == "cache_written"
    assert second["cache_status"] == "cache_hit"
    assert second["skills"][0]["slug"] == "alpha-test"


def test_reload_repo_skill_manifest_reports_added_and_changed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        skills_dir = workspace / "ai-skills"
        cache_path = workspace / ".cache" / "alpharavis" / "skills.json"
        skill_md = _write_skill(workspace, "first-skill", "First description.")
        scan_repo_skills(skills_dir, workspace_root=workspace, cache_path=cache_path)

        time.sleep(0.001)
        skill_md.write_text(skill_md.read_text(encoding="utf-8") + "\nChanged.\n", encoding="utf-8")
        _write_skill(workspace, "second-skill", "Second description.")
        result = reload_repo_skill_manifest(skills_dir, workspace_root=workspace, cache_path=cache_path)

    assert [item["slug"] for item in result["added"]] == ["second-skill"]
    assert [item["slug"] for item in result["changed"]] == ["first-skill"]
    assert result["total"] == 2


def test_supporting_files_are_indexed_and_resolved_safely() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        skills_dir = workspace / "ai-skills"
        _write_skill(workspace, "skill-with-files")
        skill_dir = skills_dir / "skill-with-files"
        for rel in ["references/patterns.md", "templates/config.txt", "scripts/run.py", "assets/readme.txt"]:
            target = skill_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("support", encoding="utf-8")

        snapshot = scan_repo_skills(skills_dir, workspace_root=workspace, use_cache=False)
        slug, reference = resolve_skill_file_path(skills_dir, "skill-with-files", "references/patterns.md")
        _slug, skill_md = resolve_skill_file_path(skills_dir, "skill-with-files", "SKILL.md")

    assert slug == "skill-with-files"
    assert reference.name == "patterns.md"
    assert skill_md.name == "SKILL.md"
    assert sorted(snapshot["skills"][0]["supporting_files"]) == [
        "assets/readme.txt",
        "references/patterns.md",
        "scripts/run.py",
        "templates/config.txt",
    ]


def test_draft_render_is_review_only_and_not_active() -> None:
    candidate = {
        "name": "Fix Pixelle Docker Flow",
        "trigger": "Use when Pixelle Docker jobs fail.",
        "steps": "1. Check logs\n2. Verify ComfyUI",
        "success_signals": "Job completes.",
        "safety_notes": "Do not restart services without approval.",
        "confidence": 0.8,
    }

    draft = render_skill_draft_from_candidate(candidate, candidate_key="abc123", approval_note="reviewed")
    frontmatter = draft.split("---", 2)[1]
    parsed = {
        line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
        for line in frontmatter.splitlines()
        if ":" in line
    }

    assert parsed["status"] == "draft"
    assert parsed["review_required"] == "true"
    assert parsed["source_candidate_key"] == "abc123"
    assert "Move out of the draft folder only after human review." in draft


def test_draft_skills_are_hidden_until_requested() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        skills_dir = workspace / "ai-skills"
        draft_dir = skills_dir / "_drafts" / "draft-skill"
        draft_dir.mkdir(parents=True, exist_ok=True)
        draft_dir.joinpath("SKILL.md").write_text(
            "\n".join(
                [
                    "---",
                    "name: draft-skill",
                    "description: Draft only.",
                    "status: draft",
                    "---",
                    "",
                    "# Draft",
                ]
            ),
            encoding="utf-8",
        )

        hidden = scan_repo_skills(skills_dir, workspace_root=workspace, use_cache=False)
        visible = scan_repo_skills(skills_dir, workspace_root=workspace, use_cache=False, include_drafts=True)

    assert hidden["skills"] == []
    assert visible["skills"][0]["slug"] == "draft-skill"


def test_snapshot_is_json_serializable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        skills_dir = workspace / "ai-skills"
        _write_skill(workspace, "json-safe")

        snapshot = scan_repo_skills(skills_dir, workspace_root=workspace, use_cache=False)

    json.dumps(snapshot, ensure_ascii=False)


def _run_all() -> None:
    tests = [
        test_slugify_skill_name_is_command_safe,
        test_scan_repo_skills_writes_and_reuses_manifest_cache,
        test_reload_repo_skill_manifest_reports_added_and_changed,
        test_supporting_files_are_indexed_and_resolved_safely,
        test_draft_render_is_review_only_and_not_active,
        test_draft_skills_are_hidden_until_requested,
        test_snapshot_is_json_serializable,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    _run_all()
