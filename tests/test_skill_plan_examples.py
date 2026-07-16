"""Verification test for issue #187 — the doc examples in the
playbook-from-corpus skill's "Assemble the corpus story" stage must stay
valid ``staging_plan.json`` shapes.

Extracts every fenced ```json code block in SKILL.md that looks like a
staging plan (parses as JSON and carries the plan's ``deals`` key), and
confirms each one is accepted by ``execute_staging_plan`` without error —
this is the "dry-parse" the issue asks for, using ``execute_staging_plan``'s
default symlink placement (issue #186), which does not require the example's
referenced file paths to exist on disk (a dangling symlink is still a valid
symlink) so the doc's illustrative filenames need no matching fixture files.

Keeps the doc's `staging_plan.json` examples from rotting as intake_plan.py
evolves — the companion doc-drift pattern to
test_skill_playbook_from_corpus.py's CLI-flag-citation check.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from playbook_engine.intake_plan import execute_staging_plan

REPO_ROOT = Path(__file__).parent.parent
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "playbook-from-corpus" / "SKILL.md"

_JSON_FENCE_RE = re.compile(r"```json[ \t]*\n(.*?)\n[ \t]*```", re.DOTALL)


def _staging_plan_examples() -> list[dict]:
    """Return every fenced ```json block in SKILL.md that is a staging plan.

    A block counts as a staging plan if it parses as JSON and has a top-level
    ``deals`` key — this is specific enough not to false-match some other
    JSON example that might be added to the doc later, but doesn't hardcode
    anything about the plan's current content.
    """
    text = SKILL_MD.read_text(encoding="utf-8")
    plans = []
    for block in _JSON_FENCE_RE.findall(text):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError as exc:
            pytest.fail(f"SKILL.md contains a ```json block that is not valid JSON: {exc}\n{block}")
        if isinstance(parsed, dict) and "deals" in parsed:
            plans.append(parsed)
    return plans


def test_skill_md_has_at_least_one_staging_plan_example() -> None:
    """SKILL.md's 'Assemble the corpus story' stage must show a plan shape."""
    assert _staging_plan_examples(), (
        "expected at least one ```json staging_plan.json example in SKILL.md"
    )


@pytest.mark.parametrize(
    "index",
    range(len(_staging_plan_examples())) if SKILL_MD.exists() else [],
)
def test_staging_plan_example_dry_parses(index: int, tmp_path: Path) -> None:
    """Each staging_plan.json example must dry-parse via execute_staging_plan.

    Uses the default symlink placement (not --copy), which only needs the
    *directory structure* to be valid — not the example's illustrative file
    paths to exist — matching how a skill/human would edit a real
    staging_plan.json without fabricating fixture files for a doc example.
    """
    plan = _staging_plan_examples()[index]
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    out_dir = tmp_path / "out"

    result = execute_staging_plan(plan, src_dir, out_dir)

    assert result.layout == "unknown"
    assert result.agreement_count == len(plan["deals"])
    expected_staged = sum(len(deal["files"]) for deal in plan["deals"])
    assert result.staged_count == expected_staged

    for deal in plan["deals"]:
        assert (out_dir / deal["deal_id"] / "hints.yaml").exists()


def test_skill_md_staging_plan_examples_have_evidence_rationale() -> None:
    """At least one example must show a skill-recorded arbitration rationale.

    The issue requires the skill to record a one-line rationale per
    arbitration decision by editing the plan's `evidence` list — this
    catches the doc example silently losing that illustration.
    """
    plans = _staging_plan_examples()
    assert any(
        "skill_rationale" in ev
        for plan in plans
        for deal in plan["deals"]
        for f in deal["files"]
        for ev in f.get("evidence", [])
    ), "expected at least one example to show a 'skill_rationale' entry in a file's evidence list"
