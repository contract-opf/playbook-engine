"""Tests for the render-prompt reference consumer (issue #179).

The renderer is a pure function of the document: six locked sections in
order, explicit markers for empty sections, deterministic output, and a
byte-for-byte snapshot of the flagship example.

To regenerate the snapshot after an INTENTIONAL renderer/example change:

    UPDATE_RENDER_SNAPSHOT=1 .venv/bin/python -m pytest tests/test_prompt_renderer.py -q

— never regenerate automatically; a diff against the committed snapshot is
exactly the review signal this test exists to produce.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from playbook_engine.prompt_renderer import render_prompt

ROOT = Path(__file__).parent.parent
FLAGSHIP = ROOT / "examples" / "our-paper-baseline.v0.2.playbook.json"
SNAPSHOT = Path(__file__).parent / "snapshots" / "render_prompt_example.md"

_SECTION_HEADERS = [
    "## RED LINES (Floor — hard)",
    "## NEGOTIATION POSTURE (soft)",
    "## EVIDENCE (advisory, cited)",
    "## DRAFTING RULES",
    "## CITATION & CONFIDENCE RULES",
]


def _flagship() -> dict[str, Any]:
    return json.loads(FLAGSHIP.read_text(encoding="utf-8"))


def test_render_matches_snapshot() -> None:
    rendered = render_prompt(_flagship())
    if os.environ.get("UPDATE_RENDER_SNAPSHOT") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(rendered, encoding="utf-8")
    assert SNAPSHOT.exists(), (
        "snapshot missing — generate once with UPDATE_RENDER_SNAPSHOT=1 (see module docstring)"
    )
    assert rendered == SNAPSHOT.read_text(encoding="utf-8"), (
        "rendered prompt differs from the committed snapshot; if the change is "
        "intentional, regenerate with UPDATE_RENDER_SNAPSHOT=1 and review the diff"
    )


def test_six_sections_present_in_order() -> None:
    rendered = render_prompt(_flagship())
    # Section 1 is the role preamble (the document title line).
    assert rendered.startswith("# Contract review playbook:")
    positions = [rendered.index(h) for h in _SECTION_HEADERS]
    assert positions == sorted(positions), "sections out of order"


def test_empty_sections_render_markers() -> None:
    doc = _flagship()
    doc["floor"] = {}
    doc["posture"] = {}
    doc["evidence"] = {"clauses": [], "clause_library": []}
    rendered = render_prompt(doc)
    assert "(this playbook defines no floor invariants)" in rendered
    assert "(this playbook carries no generated posture yet)" in rendered
    assert "(this playbook carries no compiled evidence)" in rendered
    for header in _SECTION_HEADERS:
        assert header in rendered, f"section {header!r} silently disappeared"


def test_deterministic() -> None:
    doc = _flagship()
    assert render_prompt(doc) == render_prompt(json.loads(json.dumps(doc)))


def test_stance_line_uses_stance_detail_when_present() -> None:
    doc = _flagship()
    clause = doc["evidence"]["clauses"][0]
    clause["summary"]["stance_detail"] = {"held": 7, "of": 9, "basis": "our_paper"}
    rendered = render_prompt(doc)
    assert "held 7 of 9 our-paper deals" in rendered


def test_no_network_and_no_entity_resolution() -> None:
    """The renderer must be a pure function: no anthropic import, no
    entity-registry lookups — aliases render exactly as stored."""
    import playbook_engine.prompt_renderer as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "anthropic" not in source
    assert "entity_registry" not in source

    rendered = render_prompt(_flagship())
    assert "Counterparty-1" in rendered  # stored alias, rendered as-is


def test_indefinite_article_agrees_with_agreement_name() -> None:
    """Issue #207: line 1 of the rendered prompt hardcoded "a" regardless of
    the agreement name's first sound — "a Educational Affiliation Agreement"
    was the first thing a user saw in the flagship artifact."""
    doc = _flagship()

    doc["agreement_type"]["name"] = "Educational Affiliation Agreement"
    assert "reviewing an **Educational Affiliation Agreement**" in render_prompt(doc)

    doc["agreement_type"]["name"] = "Master Services Agreement"
    assert "reviewing a **Master Services Agreement**" in render_prompt(doc)
