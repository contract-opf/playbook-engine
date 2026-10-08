"""Tests for the render-prompt reference consumer (issue #179).

The renderer is a pure function of the document: six locked sections in
order, explicit markers for empty sections, deterministic output, and a
byte-for-byte snapshot of the flagship example — the OPF 0.4 NDA reference
playbook (the 0.2 worked example and the 0.1-0.3 stance/observation
rendering were retired with those formats, issue #238).

To regenerate the snapshot after an INTENTIONAL renderer/example change:

    UPDATE_RENDER_SNAPSHOT=1 .venv/bin/python -m pytest tests/test_prompt_renderer.py -q

— never regenerate automatically; a diff against the committed snapshot is
exactly the review signal this test exists to produce.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from playbook_engine.prompt_renderer import (
    _ADVISORY_BANNER,
    _NO_INVARIANTS_MARKER,
    _NO_POSTURE_MARKER,
    is_advisory_only,
    render_prompt,
)

ROOT = Path(__file__).parent.parent
FLAGSHIP = ROOT / "examples" / "nda" / "playbook.opf.json"
SNAPSHOT = Path(__file__).parent / "snapshots" / "render_prompt_example.md"

_SECTION_HEADERS = [
    "## HARD LINES (Floor)",
    "## NEGOTIATION POSTURE (soft)",
    "## EVIDENCE (advisory, cited)",
    "## DRAFTING RULES",
    "## CITATION & CONFIDENCE RULES",
]


def _flagship() -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(FLAGSHIP.read_text(encoding="utf-8"))
    return doc


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
    doc["evidence"] = {"clauses": [], "precedent": []}
    rendered = render_prompt(doc)
    assert _NO_INVARIANTS_MARKER in rendered
    assert _NO_POSTURE_MARKER in rendered
    assert "(this playbook carries no compiled evidence)" in rendered
    for header in _SECTION_HEADERS:
        assert header in rendered, f"section {header!r} silently disappeared"


def test_deterministic() -> None:
    doc = _flagship()
    assert render_prompt(doc) == render_prompt(json.loads(json.dumps(doc)))


def test_no_network_and_no_entity_resolution() -> None:
    """The renderer must be a pure function: no anthropic import, no
    entity-registry lookups — aliases render exactly as stored."""
    import playbook_engine.prompt_renderer as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "anthropic" not in source
    assert "entity_registry" not in source

    doc = _flagship()
    record = next(p for p in doc["evidence"]["precedent"] if p["signed_text"])
    record["signed_text"]["text"] = "Counterparty-1 shall keep the terms confidential."
    record["standard"] = False
    rendered = render_prompt(copy.deepcopy(doc))
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


def test_string_form_floor_invariants_render_without_crashing() -> None:
    """Issue #73: document_renderer tolerates hand-authored bare-string
    invariants (`invariants: ["No indemnity cap below $1M"]`), so
    render-prompt must too — a GC's hand-edited playbook that works in
    `view bundle` must not traceback in `render-prompt` (pre-fix: inv.get()
    raised AttributeError on the plain str)."""
    doc = _flagship()
    doc["floor"] = {"invariants": ["No indemnity cap below $1M"]}
    rendered = render_prompt(doc)
    assert "- No indemnity cap below $1M" in rendered


def test_none_floor_invariant_renders_without_crashing() -> None:
    """The tolerance comment explicitly names JSON null as a shape a
    hand-edited/foreign playbook may carry; a bare None must not crash
    either (pre-fix: None.get() would raise the same AttributeError)."""
    doc = _flagship()
    doc["floor"] = {"invariants": [None]}
    rendered = render_prompt(doc)
    assert "- None" in rendered


# ---------------------------------------------------------------------------
# Evidence: facts only (issue #223) and loud empty states (issue #92).
# ---------------------------------------------------------------------------


def test_evidence_renders_counts_variants_and_refused_asks_not_stances() -> None:
    rendered = render_prompt(_flagship())
    evidence = rendered.split("## EVIDENCE (advisory, cited)", 1)[1].split("## DRAFTING RULES")[0]
    assert "signed our standard language." in evidence
    assert "Non-standard language we have signed:" in evidence
    assert "Asks refused before signing (proposed, then struck):" in evidence
    for jargon in ("historical_stance", "no_signal", "n_our_paper", "THIN PRECEDENT"):
        assert jargon not in rendered, jargon


def test_citation_rules_name_the_single_deal_marker_the_evidence_uses() -> None:
    rendered = render_prompt(_flagship())
    assert "[1 deal]" in rendered
    citation_section = rendered.split("## CITATION & CONFIDENCE RULES", 1)[1]
    assert "`1 deal`" in citation_section


def test_is_advisory_only_true_only_when_both_floor_and_posture_empty() -> None:
    assert is_advisory_only({}) is True
    assert is_advisory_only({"floor": {"invariants": []}, "posture": {}}) is True
    assert (
        is_advisory_only({"floor": {"invariants": [{"id": "x", "statement": "y"}]}, "posture": {}})
        is False
    )
    assert is_advisory_only({"floor": {}, "posture": {"system_prompt": "Be terse."}}) is False
    assert (
        is_advisory_only(
            {
                "floor": {"invariants": [{"id": "x", "statement": "y"}]},
                "posture": {"system_prompt": "Be terse."},
            }
        )
        is False
    )


def test_advisory_only_banner_when_floor_and_posture_both_empty() -> None:
    doc = _flagship()
    doc["floor"] = {"invariants": []}
    doc["posture"]["system_prompt"] = ""
    rendered = render_prompt(doc)
    assert rendered.startswith(_ADVISORY_BANNER)
    # Per-section empty states still render inline — the banner supplements
    # them, it does not replace them.
    assert _NO_POSTURE_MARKER in rendered
    assert _NO_INVARIANTS_MARKER in rendered


def test_advisory_banner_disappears_once_floor_gets_one_invariant_but_posture_marker_stays() -> (
    None
):
    doc = _flagship()
    doc["floor"] = {
        "invariants": [{"id": "inv.1", "statement": "Never accept uncapped liability."}]
    }
    doc["posture"]["system_prompt"] = ""
    rendered = render_prompt(doc)
    assert _ADVISORY_BANNER not in rendered
    assert _NO_POSTURE_MARKER in rendered


def test_no_banner_when_flagship_fully_populated() -> None:
    rendered = render_prompt(_flagship())
    assert _ADVISORY_BANNER not in rendered
