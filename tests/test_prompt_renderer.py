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

from playbook_engine.prompt_renderer import (
    _ADVISORY_BANNER,
    _NO_INVARIANTS_MARKER,
    _NO_POSTURE_MARKER,
    _STANCE_SENTENCES,
    _THIN_TERM,
    _UNKNOWN_STANCE_SENTENCE,
    _UNRECORDED_MOVER,
    _V01_POSITION_SENTENCES,
    is_advisory_only,
    render_prompt,
)

ROOT = Path(__file__).parent.parent
FLAGSHIP = ROOT / "examples" / "our-paper-baseline.v0.2.playbook.json"
SNAPSHOT = Path(__file__).parent / "snapshots" / "render_prompt_example.md"

_SECTION_HEADERS = [
    "## HARD LINES (Floor)",
    "## NEGOTIATION POSTURE (soft)",
    "## EVIDENCE (advisory, cited)",
    "## DRAFTING RULES",
    "## CITATION & CONFIDENCE RULES",
]


def _flagship() -> dict[str, Any]:
    return json.loads(FLAGSHIP.read_text(encoding="utf-8"))


def _position(*, precedent_count: int, provenance: str = "our_paper") -> dict[str, Any]:
    """Minimal ``observed_positions`` entry — only what ``_thin_marker`` reads."""
    return {
        "text_summary": "Synthetic observed position for testing.",
        "precedent_count": precedent_count,
        "provenance": provenance,
        "outcome": "signed",
    }


def _clause(
    *,
    historical_stance: str,
    title: str = "Test Clause",
    stance_detail: dict[str, Any] | None = None,
    n_our_paper: int | None = None,
    n_counterparty_paper: int | None = None,
    evidence_sufficient: bool = True,
    observed_positions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Minimal synthetic v0.2 clause — only the fields render_prompt reads."""
    confidence: dict[str, Any] = {"evidence_sufficient": evidence_sufficient}
    if n_our_paper is not None:
        confidence["n_our_paper"] = n_our_paper
    if n_counterparty_paper is not None:
        confidence["n_counterparty_paper"] = n_counterparty_paper
    summary: dict[str, Any] = {
        "historical_stance": historical_stance,
        "confidence": confidence,
        "acceptable_if": [],
        "fallbacks": [],
        "rejected": [],
    }
    if stance_detail is not None:
        summary["stance_detail"] = stance_detail
    slug = title.lower().replace(" ", "_")
    return {
        "id": f"clause.{slug}",
        "taxonomy_id": slug,
        "title": title,
        "observed_positions": observed_positions if observed_positions is not None else [],
        "summary": summary,
    }


def _v01_clause(
    *,
    position: str,
    title: str = "Test Clause",
    n_our_paper: int | None = None,
    n_counterparty_paper: int | None = None,
    evidence_sufficient: bool = True,
) -> dict[str, Any]:
    """Minimal synthetic OPF v0.1 clause — ``rollup.position``, not
    ``summary.historical_stance`` (issue #92 fix round 2). Mirrors
    ``_clause()`` but uses the v0.1 wrapper key so ``clause_stance()``
    (opf_accessors.py) falls through to the ``rollup.position`` branch
    instead of the v0.2 ``summary`` branch — the exact shape both shipped
    examples/our-paper-baseline.playbook.json and
    examples/emergent-no-template.playbook.json carry.
    """
    confidence: dict[str, Any] = {"score": 0.5, "evidence_sufficient": evidence_sufficient}
    if n_our_paper is not None:
        confidence["n_our_paper"] = n_our_paper
    if n_counterparty_paper is not None:
        confidence["n_counterparty_paper"] = n_counterparty_paper
    slug = title.lower().replace(" ", "_")
    return {
        "id": f"clause.{slug}",
        "taxonomy_id": slug,
        "title": title,
        "observed_positions": [],
        "rollup": {"position": position, "confidence": confidence},
    }


def _doc_with_clause(clause: dict[str, Any]) -> dict[str, Any]:
    """Flagship doc (populated floor/posture) with its clauses replaced by *clause*.

    Isolates one synthetic clause for assertions while keeping floor/posture
    populated, so is_advisory_only() stays False and the banner doesn't
    interfere with clause-level assertions.
    """
    doc = _flagship()
    doc["evidence"]["clauses"] = [clause]
    doc["evidence"]["clause_library"] = []
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
    doc["evidence"] = {"clauses": [], "clause_library": []}
    rendered = render_prompt(doc)
    assert _NO_INVARIANTS_MARKER in rendered
    assert _NO_POSTURE_MARKER in rendered
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
# Issue #92: plain-language stances, inline thin-precedent markers, jargon
# strip, loud empty states.
# ---------------------------------------------------------------------------


def test_stance_lines_use_plain_language_sentences() -> None:
    """Every historical_stance enum (besides no_signal) renders via the
    module's plain-language sentence, not `Historically **enum_value**`,
    and keeps the held/of counts (in words, not raw JSON) alongside it."""
    for stance, sentence in _STANCE_SENTENCES.items():
        clause = _clause(
            historical_stance=stance,
            stance_detail={"held": 3, "of": 4, "basis": "our_paper"},
        )
        rendered = render_prompt(_doc_with_clause(clause))
        assert f"Historically **{stance}**" not in rendered
        assert sentence in rendered
        assert "held 3 of 4 our-paper deals" in rendered


def test_no_signal_stance_renders_occurrence_and_provenance_sentence() -> None:
    """Ticket example: no_signal -> 'Not enough history to establish a
    stance (1 occurrence, counterparty paper only).'"""
    clause = _clause(
        historical_stance="no_signal",
        n_our_paper=0,
        n_counterparty_paper=1,
        evidence_sufficient=False,
    )
    rendered = render_prompt(_doc_with_clause(clause))
    assert (
        "Not enough history to establish a stance (1 occurrence, counterparty paper only)."
        in rendered
    )


def test_v01_rollup_position_renders_recorded_stance_not_undetermined() -> None:
    """Issue #92 fix round 2: a v0.1-shaped clause (``rollup.position``, no
    ``summary`` key at all) must render its recorded position via
    _V01_POSITION_SENTENCES, not fall through to "could not be determined".
    Pre-fix, ``_STANCE_SENTENCES.get(stance, _UNKNOWN_STANCE_SENTENCE)``
    mapped every one of these four v0.1 enum values to the "undeterminable"
    sentence, silently turning a recorded stance into a false negative."""
    for position, sentence in _V01_POSITION_SENTENCES.items():
        clause = _v01_clause(position=position, n_our_paper=12)
        rendered = render_prompt(_doc_with_clause(clause))
        assert _UNKNOWN_STANCE_SENTENCE not in rendered, position
        assert sentence in rendered, position


def test_v01_shipped_examples_render_recorded_stance_not_undetermined() -> None:
    """End-to-end regression against the two shipped, `playbook
    validate`-passing v0.1 example playbooks that motivated this fix: at
    HEAD (pre-fix-round-2) all 6 clauses across these two files rendered
    "This clause's historical stance could not be determined", asserting
    the opposite of what each document actually records."""
    for name in ("our-paper-baseline.playbook.json", "emergent-no-template.playbook.json"):
        doc = json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))
        rendered = render_prompt(doc)
        assert _UNKNOWN_STANCE_SENTENCE not in rendered, name
        for clause in doc["clauses"]:
            position = clause["rollup"]["position"]
            assert _V01_POSITION_SENTENCES[position] in rendered, (name, position)


def test_unmapped_recorded_stance_preserves_value_not_undetermined() -> None:
    """A stance string that is neither a v0.2 nor a v0.1 enum value this
    renderer recognizes (e.g. a future OPF enum addition, or a foreign
    playbook) must still surface the value the document actually carries —
    only clause_stance()'s literal "unknown" sentinel earns
    _UNKNOWN_STANCE_SENTENCE (issue #92 fix round 2)."""
    clause = _clause(historical_stance="some_future_stance")
    rendered = render_prompt(_doc_with_clause(clause))
    assert "Recorded stance: some_future_stance." in rendered
    assert _UNKNOWN_STANCE_SENTENCE not in rendered


def test_missing_stance_shape_renders_unknown_sentence() -> None:
    """When a clause carries neither `summary.historical_stance` nor
    `rollup.position` (clause_stance()'s true "unknown" sentinel case),
    _UNKNOWN_STANCE_SENTENCE is still the right thing to render — this is
    the one case the sentence is reserved for post-fix."""
    clause = _v01_clause(position="")  # falsy -> clause_stance() -> "unknown"
    rendered = render_prompt(_doc_with_clause(clause))
    assert _UNKNOWN_STANCE_SENTENCE in rendered


def test_fixture_render_carries_no_engine_jargon() -> None:
    """Acceptance criterion: the fixture render contains no `no_signal`, no
    `n_our_paper`, no `moved by unknown`, no `Risk profile: not recorded` —
    exercised across every stance enum plus a no-signal clause, a
    negotiation-trail entry whose `moved_by` is the literal schema value
    `"unknown"` (both OPF schemas require `moved_by`; the engine always
    writes this string, never a missing key — issue #92 fix round 1), and a
    clause-library concept with no recorded risk profile, all in one
    document."""
    # Titles are deliberately plain English (never the enum spelling itself)
    # so this test can honestly assert the enum token is absent from the
    # render — the enum only ever enters via `historical_stance`.
    friendly_titles = {
        "consistently_held": "Alpha Clause",
        "usually_held": "Bravo Clause",
        "mixed": "Charlie Clause",
        "usually_conceded": "Delta Clause",
    }
    clauses = [
        _clause(
            historical_stance=stance,
            title=friendly_titles[stance],
            stance_detail={"held": 1, "of": 2, "basis": "our_paper"},
            n_our_paper=2,
            n_counterparty_paper=0,
        )
        for stance in _STANCE_SENTENCES
    ]
    clauses.append(
        _clause(
            historical_stance="no_signal",
            title="Echo Clause",
            n_our_paper=0,
            n_counterparty_paper=1,
            evidence_sufficient=False,
        )
    )
    clauses[0]["negotiation_trail"] = [
        {
            "document_id": "doc-1",
            "round": 1,
            "moved_by": "unknown",
            "change_summary": "Change with no recorded mover.",
        }
    ]

    doc = _flagship()
    doc["evidence"]["clauses"] = clauses
    doc["evidence"]["clause_library"] = [
        {
            "taxonomy_id": "confidentiality",
            "description": "Standard mutual confidentiality obligations.",
            "accepted_forms": [],
        }
    ]
    rendered = render_prompt(doc)

    assert "no_signal" not in rendered
    assert "n_our_paper" not in rendered
    assert "moved by unknown" not in rendered
    assert "Risk profile: not recorded" not in rendered


def test_moved_by_missing_renders_unrecorded_party_not_unknown() -> None:
    """Tolerant-read case: a hand-edited/foreign playbook may simply omit
    `moved_by` even though the schema requires it (see the tolerant-reads
    contract in render_prompt's HARD LINES section) — the missing key must
    still normalize to _UNRECORDED_MOVER, not render literally as absent."""
    doc = _flagship()
    clause = doc["evidence"]["clauses"][0]
    clause["negotiation_trail"] = [
        {
            "document_id": "metro-tech-2021",
            "round": 1,
            "change_summary": "Some change with no recorded mover.",
        }
    ]
    rendered = render_prompt(doc)
    assert "moved by unknown" not in rendered
    assert f"moved by {_UNRECORDED_MOVER}" in rendered


def test_moved_by_literal_unknown_renders_unrecorded_party_not_unknown() -> None:
    """Real-world case (issue #92 fix round 1): both OPF schemas make
    `moved_by` REQUIRED with enum `["us", "counterparty", "unknown"]`
    (spec/playbook.schema-0.2.json, spec/playbook.schema-0.3.json), and the
    engine always writes one of those three strings
    (observation_builder.build_round_moves writes the literal "unknown"
    when tracked-changes attribution fails) — a missing key never actually
    occurs. This schema-valid entry is the shape a real fixture render
    produces and must also render as the unrecorded-party phrase, not the
    raw enum value."""
    doc = _flagship()
    clause = doc["evidence"]["clauses"][0]
    clause["negotiation_trail"] = [
        {
            "document_id": "metro-tech-2021",
            "round": 1,
            "moved_by": "unknown",
            "change_summary": "Some change with no recorded mover.",
            "ref": {"document_id": "metro-tech-2021", "version": 2, "clause_path": "8"},
        }
    ]
    rendered = render_prompt(doc)
    assert "moved by unknown" not in rendered
    assert f"moved by {_UNRECORDED_MOVER}" in rendered


def test_clause_library_omits_risk_profile_line_when_absent_but_keeps_description() -> None:
    doc = _flagship()
    doc["evidence"]["clause_library"] = [
        {
            "taxonomy_id": "confidentiality",
            "description": "Standard mutual confidentiality obligations.",
            "accepted_forms": [],
        },
        {
            "taxonomy_id": "termination",
            "description": "Termination for convenience with notice.",
            "risk_profile": "low",
            "accepted_forms": [],
        },
    ]
    rendered = render_prompt(doc)
    assert "Risk profile: not recorded" not in rendered
    assert "Standard mutual confidentiality obligations." in rendered
    assert "Risk profile: low" in rendered


def test_historical_stance_field_name_dropped_from_preamble_and_evidence_intro() -> None:
    """The `historical_stance` backtick-enum mention becomes plain prose —
    both occurrences (role preamble and EVIDENCE section intro)."""
    rendered = render_prompt(_flagship())
    assert "historical_stance" not in rendered


def test_thin_precedent_marker_on_low_evidence_clauses() -> None:
    """Every thin clause (5 here, covering both trigger conditions and their
    combination) carries the inline THIN PRECEDENT marker on its heading,
    AND the complete heading including the parenthetical — the parenthetical
    must name the evidence that actually triggered the marker (issue #92 fix
    round 1: it used to be derived from ``len(observed_positions)`` rather
    than each position's ``precedent_count``, so a lone position with
    ``precedent_count: 40`` and ``evidence_sufficient: false`` rendered the
    self-contradicting "(single occurrence)" instead of naming the real
    trigger)."""
    thin_clauses_and_details = [
        (
            _clause(
                title="Insufficient Evidence Only",
                historical_stance="no_signal",
                evidence_sufficient=False,
                observed_positions=[_position(precedent_count=4)],
            ),
            "low confidence",  # sole position has precedent_count=4, not 1
        ),
        (
            _clause(
                title="Single Occurrence Only",
                historical_stance="mixed",
                evidence_sufficient=True,
                observed_positions=[_position(precedent_count=1)],
            ),
            "single occurrence",  # sole position truly has precedent_count=1
        ),
        (
            _clause(
                title="Two Single Occurrence Positions",
                historical_stance="mixed",
                evidence_sufficient=True,
                observed_positions=[_position(precedent_count=1), _position(precedent_count=1)],
            ),
            "thin evidence",  # no single position to name — two, not one
        ),
        (
            _clause(
                title="Both Conditions At Once",
                historical_stance="no_signal",
                evidence_sufficient=False,
                observed_positions=[_position(precedent_count=1)],
            ),
            "single occurrence",  # sole position has precedent_count=1
        ),
        (
            _clause(
                title="No Positions Recorded",
                historical_stance="no_signal",
                evidence_sufficient=False,
                observed_positions=[],
            ),
            "low confidence",  # no position at all to call "the" occurrence
        ),
    ]
    doc = _flagship()
    doc["evidence"]["clauses"] = [clause for clause, _detail in thin_clauses_and_details]
    doc["evidence"]["clause_library"] = []
    rendered = render_prompt(doc)
    for clause, detail in thin_clauses_and_details:
        assert f"### {clause['title']} — {_THIN_TERM} ({detail})" in rendered, clause["title"]


def test_thin_precedent_marker_absent_for_sufficient_multi_precedent_clause() -> None:
    """A synthetic clause with evidence_sufficient: true and
    precedent_count > 1 renders without the marker on ITS heading.

    (The CITATION & CONFIDENCE RULES section legitimately names
    _THIN_TERM unconditionally — same as it already did for `1x
    precedent` pre-fix — so the assertion is heading-specific, not a
    whole-document absence check.)
    """
    clause = _clause(
        title="Well Evidenced",
        historical_stance="usually_held",
        evidence_sufficient=True,
        observed_positions=[_position(precedent_count=2)],
        stance_detail={"held": 2, "of": 2, "basis": "our_paper"},
    )
    rendered = render_prompt(_doc_with_clause(clause))
    assert "### Well Evidenced\n" in rendered
    assert f"### Well Evidenced — {_THIN_TERM}" not in rendered


def test_thin_marker_term_matches_citation_rules_section() -> None:
    """The heading marker and the CITATION & CONFIDENCE RULES section name
    the same term, so the model can correlate the two."""
    clause = _clause(
        title="Thin Clause",
        historical_stance="no_signal",
        evidence_sufficient=False,
        observed_positions=[],
    )
    rendered = render_prompt(_doc_with_clause(clause))
    assert f"### Thin Clause — {_THIN_TERM}" in rendered
    citation_section = rendered.split("## CITATION & CONFIDENCE RULES", 1)[1]
    assert _THIN_TERM in citation_section


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
