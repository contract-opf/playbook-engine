"""Tests for the ClausePosition compiler (L5, issue #22).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional party/document names only
(e.g., "Alice", "Bob", "Acme Corp", "Beta LLC").
"""

from __future__ import annotations

import pytest

from playbook_engine.clause_position_compiler import (
    COHERENCE_MIN_CITATIONS,
    UNCLASSIFIED_EXAMPLE_LIMIT,
    ClausePosition,
    ClauseRollup,
    CoherenceFlag,
    CoherenceJudge,
    OPFCitation,
    UnclassifiedCoverage,
    compile_clause_positions,
)
from playbook_engine.deviation_classifier import RiskDelta
from playbook_engine.observation_builder import Observation, ObservationCitation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEUTRAL = RiskDelta(direction="neutral", magnitude="none")
_WORSE_MINOR = RiskDelta(direction="worse", magnitude="minor")
_WORSE_MATERIAL = RiskDelta(direction="worse", magnitude="material")
_BETTER = RiskDelta(direction="better", magnitude="minor")


def _obs(
    taxonomy_id: str | None,
    provenance: str = "our_paper",
    outcome: str = "signed",
    deviation: str = "none",
    risk_delta: RiskDelta = _NEUTRAL,
    text: str = "Mutual indemnification.",
    doc_id: str = "deal_001",
    version: str = "v2",
    clause_path: str = "8",
    basis: str | None = None,
) -> Observation:
    return Observation(
        observation_id=f"{doc_id}/{version}/{clause_path}",
        taxonomy_id=taxonomy_id,
        text_summary=text,
        citation=ObservationCitation(
            document_id=doc_id,
            version=version,
            clause_path=clause_path,
            char_span=None,
        ),
        deviation=deviation,
        risk_delta=risk_delta.to_dict(),
        provenance=provenance,
        outcome=outcome,
        basis=basis,
    )


def _template_obs(
    taxonomy_id: str,
    text: str = "Standard mutual indemnification language.",
    clause_path: str = "8",
) -> Observation:
    return _obs(
        taxonomy_id=taxonomy_id,
        provenance="our_paper",
        outcome="signed",
        deviation="none",
        risk_delta=_NEUTRAL,
        text=text,
        doc_id="template",
        version="template",
        clause_path=clause_path,
    )


# ---------------------------------------------------------------------------
# Thin wrapper: unpack (positions, flags) so existing tests stay unchanged
# ---------------------------------------------------------------------------


def _compile(
    observations: list,
    template_observations: list,
    taxonomy_titles: dict | None = None,
) -> list[ClausePosition]:
    """Call compile_clause_positions and return only the positions list."""
    positions, _, _ = compile_clause_positions(
        observations,
        template_observations,
        taxonomy_titles=taxonomy_titles,
    )
    return positions


# ---------------------------------------------------------------------------
# compile_clause_positions: basic grouping and structure
# ---------------------------------------------------------------------------


def test_compile_returns_one_position_per_taxonomy_id() -> None:
    """Acceptance criterion: one ClausePosition per distinct taxonomy_id."""
    obs = [
        _obs("indemnification"),
        _obs("governing_law", doc_id="deal_001", version="v2", clause_path="12"),
    ]
    positions = _compile(obs, [])
    assert len(positions) == 2
    tids = {p.taxonomy_id for p in positions}
    assert tids == {"indemnification", "governing_law"}


def test_compile_none_taxonomy_id_skipped() -> None:
    """Unclassified observations (taxonomy_id=None) are silently skipped."""
    obs = [_obs(None), _obs("indemnification")]
    positions = _compile(obs, [])
    assert len(positions) == 1
    assert positions[0].taxonomy_id == "indemnification"


def test_compile_sorted_by_taxonomy_id() -> None:
    """Returned positions are in taxonomy_id sorted order."""
    obs = [
        _obs("governing_law"),
        _obs("indemnification"),
        _obs("confidentiality"),
    ]
    positions = _compile(obs, [])
    tids = [p.taxonomy_id for p in positions]
    assert tids == sorted(tids)


def test_compile_includes_template_only_taxonomy_ids() -> None:
    """A taxonomy_id present only in template_observations is included."""
    template_obs = [_template_obs("limitation_of_liability")]
    positions = _compile([], template_obs)
    assert len(positions) == 1
    assert positions[0].taxonomy_id == "limitation_of_liability"


def test_compile_id_format() -> None:
    """ClausePosition.id = 'clause.<taxonomy_id>'."""
    positions = _compile([_obs("indemnification")], [])
    assert positions[0].id == "clause.indemnification"


def test_compile_title_derived_from_taxonomy_id() -> None:
    """Title defaults to title-cased words from taxonomy_id."""
    positions = _compile([_obs("limitation_of_liability")], [])
    assert positions[0].title == "Limitation Of Liability"


def test_compile_title_overridden_by_taxonomy_titles() -> None:
    """taxonomy_titles map overrides the default title derivation."""
    positions = _compile(
        [_obs("indemnification")],
        [],
        taxonomy_titles={"indemnification": "Indemnification & Defense"},
    )
    assert positions[0].title == "Indemnification & Defense"


# ---------------------------------------------------------------------------
# our_standard: set from template, absent when counterparty-paper-only
# ---------------------------------------------------------------------------


def test_our_standard_set_from_template_observation() -> None:
    """Acceptance: our_standard text + citation come from template_observation."""
    t_obs = _template_obs("indemnification", text="Mutual indemnification.", clause_path="8")
    deal_obs = [_obs("indemnification")]
    positions = _compile(deal_obs, [t_obs])

    pos = positions[0]
    assert pos.our_standard is not None
    assert pos.our_standard.text == "Mutual indemnification."
    assert pos.our_standard.source_ref.document_id == "template"
    assert pos.our_standard.source_ref.version == "template"
    assert pos.our_standard.source_ref.clause_path == "8"


def test_our_standard_absent_when_template_text_empty() -> None:
    """An empty-text template observation yields our_standard=None (issue #182).

    Deterministic segmentation can classify a heading-only template clause,
    producing a template observation with blank full_text. Building an
    OurStandard from it would be present-with-empty-text and fail OPF
    validation ("our_standard.text is empty"), blocking projection — the clause
    must degrade to emergent (our_standard=None) instead.
    """
    empty_t_obs = Observation(
        observation_id="template/template/8",
        taxonomy_id="indemnification",
        text_summary="",
        full_text="",
        citation=ObservationCitation(
            document_id="template", version="template", clause_path="8", char_span=None
        ),
        deviation="none",
        risk_delta=_NEUTRAL.to_dict(),
        provenance="our_paper",
        outcome="signed",
    )
    deal_obs = [_obs("indemnification", provenance="counterparty_paper")]
    positions = _compile(deal_obs, [empty_t_obs])

    assert len(positions) == 1
    assert positions[0].our_standard is None


def test_our_standard_carries_full_text() -> None:
    """Regression (audit 2026-07, issue #105): our_standard.text must be the
    untruncated clause text, not the 200-char text_summary — any real
    indemnification/insurance clause exceeds 200 chars, and a truncated
    fragment is useless as a drafting standard."""
    long_text = "Each party shall indemnify the other against claims. " * 5
    assert len(long_text) > 200
    t_obs = Observation(
        observation_id="template/template/8",
        taxonomy_id="indemnification",
        text_summary=long_text[:200],
        full_text=long_text,
        citation=ObservationCitation(
            document_id="template", version="template", clause_path="8", char_span=None
        ),
        deviation="none",
        risk_delta=_NEUTRAL.to_dict(),
        provenance="our_paper",
        outcome="signed",
    )
    positions = _compile([], [t_obs])
    pos = positions[0]
    assert pos.our_standard is not None
    assert pos.our_standard.text == long_text
    assert len(pos.our_standard.text) > 200


def test_acceptable_if_carries_full_text() -> None:
    """Regression (issue #105): acceptable_if entries must be the full clause
    text — that IS the acceptable alternative language lawyers need, not a
    200-char fragment of it."""
    long_text = "Mutual indemnification limited to gross negligence. " * 5
    assert len(long_text) > 200
    t_obs = _template_obs("indemnification")
    obs = Observation(
        observation_id="deal_001/v2/8",
        taxonomy_id="indemnification",
        text_summary=long_text[:200],
        full_text=long_text,
        citation=ObservationCitation(
            document_id="deal_001", version="v2", clause_path="8", char_span=None
        ),
        deviation="reworded_equivalent",
        risk_delta=_NEUTRAL.to_dict(),
        provenance="our_paper",
        outcome="signed",
    )
    positions = _compile([obs], [t_obs])
    assert len(positions[0].rollup.acceptable_if) == 1
    entry = positions[0].rollup.acceptable_if[0]
    assert entry.to == long_text
    assert len(entry.to) > 200


def test_acceptable_if_entry_is_if_to_rationale_triple() -> None:
    """Issue #141: acceptable_if entries are structured {if,to,rationale}
    triples (the acceptable_variations shape consuming apps prove out), each citing
    its supporting observation — not free text."""
    t_obs = _template_obs("indemnification")
    obs = Observation(
        observation_id="deal_001/v2/8",
        taxonomy_id="indemnification",
        text_summary="Mutual indemnification, reworded but equivalent.",
        full_text="Mutual indemnification, reworded but equivalent (full clause text).",
        citation=ObservationCitation(
            document_id="deal_001", version="v2", clause_path="8", char_span=(0, 40)
        ),
        deviation="reworded_equivalent",
        risk_delta=_NEUTRAL.to_dict(),
        provenance="our_paper",
        outcome="signed",
    )
    positions = _compile([obs], [t_obs])
    entry = positions[0].rollup.acceptable_if[0]
    assert entry.if_ == "Mutual indemnification, reworded but equivalent."
    assert entry.to == "Mutual indemnification, reworded but equivalent (full clause text)."
    assert entry.rationale  # non-empty — cites deviation/precedent basis
    assert "reworded_equivalent" in entry.rationale
    # observation_ref resolves to the observation this entry was derived from.
    assert entry.observation_ref.document_id == "deal_001"
    assert entry.observation_ref.version == "v2"
    assert entry.observation_ref.clause_path == "8"

    d = entry.to_dict()
    assert set(d.keys()) == {"if", "to", "rationale", "observation_ref"}
    assert d["observation_ref"]["document_id"] == "deal_001"


def test_acceptable_if_serializes_as_triple_in_to_dict() -> None:
    """ClausePosition.to_dict()'s summary.acceptable_if carries the full
    {if,to,rationale,observation_ref} shape, not bare strings (issue #141)."""
    t_obs = _template_obs("indemnification")
    obs = Observation(
        observation_id="deal_001/v2/8",
        taxonomy_id="indemnification",
        text_summary="Mutual, reworded.",
        full_text="Mutual indemnification, reworded but equivalent.",
        citation=ObservationCitation(
            document_id="deal_001", version="v2", clause_path="8", char_span=None
        ),
        deviation="reworded_equivalent",
        risk_delta=_NEUTRAL.to_dict(),
        provenance="our_paper",
        outcome="signed",
    )
    positions = _compile([obs], [t_obs])
    d = positions[0].to_dict()
    entries = d["summary"]["acceptable_if"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["if"] == "Mutual, reworded."
    assert entry["to"] == "Mutual indemnification, reworded but equivalent."
    assert isinstance(entry["rationale"], str) and entry["rationale"]
    assert entry["observation_ref"]["document_id"] == "deal_001"


def test_rollup_fallback_carries_full_text() -> None:
    """Regression (issue #105): fallback text must be the full clause text,
    not the 200-char text_summary."""
    long_text = "We accept liability up to the aggregate fees paid. " * 5
    assert len(long_text) > 200
    t_obs = _template_obs("indemnification")
    obs = Observation(
        observation_id="deal_001/v2/8",
        taxonomy_id="indemnification",
        text_summary=long_text[:200],
        full_text=long_text,
        citation=ObservationCitation(
            document_id="deal_001", version="v2", clause_path="8", char_span=None
        ),
        deviation="substantive",
        risk_delta=_WORSE_MINOR.to_dict(),
        provenance="our_paper",
        outcome="signed",
    )
    positions = _compile([obs], [t_obs])
    assert len(positions[0].rollup.fallbacks) == 1
    assert positions[0].rollup.fallbacks[0].full_text == long_text


def test_our_standard_none_when_no_template_obs() -> None:
    """No our_standard if template_observations has no entry for this tid."""
    positions = _compile([_obs("indemnification")], [])
    assert positions[0].our_standard is None


# ---------------------------------------------------------------------------
# §2.2 provenance rule — structural enforcement tests
# ---------------------------------------------------------------------------


def test_provenance_rule_counterparty_only_no_our_standard() -> None:
    """Acceptance criterion: counterparty-paper-only → our_standard is None."""
    obs = [_obs("indemnification", provenance="counterparty_paper")]
    # No template passed: template would give has_our_paper=True and enable our_standard.
    positions = _compile(obs, [])
    assert positions[0].our_standard is None


def test_provenance_rule_counterparty_only_position_capped_at_negotiable() -> None:
    """Acceptance criterion: counterparty-paper-only → position='negotiable'."""
    obs = [_obs("indemnification", provenance="counterparty_paper")]
    positions = _compile(obs, [])
    # §2.2: counterparty-paper-only cannot have position stronger than "negotiable".
    assert positions[0].rollup.position == "negotiable"


def test_provenance_rule_counterparty_only_cannot_emit_stronger_position() -> None:
    """Acceptance criterion: provenance rule violation is IMPOSSIBLE to emit.

    Even with many counterparty-paper signed observations, the position must
    remain 'negotiable'. This is enforced structurally, not by validation.
    """
    obs = [
        _obs("indemnification", provenance="counterparty_paper", deviation="none"),
        _obs(
            "indemnification",
            provenance="counterparty_paper",
            deviation="none",
            doc_id="deal_002",
            version="v1",
        ),
        _obs(
            "indemnification",
            provenance="counterparty_paper",
            deviation="none",
            doc_id="deal_003",
            version="v1",
        ),
    ]
    positions = _compile(obs, [])
    pos = positions[0]
    # Structural guarantee: these values cannot appear for counterparty-paper-only.
    assert pos.our_standard is None
    assert pos.rollup.position not in {"standard", "acceptable_variants_exist", "hold_firm"}
    assert pos.rollup.position == "negotiable"


def test_provenance_rule_template_obs_provides_our_paper() -> None:
    """Template observation counts as our-paper; enables our_standard and stronger positions."""
    t_obs = _template_obs("indemnification")
    # No deal observations — only template.
    positions = _compile([], [t_obs])
    pos = positions[0]
    assert pos.our_standard is not None
    # Template-only grounding: conservative "negotiable" (no deal evidence to assert standard).
    assert pos.rollup.position == "negotiable"


def test_provenance_rule_template_must_be_our_paper() -> None:
    """ValueError if template_observations contains non-our-paper provenance."""
    bad_template = _obs(
        "indemnification", provenance="counterparty_paper", doc_id="template", version="template"
    )
    with pytest.raises(ValueError, match="provenance='our_paper'"):
        compile_clause_positions([], [bad_template])


def test_provenance_rule_template_plus_counterparty_reversal_not_hold_firm() -> None:
    """Regression (§2.2): a template match (our-paper grounding) plus counterparty-paper
    observations — including a proposed_then_reversed ask — must NOT be promoted to
    'hold_firm'. Only our-paper DEAL signal may set a position stronger than 'negotiable';
    a counterparty reversal previously rode in on the template match and produced an
    illegal hold_firm that failed schema validation on the real corpus."""
    template = [_template_obs("indemnification")]
    obs = [
        _obs(
            "indemnification",
            provenance="counterparty_paper",
            outcome="signed",
            deviation="substantive",
            doc_id="deal_a",
            version="v3",
        ),
        _obs(
            "indemnification",
            provenance="counterparty_paper",
            outcome="proposed_then_reversed",
            deviation="substantive",
            doc_id="deal_b",
            version="v2",
        ),
    ]
    pos = _compile(obs, template)[0]
    # our_standard is set from the template (template is our-paper drafting) ...
    assert pos.our_standard is not None
    # ... but the position is capped at negotiable: no our-paper DEAL evidence exists.
    assert pos.rollup.position == "negotiable"
    # the counterparty reversal is still preserved as evidence, just not as the position.
    assert len(pos.rollup.rejected) == 1


# ---------------------------------------------------------------------------
# observed_positions: citations carried on every asserted text
# ---------------------------------------------------------------------------


def test_observed_positions_count() -> None:
    """One ObservedPosition per observation in the group."""
    obs = [
        _obs("indemnification", doc_id="deal_001", version="v2"),
        _obs("indemnification", doc_id="deal_002", version="v1"),
    ]
    positions = _compile(obs, [])
    assert len(positions[0].observed_positions) == 2


def test_observed_positions_citation_carried() -> None:
    """Citations are present on every observed position (no citation omitted)."""
    obs = [_obs("indemnification", doc_id="deal_007", version="v3", clause_path="9")]
    positions = _compile(obs, [])
    op = positions[0].observed_positions[0]
    assert op.example_ref.document_id == "deal_007"
    assert op.example_ref.version == "v3"
    assert op.example_ref.clause_path == "9"


def test_observed_positions_text_summary_preserved() -> None:
    obs = [_obs("indemnification", text="Alice shall indemnify Beta LLC.")]
    positions = _compile(obs, [])
    assert positions[0].observed_positions[0].text_summary == "Alice shall indemnify Beta LLC."


def test_observed_positions_deviation_and_risk_delta() -> None:
    obs = [_obs("indemnification", deviation="substantive", risk_delta=_WORSE_MATERIAL)]
    positions = _compile(obs, [])
    op = positions[0].observed_positions[0]
    assert op.deviation == "substantive"
    assert op.risk_delta == {"direction": "worse", "magnitude": "material"}


def test_observed_positions_outcome_preserved() -> None:
    obs = [_obs("indemnification", outcome="proposed_then_reversed")]
    positions = _compile(obs, [])
    assert positions[0].observed_positions[0].outcome == "proposed_then_reversed"


def test_observed_positions_provenance_preserved() -> None:
    obs = [_obs("indemnification", provenance="counterparty_paper")]
    positions = _compile(obs, [])
    assert positions[0].observed_positions[0].provenance == "counterparty_paper"


# ---------------------------------------------------------------------------
# rollup.position derivation
# ---------------------------------------------------------------------------


def test_position_standard_all_deviation_none() -> None:
    """Standard: all our-paper observations have deviation=none, no concessions.

    Two our-paper observations (issue #107 evidence-depth floor — a single
    observation is capped at "negotiable" regardless of the cascade below;
    see test_evidence_depth_caps_single_observation_at_negotiable).
    """
    t_obs = _template_obs("indemnification")
    obs = [
        _obs("indemnification", deviation="none", outcome="signed"),
        _obs("indemnification", deviation="none", outcome="signed", doc_id="deal_002"),
    ]
    positions = _compile(obs, [t_obs])
    assert positions[0].rollup.position == "standard"


def test_position_acceptable_variants_exist() -> None:
    """acceptable_variants_exist: neutral-risk signed variant (deviation != none).

    Two our-paper observations (issue #107 evidence-depth floor).
    """
    t_obs = _template_obs("indemnification")
    obs = [
        _obs(
            "indemnification",
            deviation="reworded_equivalent",
            risk_delta=_NEUTRAL,
            outcome="signed",
        ),
        _obs(
            "indemnification",
            deviation="reworded_equivalent",
            risk_delta=_NEUTRAL,
            outcome="signed",
            doc_id="deal_002",
        ),
    ]
    positions = _compile(obs, [t_obs])
    assert positions[0].rollup.position == "acceptable_variants_exist"


def test_unjudged_deviation_does_not_fabricate_acceptable_variants() -> None:
    """A neutral-risk substantive variant whose basis is unjudged
    (``needs_review`` — e.g. stub-mode or judge-error) must NOT compile into an
    ``acceptable_variants_exist`` position: nothing actually assessed the risk,
    so the neutral risk_delta is a placeholder, not evidence of an acceptable
    variant. With no other signal this falls through to ``standard``.

    Two our-paper observations (issue #107 evidence-depth floor).
    """
    t_obs = _template_obs("indemnification")
    obs = [
        _obs(
            "indemnification",
            deviation="substantive",
            risk_delta=_NEUTRAL,
            outcome="signed",
            basis="needs_review",
        ),
        _obs(
            "indemnification",
            deviation="substantive",
            risk_delta=_NEUTRAL,
            outcome="signed",
            basis="needs_review",
            doc_id="deal_002",
        ),
    ]
    positions = _compile(obs, [t_obs])
    assert positions[0].rollup.position == "standard"
    # And it must not leak into acceptable_if either.
    assert positions[0].rollup.acceptable_if == ()


def test_judged_deviation_still_yields_acceptable_variants() -> None:
    """The counterpart to the unjudged case: an identical neutral-risk
    substantive variant WITH ``basis="judge"`` is a real assessment and must
    still compile into ``acceptable_variants_exist``.

    Two our-paper observations (issue #107 evidence-depth floor).
    """
    t_obs = _template_obs("indemnification")
    obs = [
        _obs(
            "indemnification",
            deviation="substantive",
            risk_delta=_NEUTRAL,
            outcome="signed",
            basis="judge",
        ),
        _obs(
            "indemnification",
            deviation="substantive",
            risk_delta=_NEUTRAL,
            outcome="signed",
            basis="judge",
            doc_id="deal_002",
        ),
    ]
    positions = _compile(obs, [t_obs])
    assert positions[0].rollup.position == "acceptable_variants_exist"


def test_stub_basis_caps_position_at_negotiable() -> None:
    """issue #101: an our-paper signed observation carrying ``basis="stub"``
    (no judge configured at all) must not yield "standard" or
    "acceptable_variants_exist" — even though an identical deviation="none"
    observation without a stub basis WOULD yield "standard"
    (test_position_standard_all_deviation_none). "stub" is stricter than
    "needs_review": it means no judge was ever configured, not that one
    judge call failed, so no clause type touched by it can be trusted beyond
    the §2.2-style "negotiable" cap."""
    t_obs = _template_obs("indemnification")
    obs = [
        _obs(
            "indemnification",
            deviation="none",
            risk_delta=_NEUTRAL,
            outcome="signed",
            basis="stub",
        ),
    ]
    positions = _compile(obs, [t_obs])
    assert positions[0].rollup.position == "negotiable"
    assert positions[0].rollup.position not in {
        "standard",
        "acceptable_variants_exist",
        "hold_firm",
    }


def test_stub_basis_excluded_from_acceptable_if() -> None:
    """A neutral-risk substantive variant with basis="stub" must not leak into
    acceptable_if — mirrors the needs_review/judge_error exclusion."""
    t_obs = _template_obs("indemnification")
    obs = [
        _obs(
            "indemnification",
            deviation="substantive",
            risk_delta=_NEUTRAL,
            outcome="signed",
            basis="stub",
        ),
    ]
    positions = _compile(obs, [t_obs])
    assert positions[0].rollup.acceptable_if == ()
    assert positions[0].rollup.position == "negotiable"


def test_stub_basis_caps_even_with_other_judged_observations() -> None:
    """A single stub-basis observation caps the WHOLE clause type at
    "negotiable", even when other observations for the same taxonomy_id were
    genuinely judged and would otherwise support "standard"."""
    t_obs = _template_obs("indemnification")
    obs = [
        _obs(
            "indemnification",
            deviation="none",
            risk_delta=_NEUTRAL,
            outcome="signed",
            basis="deterministic",
            doc_id="deal_001",
        ),
        _obs(
            "indemnification",
            deviation="none",
            risk_delta=_NEUTRAL,
            outcome="signed",
            basis="stub",
            doc_id="deal_002",
        ),
    ]
    positions = _compile(obs, [t_obs])
    assert positions[0].rollup.position == "negotiable"


def test_position_negotiable_when_concessions_exist() -> None:
    """negotiable: worse-risk signed our-paper observations exist."""
    t_obs = _template_obs("indemnification")
    obs = [
        _obs("indemnification", deviation="substantive", risk_delta=_WORSE_MINOR, outcome="signed"),
    ]
    positions = _compile(obs, [t_obs])
    assert positions[0].rollup.position == "negotiable"


def test_position_hold_firm_reversals_no_concessions() -> None:
    """hold_firm: proposed_then_reversed with no worse-risk signed observations.

    Two our-paper observations (issue #107 evidence-depth floor).
    """
    t_obs = _template_obs("indemnification")
    obs = [
        _obs(
            "indemnification",
            deviation="substantive",
            risk_delta=_WORSE_MINOR,
            outcome="proposed_then_reversed",
        ),
        _obs(
            "indemnification",
            deviation="substantive",
            risk_delta=_WORSE_MINOR,
            outcome="proposed_then_reversed",
            doc_id="deal_002",
        ),
    ]
    positions = _compile(obs, [t_obs])
    assert positions[0].rollup.position == "hold_firm"


def test_position_negotiable_takes_precedence_over_hold_firm() -> None:
    """negotiable when both fallbacks and reversals present (concession dominates)."""
    t_obs = _template_obs("indemnification")
    obs = [
        _obs("indemnification", deviation="substantive", risk_delta=_WORSE_MINOR, outcome="signed"),
        _obs(
            "indemnification",
            deviation="substantive",
            risk_delta=_WORSE_MINOR,
            outcome="proposed_then_reversed",
            doc_id="deal_002",
            version="v1",
        ),
    ]
    positions = _compile(obs, [t_obs])
    assert positions[0].rollup.position == "negotiable"


# ---------------------------------------------------------------------------
# rollup.fallbacks and rollup.rejected
# ---------------------------------------------------------------------------


def test_rollup_fallbacks_are_worse_risk_signed_our_paper() -> None:
    """fallbacks = signed our-paper observations with risk_delta.direction=worse."""
    t_obs = _template_obs("indemnification")
    obs = [
        _obs("indemnification", deviation="substantive", risk_delta=_WORSE_MINOR, outcome="signed"),
    ]
    positions = _compile(obs, [t_obs])
    assert len(positions[0].rollup.fallbacks) == 1
    assert positions[0].rollup.fallbacks[0].risk_delta["direction"] == "worse"


def test_rollup_fallbacks_exclude_counterparty_paper() -> None:
    """Counterparty-paper observations never appear in fallbacks."""
    obs = [
        _obs(
            "indemnification",
            provenance="counterparty_paper",
            deviation="substantive",
            risk_delta=_WORSE_MINOR,
            outcome="signed",
        ),
    ]
    positions = _compile(obs, [])
    assert len(positions[0].rollup.fallbacks) == 0


def test_rollup_rejected_are_proposed_then_reversed() -> None:
    """rejected = observations with outcome=proposed_then_reversed."""
    t_obs = _template_obs("indemnification")
    obs = [
        _obs("indemnification", outcome="proposed_then_reversed", risk_delta=_WORSE_MATERIAL),
    ]
    positions = _compile(obs, [t_obs])
    assert len(positions[0].rollup.rejected) == 1
    assert positions[0].rollup.rejected[0].outcome == "proposed_then_reversed"


def test_rollup_signed_obs_not_in_rejected() -> None:
    obs = [_obs("indemnification", outcome="signed")]
    positions = _compile(obs, [])
    assert len(positions[0].rollup.rejected) == 0


# ---------------------------------------------------------------------------
# rollup.confidence
# ---------------------------------------------------------------------------


def test_rollup_confidence_fields_present() -> None:
    obs = [_obs("indemnification")]
    positions = _compile(obs, [])
    conf = positions[0].rollup.confidence
    assert "score" in conf
    assert "n_our_paper" in conf
    assert "n_counterparty_paper" in conf


def test_rollup_confidence_counts_provenance() -> None:
    obs = [
        _obs("indemnification", provenance="our_paper"),
        _obs("indemnification", provenance="counterparty_paper", doc_id="deal_002", version="v1"),
        _obs("indemnification", provenance="counterparty_paper", doc_id="deal_003", version="v1"),
    ]
    positions = _compile(obs, [])
    conf = positions[0].rollup.confidence
    assert conf["n_our_paper"] == 1
    assert conf["n_counterparty_paper"] == 2


def test_rollup_confidence_score_range() -> None:
    obs = [
        _obs("indemnification", provenance="our_paper"),
        _obs("indemnification", provenance="our_paper", doc_id="d2", version="v1"),
    ]
    positions = _compile(obs, [])
    score = positions[0].rollup.confidence["score"]
    assert 0.0 <= score <= 1.0


def test_rollup_confidence_score_zero_for_no_observations() -> None:
    """Template-only clause (no deal observations) has confidence score 0."""
    t_obs = _template_obs("indemnification")
    positions = _compile([], [t_obs])
    assert positions[0].rollup.confidence["score"] == 0.0


# ---------------------------------------------------------------------------
# ClausePosition.to_dict: OPF-shaped output
# ---------------------------------------------------------------------------


def test_to_dict_structure() -> None:
    """to_dict() produces the expected top-level OPF keys."""
    t_obs = _template_obs("indemnification")
    obs = [_obs("indemnification")]
    positions = _compile(obs, [t_obs])
    d = positions[0].to_dict()

    assert d["id"] == "clause.indemnification"
    assert d["taxonomy_id"] == "indemnification"
    assert "title" in d
    assert "our_standard" in d
    assert "observed_positions" in d
    assert "summary" in d


def test_to_dict_our_standard_none_serializes_null() -> None:
    obs = [_obs("indemnification", provenance="counterparty_paper")]
    positions = _compile(obs, [])
    assert positions[0].to_dict()["our_standard"] is None


def test_to_dict_our_standard_carries_source_ref() -> None:
    """Acceptance: citations are present on every asserted text (our_standard)."""
    t_obs = _template_obs("indemnification", clause_path="8")
    positions = _compile([], [t_obs])
    d = positions[0].to_dict()
    assert d["our_standard"]["source_ref"]["document_id"] == "template"
    assert d["our_standard"]["source_ref"]["clause_path"] == "8"


def test_to_dict_observed_positions_each_have_example_ref() -> None:
    """Acceptance: every ObservedPosition in output carries example_ref (citation)."""
    obs = [
        _obs("indemnification", doc_id="deal_A", version="v2", clause_path="8.1"),
        _obs(
            "indemnification",
            doc_id="deal_B",
            version="v1",
            clause_path="5",
            provenance="counterparty_paper",
        ),
    ]
    positions = _compile(obs, [])
    d = positions[0].to_dict()
    for op in d["observed_positions"]:
        assert "example_ref" in op
        assert "document_id" in op["example_ref"]


def test_to_dict_summary_structure() -> None:
    """OPF v0.2 (§3.5): to_dict() emits `summary` with `historical_stance`
    (descriptive), not v0.1's `rollup.position` (prescriptive)."""
    obs = [_obs("indemnification")]
    positions = _compile(obs, [])
    summary = positions[0].to_dict()["summary"]
    assert "historical_stance" in summary
    assert "acceptable_if" in summary
    assert "fallbacks" in summary
    assert "rejected" in summary
    assert "confidence" in summary


def test_to_dict_historical_stance_no_signal_when_evidence_insufficient() -> None:
    """Below MIN_EVIDENCE_N our-paper observations → historical_stance is the
    descriptive "no_signal", never a stronger stance (mirrors the §2.2/#107
    rollup.position="negotiable" cap)."""
    obs = [_obs("indemnification")]  # single our-paper observation
    positions = _compile(obs, [])
    assert positions[0].rollup.position == "negotiable"
    assert positions[0].to_dict()["summary"]["historical_stance"] == "no_signal"


def test_to_dict_historical_stance_consistently_held() -> None:
    # A stance stronger than "mixed" requires an our_standard to point at
    # (OPF §2.2), so a template observation must be present (issue #182).
    t_obs = _template_obs("indemnification")
    obs = [_obs("indemnification"), _obs("indemnification", doc_id="deal_002")]
    positions = _compile(obs, [t_obs])
    assert positions[0].our_standard is not None
    assert positions[0].rollup.position == "standard"
    assert positions[0].to_dict()["summary"]["historical_stance"] == "consistently_held"


def test_strong_stance_capped_when_no_our_standard() -> None:
    """our-paper obs but no template clause → §2.2 cap (issue #182).

    Without an our_standard to reference, the position must cap at "negotiable"
    and historical_stance must not exceed "mixed" — otherwise the assembled
    playbook fails OPF §2.2 validation ('consistently_held' with
    our_standard: null).
    """
    obs = [_obs("indemnification"), _obs("indemnification", doc_id="deal_002")]
    positions = _compile(obs, [])  # no template observations
    pos = positions[0]
    assert pos.our_standard is None
    assert pos.rollup.position == "negotiable"
    assert pos.to_dict()["summary"]["historical_stance"] not in {
        "usually_conceded",
        "usually_held",
        "consistently_held",
    }


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_inputs_returns_empty() -> None:
    positions, flags, _ = compile_clause_positions([], [])
    assert positions == []
    assert flags == []


def test_empty_deal_observations_template_only() -> None:
    """Template-only: correct ClausePosition with no deal observations."""
    t_obs = _template_obs("confidentiality")
    positions = _compile([], [t_obs])
    assert len(positions) == 1
    assert positions[0].taxonomy_id == "confidentiality"
    assert positions[0].our_standard is not None
    assert len(positions[0].observed_positions) == 0


def test_multiple_template_observations_first_wins() -> None:
    """Only the first template observation for a taxonomy_id is used."""
    t1 = _template_obs("indemnification", text="First version.", clause_path="8")
    t2 = _template_obs("indemnification", text="Second version.", clause_path="9")
    positions = _compile([], [t1, t2])
    assert positions[0].our_standard is not None
    assert positions[0].our_standard.text == "First version."


def test_opf_citation_to_dict_no_optional_fields() -> None:
    """OPFCitation omits clause_path and char_span when None."""
    c = OPFCitation(document_id="template", version="template")
    d = c.to_dict()
    assert "clause_path" not in d
    assert "char_span" not in d


def test_opf_citation_to_dict_with_all_fields() -> None:
    c = OPFCitation(document_id="deal_x", version=3, clause_path="8.1", char_span=(0, 120))
    d = c.to_dict()
    assert d["version"] == 3
    assert d["clause_path"] == "8.1"
    assert d["char_span"] == [0, 120]


def test_clause_rollup_rejects_invalid_position() -> None:
    with pytest.raises(ValueError, match="position"):
        ClauseRollup(
            position="unknown",
            acceptable_if=(),
            fallbacks=(),
            rejected=(),
            confidence={"score": 0.0},
        )


# ---------------------------------------------------------------------------
# CoherenceJudge — acceptance criteria (issue #54)
# ---------------------------------------------------------------------------


class _FlagAllJudge:
    """Stub judge that always returns a CoherenceFlag (severity=warn)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def judge(self, clause_summary: dict) -> CoherenceFlag:
        self.calls.append(clause_summary)
        return CoherenceFlag(
            clause_id=clause_summary["clause_id"],
            reason="low citation count",
            severity="warn",
        )


class _NullJudge:
    """Stub judge that never flags (returns None)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def judge(self, clause_summary: dict) -> CoherenceFlag | None:
        self.calls.append(clause_summary)
        return None


def test_coherence_judge_called_for_low_n_our_paper() -> None:
    """Acceptance: judge is called for clause with n_our_paper < COHERENCE_MIN_CITATIONS."""
    # One our-paper observation — below threshold of 3.
    obs = [_obs("indemnification", provenance="our_paper")]
    judge = _FlagAllJudge()
    positions, flags, _ = compile_clause_positions(obs, [], coherence_judge=judge)
    assert len(judge.calls) == 1
    assert judge.calls[0]["clause_id"] == "clause.indemnification"
    assert judge.calls[0]["n_our_paper"] == 1


def test_coherence_flag_appears_in_result_for_low_n_our_paper() -> None:
    """Acceptance: CoherenceFlag for low n_our_paper clause appears in output."""
    obs = [_obs("indemnification", provenance="our_paper")]
    judge = _FlagAllJudge()
    positions, flags, _ = compile_clause_positions(obs, [], coherence_judge=judge)
    assert len(flags) == 1
    flag = flags[0]
    assert flag.clause_id == "clause.indemnification"
    assert flag.reason == "low citation count"
    assert flag.severity == "warn"


def test_coherence_judge_not_called_when_sufficient_citations_and_consistent_risk() -> None:
    """Acceptance: judge NOT called when all clauses have n_our_paper >= threshold and
    consistent risk_delta directions."""
    # COHERENCE_MIN_CITATIONS = 3; supply 3 our-paper observations with consistent neutral risk.
    obs = [
        _obs("indemnification", provenance="our_paper", doc_id="d1", version="v1"),
        _obs("indemnification", provenance="our_paper", doc_id="d2", version="v1"),
        _obs("indemnification", provenance="our_paper", doc_id="d3", version="v1"),
    ]
    judge = _NullJudge()
    positions, flags, _ = compile_clause_positions(obs, [], coherence_judge=judge)
    # All 3 observations have neutral risk and no fallbacks — no trigger conditions met.
    assert len(judge.calls) == 0
    assert len(flags) == 0


def test_coherence_judge_not_called_when_none() -> None:
    """No calls and no flags when coherence_judge=None (default)."""
    obs = [_obs("indemnification", provenance="our_paper")]
    positions, flags, _ = compile_clause_positions(obs, [], coherence_judge=None)
    assert flags == []


def test_coherence_flag_dataclass_fields() -> None:
    """CoherenceFlag carries clause_id, reason, and severity."""
    flag = CoherenceFlag(
        clause_id="clause.governing_law",
        reason="contradictory risk_delta directions",
        severity="block",
    )
    assert flag.clause_id == "clause.governing_law"
    assert flag.reason == "contradictory risk_delta directions"
    assert flag.severity == "block"


def test_coherence_min_citations_constant() -> None:
    """COHERENCE_MIN_CITATIONS is defined and equals 3 per spec."""
    assert COHERENCE_MIN_CITATIONS == 3


def test_coherence_judge_protocol_satisfied_by_stub() -> None:
    """Stub judges satisfy the CoherenceJudge protocol (runtime_checkable)."""
    assert isinstance(_FlagAllJudge(), CoherenceJudge)
    assert isinstance(_NullJudge(), CoherenceJudge)


# ---------------------------------------------------------------------------
# Unclassified coverage (issue #113)
# ---------------------------------------------------------------------------


def test_unclassified_coverage_counts_none_taxonomy_observations() -> None:
    """taxonomy_id=None observations are excluded from positions but counted,
    not silently omitted, in the returned unclassified coverage."""
    obs = [
        _obs(None, doc_id="deal_001", version="v1", clause_path="3"),
        _obs(None, doc_id="deal_002", version="v1", clause_path="7"),
        _obs("indemnification"),
    ]
    positions, _flags, unclassified = compile_clause_positions(obs, [])
    assert len(positions) == 1
    assert isinstance(unclassified, UnclassifiedCoverage)
    assert unclassified.count == 2


def test_unclassified_coverage_by_document_breakdown() -> None:
    """Per-document counts let a consumer see which documents lost content."""
    obs = [
        _obs(None, doc_id="deal_001", version="v1", clause_path="3"),
        _obs(None, doc_id="deal_001", version="v1", clause_path="5"),
        _obs(None, doc_id="deal_002", version="v1", clause_path="7"),
        _obs("indemnification"),
    ]
    _positions, _flags, unclassified = compile_clause_positions(obs, [])
    assert unclassified.by_document == {"deal_001": 2, "deal_002": 1}


def test_unclassified_coverage_example_citations_present() -> None:
    """Example citations are real OPFCitation objects pointing at the source."""
    obs = [
        _obs(None, doc_id="deal_001", version="v1", clause_path="3"),
        _obs("indemnification"),
    ]
    _positions, _flags, unclassified = compile_clause_positions(obs, [])
    assert len(unclassified.example_citations) == 1
    citation = unclassified.example_citations[0]
    assert isinstance(citation, OPFCitation)
    assert citation.document_id == "deal_001"
    assert citation.clause_path == "3"


def test_unclassified_coverage_example_citations_capped() -> None:
    """Example citations are capped at UNCLASSIFIED_EXAMPLE_LIMIT even when
    many more observations are unclassified."""
    obs = [
        _obs(None, doc_id=f"deal_{i:03d}", version="v1", clause_path=str(i))
        for i in range(UNCLASSIFIED_EXAMPLE_LIMIT + 5)
    ]
    _positions, _flags, unclassified = compile_clause_positions(obs, [])
    assert unclassified.count == UNCLASSIFIED_EXAMPLE_LIMIT + 5
    assert len(unclassified.example_citations) == UNCLASSIFIED_EXAMPLE_LIMIT


def test_unclassified_coverage_zero_when_all_classified() -> None:
    """No unclassified observations → count 0, empty breakdown/examples."""
    obs = [_obs("indemnification"), _obs("governing_law")]
    _positions, _flags, unclassified = compile_clause_positions(obs, [])
    assert unclassified.count == 0
    assert unclassified.by_document == {}
    assert unclassified.example_citations == ()


def test_unclassified_coverage_to_dict_shape() -> None:
    """to_dict() emits count/by_document/example_citations for JSON persistence."""
    obs = [_obs(None, doc_id="deal_001", version="v1", clause_path="3")]
    _positions, _flags, unclassified = compile_clause_positions(obs, [])
    d = unclassified.to_dict()
    assert d["count"] == 1
    assert d["by_document"] == {"deal_001": 1}
    assert len(d["example_citations"]) == 1
    assert d["example_citations"][0]["document_id"] == "deal_001"
