"""Tests for rollup derivation: acceptable_if, fallbacks ordering, confidence.

Issue #24: populates acceptable_if from neutral-risk signed variants, orders
fallbacks least→most costly, and applies a provenance-weighted confidence score.

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional party/document names only.
"""

from __future__ import annotations

from playbook_engine.clause_position_compiler import compile_clause_positions
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
    taxonomy_id: str,
    provenance: str = "our_paper",
    outcome: str = "signed",
    deviation: str = "none",
    risk_delta: RiskDelta = _NEUTRAL,
    text: str = "Standard clause text.",
    doc_id: str = "deal_001",
    version: str = "v2",
    clause_path: str = "8",
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
    )


def _template_obs(taxonomy_id: str) -> Observation:
    return _obs(taxonomy_id, doc_id="template", version="template")


def _positions(
    obs_list: list[Observation], with_template: bool = True, min_evidence_n: int | None = None
) -> list:
    template = [_template_obs(obs_list[0].taxonomy_id)] if with_template and obs_list else []
    kwargs = {} if min_evidence_n is None else {"min_evidence_n": min_evidence_n}
    positions, _, _ = compile_clause_positions(obs_list, template, **kwargs)
    return positions


# ---------------------------------------------------------------------------
# acceptable_if: populated from neutral-risk signed variants
# ---------------------------------------------------------------------------


def test_acceptable_if_empty_when_no_neutral_variants() -> None:
    """No neutral-risk signed variants → acceptable_if is empty."""
    obs = [_obs("ind", deviation="none", risk_delta=_NEUTRAL, outcome="signed")]
    positions = _positions(obs)
    assert positions[0].rollup.acceptable_if == ()


def test_acceptable_if_populated_from_neutral_signed_variants() -> None:
    """Acceptance: neutral-risk signed variant (deviation != none) → appears in acceptable_if."""
    obs = [
        _obs(
            "ind",
            deviation="reworded_equivalent",
            risk_delta=_NEUTRAL,
            outcome="signed",
            text="Mutual indemnification, reworded but equivalent.",
        ),
    ]
    positions = _positions(obs)
    af = positions[0].rollup.acceptable_if
    assert len(af) == 1
    assert af[0].to == "Mutual indemnification, reworded but equivalent."


def test_acceptable_if_deviation_none_excluded() -> None:
    """deviation=none signed observation is NOT an acceptable variant (it IS the standard)."""
    obs = [
        _obs(
            "ind",
            deviation="none",
            risk_delta=_NEUTRAL,
            outcome="signed",
            text="Exact standard clause text.",
        ),
    ]
    positions = _positions(obs)
    # deviation=none + neutral risk = standard, not a variant to flag as acceptable_if
    af_texts = [entry.to for entry in positions[0].rollup.acceptable_if]
    assert "Exact standard clause text." not in af_texts


def test_acceptable_if_worse_risk_excluded() -> None:
    """Worse-risk signed observations go to fallbacks, not acceptable_if."""
    obs = [
        _obs(
            "ind",
            deviation="substantive",
            risk_delta=_WORSE_MINOR,
            outcome="signed",
            text="Gross-negligence only — worse risk.",
        ),
    ]
    positions = _positions(obs)
    af_texts = [entry.to for entry in positions[0].rollup.acceptable_if]
    assert "Gross-negligence only — worse risk." not in af_texts


def test_acceptable_if_proposed_then_reversed_excluded() -> None:
    """proposed_then_reversed observations go to rejected, not acceptable_if."""
    obs = [
        _obs(
            "ind",
            deviation="substantive",
            risk_delta=_NEUTRAL,
            outcome="proposed_then_reversed",
            text="Proposed variant, rejected.",
        ),
    ]
    positions = _positions(obs)
    af_texts = [entry.to for entry in positions[0].rollup.acceptable_if]
    assert "Proposed variant, rejected." not in af_texts


def test_acceptable_if_our_paper_before_counterparty_paper() -> None:
    """Our-paper neutral variants appear before counterparty-paper variants."""
    obs = [
        _obs(
            "ind",
            provenance="counterparty_paper",
            deviation="reworded_equivalent",
            risk_delta=_NEUTRAL,
            outcome="signed",
            text="Counterparty variant — neutral risk.",
            doc_id="deal_002",
            version="v1",
        ),
        _obs(
            "ind",
            provenance="our_paper",
            deviation="reworded_equivalent",
            risk_delta=_NEUTRAL,
            outcome="signed",
            text="Our-paper variant — neutral risk.",
        ),
    ]
    positions, _, _ = compile_clause_positions(obs, [])
    af = positions[0].rollup.acceptable_if
    assert len(af) == 2
    # Our-paper comes first
    assert af[0].to == "Our-paper variant — neutral risk."
    assert af[1].to == "Counterparty variant — neutral risk."


def test_acceptable_if_counterparty_paper_included() -> None:
    """§2.2 Appendix B: counterparty-paper observations MAY inform acceptable_if."""
    obs = [
        _obs(
            "ind",
            provenance="counterparty_paper",
            deviation="reworded_equivalent",
            risk_delta=_NEUTRAL,
            outcome="signed",
            text="Counterparty acceptable variant.",
        ),
    ]
    positions, _, _ = compile_clause_positions(obs, [])
    assert len(positions[0].rollup.acceptable_if) == 1


def test_acceptable_if_deduplicates_identical_text() -> None:
    """Identical text_summary from multiple observations appears once."""
    repeated_text = "Mutual indemnification, reworded but equivalent."
    obs = [
        _obs(
            "ind",
            deviation="reworded_equivalent",
            risk_delta=_NEUTRAL,
            outcome="signed",
            text=repeated_text,
        ),
        _obs(
            "ind",
            deviation="reworded_equivalent",
            risk_delta=_NEUTRAL,
            outcome="signed",
            text=repeated_text,
            doc_id="deal_002",
            version="v1",
        ),
    ]
    positions = _positions(obs)
    af = positions[0].rollup.acceptable_if
    assert sum(1 for entry in af if entry.to == repeated_text) == 1


def test_acceptable_if_multiple_distinct_variants() -> None:
    """Multiple distinct neutral-risk signed variants each appear in acceptable_if."""
    obs = [
        _obs(
            "ind",
            deviation="reworded_equivalent",
            risk_delta=_NEUTRAL,
            outcome="signed",
            text="Variant A — mutual indemnification reworded.",
        ),
        _obs(
            "ind",
            deviation="reworded_equivalent",
            risk_delta=_NEUTRAL,
            outcome="signed",
            text="Variant B — negligence-based reworded.",
            doc_id="deal_002",
            version="v1",
        ),
    ]
    positions = _positions(obs)
    af = positions[0].rollup.acceptable_if
    assert len(af) == 2
    af_texts = [entry.to for entry in af]
    assert "Variant A — mutual indemnification reworded." in af_texts
    assert "Variant B — negligence-based reworded." in af_texts


# ---------------------------------------------------------------------------
# fallbacks: ordered least→most costly (minor before material)
# ---------------------------------------------------------------------------


def test_fallbacks_ordered_minor_before_material() -> None:
    """Acceptance: fallbacks ordered least→most costly (minor first, then material)."""
    obs = [
        _obs(
            "ind",
            deviation="substantive",
            risk_delta=_WORSE_MATERIAL,
            outcome="signed",
            text="Material-risk concession.",
            doc_id="deal_001",
            version="v2",
        ),
        _obs(
            "ind",
            deviation="substantive",
            risk_delta=_WORSE_MINOR,
            outcome="signed",
            text="Minor-risk concession.",
            doc_id="deal_002",
            version="v1",
        ),
    ]
    positions = _positions(obs)
    fallbacks = positions[0].rollup.fallbacks
    assert len(fallbacks) == 2
    assert fallbacks[0].risk_delta["magnitude"] == "minor"
    assert fallbacks[1].risk_delta["magnitude"] == "material"


def test_fallbacks_single_entry_no_reordering() -> None:
    """Single fallback returns as-is."""
    obs = [
        _obs(
            "ind",
            deviation="substantive",
            risk_delta=_WORSE_MATERIAL,
            outcome="signed",
            text="Material-risk concession.",
        ),
    ]
    positions = _positions(obs)
    fallbacks = positions[0].rollup.fallbacks
    assert len(fallbacks) == 1
    assert fallbacks[0].risk_delta["magnitude"] == "material"


def test_fallbacks_only_our_paper_worse_risk_signed() -> None:
    """fallbacks = our-paper signed worse-risk only; counterparty-paper excluded."""
    obs = [
        _obs(
            "ind",
            provenance="counterparty_paper",
            deviation="substantive",
            risk_delta=_WORSE_MINOR,
            outcome="signed",
            text="CP worse signed.",
        ),
        _obs(
            "ind",
            provenance="our_paper",
            deviation="substantive",
            risk_delta=_WORSE_MINOR,
            outcome="signed",
            text="Our worse signed.",
            doc_id="deal_002",
            version="v1",
        ),
    ]
    positions, _, _ = compile_clause_positions(obs, [])
    fallbacks = positions[0].rollup.fallbacks
    # Only our-paper worse-risk appears in fallbacks.
    assert len(fallbacks) == 1
    assert fallbacks[0].text_summary == "Our worse signed."


def test_fallbacks_reversed_obs_excluded() -> None:
    """proposed_then_reversed obs with worse risk goes to rejected, not fallbacks."""
    obs = [
        _obs(
            "ind",
            deviation="substantive",
            risk_delta=_WORSE_MINOR,
            outcome="proposed_then_reversed",
            text="Rejected ask.",
        ),
    ]
    positions = _positions(obs)
    assert len(positions[0].rollup.fallbacks) == 0


# ---------------------------------------------------------------------------
# confidence: provenance-weighted score
# ---------------------------------------------------------------------------


def test_confidence_all_our_paper_score_is_one() -> None:
    """All our-paper → weighted = 1.0 * n_our / n_our = 1.0."""
    obs = [
        _obs("ind", provenance="our_paper"),
        _obs("ind", provenance="our_paper", doc_id="deal_002", version="v1"),
    ]
    positions = _positions(obs)
    assert positions[0].rollup.confidence["score"] == 1.0


def test_confidence_all_counterparty_paper_score_is_half() -> None:
    """All counterparty-paper → weighted = 0.5 * n_cp / n_cp = 0.5."""
    obs = [
        _obs("ind", provenance="counterparty_paper"),
        _obs("ind", provenance="counterparty_paper", doc_id="deal_002", version="v1"),
    ]
    positions, _, _ = compile_clause_positions(obs, [])
    score = positions[0].rollup.confidence["score"]
    assert score == 0.5


def test_confidence_mixed_provenance_between_half_and_one() -> None:
    """Mixed provenance: score is strictly between 0.5 and 1.0."""
    obs = [
        _obs("ind", provenance="our_paper"),
        _obs("ind", provenance="counterparty_paper", doc_id="deal_002", version="v1"),
    ]
    positions, _, _ = compile_clause_positions(obs, [])
    score = positions[0].rollup.confidence["score"]
    assert 0.5 < score < 1.0


def test_confidence_zero_for_no_observations() -> None:
    """No observations (template only) → score = 0.0."""
    positions, _, _ = compile_clause_positions([], [_template_obs("ind")])
    assert positions[0].rollup.confidence["score"] == 0.0


def test_confidence_score_range_always_valid() -> None:
    """Score is always in [0.0, 1.0]."""
    obs = [
        _obs("ind", provenance="our_paper"),
        _obs("ind", provenance="counterparty_paper", doc_id="deal_002", version="v1"),
        _obs("ind", provenance="counterparty_paper", doc_id="deal_003", version="v1"),
    ]
    positions, _, _ = compile_clause_positions(obs, [])
    score = positions[0].rollup.confidence["score"]
    assert 0.0 <= score <= 1.0


def test_confidence_basis_string_present() -> None:
    obs = [_obs("ind")]
    positions = _positions(obs)
    # basis describes what the score formula actually uses
    assert "provenance_mix" in positions[0].rollup.confidence["basis"]


def test_confidence_counts_preserved() -> None:
    obs = [
        _obs("ind", provenance="our_paper"),
        _obs("ind", provenance="our_paper", doc_id="deal_002", version="v1"),
        _obs("ind", provenance="counterparty_paper", doc_id="deal_003", version="v1"),
    ]
    positions, _, _ = compile_clause_positions(obs, [])
    conf = positions[0].rollup.confidence
    assert conf["n_our_paper"] == 2
    assert conf["n_counterparty_paper"] == 1


# ---------------------------------------------------------------------------
# better-risk signed observations (fall-through path)
# ---------------------------------------------------------------------------


def test_better_risk_signed_not_in_acceptable_if() -> None:
    """better-risk variant is NOT an acceptable_if entry (those require neutral risk)."""
    obs = [
        _obs(
            "ind",
            deviation="substantive",
            risk_delta=_BETTER,
            outcome="signed",
            text="Strictly better clause we achieved.",
        ),
    ]
    positions = _positions(obs)
    af_texts = [entry.to for entry in positions[0].rollup.acceptable_if]
    assert "Strictly better clause we achieved." not in af_texts


def test_better_risk_signed_not_in_fallbacks() -> None:
    """better-risk variant is NOT a fallback (fallbacks require worse risk)."""
    obs = [
        _obs(
            "ind",
            deviation="substantive",
            risk_delta=_BETTER,
            outcome="signed",
            text="Better than standard — not a concession.",
        ),
    ]
    positions = _positions(obs)
    assert len(positions[0].rollup.fallbacks) == 0


def test_better_risk_signed_appears_in_observed_positions() -> None:
    """better-risk variant DOES appear in observed_positions (it's a real observation)."""
    obs = [
        _obs(
            "ind",
            deviation="substantive",
            risk_delta=_BETTER,
            outcome="signed",
            text="Better outcome achieved.",
        ),
    ]
    positions = _positions(obs)
    op_texts = [op.text_summary for op in positions[0].observed_positions]
    assert "Better outcome achieved." in op_texts


# ---------------------------------------------------------------------------
# Issue #107: evidence-depth floor — sparse evidence must not yield strong
# positions, and identical precedents must aggregate into precedent_count.
# ---------------------------------------------------------------------------


def test_single_observation_does_not_yield_strong_position() -> None:
    """Acceptance (issue #107): a clause with exactly ONE our-paper observation
    is demoted to "negotiable" — deviation=none + neutral risk would otherwise
    cascade to "standard" (see test_position_standard_all_deviation_none in
    test_clause_position_compiler.py), but one agreement is one data point,
    not a pattern.
    """
    obs = [_obs("ind", deviation="none", risk_delta=_NEUTRAL, outcome="signed")]
    positions = _positions(obs)
    assert positions[0].rollup.position == "negotiable"
    assert positions[0].rollup.position not in {
        "standard",
        "acceptable_variants_exist",
        "hold_firm",
    }


def test_single_observation_confidence_no_longer_reports_evidence_as_sufficient() -> None:
    """Confidence.score is a provenance-QUALITY signal (1.0 for pure our-paper)
    and legitimately stays 1.0 even for n=1 — but it must no longer be read as
    "evidence depth". evidence_sufficient is the orthogonal depth signal, and
    it must be False when n_our_paper is below MIN_EVIDENCE_N.
    """
    obs = [_obs("ind", deviation="none", risk_delta=_NEUTRAL, outcome="signed")]
    positions = _positions(obs)
    conf = positions[0].rollup.confidence
    assert conf["score"] == 1.0
    assert conf["n_our_paper"] == 1
    assert conf["evidence_sufficient"] is False


def test_two_observations_meets_evidence_floor_yields_standard() -> None:
    """Two our-paper observations meet MIN_EVIDENCE_N — "standard" is reachable
    again once there is a second, independent data point."""
    obs = [
        _obs("ind", deviation="none", risk_delta=_NEUTRAL, outcome="signed"),
        _obs("ind", deviation="none", risk_delta=_NEUTRAL, outcome="signed", doc_id="deal_002"),
    ]
    positions = _positions(obs)
    assert positions[0].rollup.position == "standard"
    assert positions[0].rollup.confidence["evidence_sufficient"] is True


# ---------------------------------------------------------------------------
# Issue #144: min_evidence_n is producer-configurable (validator + compiler
# aligned on one rule) and the threshold actually applied is recorded in
# confidence.basis.
# ---------------------------------------------------------------------------


def test_one_our_paper_among_many_counterparty_does_not_meet_default_floor() -> None:
    """A single our-paper observation surrounded by counterparty_paper
    observations must not license a strong position at the default
    min_evidence_n=2 — mixed provenance does not, by itself, satisfy the
    evidence-depth floor."""
    obs = [
        _obs(
            "ind", provenance="our_paper", deviation="none", risk_delta=_NEUTRAL, outcome="signed"
        ),
        _obs(
            "ind",
            provenance="counterparty_paper",
            deviation="none",
            risk_delta=_NEUTRAL,
            outcome="signed",
            doc_id="deal_002",
        ),
        _obs(
            "ind",
            provenance="counterparty_paper",
            deviation="none",
            risk_delta=_NEUTRAL,
            outcome="signed",
            doc_id="deal_003",
        ),
        _obs(
            "ind",
            provenance="counterparty_paper",
            deviation="none",
            risk_delta=_NEUTRAL,
            outcome="signed",
            doc_id="deal_004",
        ),
    ]
    positions = _positions(obs)
    assert positions[0].rollup.position == "negotiable"
    assert positions[0].rollup.confidence["evidence_sufficient"] is False
    assert positions[0].rollup.confidence["n_our_paper"] == 1


def test_min_evidence_n_is_producer_configurable() -> None:
    """A caller-supplied min_evidence_n overrides the module default — one
    our-paper observation meets a custom floor of 1, reaching "standard"."""
    obs = [_obs("ind", deviation="none", risk_delta=_NEUTRAL, outcome="signed")]
    positions = _positions(obs, min_evidence_n=1)
    assert positions[0].rollup.position == "standard"
    assert positions[0].rollup.confidence["evidence_sufficient"] is True


def test_min_evidence_n_configured_higher_demotes_two_observations() -> None:
    """Two our-paper observations meet the default floor (2) but not a
    producer-configured higher floor (3) — the position is demoted and
    evidence_sufficient reflects the configured threshold, not the default."""
    obs = [
        _obs("ind", deviation="none", risk_delta=_NEUTRAL, outcome="signed"),
        _obs("ind", deviation="none", risk_delta=_NEUTRAL, outcome="signed", doc_id="deal_002"),
    ]
    positions = _positions(obs, min_evidence_n=3)
    assert positions[0].rollup.position == "negotiable"
    assert positions[0].rollup.confidence["evidence_sufficient"] is False


def test_confidence_basis_records_the_min_evidence_n_threshold_applied() -> None:
    """Issue #144: the min_evidence_n threshold actually enforced must be
    recorded in confidence.basis, so a consumer reading one playbook document
    (without the producer's config) can see what N was applied."""
    obs = [_obs("ind", deviation="none", risk_delta=_NEUTRAL, outcome="signed")]
    positions = _positions(obs, min_evidence_n=5)
    assert "min_evidence_n=5" in positions[0].rollup.confidence["basis"]


def test_precedent_count_aggregates_identical_text() -> None:
    """Acceptance (issue #107): identical clause text observed across multiple
    deals is aggregated into precedent_count — it must not stay at the
    dataclass default of 1 for every observation."""
    repeated_text = "Mutual indemnification, standard language."
    obs = [
        _obs("ind", text=repeated_text, doc_id="deal_001"),
        _obs("ind", text=repeated_text, doc_id="deal_002", version="v1"),
        _obs("ind", text=repeated_text, doc_id="deal_003", version="v1"),
    ]
    positions = _positions(obs)
    op = positions[0].observed_positions[0]
    assert op.precedent_count == 3


def test_precedent_count_distinct_texts_not_merged() -> None:
    """Distinct clause texts each keep their own precedent_count=1 — dedup is
    by normalized text, not a blanket count of the whole group."""
    obs = [
        _obs("ind", text="Variant A.", doc_id="deal_001"),
        _obs("ind", text="Variant B.", doc_id="deal_002", version="v1"),
    ]
    positions = _positions(obs)
    counts = {op.text_summary: op.precedent_count for op in positions[0].observed_positions}
    assert counts["Variant A."] == 1
    assert counts["Variant B."] == 1


def test_precedent_count_ignores_whitespace_and_case_differences() -> None:
    """Normalization collapses whitespace/case so line-wrap artifacts from
    extraction don't fragment an otherwise-identical precedent."""
    obs = [
        _obs("ind", text="Mutual   Indemnification.", doc_id="deal_001"),
        _obs("ind", text="mutual indemnification.", doc_id="deal_002", version="v1"),
    ]
    positions = _positions(obs)
    for op in positions[0].observed_positions:
        assert op.precedent_count == 2


def test_better_risk_signed_yields_standard_position() -> None:
    """A strictly-better signed variant (no fallbacks, no rejections) → position='standard'.

    Two our-paper observations (issue #107 evidence-depth floor).
    """
    obs = [
        _obs(
            "ind",
            deviation="substantive",
            risk_delta=_BETTER,
            outcome="signed",
            text="More favourable variant accepted.",
        ),
        _obs(
            "ind",
            deviation="substantive",
            risk_delta=_BETTER,
            outcome="signed",
            text="More favourable variant accepted.",
            doc_id="deal_002",
            version="v1",
        ),
    ]
    positions = _positions(obs)
    assert positions[0].rollup.position == "standard"
