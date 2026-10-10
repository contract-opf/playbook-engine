"""Tests for the clause-type compiler (L5, issues #22 / #238).

The compiler decides which clause types a playbook carries and each one's
``our_standard``; the per-deal facts live in the precedent record
(``tests/test_playbook_assembler.py``). OPF 0.1-0.3's rollup, observed
positions and clause library were retired with those formats (issue #238).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional party/document names only
(e.g., "Alice", "Bob", "Acme Corp", "Beta LLC").
"""

from __future__ import annotations

import pytest

from playbook_engine.clause_differ import ClauseDiff
from playbook_engine.clause_position_compiler import (
    MIN_OBSERVATION_TEXT_LEN,
    UNCLASSIFIED_EXAMPLE_LIMIT,
    ClausePosition,
    OPFCitation,
    UnclassifiedCoverage,
    compile_clause_positions,
)
from playbook_engine.deviation_classifier import (
    DeviationResult,
    assess_deviations_deterministic,
)
from playbook_engine.observation_builder import (
    OUTCOME_CONCEDED_BEFORE_SIGNING,
    Observation,
    ObservationCitation,
    build_observations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEUTRAL = {"direction": "neutral", "magnitude": "none"}


def _obs(
    taxonomy_id: str | None,
    provenance: str = "our_paper",
    outcome: str = "signed",
    text: str = "Mutual indemnification clause language.",
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
        deviation="none",
        risk_delta=dict(_NEUTRAL),
        provenance=provenance,
        outcome=outcome,
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
        text=text,
        doc_id="template",
        version="template",
        clause_path=clause_path,
    )


def _compile(
    observations: list[Observation],
    template_observations: list[Observation],
    taxonomy_titles: dict[str, str] | None = None,
) -> list[ClausePosition]:
    """Call compile_clause_positions and return only the positions list."""
    positions, _, _ = compile_clause_positions(
        observations,
        template_observations,
        taxonomy_titles=taxonomy_titles,
    )
    return positions


# ---------------------------------------------------------------------------
# Which clause types appear
# ---------------------------------------------------------------------------


def test_compile_returns_one_position_per_taxonomy_id() -> None:
    obs = [
        _obs("indemnification"),
        _obs("governing_law", doc_id="deal_001", version="v2", clause_path="12"),
    ]
    positions = _compile(obs, [])
    assert {p.taxonomy_id for p in positions} == {"indemnification", "governing_law"}
    assert len(positions) == 2


def test_compile_none_taxonomy_id_skipped() -> None:
    """Unclassified observations never become a clause type (they are counted
    in the unclassified coverage instead — see below)."""
    positions = _compile([_obs(None), _obs("indemnification")], [])
    assert [p.taxonomy_id for p in positions] == ["indemnification"]


def test_compile_sorted_by_taxonomy_id() -> None:
    obs = [_obs("governing_law"), _obs("indemnification"), _obs("confidentiality")]
    tids = [p.taxonomy_id for p in _compile(obs, [])]
    assert tids == sorted(tids)


def test_compile_includes_template_only_taxonomy_ids() -> None:
    positions = _compile([], [_template_obs("limitation_of_liability")])
    assert [p.taxonomy_id for p in positions] == ["limitation_of_liability"]


@pytest.mark.parametrize(
    "outcome", ["signed", "proposed_then_reversed", OUTCOME_CONCEDED_BEFORE_SIGNING]
)
def test_clause_defining_outcomes_make_a_clause_type_appear(outcome: str) -> None:
    """A signed text, a refused ask and our standard struck before signing
    each make their clause type appear on their own."""
    positions = _compile([_obs("non_solicit", outcome=outcome)], [])
    assert [p.taxonomy_id for p in positions] == ["non_solicit"]


def test_unsigned_only_clause_type_does_not_appear() -> None:
    """An ``unsigned`` row (a deal with no detected executed copy, issue #83)
    never makes a clause type appear on its own — the template or a signed
    deal must."""
    assert _compile([_obs("non_solicit", outcome="unsigned")], []) == []
    positions = _compile([_obs("non_solicit", outcome="unsigned")], [_template_obs("non_solicit")])
    assert [p.taxonomy_id for p in positions] == ["non_solicit"]


def test_compile_id_format() -> None:
    assert _compile([_obs("indemnification")], [])[0].id == "clause.indemnification"


def test_compile_title_derived_from_taxonomy_id() -> None:
    assert _compile([_obs("limitation_of_liability")], [])[0].title == "Limitation Of Liability"


def test_compile_title_overridden_by_taxonomy_titles() -> None:
    positions = _compile(
        [_obs("indemnification")],
        [],
        taxonomy_titles={"indemnification": "Indemnification & Defense"},
    )
    assert positions[0].title == "Indemnification & Defense"


# ---------------------------------------------------------------------------
# our_standard comes only from the template
# ---------------------------------------------------------------------------


def test_our_standard_set_from_template_observation() -> None:
    t_obs = _template_obs("indemnification", text="Mutual indemnification.", clause_path="8")
    pos = _compile([_obs("indemnification")], [t_obs])[0]
    assert pos.our_standard is not None
    assert pos.our_standard.text == "Mutual indemnification."
    assert pos.our_standard.source_ref.document_id == "template"
    assert pos.our_standard.source_ref.version == "template"
    assert pos.our_standard.source_ref.clause_path == "8"


def test_our_standard_absent_when_template_text_empty() -> None:
    """An empty-text template observation yields our_standard=None (issue #182)."""
    empty_t_obs = Observation(
        observation_id="template/template/8",
        taxonomy_id="indemnification",
        text_summary="",
        full_text="",
        citation=ObservationCitation(
            document_id="template", version="template", clause_path="8", char_span=None
        ),
        deviation="none",
        risk_delta=dict(_NEUTRAL),
        provenance="our_paper",
        outcome="signed",
    )
    positions = _compile([_obs("indemnification", provenance="counterparty_paper")], [empty_t_obs])
    assert len(positions) == 1
    assert positions[0].our_standard is None


def test_our_standard_carries_full_text() -> None:
    """our_standard.text is the untruncated clause text (issue #105)."""
    long_text = "Each party shall indemnify the other against claims. " * 5
    t_obs = Observation(
        observation_id="template/template/8",
        taxonomy_id="indemnification",
        text_summary=long_text[:200],
        full_text=long_text,
        citation=ObservationCitation(
            document_id="template", version="template", clause_path="8", char_span=None
        ),
        deviation="none",
        risk_delta=dict(_NEUTRAL),
        provenance="our_paper",
        outcome="signed",
    )
    pos = _compile([], [t_obs])[0]
    assert pos.our_standard is not None
    assert pos.our_standard.text == long_text


@pytest.mark.parametrize("provenance", ["our_paper", "counterparty_paper", "unknown"])
def test_our_standard_never_comes_from_a_deal(provenance: str) -> None:
    """No template clause, no our_standard — whatever paper the deals sit on."""
    obs = [_obs("indemnification", provenance=provenance, doc_id=f"deal_{i}") for i in range(3)]
    assert _compile(obs, [])[0].our_standard is None


def test_template_observation_must_be_our_paper() -> None:
    bad_template = _obs(
        "indemnification", provenance="counterparty_paper", doc_id="template", version="template"
    )
    with pytest.raises(ValueError, match="provenance='our_paper'"):
        compile_clause_positions([], [bad_template])


def test_multiple_template_observations_are_joined_in_document_order() -> None:
    """Issue #242: a clause split across template nodes is ONE standard, the
    nodes joined in document order (the first node alone was a lead-in)."""
    t1 = _template_obs("indemnification", text="First version.", clause_path="8")
    t2 = _template_obs("indemnification", text="Second version.", clause_path="9")
    pos = _compile([], [t1, t2])[0]
    assert pos.our_standard is not None
    assert pos.our_standard.text == "First version.\n\nSecond version."
    assert pos.our_standard.source_ref.clause_path == "8, 9"


def test_empty_inputs_returns_empty() -> None:
    positions, flags, _ = compile_clause_positions([], [])
    assert positions == []
    assert flags == []


def test_opf_citation_to_dict_no_optional_fields() -> None:
    d = OPFCitation(document_id="template", version="template").to_dict()
    assert "clause_path" not in d
    assert "char_span" not in d


def test_opf_citation_to_dict_with_all_fields() -> None:
    d = OPFCitation(
        document_id="deal_x", version=3, clause_path="8.1", char_span=(0, 120)
    ).to_dict()
    assert d["version"] == 3
    assert d["clause_path"] == "8.1"
    assert d["char_span"] == [0, 120]


def test_opf_citation_normalizes_string_versions() -> None:
    assert OPFCitation(document_id="d", version="v2").to_dict()["version"] == 2
    assert OPFCitation(document_id="d", version="template").to_dict()["version"] == "template"


# ---------------------------------------------------------------------------
# Unclassified coverage (issue #113)
# ---------------------------------------------------------------------------


def test_unclassified_coverage_counts_none_taxonomy_observations() -> None:
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
    obs = [
        _obs(None, doc_id="deal_001", version="v1", clause_path="3"),
        _obs(None, doc_id="deal_001", version="v1", clause_path="5"),
        _obs(None, doc_id="deal_002", version="v1", clause_path="7"),
        _obs("indemnification"),
    ]
    _positions, _flags, unclassified = compile_clause_positions(obs, [])
    assert unclassified.by_document == {"deal_001": 2, "deal_002": 1}


def test_unclassified_coverage_example_citations_capped() -> None:
    obs = [
        _obs(None, doc_id=f"deal_{i:03d}", version="v1", clause_path=str(i))
        for i in range(UNCLASSIFIED_EXAMPLE_LIMIT + 5)
    ]
    _positions, _flags, unclassified = compile_clause_positions(obs, [])
    assert unclassified.count == UNCLASSIFIED_EXAMPLE_LIMIT + 5
    assert len(unclassified.example_citations) == UNCLASSIFIED_EXAMPLE_LIMIT
    assert isinstance(unclassified.example_citations[0], OPFCitation)


def test_unclassified_coverage_zero_when_all_classified() -> None:
    _positions, _flags, unclassified = compile_clause_positions(
        [_obs("indemnification"), _obs("governing_law")], []
    )
    assert unclassified.count == 0
    assert unclassified.by_document == {}
    assert unclassified.example_citations == ()


def test_unclassified_coverage_to_dict_shape() -> None:
    _positions, _flags, unclassified = compile_clause_positions(
        [_obs(None, doc_id="deal_001", version="v1", clause_path="3")], []
    )
    d = unclassified.to_dict()
    assert d["count"] == 1
    assert d["by_document"] == {"deal_001": 1}
    assert d["example_citations"][0]["document_id"] == "deal_001"


# ---------------------------------------------------------------------------
# Minimum-viable-observation guard (issue #210)
# ---------------------------------------------------------------------------


def test_degenerate_fragment_is_flagged_beside_real_text() -> None:
    real = _obs("indemnification", text="Mutual indemnification for third-party claims.")
    fragment = _obs("indemnification", text="1 6", doc_id="deal_002", version="v1", clause_path="3")
    positions, flags, _ = compile_clause_positions([real, fragment], [])
    assert [p.taxonomy_id for p in positions] == ["indemnification"]
    assert [(f.clause_id, f.severity) for f in flags] == [("clause.indemnification", "warn")]
    assert flags[0].reason.startswith("1 observation(s) excluded from precedent")


def test_taxonomy_id_with_only_fragments_has_no_position_but_is_flagged() -> None:
    positions, flags, _ = compile_clause_positions([_obs("amendments", text="1 7")], [])
    assert positions == []
    assert [f.clause_id for f in flags] == ["clause.amendments"]


def test_exactly_min_length_text_is_not_degenerate() -> None:
    obs = _obs("governing_law", text="x" * MIN_OBSERVATION_TEXT_LEN)
    positions, flags, _ = compile_clause_positions([obs], [])
    assert [p.taxonomy_id for p in positions] == ["governing_law"]
    assert flags == []


# ---------------------------------------------------------------------------
# The consumer path's observations — driven through the real producers
# (deviation_classifier.assess_deviations_deterministic -> build_observations
# (deterministic_deviations=True)), exactly what mine_corpus writes with no
# deviation judge configured (its default).
# ---------------------------------------------------------------------------

_STD_NON_SOLICIT = "For twelve months neither party shall solicit the other's employees."
_THEIR_NON_SOLICIT = "For thirty-six months neither party shall hire any of the other's staff."
_STD_ONLY = {"non_solicit": _STD_NON_SOLICIT}


def _removed(taxonomy_id: str, text: str, path: str = "9") -> ClauseDiff:
    return ClauseDiff(
        taxonomy_id=taxonomy_id,
        clause_path_before=path,
        clause_path_after=None,
        kind="removed",
        hunks=(),
        text_before=text,
        text_after="",
        clause_version_before="v1",
        clause_version_after=None,
        char_span_before=(0, len(text)),
    )


def _signed(taxonomy_id: str, text: str, path: str = "9", kind: str = "added") -> ClauseDiff:
    return ClauseDiff(
        taxonomy_id=taxonomy_id,
        clause_path_before=None if kind == "added" else path,
        clause_path_after=path,
        kind=kind,
        hunks=(),
        text_before="" if kind == "added" else text,
        text_after=text,
        clause_version_before=None if kind == "added" else "v1",
        clause_version_after="v3",
        char_span_after=(0, len(text)),
    )


def _consumer_deal(
    doc_id: str, diffs: list[ClauseDiff], provenance: str = "our_paper"
) -> list[Observation]:
    rows: list[tuple[ClauseDiff, DeviationResult]] = assess_deviations_deterministic(
        diffs, _STD_NON_SOLICIT
    )
    return build_observations(
        doc_id,
        3,
        provenance,
        rows,
        [],
        ordinal_by_vid={"v1": 1, "v2": 2, "v3": 3},
        standard_text_by_tid=_STD_ONLY,
    )


def _consumer_corpus() -> list[Observation]:
    return [
        *_consumer_deal("deal-a", [_signed("non_solicit", _STD_NON_SOLICIT, kind="unchanged")]),
        *_consumer_deal(
            "deal-b",
            [_signed("non_solicit", _STD_NON_SOLICIT, kind="unchanged")],
            provenance="counterparty_paper",
        ),
        *_consumer_deal("deal-c", [_signed("non_solicit", _THEIR_NON_SOLICIT)]),
        *_consumer_deal("deal-d", [_removed("non_solicit", _STD_NON_SOLICIT)]),
        *_consumer_deal(
            "deal-e",
            [
                _removed("non_solicit", _THEIR_NON_SOLICIT, path="4"),
                _signed("non_solicit", _STD_NON_SOLICIT, kind="unchanged"),
            ],
        ),
    ]


def test_consumer_path_observations_carry_the_standard_fact() -> None:
    obs = _consumer_corpus()
    for o in obs:
        assert o.basis == "deterministic"
        assert isinstance(o.standard, bool)
        assert o.deviation == ("none" if o.standard else "substantive")
        assert o.risk_delta == {"direction": "neutral", "magnitude": "none"}
    by_outcome = {(o.citation.document_id, o.outcome): o.standard for o in obs}
    assert by_outcome[("deal-c", "signed")] is False
    assert by_outcome[("deal-d", OUTCOME_CONCEDED_BEFORE_SIGNING)] is True
    assert by_outcome[("deal-e", "proposed_then_reversed")] is False
    assert by_outcome[("deal-e", "signed")] is True


def test_consumer_path_corpus_compiles_one_clause_type_with_the_template_standard() -> None:
    positions = _compile(_consumer_corpus(), [_template_obs("non_solicit", text=_STD_NON_SOLICIT)])
    assert [p.taxonomy_id for p in positions] == ["non_solicit"]
    assert positions[0].our_standard is not None
    assert positions[0].our_standard.text == _STD_NON_SOLICIT


def test_consumer_path_multi_node_clause_is_standard_as_a_whole() -> None:
    """A deal that splits our one-clause standard across two nodes signed our
    standard: the terminal observation's standard fact is its MERGED text
    against the whole template clause, not each node alone."""
    first = "For twelve months neither party shall solicit"
    second = "the other's employees."
    rows = assess_deviations_deterministic(
        [
            _signed("non_solicit", first, path="9", kind="unchanged"),
            _signed("non_solicit", second, path="10", kind="unchanged"),
        ],
        _STD_NON_SOLICIT,
    )
    assert [dr.deviation for _, dr in rows] == ["substantive", "substantive"]
    obs = build_observations(
        "deal-g",
        3,
        "our_paper",
        rows,
        [],
        standard_text_by_tid=_STD_ONLY,
    )
    assert len(obs) == 1
    assert obs[0].standard is True
    assert obs[0].deviation == "none"
