"""Tests for the clause-library compiler (L5, issue #23).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional party/document names only
(e.g., "Alice", "Bob", "Acme Corp", "Beta LLC").
"""

from __future__ import annotations

import pytest

from playbook_engine.clause_library_compiler import (
    AcceptedForm,
    ClauseConcept,
    compile_clause_library,
)
from playbook_engine.deviation_classifier import RiskDelta
from playbook_engine.observation_builder import Observation, ObservationCitation

# ---------------------------------------------------------------------------
# Thin wrapper: unpack (concepts, unclassified_coverage) so existing tests
# that only care about `concepts` stay unchanged.
# ---------------------------------------------------------------------------


def _compile(
    observations: list,
    taxonomy_metadata: dict | None = None,
) -> list[ClauseConcept]:
    """Call compile_clause_library and return only the concepts list."""
    concepts, _ = compile_clause_library(observations, taxonomy_metadata)
    return concepts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEUTRAL = RiskDelta(direction="neutral", magnitude="none")
_WORSE_MINOR = RiskDelta(direction="worse", magnitude="minor")
_WORSE_MATERIAL = RiskDelta(direction="worse", magnitude="material")


def _obs(
    taxonomy_id: str | None,
    provenance: str = "counterparty_paper",
    outcome: str = "signed",
    deviation: str = "reworded_equivalent",
    risk_delta: RiskDelta = _NEUTRAL,
    text: str = "Counterparty governing law clause.",
    doc_id: str = "deal_001",
    version: str = "v2",
    clause_path: str = "12",
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


# ---------------------------------------------------------------------------
# compile_clause_library: basic structure
# ---------------------------------------------------------------------------


def test_compile_returns_one_concept_per_taxonomy_id() -> None:
    """Acceptance: one ClauseConcept per distinct taxonomy_id with signed obs."""
    obs = [
        _obs("governing_law"),
        _obs("indemnification", clause_path="8"),
    ]
    concepts = _compile(obs)
    assert len(concepts) == 2
    tids = {c.taxonomy_id for c in concepts}
    assert tids == {"governing_law", "indemnification"}


def test_compile_sorted_by_taxonomy_id() -> None:
    obs = [
        _obs("governing_law"),
        _obs("indemnification", clause_path="8"),
        _obs("confidentiality", clause_path="5"),
    ]
    concepts = _compile(obs)
    tids = [c.taxonomy_id for c in concepts]
    assert tids == sorted(tids)


def test_compile_none_taxonomy_id_skipped() -> None:
    obs = [_obs(None), _obs("governing_law")]
    concepts = _compile(obs)
    assert len(concepts) == 1
    assert concepts[0].taxonomy_id == "governing_law"


def test_compile_proposed_then_reversed_excluded() -> None:
    """Acceptance: counterparty-paper-tolerated clauses appear even when others excluded."""
    obs = [
        _obs("governing_law", outcome="signed"),
        _obs("governing_law", outcome="proposed_then_reversed", doc_id="deal_002", version="v1"),
    ]
    concepts = _compile(obs)
    # Only one concept (governing_law), and only the signed observation is an accepted_form.
    assert len(concepts) == 1
    assert len(concepts[0].accepted_forms) == 1
    assert concepts[0].accepted_forms[0].example_ref.document_id == "deal_001"


def test_compile_taxonomy_with_only_reversed_excluded() -> None:
    """A taxonomy_id with only proposed_then_reversed observations produces no concept."""
    obs = [_obs("governing_law", outcome="proposed_then_reversed")]
    concepts = _compile(obs)
    assert len(concepts) == 0


def test_compile_empty_input() -> None:
    assert _compile([]) == []


# ---------------------------------------------------------------------------
# concept_id and taxonomy_id
# ---------------------------------------------------------------------------


def test_concept_id_format() -> None:
    concepts = _compile([_obs("governing_law")])
    assert concepts[0].concept_id == "concept.governing_law"


def test_taxonomy_id_preserved() -> None:
    concepts = _compile([_obs("limitation_of_liability", clause_path="10")])
    assert concepts[0].taxonomy_id == "limitation_of_liability"


# ---------------------------------------------------------------------------
# description: default vs taxonomy_metadata override
# ---------------------------------------------------------------------------


def test_description_default_from_taxonomy_id() -> None:
    """Default description is a sentence built from the taxonomy_id."""
    concepts = _compile([_obs("governing_law")])
    assert "governing law" in concepts[0].description.lower()


def test_description_overridden_by_metadata() -> None:
    meta = {
        "governing_law": {"description": "Which state's law governs and where disputes are heard."}
    }
    concepts = _compile([_obs("governing_law")], taxonomy_metadata=meta)
    assert concepts[0].description == "Which state's law governs and where disputes are heard."


def test_risk_profile_default_none() -> None:
    concepts = _compile([_obs("governing_law")])
    assert concepts[0].risk_profile is None


def test_risk_profile_from_metadata() -> None:
    meta = {"governing_law": {"risk_profile": "Venue/forum exposure; cost of litigating."}}
    concepts = _compile([_obs("governing_law")], taxonomy_metadata=meta)
    assert concepts[0].risk_profile == "Venue/forum exposure; cost of litigating."


# ---------------------------------------------------------------------------
# accepted_forms: citations carried on every form
# ---------------------------------------------------------------------------


def test_accepted_forms_count_matches_signed_obs() -> None:
    obs = [
        _obs("governing_law", doc_id="deal_001", version="v2"),
        _obs("governing_law", doc_id="deal_002", version="v1"),
    ]
    concepts = _compile(obs)
    assert len(concepts[0].accepted_forms) == 2


def test_accepted_forms_citation_carried() -> None:
    """Acceptance: citations present on every accepted form."""
    obs = [_obs("governing_law", doc_id="deal_007", version="v3", clause_path="12")]
    concepts = _compile(obs)
    af = concepts[0].accepted_forms[0]
    assert af.example_ref.document_id == "deal_007"
    assert af.example_ref.version == "v3"
    assert af.example_ref.clause_path == "12"


def test_accepted_forms_text_summary_preserved() -> None:
    obs = [_obs("governing_law", text="Beta LLC home-state law governs.")]
    concepts = _compile(obs)
    assert concepts[0].accepted_forms[0].text_summary == "Beta LLC home-state law governs."


def test_accepted_forms_provenance_preserved() -> None:
    obs = [_obs("governing_law", provenance="counterparty_paper")]
    concepts = _compile(obs)
    assert concepts[0].accepted_forms[0].provenance == "counterparty_paper"


def test_accepted_forms_our_paper_included() -> None:
    """Our-paper signed observations may also appear as accepted forms."""
    obs = [_obs("governing_law", provenance="our_paper", deviation="none", risk_delta=_NEUTRAL)]
    concepts = _compile(obs)
    assert len(concepts[0].accepted_forms) == 1
    assert concepts[0].accepted_forms[0].provenance == "our_paper"


def test_accepted_forms_risk_delta_carried() -> None:
    obs = [_obs("governing_law", risk_delta=_WORSE_MINOR)]
    concepts = _compile(obs)
    af = concepts[0].accepted_forms[0]
    assert af.risk_delta_vs_our_standard is not None
    assert af.risk_delta_vs_our_standard["direction"] == "worse"
    assert af.risk_delta_vs_our_standard["magnitude"] == "minor"


# ---------------------------------------------------------------------------
# Acceptance criterion: counterparty-paper-tolerated clauses appear in library
# even when they would be excluded from opening positions
# ---------------------------------------------------------------------------


def test_counterparty_paper_tolerated_appears_in_library() -> None:
    """Acceptance: counterparty-paper observations appear in accepted_forms.

    This is the key scenario: a clause the counterparty drafted and we signed
    belongs in the library even though it cannot set an opening position (§2.2).
    """
    obs = [
        _obs(
            "governing_law",
            provenance="counterparty_paper",
            outcome="signed",
            risk_delta=_WORSE_MINOR,
            text="Acme Corp's home-state law governs.",
        ),
    ]
    concepts = _compile(obs)
    assert len(concepts) == 1
    assert len(concepts[0].accepted_forms) == 1
    af = concepts[0].accepted_forms[0]
    assert af.provenance == "counterparty_paper"
    assert af.risk_delta_vs_our_standard is not None
    assert af.risk_delta_vs_our_standard["direction"] == "worse"


def test_multiple_counterparty_paper_deals_all_appear() -> None:
    """All counterparty-paper tolerated clauses across deals appear in the library."""
    obs = [
        _obs(
            "limitation_of_liability",
            provenance="counterparty_paper",
            doc_id="deal_001",
            version="v2",
            clause_path="10",
            text="Liability cap two times contract value.",
        ),
        _obs(
            "limitation_of_liability",
            provenance="counterparty_paper",
            doc_id="deal_002",
            version="v1",
            clause_path="10",
            text="Liability cap one times contract value.",
        ),
        _obs(
            "limitation_of_liability",
            provenance="counterparty_paper",
            doc_id="deal_003",
            version="v1",
            clause_path="10",
            text="No liability cap for gross negligence.",
        ),
    ]
    concepts = _compile(obs)
    assert len(concepts) == 1
    assert len(concepts[0].accepted_forms) == 3


# ---------------------------------------------------------------------------
# notes auto-generation
# ---------------------------------------------------------------------------


def test_notes_includes_counterparty_paper_count() -> None:
    obs = [
        _obs("governing_law", provenance="counterparty_paper"),
        _obs("governing_law", provenance="counterparty_paper", doc_id="deal_002", version="v1"),
    ]
    concepts = _compile(obs)
    assert concepts[0].notes is not None
    assert "2" in concepts[0].notes
    assert "counterparty" in concepts[0].notes.lower()


def test_notes_none_when_no_counterparty_paper() -> None:
    """No notes generated when all accepted forms are our-paper."""
    obs = [_obs("governing_law", provenance="our_paper")]
    concepts = _compile(obs)
    assert concepts[0].notes is None


# ---------------------------------------------------------------------------
# to_dict: OPF-shaped output
# ---------------------------------------------------------------------------


def test_to_dict_required_keys() -> None:
    concepts = _compile([_obs("governing_law")])
    d = concepts[0].to_dict()
    for key in ("concept_id", "taxonomy_id", "description", "accepted_forms"):
        assert key in d


def test_to_dict_optional_fields_omitted_when_none() -> None:
    concepts = _compile([_obs("governing_law", provenance="our_paper")])
    d = concepts[0].to_dict()
    # risk_profile is None → must not appear; notes is None → must not appear.
    assert "risk_profile" not in d
    assert "notes" not in d


def test_to_dict_optional_fields_present_when_set() -> None:
    meta = {"governing_law": {"risk_profile": "Forum exposure."}}
    obs = [_obs("governing_law", provenance="counterparty_paper")]
    concepts = _compile(obs, taxonomy_metadata=meta)
    d = concepts[0].to_dict()
    assert "risk_profile" in d
    assert "notes" in d  # counterparty_paper obs triggers notes


def test_to_dict_accepted_forms_each_have_example_ref() -> None:
    """Every accepted form in output carries example_ref (citation)."""
    obs = [
        _obs("governing_law", doc_id="deal_A", version="v2"),
        _obs("governing_law", doc_id="deal_B", version="v1", clause_path="9"),
    ]
    concepts = _compile(obs)
    d = concepts[0].to_dict()
    for af_dict in d["accepted_forms"]:
        assert "example_ref" in af_dict
        assert "document_id" in af_dict["example_ref"]


def test_to_dict_risk_delta_omitted_when_none() -> None:
    """risk_delta_vs_our_standard must not appear when it's None."""
    from playbook_engine.clause_position_compiler import OPFCitation

    af = AcceptedForm(
        text_summary="text",
        example_ref=OPFCitation(document_id="doc", version="v1"),
        provenance="our_paper",
        risk_delta_vs_our_standard=None,
    )
    d = af.to_dict()
    assert "risk_delta_vs_our_standard" not in d


def test_to_dict_risk_delta_present_when_set() -> None:
    from playbook_engine.clause_position_compiler import OPFCitation

    af = AcceptedForm(
        text_summary="text",
        example_ref=OPFCitation(document_id="doc", version="v1"),
        provenance="counterparty_paper",
        risk_delta_vs_our_standard={"direction": "worse", "magnitude": "minor"},
    )
    d = af.to_dict()
    assert d["risk_delta_vs_our_standard"] == {"direction": "worse", "magnitude": "minor"}


# ---------------------------------------------------------------------------
# ClauseConcept dataclass
# ---------------------------------------------------------------------------


def test_clause_concept_frozen() -> None:
    from playbook_engine.clause_position_compiler import OPFCitation

    af = AcceptedForm(
        text_summary="t",
        example_ref=OPFCitation(document_id="d", version="v1"),
        provenance="our_paper",
    )
    cc = ClauseConcept(
        concept_id="concept.ind",
        taxonomy_id="ind",
        description="Indemnification.",
        accepted_forms=(af,),
    )
    with pytest.raises((AttributeError, TypeError)):
        cc.taxonomy_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Unclassified coverage (issue #113)
# ---------------------------------------------------------------------------


def test_unclassified_coverage_counts_none_taxonomy_observations() -> None:
    """taxonomy_id=None observations are excluded from concepts but counted,
    not silently omitted, in the returned unclassified coverage."""
    from playbook_engine.clause_position_compiler import UnclassifiedCoverage

    obs = [
        _obs(None, doc_id="deal_001", version="v1", clause_path="3"),
        _obs(None, doc_id="deal_002", version="v1", clause_path="7"),
        _obs("governing_law"),
    ]
    concepts, unclassified = compile_clause_library(obs)
    assert len(concepts) == 1
    assert isinstance(unclassified, UnclassifiedCoverage)
    assert unclassified.count == 2


def test_unclassified_coverage_by_document_breakdown() -> None:
    """Per-document counts let a consumer see which documents lost content."""
    obs = [
        _obs(None, doc_id="deal_001", version="v1", clause_path="3"),
        _obs(None, doc_id="deal_001", version="v1", clause_path="5"),
        _obs(None, doc_id="deal_002", version="v1", clause_path="7"),
        _obs("governing_law"),
    ]
    _concepts, unclassified = compile_clause_library(obs)
    assert unclassified.by_document == {"deal_001": 2, "deal_002": 1}


def test_unclassified_coverage_example_citations_present() -> None:
    """Example citations are real citations pointing at the source document."""
    obs = [
        _obs(None, doc_id="deal_001", version="v1", clause_path="3"),
        _obs("governing_law"),
    ]
    _concepts, unclassified = compile_clause_library(obs)
    assert len(unclassified.example_citations) == 1
    citation = unclassified.example_citations[0]
    assert citation.document_id == "deal_001"
    assert citation.clause_path == "3"


def test_unclassified_coverage_zero_when_all_classified() -> None:
    """No unclassified observations → count 0, empty breakdown/examples."""
    obs = [_obs("governing_law"), _obs("indemnification", clause_path="8")]
    _concepts, unclassified = compile_clause_library(obs)
    assert unclassified.count == 0
    assert unclassified.by_document == {}
    assert unclassified.example_citations == ()


def test_unclassified_coverage_to_dict_shape() -> None:
    """to_dict() emits count/by_document/example_citations for JSON persistence."""
    obs = [_obs(None, doc_id="deal_001", version="v1", clause_path="3")]
    _concepts, unclassified = compile_clause_library(obs)
    d = unclassified.to_dict()
    assert d["count"] == 1
    assert d["by_document"] == {"deal_001": 1}
    assert len(d["example_citations"]) == 1
    assert d["example_citations"][0]["document_id"] == "deal_001"
