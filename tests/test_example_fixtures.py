"""Validate the worked example playbooks shipped in examples/."""

from __future__ import annotations

from pathlib import Path

from playbook_engine.validator import load_opf_file, validate_document

EXAMPLES = Path(__file__).parent.parent / "examples"


def _validate(name: str) -> None:
    doc = load_opf_file(EXAMPLES / name)
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


def test_our_paper_baseline_fixture_is_valid() -> None:
    _validate("our-paper-baseline.playbook.json")


def test_emergent_no_template_fixture_is_valid() -> None:
    _validate("emergent-no-template.playbook.json")


def test_our_paper_baseline_has_concession() -> None:
    """The fixture must demonstrate a concession (signed, worse risk_delta)."""
    doc = load_opf_file(EXAMPLES / "our-paper-baseline.playbook.json")
    concessions = [
        obs
        for clause in doc["clauses"]
        for obs in clause.get("observed_positions", [])
        if obs["outcome"] == "signed" and obs["risk_delta"]["direction"] == "worse"
    ]
    assert concessions, "our-paper-baseline fixture must contain at least one concession"


def test_our_paper_baseline_has_rejected_ask() -> None:
    """The fixture must demonstrate a rejected ask (proposed_then_reversed)."""
    doc = load_opf_file(EXAMPLES / "our-paper-baseline.playbook.json")
    rejected = [
        obs
        for clause in doc["clauses"]
        for obs in clause.get("observed_positions", [])
        if obs["outcome"] == "proposed_then_reversed"
    ]
    assert rejected, (
        "our-paper-baseline fixture must contain at least one proposed_then_reversed observation"
    )


def test_our_paper_baseline_has_acceptable_variant() -> None:
    """The fixture must demonstrate an acceptable variant (signed, neutral risk_delta)."""
    doc = load_opf_file(EXAMPLES / "our-paper-baseline.playbook.json")
    acceptable = [
        obs
        for clause in doc["clauses"]
        for obs in clause.get("observed_positions", [])
        if obs["outcome"] == "signed" and obs["risk_delta"]["direction"] == "neutral"
    ]
    assert acceptable, "our-paper-baseline fixture must contain at least one acceptable variant"


def test_our_paper_baseline_has_counterparty_paper_clause_library() -> None:
    """The fixture must have a counterparty-paper entry in clause_library."""
    doc = load_opf_file(EXAMPLES / "our-paper-baseline.playbook.json")
    cp_forms = [
        form
        for concept in doc.get("clause_library", [])
        for form in concept.get("accepted_forms", [])
        if form["provenance"] == "counterparty_paper"
    ]
    assert cp_forms, (
        "our-paper-baseline fixture must contain counterparty_paper entries in clause_library"
    )


def test_emergent_has_concession() -> None:
    doc = load_opf_file(EXAMPLES / "emergent-no-template.playbook.json")
    concessions = [
        obs
        for clause in doc["clauses"]
        for obs in clause.get("observed_positions", [])
        if obs["outcome"] == "signed" and obs["risk_delta"]["direction"] == "worse"
    ]
    assert concessions, "emergent fixture must contain at least one concession"


def test_emergent_has_rejected_ask() -> None:
    doc = load_opf_file(EXAMPLES / "emergent-no-template.playbook.json")
    rejected = [
        obs
        for clause in doc["clauses"]
        for obs in clause.get("observed_positions", [])
        if obs["outcome"] == "proposed_then_reversed"
    ]
    assert rejected, "emergent fixture must contain at least one proposed_then_reversed observation"


def test_emergent_has_acceptable_variant() -> None:
    doc = load_opf_file(EXAMPLES / "emergent-no-template.playbook.json")
    acceptable = [
        obs
        for clause in doc["clauses"]
        for obs in clause.get("observed_positions", [])
        if obs["outcome"] == "signed" and obs["risk_delta"]["direction"] == "neutral"
    ]
    assert acceptable, "emergent fixture must contain at least one acceptable variant"


def test_emergent_has_counterparty_paper_clause_library() -> None:
    doc = load_opf_file(EXAMPLES / "emergent-no-template.playbook.json")
    cp_forms = [
        form
        for concept in doc.get("clause_library", [])
        for form in concept.get("accepted_forms", [])
        if form["provenance"] == "counterparty_paper"
    ]
    assert cp_forms, "emergent fixture must contain counterparty_paper entries in clause_library"


def test_emergent_has_no_template() -> None:
    doc = load_opf_file(EXAMPLES / "emergent-no-template.playbook.json")
    assert not doc["baseline"]["has_canonical_template"]
    assert doc["baseline"]["template_ref"] is None


def test_emergent_clauses_have_no_our_standard() -> None:
    """In an emergent playbook with no template, our_standard should be null."""
    doc = load_opf_file(EXAMPLES / "emergent-no-template.playbook.json")
    for clause in doc["clauses"]:
        assert clause["our_standard"] is None, (
            f"Emergent playbook clause {clause['id']} should have our_standard=null"
        )


def test_out_of_scope_docs_have_rationale_in_both_fixtures() -> None:
    """Every out-of-scope document in both fixtures must have a scope_rationale."""
    for name in ("our-paper-baseline.playbook.json", "emergent-no-template.playbook.json"):
        doc = load_opf_file(EXAMPLES / name)
        for corpus_doc in doc["corpus"]["documents"]:
            if not corpus_doc["in_scope"]:
                assert corpus_doc.get("scope_rationale"), (
                    f"{name}: out-of-scope doc {corpus_doc['document_id']} missing scope_rationale"
                )
