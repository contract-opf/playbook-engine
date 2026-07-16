"""Tests for the OPF validator (schema + normative rules)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from playbook_engine.validator import load_opf_file, validate_document

FIXTURES = Path(__file__).parent.parent / "examples" / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(name: str) -> dict[str, Any]:
    with (FIXTURES / name).open() as f:
        result: dict[str, Any] = json.load(f)
        return result


# ---------------------------------------------------------------------------
# Valid fixture
# ---------------------------------------------------------------------------


def test_minimal_valid_passes() -> None:
    doc = _load("minimal_valid.json")
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


# ---------------------------------------------------------------------------
# Schema errors
# ---------------------------------------------------------------------------


def test_missing_opf_version_fails() -> None:
    doc = _load("invalid_missing_opf_version.json")
    result = validate_document(doc)
    assert not result.ok
    messages = [e.message for e in result.errors]
    assert any("opf_version" in m or "'opf_version' is a required" in m for m in messages)


def test_wrong_opf_version_fails() -> None:
    doc = _load("minimal_valid.json")
    doc["opf_version"] = "9.9"
    result = validate_document(doc)
    assert not result.ok


def test_unknown_opf_version_fails_loud() -> None:
    """A doc with a genuinely unrecognized opf_version (e.g. "0.3", neither
    v0.1 nor v0.2) — even one shaped like evidence.clauses — must be rejected
    with an explicit unsupported-opf_version error rather than silently
    passing normative checks that iterate an empty top-level `clauses`."""
    doc = _load("invalid_unknown_opf_version.json")
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "unsupported" in messages.lower() and "opf_version" in messages.lower()


def test_v01_fixtures_unaffected_by_version_gate() -> None:
    """Existing v0.1 fixtures must validate/behave exactly as before the
    opf_version gate was introduced."""
    doc = _load("minimal_valid.json")
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


def test_missing_required_top_level_field_fails() -> None:
    doc = _load("minimal_valid.json")
    del doc["compiler"]
    result = validate_document(doc)
    assert not result.ok


# ---------------------------------------------------------------------------
# OPF §2.2 — Provenance rule
# ---------------------------------------------------------------------------


def test_provenance_rule_violation_detected() -> None:
    doc = _load("invalid_provenance_rule.json")
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "counterparty" in messages.lower() or "§2.2" in messages


def test_our_standard_citing_unknown_document_fails() -> None:
    """B1 regression: our_standard citing a document_id absent from corpus must be caught."""
    doc = _load("minimal_valid.json")
    doc["clauses"][0]["our_standard"]["source_ref"]["document_id"] = "not-in-corpus"
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert (
        "unknown" in messages.lower()
        or "dangling" in messages.lower()
        or "not-in-corpus" in messages
    )


def test_mixed_provenance_clause_with_strong_position_passes() -> None:
    """B2 regression: a clause with >=1 our_paper observation MAY have a strong rollup.position."""
    doc = _load("minimal_valid.json")
    clause = doc["clauses"][0]
    # Add a counterparty_paper observation alongside the existing our_paper one
    clause["observed_positions"].append(
        {
            "text_summary": "Unilateral indemnification.",
            "example_ref": {
                "document_id": "university-of-example",
                "version": 1,
                "clause_path": "8",
                "char_span": [0, 30],
            },
            "deviation": "substantive",
            "risk_delta": {"direction": "worse", "magnitude": "material"},
            "provenance": "counterparty_paper",
            "outcome": "proposed_then_reversed",
            "precedent_count": 1,
        }
    )
    clause["rollup"]["position"] = "standard"
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


def test_all_counterparty_paper_with_strong_position_fails() -> None:
    """B2 complement: when ALL observations are counterparty_paper, strong position is §2.2 violation."""
    doc = _load("minimal_valid.json")
    clause = doc["clauses"][0]
    # Flip the single our_paper observation to counterparty_paper and remove our_standard source
    clause["observed_positions"][0]["provenance"] = "counterparty_paper"
    clause["our_standard"] = None
    clause["rollup"]["position"] = "standard"
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "counterparty" in messages.lower() or "§2.2" in messages


def test_counterparty_observations_allowed_in_clause_library() -> None:
    """Counterparty-paper entries in clause_library should not trigger §2.2."""
    doc = _load("minimal_valid.json")
    doc["clause_library"] = [
        {
            "concept_id": "concept.gov_law",
            "taxonomy_id": "indemnification",
            "description": "Governing law as used by counterparty.",
            "accepted_forms": [
                {
                    "text_summary": "Counterparty home-state law.",
                    "example_ref": {
                        "document_id": "university-of-example",
                        "version": 1,
                        "clause_path": "12",
                        "char_span": [0, 30],
                    },
                    "provenance": "counterparty_paper",
                    "risk_delta_vs_our_standard": {"direction": "worse", "magnitude": "minor"},
                }
            ],
        }
    ]
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


# ---------------------------------------------------------------------------
# OPF §3.6 — Out-of-scope docs must carry scope_rationale
# ---------------------------------------------------------------------------


def test_out_of_scope_without_rationale_fails() -> None:
    doc = _load("invalid_out_of_scope_no_rationale.json")
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "scope_rationale" in messages or "§3.6" in messages


def test_out_of_scope_with_rationale_passes() -> None:
    doc = _load("minimal_valid.json")
    doc["corpus"]["documents"].append(
        {
            "document_id": "stray-nda",
            "title": "Stray NDA",
            "provenance": "our_paper",
            "in_scope": False,
            "scope_rationale": "Non-disclosure agreement, not an affiliation agreement.",
        }
    )
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


# ---------------------------------------------------------------------------
# OPF §4 — Citations required
# ---------------------------------------------------------------------------


def test_our_standard_missing_citation_fails() -> None:
    doc = _load("minimal_valid.json")
    del doc["clauses"][0]["our_standard"]["source_ref"]
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "citation" in messages.lower() or "source_ref" in messages.lower()


def test_empty_document_id_in_citation_fails() -> None:
    """B3 regression: a citation with document_id='' must be caught (vacuous citation)."""
    doc = _load("minimal_valid.json")
    doc["clauses"][0]["our_standard"]["source_ref"]["document_id"] = ""
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "empty" in messages.lower() or "minLength" in messages or "document_id" in messages


def test_observation_missing_example_ref_schema_fails() -> None:
    """Schema requires example_ref — missing it should fail schema validation."""
    doc = _load("minimal_valid.json")
    doc["clauses"][0]["observed_positions"][0]["example_ref"] = None
    result = validate_document(doc)
    assert not result.ok


def test_bare_citation_missing_version_clause_path_rejected() -> None:
    """A citation with only document_id (no version/clause_path) is untraceable
    in practice and must be rejected — OPF §4 / playbook.schema.json citation def."""
    doc = _load("invalid_bare_citation.json")
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "version" in messages.lower() or "clause_path" in messages.lower()


def test_dangling_observation_example_ref_rejected() -> None:
    """N2 regression: dangling-citation detection must cover observation.example_ref,
    not just our_standard.source_ref — a citation to a document_id absent from
    corpus.documents is unresolvable regardless of which field holds it."""
    doc = _load("invalid_dangling_observation_citation.json")
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "not-in-corpus" in messages or "dangling" in messages.lower()


def test_citation_version_exceeding_corpus_versions_rejected() -> None:
    """A citation's version ordinal must not exceed corpus.documents[id].versions —
    otherwise it points at a version that was never ingested."""
    doc = _load("minimal_valid.json")
    doc["clauses"][0]["observed_positions"][0]["example_ref"]["version"] = 99
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "version" in messages.lower() and (
        "exceeds" in messages.lower() or "dangling" in messages.lower()
    )


def test_rollup_fallback_missing_citation_fails() -> None:
    """N1 regression: citations in rollup.fallbacks must also be checked (§4)."""
    doc = _load("minimal_valid.json")
    obs = {
        "text_summary": "Mutual indemnification, higher risk.",
        "example_ref": None,
        "deviation": "substantive",
        "risk_delta": {"direction": "worse", "magnitude": "minor"},
        "provenance": "our_paper",
        "outcome": "signed",
        "precedent_count": 1,
    }
    doc["clauses"][0]["rollup"]["fallbacks"] = [obs]
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "citation" in messages.lower() or "example_ref" in messages.lower() or "§4" in messages


# ---------------------------------------------------------------------------
# OPF v0.2 — schema + validator dispatch on opf_version
# ---------------------------------------------------------------------------


def test_v0_2_minimal_valid_passes() -> None:
    doc = _load("valid_v0_2_minimal.json")
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


def test_v0_2_top_level_clauses_is_rejected_by_schema() -> None:
    """A v0.2 doc must not carry a top-level `clauses` — v0.2's clauses live
    under `evidence.clauses`. (additionalProperties: false on the v0.2 schema.)"""
    doc = _load("valid_v0_2_minimal.json")
    doc["clauses"] = []
    result = validate_document(doc)
    assert not result.ok


def test_v0_2_provenance_rule_violation_detected() -> None:
    """v0.2 §2.2: our_standard sourced from counterparty_paper, and a
    historical_stance stronger than 'mixed' when all observations are
    counterparty_paper, must both be rejected."""
    doc = _load("invalid_v0_2_provenance_rule.json")
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "counterparty" in messages.lower() or "§2.2" in messages


def test_v0_2_dangling_citation_detected() -> None:
    """v0.2 §4: a citation in evidence.clauses[].observed_positions pointing
    at a document_id absent from corpus.documents must be dangling."""
    doc = _load("invalid_v0_2_dangling_citation.json")
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "not-in-corpus" in messages or "dangling" in messages.lower()


def test_v0_2_dangling_acceptable_if_citation_detected() -> None:
    """Issue #141: acceptable_if entries are {if,to,rationale} triples citing
    their supporting observation via observation_ref — a dangling
    observation_ref (unknown document_id) must be rejected, same as any other
    OPF §4 citation."""
    doc = _load("invalid_v0_2_dangling_acceptable_if_citation.json")
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "not-in-corpus" in messages or "dangling" in messages.lower()


def test_v0_2_acceptable_if_legacy_string_form_still_valid() -> None:
    """Backward compatibility: a bare-string acceptable_if entry (v0.1-era /
    hand-authored) still validates — the schema accepts it on input even
    though this engine's compiler only ever emits the {if,to,rationale}
    triple."""
    doc = _load("valid_v0_2_minimal.json")
    doc["evidence"]["clauses"][0]["summary"]["acceptable_if"] = ["mutual", "negligence-limited"]
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


def test_v0_2_out_of_scope_without_rationale_fails() -> None:
    doc = _load("invalid_v0_2_out_of_scope_no_rationale.json")
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "scope_rationale" in messages or "§3.6" in messages


def test_v0_2_mixed_provenance_clause_with_strong_stance_passes() -> None:
    """A clause with >= min_evidence_n (default 2) our_paper observations MAY
    carry a historical_stance stronger than 'mixed' even when counterparty_paper
    observations are also present — the §2.2 restriction only bites when
    n_our_paper is below the evidence-depth floor (issue #144)."""
    doc = _load("valid_v0_2_minimal.json")
    clause = doc["evidence"]["clauses"][0]
    # Base fixture already carries 2 our_paper observations (meets the
    # default min_evidence_n=2) — add a counterparty_paper observation
    # alongside them; the mix must not demote the stance.
    clause["observed_positions"].append(
        {
            "text_summary": "Unilateral indemnification.",
            "example_ref": {
                "document_id": "university-of-example",
                "version": 3,
                "clause_path": "8",
                "char_span": [0, 30],
            },
            "deviation": "substantive",
            "risk_delta": {"direction": "worse", "magnitude": "material"},
            "provenance": "counterparty_paper",
            "outcome": "proposed_then_reversed",
            "precedent_count": 1,
        }
    )
    clause["summary"]["historical_stance"] = "consistently_held"
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


def test_v0_2_mixed_provenance_below_min_evidence_n_fails() -> None:
    """Issue #144: 1 our-paper observation among many counterparty_paper
    observations does NOT license a historical_stance stronger than 'mixed'
    at the default min_evidence_n=2 — even though our_standard is validly
    sourced from the template (not counterparty_paper), so this is caught
    only by the evidence-depth check, not the our_standard-provenance check."""
    doc = _load("invalid_v0_2_insufficient_evidence.json")
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "min_evidence_n" in messages or "§2.2" in messages


def test_v0_2_mixed_provenance_meets_custom_higher_min_evidence_n_still_fails() -> None:
    """A producer-configured min_evidence_n higher than the compiler default
    must also be enforced by the validator when explicitly passed — 2
    our_paper observations pass at the default (N=2) but must fail at N=3."""
    doc = _load("valid_v0_2_minimal.json")
    result = validate_document(doc, min_evidence_n=3)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "min_evidence_n=3" in messages


def test_v0_2_agreement_type_aliases_accepted() -> None:
    doc = _load("valid_v0_2_minimal.json")
    assert "eiaa" in doc["agreement_type"]["aliases"]
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


def test_v0_2_empty_posture_and_floor_are_valid() -> None:
    """posture/floor MAY be empty-but-present — a corpus-only compile with no
    interview run and no attorney-authored invariants still validates."""
    doc = _load("valid_v0_2_minimal.json")
    doc["posture"] = {}
    doc["floor"] = {}
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


# ---------------------------------------------------------------------------
# YAML input
# ---------------------------------------------------------------------------


def test_yaml_input_accepted(tmp_path: Path) -> None:
    import yaml

    doc = _load("minimal_valid.json")
    yaml_path = tmp_path / "playbook.yaml"
    yaml_path.write_text(yaml.dump(doc), encoding="utf-8")
    loaded = load_opf_file(yaml_path)
    result = validate_document(loaded)
    assert result.ok, [str(e) for e in result.errors]


# ---------------------------------------------------------------------------
# load_opf_file errors
# ---------------------------------------------------------------------------


def test_load_non_object_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="Expected a JSON/YAML object"):
        load_opf_file(bad)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_validate_valid_file(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from playbook_engine.cli import cli

    dest = tmp_path / "playbook.json"
    dest.write_text(json.dumps(_load("minimal_valid.json")))
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", str(dest)])
    assert result.exit_code == 0, result.output


def test_cli_validate_invalid_file(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from playbook_engine.cli import cli

    dest = tmp_path / "bad.json"
    dest.write_text(json.dumps(_load("invalid_missing_opf_version.json")))
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", str(dest)])
    assert result.exit_code != 0
