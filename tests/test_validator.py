"""Tests for the OPF validator (schema + normative rules)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.digest import build_digest
from playbook_engine.validator import ValidationResult, load_opf_file, validate_document

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
    """A doc with a genuinely unrecognized opf_version (e.g. "0.9" — no supported
    schema) — even one shaped like evidence.clauses — must be rejected
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
# Issue #72 regression — malformed (hand-edited/foreign) documents must
# produce a ValidationResult with blocking errors, never raise.
# ---------------------------------------------------------------------------


def test_null_document_id_citation_and_missing_corpus_document_id_does_not_crash() -> None:
    """A citation whose document_id is JSON null (not merely absent), combined
    with a corpus document that itself lacks document_id, used to raise
    AttributeError (ref.get("document_id", "").strip() on None) or KeyError
    (_corpus_docs' d["document_id"]) instead of returning a ValidationResult."""
    doc = _load("minimal_valid.json")
    doc["clauses"][0]["observed_positions"][0]["example_ref"]["document_id"] = None
    doc["corpus"]["documents"].append({})
    result = validate_document(doc)
    assert not result.ok
    assert any(e.blocking for e in result.errors)


def test_null_our_standard_text_does_not_crash() -> None:
    """our_standard.text = null used to raise AttributeError on
    std.get("text", "").strip() instead of reporting a blocking error."""
    doc = _load("minimal_valid.json")
    doc["clauses"][0]["our_standard"]["text"] = None
    result = validate_document(doc)
    assert not result.ok
    assert any(e.blocking for e in result.errors)


def test_v0_2_null_our_standard_text_does_not_crash() -> None:
    """Same defect in the v0.2 evidence-wrapped citation walker
    (_check_citations_v2)."""
    doc = _load("valid_v0_2_minimal.json")
    doc["evidence"]["clauses"][0]["our_standard"]["text"] = None
    result = validate_document(doc)
    assert not result.ok
    assert any(e.blocking for e in result.errors)


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
# _check_identity_hash_v2 — issue #178
# ---------------------------------------------------------------------------


def test_identity_absent_does_not_affect_validity() -> None:
    """No identity section at all (e.g. a corpus-only compile) has nothing to
    verify — must not be treated as a mismatch."""
    doc = _load("valid_v0_2_minimal.json")
    assert "identity" not in doc
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


def test_identity_matching_hashes_pass() -> None:
    """A genuinely fresh identity — computed from this exact document — must
    validate clean."""
    doc = _load("valid_v0_2_minimal.json")
    doc["identity"] = {
        "content_hash": content_hash(doc),
        "section_digests": compute_section_digests(doc),
    }
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


def test_identity_stale_content_hash_fails_blocking() -> None:
    """A hand-edited playbook (content changed after identity was stamped)
    must fail validate — the exact gap the ticket's Consequence describes:
    the one local integrity gate must catch what it's positioned to catch."""
    doc = _load("valid_v0_2_minimal.json")
    doc["identity"] = {
        "content_hash": content_hash(doc),
        "section_digests": compute_section_digests(doc),
    }
    # Hand-edit content without recomputing identity.
    doc["posture"]["system_prompt"] = "hand-edited after stamping"
    result = validate_document(doc)

    assert not result.ok
    offending = [e for e in result.errors if e.path == "identity.content_hash"]
    assert offending, [str(e) for e in result.errors]
    assert offending[0].blocking


def test_identity_stale_section_digest_fails_blocking() -> None:
    """A section_digests entry that no longer matches its own section must
    fail validate, even reported independently of content_hash."""
    doc = _load("valid_v0_2_minimal.json")
    doc["identity"] = {
        "content_hash": content_hash(doc),
        "section_digests": compute_section_digests(doc),
    }
    doc["floor"]["invariants"][0]["statement"] = "hand-edited after stamping"
    # Keep content_hash consistent with the edit so only the digest mismatch fires.
    doc["identity"]["content_hash"] = content_hash(doc)
    result = validate_document(doc)

    assert not result.ok
    offending = [e for e in result.errors if e.path == "identity.section_digests.floor"]
    assert offending, [str(e) for e in result.errors]
    assert offending[0].blocking


def test_identity_missing_optional_curation_digest_is_not_a_mismatch() -> None:
    """section_digests.curation is optional (schema `required` omits it) — a
    document that never populated it has asserted nothing about it, so its
    absence must not be flagged even though compute_section_digests always
    returns a curation entry."""
    doc = _load("valid_v0_2_minimal.json")
    digests = compute_section_digests(doc)
    del digests["curation"]
    doc["identity"] = {"content_hash": content_hash(doc), "section_digests": digests}
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


# ---------------------------------------------------------------------------
# _check_posture_interview_provenance_v2 — issue #133
# ---------------------------------------------------------------------------


def test_posture_with_no_interview_record_gets_should_warn() -> None:
    """A drafted system_prompt with no generation.interview behind it -- the
    exact gap the ticket's Consequence describes: a hand-edited or
    third-party playbook that strips the interview provenance still
    validates clean without this check."""
    doc = _load("valid_v0_2_minimal.json")
    del doc["posture"]["generation"]["interview"]
    result = validate_document(doc)

    warn_messages = [e for e in result.errors if not e.blocking]
    offending = [e for e in warn_messages if "interview" in e.message and "§3.6" in e.message]
    assert offending, [str(e) for e in result.errors]
    assert offending[0].path == "posture.generation.interview"
    # Advisory only -- a Posture with no traceable provenance is still a
    # structurally valid document, just one a human should look at.
    assert all(not e.blocking for e in offending)


def test_posture_with_empty_interview_list_still_warns() -> None:
    doc = _load("valid_v0_2_minimal.json")
    doc["posture"]["generation"]["interview"] = []
    result = validate_document(doc)

    assert any("§3.6" in e.message for e in result.errors)


def test_posture_with_interview_record_suppresses_the_warning() -> None:
    doc = _load("valid_v0_2_minimal.json")
    result = validate_document(doc)

    assert not any("§3.6" in e.message and "interview" in e.message for e in result.errors)


def test_empty_posture_does_not_trigger_interview_provenance_warning() -> None:
    """No system_prompt at all -- e.g. a corpus-only compile -- has nothing
    to provide provenance for, so it must not warn (would otherwise
    contradict test_v0_2_empty_posture_and_floor_are_valid's `result.ok`)."""
    doc = _load("valid_v0_2_minimal.json")
    doc["posture"] = {}
    result = validate_document(doc)

    assert not any("interview" in e.message and "§3.6" in e.message for e in result.errors)
    assert result.ok, [str(e) for e in result.errors]


# ---------------------------------------------------------------------------
# _check_perspective_present_v2 — issue #212
# ---------------------------------------------------------------------------


def test_missing_perspective_gets_should_warn() -> None:
    """The reported production failure: a playbook with no top-level
    `perspective` validates clean, so a consumer never learns which side it
    acts for and applies its floors symmetrically to a one-sided clause.
    OPF-SPEC §3.1 says an instance MUST say who "us" is, but the field is
    absent from the schema's `required` list and the 1.0 stability policy
    forbids adding one in a 1.x release -- so a warn is the enforcement."""
    doc = _load("valid_v0_2_minimal.json")
    del doc["perspective"]
    # Re-stamp identity: dropping a top-level key changes the content hash,
    # and an unrelated blocking hash error would muddy the assertions below.
    digests = compute_section_digests(doc)
    doc["identity"] = {"content_hash": content_hash(doc), "section_digests": digests}
    result = validate_document(doc)

    offending = [e for e in result.errors if e.path == "perspective"]
    assert offending, [str(e) for e in result.errors]
    assert "§3.1" in offending[0].message
    # Advisory only -- a perspective-less playbook is still structurally
    # valid, which is exactly why nothing caught this in production.
    assert all(not e.blocking for e in offending)
    assert result.ok, [str(e) for e in result.errors]


def test_perspective_present_suppresses_the_warning() -> None:
    doc = _load("valid_v0_2_minimal.json")
    assert doc["perspective"] == {
        "party": "FixtureCorp",
        "counterparty_type": "Educational Institution",
    }
    result = validate_document(doc)

    assert not any(e.path == "perspective" for e in result.errors)


def test_partial_perspective_is_a_blocking_schema_error_not_a_warn() -> None:
    """`party` and `counterparty_type` are required together by the schema,
    so an incomplete block must fail hard rather than draw the advisory --
    the warn exists only for the whole-key-absent case."""
    doc = _load("valid_v0_2_minimal.json")
    del doc["perspective"]["counterparty_type"]
    result = validate_document(doc)

    assert not result.ok
    assert not any(
        e.path == "perspective" and not e.blocking and "§3.1" in e.message for e in result.errors
    )


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


# ---------------------------------------------------------------------------
# Invisible-character check (pre-derivation QA)
# ---------------------------------------------------------------------------


def test_validate_rejects_zero_width_chars() -> None:
    doc = _load("minimal_valid.json")
    doc["agreement_type"]["name"] = "Educational​ Affiliation"
    result = validate_document(doc)
    assert not result.ok
    assert any("zero-width" in str(e) for e in result.errors)


def test_validate_reports_invisible_char_path() -> None:
    doc = _load("minimal_valid.json")
    doc["agreement_type"]["name"] = "Name﻿"
    result = validate_document(doc)
    offending = [e for e in result.errors if "zero-width" in str(e)]
    assert offending and "agreement_type.name" in offending[0].path


# ---------------------------------------------------------------------------
# Duplicate clause/document id check (issue #70) — nothing in the schema
# enforces uniqueness; export_profile's path-keyed sample/location maps
# silently collapse two duplicates onto one path without this check.
# ---------------------------------------------------------------------------


def test_validate_rejects_duplicate_clause_id() -> None:
    doc = _load("minimal_valid.json")
    dup = copy.deepcopy(doc["clauses"][0])
    doc["clauses"].append(dup)  # same "id" as doc["clauses"][0]

    result = validate_document(doc)
    assert not result.ok
    offending = [e for e in result.errors if "duplicate clause id" in e.message]
    assert offending and offending[0].path == "clauses[1].id"


def test_validate_rejects_duplicate_corpus_document_id() -> None:
    doc = _load("minimal_valid.json")
    dup = copy.deepcopy(doc["corpus"]["documents"][0])
    doc["corpus"]["documents"].append(dup)  # same "document_id" as index 0

    result = validate_document(doc)
    assert not result.ok
    offending = [e for e in result.errors if "duplicate corpus document_id" in e.message]
    assert offending and offending[0].path == "corpus.documents[1].document_id"


def test_validate_rejects_duplicate_clause_id_v0_2_evidence_shape() -> None:
    # v0.1's top-level `clauses` and v0.2/v0.3's `evidence.clauses` are two
    # different JSON shapes for the same normative rule (issue #70) — this
    # covers the evidence.clauses shape used by every engine-produced
    # document and the whole publish pipeline, so a regression in that
    # branch (e.g. a path that names the wrong shape) is caught here rather
    # than only on the v0.1 fixture above.
    doc = _load("valid_v0_2_minimal.json")
    dup = copy.deepcopy(doc["evidence"]["clauses"][0])
    doc["evidence"]["clauses"].append(dup)  # same "id" as evidence.clauses[0]

    result = validate_document(doc)
    assert not result.ok
    offending = [e for e in result.errors if "duplicate clause id" in e.message]
    assert offending and offending[0].path == "evidence.clauses[1].id"


# ---------------------------------------------------------------------------
# _check_floor_attribution — issue #127
# ---------------------------------------------------------------------------


def test_floor_invariant_with_no_attribution_gets_should_warn() -> None:
    """The fixture's own floor invariant carries no x_signed_by and no
    Posture-interview/candidate rationale marker -- exactly the shape an
    LLM agent could fabricate end-to-end (issue #127's Consequence)."""
    doc = _load("valid_v0_2_minimal.json")
    result = validate_document(doc)

    warn_messages = [e for e in result.errors if not e.blocking]
    offending = [e for e in warn_messages if "structural attribution" in e.message]
    assert offending, [str(e) for e in result.errors]
    assert offending[0].path == "floor.invariants[0]"
    assert "no-uncapped-liability" in offending[0].message
    # Never blocking -- an unattributed invariant is still a structurally
    # valid document, just one a human should look at.
    assert all(not e.blocking for e in offending)


def test_floor_invariant_signed_by_suppresses_the_warning() -> None:
    doc = _load("valid_v0_2_minimal.json")
    doc["floor"]["invariants"][0]["x_signed_by"] = "Test Legal Owner"
    result = validate_document(doc)

    assert not any("structural attribution" in e.message for e in result.errors)


def test_floor_invariant_posture_interview_marker_suppresses_the_warning() -> None:
    """The exact rationale `promote_interview_q4_invariants` stamps -- see
    `floor_candidates._Q4_ATTRIBUTION_RE`."""
    doc = _load("valid_v0_2_minimal.json")
    doc["floor"]["invariants"][0]["rationale"] = (
        "Authored by the legal owner in posture interview v1, question sacred_clauses."
    )
    result = validate_document(doc)

    assert not any("structural attribution" in e.message for e in result.errors)


def test_floor_invariant_review_feedback_marker_suppresses_the_warning() -> None:
    """The exact rationale suffix `promote_floor_candidate` stamps on an
    accepted candidate -- see `floor_candidates._CANDIDATE_ATTRIBUTION_RE`."""
    doc = _load("valid_v0_2_minimal.json")
    doc["floor"]["invariants"][0]["rationale"] = (
        "Proposed then reversed before signing. Accepted via review feedback (floor candidate cand-001)."
    )
    result = validate_document(doc)

    assert not any("structural attribution" in e.message for e in result.errors)


def test_floor_invariant_agent_typed_rationale_still_warns() -> None:
    """Text that merely CLAIMS a sign-off, with no structural marker, must
    still trip the warning -- this is the exact gap the ticket's
    Consequence section describes (an agent-fabricated 'signed' hard line
    indistinguishable from a genuine one by rationale text alone)."""
    doc = _load("valid_v0_2_minimal.json")
    doc["floor"]["invariants"][0]["rationale"] = (
        "Hand-authored and signed by the legal owner, 2026-08-21."
    )
    result = validate_document(doc)

    assert any("structural attribution" in e.message for e in result.errors)


def test_validate_tolerates_null_floor_invariants() -> None:
    # Hand-authored YAML `floor:\n  invariants:` maps `invariants` to `None`
    # (a valueless key), not `[]` — _check_duplicate_ids runs before schema
    # validation, so it must not raise on this shape; the schema check still
    # reports it as a normal error (issue #70 round 2).
    result = validate_document({"opf_version": "0.2", "floor": {"invariants": None}})
    assert isinstance(result, ValidationResult)
    assert not result.ok


def test_validate_tolerates_non_list_floor_invariants() -> None:
    # `floor: {invariants: ["a"]}` — a non-dict item inside the list — must
    # not raise either; _check_duplicate_ids skips non-dict entries.
    result = validate_document({"opf_version": "0.2", "floor": {"invariants": ["a"]}})
    assert isinstance(result, ValidationResult)
    assert not result.ok


# ---------------------------------------------------------------------------
# _check_duplicate_ids runs BEFORE schema validation (it is the second check
# validate_document runs, right after _check_invisible_chars), so a non-dict
# entry in any of the containers it walks must not raise — it must fall
# through to the schema check, which reports it as a normal error. Round-3
# review finding: `clause.get(...)`/`concept.get(...)` had no isinstance
# guard for the `clauses`/`evidence.clauses` and `clause_library` families
# (only `floor.invariants` and `corpus.documents` were covered above), so
# `["a"]`, `[None]`, and a mixed-type list all raised AttributeError on the
# unfixed code instead of returning a ValidationResult (issue #70).
# ---------------------------------------------------------------------------


def test_validate_tolerates_non_dict_top_level_clauses_string() -> None:
    result = validate_document({"opf_version": "0.2", "clauses": ["a"]})
    assert isinstance(result, ValidationResult)
    assert not result.ok


def test_validate_tolerates_non_dict_top_level_clauses_none() -> None:
    result = validate_document({"opf_version": "0.2", "clauses": [None]})
    assert isinstance(result, ValidationResult)
    assert not result.ok


def test_validate_tolerates_mixed_type_top_level_clauses() -> None:
    result = validate_document({"opf_version": "0.2", "clauses": [{"id": "a"}, 5]})
    assert isinstance(result, ValidationResult)
    assert not result.ok


# NOTE: an `evidence.clauses` (v0.2 evidence-wrapped shape) sibling of the
# three tests above is intentionally NOT included here. `evidence.clauses:
# ["a"]` / `[None]` / a mixed-type list all still raise AttributeError from
# `validate_document` even with `_check_duplicate_ids` fully guarded (verified
# against this fixed tree) — but the crash is NOT in `_check_duplicate_ids`;
# it is in `_check_provenance_rule_v2` (validator.py, `clause.get("our_standard")`
# with no isinstance guard), and after guarding that too, the identical
# unguarded-`clause.get(...)` pattern recurs in `_check_evidence_depth_rule_v2`
# (`clause.get("summary", {})`). Both are pre-existing, unrelated checks —
# confirmed still present and already crashing on `git stash`/pristine `main`
# before any of this ticket's changes — not something this ticket's new
# `_check_duplicate_ids` check introduces or widens. Hardening every v0.2 (and
# v0.1) normative check against non-dict clause entries is a much larger,
# systemic fix spanning functions this ticket's Goal/Scope/DECISION comment
# never names, and is out of this ticket's authorized scope ("Out of scope:
# Only this defect. No drive-by refactors"). Flagged for a scope decision
# rather than silently included or silently dropped.


def test_validate_tolerates_non_dict_clause_library_string() -> None:
    result = validate_document({"opf_version": "0.2", "clause_library": ["a"]})
    assert isinstance(result, ValidationResult)
    assert not result.ok


def test_validate_tolerates_non_dict_clause_library_none() -> None:
    result = validate_document({"opf_version": "0.2", "clause_library": [None]})
    assert isinstance(result, ValidationResult)
    assert not result.ok


def test_validate_tolerates_mixed_type_clause_library() -> None:
    result = validate_document({"opf_version": "0.2", "clause_library": [{"concept_id": "a"}, 5]})
    assert isinstance(result, ValidationResult)
    assert not result.ok


# ---------------------------------------------------------------------------
# OPF 0.4 (issue #223) — the verdict-free per-deal precedent record. Every
# case mutates the REAL compiled NDA example (examples/nda/playbook.opf.json,
# produced by `playbook project`), so the valid baseline is a shape the
# compiler actually emits; each mutation is one a hand edit or a buggy
# producer could make, and must be rejected.
# ---------------------------------------------------------------------------

_NDA_PLAYBOOK = Path(__file__).parent.parent / "examples" / "nda" / "playbook.opf.json"


def _nda_04() -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(_NDA_PLAYBOOK.read_text(encoding="utf-8"))
    assert doc["opf_version"] == "0.4"
    return doc


def _restamp_identity(doc: dict[str, Any]) -> dict[str, Any]:
    """Re-stamp identity so a test isolates the rule it targets rather than
    tripping the content-hash check first."""
    doc["identity"]["content_hash"] = content_hash(doc)
    doc["identity"]["section_digests"] = compute_section_digests(doc)
    return doc


def _blocking(doc: dict[str, Any]) -> list[str]:
    return [str(e) for e in validate_document(doc).errors if e.blocking]


def test_v04_compiled_example_validates() -> None:
    assert _blocking(_nda_04()) == []


def test_v04_all_older_versions_still_validate() -> None:
    """0.1/0.2/0.3 must keep validating (their schemas are frozen)."""
    for name in ("minimal_valid.json", "valid_v0_2_minimal.json"):
        assert validate_document(_load(name)).ok, name
    v03 = _load("valid_v0_2_minimal.json")
    v03["opf_version"] = "0.3"
    assert validate_document(v03).ok


def test_v04_rejects_a_0_3_evidence_shape() -> None:
    doc = _nda_04()
    doc["evidence"]["clause_library"] = []
    errors = _blocking(_restamp_identity(doc))
    assert any("clause_library" in e for e in errors), errors


def test_v04_rejects_a_judged_field_on_a_precedent() -> None:
    """Judged verdicts never enter evidence.precedent (closed records)."""
    doc = _nda_04()
    doc["evidence"]["precedent"][0]["x_deviation"] = "substantive"
    errors = _blocking(_restamp_identity(doc))
    assert any("x_deviation" in e for e in errors), errors


def test_v04_rejects_tampered_precedent_id() -> None:
    doc = _nda_04()
    doc["evidence"]["precedent"][0]["id"] = "prec.0000000000000000"
    errors = _blocking(_restamp_identity(doc))
    assert any("does not match the id recomputed" in e for e in errors), errors


def test_v04_rejects_signed_text_edit_without_restamp() -> None:
    """Editing a signed text changes what the id hashes, the digest, and
    possibly the counts — the id check alone catches the edit."""
    doc = _nda_04()
    record = next(p for p in doc["evidence"]["precedent"] if p["signed_text"])
    record["signed_text"]["text"] += " Edited."
    errors = _blocking(_restamp_identity(doc))
    assert any("does not match the id recomputed" in e for e in errors), errors


def test_v04_rejects_two_precedents_for_one_deal_and_clause() -> None:
    from playbook_engine.opf_accessors import perspective_party
    from playbook_engine.precedent import restamp_evidence

    doc = _nda_04()
    first = doc["evidence"]["precedent"][0]
    twin = copy.deepcopy(first)
    twin["signed_text"] = {"text": "A different text.", "ref": first["signed_text"]["ref"]}
    twin["standard"] = False
    doc["evidence"]["precedent"].append(twin)
    restamp_evidence(doc["evidence"], doc["agreement_type"]["id"], party=perspective_party(doc))
    errors = _blocking(_restamp_identity(doc))
    assert any("more than one precedent" in e for e in errors), errors


def test_v04_rejects_count_that_disagrees_with_precedent() -> None:
    doc = _nda_04()
    doc["evidence"]["clauses"][0]["n_deals"] += 1
    errors = _blocking(_restamp_identity(doc))
    assert any("n_deals=" in e and "implies" in e for e in errors), errors


def test_v04_rejects_dangling_deal_and_signed_mismatch() -> None:
    doc = _nda_04()
    record = doc["evidence"]["precedent"][0]
    record["signed"] = not record["signed"]
    errors = _blocking(_restamp_identity(doc))
    assert any("signed_version" in e for e in errors), errors

    doc = _nda_04()
    doc["corpus"]["documents"] = [
        d
        for d in doc["corpus"]["documents"]
        if d["document_id"] != doc["evidence"]["precedent"][0]["document_id"]
    ]
    errors = _blocking(_restamp_identity(doc))
    assert any("not in corpus.documents" in e for e in errors), errors


def test_v04_rejects_standard_true_without_signed_text() -> None:
    doc = _nda_04()
    record = next(p for p in doc["evidence"]["precedent"] if p["standard"])
    record["signed_text"] = None
    errors = _blocking(_restamp_identity(doc))
    assert any("standard=true but signed_text is null" in e for e in errors), errors


def test_v04_rejects_edited_digest() -> None:
    doc = _nda_04()
    clause = next(c for c in doc["digest"]["clauses"] if c["signed_variants"])
    clause["signed_variants"][0]["n_deals"] += 1
    errors = _blocking(_restamp_identity(doc))
    assert any("digest does not equal build_digest" in e for e in errors), errors


def test_v04_rejects_digest_without_perspective_key() -> None:
    doc = _nda_04()
    del doc["digest"]["perspective"]
    errors = _blocking(_restamp_identity(doc))
    assert any("perspective" in e for e in errors), errors


def test_v04_rejects_full_text_in_digest() -> None:
    doc = _nda_04()
    doc["digest"]["clauses"][0]["signed_variants"].append(
        {
            "full_text": "x",
            "text": "x",
            "n_deals": 1,
            "last_signed": None,
            "ref": {"document_id": "template", "version": "template", "clause_path": "1"},
            "precedent_ids": ["prec.0000000000000000"],
        }
    )
    errors = _blocking(_restamp_identity(doc))
    assert any("full_text" in e for e in errors), errors


def test_v04_rejects_impossible_signed_at() -> None:
    doc = _nda_04()
    from playbook_engine.digest import build_digest

    doc["evidence"]["precedent"][0]["signed_at"] = "2025-13-45"
    doc["digest"] = build_digest(doc)
    errors = _blocking(_restamp_identity(doc))
    assert any("signed_at" in e for e in errors), errors


# ---------------------------------------------------------------------------
# OPF 0.4 paper side (issue #225) — three-valued deal metadata that gates
# nothing; the validator only checks the document tells one consistent story
# about it. Same real compiled NDA example as above.
# ---------------------------------------------------------------------------


def _make_deal_ambiguous(doc: dict[str, Any], deal: str) -> None:
    """What the pipeline writes for an ambiguous detection: the corpus document
    keeps today's two-valued value (counterparty_paper) flagged ambiguous,
    every precedent says unknown."""
    corpus_doc = next(d for d in doc["corpus"]["documents"] if d["document_id"] == deal)
    corpus_doc["provenance"] = "counterparty_paper"
    corpus_doc["provenance_is_ambiguous"] = True
    corpus_doc["provenance_confidence"] = 0.65
    for record in doc["evidence"]["precedent"]:
        if record["document_id"] == deal:
            record["paper"] = "unknown"
            record["paper_basis"] = "alias_present"
            record["paper_confidence"] = 0.65


def test_v04_paper_accepts_an_ambiguous_deal_recorded_unknown() -> None:
    doc = _nda_04()
    _make_deal_ambiguous(doc, "beta-industries")
    assert _blocking(_restamp_identity(doc)) == []


def test_v04_paper_rejects_a_side_on_an_ambiguous_deal() -> None:
    """An ambiguous detection is "unknown" — never coerced to a side."""
    doc = _nda_04()
    _make_deal_ambiguous(doc, "beta-industries")
    for record in doc["evidence"]["precedent"]:
        if record["document_id"] == "beta-industries":
            record["paper"] = "theirs"
    errors = _blocking(_restamp_identity(doc))
    assert any("provenance_is_ambiguous=true" in e and "paper='theirs'" in e for e in errors), (
        errors
    )


def test_v04_paper_rejects_a_side_that_contradicts_the_corpus_document() -> None:
    doc = _nda_04()
    for record in doc["evidence"]["precedent"]:
        if record["document_id"] == "zeta-diagnostics":  # counterparty_paper
            record["paper"] = "ours"
    errors = _blocking(_restamp_identity(doc))
    assert any("paper='ours'" in e and "provenance='counterparty_paper'" in e for e in errors), (
        errors
    )


def test_v04_paper_accepts_unknown_against_an_unflagged_corpus_document() -> None:
    """The two-valued corpus field cannot say "unknown"; a record may honestly
    withhold a side (the 0.4 conformance vectors model this)."""
    doc = _nda_04()
    for record in doc["evidence"]["precedent"]:
        if record["document_id"] == "zeta-diagnostics":
            record["paper"] = "unknown"
    assert _blocking(_restamp_identity(doc)) == []


def test_v04_paper_rejects_confidence_that_disagrees_with_the_corpus_document() -> None:
    doc = _nda_04()
    record = next(p for p in doc["evidence"]["precedent"] if p["document_id"] == "beta-industries")
    record["paper_confidence"] = 0.5
    errors = _blocking(_restamp_identity(doc))
    assert any("paper_confidence=0.5" in e for e in errors), errors


def test_v04_paper_rejects_a_deal_whose_records_disagree() -> None:
    """Paper side is a fact about the deal — every record of it agrees."""
    doc = _nda_04()
    records = [p for p in doc["evidence"]["precedent"] if p["document_id"] == "beta-industries"]
    records[-1]["paper_basis"] = "alias_first_party"
    errors = _blocking(_restamp_identity(doc))
    assert any("paper side is a fact about the deal" in e for e in errors), errors


def test_v04_paper_rejects_our_standard_from_an_unknown_paper_deal() -> None:
    doc = _nda_04()
    _make_deal_ambiguous(doc, "beta-industries")
    record = next(
        p
        for p in doc["evidence"]["precedent"]
        if p["document_id"] == "beta-industries" and p["signed_text"]
    )
    clause = next(
        c for c in doc["evidence"]["clauses"] if c["taxonomy_id"] == record["taxonomy_id"]
    )
    clause["our_standard"] = {
        "text": record["signed_text"]["text"],
        "source_ref": copy.deepcopy(record["signed_text"]["ref"]),
    }
    doc["digest"] = build_digest(doc)
    errors = _blocking(_restamp_identity(doc))
    assert any("an unknown-paper deal contributes no our_standard" in e for e in errors), errors


def test_v04_paper_unknown_deal_standard_needs_a_template() -> None:
    """With a template, ``standard`` is the exact match against it and an
    unknown-paper deal counts in n_signed_standard; with no template there is
    no standard for it to match, so it must not be standard."""
    doc = _nda_04()
    _make_deal_ambiguous(doc, "beta-industries")
    assert any(
        p["standard"] for p in doc["evidence"]["precedent"] if p["document_id"] == "beta-industries"
    )
    assert _blocking(_restamp_identity(doc)) == []

    doc["baseline"]["has_canonical_template"] = False
    errors = _blocking(_restamp_identity(doc))
    assert any("standard=true on an unknown-paper deal" in e for e in errors), errors
