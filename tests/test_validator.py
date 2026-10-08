"""Tests for the OPF validator (schema + normative rules).

The engine reads and writes exactly one format, OPF 0.4 (issue #238): every
other ``opf_version`` — the retired 0.1, 0.2 and 0.3 included — is rejected
as unsupported. Valid baselines are produced by the real producer: the
compiled NDA example (``examples/nda/playbook.opf.json``, ``playbook
project``) or a small document built through ``assemble_playbook``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.clause_position_compiler import compile_clause_positions
from playbook_engine.digest import build_digest
from playbook_engine.observation_builder import Observation, ObservationCitation
from playbook_engine.playbook_assembler import assemble_playbook
from playbook_engine.validator import ValidationResult, load_opf_file, validate_document

FIXTURES = Path(__file__).parent.parent / "examples" / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(name: str) -> dict[str, Any]:
    with (FIXTURES / name).open() as f:
        result: dict[str, Any] = json.load(f)
        return result


_STANDARD = "This Agreement is governed by the laws of the State of New York."


def _minimal() -> dict[str, Any]:
    """A small valid OPF 0.4 document, built by the real producer: one
    template clause (so ``our_standard`` cites the template) and one in-scope
    corpus document. ``identity`` is dropped so a mutation trips only the
    rule under test (identity is optional; ``_refresh`` re-derives the
    digest)."""
    template = Observation(
        observation_id="template/governing_law",
        taxonomy_id="governing_law",
        text_summary=_STANDARD,
        citation=ObservationCitation(
            document_id="template", version="template", clause_path="4", char_span=None
        ),
        deviation="none",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome="signed",
    )
    positions, _, _ = compile_clause_positions([], [template])
    doc = assemble_playbook(
        agreement_type={"id": "test-agreement", "name": "Test Agreement"},
        baseline={"has_canonical_template": True},
        taxonomy={"source": "custom", "entries": []},
        clause_positions=positions,
        corpus_documents=[
            {
                "document_id": "deal-a",
                "provenance": "our_paper",
                "in_scope": True,
                "versions": 2,
                "signed_version": 2,
            }
        ],
        generated_at="2026-01-01T00:00:00Z",
        perspective={"party": "FixtureCorp", "counterparty_type": "Educational Institution"},
    )
    del doc["identity"]
    return doc


def _refresh(doc: dict[str, Any]) -> dict[str, Any]:
    """Re-derive the digest (and identity, when present) after a mutation."""
    doc["digest"] = build_digest(doc)
    if "identity" in doc:
        doc["identity"]["content_hash"] = content_hash(doc)
        doc["identity"]["section_digests"] = compute_section_digests(doc)
    return doc


def _messages(doc: dict[str, Any]) -> str:
    return " ".join(e.message for e in validate_document(doc).errors)


# ---------------------------------------------------------------------------
# Valid documents
# ---------------------------------------------------------------------------


def test_minimal_valid_passes() -> None:
    result = validate_document(_minimal())
    assert result.ok, [str(e) for e in result.errors]


# ---------------------------------------------------------------------------
# opf_version — one supported format (issue #238)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["0.1", "0.2", "0.3"])
def test_retired_opf_version_is_rejected_as_unsupported(version: str) -> None:
    """Acceptance (issue #238): a document claiming a retired version gets one
    clear, blocking "unsupported opf_version" error — no schema noise."""
    doc = _minimal()
    doc["opf_version"] = version
    result = validate_document(doc)
    assert not result.ok
    assert len(result.errors) == 1, [str(e) for e in result.errors]
    (error,) = result.errors
    assert error.blocking
    assert error.path == "opf_version"
    assert f"unsupported opf_version {version!r}" in error.message
    assert "supported: 0.4" in error.message


@pytest.mark.parametrize("version", [None, 0.4, ["0.4"], {"v": "0.4"}])
def test_non_string_opf_version_is_unsupported_not_a_crash(version: Any) -> None:
    doc = _minimal()
    doc["opf_version"] = version
    result = validate_document(doc)
    assert not result.ok
    assert [e.path for e in result.errors] == ["opf_version"]


def test_missing_opf_version_fails() -> None:
    doc = _load("invalid_missing_opf_version.json")
    result = validate_document(doc)
    assert not result.ok
    messages = [e.message for e in result.errors]
    assert any("unsupported opf_version None" in m for m in messages)


def test_unknown_opf_version_fails_loud() -> None:
    doc = _load("invalid_unknown_opf_version.json")
    result = validate_document(doc)
    assert not result.ok
    messages = " ".join(e.message for e in result.errors)
    assert "unsupported" in messages.lower() and "opf_version" in messages.lower()


def test_missing_required_top_level_field_fails() -> None:
    doc = _minimal()
    del doc["compiler"]
    assert not validate_document(doc).ok


# ---------------------------------------------------------------------------
# OPF §3.6 — Out-of-scope docs must carry scope_rationale
# ---------------------------------------------------------------------------


def test_out_of_scope_without_rationale_fails() -> None:
    doc = _minimal()
    doc["corpus"]["documents"].append(
        {"document_id": "stray", "provenance": "our_paper", "in_scope": False, "versions": 1}
    )
    assert "scope_rationale" in _messages(_refresh(doc))


def test_out_of_scope_with_rationale_passes() -> None:
    doc = _minimal()
    doc["corpus"]["documents"].append(
        {
            "document_id": "stray-nda",
            "title": "Stray NDA",
            "provenance": "our_paper",
            "in_scope": False,
            "versions": 1,
            "scope_rationale": "Non-disclosure agreement, not an affiliation agreement.",
        }
    )
    result = validate_document(_refresh(doc))
    assert result.ok, [str(e) for e in result.errors]


# ---------------------------------------------------------------------------
# OPF §4 — Citations required and resolvable
# ---------------------------------------------------------------------------


def test_our_standard_citing_unknown_document_fails() -> None:
    doc = _minimal()
    doc["evidence"]["clauses"][0]["our_standard"]["source_ref"]["document_id"] = "not-in-corpus"
    messages = _messages(_refresh(doc))
    assert "not-in-corpus" in messages and "dangling" in messages


def test_our_standard_missing_citation_fails() -> None:
    doc = _minimal()
    del doc["evidence"]["clauses"][0]["our_standard"]["source_ref"]
    assert not validate_document(_refresh(doc)).ok


def test_empty_document_id_in_citation_fails() -> None:
    doc = _minimal()
    doc["evidence"]["clauses"][0]["our_standard"]["source_ref"]["document_id"] = ""
    result = validate_document(_refresh(doc))
    assert not result.ok
    assert "empty" in " ".join(e.message for e in result.errors).lower()


def test_citation_version_exceeding_corpus_versions_rejected() -> None:
    doc = _minimal()
    doc["evidence"]["clauses"][0]["our_standard"]["source_ref"] = {
        "document_id": "deal-a",
        "version": 99,
        "clause_path": "4",
    }
    messages = _messages(_refresh(doc)).lower()
    assert "version" in messages and "exceeds" in messages


# ---------------------------------------------------------------------------
# Issue #72 regression — malformed (hand-edited/foreign) documents must
# produce a ValidationResult with blocking errors, never raise.
# ---------------------------------------------------------------------------


def test_null_document_id_citation_and_missing_corpus_document_id_does_not_crash() -> None:
    doc = _minimal()
    doc["evidence"]["clauses"][0]["our_standard"]["source_ref"]["document_id"] = None
    doc["corpus"]["documents"].append({})
    result = validate_document(doc)
    assert not result.ok
    assert any(e.blocking for e in result.errors)


def test_null_our_standard_text_does_not_crash() -> None:
    doc = _minimal()
    doc["evidence"]["clauses"][0]["our_standard"]["text"] = None
    result = validate_document(doc)
    assert not result.ok
    assert any("our_standard.text is empty" in e.message for e in result.errors)


def test_agreement_type_aliases_accepted() -> None:
    doc = _minimal()
    doc["agreement_type"]["aliases"] = ["eaa", "Educational Affiliation"]
    result = validate_document(_refresh(doc))
    assert result.ok, [str(e) for e in result.errors]


def test_empty_posture_and_floor_are_valid() -> None:
    doc = _minimal()
    assert doc["posture"] == {} and doc["floor"] == {}
    assert validate_document(doc).ok


# ---------------------------------------------------------------------------
# _check_identity_hash — issue #178
# ---------------------------------------------------------------------------


def _nda() -> dict[str, Any]:
    path = Path(__file__).parent.parent / "examples" / "nda" / "playbook.opf.json"
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


def test_identity_absent_does_not_affect_validity() -> None:
    doc = _minimal()
    assert "identity" not in doc
    assert validate_document(doc).ok


def test_identity_matching_hashes_pass() -> None:
    result = validate_document(_nda())
    assert result.ok, [str(e) for e in result.errors]


def test_identity_stale_content_hash_fails_blocking() -> None:
    doc = _nda()
    doc["posture"]["system_prompt"] = "hand-edited after stamping"
    result = validate_document(doc)
    assert not result.ok
    offending = [e for e in result.errors if e.path == "identity.content_hash"]
    assert offending and offending[0].blocking


def test_identity_stale_section_digest_fails_blocking() -> None:
    doc = _nda()
    doc["floor"]["invariants"][0]["statement"] = "hand-edited after stamping"
    # Keep content_hash consistent with the edit so only the digest mismatch fires.
    doc["identity"]["content_hash"] = content_hash(doc)
    result = validate_document(doc)
    assert not result.ok
    offending = [e for e in result.errors if e.path == "identity.section_digests.floor"]
    assert offending and offending[0].blocking


def test_identity_missing_optional_curation_digest_is_not_a_mismatch() -> None:
    doc = _nda()
    doc["identity"]["section_digests"].pop("curation", None)
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


# ---------------------------------------------------------------------------
# _check_posture_interview_provenance — issue #133
# ---------------------------------------------------------------------------


def test_posture_with_no_interview_record_gets_should_warn() -> None:
    doc = _nda()
    del doc["posture"]["generation"]["interview"]
    result = validate_document(_refresh(doc))
    offending = [
        e
        for e in result.errors
        if not e.blocking and "interview" in e.message and "§3.6" in e.message
    ]
    assert offending, [str(e) for e in result.errors]
    assert offending[0].path == "posture.generation.interview"


def test_posture_with_empty_interview_list_still_warns() -> None:
    doc = _nda()
    doc["posture"]["generation"]["interview"] = []
    assert "§3.6" in _messages(_refresh(doc))


def test_posture_with_interview_record_suppresses_the_warning() -> None:
    assert not any(
        "§3.6" in e.message and "interview" in e.message for e in validate_document(_nda()).errors
    )


def test_empty_posture_does_not_trigger_interview_provenance_warning() -> None:
    result = validate_document(_minimal())
    assert not any("interview" in e.message and "§3.6" in e.message for e in result.errors)
    assert result.ok


# ---------------------------------------------------------------------------
# _check_perspective_present — issue #212
# ---------------------------------------------------------------------------


def test_missing_perspective_gets_should_warn() -> None:
    doc = _minimal()
    del doc["perspective"]
    result = validate_document(_refresh(doc))
    offending = [e for e in result.errors if e.path == "perspective"]
    assert offending, [str(e) for e in result.errors]
    assert "§3.1" in offending[0].message
    assert all(not e.blocking for e in offending)
    assert result.ok, [str(e) for e in result.errors]


def test_perspective_present_suppresses_the_warning() -> None:
    assert not any(e.path == "perspective" for e in validate_document(_minimal()).errors)


def test_partial_perspective_is_a_blocking_schema_error_not_a_warn() -> None:
    doc = _minimal()
    del doc["perspective"]["counterparty_type"]
    result = validate_document(doc)
    assert not result.ok
    assert not any(
        e.path == "perspective" and not e.blocking and "§3.1" in e.message for e in result.errors
    )


# ---------------------------------------------------------------------------
# YAML input / load_opf_file / CLI
# ---------------------------------------------------------------------------


def test_yaml_input_accepted(tmp_path: Path) -> None:
    import yaml

    yaml_path = tmp_path / "playbook.yaml"
    yaml_path.write_text(yaml.dump(_minimal()), encoding="utf-8")
    result = validate_document(load_opf_file(yaml_path))
    assert result.ok, [str(e) for e in result.errors]


def test_load_non_object_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="Expected a JSON/YAML object"):
        load_opf_file(bad)


def test_cli_validate_valid_file(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from playbook_engine.cli import cli

    dest = tmp_path / "playbook.json"
    dest.write_text(json.dumps(_minimal()))
    result = CliRunner().invoke(cli, ["validate", str(dest)])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("version", ["0.2", "0.3"])
def test_cli_validate_rejects_a_retired_version(tmp_path: Path, version: str) -> None:
    from click.testing import CliRunner

    from playbook_engine.cli import cli

    doc = _minimal()
    doc["opf_version"] = version
    dest = tmp_path / "old.json"
    dest.write_text(json.dumps(doc))
    result = CliRunner().invoke(cli, ["validate", str(dest)])
    assert result.exit_code != 0
    assert f"unsupported opf_version '{version}'" in result.output


def test_cli_validate_invalid_file(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from playbook_engine.cli import cli

    dest = tmp_path / "bad.json"
    dest.write_text(json.dumps(_load("invalid_missing_opf_version.json")))
    result = CliRunner().invoke(cli, ["validate", str(dest)])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Invisible-character check (pre-derivation QA)
# ---------------------------------------------------------------------------


def test_validate_rejects_zero_width_chars() -> None:
    doc = _minimal()
    doc["agreement_type"]["name"] = "Educational\u200b Affiliation"
    result = validate_document(doc)
    assert not result.ok
    assert any("zero-width" in str(e) for e in result.errors)


def test_validate_reports_invisible_char_path() -> None:
    doc = _minimal()
    doc["agreement_type"]["name"] = "Name\ufeff"
    offending = [e for e in validate_document(doc).errors if "zero-width" in str(e)]
    assert offending and "agreement_type.name" in offending[0].path


# ---------------------------------------------------------------------------
# Duplicate id check (issue #70)
# ---------------------------------------------------------------------------


def test_validate_rejects_duplicate_clause_id() -> None:
    doc = _minimal()
    doc["evidence"]["clauses"].append(copy.deepcopy(doc["evidence"]["clauses"][0]))
    result = validate_document(doc)
    assert not result.ok
    offending = [e for e in result.errors if "duplicate clause id" in e.message]
    assert offending and offending[0].path == "evidence.clauses[1].id"


def test_validate_rejects_duplicate_corpus_document_id() -> None:
    doc = _minimal()
    doc["corpus"]["documents"].append(copy.deepcopy(doc["corpus"]["documents"][0]))
    result = validate_document(doc)
    assert not result.ok
    offending = [e for e in result.errors if "duplicate corpus document_id" in e.message]
    assert offending and offending[0].path == "corpus.documents[1].document_id"


# ---------------------------------------------------------------------------
# _check_floor_attribution — issue #127
# ---------------------------------------------------------------------------


def _with_invariant(rationale: str | None = None, **extra: Any) -> dict[str, Any]:
    doc = _minimal()
    invariant: dict[str, Any] = {
        "id": "no-uncapped-liability",
        "statement": "Never accept uncapped liability.",
        **extra,
    }
    if rationale is not None:
        invariant["rationale"] = rationale
    doc["floor"] = {"invariants": [invariant]}
    return doc


def test_floor_invariant_with_no_attribution_gets_should_warn() -> None:
    result = validate_document(_with_invariant("Our own rationale."))
    offending = [e for e in result.errors if "structural attribution" in e.message]
    assert offending, [str(e) for e in result.errors]
    assert offending[0].path == "floor.invariants[0]"
    assert "no-uncapped-liability" in offending[0].message
    assert all(not e.blocking for e in offending)


def test_floor_invariant_signed_by_suppresses_the_warning() -> None:
    result = validate_document(_with_invariant("Ours.", x_signed_by="Test Legal Owner"))
    assert not any("structural attribution" in e.message for e in result.errors)


def test_floor_invariant_posture_interview_marker_suppresses_the_warning() -> None:
    result = validate_document(
        _with_invariant(
            "Authored by the legal owner in posture interview v1, question sacred_clauses."
        )
    )
    assert not any("structural attribution" in e.message for e in result.errors)


def test_floor_invariant_review_feedback_marker_suppresses_the_warning() -> None:
    result = validate_document(
        _with_invariant(
            "Proposed then reversed before signing. Accepted via review feedback "
            "(floor candidate cand-001)."
        )
    )
    assert not any("structural attribution" in e.message for e in result.errors)


def test_floor_invariant_agent_typed_rationale_still_warns() -> None:
    result = validate_document(
        _with_invariant("Hand-authored and signed by the legal owner, 2026-08-21.")
    )
    assert any("structural attribution" in e.message for e in result.errors)


# ---------------------------------------------------------------------------
# Malformed containers must yield a ValidationResult, never raise (issue #70)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation",
    [
        {"floor": {"invariants": None}},
        {"floor": {"invariants": ["a"]}},
        {"evidence": {"clauses": ["a"], "precedent": []}},
        {"evidence": {"clauses": [None], "precedent": [None]}},
        {"evidence": {"clauses": [{"id": "a"}, 5], "precedent": ["x", 5]}},
        {"corpus": {"documents": ["a", None]}},
    ],
)
def test_validate_tolerates_malformed_containers(mutation: dict[str, Any]) -> None:
    doc = _minimal()
    doc.update(mutation)
    result = validate_document(doc)
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
