"""Tests for the playbook assembler (L5, issue #25).

Acceptance criterion: assemble_playbook() produces a schema-valid playbook
from the observations of a small fixture corpus.

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional party/document names only
(e.g., "Alice", "Bob", "Acme Corp", "Beta LLC").
"""

from __future__ import annotations

import json
import re

import pytest

from playbook_engine.clause_library_compiler import compile_clause_library
from playbook_engine.clause_position_compiler import (
    compile_clause_positions,
)
from playbook_engine.deviation_classifier import RiskDelta
from playbook_engine.observation_builder import Observation, ObservationCitation
from playbook_engine.playbook_assembler import (
    AssemblyError,
    assemble_playbook,
    write_playbook,
)
from playbook_engine.validator import validate_document

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GENERATED_AT = "2024-01-15T10:30:00Z"

_AGREEMENT_TYPE = {
    "id": "educational-affiliation",
    "name": "Educational Affiliation Agreement",
    "description": "Governs clinical placements at academic institutions.",
}

_BASELINE = {
    "has_canonical_template": True,
    "template_ref": {
        "document_id": "template",
        "title": "FixtureCorp Standard EAA Template v2",
        "source": "corpus/template.docx",
    },
    "notes": "Standard template used since 2020.",
}

_TAXONOMY = {
    "source": "CUAD-v1",
    "entries": [
        {
            "id": "indemnification",
            "label": "Indemnification",
            "status": "active",
            "cuad_origin": "Indemnification",
            "description": "Who bears third-party claim risk.",
        },
        {
            "id": "governing_law",
            "label": "Governing Law",
            "status": "active",
            "cuad_origin": "Governing Laws",
            "description": "Which law governs the agreement.",
        },
    ],
}

_NEUTRAL = RiskDelta(direction="neutral", magnitude="none")
_WORSE_MINOR = RiskDelta(direction="worse", magnitude="minor")


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


def _template_obs(taxonomy_id: str, clause_path: str = "8") -> Observation:
    return _obs(
        taxonomy_id,
        provenance="our_paper",
        doc_id="template",
        version="template",
        clause_path=clause_path,
    )


def _corpus_doc(
    doc_id: str,
    provenance: str = "our_paper",
    in_scope: bool = True,
    versions: int = 3,
    scope_rationale: str | None = None,
) -> dict:
    d: dict = {
        "document_id": doc_id,
        "provenance": provenance,
        "in_scope": in_scope,
        "versions": versions,
        "signed_version": versions,
        "version_order_basis": "edit_distance_chain",
    }
    if scope_rationale is not None:
        d["scope_rationale"] = scope_rationale
    elif not in_scope:
        d["scope_rationale"] = "Not an EAA — excluded at scope gate."
    return d


def _minimal_playbook(
    obs_list: list[Observation] | None = None,
    cp_obs_list: list[Observation] | None = None,
    corpus_docs: list[dict] | None = None,
    run_id: str | None = None,
    scope_bases: list[str] | None = None,
    perspective: dict[str, str] | None = None,
    de_minimis: list[str] | None = None,
    playbook_id: str | None = None,
    playbook_version: str | None = None,
    supersedes: str | None = None,
) -> dict:
    """Helper: build and return a schema-valid playbook dict."""
    deal_obs = obs_list or [
        _obs("indemnification"),
        _obs("governing_law", clause_path="12"),
    ]
    template_obs = [_template_obs("indemnification"), _template_obs("governing_law", "12")]
    all_obs = deal_obs + (cp_obs_list or [])

    positions, _, _ = compile_clause_positions(all_obs, template_obs)
    library, _ = compile_clause_library(all_obs)
    docs = corpus_docs or [_corpus_doc("deal_001")]

    return assemble_playbook(
        agreement_type=_AGREEMENT_TYPE,
        baseline=_BASELINE,
        taxonomy=_TAXONOMY,
        clause_positions=positions,
        clause_library=library,
        corpus_documents=docs,
        generated_at=_GENERATED_AT,
        run_id=run_id,
        observations=all_obs,
        scope_bases=scope_bases,
        perspective=perspective,
        de_minimis=de_minimis,
        playbook_id=playbook_id,
        playbook_version=playbook_version,
        supersedes=supersedes,
    )


# ---------------------------------------------------------------------------
# Acceptance criterion: produces a schema-valid playbook
# ---------------------------------------------------------------------------


def test_assemble_produces_schema_valid_playbook() -> None:
    """Acceptance: assemble_playbook() produces a schema-valid OPF document."""
    playbook = _minimal_playbook()
    result = validate_document(playbook)
    # No blocking errors
    errors = [str(e) for e in result.errors if e.blocking]
    assert errors == [], f"Blocking validation errors: {errors}"


def test_assemble_small_corpus_end_to_end() -> None:
    """Acceptance: multi-clause corpus with both provenances assembles cleanly."""
    deal_obs = [
        _obs("indemnification", provenance="our_paper", deviation="none"),
        _obs(
            "indemnification",
            provenance="our_paper",
            deviation="substantive",
            risk_delta=_WORSE_MINOR,
            outcome="signed",
            doc_id="deal_002",
            version="v1",
        ),
        _obs(
            "governing_law",
            provenance="counterparty_paper",
            deviation="substantive",
            risk_delta=_WORSE_MINOR,
            text="Counterparty home-state law.",
            clause_path="12",
        ),
    ]
    template_obs = [_template_obs("indemnification"), _template_obs("governing_law", "12")]
    positions, _, _ = compile_clause_positions(deal_obs, template_obs)
    library, _ = compile_clause_library(deal_obs)
    docs = [_corpus_doc("deal_001"), _corpus_doc("deal_002")]

    playbook = assemble_playbook(
        agreement_type=_AGREEMENT_TYPE,
        baseline=_BASELINE,
        taxonomy=_TAXONOMY,
        clause_positions=positions,
        clause_library=library,
        corpus_documents=docs,
        generated_at=_GENERATED_AT,
    )
    result = validate_document(playbook)
    errors = [str(e) for e in result.errors if e.blocking]
    assert errors == [], errors


# ---------------------------------------------------------------------------
# Document structure
# ---------------------------------------------------------------------------


def test_assemble_top_level_keys() -> None:
    """All required (OPF v0.2 schema) top-level keys are present."""
    playbook = _minimal_playbook()
    required = {
        "opf_version",
        "agreement_type",
        "baseline",
        "taxonomy",
        "evidence",
        "posture",
        "floor",
        "corpus",
        "compiler",
    }
    assert required.issubset(playbook.keys())


def test_assemble_opf_version() -> None:
    assert _minimal_playbook()["opf_version"] == "0.2"


def test_assemble_agreement_type_preserved() -> None:
    pb = _minimal_playbook()
    assert pb["agreement_type"]["id"] == "educational-affiliation"
    assert pb["agreement_type"]["name"] == "Educational Affiliation Agreement"


def test_assemble_baseline_preserved() -> None:
    pb = _minimal_playbook()
    assert pb["baseline"]["has_canonical_template"] is True
    assert pb["baseline"]["template_ref"]["document_id"] == "template"


def test_assemble_taxonomy_preserved() -> None:
    pb = _minimal_playbook()
    assert pb["taxonomy"]["source"] == "CUAD-v1"
    assert len(pb["taxonomy"]["entries"]) == 2


def test_assemble_clauses_present() -> None:
    """OPF v0.2 (§3.5): clauses live under `evidence`, not top-level."""
    pb = _minimal_playbook()
    assert isinstance(pb["evidence"]["clauses"], list)
    assert len(pb["evidence"]["clauses"]) > 0


def test_assemble_clause_library_present() -> None:
    pb = _minimal_playbook()
    assert "clause_library" in pb["evidence"]
    assert isinstance(pb["evidence"]["clause_library"], list)


def test_assemble_clauses_carry_historical_stance() -> None:
    """OPF v0.2 (§3.5, §2.2): every clause's `summary` carries the
    descriptive `historical_stance`, not v0.1's prescriptive `rollup.position`."""
    pb = _minimal_playbook()
    for clause in pb["evidence"]["clauses"]:
        assert "summary" in clause
        assert "historical_stance" in clause["summary"]
        assert "rollup" not in clause


def test_assemble_posture_and_floor_empty_but_present() -> None:
    """OPF v0.2 (§3.6/§3.7): Posture/Floor are structurally present even when
    no interview has been run and no invariants authored (issue #140 scope
    excludes Floor invariant content)."""
    pb = _minimal_playbook()
    assert pb["posture"] == {}
    assert pb["floor"] == {}


def test_assemble_perspective_omitted_when_not_supplied() -> None:
    """perspective/de_minimis are optional (issue #140: no config source
    exists yet) — must never be fabricated, so they're omitted, not defaulted
    to placeholder values, when the caller supplies nothing."""
    pb = _minimal_playbook()
    assert "perspective" not in pb
    assert "de_minimis" not in pb


def test_assemble_perspective_and_de_minimis_passed_through() -> None:
    pb = _minimal_playbook(
        perspective={"party": "FixtureCorp", "counterparty_type": "Educational Institution"},
        de_minimis=["typo fixes"],
    )
    assert pb["perspective"] == {
        "party": "FixtureCorp",
        "counterparty_type": "Educational Institution",
    }
    assert pb["de_minimis"] == ["typo fixes"]
    result = validate_document(pb)
    errors = [str(e) for e in result.errors if e.blocking]
    assert errors == [], errors


def test_assemble_compiler_fields() -> None:
    pb = _minimal_playbook(run_id="run-abc-123")
    comp = pb["compiler"]
    assert comp["name"] == "playbook-engine"
    assert "version" in comp
    assert comp["generated_at"] == _GENERATED_AT
    assert comp["run_id"] == "run-abc-123"


def test_assemble_compiler_no_run_id_when_none() -> None:
    pb = _minimal_playbook()
    assert "run_id" not in pb["compiler"]


# ---------------------------------------------------------------------------
# identity — issue #143
# ---------------------------------------------------------------------------

_HASH_RE = r"^sha256:[0-9a-f]{64}$"


def test_assemble_identity_present_with_content_hash_and_section_digests() -> None:
    pb = _minimal_playbook()
    identity = pb["identity"]
    assert re.match(_HASH_RE, identity["content_hash"])
    # Issue #147: "curation" is a fourth digest, always computed.
    assert set(identity["section_digests"].keys()) == {"evidence", "posture", "floor", "curation"}
    for h in identity["section_digests"].values():
        assert re.match(_HASH_RE, h)


def test_assemble_identity_id_version_supersedes_omitted_when_not_supplied() -> None:
    """Like perspective/de_minimis, id/version/supersedes are producer-
    assigned lineage the engine cannot derive — never fabricated."""
    pb = _minimal_playbook()
    identity = pb["identity"]
    assert "id" not in identity
    assert "version" not in identity
    assert "supersedes" not in identity


def test_assemble_identity_id_version_supersedes_passed_through() -> None:
    pb = _minimal_playbook(
        playbook_id="eiaa-fixturecorp",
        playbook_version="1.0.0",
        supersedes="eiaa-fixturecorp@0.9.0",
    )
    identity = pb["identity"]
    assert identity["id"] == "eiaa-fixturecorp"
    assert identity["version"] == "1.0.0"
    assert identity["supersedes"] == "eiaa-fixturecorp@0.9.0"
    result = validate_document(pb)
    errors = [str(e) for e in result.errors if e.blocking]
    assert errors == [], errors


def test_assemble_identity_content_hash_stable_across_run_id_and_generated_at() -> None:
    """Two compiles of the same corpus content but different run_id/generated_at
    (e.g. a re-run a minute later) must produce the same content_hash."""
    pb_a = _minimal_playbook(run_id="run-1")
    pb_b = assemble_playbook(
        agreement_type=_AGREEMENT_TYPE,
        baseline=_BASELINE,
        taxonomy=_TAXONOMY,
        clause_positions=compile_clause_positions(
            [_obs("indemnification"), _obs("governing_law", clause_path="12")],
            [_template_obs("indemnification"), _template_obs("governing_law", "12")],
        )[0],
        clause_library=compile_clause_library(
            [_obs("indemnification"), _obs("governing_law", clause_path="12")]
        )[0],
        corpus_documents=[_corpus_doc("deal_001")],
        generated_at="2099-01-01T00:00:00Z",
        run_id="run-2-completely-different",
    )
    assert pb_a["identity"]["content_hash"] == pb_b["identity"]["content_hash"]


def test_assemble_identity_content_hash_changes_when_corpus_content_changes() -> None:
    pb_a = _minimal_playbook()
    pb_b = _minimal_playbook(corpus_docs=[_corpus_doc("deal_001", versions=99)])
    assert pb_a["identity"]["content_hash"] != pb_b["identity"]["content_hash"]


def test_assemble_identity_schema_valid() -> None:
    pb = _minimal_playbook()
    result = validate_document(pb)
    errors = [str(e) for e in result.errors if e.blocking]
    assert errors == [], errors


# ---------------------------------------------------------------------------
# stub-basis watermark (issue #101)
# ---------------------------------------------------------------------------


def test_assemble_compiler_stub_watermark_false_by_default() -> None:
    """No stub-basis observations → the playbook is not watermarked."""
    pb = _minimal_playbook()
    assert pb["compiler"]["stub_basis_present"] is False


def test_assemble_compiler_stub_watermark_true_when_observation_is_stub() -> None:
    """issue #101: when any observation fed into the playbook has
    basis="stub" (no judge configured at all), the assembled playbook's
    compiler block must be watermarked so a consuming review application can
    refuse to run against it without human review."""
    pb = _minimal_playbook(
        obs_list=[
            _obs("indemnification", basis="stub"),
            _obs("governing_law", clause_path="12"),
        ]
    )
    assert pb["compiler"]["stub_basis_present"] is True


def test_assemble_compiler_stub_watermark_schema_valid() -> None:
    """The watermarked playbook must still be schema-valid (stub_basis_present
    is a declared optional property, not an ad hoc extra field)."""
    pb = _minimal_playbook(obs_list=[_obs("indemnification", basis="stub")])
    result = validate_document(pb)
    errors = [str(e) for e in result.errors if e.blocking]
    assert errors == [], errors


def test_assemble_compiler_stub_watermark_true_when_observation_needs_review() -> None:
    """issue #101: the *default zero-LLM* deviation stub (``_NullDeviationJudge``)
    emits basis="needs_review", never "stub" (a judge protocol IS wired, it's
    just the stub default) — so the watermark must also fire on
    "needs_review" (and "judge_error"), not only the strict "stub" basis, or
    a real default ``playbook compile`` run never watermarks its output."""
    pb = _minimal_playbook(
        obs_list=[
            _obs("indemnification", basis="needs_review"),
            _obs("governing_law", clause_path="12"),
        ]
    )
    assert pb["compiler"]["stub_basis_present"] is True


def test_assemble_compiler_stub_watermark_true_when_scope_basis_is_stub() -> None:
    """issue #101: the default zero-LLM scope stub (``_AllInScopeJudge``)
    puts basis="stub" on the ScopeDecision, never on an Observation — the
    watermark must also fire from ``scope_bases`` (threaded in from
    scope.json by the caller), or a default compile with an otherwise fully
    LLM-judged deviation path still fails to watermark."""
    pb = _minimal_playbook(scope_bases=["stub"])
    assert pb["compiler"]["stub_basis_present"] is True


def test_assemble_compiler_stub_watermark_false_with_only_judge_scope_basis() -> None:
    """A real (non-stub) scope basis must not trigger the watermark."""
    pb = _minimal_playbook(scope_bases=["judge", "deterministic_empty"])
    assert pb["compiler"]["stub_basis_present"] is False


def test_assemble_no_observations_arg_watermarks_false() -> None:
    """Backward compatibility: omitting ``observations`` entirely (the
    pre-#101 call signature) must not raise and must watermark False."""
    positions, _, _ = compile_clause_positions(
        [_obs("indemnification")], [_template_obs("indemnification")]
    )
    library, _ = compile_clause_library([_obs("indemnification")])
    playbook = assemble_playbook(
        agreement_type=_AGREEMENT_TYPE,
        baseline=_BASELINE,
        taxonomy=_TAXONOMY,
        clause_positions=positions,
        clause_library=library,
        corpus_documents=[_corpus_doc("deal_001")],
        generated_at=_GENERATED_AT,
    )
    assert playbook["compiler"]["stub_basis_present"] is False


# ---------------------------------------------------------------------------
# corpus stats auto-computation
# ---------------------------------------------------------------------------


def test_corpus_stats_total_documents() -> None:
    docs = [_corpus_doc("deal_001"), _corpus_doc("deal_002"), _corpus_doc("deal_003")]
    pb = _minimal_playbook(corpus_docs=docs)
    assert pb["corpus"]["stats"]["documents_total"] == 3


def test_corpus_stats_in_scope_count() -> None:
    docs = [
        _corpus_doc("deal_001", in_scope=True),
        _corpus_doc("deal_002", in_scope=True),
        _corpus_doc("deal_003", in_scope=False),
    ]
    pb = _minimal_playbook(corpus_docs=docs)
    assert pb["corpus"]["stats"]["documents_in_scope"] == 2


def test_corpus_stats_versions_total() -> None:
    docs = [_corpus_doc("deal_001", versions=5), _corpus_doc("deal_002", versions=3)]
    pb = _minimal_playbook(corpus_docs=docs)
    assert pb["corpus"]["stats"]["versions_total"] == 8


def test_out_of_scope_docs_retained_in_corpus() -> None:
    """§3.6: out-of-scope docs MUST appear in corpus with scope_rationale."""
    docs = [
        _corpus_doc("deal_001", in_scope=True),
        _corpus_doc(
            "deal_oos", in_scope=False, scope_rationale="Not an EAA — excluded at scope gate."
        ),
    ]
    pb = _minimal_playbook(corpus_docs=docs)
    corpus_ids = [d["document_id"] for d in pb["corpus"]["documents"]]
    assert "deal_oos" in corpus_ids


# ---------------------------------------------------------------------------
# Validation enforcement
# ---------------------------------------------------------------------------


def test_assemble_raises_on_invalid_document() -> None:
    """AssemblyError raised when validation fails (blocking errors)."""
    # Pass an invalid taxonomy entry with bad status to trigger schema error.
    bad_taxonomy = {
        "source": "CUAD-v1",
        "entries": [
            {"id": "indemnification", "label": "Indemnification", "status": "INVALID_STATUS"}
        ],
    }
    positions, _, _ = compile_clause_positions([], [])
    library, _ = compile_clause_library([])
    with pytest.raises(AssemblyError) as exc_info:
        assemble_playbook(
            agreement_type=_AGREEMENT_TYPE,
            baseline=_BASELINE,
            taxonomy=bad_taxonomy,
            clause_positions=positions,
            clause_library=library,
            corpus_documents=[_corpus_doc("deal_001")],
            generated_at=_GENERATED_AT,
        )
    assert exc_info.value.blocking_errors  # at least one blocking error


def test_assemble_error_message_contains_error_info() -> None:
    """AssemblyError.__str__ includes error information."""
    bad_taxonomy = {
        "source": "CUAD-v1",
        "entries": [{"id": "ind", "label": "Ind", "status": "bad"}],
    }
    positions, _, _ = compile_clause_positions([], [])
    library, _ = compile_clause_library([])
    with pytest.raises(AssemblyError) as exc_info:
        assemble_playbook(
            agreement_type=_AGREEMENT_TYPE,
            baseline=_BASELINE,
            taxonomy=bad_taxonomy,
            clause_positions=positions,
            clause_library=library,
            corpus_documents=[_corpus_doc("deal_001")],
            generated_at=_GENERATED_AT,
        )
    err_str = str(exc_info.value)
    assert "validation" in err_str.lower()


def test_assemble_out_of_scope_without_rationale_raises() -> None:
    """§3.6: out-of-scope doc without scope_rationale must fail validation."""
    docs = [{"document_id": "deal_oos", "provenance": "our_paper", "in_scope": False}]
    positions, _, _ = compile_clause_positions([], [])
    library, _ = compile_clause_library([])
    with pytest.raises(AssemblyError):
        assemble_playbook(
            agreement_type=_AGREEMENT_TYPE,
            baseline=_BASELINE,
            taxonomy=_TAXONOMY,
            clause_positions=positions,
            clause_library=library,
            corpus_documents=docs,
            generated_at=_GENERATED_AT,
        )


# ---------------------------------------------------------------------------
# write_playbook
# ---------------------------------------------------------------------------


def test_write_playbook_creates_file(tmp_path) -> None:
    playbook = _minimal_playbook()
    out = tmp_path / "out" / "playbook.opf.json"
    write_playbook(playbook, out)
    assert out.exists()


def test_write_playbook_valid_json(tmp_path) -> None:
    playbook = _minimal_playbook()
    out = tmp_path / "playbook.opf.json"
    write_playbook(playbook, out)
    parsed = json.loads(out.read_text())
    assert parsed["opf_version"] == "0.2"


def test_write_playbook_atomic_no_tmp_left(tmp_path) -> None:
    """No .json.tmp file left after successful write."""
    playbook = _minimal_playbook()
    out = tmp_path / "playbook.opf.json"
    write_playbook(playbook, out)
    assert not (out.with_suffix(".json.tmp")).exists()


def test_write_playbook_creates_parent_dirs(tmp_path) -> None:
    out = tmp_path / "deep" / "nested" / "playbook.opf.json"
    write_playbook(_minimal_playbook(), out)
    assert out.exists()


def test_write_playbook_pretty_printed(tmp_path) -> None:
    """Written JSON is indented (pretty-printed) for human readability."""
    out = tmp_path / "playbook.opf.json"
    write_playbook(_minimal_playbook(), out)
    text = out.read_text()
    # Indented JSON has newlines and spaces
    assert "\n" in text
    assert "  " in text


def test_write_playbook_round_trip(tmp_path) -> None:
    """Written and re-read playbook is structurally identical."""
    playbook = _minimal_playbook()
    out = tmp_path / "playbook.opf.json"
    write_playbook(playbook, out)
    loaded = json.loads(out.read_text())
    assert loaded["evidence"]["clauses"] == playbook["evidence"]["clauses"]
    assert loaded["compiler"]["generated_at"] == _GENERATED_AT
