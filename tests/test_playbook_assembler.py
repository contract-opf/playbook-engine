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
from pathlib import Path

import pytest

from playbook_engine.clause_position_compiler import (
    compile_clause_positions,
)
from playbook_engine.digest import build_digest
from playbook_engine.observation_builder import Observation, ObservationCitation
from playbook_engine.playbook_assembler import (
    _VERSION_INGEST_SCHEMA_KEYS,
    AssemblyError,
    _sanitize_corpus_documents_for_schema,
    assemble_playbook,
    write_playbook,
)
from playbook_engine.validator import validate_document

_REPO_ROOT = Path(__file__).parent.parent

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

_NEUTRAL = {"direction": "neutral", "magnitude": "none"}


def _obs(
    taxonomy_id: str | None,
    provenance: str = "our_paper",
    outcome: str = "signed",
    deviation: str = "none",
    risk_delta: dict[str, str] = _NEUTRAL,
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
        risk_delta=dict(risk_delta),
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
    docs = corpus_docs or [_corpus_doc("deal_001")]

    return assemble_playbook(
        agreement_type=_AGREEMENT_TYPE,
        baseline=_BASELINE,
        taxonomy=_TAXONOMY,
        clause_positions=positions,
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
            outcome="signed",
            doc_id="deal_002",
            version="v1",
        ),
        _obs(
            "governing_law",
            provenance="counterparty_paper",
            deviation="substantive",
            text="Counterparty home-state law.",
            clause_path="12",
        ),
    ]
    template_obs = [_template_obs("indemnification"), _template_obs("governing_law", "12")]
    positions, _, _ = compile_clause_positions(deal_obs, template_obs)
    docs = [_corpus_doc("deal_001"), _corpus_doc("deal_002")]

    playbook = assemble_playbook(
        agreement_type=_AGREEMENT_TYPE,
        baseline=_BASELINE,
        taxonomy=_TAXONOMY,
        clause_positions=positions,
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
    """All required (OPF 0.5 schema) top-level keys are present."""
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
    """Issue #238: the assembler emits exactly one format, OPF 0.5, and takes
    no version argument."""
    import inspect

    pb = _minimal_playbook()
    assert pb["opf_version"] == "0.5"
    assert pb["digest"]["digest_version"] == "3"
    assert "opf_version" not in inspect.signature(assemble_playbook).parameters


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
    """Clauses live under `evidence`, not top-level."""
    pb = _minimal_playbook()
    assert isinstance(pb["evidence"]["clauses"], list)
    assert len(pb["evidence"]["clauses"]) > 0


def test_assemble_evidence_is_clauses_and_precedent() -> None:
    """Issue #223: evidence is {clauses, precedent} — no clause library."""
    assert set(_minimal_playbook()["evidence"]) == {"clauses", "precedent"}


def test_assemble_clauses_carry_counts_and_no_stance() -> None:
    """Issue #223: a clause carries no stance at all — only counts."""
    for clause in _minimal_playbook()["evidence"]["clauses"]:
        assert set(clause) == {
            "id",
            "taxonomy_id",
            "title",
            "our_standard",
            "n_deals",
            "n_signed_standard",
            "n_variants",
            "n_refused",
        }


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
    # OPF 0.5 has no curation section (issue #233), so no fourth digest.
    assert set(identity["section_digests"].keys()) == {"evidence", "posture", "floor"}
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
    """No stub scope decision → the playbook is not watermarked."""
    pb = _minimal_playbook()
    assert pb["compiler"]["stub_basis_present"] is False


def test_assemble_compiler_stub_watermark_schema_valid() -> None:
    """The watermarked playbook must still be schema-valid (stub_basis_present
    is a declared optional property, not an ad hoc extra field)."""
    pb = _minimal_playbook(scope_bases=["stub"])
    assert pb["compiler"]["stub_basis_present"] is True
    result = validate_document(pb)
    errors = [str(e) for e in result.errors if e.blocking]
    assert errors == [], errors


def test_assemble_compiler_stub_watermark_true_when_scope_basis_is_stub() -> None:
    """issue #101: the default zero-LLM scope stub (``_AllInScopeJudge``)
    puts basis="stub" on the ScopeDecision — the watermark fires from
    ``scope_bases`` (threaded in from scope.json by the caller). An
    observation's basis never watermarks: its deviation is the deterministic
    standard check (issue #239)."""
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
    playbook = assemble_playbook(
        agreement_type=_AGREEMENT_TYPE,
        baseline=_BASELINE,
        taxonomy=_TAXONOMY,
        clause_positions=positions,
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


def test_corpus_stats_sum_dropped_observations_and_strip_internal_key() -> None:
    """Issue #216: pipeline._compute_doc_result records each document's
    dropped net-diff rows by reason (text with no signed slot that survives
    in the signed version; text removed before signing whose origin cannot
    be determined) as ``dropped_observations``; the assembler sums them into
    corpus.stats and
    never ships the engine-internal key inside corpus.documents[] (whose
    schema is additionalProperties:false)."""
    d1 = _corpus_doc("deal_001")
    d1["dropped_observations"] = {"survives_in_terminal": 2}
    d2 = _corpus_doc("deal_002")
    d2["dropped_observations"] = {}
    d3 = _corpus_doc("deal_003")
    d3["dropped_observations"] = {"survives_in_terminal": 1, "removed_origin_undetermined": 1}
    pb = _minimal_playbook(corpus_docs=[d1, d2, d3])
    assert pb["corpus"]["stats"]["dropped_observations"] == {
        "count": 4,
        "by_reason": {"removed_origin_undetermined": 1, "survives_in_terminal": 3},
        "by_document": {"deal_001": 2, "deal_003": 2},
    }
    assert all("dropped_observations" not in d for d in pb["corpus"]["documents"])


def test_corpus_stats_omit_dropped_observations_when_nothing_dropped() -> None:
    d1 = _corpus_doc("deal_001")
    d1["dropped_observations"] = {}
    pb = _minimal_playbook(corpus_docs=[d1, _corpus_doc("deal_002")])
    assert "dropped_observations" not in pb["corpus"]["stats"]


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
    with pytest.raises(AssemblyError) as exc_info:
        assemble_playbook(
            agreement_type=_AGREEMENT_TYPE,
            baseline=_BASELINE,
            taxonomy=bad_taxonomy,
            clause_positions=positions,
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
    with pytest.raises(AssemblyError) as exc_info:
        assemble_playbook(
            agreement_type=_AGREEMENT_TYPE,
            baseline=_BASELINE,
            taxonomy=bad_taxonomy,
            clause_positions=positions,
            corpus_documents=[_corpus_doc("deal_001")],
            generated_at=_GENERATED_AT,
        )
    err_str = str(exc_info.value)
    assert "validation" in err_str.lower()


def test_assemble_out_of_scope_without_rationale_raises() -> None:
    """§3.6: out-of-scope doc without scope_rationale must fail validation."""
    docs = [{"document_id": "deal_oos", "provenance": "our_paper", "in_scope": False}]
    positions, _, _ = compile_clause_positions([], [])
    with pytest.raises(AssemblyError):
        assemble_playbook(
            agreement_type=_AGREEMENT_TYPE,
            baseline=_BASELINE,
            taxonomy=_TAXONOMY,
            clause_positions=positions,
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
    assert parsed["opf_version"] == "0.5"


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


# ---------------------------------------------------------------------------
# Invisible-character stripping (pre-derivation QA)
# ---------------------------------------------------------------------------


def test_assemble_strips_zero_width_chars_from_all_strings() -> None:
    """Zero-width/bidi chars from extraction must never reach the document."""
    dirty = _obs(
        "indemnification",
        text="Mutual​ indemnification﻿ obligations‪.",
    )
    playbook = _minimal_playbook(obs_list=[dirty, _obs("governing_law", clause_path="12")])
    serialized = json.dumps(playbook, ensure_ascii=False)
    assert not re.search("[​-‍﻿‪-‮]", serialized)
    # The visible text survives, minus the invisible characters.
    assert "Mutual indemnification obligations." in serialized


def test_assemble_content_hash_covers_stripped_content() -> None:
    """content_hash must be computed AFTER stripping (hash of what is written)."""
    dirty = _obs("indemnification", text="Indemnification​ text.")
    clean = _obs("indemnification", text="Indemnification text.")
    dirty_doc = _minimal_playbook(obs_list=[dirty, _obs("governing_law", clause_path="12")])
    clean_doc = _minimal_playbook(obs_list=[clean, _obs("governing_law", clause_path="12")])
    assert dirty_doc["identity"]["content_hash"] == clean_doc["identity"]["content_hash"]


def test_assemble_digest_matches_recompute_over_stripped_playbook() -> None:
    """issue #35: embedded digest must be a pure function of the shipped evidence.

    Two observations whose full_text is identical after stripping (one has a
    ZWSP inside a word, the other is its clean twin) must land in the SAME
    dedupe group. If the digest were built before stripping, they would form
    two separate n=1 groups instead of one n=2 group, and recomputing
    build_digest() over the final (stripped) playbook would then disagree
    with the embedded digest.
    """
    dirty = _obs(
        "indemnification",
        text="Mutual inde​mnity obligations.",
        doc_id="deal_002",
        clause_path="8",
    )
    clean = _obs(
        "indemnification",
        text="Mutual indemnity obligations.",
        doc_id="deal_003",
        clause_path="8",
    )
    playbook = _minimal_playbook(
        obs_list=[dirty, clean, _obs("governing_law", clause_path="12")],
        corpus_docs=[_corpus_doc("deal_001"), _corpus_doc("deal_002"), _corpus_doc("deal_003")],
    )
    assert playbook["digest"] == build_digest(playbook)


# ---------------------------------------------------------------------------
# version_ingest schema stripping (issue #81) — corpus_manifest.json/
# ExtractorLabel gained a "reason" field that the published OPF schema does
# not (and, per additionalProperties:false, must not) accept.
# ---------------------------------------------------------------------------


def _version_ingest_schema_properties(schema_filename: str) -> set[str]:
    schema = json.loads((_REPO_ROOT / "spec" / schema_filename).read_text(encoding="utf-8"))
    props = schema["properties"]["corpus"]["properties"]["documents"]["items"]["properties"][
        "version_ingest"
    ]["items"]["properties"]
    return set(props)


def test_version_ingest_schema_keys_matches_schema_0_4() -> None:
    """_VERSION_INGEST_SCHEMA_KEYS (the strip-list assemble_playbook applies
    to every version_ingest entry) must stay in sync with
    spec/playbook.schema-0.5.json's actual property set — the schema
    assemble_playbook's self-validation enforces (issue #81). Used as a
    strip-list, drift in the OTHER direction (a future schema addition
    silently stripped from every published playbook) would otherwise fail
    silently — this test exists so that drift fails LOUDLY instead."""
    assert (
        _version_ingest_schema_properties("playbook.schema-0.5.json") == _VERSION_INGEST_SCHEMA_KEYS
    )


def test_version_ingest_schema_keys_has_no_additional_engine_only_fields() -> None:
    """Sanity check on the guard itself: "reason" (an engine-internal-only
    field the schema does not accept) must NOT be in the whitelist, or the
    strip would be a no-op for the exact field this ticket introduced."""
    assert "reason" not in _VERSION_INGEST_SCHEMA_KEYS


def test_sanitize_corpus_documents_strips_reason_from_version_ingest() -> None:
    corpus_documents = [
        {
            "document_id": "deal_001",
            "in_scope": True,
            "version_ingest": [
                {
                    "version": "v1",
                    "status": "ok",
                    "error": None,
                    "extractor": "legacy",
                    "reason": "backend-error",
                }
            ],
        }
    ]
    sanitized = _sanitize_corpus_documents_for_schema(corpus_documents)
    assert sanitized[0]["version_ingest"][0] == {
        "version": "v1",
        "status": "ok",
        "error": None,
        "extractor": "legacy",
    }
    # Every other field on the document dict survives untouched.
    assert sanitized[0]["document_id"] == "deal_001"
    assert sanitized[0]["in_scope"] is True
    # The original input is not mutated.
    assert "reason" in corpus_documents[0]["version_ingest"][0]


def test_sanitize_corpus_documents_tolerates_missing_version_ingest() -> None:
    """A document with no version_ingest key at all (e.g. an out-of-scope
    document shape predating issue #89) must pass through unchanged, not
    raise."""
    corpus_documents = [{"document_id": "deal_001", "in_scope": True}]
    sanitized = _sanitize_corpus_documents_for_schema(corpus_documents)
    assert sanitized == corpus_documents


def test_assemble_playbook_strips_reason_and_still_validates() -> None:
    """End-to-end: a corpus_documents entry carrying "reason" (as
    corpus_manifest.json now does — issue #81) must not break
    assemble_playbook's self-validation — additionalProperties:false on
    version_ingest.items would otherwise reject the whole document."""
    corpus_docs = [
        {
            **_corpus_doc("deal_001"),
            "version_ingest": [
                {
                    "version": "v1",
                    "status": "ok",
                    "error": None,
                    "extractor": "legacy",
                    "reason": "backend-error",
                },
                {
                    "version": "v2",
                    "status": "ok",
                    "error": None,
                    "extractor": "docling",
                    "reason": None,
                },
            ],
        }
    ]
    playbook = _minimal_playbook(corpus_docs=corpus_docs)

    result = validate_document(playbook)
    errors = [str(e) for e in result.errors if e.blocking]
    assert errors == [], f"Blocking validation errors: {errors}"

    published_ingest = playbook["corpus"]["documents"][0]["version_ingest"]
    assert all("reason" not in vi for vi in published_ingest), (
        "reason must never reach the published playbook.opf.json"
    )
    assert published_ingest[0]["extractor"] == "legacy"
    assert published_ingest[1]["extractor"] == "docling"


# ---------------------------------------------------------------------------
# OPF 0.5 precedent record (issue #223). Observations below are the shapes
# observation_builder writes (one terminal signed/unsigned row per deal and
# clause, proposed_then_reversed rows, conceded_before_signing rows, the
# deterministic `standard` fact); RoundMove is what build_round_moves writes.
# ---------------------------------------------------------------------------


#: A clause text above MIN_OBSERVATION_TEXT_LEN (the default _obs text is a
#: 24-char fragment the compiler quarantines).
_LONG_STD = "Mutual indemnification for third-party claims."


def _assemble_04(
    observations: list[Observation],
    docs: list[dict],
    round_moves: list | None = None,
) -> dict:
    # The template's indemnification text IS _LONG_STD, so a row marked
    # standard=True below is exactly what is_standard_text would compute.
    template_obs = [
        _obs(
            "indemnification",
            doc_id="template",
            version="template",
            clause_path="8",
            text=_LONG_STD,
        ),
        _template_obs("governing_law", "12"),
    ]
    positions, _, _ = compile_clause_positions(observations, template_obs)
    return assemble_playbook(
        agreement_type=_AGREEMENT_TYPE,
        baseline=_BASELINE,
        taxonomy=_TAXONOMY,
        clause_positions=positions,
        corpus_documents=docs,
        generated_at=_GENERATED_AT,
        observations=observations,
        round_moves=round_moves,
    )


def _std(obs: Observation, standard: bool) -> Observation:
    import dataclasses

    return dataclasses.replace(obs, standard=standard)


def test_v05_one_precedent_per_deal_and_clause_with_facts_only() -> None:
    from playbook_engine.observation_builder import RoundMove

    observations = [
        _std(_obs("indemnification", text=_LONG_STD, basis="deterministic"), True),
        _std(
            _obs(
                "indemnification",
                doc_id="deal_002",
                text="Supplier indemnifies only for gross negligence.",
                basis="deterministic",
            ),
            False,
        ),
        _std(
            _obs(
                "indemnification",
                doc_id="deal_002",
                outcome="proposed_then_reversed",
                text="Supplier indemnifies nobody for anything at all.",
                version="v2",
                basis="deterministic",
            ),
            False,
        ),
        _std(
            _obs(
                "governing_law",
                doc_id="deal_003",
                outcome="unsigned",
                clause_path="12",
                text="Delaware law governs this agreement.",
                basis="deterministic",
            ),
            False,
        ),
    ]
    docs = [_corpus_doc("deal_001"), _corpus_doc("deal_002"), _corpus_doc("deal_003")]
    docs[2]["signed_version"] = None
    docs[1]["provenance_is_ambiguous"] = True
    moves = [
        RoundMove(
            document_id="deal_002",
            round=1,
            taxonomy_id="indemnification",
            moved_by="counterparty",
            change_summary="Clause modified",
            citation=ObservationCitation(
                document_id="deal_002", version=2, clause_path="8", char_span=None
            ),
        ),
        RoundMove(
            document_id="deal_002",
            round=2,
            taxonomy_id="indemnification",
            moved_by="us",
            change_summary="Clause modified",
            citation=ObservationCitation(
                document_id="deal_002", version=3, clause_path="8", char_span=None
            ),
        ),
    ]
    pb = _assemble_04(observations, docs, round_moves=moves)
    assert validate_document(pb).ok
    by_key = {(p["document_id"], p["taxonomy_id"]): p for p in pb["evidence"]["precedent"]}
    assert set(by_key) == {
        ("deal_001", "indemnification"),
        ("deal_002", "indemnification"),
        ("deal_003", "governing_law"),
    }
    d1 = by_key[("deal_001", "indemnification")]
    assert d1["standard"] is True and d1["moved"] is False and d1["rounds"] == 0
    d2 = by_key[("deal_002", "indemnification")]
    assert d2["standard"] is False and d2["rounds"] == 2 and d2["moved"] is True
    assert [a["round"] for a in d2["refused_asks"]] == [1]  # cited v2 -> round 1
    assert d2["paper"] == "unknown" and d2["paper_basis"] == "ambiguous_detection"
    d3 = by_key[("deal_003", "governing_law")]
    assert d3["signed"] is False and d3["signed_text"]["text"].startswith("Delaware")

    clauses = {c["taxonomy_id"]: c for c in pb["evidence"]["clauses"]}
    assert clauses["indemnification"] == {
        **clauses["indemnification"],
        "n_deals": 2,
        "n_signed_standard": 1,
        "n_variants": 1,
        "n_refused": 1,
    }
    # The unsigned deal counts as a deal but never as a signed variant.
    assert (clauses["governing_law"]["n_deals"], clauses["governing_law"]["n_variants"]) == (1, 0)
    assert "x_judgments" not in pb and "curation" not in pb


def test_v05_precedent_ids_stable_across_recompile_and_run_metadata() -> None:
    observations = [_std(_obs("indemnification", text=_LONG_STD, basis="deterministic"), True)]
    a = _assemble_04(observations, [_corpus_doc("deal_001")])
    b = _assemble_04(observations, [_corpus_doc("deal_001")])
    assert [p["id"] for p in a["evidence"]["precedent"]] == [
        p["id"] for p in b["evidence"]["precedent"]
    ]
    assert a["identity"]["content_hash"] == b["identity"]["content_hash"]


def test_v05_fragment_rows_never_become_precedent() -> None:
    """Sub-sentence fragments are excluded exactly as the clause-type
    compiler excludes them (MIN_OBSERVATION_TEXT_LEN)."""
    observations = [
        _std(_obs("indemnification", text=_LONG_STD, basis="deterministic"), True),
        _std(_obs("governing_law", clause_path="12", text="1 6", basis="deterministic"), False),
    ]
    pb = _assemble_04(observations, [_corpus_doc("deal_001")])
    assert {p["taxonomy_id"] for p in pb["evidence"]["precedent"]} == {"indemnification"}
