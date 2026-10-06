"""Pipeline provenance integration tests — P1.1 hardening.

Verifies three properties introduced in issue #43:

1. detect_provenance is called on the inferred-earliest version tree, NOT the
   first version by filename sort order.
2. trail/<doc>.json and the corpus_manifest.json entries carry
   provenance_confidence and provenance_is_ambiguous fields.
3. A document whose provenance is ambiguous (confidence < AMBIGUITY_THRESHOLD)
   does not yield a strong opening position (standard / hold_firm /
   acceptable_variants_exist) in the compiled playbook for clauses sourced
   only from that document.

SECURITY NOTE: All fixtures use programmatically constructed RTF text with
synthetic, fictional content.  No real agreement files are referenced.
Fictional party names only (e.g. "Alpha Corp", "ACME Works").

Fixture design notes (deal-order corpus):
  a1.rtf — counterparty heavy redline: Alpha Corp is first-named party
            (provenance = counterparty_paper).  Filename sorts BEFORE v1.rtf
            alphabetically, so the old pipeline code (first_tree) would
            incorrectly use this for provenance detection.
  v1.rtf — our opening form: ACME Works is first-named party
            (provenance = our_paper).  Version orderer infers this as
            the earliest version (most different from the signed copy, v2).
  v2.rtf — signed executed copy: similar language to a1 (counterparty
            terms accepted), contains filled signature block.
            detect_signed() identifies it as signed → anchored last.

  Version ordering result: ('v1', 'a1', 'v2')
    v1 = inferred earliest  (our paper — correct provenance source)
    a1 = intermediate       (counterparty redline)
    v2 = signed / latest    (always last when signed)

  With the fix: detect_provenance receives v1's tree → our_paper (correct).
  Without the fix: detect_provenance receives a1's tree → counterparty_paper (wrong).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from playbook_engine.clause_position_compiler import CoherenceFlag
from playbook_engine.config import load_config
from playbook_engine.pipeline import compile_corpus
from playbook_engine.provenance_detector import AMBIGUITY_THRESHOLD, ProvenanceResult
from playbook_engine.signed_detector import SignedStatus
from playbook_engine.taxonomy import load_taxonomy
from playbook_engine.validator import validate_document

# ---------------------------------------------------------------------------
# RTF fixture helpers
# ---------------------------------------------------------------------------

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
)
_RTF_EPILOGUE = r"}"


def _rtf(body: str) -> str:
    return _RTF_PROLOGUE + body + _RTF_EPILOGUE


def _write_rtf(path: Path, body: str) -> None:
    path.write_text(_rtf(body), encoding="utf-8")


# ---------------------------------------------------------------------------
# Synthetic RTF bodies (fictional parties: ACME Works, Alpha Corp)
# ---------------------------------------------------------------------------

_TAXONOMY_PATH = Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"

# a1.rtf — counterparty heavy redline.
# Alpha Corp is first-named party → provenance = counterparty_paper.
# Filename 'a1' sorts before 'v1' alphabetically, so old code (first_tree)
# would incorrectly use this for provenance detection.
# Content is similar to the signed copy v2 (negotiation converged toward a1's terms).
_A1_BODY = (
    r"1. Parties\par "
    r"This Agreement is by and between Alpha Corp (Company) "
    r"and ACME Works (Service Provider).\par "
    r"2. Indemnification\par "
    r"The parties shall mutually indemnify each other. "
    r"Client has broader indemnification rights.\par "
    r"3. Governing Law\par "
    r"This Agreement is governed by New York law. "
    r"Disputes resolved in New York courts.\par "
    r"4. Term\par "
    r"Two years with automatic renewal unless terminated on ninety days notice.\par "
    r"5. Liability Cap\par "
    r"Liability capped at greater of fees paid or one million dollars.\par "
    r"6. IP Ownership\par "
    r"Client owns all work product created specifically for Client.\par "
)

# v1.rtf — our original opening draft.
# ACME Works is first-named party → provenance = our_paper.
# Content is most different from the signed copy (we gave ground in negotiation).
# Version orderer infers this as the earliest version.
_V1_BODY = (
    r"1. Parties\par "
    r"This Agreement is by and between ACME Works (Company) "
    r"and Alpha Corp (Client).\par "
    r"2. Indemnification\par "
    r"Company shall indemnify Client against third-party claims.\par "
    r"3. Governing Law\par "
    r"This Agreement is governed by California law.\par "
    r"4. Term\par "
    r"One year.\par "
    r"5. Liability Cap\par "
    r"Liability capped at fees paid in the prior twelve months.\par "
    r"6. IP Ownership\par "
    r"All work product is owned exclusively by Company.\par "
)

# v2.rtf — executed signed copy.
# Mostly mirrors a1's terms (we accepted counterparty terms in negotiation).
# Contains filled "By:" lines under "7. Signatures" so detect_signed() fires.
_V2_SIGNED_BODY = (
    r"1. Parties\par "
    r"This Agreement is by and between Alpha Corp (Company) "
    r"and ACME Works (Service Provider).\par "
    r"2. Indemnification\par "
    r"The parties shall mutually indemnify each other. "
    r"Client has broader indemnification rights.\par "
    r"3. Governing Law\par "
    r"This Agreement is governed by New York law. "
    r"Disputes resolved in New York courts.\par "
    r"4. Term\par "
    r"Two years with automatic renewal unless terminated on ninety days notice.\par "
    r"5. Liability Cap\par "
    r"Liability capped at greater of fees paid or one million dollars.\par "
    r"6. IP Ownership\par "
    r"Client owns all work product created specifically for Client.\par "
    r"7. Signatures\par "
    r"By: Alice Johnson, VP Operations, ACME Works\par "
    r"By: Robert Chen, Chief Procurement Officer, Alpha Corp\par "
)

# Single-version document whose provenance will be ambiguous.
# No ACME alias mentioned → alias_absent → confidence=0.65 < AMBIGUITY_THRESHOLD=0.70
#
# Carries a filled "5. Signatures" section (no ACME alias in the signatory
# names, so provenance ambiguity is unaffected) so detect_signed() identifies
# this as a genuinely signed copy — otherwise, per issue #83, the pipeline
# correctly records outcome="unsigned" and withholds these observations from
# OPF-conformant clause positions entirely, which would starve out the
# coherence-judge tests below that need at least one real position.
_AMBIG_BODY = (
    r"1. Parties\par "
    r"This Agreement is entered into by and between Alpha Corp and Beta University.\par "
    r"2. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against all third-party claims "
    r"arising from student placement activities.\par "
    r"3. Governing Law\par "
    r"This Agreement is governed by the laws of the State of California.\par "
    r"4. Term\par "
    r"This Agreement commences upon execution and continues for one academic year.\par "
    r"5. Signatures\par "
    r"By: Maria Garcia, General Counsel\par "
    r"By: David Kim, Managing Director\par "
)

# Issue #225: same deal, but our alias appears outside the recital's party
# slots → alias_present → our_paper lean at 0.65 (ambiguous). The coercion
# issue #225 removed flipped exactly this lean to counterparty_paper.
_AMBIG_OUR_LEAN_BODY = _AMBIG_BODY.replace(
    r"4. Term\par ",
    r"4. Term\par Placements are coordinated with ACME Works staff. ",
)

# Issue #225: our template — the deal below signs these four sections
# verbatim, but adds enough of its own sections that its similarity to the
# template lands in the ambiguous middle band (no alias anywhere either), so
# the deal's paper is "unknown" while its clauses are exactly our standard.
_TEMPLATE_BODY = _AMBIG_BODY.split(r"5. Signatures")[0]
_AMBIG_TEMPLATE_DEAL_BODY = (
    _TEMPLATE_BODY
    + r"5. Insurance\par Alpha Corp shall maintain general liability insurance.\par "
    + r"6. Confidentiality\par Each party shall keep student records confidential.\par "
    + r"7. Notices\par Notices shall be delivered in writing to the addresses above.\par "
    + r"8. Assignment\par Neither party may assign this Agreement without consent.\par "
    + r"9. Supervision\par Beta University shall designate a faculty coordinator.\par "
    + r"10. Compliance\par Each party shall comply with applicable accreditation rules.\par "
    + r"11. Signatures\par "
    + r"By: Maria Garcia, General Counsel\par "
    + r"By: David Kim, Managing Director\par "
)


# ---------------------------------------------------------------------------
# Corpus + config factory helpers
# ---------------------------------------------------------------------------


def _make_corpus_earliest_test(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Corpus where a1.rtf (counterparty redline) sorts first by filename
    but v1.rtf is the inferred-earliest version (most different from signed v2).

    Layout:
      corpus/
        deal-order/
          a1.rtf   ← counterparty redline (sorts first alphabetically)
          v1.rtf   ← our opening form (version orderer infers as earliest)
          v2.rtf   ← signed executed copy (always last in ordered chain)
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-order"
    deal_dir.mkdir(parents=True)

    _write_rtf(deal_dir / "a1.rtf", _A1_BODY)
    _write_rtf(deal_dir / "v1.rtf", _V1_BODY)
    _write_rtf(deal_dir / "v2.rtf", _V2_SIGNED_BODY)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["ACME Works", "ACME"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


def _make_corpus_ambiguous(
    tmp_path: Path, *, body: str = _AMBIG_BODY, template_body: str | None = None
) -> tuple[Path, Path, Path]:
    """Corpus with a single document that has no ACME alias in text.

    *body* overrides the document's RTF body; *template_body*, when given, is
    written as ``template.rtf`` and configured as the baseline template.

    alias_absent → confidence=0.65 < AMBIGUITY_THRESHOLD=0.70 → is_ambiguous=True.

    Layout:
      corpus/
        deal-ambig/
          v1.rtf   ← no ACME alias anywhere in the text
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-ambig"
    deal_dir.mkdir(parents=True)

    _write_rtf(deal_dir / "v1.rtf", body)
    template: str | None = None
    if template_body is not None:
        template_path = tmp_path / "template.rtf"
        _write_rtf(template_path, template_body)
        template = str(template_path)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": template},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["ACME Works", "ACME"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


def _make_corpus_ambiguous_with_template(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Issue #225: an unknown-paper deal that signs our template's sections
    verbatim, with the template configured (see ``_AMBIG_TEMPLATE_DEAL_BODY``)."""
    return _make_corpus_ambiguous(
        tmp_path, body=_AMBIG_TEMPLATE_DEAL_BODY, template_body=_TEMPLATE_BODY
    )


# ---------------------------------------------------------------------------
# AC-1: detect_provenance receives the inferred-earliest tree
# ---------------------------------------------------------------------------


def test_provenance_computed_on_inferred_earliest_tree(tmp_path: Path) -> None:
    """AC-1: Provenance is detected on the version orderer's inferred-earliest
    version, not the first-by-filename version.

    Setup (see module docstring):
      a1.rtf sorts first alphabetically (counterparty_paper).
      v1.rtf is inferred-earliest by the version orderer (our_paper).

    Before the fix: detect_provenance received a1's tree → counterparty_paper.
    After the fix:  detect_provenance receives v1's tree → our_paper.

    The compiled manifest must record provenance=our_paper for deal-order.
    """
    corpus_dir, config_path, out_dir = _make_corpus_earliest_test(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    manifest = json.loads((out_dir / "corpus_manifest.json").read_text())
    deal_entry = next(d for d in manifest if d["document_id"] == "deal-order")
    assert deal_entry["provenance"] == "our_paper", (
        f"Provenance should be our_paper (earliest version v1 is our opening form), "
        f"got {deal_entry['provenance']!r}.  "
        f"If this is counterparty_paper the fix is not working — the pipeline is still "
        f"using the first-by-filename tree (a1, counterparty redline) instead of the "
        f"inferred-earliest tree (v1, our opening form)."
    )


def test_provenance_spy_receives_earliest_tree(tmp_path: Path) -> None:
    """AC-1 (spy variant): detect_provenance must be called with the inferred-earliest
    tree, not the first-by-filename tree.

    We spy on detect_provenance and verify the document_id/version of the tree
    it receives matches the version orderer's inferred-earliest, not a1.
    """
    corpus_dir, config_path, out_dir = _make_corpus_earliest_test(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    spy_trees: list[Any] = []

    import playbook_engine.pipeline as _pipeline

    original_detect = _pipeline.detect_provenance

    def _spy(tree: Any, *args: Any, **kwargs: Any) -> ProvenanceResult:
        spy_trees.append(tree)
        return original_detect(tree, *args, **kwargs)

    with patch.object(_pipeline, "detect_provenance", side_effect=_spy):
        compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    assert len(spy_trees) >= 1, "detect_provenance was never called"
    # The tree version must NOT be 'a1' (the first-by-filename counterparty redline).
    # The fix must have selected v1 (our opening form) as the earliest.
    for tree in spy_trees:
        assert tree.version != "a1", (
            "detect_provenance was called with tree version='a1' (the first-by-filename "
            "counterparty redline).  The fix should pass the inferred-earliest tree instead."
        )


# ---------------------------------------------------------------------------
# AC-2: trail/<doc>.json and manifest carry confidence + is_ambiguous
# ---------------------------------------------------------------------------


def test_trail_carries_provenance_confidence_and_is_ambiguous(tmp_path: Path) -> None:
    """AC-2: trail/<doc>.json must include provenance_confidence and
    provenance_is_ambiguous fields after compile."""
    corpus_dir, config_path, out_dir = _make_corpus_earliest_test(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    trail = json.loads((out_dir / "trail" / "deal-order.json").read_text())
    assert "provenance_confidence" in trail, (
        "trail/deal-order.json missing provenance_confidence field"
    )
    assert "provenance_is_ambiguous" in trail, (
        "trail/deal-order.json missing provenance_is_ambiguous field"
    )
    assert isinstance(trail["provenance_confidence"], float), (
        "provenance_confidence must be a float"
    )
    assert isinstance(trail["provenance_is_ambiguous"], bool), (
        "provenance_is_ambiguous must be a bool"
    )
    assert 0.0 <= trail["provenance_confidence"] <= 1.0


def test_manifest_carries_provenance_confidence_and_is_ambiguous(tmp_path: Path) -> None:
    """AC-2: corpus_manifest.json entries must include provenance_confidence
    and provenance_is_ambiguous fields after compile."""
    corpus_dir, config_path, out_dir = _make_corpus_earliest_test(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    manifest = json.loads((out_dir / "corpus_manifest.json").read_text())
    deal_entry = next(d for d in manifest if d["document_id"] == "deal-order")

    assert "provenance_confidence" in deal_entry, (
        "corpus_manifest entry missing provenance_confidence"
    )
    assert "provenance_is_ambiguous" in deal_entry, (
        "corpus_manifest entry missing provenance_is_ambiguous"
    )
    assert isinstance(deal_entry["provenance_confidence"], float)
    assert isinstance(deal_entry["provenance_is_ambiguous"], bool)
    assert 0.0 <= deal_entry["provenance_confidence"] <= 1.0


# ---------------------------------------------------------------------------
# AC-3: Ambiguous-provenance document does not set strong opening position
# ---------------------------------------------------------------------------


def _precedent_for(playbook: dict[str, Any], document_id: str) -> list[dict[str, Any]]:
    return [p for p in playbook["evidence"]["precedent"] if p["document_id"] == document_id]


def _corpus_doc(playbook: dict[str, Any], document_id: str) -> dict[str, Any]:
    return next(d for d in playbook["corpus"]["documents"] if d["document_id"] == document_id)


def test_ambiguous_provenance_is_unknown_paper_on_every_precedent(tmp_path: Path) -> None:
    """Issue #225 (rewrites a test that iterated the v0.1 ``clauses`` key and so
    asserted nothing on current output): an ambiguous detection (alias_absent ->
    0.65 < AMBIGUITY_THRESHOLD) is paper "unknown" in the trail, on every
    observation and on every OPF 0.4 precedent record of the deal, with the
    detection's own basis and confidence -- and, with no template configured,
    it contributes no our_standard and no n_signed_standard."""
    corpus_dir, config_path, out_dir = _make_corpus_ambiguous(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    manifest = json.loads((out_dir / "corpus_manifest.json").read_text())
    deal_entry = next(d for d in manifest if d["document_id"] == "deal-ambig")
    assert deal_entry["provenance_is_ambiguous"] is True
    assert deal_entry["provenance_confidence"] < AMBIGUITY_THRESHOLD

    trail = json.loads((out_dir / "trail" / "deal-ambig.json").read_text())
    assert trail["provenance_is_ambiguous"] is True
    assert trail["provenance"] == "unknown", (
        f"An ambiguous detection must be recorded as 'unknown', never coerced to a "
        f"side; got {trail['provenance']!r}"
    )

    playbook = json.loads((out_dir / "playbook.opf.json").read_text())
    assert playbook["opf_version"] == "0.4"
    assert validate_document(playbook).ok
    records = _precedent_for(playbook, "deal-ambig")
    assert records, "deal-ambig produced no precedent records -- nothing would be checked"
    for record in records:
        assert record["paper"] == "unknown", record
        assert record["paper_basis"] == "alias_absent", record
        assert record["paper_confidence"] == deal_entry["provenance_confidence"]
        assert record["standard"] is False
    # No template configured -> no our_standard anywhere, nothing signed-standard.
    for clause in playbook["evidence"]["clauses"]:
        assert clause["our_standard"] is None, clause
        assert clause["n_signed_standard"] == 0, clause
    # The observation store carries the deal's paper side, basis and confidence.
    obs_rows = [
        json.loads(line)
        for line in (out_dir / "observations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    deal_rows = [o for o in obs_rows if o["citation"]["document_id"] == "deal-ambig"]
    assert deal_rows
    assert {o["provenance"] for o in deal_rows} == {"unknown"}
    assert {o["paper_basis"] for o in deal_rows} == {"alias_absent"}
    assert {o["paper_confidence"] for o in deal_rows} == {deal_entry["provenance_confidence"]}


def test_pre_225_stage_cache_entry_is_not_replayed(tmp_path: Path) -> None:
    """Issue #225 changes what _compute_doc_result records (unknown paper side,
    paper_basis/paper_confidence on observations), so a warm out/.cache entry
    written before it (deviation_vs_template_version 10) must miss — otherwise
    the relabelled counterparty_paper side would replay forever."""
    import playbook_engine.pipeline as pipeline_mod

    assert pipeline_mod._DEVIATION_VS_TEMPLATE_VERSION >= 11
    corpus_dir, config_path, out_dir = _make_corpus_ambiguous(tmp_path, body=_AMBIG_OUR_LEAN_BODY)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    def _run() -> list[str]:
        lines: list[str] = []
        compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=True, progress=lines.append)
        return [line for line in lines if "cache hits=" in line]

    # Run 1: a pre-#225 engine populates the stage cache.
    with patch.object(pipeline_mod, "_DEVIATION_VS_TEMPLATE_VERSION", 10):
        assert any("misses=1" in line for line in _run())

    # Run 2: the current engine over the same warm out dir must recompute.
    run_2 = _run()
    assert any("hits=0" in line and "misses=1" in line for line in run_2), run_2
    trail = json.loads((out_dir / "trail" / "deal-ambig.json").read_text())
    assert trail["provenance"] == "unknown"

    # Run 3: a current-version entry IS replayed (the miss above is not vacuous).
    run_3 = _run()
    assert any("hits=1" in line and "misses=0" in line for line in run_3), run_3


def test_ambiguous_our_paper_lean_is_never_relabelled(tmp_path: Path) -> None:
    """Issue #225: the coercion this replaces flipped an alias_present our_paper
    lean (0.65) to counterparty_paper. Now the trail and every precedent say
    "unknown"; the frozen two-valued corpus.documents[].provenance keeps
    today's value (counterparty_paper) with provenance_is_ambiguous: true
    recording that the side was not determined."""
    corpus_dir, config_path, out_dir = _make_corpus_ambiguous(tmp_path, body=_AMBIG_OUR_LEAN_BODY)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    trail = json.loads((out_dir / "trail" / "deal-ambig.json").read_text())
    assert trail["provenance"] == "unknown"
    assert trail["provenance_is_ambiguous"] is True

    playbook = json.loads((out_dir / "playbook.opf.json").read_text())
    assert validate_document(playbook).ok
    corpus_doc = _corpus_doc(playbook, "deal-ambig")
    assert corpus_doc["provenance"] == "counterparty_paper"
    assert corpus_doc["provenance_is_ambiguous"] is True
    records = _precedent_for(playbook, "deal-ambig")
    assert records
    assert {(r["paper"], r["paper_basis"]) for r in records} == {("unknown", "alias_present")}


def test_v03_unknown_paper_counts_match_the_listed_observed_positions(tmp_path: Path) -> None:
    """Issue #225: in 0.3 output an "unknown" paper side is emitted through
    two_valued_side, and summary.confidence's counts (and the clause-library
    note) are counted through the same mapping — the document never says one
    thing in observed_positions and another in its counts."""
    corpus_dir, config_path, out_dir = _make_corpus_ambiguous(tmp_path, body=_AMBIG_OUR_LEAN_BODY)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False, opf_version="0.3")

    playbook = json.loads((out_dir / "playbook.opf.json").read_text())
    assert playbook["opf_version"] == "0.3"
    assert validate_document(playbook).ok
    corpus_doc = _corpus_doc(playbook, "deal-ambig")
    assert corpus_doc["provenance"] == "counterparty_paper"
    assert corpus_doc["provenance_is_ambiguous"] is True

    clauses = playbook["evidence"]["clauses"]
    assert clauses, "no clauses compiled -- nothing would be checked"
    for clause in clauses:
        positions = clause["observed_positions"]
        assert positions
        confidence = clause["summary"]["confidence"]
        for side, key in (
            ("our_paper", "n_our_paper"),
            ("counterparty_paper", "n_counterparty_paper"),
        ):
            listed = {p["example_ref"]["document_id"] for p in positions if p["provenance"] == side}
            assert confidence[key] == len(listed), (key, confidence, positions)
        assert confidence["n_counterparty_paper"] == 1
        assert confidence["score"] == 0.5

    library = playbook["evidence"]["clause_library"]
    assert library
    for concept in library:
        n_cp = sum(1 for f in concept["accepted_forms"] if f["provenance"] == "counterparty_paper")
        assert n_cp == 1
        assert concept["notes"] == f"Accepted in {n_cp} signed counterparty-paper observation(s)."


def test_store_backed_judge_miss_is_unknown_never_counterparty(tmp_path: Path) -> None:
    """Issue #225: a store-backed provenance judge with no verdict returns
    "unknown" at 0.0 (basis needs_review) -- previously counterparty_paper at
    0.0. Driven through the real StoreBackedProvenanceJudge on an empty store
    (the alias_absent document is ambiguous, so the judge is consulted)."""
    from playbook_engine.agent_judge import (
        PendingQueue,
        StoreBackedProvenanceJudge,
        VerdictStore,
    )

    corpus_dir, config_path, out_dir = _make_corpus_ambiguous(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)
    judge = StoreBackedProvenanceJudge(
        store=VerdictStore(tmp_path / "verdicts.jsonl"),
        pending=PendingQueue(tmp_path / "pending.jsonl"),
    )

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False, provenance_judge=judge)

    trail = json.loads((out_dir / "trail" / "deal-ambig.json").read_text())
    assert trail["provenance"] == "unknown"
    assert trail["provenance_confidence"] == 0.0
    queued = [
        json.loads(line)
        for line in (tmp_path / "pending.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [q["kind"] for q in queued] == ["provenance"]

    playbook = json.loads((out_dir / "playbook.opf.json").read_text())
    assert validate_document(playbook).ok
    corpus_doc = _corpus_doc(playbook, "deal-ambig")
    # The frozen two-valued field must carry a side; it is flagged ambiguous.
    assert corpus_doc["provenance"] == "counterparty_paper"
    assert corpus_doc["provenance_is_ambiguous"] is True
    assert corpus_doc["provenance_confidence"] == 0.0
    records = _precedent_for(playbook, "deal-ambig")
    assert records
    for record in records:
        assert (record["paper"], record["paper_basis"], record["paper_confidence"]) == (
            "unknown",
            "needs_review",
            0.0,
        )


def test_unknown_paper_deal_signing_our_template_counts_as_signed_standard(
    tmp_path: Path,
) -> None:
    """Issue #225: with a template configured, ``standard`` is the exact match
    against it and paper side does not enter into it -- an unknown-paper deal
    that signed our template language counts in n_signed_standard, while
    our_standard still comes only from the configured template."""
    corpus_dir, config_path, out_dir = _make_corpus_ambiguous_with_template(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    playbook = json.loads((out_dir / "playbook.opf.json").read_text())
    assert validate_document(playbook).ok
    assert playbook["baseline"]["has_canonical_template"] is True
    assert _corpus_doc(playbook, "deal-ambig")["provenance_is_ambiguous"] is True
    records = _precedent_for(playbook, "deal-ambig")
    assert records and {r["paper"] for r in records} == {"unknown"}
    standard_tids = {r["taxonomy_id"] for r in records if r["standard"] and r["signed"]}
    assert standard_tids, "the deal signed template text verbatim -- some clause must be standard"
    clauses = {c["taxonomy_id"]: c for c in playbook["evidence"]["clauses"]}
    for tid in standard_tids:
        assert clauses[tid]["n_signed_standard"] == 1, clauses[tid]
    assert any(c["our_standard"] is not None for c in clauses.values())
    for clause in clauses.values():
        if clause["our_standard"] is not None:
            assert clause["our_standard"]["source_ref"]["document_id"] == "template"


# ---------------------------------------------------------------------------
# Non-ambiguous case: high-confidence provenance is not down-graded
# ---------------------------------------------------------------------------


def test_non_ambiguous_provenance_preserved(tmp_path: Path) -> None:
    """Sanity: a high-confidence our-paper detection is NOT down-graded to
    counterparty_paper — the ambiguity gate only fires when confidence < threshold."""
    corpus_dir, config_path, out_dir = _make_corpus_earliest_test(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    trail = json.loads((out_dir / "trail" / "deal-order.json").read_text())
    # The inferred-earliest version (v1.rtf) is our_paper with confidence=0.85
    # (alias_first_party basis) — well above AMBIGUITY_THRESHOLD=0.70.
    assert trail["provenance_is_ambiguous"] is False, (
        f"v1's our_paper result (confidence={trail['provenance_confidence']}) "
        f"should NOT be ambiguous (threshold={AMBIGUITY_THRESHOLD})"
    )
    assert trail["provenance"] == "our_paper", (
        f"High-confidence our_paper must not be down-graded, got {trail['provenance']!r}"
    )
    assert trail["provenance_confidence"] >= AMBIGUITY_THRESHOLD


# ---------------------------------------------------------------------------
# P3.1 — stop_after="intermediates" checkpoint (issue #55)
# ---------------------------------------------------------------------------


def test_stop_after_intermediates_writes_intermediates_not_playbook(tmp_path: Path) -> None:
    """AC-1: stop_after='intermediates' writes scope.json, observations.jsonl,
    corpus_manifest.json, and trail/<doc>.json but NOT playbook.opf.json."""
    corpus_dir, config_path, out_dir = _make_corpus_earliest_test(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    result = compile_corpus(
        corpus_dir, config, taxonomy, out_dir, resume=False, stop_after="intermediates"
    )

    # Intermediates must be present.
    assert (out_dir / "scope.json").exists(), "scope.json must be written"
    assert (out_dir / "observations.jsonl").exists(), "observations.jsonl must be written"
    assert (out_dir / "corpus_manifest.json").exists(), "corpus_manifest.json must be written"
    assert (out_dir / "trail" / "deal-order.json").exists(), "trail/<doc>.json must be written"

    # Playbook must NOT be written.
    assert not (out_dir / "playbook.opf.json").exists(), (
        "playbook.opf.json must NOT be written when stop_after='intermediates'"
    )

    # Return value must be the status dict.
    assert result["stopped_after"] == "intermediates"
    assert result["out_dir"] == str(out_dir)
    assert isinstance(result["documents"], int)
    assert result["documents"] >= 1


def test_stop_after_none_full_run_unchanged(tmp_path: Path) -> None:
    """AC-2: A full run (no stop_after) still writes playbook.opf.json and
    returns the playbook dict, not a status dict."""
    corpus_dir, config_path, out_dir = _make_corpus_earliest_test(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    result = compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    # Playbook must be written.
    assert (out_dir / "playbook.opf.json").exists(), "playbook.opf.json must be written"

    # Return value must look like a playbook dict, not a status dict.
    assert "stopped_after" not in result, "Full run must not return a stopped_after status dict"
    assert "opf_version" in result, "Full run must return the playbook dict"


def test_stop_after_intermediates_status_dict_document_count(tmp_path: Path) -> None:
    """AC-1 (document count): the status dict 'documents' key equals the number of
    documents recorded in corpus_manifest.json."""
    corpus_dir, config_path, out_dir = _make_corpus_earliest_test(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    result = compile_corpus(
        corpus_dir, config, taxonomy, out_dir, resume=False, stop_after="intermediates"
    )

    manifest = json.loads((out_dir / "corpus_manifest.json").read_text())
    assert result["documents"] == len(manifest), (
        f"Status dict 'documents' ({result['documents']}) must equal len(corpus_manifest) "
        f"({len(manifest)})"
    )


# ---------------------------------------------------------------------------
# P3.2 — ProvenanceJudge/SignedJudge/CoherenceJudge injection (issue #56)
# ---------------------------------------------------------------------------


class _RecordingProvenanceJudge:
    """Recording stub: captures all judge calls; returns a fixed our_paper result."""

    def __init__(self) -> None:
        self.received_calls: list[dict] = []

    def judge(
        self,
        preamble: str,
        letterhead: str,
        agreement_type: str,
    ) -> ProvenanceResult:
        self.received_calls.append(
            {"preamble": preamble, "letterhead": letterhead, "agreement_type": agreement_type}
        )
        return ProvenanceResult(
            provenance="our_paper",
            confidence=0.95,
            basis="llm",
        )


class _RecordingSignedJudge:
    """Recording stub: captures subtrees; returns a fixed not-signed result."""

    def __init__(self) -> None:
        self.received_subtrees: list[str] = []

    def judge(self, signature_subtree: str) -> SignedStatus:
        self.received_subtrees.append(signature_subtree)
        return SignedStatus(signed=False, basis="llm", confidence=0.50)


class _RecordingCoherenceJudge:
    """Recording stub: captures all clause summaries; flags all with severity=warn."""

    def __init__(self) -> None:
        self.received_summaries: list[dict] = []

    def judge(self, clause_summary: dict) -> CoherenceFlag | None:
        self.received_summaries.append(clause_summary)
        return CoherenceFlag(
            clause_id=clause_summary["clause_id"],
            reason="stub flag",
            severity="warn",
        )


class _RaisingProvenanceJudge:
    """Judge that always raises — simulates LLM timeout / network failure."""

    def judge(self, preamble: str, letterhead: str, agreement_type: str) -> None:
        raise RuntimeError("LLM service unavailable")


class _RaisingSignedJudge:
    """Judge that always raises."""

    def judge(self, signature_subtree: str) -> None:
        raise RuntimeError("LLM service unavailable")


class _RaisingCoherenceJudge:
    """Judge that always raises."""

    def judge(self, clause_summary: dict) -> None:
        raise RuntimeError("LLM service unavailable")


def test_provenance_judge_called_via_compile_corpus(tmp_path: Path) -> None:
    """AC: provenance_judge passed to compile_corpus is called by detect_provenance.

    The _RecordingProvenanceJudge is only invoked when the deterministic detector
    is ambiguous.  We use the ambiguous corpus (no ACME alias → alias_absent basis →
    confidence=0.65 < AMBIGUITY_THRESHOLD) to guarantee the judge fires.
    """
    corpus_dir, config_path, out_dir = _make_corpus_ambiguous(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    judge = _RecordingProvenanceJudge()
    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False, provenance_judge=judge)

    assert len(judge.received_calls) >= 1, (
        "provenance_judge was never called — the judge was not threaded through compile_corpus. "
        "The ambiguous corpus (no alias → alias_absent basis) should always invoke the judge."
    )
    for call in judge.received_calls:
        assert "preamble" in call
        assert "letterhead" in call
        assert "agreement_type" in call


def test_signed_judge_called_via_compile_corpus(tmp_path: Path) -> None:
    """AC: signed_judge passed to compile_corpus reaches detect_signed for each version.

    The _RecordingSignedJudge fires only when the deterministic detector is in
    the ambiguous range.  We use the signed corpus (v2.rtf has filled signature
    blocks) and verify the judge was invoked.  Because detect_signed calls the
    judge only on ambiguous confidence ranges, we patch detect_signed to always
    delegate to the judge.
    """
    corpus_dir, config_path, out_dir = _make_corpus_earliest_test(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    import playbook_engine.pipeline as _pipeline
    from playbook_engine.signed_detector import detect_signed

    judge = _RecordingSignedJudge()
    call_count = [0]

    def _spy_detect_signed(tree: Any, *, signed_judge: Any = None) -> SignedStatus:
        result = detect_signed(tree, signed_judge=signed_judge)
        if signed_judge is not None:
            call_count[0] += 1
        return result

    with patch.object(_pipeline, "detect_signed", side_effect=_spy_detect_signed):
        compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False, signed_judge=judge)

    # The judge object was forwarded — verify detect_signed received it (non-zero calls
    # through the spy path that forwards signed_judge).
    assert call_count[0] >= 1, "signed_judge was not forwarded to detect_signed via compile_corpus."


def test_coherence_judge_called_via_compile_corpus(tmp_path: Path) -> None:
    """AC: coherence_judge passed to compile_corpus is called during L5 compile.

    The _RecordingCoherenceJudge is called for every clause with low n_our_paper
    (< COHERENCE_MIN_CITATIONS = 3).  The test corpus has only one document so
    all clause positions have n_our_paper < 3 — guaranteeing judge invocations.
    """
    corpus_dir, config_path, out_dir = _make_corpus_ambiguous(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    judge = _RecordingCoherenceJudge()
    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False, coherence_judge=judge)

    assert len(judge.received_summaries) >= 1, (
        "coherence_judge was never called — the judge was not threaded through compile_corpus. "
        "With a single-document corpus every clause has n_our_paper < COHERENCE_MIN_CITATIONS."
    )
    for summary in judge.received_summaries:
        assert "clause_id" in summary


def test_coherence_flags_written_to_json(tmp_path: Path) -> None:
    """AC: coherence_flags.json is written when coherence_judge is set.

    The file must be a JSON array of flag dicts, each with clause_id/reason/severity.
    """
    corpus_dir, config_path, out_dir = _make_corpus_ambiguous(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    judge = _RecordingCoherenceJudge()
    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False, coherence_judge=judge)

    flags_path = out_dir / "coherence_flags.json"
    assert flags_path.exists(), "coherence_flags.json was not written"

    flags = json.loads(flags_path.read_text())
    assert isinstance(flags, list), "coherence_flags.json must be a JSON array"
    assert len(flags) >= 1, "Expected at least one flag from the recording judge"

    flag = flags[0]
    assert "clause_id" in flag
    assert "reason" in flag
    assert "severity" in flag
    assert flag["severity"] == "warn"


def test_coherence_flags_written_empty_when_no_judge(tmp_path: Path) -> None:
    """Coherence_flags.json is written as an empty array when no judge is configured."""
    corpus_dir, config_path, out_dir = _make_corpus_ambiguous(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    flags_path = out_dir / "coherence_flags.json"
    assert flags_path.exists(), "coherence_flags.json must always be written (even empty)"

    flags = json.loads(flags_path.read_text())
    assert flags == [], f"Expected empty list when no judge, got {flags!r}"


def test_no_judges_behavior_unchanged(tmp_path: Path) -> None:
    """With no judges passed, compile_corpus output is unchanged (backward-compat)."""
    corpus_dir, config_path, out_dir = _make_corpus_earliest_test(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    playbook = compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    assert "opf_version" in playbook, "Full run must return the playbook dict"
    assert (out_dir / "playbook.opf.json").exists()
    assert (out_dir / "coherence_flags.json").exists()


# ---------------------------------------------------------------------------
# Issue #58 — hints.yaml signed_version + provenance overrides
# ---------------------------------------------------------------------------


def _make_corpus_hint_signed_version(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Corpus where a1.rtf is the signed copy according to the heuristic
    (it has filled signature blocks) but hints.yaml overrides signed_version to v1.

    Layout:
      corpus/
        deal-hints/
          a1.rtf   ← signed by heuristic (has signature block)
          v1.rtf   ← NOT signed by heuristic; hints.yaml declares it signed
          hints.yaml  ← signed_version: a1_stem  (stem = filename without ext)

    We name the files v1.rtf and v2.rtf and set signed_version to v1 so the
    hint overrides the heuristic (which would pick v2 as signed).
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-hints"
    deal_dir.mkdir(parents=True)

    # v2.rtf has signature block → heuristic would pick v2 as signed.
    _write_rtf(deal_dir / "v2.rtf", _V2_SIGNED_BODY)
    # v1.rtf has no signature block → heuristic does NOT pick v1 as signed.
    _write_rtf(deal_dir / "v1.rtf", _V1_BODY)

    # hints.yaml overrides: v1 is the signed copy.
    hints_content = "signed_version: v1\n"
    (deal_dir / "hints.yaml").write_text(hints_content, encoding="utf-8")

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["ACME Works", "ACME"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


def _make_corpus_hint_provenance(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Corpus where provenance heuristic would return counterparty_paper (ambiguous
    alias_absent basis) but hints.yaml overrides provenance to our_paper.

    Layout:
      corpus/
        deal-prov-hint/
          v1.rtf   ← no ACME alias → alias_absent → counterparty_paper (ambiguous)
          hints.yaml  ← provenance: our_paper
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-prov-hint"
    deal_dir.mkdir(parents=True)

    _write_rtf(deal_dir / "v1.rtf", _AMBIG_BODY)

    hints_content = "provenance: our_paper\n"
    (deal_dir / "hints.yaml").write_text(hints_content, encoding="utf-8")

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["ACME Works", "ACME"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


def test_hint_signed_version_overrides_heuristic(tmp_path: Path) -> None:
    """AC-1 (signed_version): when hints.yaml declares signed_version=v1 the trail
    must record signed_version=v1 even though the heuristic would have picked v2
    (which has a filled signature block).
    """
    corpus_dir, config_path, out_dir = _make_corpus_hint_signed_version(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    trail = json.loads((out_dir / "trail" / "deal-hints.json").read_text())

    # The hint says v1 is signed; trail["signed_version"] (from VersionOrder.to_dict)
    # is the version ID string of the signed copy.
    assert trail["signed_version"] == "v1", (
        f"Trail signed_version should be 'v1' (hint-declared signed copy), "
        f"got {trail['signed_version']!r}. The signed_version hint is not being honored."
    )


def test_hint_provenance_overrides_heuristic(tmp_path: Path) -> None:
    """AC-2 (provenance): when hints.yaml declares provenance=our_paper the trail
    must record provenance=our_paper even though the heuristic would detect
    counterparty_paper (alias_absent basis → ambiguous).
    """
    corpus_dir, config_path, out_dir = _make_corpus_hint_provenance(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    trail = json.loads((out_dir / "trail" / "deal-prov-hint.json").read_text())

    assert trail["provenance"] == "our_paper", (
        f"Provenance hint (our_paper) must override heuristic result; "
        f"got {trail['provenance']!r}. The provenance hint is not being honored."
    )
    # With a hint confidence=1.0, is_ambiguous must be False.
    assert trail["provenance_is_ambiguous"] is False, (
        "Hint-overridden provenance must not be flagged as ambiguous (confidence=1.0)."
    )


def test_hint_provenance_manifest_reflects_override(tmp_path: Path) -> None:
    """corpus_manifest.json must also reflect the provenance hint override."""
    corpus_dir, config_path, out_dir = _make_corpus_hint_provenance(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    manifest = json.loads((out_dir / "corpus_manifest.json").read_text())
    deal_entry = next(d for d in manifest if d["document_id"] == "deal-prov-hint")
    assert deal_entry["provenance"] == "our_paper", (
        f"Manifest provenance must reflect the hint; got {deal_entry['provenance']!r}"
    )


def test_hints_order_timestamps_still_work(tmp_path: Path) -> None:
    """AC-3 (backward compat): order/timestamps hints still function as before.

    Build a corpus whose hints.yaml has only order/timestamps (no new keys),
    and verify the version ordering respects the hint-supplied order.
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-order-hint"
    deal_dir.mkdir(parents=True)

    _write_rtf(deal_dir / "v1.rtf", _V1_BODY)
    _write_rtf(deal_dir / "v2.rtf", _V2_SIGNED_BODY)

    hints_content = "timestamps:\n  v1: '2025-01-01'\n  v2: '2025-01-20'\n"
    (deal_dir / "hints.yaml").write_text(hints_content, encoding="utf-8")

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["ACME Works", "ACME"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "out"

    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    trail = json.loads((out_dir / "trail" / "deal-order-hint.json").read_text())

    # v2 has signature block → must be last (signed).
    ordered = trail["ordered_versions"]
    assert ordered[-1] == "v2", f"v2 (signed) must be last; got {ordered}"
    # v1 must be before v2.
    assert ordered.index("v1") < ordered.index("v2"), f"v1 must precede v2; got {ordered}"


# ---------------------------------------------------------------------------
# corpus_manifest signed_version honesty (issue #202)
# ---------------------------------------------------------------------------


def test_signed_version_null_when_no_signed_copy_detected(tmp_path: Path) -> None:
    """corpus.documents[].signed_version must be null when no version was
    detected as an executed copy (issue #202).

    signed_ordinal is a positional fallback (last ordered version) used for
    diffing; publishing it as signed_version made the projected playbook claim
    an execution that the trail (signed_version: null), the report ("0/N
    signed copies"), and every observation (outcome="unsigned") all denied.
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-unsigned"
    deal_dir.mkdir(parents=True)
    # Two drafts, no signature blocks, no hints.yaml → nothing detected signed.
    _write_rtf(deal_dir / "v1.rtf", _V1_BODY)
    _write_rtf(deal_dir / "v2.rtf", _A1_BODY)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["ACME Works", "ACME"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "out"

    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)
    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    manifest = json.loads((out_dir / "corpus_manifest.json").read_text())
    entry = next(d for d in manifest if d["document_id"] == "deal-unsigned")
    assert entry["signed_version"] is None, (
        f"no signed copy was detected, yet signed_version={entry['signed_version']!r} "
        "was published — the manifest is claiming an execution the trail denies"
    )
    trail = json.loads((out_dir / "trail" / "deal-unsigned.json").read_text())
    assert trail["signed_version"] is None

    # Control: the signed corpus still records its ordinal.
    corpus_dir2 = tmp_path / "corpus-signed"
    deal_dir2 = corpus_dir2 / "deal-signed"
    deal_dir2.mkdir(parents=True)
    _write_rtf(deal_dir2 / "v1.rtf", _V1_BODY)
    _write_rtf(deal_dir2 / "v2.rtf", _V2_SIGNED_BODY)
    out_dir2 = tmp_path / "out-signed"
    compile_corpus(corpus_dir2, config, taxonomy, out_dir2, resume=False)
    manifest2 = json.loads((out_dir2 / "corpus_manifest.json").read_text())
    entry2 = next(d for d in manifest2 if d["document_id"] == "deal-signed")
    assert entry2["signed_version"] == 2
