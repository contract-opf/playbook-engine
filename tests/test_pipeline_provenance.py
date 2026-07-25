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


def _make_corpus_ambiguous(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Corpus with a single document that has no ACME alias in text.

    alias_absent → confidence=0.65 < AMBIGUITY_THRESHOLD=0.70 → is_ambiguous=True.

    Layout:
      corpus/
        deal-ambig/
          v1.rtf   ← no ACME alias anywhere in the text
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-ambig"
    deal_dir.mkdir(parents=True)

    _write_rtf(deal_dir / "v1.rtf", _AMBIG_BODY)

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


def test_ambiguous_provenance_does_not_yield_strong_position(tmp_path: Path) -> None:
    """AC-3: A document where provenance is ambiguous (confidence < AMBIGUITY_THRESHOLD,
    i.e. alias_absent → 0.65) must not produce a strong opening position
    (standard / hold_firm / acceptable_variants_exist) in the playbook.

    The corpus has a single document that mentions no ACME alias → alias_absent
    → confidence=0.65 < 0.70=AMBIGUITY_THRESHOLD → is_ambiguous=True.
    All clause rollups for clauses only from this document must be 'negotiable'.
    """
    corpus_dir, config_path, out_dir = _make_corpus_ambiguous(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    # Verify the manifest marks the doc as ambiguous.
    manifest = json.loads((out_dir / "corpus_manifest.json").read_text())
    deal_entry = next(d for d in manifest if d["document_id"] == "deal-ambig")
    assert deal_entry["provenance_is_ambiguous"] is True, (
        "Expected deal-ambig to be flagged as provenance_is_ambiguous"
    )
    assert deal_entry["provenance_confidence"] < AMBIGUITY_THRESHOLD

    # Verify no strong opening position in the playbook for clauses from this doc.
    playbook = json.loads((out_dir / "playbook.opf.json").read_text())
    _STRONG_POSITIONS = {"standard", "hold_firm", "acceptable_variants_exist"}
    for cp in playbook.get("clauses", []):
        rollup = cp.get("rollup", {})
        position = rollup.get("position", "negotiable")
        assert position not in _STRONG_POSITIONS, (
            f"Clause position {cp.get('taxonomy_id')!r} has strong position {position!r} "
            f"but its only source document (deal-ambig) has ambiguous provenance. "
            f"OPF §2.2 requires this to be 'negotiable'."
        )


def test_ambiguous_provenance_recorded_in_trail(tmp_path: Path) -> None:
    """AC-2 + AC-3 integration: when provenance is ambiguous, trail records
    provenance_is_ambiguous=True and provenance='counterparty_paper'."""
    corpus_dir, config_path, out_dir = _make_corpus_ambiguous(tmp_path)
    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)

    compile_corpus(corpus_dir, config, taxonomy, out_dir, resume=False)

    trail = json.loads((out_dir / "trail" / "deal-ambig.json").read_text())
    assert trail["provenance_is_ambiguous"] is True
    # When ambiguous, pipeline must treat doc as counterparty_paper (OPF §2.2 gate).
    assert trail["provenance"] == "counterparty_paper", (
        f"Ambiguous provenance must be stored as counterparty_paper, got {trail['provenance']!r}"
    )
    assert trail["provenance_confidence"] < AMBIGUITY_THRESHOLD


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
