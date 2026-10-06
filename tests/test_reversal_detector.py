"""Tests for reversal detection (L4, issue #18).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional party names only
('Alice Corp', 'Beta Ltd').
"""

from __future__ import annotations

from playbook_engine.clause_aligner import align_versions
from playbook_engine.clause_classifier import ClassifiedClause, ClauseClassification
from playbook_engine.clause_differ import diff_aligned
from playbook_engine.clause_tree import ClauseNode
from playbook_engine.reversal_detector import (
    REVERSAL_RETAINED_THRESHOLD,
    ReversalRecord,
    detect_reversals,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(path: str, heading: str | None = None, text: str = "") -> ClauseNode:
    return ClauseNode(
        clause_path=path,
        heading=heading,
        text=text,
        char_span=(0, max(1, len(text or "x"))),
    )


def _cc(path: str, taxonomy_id: str | None, text: str = "") -> ClassifiedClause:
    if taxonomy_id is None:
        cls = ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
    else:
        cls = ClauseClassification(taxonomy_id=taxonomy_id, confidence=1.0, basis="exact_match")
    return ClassifiedClause(node=_node(path, taxonomy_id, text), classification=cls)


def _doc_diff(versions: list[tuple[str, list[ClassifiedClause]]]):
    alignments = align_versions(versions)
    version_order = [vid for vid, _ in versions]
    return diff_aligned(alignments, version_order)


# ---------------------------------------------------------------------------
# detect_reversals: acceptance criterion fixture
# ---------------------------------------------------------------------------


def test_reversal_detect_planted_insert_then_revert() -> None:
    """Acceptance criterion: flags a planted insert-then-revert.

    v1 (template): 'Alice Corp shall indemnify Beta Ltd for all losses.'
    v2 (draft):    '... for all losses, including consequential damages.'  <- insert
    v3 (signed):   'Alice Corp shall indemnify Beta Ltd for all losses.'  <- revert to v1
    """
    original = "Alice Corp shall indemnify Beta Ltd for all losses."
    v1 = [_cc("1", "ind", original)]
    v2 = [
        _cc(
            "1",
            "ind",
            "Alice Corp shall indemnify Beta Ltd for all losses, including consequential damages.",
        )
    ]
    v3 = [_cc("1", "ind", original)]

    doc = _doc_diff([("v1", v1), ("v2", v2), ("v3", v3)])
    reversals = detect_reversals(doc)

    assert len(reversals) >= 1
    rev = next(r for r in reversals if r.taxonomy_id == "ind")
    assert rev.version_inserted == "v2"
    assert rev.version_removed == "v3"
    assert "consequential" in rev.proposed_text or "damages" in rev.proposed_text


def test_reversal_no_misfire_on_accepted_edit() -> None:
    """Acceptance criterion: does not misfire on an edit that survives to signed.

    v1 → v2: 'all losses' → 'direct losses and reasonable attorney fees'
    v3 (signed): retains v2 text — no reversal.
    """
    v1 = [_cc("1", "ind", "Alice Corp shall indemnify Beta Ltd for all losses.")]
    v2 = [
        _cc(
            "1",
            "ind",
            "Alice Corp shall indemnify Beta Ltd for direct losses and reasonable attorney fees.",
        )
    ]
    v3 = [
        _cc(
            "1",
            "ind",
            "Alice Corp shall indemnify Beta Ltd for direct losses and reasonable attorney fees.",
        )
    ]

    doc = _doc_diff([("v1", v1), ("v2", v2), ("v3", v3)])
    reversals = detect_reversals(doc)

    ind_reversals = [r for r in reversals if r.taxonomy_id == "ind"]
    assert len(ind_reversals) == 0


# ---------------------------------------------------------------------------
# detect_reversals: two-version documents
# ---------------------------------------------------------------------------


def test_reversal_two_versions_no_reversals() -> None:
    """With only two versions, any change that ends in v2 is not reversed."""
    v1 = [_cc("1", "ind", "Alice shall indemnify.")]
    v2 = [_cc("1", "ind", "Alice shall fully indemnify.")]
    doc = _doc_diff([("v1", v1), ("v2", v2)])
    reversals = detect_reversals(doc)
    assert reversals == []


def test_reversal_no_consecutive_diffs_returns_empty() -> None:
    """DocumentDiff with no consecutive diffs (single version) returns empty."""
    # build a 2-version doc but empty version list
    v1 = [_cc("1", "ind", "text")]
    v2 = [_cc("1", "ind", "text")]
    doc = _doc_diff([("v1", v1), ("v2", v2)])
    # force consecutive to be empty to test guard
    from playbook_engine.clause_differ import DocumentDiff

    empty_doc = DocumentDiff(
        consecutive=(),
        net=doc.net,
        version_order=doc.version_order,
    )
    assert detect_reversals(empty_doc) == []


# ---------------------------------------------------------------------------
# detect_reversals: inserted clause then removed (whole clause)
# ---------------------------------------------------------------------------


def test_reversal_whole_clause_added_then_removed() -> None:
    """A clause added in v2 but absent in v3 (signed) → reversal."""
    v1 = [_cc("1", "ind", "Alice shall indemnify Beta.")]
    v2 = [
        _cc("1", "ind", "Alice shall indemnify Beta."),
        _cc("2", "ins", "Insurance at ten million dollars."),
    ]
    v3 = [_cc("1", "ind", "Alice shall indemnify Beta.")]  # insurance clause removed

    doc = _doc_diff([("v1", v1), ("v2", v2), ("v3", v3)])
    reversals = detect_reversals(doc)

    ins_reversals = [r for r in reversals if r.taxonomy_id == "ins"]
    assert len(ins_reversals) >= 1
    rev = ins_reversals[0]
    assert rev.version_inserted == "v2"
    assert "insurance" in rev.proposed_text.lower() or "million" in rev.proposed_text.lower()


def test_reversal_whole_clause_added_and_kept() -> None:
    """A clause added in v2 and retained in v3 → no reversal."""
    v1 = [_cc("1", "ind", "Alice shall indemnify Beta.")]
    v2 = [
        _cc("1", "ind", "Alice shall indemnify Beta."),
        _cc("2", "ins", "Insurance at ten million."),
    ]
    v3 = [
        _cc("1", "ind", "Alice shall indemnify Beta."),
        _cc("2", "ins", "Insurance at ten million."),
    ]

    doc = _doc_diff([("v1", v1), ("v2", v2), ("v3", v3)])
    reversals = detect_reversals(doc)

    ins_reversals = [r for r in reversals if r.taxonomy_id == "ins"]
    assert ins_reversals == []


# ---------------------------------------------------------------------------
# detect_reversals: multiple clauses, independent reversals
# ---------------------------------------------------------------------------


def test_reversal_only_reversed_clause_flagged() -> None:
    """Only the reversed clause is flagged; accepted edits are not."""
    v1 = [
        _cc("1", "ind", "Alice Corp shall indemnify Beta Ltd."),
        _cc("2", "gov", "Delaware law governs."),
    ]
    v2 = [
        _cc("1", "ind", "Alice Corp shall indemnify Beta Ltd for consequential damages."),
        _cc("2", "gov", "New York law governs."),  # accepted change
    ]
    v3 = [
        _cc("1", "ind", "Alice Corp shall indemnify Beta Ltd."),  # reverted
        _cc("2", "gov", "New York law governs."),  # kept
    ]

    doc = _doc_diff([("v1", v1), ("v2", v2), ("v3", v3)])
    reversals = detect_reversals(doc)

    flagged_tids = {r.taxonomy_id for r in reversals}
    assert "ind" in flagged_tids  # reversed
    assert "gov" not in flagged_tids  # accepted


# ---------------------------------------------------------------------------
# ReversalRecord dataclass
# ---------------------------------------------------------------------------


def test_reversal_record_fields() -> None:
    r = ReversalRecord(
        taxonomy_id="ind",
        clause_path="1",
        version_inserted="v2",
        version_removed="v3",
        proposed_text="consequential damages",
    )
    assert r.taxonomy_id == "ind"
    assert r.clause_path == "1"
    assert r.version_inserted == "v2"
    assert r.version_removed == "v3"
    assert r.proposed_text == "consequential damages"


def test_reversal_record_to_dict() -> None:
    r = ReversalRecord(
        taxonomy_id="ind",
        clause_path="2",
        version_inserted="v2",
        version_removed="v4",
        proposed_text="including consequential",
    )
    d = r.to_dict()
    assert d["taxonomy_id"] == "ind"
    assert d["clause_path"] == "2"
    assert d["version_inserted"] == "v2"
    assert d["version_removed"] == "v4"
    assert d["proposed_text"] == "including consequential"


def test_reversal_record_frozen() -> None:
    r = ReversalRecord(
        taxonomy_id="ind",
        clause_path="1",
        version_inserted="v2",
        version_removed="v3",
        proposed_text="text",
    )
    import pytest

    with pytest.raises((AttributeError, TypeError)):
        r.taxonomy_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# detect_reversals: clause_path instance matching (acceptance criterion P1.2)
# ---------------------------------------------------------------------------


def test_reversal_record_carries_clause_path() -> None:
    """ReversalRecord emitted by detect_reversals carries the originating clause_path."""
    v1 = [_cc("1", "ind", "Alice Corp shall indemnify Beta Ltd for all losses.")]
    v2 = [
        _cc(
            "1",
            "ind",
            "Alice Corp shall indemnify Beta Ltd for all losses, including consequential damages.",
        )
    ]
    v3 = [_cc("1", "ind", "Alice Corp shall indemnify Beta Ltd for all losses.")]

    doc = _doc_diff([("v1", v1), ("v2", v2), ("v3", v3)])
    reversals = detect_reversals(doc)

    assert len(reversals) >= 1
    rev = next(r for r in reversals if r.taxonomy_id == "ind")
    assert rev.clause_path == "1"


def test_reversal_same_taxonomy_id_two_clauses_no_cross_contamination() -> None:
    """Two clauses share a taxonomy_id; only the reversed one is flagged.

    Acceptance criterion (P1.2): instance-level, not bucket-level matching.
    v1: clause "1" (ind) + clause "2" (ind) — both present
    v2: clause "1" (ind) gets an extra phrase; clause "2" unchanged
    v3 (signed): clause "1" reverts; clause "2" unchanged
    → Only clause "1" should appear in reversals.
    """
    base_text_1 = "Alice Corp shall indemnify Beta Ltd for all losses."
    base_text_2 = "Beta Ltd shall indemnify Alice Corp for direct losses."

    v1 = [
        _cc("1", "ind", base_text_1),
        _cc("2", "ind", base_text_2),
    ]
    v2 = [
        _cc("1", "ind", base_text_1 + " Including consequential damages."),
        _cc("2", "ind", base_text_2),
    ]
    v3 = [
        _cc("1", "ind", base_text_1),  # reverted
        _cc("2", "ind", base_text_2),  # unchanged / signed as-is
    ]

    doc = _doc_diff([("v1", v1), ("v2", v2), ("v3", v3)])
    reversals = detect_reversals(doc)

    reversed_paths = {r.clause_path for r in reversals}
    assert "1" in reversed_paths, "clause '1' should be flagged as reversed"
    assert "2" not in reversed_paths, "clause '2' must NOT be cross-contaminated"


# ---------------------------------------------------------------------------
# detect_reversals: cross-version clause_path renumbering (issue #244)
# ---------------------------------------------------------------------------


def test_reversal_no_false_positive_on_cross_version_renumbering() -> None:
    """A clause retained into the signed version, but renumbered by an
    intervening insertion, must not be flagged as reversed.

    v1: [A@"1"]
    v2: [A@"1", B@"2"]                       <- B added
    v3 (signed): [A@"1", C@"2", B@"3"]       <- C inserted before B, B renumbered

    B's text is identical in v2 and v3 — it survived the negotiation
    unchanged, just at a different clause_path because C pushed it down.
    Keying the signed-token lookup by clause_path (rather than the stable
    alignment identity) would look up path "2" for B and find C's tokens
    instead, fabricating a reversal.
    """
    b_text = "Beta Ltd shall receive quarterly compliance reports."
    v1 = [_cc("1", "a", "Alice Corp shall perform its obligations in full.")]
    v2 = [
        _cc("1", "a", "Alice Corp shall perform its obligations in full."),
        _cc("2", "b", b_text),
    ]
    v3 = [
        _cc("1", "a", "Alice Corp shall perform its obligations in full."),
        _cc("2", "c", "Gamma Inc shall audit compliance annually."),
        _cc("3", "b", b_text),
    ]

    doc = _doc_diff([("v1", v1), ("v2", v2), ("v3", v3)])
    reversals = detect_reversals(doc)

    b_reversals = [r for r in reversals if r.taxonomy_id == "b"]
    assert b_reversals == [], f"clause B must not be flagged as reversed: {b_reversals!r}"


def test_reversal_none_taxonomy_id_no_cross_contamination() -> None:
    """Two unclassified clauses (taxonomy_id=None); only the reversed one is flagged.

    Acceptance criterion (P1.2): the None bucket must not cross-contaminate.
    """
    v1 = [
        _cc("1", None, "Miscellaneous provision alpha."),
        _cc("2", None, "Miscellaneous provision beta."),
    ]
    v2 = [
        _cc("1", None, "Miscellaneous provision alpha, with added language."),
        _cc("2", None, "Miscellaneous provision beta."),
    ]
    v3 = [
        _cc("1", None, "Miscellaneous provision alpha."),  # reverted
        _cc("2", None, "Miscellaneous provision beta."),  # signed as-is
    ]

    doc = _doc_diff([("v1", v1), ("v2", v2), ("v3", v3)])
    reversals = detect_reversals(doc)

    reversed_paths = {r.clause_path for r in reversals}
    assert "1" in reversed_paths, "clause '1' (None tid) should be flagged"
    assert "2" not in reversed_paths, "clause '2' (None tid) must NOT be cross-contaminated"


# ---------------------------------------------------------------------------
# No signed terminal → no reversals (issue #221)
# ---------------------------------------------------------------------------


def _planted_insert_then_revert():
    original = "Alice Corp shall indemnify Beta Ltd for all losses."
    v1 = [_cc("1", "ind", original)]
    v2 = [
        _cc(
            "1",
            "ind",
            "Alice Corp shall indemnify Beta Ltd for all losses, including consequential damages.",
        )
    ]
    v3 = [_cc("1", "ind", original)]
    return _doc_diff([("v1", v1), ("v2", v2), ("v3", v3)])


def test_reversal_skipped_without_signed_copy() -> None:
    """The planted insert-then-revert is a reversal only against a SIGNED
    terminal: with no detected signed copy the last version is just the last
    draft, so nothing is reported as refused before signing."""
    doc = _planted_insert_then_revert()
    assert detect_reversals(doc, has_signed_copy=True), "fixture premise: a reversal exists"
    assert detect_reversals(doc, has_signed_copy=False) == []


def test_reversal_has_signed_copy_defaults_true() -> None:
    doc = _planted_insert_then_revert()
    assert detect_reversals(doc) == detect_reversals(doc, has_signed_copy=True)


# ---------------------------------------------------------------------------
# detect_reversals: retained-token ratio (issue #222)
# ---------------------------------------------------------------------------

_IND_BASE = (
    "Alice Corp shall defend, indemnify and hold harmless Beta Ltd from all third "
    "party claims arising from Alice Corp's "
)


_CONF_BASE = (
    "Beta Ltd shall hold all Confidential Information received from Alice Corp in "
    "strict confidence, shall use it solely to evaluate the proposed transaction, "
    "and may disclose Confidential Information "
)


def test_accepted_with_one_word_change_is_not_a_reversal() -> None:
    """Beta Ltd counter-proposes ONE hunk in v2; the signed v3 keeps that
    counter-proposal with one word inside it changed. Most of the hunk
    survived, so it was accepted with an edit — not refused. (The old
    token-subset rule called it a reversal because one proposed word was
    missing.)"""
    v1 = [_cc("1", "conf", _CONF_BASE + "only as required by law.")]
    v2 = [
        _cc(
            "1",
            "conf",
            _CONF_BASE + "only as required by law or to its professional advisers bound by "
            "duties of confidence.",
        )
    ]
    v3 = [
        _cc(
            "1",
            "conf",
            _CONF_BASE + "only as required by law or to its legal advisers bound by "
            "duties of confidence.",
        )
    ]

    doc = _doc_diff([("v1", v1), ("v2", v2), ("v3", v3)])
    proposal = doc.consecutive[0].diffs[0]
    assert proposal.kind == "modified"
    edits = [h for h in proposal.hunks if h.kind in ("insert", "replace")]
    assert len(edits) == 1, "fixture: a single-hunk counter-proposal"
    proposed = set(edits[0].new_text.lower().replace(".", " ").split())
    signed = set(v3[0].node.text.lower().replace(".", " ").split())
    assert not proposed <= signed, "fixture: the old subset rule would have fired"

    assert detect_reversals(doc) == []


def test_refused_hunk_is_not_masked_by_accepted_sibling_hunk() -> None:
    """One v2 diff holds two hunks: an accepted insert (kept in the signed
    copy) and a refused replace (the signed copy restores the original).
    ``retained`` is per hunk, so the accepted sibling cannot dilute the refused
    one below detection: exactly one reversal, for the refused hunk."""
    base = (
        "Alice Corp shall defend, indemnify and hold harmless Beta Ltd, its "
        "officers, directors and employees from all third party claims arising "
        "out of or relating to Alice Corp's negligence"
    )
    v1 = [
        _cc(
            "1",
            "ind",
            base + ". This obligation survives termination of this Agreement for three years.",
        )
    ]
    v2 = [
        _cc(
            "1",
            "ind",
            base + ", willful misconduct or fraud. This obligation survives termination "
            "of this Agreement perpetually.",
        )
    ]
    v3 = [
        _cc(
            "1",
            "ind",
            base + ", willful misconduct or fraud. This obligation survives termination "
            "of this Agreement for three years.",
        )
    ]

    doc = _doc_diff([("v1", v1), ("v2", v2), ("v3", v3)])
    proposal = doc.consecutive[0].diffs[0]
    assert proposal.kind == "modified"
    edits = [h for h in proposal.hunks if h.kind in ("insert", "replace")]
    assert len(edits) == 2, "fixture: one accepted hunk and one refused hunk"

    reversals = detect_reversals(doc)
    assert len(reversals) == 1
    rev = reversals[0]
    assert "perpetually" in rev.proposed_text
    assert "misconduct" not in rev.proposed_text
    assert rev.retained == 0.0
    assert rev.version_inserted == "v2"
    assert rev.version_removed == "v3"


def test_mostly_refused_proposal_is_a_reversal_and_records_retained() -> None:
    """A proposal most of whose words are absent from the signed clause is a
    reversal, and the record carries the retained share (< 0.5)."""
    v1 = [_cc("1", "ind", _IND_BASE + "negligence.")]
    v2 = [_cc("1", "ind", _IND_BASE + "negligence, including consequential and punitive damages.")]
    v3 = [_cc("1", "ind", _IND_BASE + "negligence.")]

    reversals = detect_reversals(_doc_diff([("v1", v1), ("v2", v2), ("v3", v3)]))
    assert len(reversals) == 1
    rev = reversals[0]
    assert rev.retained is not None
    assert rev.retained < REVERSAL_RETAINED_THRESHOLD
    assert rev.retained == 0.0
    assert rev.to_dict()["retained"] == 0.0
    # The draft clause was bound to the signed clause's row by similarity.
    assert rev.alignment_confidence is not None
    assert rev.to_dict()["alignment_confidence"] == round(rev.alignment_confidence, 6)


def test_retained_ratio_boundary() -> None:
    """Exactly half the proposed content tokens retained is accepted (the
    rule is retained < 0.5); fewer than half is a reversal."""
    # Proposed (v1 -> v2) tokens: {fraud, misconduct}; signed keeps "fraud".
    v1 = [_cc("1", "ind", _IND_BASE + "negligence.")]
    v2 = [_cc("1", "ind", _IND_BASE + "negligence, fraud, misconduct.")]
    half = [_cc("1", "ind", _IND_BASE + "negligence, fraud.")]
    assert detect_reversals(_doc_diff([("v1", v1), ("v2", v2), ("v3", half)])) == []

    # Proposed tokens: {fraud, misconduct, recklessness}; signed keeps one.
    v2b = [_cc("1", "ind", _IND_BASE + "negligence, fraud, misconduct, recklessness.")]
    reversals = detect_reversals(_doc_diff([("v1", v1), ("v2", v2b), ("v3", half)]))
    assert len(reversals) == 1
    assert reversals[0].retained is not None
    assert abs(reversals[0].retained - 1 / 3) < 1e-9


def test_reversal_record_to_dict_omits_absent_alignment_confidence() -> None:
    """detect_reversals always sets ``retained``; ``alignment_confidence`` is
    None when the diff's row binds nothing across versions (e.g. a clause
    added in a draft), and is then omitted from the record."""
    r = ReversalRecord(
        taxonomy_id="ind",
        clause_path="1",
        version_inserted="v2",
        version_removed="v3",
        proposed_text="text",
        retained=0.0,
        alignment_confidence=None,
    )
    d = r.to_dict()
    assert "alignment_confidence" not in d
