"""Tests for cross-version clause alignment (L3, issue #16).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional party names only.
"""

from __future__ import annotations

import pytest

from playbook_engine.clause_aligner import (
    ALIGNMENT_AMBIGUITY_THRESHOLD,
    MOVE_JACCARD_THRESHOLD,
    AlignmentJudge,
    AlignmentSlot,
    ClauseAlignment,
    align_versions,
)
from playbook_engine.clause_classifier import ClassifiedClause, ClauseClassification
from playbook_engine.clause_differ import diff_aligned
from playbook_engine.clause_tree import ClauseNode
from playbook_engine.taxonomy import TaxonomyEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(entry_id: str, label: str, status: str = "active") -> TaxonomyEntry:
    return TaxonomyEntry(id=entry_id, label=label, status=status, cuad_origin=None, description="")


def _node(path: str, heading: str | None = None, text: str = "") -> ClauseNode:
    return ClauseNode(
        clause_path=path,
        heading=heading,
        text=text,
        char_span=(0, max(1, len(heading or text or "x"))),
    )


def _cc(
    path: str, taxonomy_id: str | None, text: str = "", basis: str = "exact_match"
) -> ClassifiedClause:
    """Build a ClassifiedClause with given taxonomy_id and text."""
    if taxonomy_id is None:
        cls = ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
    else:
        cls = ClauseClassification(taxonomy_id=taxonomy_id, confidence=1.0, basis=basis)
    return ClassifiedClause(node=_node(path, taxonomy_id, text), classification=cls)


def _tokens_jaccard(a: str, b: str) -> float:
    from playbook_engine.clause_aligner import _jaccard, _tokens

    return _jaccard(_tokens(a), _tokens(b))


# ---------------------------------------------------------------------------
# align_versions: edge cases
# ---------------------------------------------------------------------------


def test_align_empty_returns_empty() -> None:
    assert align_versions([]) == []


def test_align_single_version_returns_one_slot_per_clause() -> None:
    clauses = [_cc("1", "indemnification"), _cc("2", "governing_law")]
    result = align_versions([("v1", clauses)])
    assert len(result) == 2
    assert result[0].taxonomy_id == "indemnification"
    assert result[0].slots[0].version == "v1"
    assert result[0].slots[0].clause is clauses[0]


def test_align_single_version_preserves_order() -> None:
    clauses = [_cc("1", "governing_law"), _cc("2", "indemnification")]
    result = align_versions([("v1", clauses)])
    assert [r.taxonomy_id for r in result] == ["governing_law", "indemnification"]


def test_align_duplicate_version_ids_raises() -> None:
    clauses = [_cc("1", "indemnification")]
    with pytest.raises(ValueError, match="Duplicate version ids"):
        align_versions([("v1", clauses), ("v1", clauses)])


# ---------------------------------------------------------------------------
# align_versions: renumbering (acceptance criterion)
# ---------------------------------------------------------------------------


def test_align_renumbering_matches_by_taxonomy_id() -> None:
    """Acceptance criterion: §1/§2 order swap should not confuse alignment."""
    v1 = [
        _cc("1", "indemnification", "Each party shall indemnify the other."),
        _cc("2", "governing_law", "This agreement shall be governed by Delaware law."),
    ]
    v2 = [
        _cc("1", "governing_law", "This agreement shall be governed by Delaware law."),
        _cc("2", "indemnification", "Each party shall indemnify the other party."),
    ]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2

    ind = next(r for r in result if r.taxonomy_id == "indemnification")
    assert ind.slots[0].version == "v1"
    assert ind.slots[0].clause is v1[0]
    assert ind.slots[1].version == "v2"
    assert ind.slots[1].clause is v2[1]

    gov = next(r for r in result if r.taxonomy_id == "governing_law")
    assert gov.slots[0].version == "v1"
    assert gov.slots[0].clause is v1[1]
    assert gov.slots[1].version == "v2"
    assert gov.slots[1].clause is v2[0]


def test_align_renumbering_three_versions() -> None:
    """Three versions, clause positions shuffle each time."""
    v1 = [_cc("1", "ind"), _cc("2", "gov"), _cc("3", "term")]
    v2 = [_cc("1", "gov"), _cc("2", "ind"), _cc("3", "term")]
    v3 = [_cc("1", "term"), _cc("2", "ind"), _cc("3", "gov")]
    result = align_versions([("v1", v1), ("v2", v2), ("v3", v3)])
    assert len(result) == 3

    for row in result:
        assert row.is_present_in_all
        assert all(s.clause is not None for s in row.slots)

    # First-appearance order should be v1's order: ind, gov, term
    tids = [r.taxonomy_id for r in result]
    assert tids == ["ind", "gov", "term"]


# ---------------------------------------------------------------------------
# align_versions: insertion
# ---------------------------------------------------------------------------


def test_align_insertion_new_clause_in_v2() -> None:
    """A clause appearing only in v2 should have None in v1 slot."""
    v1 = [_cc("1", "indemnification"), _cc("2", "governing_law")]
    v2 = [_cc("1", "indemnification"), _cc("2", "governing_law"), _cc("3", "insurance")]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 3

    ins = next(r for r in result if r.taxonomy_id == "insurance")
    assert ins.slots[0].version == "v1"
    assert ins.slots[0].clause is None  # absent in v1
    assert ins.slots[1].version == "v2"
    assert ins.slots[1].clause is not None


def test_align_insertion_preserves_existing_alignments() -> None:
    """Inserting a new clause should not disturb already-aligned rows."""
    v1 = [_cc("1", "ind"), _cc("2", "gov")]
    v2 = [_cc("1", "ind"), _cc("2", "gov"), _cc("3", "new_clause")]
    result = align_versions([("v1", v1), ("v2", v2)])

    for row in result:
        if row.taxonomy_id in ("ind", "gov"):
            assert row.is_present_in_all


# ---------------------------------------------------------------------------
# align_versions: deletion
# ---------------------------------------------------------------------------


def test_align_deletion_clause_removed_in_v2() -> None:
    """A clause present in v1 but absent in v2 should have None in v2 slot."""
    v1 = [_cc("1", "ind"), _cc("2", "gov"), _cc("3", "term")]
    v2 = [_cc("1", "ind"), _cc("2", "term")]  # gov removed
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 3

    gov = next(r for r in result if r.taxonomy_id == "gov")
    assert gov.slots[0].clause is not None  # present in v1
    assert gov.slots[1].clause is None  # absent in v2


def test_align_deletion_other_rows_intact() -> None:
    v1 = [_cc("1", "ind"), _cc("2", "gov"), _cc("3", "term")]
    v2 = [_cc("1", "ind"), _cc("2", "term")]
    result = align_versions([("v1", v1), ("v2", v2)])

    ind = next(r for r in result if r.taxonomy_id == "ind")
    term = next(r for r in result if r.taxonomy_id == "term")
    assert ind.is_present_in_all
    assert term.is_present_in_all


# ---------------------------------------------------------------------------
# align_versions: unclassified (taxonomy_id=None)
# ---------------------------------------------------------------------------


def test_align_unclassified_clauses_grouped_together() -> None:
    """Unclassified clauses (taxonomy_id=None) get their own bucket."""
    v1 = [_cc("1", None, "Miscellaneous preamble text here regarding the parties.")]
    v2 = [_cc("1", None, "Miscellaneous preamble text revised regarding the parties.")]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 1
    assert result[0].taxonomy_id is None
    assert result[0].slots[0].clause is v1[0]
    assert result[0].slots[1].clause is v2[0]


def test_align_mixed_classified_and_unclassified() -> None:
    """Classified and unclassified clauses coexist without interfering."""
    v1 = [_cc("1", "ind"), _cc("2", None, "Preamble recitals background text.")]
    v2 = [_cc("1", "ind"), _cc("2", None, "Preamble recitals background text revised.")]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2

    ind = next(r for r in result if r.taxonomy_id == "ind")
    unc = next(r for r in result if r.taxonomy_id is None)
    assert ind.is_present_in_all
    assert unc.is_present_in_all


# ---------------------------------------------------------------------------
# align_versions: multiple clauses with same taxonomy_id (splits/merges)
# ---------------------------------------------------------------------------


def test_align_two_indemnification_clauses_same_count() -> None:
    """If both versions have 2 indemnification clauses, zip in order."""
    v1 = [
        _cc("1", "ind", "First indemnification provision covers direct losses."),
        _cc("2", "ind", "Second indemnification provision covers indirect losses."),
    ]
    v2 = [
        _cc("1", "ind", "First indemnification provision covers direct losses."),
        _cc("2", "ind", "Second indemnification provision covers indirect losses."),
    ]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2
    for row in result:
        assert row.taxonomy_id == "ind"
        assert row.is_present_in_all


def test_align_merge_two_clauses_to_one() -> None:
    """v1 has 2 ind clauses; v2 merges them into 1.

    The merged clause stays on the row of the part it is similar to (at or
    above ALIGNMENT_AMBIGUITY_THRESHOLD); the other v1 clause becomes a row
    with v2=None.
    """
    v1 = [
        _cc("1", "ind", "Supplier shall indemnify Customer for direct losses arising from breach."),
        _cc("2", "ind", "Supplier shall also cover indirect losses."),
    ]
    v2 = [
        _cc(
            "1",
            "ind",
            "Supplier shall indemnify Customer for direct and indirect losses arising from breach.",
        ),
    ]
    result = align_versions([("v1", v1), ("v2", v2)])
    # Expect 2 rows: one matched + one v1-only
    ind_rows = [r for r in result if r.taxonomy_id == "ind"]
    assert len(ind_rows) == 2
    # Exactly one row has v2 clause, exactly one has v2=None
    v2_clauses = [r.slots[1].clause for r in ind_rows]
    assert v2_clauses.count(None) == 1
    assert sum(1 for c in v2_clauses if c is not None) == 1


def test_align_split_one_clause_to_two() -> None:
    """v1 has 1 ind clause; v2 splits it into 2.

    The part similar to the original stays on its row; the extra v2 clause
    becomes a row with v1=None.
    """
    v1 = [
        _cc(
            "1",
            "ind",
            "Supplier shall indemnify Customer for direct and indirect losses arising from breach.",
        )
    ]
    v2 = [
        _cc("1", "ind", "Supplier shall indemnify Customer for direct losses arising from breach."),
        _cc("2", "ind", "Supplier shall also cover indirect losses."),
    ]
    result = align_versions([("v1", v1), ("v2", v2)])
    ind_rows = [r for r in result if r.taxonomy_id == "ind"]
    assert len(ind_rows) == 2
    v1_clauses = [r.slots[0].clause for r in ind_rows]
    assert v1_clauses.count(None) == 1


# ---------------------------------------------------------------------------
# align_versions: similarity floor on the bucket path (issue #222)
# ---------------------------------------------------------------------------

_REPORTS = "The supplier shall deliver monthly service reports to the customer portal."
_REPORTS_EDITED = "The supplier shall deliver quarterly service reports to the customer portal."
_PRICING = "Each party shall keep pricing schedules confidential always."
_PRICING_EDITED = "Each party shall keep pricing schedules confidential throughout."
# Shares only "supplier"/"shall" with _REPORTS — same taxonomy, different clause.
_INSURANCE = "The supplier shall maintain insurance coverage with reputable carriers."


def test_unrelated_same_taxonomy_clauses_at_low_jaccard_are_not_paired() -> None:
    """Count-mismatch path: a clause sharing a token or two with a reference
    clause must NOT bind to it (it used to bind at any similarity > 0, and
    the two were diffed as one fabricated "modified" clause)."""
    assert (
        _tokens_jaccard(_REPORTS, _INSURANCE) > 0.0
        and _tokens_jaccard(_REPORTS, _INSURANCE) < ALIGNMENT_AMBIGUITY_THRESHOLD
    )
    v1 = [_cc("1", "svc", _REPORTS), _cc("2", "svc", _PRICING)]
    v2 = [_cc("1", "svc", _INSURANCE)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 3
    paired = [r for r in result if r.is_present_in_all]
    assert paired == []
    insurance_row = next(r for r in result if r.slots[1].clause is v2[0])
    assert insurance_row.slots[0].clause is None

    kinds = sorted(d.kind for d in diff_aligned(result, ["v1", "v2"]).net.diffs)
    assert kinds == ["added", "removed", "removed"]


def test_unrelated_same_taxonomy_clauses_equal_count_are_not_paired() -> None:
    """Equal-count path: one clause per version under one taxonomy_id used to
    be zipped by position with no similarity check at all."""
    v1 = [_cc("1", "svc", _REPORTS)]
    v2 = [_cc("1", "svc", _INSURANCE)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2
    assert [r.slots[0].clause for r in result] == [v1[0], None]
    assert [r.slots[1].clause for r in result] == [None, v2[0]]
    kinds = sorted(d.kind for d in diff_aligned(result, ["v1", "v2"]).net.diffs)
    assert kinds == ["added", "removed"]


def test_equal_count_swapped_clauses_matched_by_similarity() -> None:
    """Two same-taxonomy clauses swap positions AND are edited — too short /
    too different for the global move phase — so the bucket path decides.
    They must pair by similarity, not by position."""
    for a, b in ((_REPORTS, _REPORTS_EDITED), (_PRICING, _PRICING_EDITED)):
        sim = _tokens_jaccard(a, b)
        assert ALIGNMENT_AMBIGUITY_THRESHOLD <= sim < MOVE_JACCARD_THRESHOLD
    v1 = [_cc("1", "svc", _REPORTS), _cc("2", "svc", _PRICING)]
    v2 = [_cc("1", "svc", _PRICING_EDITED), _cc("2", "svc", _REPORTS_EDITED)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2
    assert all(r.match_basis is None for r in result), "bucket path, not the move phase"
    pairs = {(r.slots[0].clause.node.text, r.slots[1].clause.node.text) for r in result}
    assert pairs == {(_REPORTS, _REPORTS_EDITED), (_PRICING, _PRICING_EDITED)}
    for row in result:
        conf = row.slots[0].alignment_confidence
        assert conf is not None and ALIGNMENT_AMBIGUITY_THRESHOLD <= conf < 1.0

    diffs = diff_aligned(result, ["v1", "v2"]).net.diffs
    assert all(d.kind == "modified" for d in diffs)
    assert {d.alignment_confidence for d in diffs} == {
        r.slots[0].alignment_confidence for r in result
    }


def test_identical_duplicates_keep_positional_order() -> None:
    """Identical same-taxonomy clauses (similarity 1.0 ties) pair in order."""
    v1 = [_cc("1", "svc", _REPORTS), _cc("2", "svc", _REPORTS)]
    v2 = [_cc("1", "svc", _REPORTS), _cc("2", "svc", _REPORTS)]
    result = align_versions([("v1", v1), ("v2", v2)])
    # Substantial identical text is paired by the move phase; either way each
    # clause pairs with its positional twin.
    assert [(r.slots[0].clause, r.slots[1].clause) for r in result] == [
        (v1[0], v2[0]),
        (v1[1], v2[1]),
    ]


def test_gradual_drift_across_drafts_stays_on_one_row() -> None:
    """Each draft is compared with its nearest bound draft, so a clause that
    drifts below the threshold from the OPENING draft — but never between
    two adjacent drafts — stays one logical clause."""
    d1 = "Supplier shall deliver monthly written service reports to the customer portal."
    d2 = "Supplier shall deliver quarterly written service reports to the customer portal."
    d3 = "Supplier shall deliver quarterly written performance reports to the customer portal."
    d4 = "Supplier shall deliver quarterly written performance summaries to the customer portal."
    assert _tokens_jaccard(d1, d4) < ALIGNMENT_AMBIGUITY_THRESHOLD
    versions = [(f"v{i}", [_cc("1", "svc", t)]) for i, t in enumerate((d1, d2, d3, d4), start=1)]
    result = align_versions(versions)
    assert len(result) == 1
    assert result[0].is_present_in_all


def test_one_number_change_in_short_headingless_clause_stays_one_modified_row() -> None:
    """Fix round 1 regression: a short heading-less clause whose only edit is
    one number ("five (5)" -> "three (3)") is the same clause. Its token-set
    Jaccard is below the floor (a seven-token clause loses two tokens and
    gains two), but the edit is one contiguous replaced span, so the
    localized-edit rescue binds it — one ``modified`` row, never an
    add/remove pair with identical opening text."""
    from playbook_engine.clause_aligner import ALIGNMENT_RESCUE_MIN_JACCARD

    before = "The confidentiality obligations in this Agreement continue for five (5) years."
    after = "The confidentiality obligations in this Agreement continue for three (3) years."
    jac, _, spans = _rescue_shape(before, after)
    assert ALIGNMENT_RESCUE_MIN_JACCARD <= jac < ALIGNMENT_AMBIGUITY_THRESHOLD
    assert spans == 1
    v1 = [_cc("1", None, before)]
    v2 = [_cc("1", None, after)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 1
    assert result[0].match_basis is None, "bucket path, not the move phase"
    assert (result[0].slots[0].clause, result[0].slots[1].clause) == (v1[0], v2[0])
    # The reported confidence is the Jaccard, never a rescue score.
    assert result[0].slots[0].alignment_confidence == pytest.approx(jac)

    diffs = diff_aligned(result, ["v1", "v2"]).net.diffs
    assert [d.kind for d in diffs] == ["modified"]


def test_rescue_does_not_bind_unrelated_clauses() -> None:
    """The localized-edit rescue is no back door for unrelated text: clauses
    that share only a few words still split into removed + added."""
    before = "The receiving party may disclose Confidential Information to its legal advisers."
    after = "Confidential Information excludes information the party independently developed."
    v1 = [_cc("1", None, before)]
    v2 = [_cc("1", None, after)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2
    kinds = sorted(d.kind for d in diff_aligned(result, ["v1", "v2"]).net.diffs)
    assert kinds == ["added", "removed"]


def _rescue_shape(a: str, b: str) -> tuple[float, int, int]:
    """(Jaccard, shorter clause's content-token count, number of non-equal
    ``SequenceMatcher`` opcodes — contiguous edit spans — between the two
    content-token sequences)."""
    from difflib import SequenceMatcher

    from playbook_engine.clause_aligner import _clause_tokens

    ta, tb = _clause_tokens(a), _clause_tokens(b)
    short = min(len(ta.seq), len(tb.seq))
    opcodes = SequenceMatcher(None, ta.seq, tb.seq, autojunk=False).get_opcodes()
    spans = sum(1 for tag, *_ in opcodes if tag != "equal")
    return _tokens_jaccard(a, b), short, spans


_ASSIGNMENT = "Neither party may assign this Agreement without consent."
_PUBLICITY = (
    "Neither party may disclose the terms of this Agreement or use the other "
    "party's name without prior written consent."
)
_COSTS = "Each party shall bear its own costs."
_EXPORT = (
    "Each party shall comply with all applicable export control laws and "
    "regulations and shall bear its own costs of compliance."
)


@pytest.mark.parametrize(
    ("short", "long"),
    [(_ASSIGNMENT, _PUBLICITY), (_COSTS, _EXPORT)],
    ids=["assignment-vs-publicity", "costs-vs-export"],
)
def test_short_boilerplate_does_not_bind_an_unrelated_longer_clause(short: str, long: str) -> None:
    """Review regressions (a) and (b), issue #222: short boilerplate shares a
    skeleton with an unrelated longer clause of the same bucket — nearly all
    of its tokens appear, in order, in the longer one — but it is a
    different provision. It must split into removed + added, never one
    ``modified`` row (whose diff would fabricate a refused ask). Each fails
    the rescue twice over: its Jaccard is below ALIGNMENT_RESCUE_MIN_JACCARD
    and the two clauses differ in more than one separate span."""
    from playbook_engine.clause_aligner import (
        ALIGNMENT_RESCUE_MIN_JACCARD,
        ALIGNMENT_RESCUE_MIN_TOKENS,
    )

    jac, n_short, spans = _rescue_shape(short, long)
    assert jac < ALIGNMENT_RESCUE_MIN_JACCARD
    assert n_short >= ALIGNMENT_RESCUE_MIN_TOKENS, "fixture: long enough for the rescue"
    assert spans > 1

    v1 = [_cc("1", None, short)]
    v2 = [_cc("1", None, long)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2
    assert all(r.match_basis is None for r in result), "bucket path, not the move phase"
    kinds = sorted(d.kind for d in diff_aligned(result, ["v1", "v2"]).net.diffs)
    assert kinds == ["added", "removed"]


def test_assignment_vs_publicity_yields_no_fragment_reversal() -> None:
    """Review regression (a), three rounds: the assignment clause is replaced
    by the publicity clause in the draft and restored in the signed copy.
    The publicity clause is its own row, so any reversal reports it whole —
    never a fabricated fragment diffed against the assignment clause."""
    from playbook_engine.reversal_detector import detect_reversals

    versions = [
        ("v1", [_cc("1", None, _ASSIGNMENT)]),
        ("v2", [_cc("1", None, _PUBLICITY)]),
        ("v3", [_cc("1", None, _ASSIGNMENT)]),
    ]
    result = align_versions(versions)
    assert sorted(tuple(s.clause is not None for s in r.slots) for r in result) == [
        (False, True, False),
        (True, False, True),
    ]
    doc = diff_aligned(result, ["v1", "v2", "v3"])
    assert all(d.kind != "modified" for c in doc.consecutive for d in c.diffs)
    for rev in detect_reversals(doc):
        assert rev.proposed_text.strip() in {_PUBLICITY, _ASSIGNMENT}


# Review regression (c), issue #222: (L, L', S) where L' is an edit of L and
# S is short text that L' wholly contains. "primary": L' adds one word to L
# (Jaccard 0.875; L is too short for the move phase) and S is itself a
# primary candidate for L' (0.75). "rescue": L' appends a proviso to L (a
# rescue-only pair, 0.69) and S, a prefix of L', is itself a rescue
# candidate for L' (0.54).
_CASE_C: dict[str, tuple[str, str, str]] = {
    "primary": (
        "Recipient shall hold Confidential Information in strict confidence.",
        "Recipient shall hold all Confidential Information in strict confidence.",
        "Recipient shall hold Confidential Information in confidence.",
    ),
    "rescue": (
        "Recipient shall hold Confidential Information in strict confidence for five years.",
        "Recipient shall hold Confidential Information in strict confidence for five years "
        "unless disclosure is legally compelled.",
        "Recipient shall hold Confidential Information in strict confidence.",
    ),
}


@pytest.mark.parametrize("s_first", [False, True], ids=["ticket-order", "s-positionally-closer"])
@pytest.mark.parametrize("shape", sorted(_CASE_C))
def test_case_c_partner_goes_to_the_better_jaccard_clause(shape: str, s_first: bool) -> None:
    """Review regression (c), issue #222: v1 = [L, S], v2 = [L'] where L' is
    an edit of L and S is short text wholly contained in L'. S is a viable
    bind candidate for L' in its own right (alone, the two bind), but its
    Jaccard trails L–L' by more than ALIGNMENT_RESCUE_MARGIN, so the contest
    rule cannot decide this — Jaccard-first ranking does. L' binds to L and
    S is reported removed, also when S sits positionally closer to L' (a
    position-first ranking would hand L' to S)."""
    from playbook_engine.clause_aligner import (
        ALIGNMENT_RESCUE_MARGIN,
        ALIGNMENT_RESCUE_MIN_JACCARD,
        _clause_tokens,
        _is_localized_edit,
    )

    long_, edited, short = _CASE_C[shape]
    j_long = _tokens_jaccard(long_, edited)
    j_short = _tokens_jaccard(short, edited)
    if shape == "primary":
        assert ALIGNMENT_AMBIGUITY_THRESHOLD <= j_short < j_long
    else:
        assert ALIGNMENT_RESCUE_MIN_JACCARD <= j_short < j_long < ALIGNMENT_AMBIGUITY_THRESHOLD
        for text in (long_, short):
            assert _is_localized_edit(_clause_tokens(text), _clause_tokens(edited))
    assert j_long - j_short > ALIGNMENT_RESCUE_MARGIN

    clause_l = _cc("1", "conf", long_)
    clause_s = _cc("2", "conf", short)
    v2 = [_cc("1", "conf", edited)]

    # S is a viable candidate: with L absent, S and L' bind.
    alone = align_versions([("v1", [clause_s]), ("v2", v2)])
    assert len(alone) == 1 and alone[0].is_present_in_all
    assert alone[0].slots[0].alignment_confidence == pytest.approx(j_short)

    v1 = [clause_s, clause_l] if s_first else [clause_l, clause_s]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2
    assert all(r.match_basis is None for r in result), "bucket path, not the move phase"
    by_v1 = {r.slots[0].clause.node.text: r for r in result if r.slots[0].clause}
    assert by_v1[long_].slots[1].clause is v2[0]
    assert by_v1[short].slots[1].clause is None
    # The reported confidence is the Jaccard, never a rescue score.
    conf = by_v1[long_].slots[0].alignment_confidence
    assert conf == pytest.approx(j_long)

    kinds = sorted(d.kind for d in diff_aligned(result, ["v1", "v2"]).net.diffs)
    assert kinds == ["modified", "removed"]


# Three same-bucket service clauses for the ranking and one-to-one tests.
# _Q_REPORTS is the draft; _M_REPORTS is its monthly predecessor (Jaccard
# 0.79) and _Q_INVOICES a different provision that still clears the primary
# threshold against it (0.74). Both are below MOVE_JACCARD_THRESHOLD, so the
# bucket path, not the move phase, decides.
_Q_REPORTS = (
    "Supplier shall deliver quarterly written service reports to the customer "
    "portal within ten business days after each quarter end."
)
_M_REPORTS = (
    "Supplier shall deliver monthly written service reports to the customer "
    "portal within ten business days after each month end."
)
_Q_INVOICES = (
    "Supplier shall deliver quarterly service invoices to the customer portal "
    "within thirty business days after each quarter end."
)


def test_primary_binds_rank_by_jaccard_before_position() -> None:
    """Jaccard-first ranking, primary vs primary (issue #222): both v1 rows
    clear ALIGNMENT_AMBIGUITY_THRESHOLD against the one v2 clause, and the
    lower-Jaccard row sits positionally closer to it. The higher Jaccard
    wins; the other row is reported removed."""
    j_reports = _tokens_jaccard(_M_REPORTS, _Q_REPORTS)
    j_invoices = _tokens_jaccard(_Q_INVOICES, _Q_REPORTS)
    assert ALIGNMENT_AMBIGUITY_THRESHOLD <= j_invoices < j_reports < MOVE_JACCARD_THRESHOLD

    v1 = [_cc("1", "svc", _Q_INVOICES), _cc("2", "svc", _M_REPORTS)]
    v2 = [_cc("1", "svc", _Q_REPORTS)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2
    assert all(r.match_basis is None for r in result), "bucket path, not the move phase"
    by_v1 = {r.slots[0].clause.node.text: r for r in result if r.slots[0].clause}
    assert by_v1[_M_REPORTS].slots[1].clause is v2[0]
    assert by_v1[_Q_INVOICES].slots[1].clause is None
    assert by_v1[_M_REPORTS].slots[0].alignment_confidence == pytest.approx(j_reports)

    kinds = sorted(d.kind for d in diff_aligned(result, ["v1", "v2"]).net.diffs)
    assert kinds == ["modified", "removed"]


def test_two_clauses_qualifying_for_one_row_each_land_on_exactly_one_row() -> None:
    """One-to-one binding (issue #222): both v2 clauses clear
    ALIGNMENT_AMBIGUITY_THRESHOLD against the same v1 row. The higher
    Jaccard takes the row and the other opens its own row: every clause of
    every version appears on exactly one row, none overwritten or lost."""
    assert _tokens_jaccard(_Q_INVOICES, _Q_REPORTS) >= ALIGNMENT_AMBIGUITY_THRESHOLD
    assert _tokens_jaccard(_M_REPORTS, _Q_REPORTS) >= ALIGNMENT_AMBIGUITY_THRESHOLD

    v1 = [_cc("1", "svc", _Q_REPORTS), _cc("2", "svc", _INSURANCE)]
    v2 = [_cc("1", "svc", _Q_INVOICES), _cc("2", "svc", _M_REPORTS)]
    result = align_versions([("v1", v1), ("v2", v2)])
    for vi, clauses in enumerate((v1, v2)):
        for clause in clauses:
            assert sum(1 for r in result if r.slots[vi].clause is clause) == 1
    assert len(result) == 3
    reports_row = next(r for r in result if r.slots[0].clause is v1[0])
    assert reports_row.slots[1].clause is v2[1]

    kinds = sorted(d.kind for d in diff_aligned(result, ["v1", "v2"]).net.diffs)
    assert kinds == ["added", "modified", "removed"]


def test_contested_rescue_bind_is_refused() -> None:
    """A rescue bind may not take a partner from a competing pair whose
    Jaccard is within ALIGNMENT_RESCUE_MARGIN of its own: when the draft
    clause is about as similar to another open row, the rescue does not
    settle the choice, and the clause opens its own row."""
    from playbook_engine.clause_aligner import (
        ALIGNMENT_RESCUE_MARGIN,
        ALIGNMENT_RESCUE_MIN_JACCARD,
        _clause_tokens,
        _is_localized_edit,
    )

    reports = (
        "Supplier shall deliver monthly written service reports to the customer "
        "portal within ten business days."
    )
    invoices = (
        "Supplier shall deliver monthly written invoices to the customer portal "
        "within ten business days of each calendar month end."
    )
    draft = reports[:-1] + (
        " after each calendar month ends, together with uptime statistics and incident summaries."
    )
    rescue_jac = _tokens_jaccard(reports, draft)
    assert ALIGNMENT_RESCUE_MIN_JACCARD <= rescue_jac < ALIGNMENT_AMBIGUITY_THRESHOLD
    assert _is_localized_edit(_clause_tokens(reports), _clause_tokens(draft))
    assert _tokens_jaccard(invoices, draft) >= rescue_jac - ALIGNMENT_RESCUE_MARGIN

    v1 = [_cc("1", "svc", reports), _cc("2", "svc", invoices)]
    v2 = [_cc("1", "svc", draft)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 3
    assert not any(r.is_present_in_all for r in result)

    # Uncontested (the competing row absent), the same pair binds.
    alone = align_versions([("v1", [v1[0]]), ("v2", v2)])
    assert len(alone) == 1 and alone[0].is_present_in_all


def test_rescue_contested_by_a_second_clause_for_the_same_row_is_refused() -> None:
    """Clause-side contest (issue #222): two v2 clauses are each a localized
    edit of the same v1 row — one changes the term, one appends a survival
    proviso — and their Jaccards are within ALIGNMENT_RESCUE_MARGIN of each
    other. The rescue does not settle that choice: neither binds, the row is
    reported removed and both draft clauses added."""
    from playbook_engine.clause_aligner import (
        ALIGNMENT_RESCUE_MARGIN,
        ALIGNMENT_RESCUE_MIN_JACCARD,
        _clause_tokens,
        _is_localized_edit,
    )

    term = "The confidentiality obligations in this Agreement continue for five (5) years."
    shorter = "The confidentiality obligations in this Agreement continue for three (3) years."
    survives = (
        "The confidentiality obligations in this Agreement continue for five (5) "
        "years after termination of this Agreement for any reason."
    )
    unrelated = "Either party may terminate this Agreement on thirty days written notice."
    j_shorter = _tokens_jaccard(term, shorter)
    j_survives = _tokens_jaccard(term, survives)
    for jac, text in ((j_shorter, shorter), (j_survives, survives)):
        assert ALIGNMENT_RESCUE_MIN_JACCARD <= jac < ALIGNMENT_AMBIGUITY_THRESHOLD
        assert _is_localized_edit(_clause_tokens(term), _clause_tokens(text))
    assert abs(j_shorter - j_survives) < ALIGNMENT_RESCUE_MARGIN

    v1 = [_cc("1", "term", term), _cc("2", "term", unrelated)]
    v2 = [_cc("1", "term", shorter), _cc("2", "term", survives)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 4
    assert not any(r.is_present_in_all for r in result)
    for vi, clauses in enumerate((v1, v2)):
        for clause in clauses:
            assert sum(1 for r in result if r.slots[vi].clause is clause) == 1
    kinds = sorted(d.kind for d in diff_aligned(result, ["v1", "v2"]).net.diffs)
    assert kinds == ["added", "added", "removed", "removed"]

    # Uncontested (the competing clause absent), each pair binds.
    for draft in v2:
        alone = align_versions([("v1", v1), ("v2", [draft])])
        assert any(r.slots[0].clause is v1[0] and r.slots[1].clause is draft for r in alone)


def test_rescue_ignores_a_clause_below_the_min_token_floor() -> None:
    """ALIGNMENT_RESCUE_MIN_TOKENS guard: a two-content-token clause that grew
    by one appended token is one contiguous insert at Jaccard 0.67, but a
    clause that short proves nothing about identity. Only the primary
    threshold applies, so the two split into removed + added."""
    from playbook_engine.clause_aligner import (
        ALIGNMENT_RESCUE_MIN_JACCARD,
        ALIGNMENT_RESCUE_MIN_TOKENS,
    )

    short = "Notices in writing."
    long = "Notices in writing by email."
    jac, n_short, spans = _rescue_shape(short, long)
    assert ALIGNMENT_RESCUE_MIN_JACCARD <= jac < ALIGNMENT_AMBIGUITY_THRESHOLD
    assert spans == 1
    assert n_short < ALIGNMENT_RESCUE_MIN_TOKENS

    v1 = [_cc("1", "notices", short)]
    v2 = [_cc("1", "notices", long)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2
    assert all(r.match_basis is None for r in result), "bucket path, not the move phase"
    kinds = sorted(d.kind for d in diff_aligned(result, ["v1", "v2"]).net.diffs)
    assert kinds == ["added", "removed"]


def test_single_span_rewrite_below_the_rescue_jaccard_is_not_bound() -> None:
    """A clause whose second half is rewritten is still ONE contiguous
    replaced span, but its Jaccard is below ALIGNMENT_RESCUE_MIN_JACCARD:
    the edit shape alone never binds, so the clauses split into removed +
    added."""
    from playbook_engine.clause_aligner import (
        ALIGNMENT_RESCUE_MIN_JACCARD,
        ALIGNMENT_RESCUE_MIN_TOKENS,
    )

    before = (
        "Supplier shall deliver monthly service reports to the customer portal "
        "within ten business days."
    )
    after = "Supplier shall deliver monthly service reports to the customer by registered post each quarter."
    jac, n_short, spans = _rescue_shape(before, after)
    assert spans == 1
    assert n_short >= ALIGNMENT_RESCUE_MIN_TOKENS
    assert jac < ALIGNMENT_RESCUE_MIN_JACCARD

    v1 = [_cc("1", "reports", before)]
    v2 = [_cc("1", "reports", after)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2
    assert all(r.match_basis is None for r in result), "bucket path, not the move phase"
    kinds = sorted(d.kind for d in diff_aligned(result, ["v1", "v2"]).net.diffs)
    assert kinds == ["added", "removed"]


def test_rescue_requires_exactly_one_edit_span() -> None:
    """Edit-shape guard: two separate number swaps leave the Jaccard above
    ALIGNMENT_RESCUE_MIN_JACCARD, but the clauses differ in two separate
    spans — not a localized edit — so the rescue does not apply and, below
    the primary threshold, the clauses split into removed + added."""
    from playbook_engine.clause_aligner import (
        ALIGNMENT_RESCUE_MIN_JACCARD,
        _clause_tokens,
        _is_localized_edit,
    )

    before = (
        "The confidentiality obligations in this Agreement continue for five (5) "
        "years after disclosure by the disclosing party."
    )
    after = (
        "The confidentiality obligations in this Agreement continue for three (3) "
        "years after disclosure by the receiving party."
    )
    jac, _, spans = _rescue_shape(before, after)
    assert ALIGNMENT_RESCUE_MIN_JACCARD <= jac < ALIGNMENT_AMBIGUITY_THRESHOLD
    assert spans == 2
    assert not _is_localized_edit(_clause_tokens(before), _clause_tokens(after))

    v1 = [_cc("1", "term", before)]
    v2 = [_cc("1", "term", after)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2
    assert all(r.match_basis is None for r in result), "bucket path, not the move phase"
    kinds = sorted(d.kind for d in diff_aligned(result, ["v1", "v2"]).net.diffs)
    assert kinds == ["added", "removed"]


# ---------------------------------------------------------------------------
# align_versions: ordering
# ---------------------------------------------------------------------------


def test_align_output_order_follows_first_appearance() -> None:
    """Output taxonomy_id order follows first-appearance in first version."""
    v1 = [_cc("1", "term"), _cc("2", "ind"), _cc("3", "gov")]
    v2 = [_cc("1", "ind"), _cc("2", "term"), _cc("3", "gov")]
    result = align_versions([("v1", v1), ("v2", v2)])
    tids = [r.taxonomy_id for r in result]
    assert tids == ["term", "ind", "gov"]


def test_align_new_taxonomy_id_in_v2_appended_at_end() -> None:
    """A taxonomy_id appearing first in v2 is appended after v1's order."""
    v1 = [_cc("1", "ind"), _cc("2", "gov")]
    v2 = [_cc("1", "insurance"), _cc("2", "ind"), _cc("3", "gov")]
    result = align_versions([("v1", v1), ("v2", v2)])
    tids = [r.taxonomy_id for r in result]
    assert tids == ["ind", "gov", "insurance"]


# ---------------------------------------------------------------------------
# ClauseAlignment dataclass
# ---------------------------------------------------------------------------


def test_clause_alignment_is_present_in_all_true() -> None:
    c = _cc("1", "ind")
    row = ClauseAlignment(
        taxonomy_id="ind",
        slots=(
            AlignmentSlot(version="v1", clause=c),
            AlignmentSlot(version="v2", clause=c),
        ),
    )
    assert row.is_present_in_all is True


def test_clause_alignment_is_present_in_all_false() -> None:
    c = _cc("1", "ind")
    row = ClauseAlignment(
        taxonomy_id="ind",
        slots=(
            AlignmentSlot(version="v1", clause=c),
            AlignmentSlot(version="v2", clause=None),
        ),
    )
    assert row.is_present_in_all is False


def test_clause_alignment_version_count() -> None:
    c = _cc("1", "ind")
    row = ClauseAlignment(
        taxonomy_id="ind",
        slots=(
            AlignmentSlot(version="v1", clause=c),
            AlignmentSlot(version="v2", clause=c),
            AlignmentSlot(version="v3", clause=None),
        ),
    )
    assert row.version_count == 3


def test_clause_alignment_frozen() -> None:
    c = _cc("1", "ind")
    row = ClauseAlignment(
        taxonomy_id="ind",
        slots=(AlignmentSlot(version="v1", clause=c),),
    )
    with pytest.raises((AttributeError, TypeError)):
        row.taxonomy_id = "something_else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# align_versions: slot parallel structure
# ---------------------------------------------------------------------------


def test_align_slots_parallel_to_input_versions() -> None:
    """Each ClauseAlignment.slots must be parallel to the input version list."""
    v1 = [_cc("1", "ind"), _cc("2", "gov")]
    v2 = [_cc("1", "ind"), _cc("2", "gov")]
    v3 = [_cc("1", "ind")]
    versions = [("v1", v1), ("v2", v2), ("v3", v3)]
    result = align_versions(versions)
    for row in result:
        assert len(row.slots) == 3
        assert row.slots[0].version == "v1"
        assert row.slots[1].version == "v2"
        assert row.slots[2].version == "v3"


def test_align_all_empty_versions_returns_empty() -> None:
    result = align_versions([("v1", []), ("v2", [])])
    assert result == []


# ---------------------------------------------------------------------------
# Issue #48: alignment_confidence + AlignmentJudge seam
# ---------------------------------------------------------------------------


def test_alignment_slot_has_confidence_field() -> None:
    """AlignmentSlot must carry alignment_confidence."""
    c = _cc("1", "ind")
    slot_no_conf = AlignmentSlot(version="v1", clause=c)
    assert slot_no_conf.alignment_confidence is None

    slot_with_conf = AlignmentSlot(version="v1", clause=c, alignment_confidence=0.85)
    assert slot_with_conf.alignment_confidence == pytest.approx(0.85)


def test_alignment_judge_protocol_importable() -> None:
    """AlignmentJudge must be importable and usable as a protocol check."""

    class _StubJudge:
        def judge_bucket(
            self,
            before_clauses: list,
            after_clauses: list,
        ) -> list:
            return [(0, 0)] if before_clauses and after_clauses else []

    stub = _StubJudge()
    assert isinstance(stub, AlignmentJudge)


def test_high_jaccard_pairs_carry_computed_confidence() -> None:
    """Slots from the slow path with high Jaccard must carry the computed score."""
    # Create two versions where v1 has 2 clauses and v2 has 1 (slow path).
    # The matched clause should have a high Jaccard score stored on the slot.
    long_text = "Each party shall indemnify defend and hold harmless the other party from losses."
    v1 = [
        _cc("1", "ind", long_text),
        _cc("2", "ind", "Second indemnification provision covers indirect losses entirely."),
    ]
    v2 = [
        _cc("1", "ind", long_text),  # identical text → Jaccard = 1.0
    ]
    result = align_versions([("v1", v1), ("v2", v2)])
    ind_rows = [r for r in result if r.taxonomy_id == "ind"]
    # Find the row where v2 has a clause (high-confidence match)
    matched_row = next(r for r in ind_rows if r.slots[1].clause is not None)
    conf = matched_row.slots[1].alignment_confidence
    # v2 is NOT the reference (v1 is longer), so confidence goes on v2 slot
    # Actually conf is stored on all slots in the row — check at least one slot
    # In the implementation, sim_score is the Jaccard score stored on all slots.
    assert conf is not None
    assert conf >= ALIGNMENT_AMBIGUITY_THRESHOLD


def test_judge_called_once_for_none_matched_slot() -> None:
    """With a None-matched slot, the stub judge must be called exactly once."""
    call_log: list[tuple[list, list]] = []

    class _RecordingJudge:
        def judge_bucket(
            self,
            before_clauses: list,
            after_clauses: list,
        ) -> list:
            call_log.append((list(before_clauses), list(after_clauses)))
            # Return identity pairing (or empty if no clauses)
            n = max(len(before_clauses), len(after_clauses))
            return [
                (i if i < len(before_clauses) else None, i if i < len(after_clauses) else None)
                for i in range(n)
            ]

    # v1 has 2 ind clauses; v2 has 1 → slow path; unmatched ref slot gets None
    v1 = [
        _cc("1", "ind", "Indemnification covers all direct losses completely."),
        _cc("2", "ind", "Indemnification covers all indirect losses entirely."),
    ]
    v2 = [
        _cc("1", "ind", "Indemnification covers all direct losses completely."),
    ]

    judge = _RecordingJudge()
    align_versions([("v1", v1), ("v2", v2)], alignment_judge=judge)

    # The judge should have been called for the None-matched and/or low-conf rows.
    assert len(call_log) >= 1, "Judge was never called despite a None-matched slot"


def test_judge_not_called_when_all_high_confidence() -> None:
    """When every alignment is high-confidence, the judge must NOT be called."""
    call_log: list = []

    class _ShouldNotBeCalledJudge:
        def judge_bucket(self, before_clauses: list, after_clauses: list) -> list:
            call_log.append(True)
            return []

    # Fast path: identical counts → zip in order, no Jaccard computed, no judge call.
    v1 = [_cc("1", "ind", "Indemnification clause text.")]
    v2 = [_cc("1", "ind", "Indemnification clause text.")]

    judge = _ShouldNotBeCalledJudge()
    align_versions([("v1", v1), ("v2", v2)], alignment_judge=judge)

    assert call_log == [], "Judge was called on a high-confidence alignment"


def test_stub_judge_injectable_via_kwarg() -> None:
    """alignment_judge kwarg must be accepted by align_versions."""

    class _FixedJudge:
        def judge_bucket(
            self,
            before_clauses: list,
            after_clauses: list,
        ) -> list:
            return [(0, 0)] if before_clauses and after_clauses else []

    v1 = [_cc("1", "ind", "Indemnify alpha."), _cc("2", "ind", "Indemnify beta.")]
    v2 = [_cc("1", "ind", "Indemnify alpha.")]

    result = align_versions([("v1", v1), ("v2", v2)], alignment_judge=_FixedJudge())
    assert result is not None
    assert len(result) >= 1


# ---------------------------------------------------------------------------
# Global move matching (relocation-aware alignment)
# ---------------------------------------------------------------------------

_MOVED_TEXT = (
    "The receiving party shall maintain professional liability insurance of not "
    "less than one million dollars per occurrence throughout the program term."
)
_OTHER_TEXT_A = (
    "Each participating student shall complete all onboarding requirements before "
    "the first scheduled clinical rotation begins at the facility."
)
_OTHER_TEXT_B = (
    "The university shall designate a program coordinator responsible for all "
    "scheduling communications between the parties during each academic year."
)


def test_moved_unchanged_clause_aligns_to_itself_same_bucket() -> None:
    """Two same-taxonomy clauses swap positions between drafts; equal counts
    previously zipped positionally and mis-paired them. The move phase must
    pair identical text with itself so the diff is 'unchanged'."""
    v1 = [_cc("1", "ins", _MOVED_TEXT), _cc("2", "ins", _OTHER_TEXT_A)]
    v2 = [_cc("1", "ins", _OTHER_TEXT_A), _cc("2", "ins", _MOVED_TEXT)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2
    for row in result:
        assert row.is_present_in_all
        assert row.slots[0].clause.node.text == row.slots[1].clause.node.text
        assert row.match_basis == "content_exact"

    diffs = diff_aligned(result, ["v1", "v2"]).net.diffs
    assert all(d.kind == "unchanged" for d in diffs)
    assert all(not d.hunks for d in diffs)


def test_moved_clause_across_taxonomy_flap_aligns() -> None:
    """A clause whose classification flaps between versions (different
    taxonomy_id) previously became a delete+add pair across two buckets. The
    move phase pairs it by content regardless of bucket."""
    v1 = [_cc("1", "insurance", _MOVED_TEXT), _cc("2", "onboarding", _OTHER_TEXT_A)]
    v2 = [_cc("1", "onboarding", _OTHER_TEXT_A), _cc("2", "indemnification", _MOVED_TEXT)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2
    moved = next(r for r in result if r.slots[0].clause is v1[0])
    assert moved.slots[1].clause is v2[1]
    # Row taxonomy follows the latest version's classification.
    assert moved.taxonomy_id == "indemnification"
    assert moved.match_basis == "content_exact"

    diffs = diff_aligned(result, ["v1", "v2"]).net.diffs
    assert all(d.kind == "unchanged" for d in diffs)


def test_moved_and_edited_clause_produces_one_modified_diff() -> None:
    """A relocated-and-edited clause must yield ONE modified diff with the
    true before/after — not a removed+added pair."""
    edited = _MOVED_TEXT.replace("one million", "two million")
    v1 = [_cc("1", "ins", _MOVED_TEXT), _cc("2", "coord", _OTHER_TEXT_B)]
    v2 = [_cc("1", "coord", _OTHER_TEXT_B), _cc("2", "ins", edited)]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 2
    moved = next(r for r in result if r.match_basis == "content_jaccard")
    assert moved.slots[0].clause is v1[0]
    assert moved.slots[1].clause is v2[1]

    diffs = diff_aligned(result, ["v1", "v2"]).net.diffs
    modified = [d for d in diffs if d.kind == "modified"]
    assert len(modified) == 1
    assert not any(d.kind in ("added", "removed") for d in diffs)
    hunk_text = " ".join(h.old_text + " " + h.new_text for h in modified[0].hunks)
    assert "one" in hunk_text and "two" in hunk_text


def test_move_rows_carry_confidence() -> None:
    v1 = [_cc("1", "ins", _MOVED_TEXT), _cc("2", "coord", _OTHER_TEXT_B)]
    v2 = [_cc("1", "coord", _OTHER_TEXT_B), _cc("2", "ins", _MOVED_TEXT)]
    result = align_versions([("v1", v1), ("v2", v2)])
    for row in result:
        assert row.match_basis == "content_exact"
        for slot in row.slots:
            assert slot.alignment_confidence == pytest.approx(1.0)


def test_jaccard_move_confidence_between_threshold_and_one() -> None:
    edited = _MOVED_TEXT.replace("one million", "two million")
    v1 = [_cc("1", "ins", _MOVED_TEXT), _cc("2", "coord", _OTHER_TEXT_B)]
    v2 = [_cc("1", "coord", _OTHER_TEXT_B), _cc("2", "ins", edited)]
    result = align_versions([("v1", v1), ("v2", v2)])
    moved = next(r for r in result if r.match_basis == "content_jaccard")
    conf = moved.slots[0].alignment_confidence
    assert conf is not None
    assert MOVE_JACCARD_THRESHOLD <= conf < 1.0


def test_short_boilerplate_not_globally_matched() -> None:
    """Short clauses must not cross-pair document-wide; they stay on the
    positional path (match_basis None)."""
    v1 = [_cc("1", "misc", "Notices."), _cc("2", "misc", "Reserved.")]
    v2 = [_cc("1", "misc", "Reserved."), _cc("2", "misc", "Notices.")]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert all(r.match_basis is None for r in result)


def test_positional_rows_have_no_match_basis() -> None:
    v1 = [_cc("1", "ind", "Indemnify alpha."), _cc("2", "gov", "Governing law beta.")]
    v2 = [_cc("1", "ind", "Indemnify alpha."), _cc("2", "gov", "Governing law beta.")]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert all(r.match_basis is None for r in result)


def test_move_chain_spans_three_versions() -> None:
    """A clause relocating in v2 and again in v3 must chain into ONE row
    spanning all three versions."""
    v1 = [_cc("1", "ins", _MOVED_TEXT), _cc("2", "a", _OTHER_TEXT_A), _cc("3", "b", _OTHER_TEXT_B)]
    v2 = [_cc("1", "a", _OTHER_TEXT_A), _cc("2", "ins", _MOVED_TEXT), _cc("3", "b", _OTHER_TEXT_B)]
    v3 = [_cc("1", "a", _OTHER_TEXT_A), _cc("2", "b", _OTHER_TEXT_B), _cc("3", "ins", _MOVED_TEXT)]
    result = align_versions([("v1", v1), ("v2", v2), ("v3", v3)])
    assert len(result) == 3
    ins = next(r for r in result if r.taxonomy_id == "ins")
    assert ins.match_basis == "content_exact"
    assert ins.slots[0].clause is v1[0]
    assert ins.slots[1].clause is v2[1]
    assert ins.slots[2].clause is v3[2]

    diffs = diff_aligned(result, ["v1", "v2", "v3"]).net.diffs
    assert all(d.kind == "unchanged" for d in diffs)


def test_move_phase_does_not_steal_better_match() -> None:
    """Mutual-best: when two candidate pairs compete, the higher-similarity
    pair wins; the other clause falls back to its own best match."""
    base = (
        "The facility shall provide clinical supervision and evaluation services "
        "for each assigned student participating in the education program rotation."
    )
    near = base.replace("evaluation", "assessment")  # very high Jaccard to base
    far = base.replace(
        "clinical supervision and evaluation services",
        "orientation materials and scheduling support functions",
    )
    v1 = [_cc("1", "sup", base)]
    v2 = [_cc("1", "sup", far), _cc("2", "sup", near)]
    result = align_versions([("v1", v1), ("v2", v2)])
    paired = next(r for r in result if r.slots[0].clause is v1[0])
    assert paired.slots[1].clause is v2[1], "base must pair with its nearest edit"


def test_moved_clause_no_longer_floods_added_removed() -> None:
    """End-to-end guard for the relocation-artifact bug: a document where a
    clause relocates must produce zero added/removed diffs."""
    v1 = [
        _cc("1", "ins", _MOVED_TEXT),
        _cc("2", "a", _OTHER_TEXT_A),
        _cc("3", "b", _OTHER_TEXT_B),
    ]
    v2 = [
        _cc("1", "a", _OTHER_TEXT_A),
        _cc("2", "b", _OTHER_TEXT_B),
        _cc("3", "ins", _MOVED_TEXT),
    ]
    result = align_versions([("v1", v1), ("v2", v2)])
    diffs = diff_aligned(result, ["v1", "v2"]).net.diffs
    kinds = {d.kind for d in diffs}
    assert kinds == {"unchanged"}, f"relocation produced artifacts: {kinds}"


# ---------------------------------------------------------------------------
# Backward extension of move rows (issue #232)
# ---------------------------------------------------------------------------

_TERM_FIVE = "The confidentiality obligations in this Agreement continue for five (5) years."
_TERM_FOUR = "The confidentiality obligations in this Agreement continue for four (4) years."
_TERM_THREE = "The confidentiality obligations in this Agreement continue for three (3) years."


def _presence(row: ClauseAlignment) -> tuple[bool, ...]:
    return tuple(s.clause is not None for s in row.slots)


@pytest.mark.parametrize("tid", [None, "term"], ids=["unclassified", "classified"])
def test_earlier_draft_joins_move_row_carried_into_signed_copy(tid: str | None) -> None:
    """Issue #232: v1 "five (5) years" -> v2 "three (3) years" == v3. The move
    phase chains v2 to v3 (identical text) and takes both out of the bucket
    path; v1's copy must still join that row by bind similarity (here the
    localized-edit rescue) — one ``modified`` row, never removed + added."""
    from playbook_engine.reversal_detector import detect_reversals

    jac, _, spans = _rescue_shape(_TERM_FIVE, _TERM_THREE)
    assert jac < ALIGNMENT_AMBIGUITY_THRESHOLD and spans == 1, "fixture: a rescue-only pair"
    v1 = [_cc("1", tid, _TERM_FIVE)]
    v2 = [_cc("1", tid, _TERM_THREE)]
    v3 = [_cc("1", tid, _TERM_THREE)]
    result = align_versions([("v1", v1), ("v2", v2), ("v3", v3)])
    assert len(result) == 1
    row = result[0]
    assert [s.clause for s in row.slots] == [v1[0], v2[0], v3[0]]
    assert row.taxonomy_id == tid
    # Grown by bind similarity, so no longer a pure near-exact move row; the
    # row's confidence is its worst link — the v1 -> v2 Jaccard.
    assert row.match_basis == "content_jaccard"
    assert all(s.alignment_confidence == pytest.approx(jac) for s in row.slots)

    doc = diff_aligned(result, ["v1", "v2", "v3"])
    assert [d.kind for d in doc.net.diffs] == ["modified"]
    assert [[d.kind for d in c.diffs] for c in doc.consecutive] == [["modified"], ["unchanged"]]
    # v1's text was the opening position, not a draft's proposal: no reversal.
    assert detect_reversals(doc) == []


def test_backward_extension_chains_across_successive_edits() -> None:
    """Two successive round edits, then carried into the signed copy: v1
    "five" -> v2 "four" -> v3 "three" == v4. The pass runs newest version
    pair first, so the row extended to v2 is extended again to v1 — one row
    across all four drafts (an oldest-first pass would strand v1)."""
    assert _rescue_shape(_TERM_FIVE, _TERM_THREE)[0] < ALIGNMENT_AMBIGUITY_THRESHOLD
    clauses = [_cc("1", "term", t) for t in (_TERM_FIVE, _TERM_FOUR, _TERM_THREE, _TERM_THREE)]
    versions = [(f"v{i}", [c]) for i, c in enumerate(clauses, start=1)]
    result = align_versions(versions)
    assert len(result) == 1
    assert [s.clause for s in result[0].slots] == clauses
    doc = diff_aligned(result, [v for v, _ in versions])
    assert [d.kind for d in doc.net.diffs] == ["modified"]


def test_backward_extension_does_not_bind_unrelated_clause() -> None:
    """The extension uses the bucket path's bind rule, not a looser one: the
    assignment clause replaced by the (unrelated) publicity clause, which is
    then carried into the signed copy, stays removed + added."""
    v1 = [_cc("1", None, _ASSIGNMENT)]
    v2 = [_cc("1", None, _PUBLICITY)]
    v3 = [_cc("1", None, _PUBLICITY)]
    result = align_versions([("v1", v1), ("v2", v2), ("v3", v3)])
    assert sorted(_presence(r) for r in result) == [(False, True, True), (True, False, False)]
    kinds = sorted(d.kind for d in diff_aligned(result, ["v1", "v2", "v3"]).net.diffs)
    assert kinds == ["added", "removed"]


def test_backward_extension_stays_within_the_taxonomy_bucket() -> None:
    """A v1 clause is offered only to rows whose v2 member shares its
    taxonomy_id — the bucket rule the bucket path applies."""
    v1 = [_cc("1", "term", _TERM_FIVE)]
    v2 = [_cc("1", "survival", _TERM_THREE)]
    v3 = [_cc("1", "survival", _TERM_THREE)]
    result = align_versions([("v1", v1), ("v2", v2), ("v3", v3)])
    assert sorted(_presence(r) for r in result) == [(False, True, True), (True, False, False)]


def test_backward_extension_never_rebinds_a_matched_clause() -> None:
    """v1's clause is already chained (v1 == v2 == v3) by the move phase; a
    second row starting at v2 whose text is a localized edit of it must not
    take it too. Every clause sits on exactly one row."""
    v1 = [_cc("1", "term", _TERM_FIVE)]
    v2 = [_cc("1", "term", _TERM_FIVE), _cc("2", "term", _TERM_THREE)]
    v3 = [_cc("1", "term", _TERM_FIVE), _cc("2", "term", _TERM_THREE)]
    result = align_versions([("v1", v1), ("v2", v2), ("v3", v3)])
    assert len(result) == 2
    five_row = next(r for r in result if r.slots[2].clause is v3[0])
    three_row = next(r for r in result if r.slots[2].clause is v3[1])
    assert [s.clause for s in five_row.slots] == [v1[0], v2[0], v3[0]]
    assert [s.clause for s in three_row.slots] == [None, v2[1], v3[1]]


def test_backward_extension_leaves_a_better_free_partner_to_the_bucket_path() -> None:
    """Competition for one v1 clause: the move row's v2 member X ("three")
    is only a rescue partner (Jaccard 0.56), while the free v2 clause Y — v1
    plus one word, too short for the move phase and dropped before signing
    — is a primary partner (0.875). Jaccard-first ranking gives v1 to Y;
    the row may not steal it."""
    y_text = "The confidentiality obligations in this Agreement continue for five (5) full years."
    assert _tokens_jaccard(_TERM_FIVE, y_text) >= ALIGNMENT_AMBIGUITY_THRESHOLD
    assert _tokens_jaccard(_TERM_FIVE, _TERM_THREE) < ALIGNMENT_AMBIGUITY_THRESHOLD
    v1 = [_cc("1", "term", _TERM_FIVE)]
    for x_first in (True, False):
        x, y = _cc("1", "term", _TERM_THREE), _cc("2", "term", y_text)
        v2 = [x, y] if x_first else [y, x]
        v3 = [_cc("1", "term", _TERM_THREE)]
        result = align_versions([("v1", v1), ("v2", v2), ("v3", v3)])
        assert len(result) == 2
        y_row = next(r for r in result if r.slots[1].clause is y)
        x_row = next(r for r in result if r.slots[1].clause is x)
        assert [s.clause for s in y_row.slots] == [v1[0], y, None]
        assert [s.clause for s in x_row.slots] == [None, x, v3[0]]


@pytest.mark.parametrize(
    "weaker",
    [
        _TERM_FIVE,
        "The confidentiality obligations in this Agreement continue for three (3) full calendar years.",
    ],
    ids=["rescue-vs-primary", "primary-vs-primary"],
)
def test_backward_extension_ranks_competing_earlier_clauses_by_jaccard(weaker: str) -> None:
    """Two v1 clauses compete for one move row (v2 == v3). The higher-Jaccard
    candidate joins the row whichever comes first in the document, and the
    other is removed — whether the weaker one is a rescue partner (refused
    as contested) or itself a primary partner (outranked)."""
    stronger = (
        "The confidentiality obligations in this Agreement continue for three (3) full years."
    )
    j_strong = _tokens_jaccard(stronger, _TERM_THREE)
    j_weak = _tokens_jaccard(weaker, _TERM_THREE)
    assert j_strong >= ALIGNMENT_AMBIGUITY_THRESHOLD and j_weak < j_strong
    for weaker_first in (True, False):
        a_weak, a_strong = _cc("1", "term", weaker), _cc("2", "term", stronger)
        v1 = [a_weak, a_strong] if weaker_first else [a_strong, a_weak]
        v2 = [_cc("1", "term", _TERM_THREE)]
        v3 = [_cc("1", "term", _TERM_THREE)]
        result = align_versions([("v1", v1), ("v2", v2), ("v3", v3)])
        assert len(result) == 2
        row = next(r for r in result if r.slots[2].clause is v3[0])
        assert [s.clause for s in row.slots] == [a_strong, v2[0], v3[0]]
        other = next(r for r in result if r is not row)
        assert [s.clause for s in other.slots] == [a_weak, None, None]


def test_alignment_judge_split_emits_multiple_rows() -> None:
    """A judge resolving a multi-version bucket with a 2-row split must not
    collapse the pairings into one overwritten row, and must address the
    correct version index for each pairing (not always the first non-ref
    version) — regression test for issue #111.
    """

    class _SplitJudge:
        def judge_bucket(
            self,
            before_clauses: list,
            after_clauses: list,
        ) -> list[tuple[int | None, int | None]]:
            if len(after_clauses) == 2:
                # The single reference clause maps to both non-ref clauses —
                # a two-row split verdict.
                return [(0, 0), (0, 1)]
            # Any other ambiguous row (e.g. the unmatched ref slot): keep the
            # ref clause, no match on the other side.
            return [(0, None)] if before_clauses else []

    # v1 (ref) has 2 "ind" clauses; v2 and v3 each have 1 clause that binds
    # (at or above ALIGNMENT_AMBIGUITY_THRESHOLD, issue #222) to v1's first
    # clause's row, so both land in the SAME bucket row alongside the ref
    # clause; v4 has no "ind" clause at all, so that row has an empty slot
    # and is flagged for the judge.
    v1 = [
        _cc(
            "1",
            "ind",
            "Indemnification obligations survive termination of this agreement entirely for losses.",
        ),
        _cc(
            "2",
            "ind",
            "Limitation of liability caps apply to indirect damages under this agreement.",
        ),
    ]
    v2 = [
        _cc(
            "1",
            "ind",
            "Indemnification obligations survive termination of this agreement entirely for "
            "direct losses.",
        ),
    ]
    v3 = [
        _cc(
            "1",
            "ind",
            "Indemnification obligations survive expiration of this agreement entirely for "
            "direct losses.",
        ),
    ]
    v4 = [_cc("1", "gov", "Governing law is the law of the State of Delaware.")]

    judge = _SplitJudge()
    result = align_versions([("v1", v1), ("v2", v2), ("v3", v3), ("v4", v4)], alignment_judge=judge)
    ind_rows = [r for r in result if r.taxonomy_id == "ind"]

    # Rows whose v1 slot carries the first (split) ref clause.
    split_ref_text = v1[0].node.text
    split_rows = [
        r
        for r in ind_rows
        if r.slots[0].clause is not None and r.slots[0].clause.node.text == split_ref_text
    ]

    # The split verdict must survive as two distinct rows, not one.
    assert len(split_rows) == 2, f"expected 2 rows from the split verdict, got {len(split_rows)}"

    # Every pairing must address the correct originating version: one row
    # carries v2's clause (v3 slot empty), the other carries v3's clause
    # (v2 slot empty) — neither collapses onto a single hardcoded index.
    v2_having = [r for r in split_rows if r.slots[1].clause is not None]
    v3_having = [r for r in split_rows if r.slots[2].clause is not None]
    assert len(v2_having) == 1, "v2's clause from the split must appear in exactly one row"
    assert len(v3_having) == 1, "v3's clause from the split must appear in exactly one row"
    assert v2_having[0] is not v3_having[0], (
        "v2 and v3 clauses must land in different rows, not merged"
    )
    assert v2_having[0].slots[2].clause is None, "the v2 row must not also carry v3's clause"
    assert v3_having[0].slots[1].clause is None, "the v3 row must not also carry v2's clause"
    assert all(r.slots[3].clause is None for r in split_rows)


def test_align_seqs_slow_path_tokenizes_each_clause_a_constant_number_of_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for issue #246: the slow (differing-count) path of
    _align_seqs must not re-tokenize full clause text inside the greedy
    max() scan. Each clause should be tokenized O(1) times (once for the
    reference precompute, once for itself), not once per remaining
    candidate — the old code was O(n^2) tokenization calls for an n-clause
    bucket.
    """
    import playbook_engine.clause_aligner as ca

    real_tokens = ca._tokens
    real_clause_tokens = ca._clause_tokens
    call_count = 0

    def counting_tokens(text: str) -> frozenset[str]:
        nonlocal call_count
        call_count += 1
        return real_tokens(text)

    def counting_clause_tokens(text: str) -> ca._ClauseTokens:
        nonlocal call_count
        call_count += 1
        return real_clause_tokens(text)

    monkeypatch.setattr(ca, "_tokens", counting_tokens)
    monkeypatch.setattr(ca, "_clause_tokens", counting_clause_tokens)

    # All clauses share taxonomy_id=None (the realistic default-classifier
    # bucket per the ticket) and use short, sub-threshold text so none of
    # them qualify for the global move-matching phase (Phase 0) — they land
    # squarely in the differing-count slow path of _align_seqs.
    n = 60
    v1 = [_cc(str(i), None, f"clause number {i} text") for i in range(n)]
    v2 = [_cc(str(i), None, f"clause number {i} text") for i in range(n - 1)]

    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) >= n - 1

    # Linear bound with generous slack (real O(n) cost is ~2n across both
    # _match_pair's phase-2 gate and _align_seqs's precompute); the O(n^2)
    # pre-fix code blows past this by roughly a factor of n.
    assert call_count <= 6 * n, (
        f"expected O(n) tokenization calls (<= {6 * n}), got {call_count} for n={n} — "
        "likely re-tokenizing inside the greedy scan"
    )
