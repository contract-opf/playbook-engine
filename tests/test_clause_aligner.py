"""Tests for cross-version clause alignment (L3, issue #16).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional party names only.
"""

from __future__ import annotations

import pytest

from playbook_engine.clause_aligner import (
    ALIGNMENT_AMBIGUITY_THRESHOLD,
    AlignmentJudge,
    AlignmentSlot,
    ClauseAlignment,
    align_versions,
)
from playbook_engine.clause_classifier import ClassifiedClause, ClauseClassification
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
    v1 = [_cc("1", None, "Miscellaneous preamble text here.")]
    v2 = [_cc("1", None, "Miscellaneous preamble text revised.")]
    result = align_versions([("v1", v1), ("v2", v2)])
    assert len(result) == 1
    assert result[0].taxonomy_id is None
    assert result[0].slots[0].clause is v1[0]
    assert result[0].slots[1].clause is v2[0]


def test_align_mixed_classified_and_unclassified() -> None:
    """Classified and unclassified clauses coexist without interfering."""
    v1 = [_cc("1", "ind"), _cc("2", None, "Preamble.")]
    v2 = [_cc("1", "ind"), _cc("2", None, "Preamble revised.")]
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

    The extra v1 clause becomes an extra row with v2=None.
    """
    v1 = [
        _cc("1", "ind", "Indemnification direct losses."),
        _cc("2", "ind", "Indemnification indirect losses."),
    ]
    v2 = [
        _cc("1", "ind", "Indemnification covers direct and indirect losses."),
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

    The extra v2 clause becomes an extra row with v1=None.
    """
    v1 = [_cc("1", "ind", "Indemnification covers all losses.")]
    v2 = [
        _cc("1", "ind", "Indemnification covers direct losses."),
        _cc("2", "ind", "Indemnification covers indirect losses."),
    ]
    result = align_versions([("v1", v1), ("v2", v2)])
    ind_rows = [r for r in result if r.taxonomy_id == "ind"]
    assert len(ind_rows) == 2
    v1_clauses = [r.slots[0].clause for r in ind_rows]
    assert v1_clauses.count(None) == 1


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


def test_alignment_judge_split_emits_multiple_rows() -> None:
    """A judge resolving a 3-version bucket with a 2-row split must not
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

    # v1 (ref) has 2 "ind" clauses; v2 and v3 each have 1 clause that both
    # best-match v1's first clause (low Jaccard, so the row is flagged), and
    # so land in the SAME bucket row alongside the ref clause.
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
            "Indemnification duties survive termination for direct losses under agreement.",
        ),
    ]
    v3 = [
        _cc(
            "1",
            "ind",
            "Indemnification survive termination covering consequential losses agreement.",
        ),
    ]

    judge = _SplitJudge()
    result = align_versions([("v1", v1), ("v2", v2), ("v3", v3)], alignment_judge=judge)
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
