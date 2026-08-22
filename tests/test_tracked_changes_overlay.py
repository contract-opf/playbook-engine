"""Tests for tracked-changes enrichment overlay (L4, issue #19).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional author names ('Alice',
'Bob') and party names only.
"""

from __future__ import annotations

import pytest

from playbook_engine.clause_differ import ClauseDiff, TextHunk
from playbook_engine.docx_ingester import TrackedChange, TrackedChanges
from playbook_engine.tracked_changes_overlay import (
    EnrichedHunk,
    HunkEnrichment,
    _jaccard,
    enrich_clause_diff,
    round_level_fallback_attribution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tracked_change(
    change_type: str,
    author: str,
    text: str,
    clause_path: str = "1",
    date: str | None = "2024-01-01",
) -> TrackedChange:
    return TrackedChange(
        change_type=change_type,  # type: ignore[arg-type]
        author=author,
        date=date,
        text=text,
        clause_path=clause_path,
        char_span=None,
    )


def _tracked_changes(
    doc_id: str,
    version: str,
    changes: list[TrackedChange],
) -> TrackedChanges:
    return TrackedChanges(document_id=doc_id, version=version, changes=changes)


def _clause_diff(
    hunks: tuple[TextHunk, ...],
    clause_path: str = "1",
    kind: str = "modified",
) -> ClauseDiff:
    return ClauseDiff(
        taxonomy_id="ind",
        clause_path_before=clause_path,
        clause_path_after=clause_path,
        kind=kind,
        hunks=hunks,
        text_before="original text",
        text_after="revised text",
    )


# ---------------------------------------------------------------------------
# enrich_clause_diff: acceptance criterion (enriches when present)
# ---------------------------------------------------------------------------


def test_enrich_insert_hunk_with_tracked_insertion() -> None:
    """Acceptance: enriches insert hunk with author from matching tracked change."""
    hunk = TextHunk(kind="insert", old_text="", new_text="consequential damages")
    tc = _tracked_change("insertion", "Alice", "consequential damages")
    cd = _clause_diff((hunk,))
    result = enrich_clause_diff(cd, _tracked_changes("doc", "v2", [tc]))

    assert len(result) == 1
    assert result[0].hunk is hunk
    assert result[0].enrichment is not None
    assert result[0].enrichment.author == "Alice"
    assert result[0].enrichment.tracked_type == "insertion"


def test_enrich_delete_hunk_with_tracked_deletion() -> None:
    """Acceptance: enriches delete hunk with author from matching tracked deletion."""
    hunk = TextHunk(kind="delete", old_text="limitation liability cap", new_text="")
    tc = _tracked_change("deletion", "Bob", "limitation liability cap")
    cd = _clause_diff((hunk,))
    result = enrich_clause_diff(cd, _tracked_changes("doc", "v2", [tc]))

    assert result[0].enrichment is not None
    assert result[0].enrichment.author == "Bob"
    assert result[0].enrichment.tracked_type == "deletion"


def test_enrich_replace_hunk_matches_insertion_side() -> None:
    """Replace hunks prefer insertion-side match (authorship of proposed new text)."""
    hunk = TextHunk(kind="replace", old_text="all losses", new_text="direct losses only")
    tc = _tracked_change("insertion", "Alice", "direct losses only")
    cd = _clause_diff((hunk,))
    result = enrich_clause_diff(cd, _tracked_changes("doc", "v2", [tc]))

    assert result[0].enrichment is not None
    assert result[0].enrichment.tracked_type == "insertion"
    assert result[0].enrichment.author == "Alice"


# ---------------------------------------------------------------------------
# enrich_clause_diff: degrades silently (acceptance criterion)
# ---------------------------------------------------------------------------


def test_enrich_no_tracked_changes_all_none() -> None:
    """Acceptance: no effect when TrackedChanges is None."""
    hunk = TextHunk(kind="insert", old_text="", new_text="new clause text")
    cd = _clause_diff((hunk,))
    result = enrich_clause_diff(cd, None)
    assert len(result) == 1
    assert result[0].enrichment is None


def test_enrich_empty_tracked_changes_all_none() -> None:
    """No effect when TrackedChanges has empty changes list."""
    hunk = TextHunk(kind="insert", old_text="", new_text="new clause text")
    cd = _clause_diff((hunk,))
    result = enrich_clause_diff(cd, _tracked_changes("doc", "v2", []))
    assert result[0].enrichment is None


def test_enrich_different_clause_path_no_match() -> None:
    """TrackedChange in a different clause path does not enrich this diff."""
    hunk = TextHunk(kind="insert", old_text="", new_text="indemnification text here")
    tc = _tracked_change("insertion", "Alice", "indemnification text here", clause_path="5")
    cd = _clause_diff((hunk,), clause_path="1")  # diff is for clause 1, not 5
    result = enrich_clause_diff(cd, _tracked_changes("doc", "v2", [tc]))
    assert result[0].enrichment is None


def test_enrich_low_similarity_text_no_match() -> None:
    """TrackedChange with dissimilar text does not enrich."""
    hunk = TextHunk(kind="insert", old_text="", new_text="indemnification liability clause")
    tc = _tracked_change("insertion", "Alice", "governing Delaware choice law")
    cd = _clause_diff((hunk,))
    result = enrich_clause_diff(cd, _tracked_changes("doc", "v2", [tc]))
    assert result[0].enrichment is None


def test_jaccard_empty_vs_empty_is_zero() -> None:
    """Issue #118: two empty token sets are absence of signal, not a match.

    Regression coverage for the ``_jaccard`` empty-set case — previously
    returned ``1.0`` (a false "perfect match"), now ``0.0``.
    """
    assert _jaccard(frozenset(), frozenset()) == 0.0


def test_enrich_all_stopword_hunk_no_false_match() -> None:
    """Issue #118: an all-stopword hunk must not be falsely attributed to an
    all-stopword tracked change on the same clause path.

    Before the ``_jaccard`` empty-set fix, both sides tokenize to the empty
    set, ``_jaccard`` returned a false "perfect" ``1.0``, and ``_match_hunk``
    accepted it as a confident match. After the fix, empty-vs-empty yields
    ``0.0``, which is below ``_MATCH_THRESHOLD`` (0.50), so no enrichment.
    """
    hunk = TextHunk(kind="insert", old_text="", new_text="and the")
    tc = _tracked_change("insertion", "Alice", "of a")
    cd = _clause_diff((hunk,))
    result = enrich_clause_diff(cd, _tracked_changes("doc", "v2", [tc]))
    assert result[0].enrichment is None


# ---------------------------------------------------------------------------
# enrich_clause_diff: span-overlap candidate selection across mismatched
# clause-path namespaces (issue #112 regression coverage)
#
# With segmentation.agent=True, clause trees come from the LLM/agent
# segmenter (flat integer paths: '1', '2') while the tracked-changes
# side-channel comes from docx_ingester with its own hierarchical numbering
# ('0.4.1'). The old candidate filter matched on exact clause_path string
# equality, so these fixtures reproduce zero candidates under that filter —
# these tests fail against that filter and pass only once candidate
# selection uses char_span interval overlap instead.
# ---------------------------------------------------------------------------


def test_enrich_mismatched_namespace_matches_via_span_containment() -> None:
    """Agent-style flat path ('2') vs docx-style hierarchical path ('0.4.1')
    on the tracked change: exact clause_path equality finds zero candidates,
    but the TrackedChange.char_span is fully contained within the diff's
    char_span_after, so span-overlap candidate selection still finds it."""
    hunk = TextHunk(kind="insert", old_text="", new_text="consequential damages excluded")
    tc = TrackedChange(
        change_type="insertion",
        author="Alice",
        date="2024-01-01",
        text="consequential damages excluded",
        clause_path="0.4.1",  # docx-ingester namespace — disjoint from clause_diff's paths
        char_span=(120, 151),  # fully contained within char_span_after below
    )
    cd = ClauseDiff(
        taxonomy_id="ind",
        clause_path_before="1",  # agent/LLM segmenter namespace
        clause_path_after="2",
        kind="modified",
        hunks=(hunk,),
        text_before="original text",
        text_after="revised text",
        char_span_before=(90, 160),
        char_span_after=(100, 170),
    )
    # Sanity: the old exact-path filter would have zero candidates here.
    assert tc.clause_path not in {cd.clause_path_before, cd.clause_path_after}

    result = enrich_clause_diff(cd, _tracked_changes("doc", "v2", [tc]))

    assert result[0].enrichment is not None
    assert result[0].enrichment.author == "Alice"


def test_enrich_mismatched_namespace_matches_via_partial_span_overlap() -> None:
    """Same namespace mismatch, but the tracked change's span only partially
    intersects the diff's char_span_after (not full containment) — overlap
    alone must still be sufficient to select it as a candidate."""
    hunk = TextHunk(kind="insert", old_text="", new_text="material adverse effect carveout")
    tc = TrackedChange(
        change_type="insertion",
        author="Priya",
        date="2024-02-02",
        text="material adverse effect carveout",
        clause_path="0.9.2",
        char_span=(165, 210),  # starts inside char_span_after, ends past it
    )
    cd = ClauseDiff(
        taxonomy_id="ind",
        clause_path_before="4",
        clause_path_after="5",
        kind="modified",
        hunks=(hunk,),
        text_before="original text",
        text_after="revised text",
        char_span_before=(130, 170),
        char_span_after=(140, 170),
    )
    assert tc.clause_path not in {cd.clause_path_before, cd.clause_path_after}

    result = enrich_clause_diff(cd, _tracked_changes("doc", "v3", [tc]))

    assert result[0].enrichment is not None
    assert result[0].enrichment.author == "Priya"


def test_enrich_mismatched_namespace_no_span_overlap_still_no_match() -> None:
    """Negative control: mismatched clause paths AND non-overlapping spans
    means span-overlap selection correctly excludes the candidate too — the
    fix adds a new way to match, it does not make matching unconditional."""
    hunk = TextHunk(kind="insert", old_text="", new_text="consequential damages excluded")
    tc = TrackedChange(
        change_type="insertion",
        author="Alice",
        date="2024-01-01",
        text="consequential damages excluded",
        clause_path="0.4.1",
        char_span=(500, 531),  # nowhere near the diff's spans
    )
    cd = ClauseDiff(
        taxonomy_id="ind",
        clause_path_before="1",
        clause_path_after="2",
        kind="modified",
        hunks=(hunk,),
        text_before="original text",
        text_after="revised text",
        char_span_before=(90, 160),
        char_span_after=(100, 170),
    )
    result = enrich_clause_diff(cd, _tracked_changes("doc", "v2", [tc]))
    assert result[0].enrichment is None


def test_enrich_deletion_no_span_falls_back_to_clause_path_within_mismatched_diff() -> None:
    """A deletion (char_span=None) can't be span-matched even when the diff
    itself carries span data; it still falls back to clause-path matching,
    so it enriches when the path happens to match and stays unmatched when
    it doesn't (same-namespace deletion coverage is not regressed by #112)."""
    hunk = TextHunk(kind="delete", old_text="limitation liability cap", new_text="")
    tc_matching_path = TrackedChange(
        change_type="deletion",
        author="Bob",
        date="2024-01-01",
        text="limitation liability cap",
        clause_path="2",  # matches clause_diff.clause_path_after
        char_span=None,
    )
    cd = ClauseDiff(
        taxonomy_id="ind",
        clause_path_before="1",
        clause_path_after="2",
        kind="modified",
        hunks=(hunk,),
        text_before="original text",
        text_after="revised text",
        char_span_before=(90, 160),
        char_span_after=(100, 170),
    )
    result = enrich_clause_diff(cd, _tracked_changes("doc", "v2", [tc_matching_path]))
    assert result[0].enrichment is not None
    assert result[0].enrichment.author == "Bob"


def test_enrich_narrow_heading_span_still_matches_via_clause_path() -> None:
    """Same-namespace docx-to-docx diff where the clause's char_span covers
    only its heading line (docx_ingester._ClauseBuilder.add_body), not the
    body — a real tracked change with a non-None char_span in the clause
    body does NOT overlap that narrow span, but clause_path still matches
    exactly, so the additive OR must still fire via the path branch.

    Pins the reason the clause-path branch was kept alongside span overlap
    (see enrich_clause_diff's module comment): every other clause-path-
    branch test uses char_span=None (via the _tracked_change() helper),
    so none of them distinguish "path branch needed because span is
    absent" from "path branch needed because span is present but narrow"."""
    hunk = TextHunk(kind="insert", old_text="", new_text="consequential damages excluded")
    tc = TrackedChange(
        change_type="insertion",
        author="Chen",
        date="2024-03-03",
        text="consequential damages excluded",
        clause_path="2",  # matches clause_diff.clause_path_after exactly
        char_span=(400, 431),  # body text, well past the heading-only spans below
    )
    cd = ClauseDiff(
        taxonomy_id="ind",
        clause_path_before="1",
        clause_path_after="2",
        kind="modified",
        hunks=(hunk,),
        text_before="original text",
        text_after="revised text",
        char_span_before=(90, 100),  # heading-only span, does not reach the body
        char_span_after=(100, 110),  # heading-only span, does not reach the body
    )
    # Sanity: tc's span starts after both diff spans end, so span-overlap
    # selection alone would find zero candidates here.
    assert tc.char_span is not None
    assert cd.char_span_before is not None
    assert cd.char_span_after is not None
    assert tc.char_span[0] >= cd.char_span_before[1]
    assert tc.char_span[0] >= cd.char_span_after[1]

    result = enrich_clause_diff(cd, _tracked_changes("doc", "v4", [tc]))

    assert result[0].enrichment is not None
    assert result[0].enrichment.author == "Chen"


# ---------------------------------------------------------------------------
# enrich_clause_diff: partial enrichment (some hunks match, some don't)
# ---------------------------------------------------------------------------


def test_enrich_two_hunks_one_match_one_no_match() -> None:
    """Only the matching hunk gets enrichment; the other stays None."""
    hunk1 = TextHunk(kind="insert", old_text="", new_text="consequential damages clause")
    hunk2 = TextHunk(kind="delete", old_text="limitation liability cap", new_text="")
    tc = _tracked_change("insertion", "Alice", "consequential damages clause")
    cd = _clause_diff((hunk1, hunk2))
    result = enrich_clause_diff(cd, _tracked_changes("doc", "v2", [tc]))

    assert len(result) == 2
    assert result[0].enrichment is not None  # hunk1 matched
    assert result[1].enrichment is None  # hunk2 did not match (no deletion tracked change)


def test_enrich_each_tracked_change_used_at_most_once() -> None:
    """A single TrackedChange cannot enrich two different hunks."""
    tc_text = "direct losses consequential damages"
    hunk1 = TextHunk(kind="insert", old_text="", new_text=tc_text)
    hunk2 = TextHunk(kind="insert", old_text="", new_text=tc_text)  # same text
    tc = _tracked_change("insertion", "Alice", tc_text)
    cd = _clause_diff((hunk1, hunk2))
    result = enrich_clause_diff(cd, _tracked_changes("doc", "v2", [tc]))

    enriched_count = sum(1 for r in result if r.enrichment is not None)
    assert enriched_count == 1  # only one match allowed


# ---------------------------------------------------------------------------
# enrich_clause_diff: empty hunks
# ---------------------------------------------------------------------------


def test_enrich_empty_hunks_returns_empty() -> None:
    cd = _clause_diff(())
    result = enrich_clause_diff(cd, None)
    assert result == []


# ---------------------------------------------------------------------------
# HunkEnrichment / EnrichedHunk dataclasses
# ---------------------------------------------------------------------------


def test_hunk_enrichment_fields() -> None:
    e = HunkEnrichment(author="Alice", date="2024-01-15", tracked_type="insertion")
    assert e.author == "Alice"
    assert e.date == "2024-01-15"
    assert e.tracked_type == "insertion"


def test_enriched_hunk_frozen() -> None:
    h = TextHunk(kind="insert", old_text="", new_text="x")
    eh = EnrichedHunk(hunk=h, enrichment=None)
    with pytest.raises((AttributeError, TypeError)):
        eh.enrichment = HunkEnrichment("x", None, "insertion")  # type: ignore[misc]


def test_enriched_hunk_date_none() -> None:
    tc = _tracked_change("insertion", "Alice", "test text", date=None)
    hunk = TextHunk(kind="insert", old_text="", new_text="test text here")
    cd = _clause_diff((hunk,))
    result = enrich_clause_diff(cd, _tracked_changes("doc", "v2", [tc]))
    if result[0].enrichment:
        assert result[0].enrichment.date is None


# ---------------------------------------------------------------------------
# round_level_fallback_attribution (issue #118): a SEPARATE, coarser tier —
# not called by enrich_clause_diff, not exercised by any test above. See
# pipeline._attribution_for_diff for the only real caller.
# ---------------------------------------------------------------------------


def test_round_level_fallback_fires_with_single_distinct_author() -> None:
    hunk = TextHunk(kind="insert", old_text="", new_text="anything at all")
    tc1 = _tracked_change("insertion", "Alice", "unrelated text one")
    tc2 = _tracked_change("deletion", "Alice", "unrelated text two")
    tracked = _tracked_changes("doc", "v2", [tc1, tc2])

    result = round_level_fallback_attribution(hunk, tracked)

    assert result is not None
    assert result.author == "Alice"
    assert result.date is None


def test_round_level_fallback_refuses_with_two_distinct_authors() -> None:
    """The 'prove it doesn't guess' test (issue #118's own required
    verification): 14/30 real corpus versions have two distinct authors in
    one file (both parties' marks) and must not be treated as
    single-author."""
    hunk = TextHunk(kind="insert", old_text="", new_text="anything at all")
    tc1 = _tracked_change("insertion", "Alice", "unrelated text one")
    tc2 = _tracked_change("insertion", "Bob", "unrelated text two")
    tracked = _tracked_changes("doc", "v2", [tc1, tc2])

    result = round_level_fallback_attribution(hunk, tracked)

    assert result is None


def test_round_level_fallback_refuses_on_empty_changes() -> None:
    hunk = TextHunk(kind="insert", old_text="", new_text="anything at all")
    tracked = _tracked_changes("doc", "v2", [])

    assert round_level_fallback_attribution(hunk, tracked) is None


def test_round_level_fallback_gates_on_raw_author_string_not_party_side() -> None:
    """Explicitly NOT gated on which party ("us"/"counterparty") the author
    resolves to (issue #119 is a separate, later stage) — two distinct raw
    author strings refuse to fire even though this function has no way to
    know or care which side either one is on."""
    hunk = TextHunk(kind="delete", old_text="x", new_text="")
    tc1 = _tracked_change("deletion", "J. Smith", "x")
    tc2 = _tracked_change("deletion", "M. Chen", "y", clause_path="2")
    tracked = _tracked_changes("doc", "v2", [tc1, tc2])

    assert round_level_fallback_attribution(hunk, tracked) is None


def test_round_level_fallback_infers_tracked_type_from_hunk_kind() -> None:
    tc = _tracked_change("insertion", "Alice", "text")
    tracked = _tracked_changes("doc", "v2", [tc])

    insert_hunk = TextHunk(kind="insert", old_text="", new_text="x")
    delete_hunk = TextHunk(kind="delete", old_text="x", new_text="")
    replace_hunk = TextHunk(kind="replace", old_text="x", new_text="y")

    assert round_level_fallback_attribution(insert_hunk, tracked).tracked_type == "insertion"  # type: ignore[union-attr]
    assert round_level_fallback_attribution(delete_hunk, tracked).tracked_type == "deletion"  # type: ignore[union-attr]
    # replace prefers the insertion-side convention, same as _match_hunk.
    assert round_level_fallback_attribution(replace_hunk, tracked).tracked_type == "insertion"  # type: ignore[union-attr]
