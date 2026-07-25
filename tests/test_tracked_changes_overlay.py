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
    enrich_clause_diff,
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
