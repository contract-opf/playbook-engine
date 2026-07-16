"""Tests for the diff engine (L4, issue #17).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional party names only
('Alice Corp', 'Beta Ltd').
"""

from __future__ import annotations

import pytest

from playbook_engine.clause_aligner import align_versions
from playbook_engine.clause_classifier import ClassifiedClause, ClauseClassification
from playbook_engine.clause_differ import (
    ClauseDiff,
    DocumentDiff,
    TextHunk,
    diff_aligned,
)
from playbook_engine.clause_tree import ClauseNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(path: str, heading: str | None = None, text: str = "") -> ClauseNode:
    return ClauseNode(
        clause_path=path,
        heading=heading,
        text=text,
        char_span=(0, max(1, len(text or heading or "x"))),
    )


def _cc(path: str, taxonomy_id: str | None, text: str = "") -> ClassifiedClause:
    if taxonomy_id is None:
        cls = ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
    else:
        cls = ClauseClassification(taxonomy_id=taxonomy_id, confidence=1.0, basis="exact_match")
    return ClassifiedClause(node=_node(path, taxonomy_id, text), classification=cls)


def _align_and_diff(
    versions: list[tuple[str, list[ClassifiedClause]]],
) -> DocumentDiff:
    alignments = align_versions(versions)
    version_order = [vid for vid, _ in versions]
    return diff_aligned(alignments, version_order)


# ---------------------------------------------------------------------------
# TextHunk dataclass
# ---------------------------------------------------------------------------


def test_text_hunk_valid_kinds() -> None:
    for kind in ("insert", "delete", "replace"):
        h = TextHunk(kind=kind, old_text="old", new_text="new")
        assert h.kind == kind


def test_text_hunk_invalid_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        TextHunk(kind="equal", old_text="x", new_text="x")


def test_text_hunk_to_dict() -> None:
    h = TextHunk(kind="insert", old_text="", new_text="new clause text")
    d = h.to_dict()
    assert d == {"kind": "insert", "old_text": "", "new_text": "new clause text"}


def test_text_hunk_frozen() -> None:
    h = TextHunk(kind="insert", old_text="", new_text="x")
    with pytest.raises((AttributeError, TypeError)):
        h.kind = "delete"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ClauseDiff dataclass
# ---------------------------------------------------------------------------


def test_clause_diff_invalid_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        ClauseDiff(
            taxonomy_id="ind",
            clause_path_before="1",
            clause_path_after="1",
            kind="bogus",
            hunks=(),
            text_before="x",
            text_after="x",
        )


def test_clause_diff_to_dict_includes_hunks() -> None:
    h = TextHunk(kind="insert", old_text="", new_text="new text")
    cd = ClauseDiff(
        taxonomy_id="ind",
        clause_path_before="1",
        clause_path_after="1",
        kind="modified",
        hunks=(h,),
        text_before="old text",
        text_after="new text",
    )
    d = cd.to_dict()
    assert d["taxonomy_id"] == "ind"
    assert len(d["hunks"]) == 1
    assert d["hunks"][0]["kind"] == "insert"


# ---------------------------------------------------------------------------
# diff_aligned: validation
# ---------------------------------------------------------------------------


def test_diff_aligned_requires_at_least_two_versions() -> None:
    alignments = align_versions([("v1", [_cc("1", "ind", "text")])])
    with pytest.raises(ValueError, match="2 versions"):
        diff_aligned(alignments, ["v1"])


# ---------------------------------------------------------------------------
# diff_aligned: identical versions → unchanged, no hunks
# (acceptance criterion: unchanged clauses produce no hunks)
# ---------------------------------------------------------------------------


def test_unchanged_clause_has_no_hunks() -> None:
    """Acceptance criterion: unchanged clauses produce no hunks."""
    v1 = [_cc("1", "ind", "Alice Corp shall indemnify Beta Ltd for all losses.")]
    v2 = [_cc("1", "ind", "Alice Corp shall indemnify Beta Ltd for all losses.")]
    doc = _align_and_diff([("v1", v1), ("v2", v2)])

    assert len(doc.consecutive) == 1
    consecutive = doc.consecutive[0]
    assert len(consecutive.diffs) == 1
    cd = consecutive.diffs[0]
    assert cd.kind == "unchanged"
    assert cd.hunks == ()


def test_unchanged_clause_net_diff_no_hunks() -> None:
    v1 = [_cc("1", "ind", "Alice Corp shall indemnify Beta Ltd.")]
    v2 = [_cc("1", "ind", "Alice Corp shall indemnify Beta Ltd.")]
    doc = _align_and_diff([("v1", v1), ("v2", v2)])
    assert doc.net.diffs[0].kind == "unchanged"
    assert doc.net.diffs[0].hunks == ()


def test_two_unchanged_clauses_no_hunks() -> None:
    v1 = [_cc("1", "ind", "Alice indemnifies."), _cc("2", "gov", "Delaware law.")]
    v2 = [_cc("1", "ind", "Alice indemnifies."), _cc("2", "gov", "Delaware law.")]
    doc = _align_and_diff([("v1", v1), ("v2", v2)])
    for cd in doc.consecutive[0].diffs:
        assert cd.hunks == ()
        assert cd.kind == "unchanged"


# ---------------------------------------------------------------------------
# diff_aligned: modified clause → correct hunks
# (acceptance criterion: correct hunks on a fixture)
# ---------------------------------------------------------------------------


def test_modified_clause_produces_hunks() -> None:
    """Acceptance criterion: correct hunks on a modified clause."""
    v1 = [_cc("1", "ind", "Alice Corp shall indemnify Beta Ltd for all losses.")]
    v2 = [_cc("1", "ind", "Alice Corp shall indemnify Beta Ltd for all direct losses.")]
    doc = _align_and_diff([("v1", v1), ("v2", v2)])

    cd = doc.consecutive[0].diffs[0]
    assert cd.kind == "modified"
    assert len(cd.hunks) > 0

    # "all" replaced/removed in favour of "all direct"
    hunk_kinds = {h.kind for h in cd.hunks}
    assert hunk_kinds <= {"insert", "delete", "replace"}


def test_modified_clause_hunks_no_equal_spans() -> None:
    """Token discipline: equal spans are never in hunks."""
    v1 = [_cc("1", "ind", "This agreement covers all losses incurred by either party.")]
    v2 = [_cc("1", "ind", "This agreement covers direct losses incurred by either party.")]
    doc = _align_and_diff([("v1", v1), ("v2", v2)])
    for cd in doc.consecutive[0].diffs:
        for hunk in cd.hunks:
            assert hunk.kind != "equal"


def test_modified_clause_text_before_after_correct() -> None:
    text_v1 = "Alice Corp shall indemnify Beta Ltd for all losses."
    text_v2 = "Alice Corp shall indemnify Beta Ltd for direct losses only."
    v1 = [_cc("1", "ind", text_v1)]
    v2 = [_cc("1", "ind", text_v2)]
    doc = _align_and_diff([("v1", v1), ("v2", v2)])
    cd = doc.consecutive[0].diffs[0]
    assert cd.text_before == text_v1
    assert cd.text_after == text_v2


# ---------------------------------------------------------------------------
# diff_aligned: added / removed clauses
# ---------------------------------------------------------------------------


def test_added_clause_kind_and_no_hunks() -> None:
    v1 = [_cc("1", "ind", "Alice shall indemnify.")]
    v2 = [_cc("1", "ind", "Alice shall indemnify."), _cc("2", "ins", "Insurance required.")]
    doc = _align_and_diff([("v1", v1), ("v2", v2)])

    ins_diffs = [cd for cd in doc.consecutive[0].diffs if cd.taxonomy_id == "ins"]
    assert len(ins_diffs) == 1
    assert ins_diffs[0].kind == "added"
    assert ins_diffs[0].hunks == ()
    assert ins_diffs[0].clause_path_before is None
    assert ins_diffs[0].clause_path_after == "2"


def test_removed_clause_kind_and_no_hunks() -> None:
    v1 = [_cc("1", "ind", "Alice shall indemnify."), _cc("2", "ins", "Insurance required.")]
    v2 = [_cc("1", "ind", "Alice shall indemnify.")]
    doc = _align_and_diff([("v1", v1), ("v2", v2)])

    ins_diffs = [cd for cd in doc.consecutive[0].diffs if cd.taxonomy_id == "ins"]
    assert len(ins_diffs) == 1
    assert ins_diffs[0].kind == "removed"
    assert ins_diffs[0].hunks == ()
    assert ins_diffs[0].clause_path_before == "2"
    assert ins_diffs[0].clause_path_after is None


# ---------------------------------------------------------------------------
# diff_aligned: consecutive vs net diff
# ---------------------------------------------------------------------------


def test_consecutive_diff_count() -> None:
    """k versions produce k-1 consecutive diffs."""
    text = "Some clause text here."
    versions = [(f"v{i}", [_cc("1", "ind", text)]) for i in range(1, 5)]
    doc = _align_and_diff(versions)
    assert len(doc.consecutive) == 3


def test_net_diff_is_first_to_last() -> None:
    """Net diff always spans first → last version."""
    v1 = [_cc("1", "ind", "Original indemnification text.")]
    v2 = [_cc("1", "ind", "Revised indemnification text.")]
    v3 = [_cc("1", "ind", "Final indemnification clause text.")]
    doc = _align_and_diff([("v1", v1), ("v2", v2), ("v3", v3)])
    assert doc.net.version_before == "v1"
    assert doc.net.version_after == "v3"


def test_net_diff_unchanged_when_reverted() -> None:
    """If v3 reverts to v1 text, the net diff sees no change."""
    original = "Alice Corp shall indemnify Beta Ltd."
    v1 = [_cc("1", "ind", original)]
    v2 = [_cc("1", "ind", "Alice Corp shall fully indemnify Beta Ltd against all claims.")]
    v3 = [_cc("1", "ind", original)]
    doc = _align_and_diff([("v1", v1), ("v2", v2), ("v3", v3)])
    net_cd = doc.net.diffs[0]
    assert net_cd.kind == "unchanged"
    assert net_cd.hunks == ()


def test_consecutive_captures_intermediate_changes() -> None:
    """Consecutive diffs capture v1→v2 even when v3 reverts."""
    original = "Alice Corp shall indemnify Beta Ltd."
    v1 = [_cc("1", "ind", original)]
    v2 = [_cc("1", "ind", "Alice Corp shall fully indemnify Beta Ltd against all claims.")]
    v3 = [_cc("1", "ind", original)]
    doc = _align_and_diff([("v1", v1), ("v2", v2), ("v3", v3)])

    v1_to_v2 = doc.consecutive[0]
    v2_to_v3 = doc.consecutive[1]

    assert v1_to_v2.diffs[0].kind == "modified"
    assert v2_to_v3.diffs[0].kind == "modified"


# ---------------------------------------------------------------------------
# diff_aligned: version_order
# ---------------------------------------------------------------------------


def test_version_order_stored_in_document_diff() -> None:
    v1 = [_cc("1", "ind", "text")]
    v2 = [_cc("1", "ind", "text")]
    doc = _align_and_diff([("v1", v1), ("v2", v2)])
    assert doc.version_order == ("v1", "v2")


# ---------------------------------------------------------------------------
# VersionDiff.changed()
# ---------------------------------------------------------------------------


def test_version_diff_changed_filters_unchanged() -> None:
    v1 = [_cc("1", "ind", "same text."), _cc("2", "gov", "different text.")]
    v2 = [_cc("1", "ind", "same text."), _cc("2", "gov", "revised text.")]
    doc = _align_and_diff([("v1", v1), ("v2", v2)])
    changed = doc.consecutive[0].changed()
    assert len(changed) == 1
    assert changed[0].taxonomy_id == "gov"


# ---------------------------------------------------------------------------
# DocumentDiff.to_dict()
# ---------------------------------------------------------------------------


def test_document_diff_to_dict_structure() -> None:
    v1 = [_cc("1", "ind", "Alice indemnifies.")]
    v2 = [_cc("1", "ind", "Alice fully indemnifies.")]
    doc = _align_and_diff([("v1", v1), ("v2", v2)])
    d = doc.to_dict()
    assert "version_order" in d
    assert "consecutive" in d
    assert "net" in d
    assert d["version_order"] == ["v1", "v2"]
