"""Tests for segmentation_grounding — deterministic block-ID → ClauseTree grounding.

All fixtures use synthetic text; no real corpus, no LLM calls.
"""

from __future__ import annotations

import pytest

from playbook_engine.segmentation_grounding import (
    Block,
    GroundingError,
    SegNode,
    ground_segmentation,
)


def _stream(segments: list[str]) -> tuple[str, list[Block]]:
    """Build a canonical text ("\\n"-joined) + block stream with exact char spans."""
    blocks: list[Block] = []
    offset = 0
    for i, seg in enumerate(segments):
        if i > 0:
            offset += 1  # the "\n" separator
        blocks.append(
            Block(block_id=f"b{i}", page=0, char_span=(offset, offset + len(seg)), text=seg)
        )
        offset += len(seg)
    return "\n".join(segments), blocks


def _ground(canonical_text, blocks, seg_nodes):
    return ground_segmentation(
        document_id="doc",
        version="v1",
        source_file="v1.rtf",
        canonical_text=canonical_text,
        blocks=blocks,
        seg_nodes=seg_nodes,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_grounds_tree_with_child_and_taxonomy() -> None:
    text, blocks = _stream(
        [
            "Indemnification clause text here",
            "sub clause a text",
            "Governing Law clause text",
        ]
    )
    nodes = [
        SegNode(
            "n1",
            None,
            1,
            "Indemnification",
            "indemnification",
            "b0",
            "b0",
            start_quote="Indemnification",
            end_quote="here",
        ),
        SegNode("n1a", "n1", 1, None, "indemnification", "b1", "b1"),
        SegNode("n2", None, 2, "Governing Law", "governing_law", "b2", "b2"),
    ]
    result = _ground(text, blocks, nodes)

    roots = result.tree.nodes
    assert [r.clause_path for r in roots] == ["1", "2"]
    # verbatim text reconstructed from the block span
    assert roots[0].text == "Indemnification clause text here"
    assert roots[0].char_span == (0, 32)
    assert roots[0].heading == "Indemnification"
    # child nested with dotted path
    assert [c.clause_path for c in roots[0].children] == ["1.1"]
    assert roots[0].children[0].text == "sub clause a text"
    assert roots[1].text == "Governing Law clause text"
    # classification carried per path
    assert result.taxonomy_by_path == {
        "1": "indemnification",
        "1.1": "indemnification",
        "2": "governing_law",
    }


def test_multi_block_span_reconstructs_full_text() -> None:
    text, blocks = _stream(["Clause part one", "clause part two", "next clause"])
    nodes = [
        SegNode("n1", None, 1, "Combined", "term", "b0", "b1"),  # spans b0..b1
        SegNode("n2", None, 2, "Next", "notices", "b2", "b2"),
    ]
    result = _ground(text, blocks, nodes)
    # span covers both blocks and the separator between them
    assert result.tree.nodes[0].text == "Clause part one\nclause part two"


def test_order_field_determines_clause_numbering() -> None:
    text, blocks = _stream(["first", "second"])
    # deliberately list out of order; `order` should drive numbering
    nodes = [
        SegNode("n2", None, 2, "B", "b", "b1", "b1"),
        SegNode("n1", None, 1, "A", "a", "b0", "b0"),
    ]
    result = _ground(text, blocks, nodes)
    assert [(r.clause_path, r.heading) for r in result.tree.nodes] == [("1", "A"), ("2", "B")]


# ---------------------------------------------------------------------------
# Fail-loud grounding gate
# ---------------------------------------------------------------------------


def test_unknown_block_id_raises() -> None:
    text, blocks = _stream(["only block"])
    nodes = [SegNode("n1", None, 1, None, None, "b0", "b9")]
    with pytest.raises(GroundingError, match="unknown end_block_id"):
        _ground(text, blocks, nodes)


def test_inverted_block_range_raises() -> None:
    text, blocks = _stream(["a", "b"])
    nodes = [SegNode("n1", None, 1, None, None, "b1", "b0")]
    with pytest.raises(GroundingError, match="start_block after end_block"):
        _ground(text, blocks, nodes)


def test_boundary_quote_mismatch_raises() -> None:
    text, blocks = _stream(["Indemnification clause"])
    nodes = [SegNode("n1", None, 1, "X", None, "b0", "b0", start_quote="Governing Law")]
    with pytest.raises(GroundingError, match="start_quote"):
        _ground(text, blocks, nodes)


def test_orphan_parent_raises() -> None:
    text, blocks = _stream(["a"])
    nodes = [SegNode("n1", "ghost", 1, None, None, "b0", "b0")]
    with pytest.raises(GroundingError, match="parent_id"):
        _ground(text, blocks, nodes)


def test_no_roots_raises() -> None:
    text, blocks = _stream(["a"])
    # single node whose parent is itself -> no root
    nodes = [SegNode("n1", "n1", 1, None, None, "b0", "b0")]
    with pytest.raises(GroundingError):
        _ground(text, blocks, nodes)


def test_empty_segmentation_yields_empty_tree() -> None:
    text, blocks = _stream(["a"])
    result = _ground(text, blocks, [])
    assert result.tree.nodes == []
    assert result.taxonomy_by_path == {}


# ---------------------------------------------------------------------------
# #86 — page threaded from the start block
# ---------------------------------------------------------------------------


def test_grounds_tree_sets_page_from_start_block() -> None:
    blocks = [
        Block(block_id="b0", page=1, char_span=(0, 5), text="first"),
        Block(block_id="b1", page=2, char_span=(6, 12), text="second"),
    ]
    text = "first\nsecond"
    nodes = [
        SegNode("n1", None, 1, "A", None, "b0", "b0"),
        SegNode("n2", None, 2, "B", None, "b1", "b1"),
    ]
    result = _ground(text, blocks, nodes)
    assert result.tree.nodes[0].page == 1
    assert result.tree.nodes[1].page == 2


def test_grounds_tree_maps_unpaginated_zero_to_none() -> None:
    """page=0 (RTF/docling — 'not paginated') must surface as None, matching
    ClauseNode.page's documented contract."""
    text, blocks = _stream(["only block"])  # _stream() always uses page=0
    nodes = [SegNode("n1", None, 1, None, None, "b0", "b0")]
    result = _ground(text, blocks, nodes)
    assert result.tree.nodes[0].page is None


def test_grounds_tree_records_start_page_when_span_crosses_pages() -> None:
    """A clause can legitimately span a page break — only the START page is
    recorded (the page the clause begins on), never the end block's page."""
    blocks = [
        Block(block_id="b0", page=3, char_span=(0, 5), text="first"),
        Block(block_id="b1", page=4, char_span=(6, 12), text="second"),
    ]
    text = "first\nsecond"
    nodes = [SegNode("n1", None, 1, "Combined", None, "b0", "b1")]
    result = _ground(text, blocks, nodes)
    assert result.tree.nodes[0].page == 3


def test_grounds_tree_page_on_parent_survives_child_truncation() -> None:
    """A parent node's page comes from its own start block and must be
    unaffected by the end-truncation applied when a parent's nominal range
    encloses its first child's range (see build()'s truncation comment)."""
    text, base_blocks = _stream(
        [
            "Indemnification clause text here",
            "sub clause a text",
            "Governing Law clause text",
        ]
    )
    pages = [7, 7, 8]
    blocks = [
        Block(block_id=b.block_id, page=p, char_span=b.char_span, text=b.text)
        for b, p in zip(base_blocks, pages, strict=True)
    ]
    nodes = [
        SegNode("n1", None, 1, "Indemnification", "indemnification", "b0", "b1"),
        SegNode("n1a", "n1", 1, None, "indemnification", "b1", "b1"),
        SegNode("n2", None, 2, "Governing Law", "governing_law", "b2", "b2"),
    ]
    result = _ground(text, blocks, nodes)
    roots = result.tree.nodes
    assert roots[0].page == 7
    assert roots[0].children[0].page == 7
    assert roots[1].page == 8
