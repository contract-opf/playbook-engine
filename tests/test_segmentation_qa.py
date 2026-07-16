"""Tests for segmentation_qa — deterministic QA gates + verify/repair loop.

All fixtures use synthetic text; no real corpus, no LLM calls. Mirrors the
``_stream`` fixture helper from ``test_segmentation_grounding.py``.
"""

from __future__ import annotations

import pytest

from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.segmentation_grounding import Block, GroundingResult, SegNode
from playbook_engine.segmentation_qa import (
    SegmentationQAError,
    _check_reconstruction,
    run_gates,
    segment_verify_repair,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


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


def _run(canonical_text, blocks, seg_nodes, taxonomy_ids):
    return run_gates(canonical_text, blocks, seg_nodes, taxonomy_ids=taxonomy_ids)


# ---------------------------------------------------------------------------
# Happy path — a clean seg_nodes set passes run_gates
# ---------------------------------------------------------------------------


def test_clean_segmentation_passes_all_gates_and_returns_grounding_result() -> None:
    text, blocks = _stream(["Indemnification clause text", "Governing Law clause text"])
    nodes = [
        SegNode("n1", None, 1, "Indemnification", "indemnification", "b0", "b0"),
        SegNode("n2", None, 2, "Governing Law", "governing_law", "b1", "b1"),
    ]
    result = _run(text, blocks, nodes, ["indemnification", "governing_law"])

    assert [r.clause_path for r in result.tree.nodes] == ["1", "2"]
    assert result.tree.nodes[0].text == "Indemnification clause text"
    assert result.taxonomy_by_path == {"1": "indemnification", "2": "governing_law"}


def test_clean_segmentation_with_nested_children_and_whitespace_passes() -> None:
    # Extra internal whitespace between blocks tolerated by coverage/reconstruction.
    text, blocks = _stream(["Parent heading", "  child text  ", "Next clause"])
    nodes = [
        SegNode("n1", None, 1, "Parent", "term", "b0", "b0"),
        SegNode("n1a", "n1", 1, None, "term", "b1", "b1"),
        SegNode("n2", None, 2, "Next", "notices", "b2", "b2"),
    ]
    result = _run(text, blocks, nodes, ["term", "notices"])
    assert result.tree.nodes[0].children[0].text == "  child text  "


def test_clean_segmentation_with_enclosing_parent_span_passes() -> None:
    # The parent's own block range (b0..b2) ENCLOSES both children's ranges
    # (b1, b2) rather than sitting disjoint from them — the ordinary nested
    # case where a heading block is followed by sub-clause blocks and the
    # model reports the parent's range as spanning the whole family. Grounding
    # must truncate the parent's own span/text to just its heading (b0) so
    # coverage/reconstruction see every node's span as disjoint.
    text, blocks = _stream(["Parent heading", "sub-clause one", "sub-clause two"])
    nodes = [
        SegNode("n1", None, 1, "Parent", "term", "b0", "b2"),
        SegNode("n1a", "n1", 1, None, "term", "b1", "b1"),
        SegNode("n1b", "n1", 2, None, "term", "b2", "b2"),
    ]
    result = _run(text, blocks, nodes, ["term"])
    assert isinstance(result, GroundingResult)
    parent = result.tree.nodes[0]
    # Parent's own text stops where its first child begins — it still
    # includes the "\n" separator before that child, exactly as a
    # multi-block LEAF's text includes inter-block separators (see
    # test_multi_block_span_reconstructs_full_text in
    # test_segmentation_grounding.py).
    assert parent.text == "Parent heading\n"
    assert [c.text for c in parent.children] == ["sub-clause one", "sub-clause two"]


# ---------------------------------------------------------------------------
# Gate 1 — grounding
# ---------------------------------------------------------------------------


def test_grounding_gate_pass() -> None:
    text, blocks = _stream(["a clause"])
    nodes = [SegNode("n1", None, 1, None, None, "b0", "b0")]
    result = _run(text, blocks, nodes, [])
    assert result.tree.nodes[0].text == "a clause"


def test_grounding_gate_failure_raises_naming_grounding() -> None:
    text, blocks = _stream(["only block"])
    nodes = [SegNode("n1", None, 1, None, None, "b0", "b9")]  # unknown end_block_id
    with pytest.raises(SegmentationQAError, match="grounding gate"):
        _run(text, blocks, nodes, [])


# ---------------------------------------------------------------------------
# Gate 2 — coverage
# ---------------------------------------------------------------------------


def test_coverage_gate_failure_on_gap() -> None:
    # Three blocks but only b0 and b2 are claimed — b1's text is an
    # uncovered non-whitespace gap between them.
    text, blocks = _stream(["first clause", "SKIPPED clause text", "last clause"])
    nodes = [
        SegNode("n1", None, 1, None, None, "b0", "b0"),
        SegNode("n2", None, 2, None, None, "b2", "b2"),
    ]
    with pytest.raises(SegmentationQAError, match="coverage gate"):
        _run(text, blocks, nodes, [])


def test_coverage_gate_failure_on_overlap() -> None:
    # n1 spans b0..b1 and n2 spans b1..b1 again as a sibling leaf: b1's text
    # is claimed twice, which is an overlap between leaf spans.
    text, blocks = _stream(["clause one", "clause two"])
    nodes = [
        SegNode("n1", None, 1, None, None, "b0", "b1"),
        SegNode("n2", None, 2, None, None, "b1", "b1"),
    ]
    with pytest.raises(SegmentationQAError, match="coverage gate"):
        _run(text, blocks, nodes, [])


def test_coverage_gate_tolerates_pure_whitespace_gap() -> None:
    # A gap of only whitespace (e.g. between block spans) does not fail.
    text = "first block\n\n\nsecond block"
    blocks = [
        Block(block_id="b0", page=0, char_span=(0, 11), text="first block"),
        Block(block_id="b1", page=0, char_span=(14, 26), text="second block"),
    ]
    nodes = [
        SegNode("n1", None, 1, None, None, "b0", "b0"),
        SegNode("n2", None, 2, None, None, "b1", "b1"),
    ]
    result = _run(text, blocks, nodes, [])
    assert [r.clause_path for r in result.tree.nodes] == ["1", "2"]


# ---------------------------------------------------------------------------
# Gate 3 — reconstruction
# ---------------------------------------------------------------------------


def test_reconstruction_gate_pass_via_run_gates() -> None:
    """Through the real ``run_gates`` path, grounding always slices node text
    directly from ``canonical_text`` at the resolved span, so a clean
    segmentation always passes reconstruction (exercised already by the
    happy-path tests above). This gate's fail case (below) targets the
    private check directly, since grounding's construction makes a
    text/span mismatch unreachable through the public ``run_gates`` surface —
    reconstruction exists as an independent, second verification of the same
    "every character is accounted for" property that ``run_gates`` composes,
    defending against a coverage-gate arithmetic bug that content comparison
    alone would still catch."""
    text, blocks = _stream(["a clause"])
    nodes = [SegNode("n1", None, 1, None, None, "b0", "b0")]
    result = _run(text, blocks, nodes, [])
    assert result.tree.nodes[0].text == "a clause"


def test_reconstruction_gate_failure_on_direct_text_mismatch() -> None:
    """Unit-test the private check directly against a hand-built tree whose
    node ``.text`` disagrees with ``canonical_text`` at its own ``char_span``
    — a state ``ground_segmentation`` can never produce (it always slices
    ``canonical_text[start:end]``), but one this gate must still reject if it
    ever occurred (e.g. a future bug bypassing grounding)."""
    text = "ABCDEFGH"
    tree = ClauseTree(
        document_id="doc",
        version="v1",
        source_file="v1.rtf",
        nodes=[ClauseNode(clause_path="1", heading=None, text="ZZZZZZZZ", char_span=(0, 8))],
    )
    with pytest.raises(SegmentationQAError, match="reconstruction gate"):
        _check_reconstruction(text, tree)


# ---------------------------------------------------------------------------
# Gate 4 — tree
# ---------------------------------------------------------------------------


def test_tree_gate_failure_on_child_starting_before_parent() -> None:
    """Child leaf's block range starts before the parent's own start block,
    producing a child span that starts before its parent — a tree invariant
    violation that grounding's own checks do not catch (grounding only
    orders a node's own start/end block; it does not compare across nodes)."""
    text, blocks = _stream(["parent heading text", "text that precedes it"])
    nodes = [
        SegNode("n1", None, 1, "Parent", None, "b1", "b1"),  # parent = block b1
        SegNode("n1a", "n1", 1, None, None, "b0", "b0"),  # child = block b0 (precedes parent)
    ]
    with pytest.raises(SegmentationQAError, match="tree gate"):
        _run(text, blocks, nodes, [])


# ---------------------------------------------------------------------------
# Gate 5 — taxonomy
# ---------------------------------------------------------------------------


def test_taxonomy_gate_pass_with_null_taxonomy() -> None:
    text, blocks = _stream(["noise block"])
    nodes = [SegNode("n1", None, 1, None, None, "b0", "b0")]  # taxonomy_id=None always allowed
    result = _run(text, blocks, nodes, ["indemnification"])
    assert result.taxonomy_by_path == {"1": None}


def test_taxonomy_gate_failure_on_out_of_enum_id() -> None:
    text, blocks = _stream(["a clause"])
    nodes = [SegNode("n1", None, 1, None, "not_a_real_taxonomy_id", "b0", "b0")]
    with pytest.raises(SegmentationQAError, match="taxonomy gate"):
        _run(text, blocks, nodes, ["indemnification", "governing_law"])


# ---------------------------------------------------------------------------
# segment_verify_repair — verify/repair loop
# ---------------------------------------------------------------------------


def _good_nodes() -> list[SegNode]:
    return [SegNode("n1", None, 1, "Clause", "indemnification", "b0", "b0")]


def _bad_nodes() -> list[SegNode]:
    # Unknown block id -> fails the grounding gate every time.
    return [SegNode("n1", None, 1, "Clause", "indemnification", "b0", "b999")]


def test_repair_succeeds_after_one_bad_attempt() -> None:
    text, blocks = _stream(["Indemnification clause"])
    calls: list[int] = []

    def fake_segment_fn(canonical_text, blocks_arg):
        calls.append(1)
        if len(calls) == 1:
            return _bad_nodes()
        return _good_nodes()

    result = segment_verify_repair(
        text,
        blocks,
        taxonomy_ids=["indemnification"],
        segment_fn=fake_segment_fn,
        max_repairs=2,
    )
    assert len(calls) == 2
    assert result.tree.nodes[0].text == "Indemnification clause"


def test_repair_receives_expected_call_arguments() -> None:
    # Per the ticket's call contract, segment_fn is called as
    # segment_fn(canonical_text, blocks) — taxonomy_ids is not threaded
    # through; a production caller pre-binds it into segment_fn itself.
    text, blocks = _stream(["Indemnification clause"])
    seen: list[tuple] = []

    def fake_segment_fn(canonical_text, blocks_arg):
        seen.append((canonical_text, blocks_arg))
        return _good_nodes()

    segment_verify_repair(
        text,
        blocks,
        taxonomy_ids=["indemnification"],
        segment_fn=fake_segment_fn,
        max_repairs=2,
    )
    assert seen == [(text, blocks)]


def test_always_bad_output_raises_after_max_repairs_with_no_fallback() -> None:
    text, blocks = _stream(["Indemnification clause"])
    calls: list[int] = []

    def always_bad_segment_fn(canonical_text, blocks_arg):
        calls.append(1)
        return _bad_nodes()

    with pytest.raises(SegmentationQAError, match="exhausted"):
        segment_verify_repair(
            text,
            blocks,
            taxonomy_ids=["indemnification"],
            segment_fn=always_bad_segment_fn,
            max_repairs=2,
        )
    # 1 initial call + 2 repairs = 3 total attempts; never falls back to any
    # heuristic/deterministic segmenter — the only calls made are to the
    # injected segment_fn, and the final outcome is a raised error.
    assert len(calls) == 3


def test_repair_passes_gate_failure_feedback() -> None:
    """A repair-aware ``segment_fn`` (one declaring a third parameter) gets
    the previous attempt's ``SegmentationQAError`` on retry: ``None`` on the
    first call, the exact exception the first attempt's gate failure raised
    on the second — so a repair can fold the specific failure into its next
    prompt instead of retrying with byte-identical input."""
    text, blocks = _stream(["Indemnification clause"])
    seen_errors: list[SegmentationQAError | None] = []

    def fake_segment_fn(canonical_text, blocks_arg, last_error=None):
        seen_errors.append(last_error)
        if len(seen_errors) == 1:
            return _bad_nodes()
        return _good_nodes()

    result = segment_verify_repair(
        text,
        blocks,
        taxonomy_ids=["indemnification"],
        segment_fn=fake_segment_fn,
        max_repairs=2,
    )

    assert len(seen_errors) == 2
    assert seen_errors[0] is None
    assert isinstance(seen_errors[1], SegmentationQAError)
    assert "grounding gate" in str(seen_errors[1])
    assert result.tree.nodes[0].text == "Indemnification clause"


def test_repair_unaware_segment_fn_still_called_with_two_args() -> None:
    """A ``segment_fn`` matching only the base two-argument shape (no third
    parameter) is never passed a third argument — the repair-aware behavior
    is opt-in, not forced on every injected callable."""
    text, blocks = _stream(["Indemnification clause"])
    calls: list[tuple] = []

    def fake_segment_fn(canonical_text, blocks_arg):
        calls.append((canonical_text, blocks_arg))
        if len(calls) == 1:
            return _bad_nodes()
        return _good_nodes()

    segment_verify_repair(
        text,
        blocks,
        taxonomy_ids=["indemnification"],
        segment_fn=fake_segment_fn,
        max_repairs=2,
    )
    assert calls == [(text, blocks), (text, blocks)]


def test_default_max_repairs_is_two() -> None:
    text, blocks = _stream(["Indemnification clause"])
    calls: list[int] = []

    def always_bad_segment_fn(canonical_text, blocks_arg):
        calls.append(1)
        return _bad_nodes()

    with pytest.raises(SegmentationQAError):
        segment_verify_repair(
            text,
            blocks,
            taxonomy_ids=["indemnification"],
            segment_fn=always_bad_segment_fn,
        )
    assert len(calls) == 3  # 1 + default max_repairs(2)
