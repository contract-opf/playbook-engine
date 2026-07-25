"""Deterministic grounding of LLM segmentation output — LLM-first segmentation.

The LLM segmenter returns clause boundaries by **block ID** (never character
offsets or verbatim clause text). This module reconstructs each clause's
verbatim ``text`` and ``char_span`` from the canonical block stream and builds
the engine's :class:`~playbook_engine.clause_tree.ClauseTree`.

The LLM is never trusted for offsets or content: everything here is
deterministic, and any inconsistency between the model's block references /
boundary quotes and the actual source raises :class:`GroundingError` (the
grounding gate — one of the fail-loud QA gates).

Contract types (:class:`Block`, :class:`SegNode`) are the seam between the
extractor (produces ``Block`` stream + canonical text) and the LLM segmenter
(produces ``SegNode`` list).

Internal-node span/text: a node WITHOUT children keeps its full resolved
block range as both ``char_span`` and ``text`` (a leaf may legitimately span
several blocks — see ``test_multi_block_span_reconstructs_full_text``). A
node WITH children is truncated to end where its first child's span begins,
so its own ``char_span``/``text`` cover only its own heading/preamble — never
its children's content — matching :class:`~playbook_engine.clause_tree.ClauseNode`'s
contract ("text — body text of this node, not including children") and
keeping every node's span disjoint from every other node's, which
:mod:`playbook_engine.segmentation_qa`'s coverage/reconstruction gates
require.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from playbook_engine.clause_tree import ClauseNode, ClauseTree


class GroundingError(Exception):
    """Raised when LLM segmentation output cannot be grounded to the source.

    A grounding failure means the model's structured output is internally
    inconsistent with the canonical block stream (unknown block id, inverted
    span, boundary quote that doesn't match the resolved text, cyclic/orphaned
    parent). By design there is no fallback — the document is flagged for human
    review.
    """


@dataclass(frozen=True)
class Block:
    """One unit of the canonical extracted text stream.

    Attributes:
        block_id:   Stable identifier the LLM references (e.g. ``"b042"``).
        page:       1-based source page number (0 when not paginated, e.g. RTF).
        char_span:  ``[start, end)`` offsets into the canonical document text.
        text:       The block's verbatim text (``canonical_text[start:end]``).
    """

    block_id: str
    page: int
    char_span: tuple[int, int]
    text: str


@dataclass(frozen=True)
class SegNode:
    """One node of the LLM's structured segmentation output (block-ID anchored).

    The model returns these; grounding turns them into ``ClauseNode``s. It never
    supplies ``char_span`` or full ``text`` — only block references and short
    boundary quotes used to verify the references.
    """

    node_id: str
    parent_id: str | None
    order: int
    heading: str | None
    taxonomy_id: str | None
    start_block_id: str
    end_block_id: str
    start_quote: str = ""
    end_quote: str = ""


@dataclass
class GroundingResult:
    """Output of grounding.

    Attributes:
        tree:             The reconstructed clause tree (structure + verbatim
                          text + citation-grade char spans).
        taxonomy_by_path: Map of ``clause_path`` → LLM-assigned ``taxonomy_id``
                          (or ``None``). Segmentation and classification happen
                          in one LLM pass, so this carries the classification
                          for the downstream observation stage — no separate
                          classify judge needed.
    """

    tree: ClauseTree
    taxonomy_by_path: dict[str, str | None] = field(default_factory=dict)


def _norm_ws(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip — for tolerant
    boundary-quote comparison against extractor/LLM whitespace differences."""
    return re.sub(r"\s+", " ", text).strip()


def ground_segmentation(
    *,
    document_id: str,
    version: str,
    source_file: str,
    canonical_text: str,
    blocks: list[Block],
    seg_nodes: list[SegNode],
) -> GroundingResult:
    """Ground LLM ``seg_nodes`` against the canonical block stream.

    Args:
        document_id:    Document id for the resulting tree.
        version:        Version id (stem) for the resulting tree.
        source_file:    Source filename for the resulting tree.
        canonical_text: The full canonical extracted text the blocks index into.
        blocks:         Block stream in reading order (``char_span`` into
                        ``canonical_text``).
        seg_nodes:      The LLM's structured segmentation output.

    Returns:
        :class:`GroundingResult` (tree + taxonomy-by-path).

    Raises:
        GroundingError: on any block-reference / span / quote / tree
                        inconsistency (fail loud; no fallback).
    """
    # --- Index blocks by id, and record stream position for ordering checks.
    pos_by_id: dict[str, int] = {}
    block_by_id: dict[str, Block] = {}
    for i, b in enumerate(blocks):
        if b.block_id in block_by_id:
            raise GroundingError(f"duplicate block_id {b.block_id!r} in block stream")
        pos_by_id[b.block_id] = i
        block_by_id[b.block_id] = b

    text_len = len(canonical_text)

    # --- Resolve each node's char_span + verbatim text from its block range.
    span_by_node: dict[str, tuple[int, int]] = {}
    text_by_node: dict[str, str] = {}
    for n in seg_nodes:
        if n.start_block_id not in block_by_id:
            raise GroundingError(f"node {n.node_id!r}: unknown start_block_id {n.start_block_id!r}")
        if n.end_block_id not in block_by_id:
            raise GroundingError(f"node {n.node_id!r}: unknown end_block_id {n.end_block_id!r}")
        if pos_by_id[n.start_block_id] > pos_by_id[n.end_block_id]:
            raise GroundingError(f"node {n.node_id!r}: start_block after end_block in stream order")
        start = block_by_id[n.start_block_id].char_span[0]
        end = block_by_id[n.end_block_id].char_span[1]
        if not (0 <= start <= end <= text_len):
            raise GroundingError(
                f"node {n.node_id!r}: span [{start}, {end}] out of bounds "
                f"for canonical text of length {text_len}"
            )
        clause_text = canonical_text[start:end]
        # Boundary-quote gate: the model's own start/end quotes must match the
        # resolved text (whitespace-normalised). A mismatch means the model
        # referenced the wrong blocks — grounding fails rather than emitting a
        # plausible-but-wrong clause.
        norm = _norm_ws(clause_text)
        if n.start_quote and not norm.startswith(_norm_ws(n.start_quote)):
            raise GroundingError(f"node {n.node_id!r}: start_quote does not match resolved text")
        if n.end_quote and not norm.endswith(_norm_ws(n.end_quote)):
            raise GroundingError(f"node {n.node_id!r}: end_quote does not match resolved text")
        span_by_node[n.node_id] = (start, end)
        text_by_node[n.node_id] = clause_text

    # --- Group children by parent, ordered by the model's `order`.
    children_by_parent: dict[str | None, list[SegNode]] = {}
    node_ids = {n.node_id for n in seg_nodes}
    for n in seg_nodes:
        if n.parent_id is not None and n.parent_id not in node_ids:
            raise GroundingError(
                f"node {n.node_id!r}: parent_id {n.parent_id!r} is not a known node"
            )
        children_by_parent.setdefault(n.parent_id, []).append(n)
    for kids in children_by_parent.values():
        kids.sort(key=lambda s: s.order)

    taxonomy_by_path: dict[str, str | None] = {}

    def build(parent_id: str | None, prefix: str, seen: frozenset[str]) -> list[ClauseNode]:
        out: list[ClauseNode] = []
        for idx, n in enumerate(children_by_parent.get(parent_id, []), start=1):
            if n.node_id in seen:  # cycle guard
                raise GroundingError(f"cycle detected at node {n.node_id!r}")
            clause_path = f"{prefix}{idx}" if not prefix else f"{prefix}.{idx}"
            children = build(n.node_id, clause_path, seen | {n.node_id})
            start, end = span_by_node[n.node_id]
            text = text_by_node[n.node_id]
            if children:
                # This node has children: per ClauseNode's contract, `text` is
                # this node's OWN body (heading/preamble), not including
                # children — so when the model's own start/end block range
                # nominally ENCLOSES its first child's range too (the
                # ordinary nested case: a heading block followed by
                # sub-clause blocks, all within the parent's own reported
                # range), truncate this node's own span to stop where that
                # child begins. Without this truncation an enclosing parent
                # range would make the coverage gate see an overlap (parent +
                # child both claim the child's text) and the reconstruction
                # gate double-count the child's text (see segmentation_qa.py's
                # coverage/reconstruction gates, which rely on every node —
                # leaf or internal — owning a disjoint slice of
                # canonical_text).
                #
                # Only truncate when the child's span genuinely starts inside
                # this node's own range (start <= first_child_start < end) —
                # a child whose span lies entirely before or after the
                # parent's own range is a different (and likely invalid)
                # shape that must be left for the tree gate to catch as a
                # sibling/parent-order violation, not silently coerced into
                # an empty or inverted parent span here.
                first_child_start = children[0].char_span[0]
                if start <= first_child_start < end:
                    end = first_child_start
                    text = canonical_text[start:end]
            out.append(
                ClauseNode(
                    clause_path=clause_path,
                    heading=n.heading,
                    text=text,
                    char_span=(start, end),
                    children=children,
                )
            )
            taxonomy_by_path[clause_path] = n.taxonomy_id
        return out

    roots = build(None, "", frozenset())
    if not roots and seg_nodes:
        raise GroundingError("segmentation produced no root nodes (no node has parent_id=None)")

    tree = ClauseTree(
        document_id=document_id,
        version=version,
        source_file=source_file,
        nodes=roots,
    )
    return GroundingResult(tree=tree, taxonomy_by_path=taxonomy_by_path)
