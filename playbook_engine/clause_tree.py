"""Normalized clause-tree data model.

Every parser (DOCX, PDF, RTF) emits a ClauseTree. Every downstream stage
reads ClauseTree. The wire format is a JSON file at
``normalized/<doc>/<version>.clauses.json``.

Terminology (matches docs/ARCHITECTURE.md L1 description):
  clause_path — dotted numbering ("1", "2.3", "10.1.2")
  heading      — section title if present, else None
  text         — body text of this node (not including children)
  char_span    — (start, end) character indices in the document's full
                 normalized text (exclusive end, like Python slice notation)
                 covering the WHOLE clause: from the start of its heading
                 line through the end of its own body text (children
                 excluded — each child carries its own span). This is the
                 span an OPF citation's ``char_span`` resolves to (OPF-SPEC
                 §4), so a consumer lands on the clause language, not just
                 its heading (issue #217).
  heading_span — (start, end) of the heading line alone, or None when the
                 node has no separate heading line (the synthetic pre-heading
                 ``clause_path="0"`` node, sub-clauses the segmenter promotes
                 out of body text, and LLM/agent-grounded nodes, whose
                 heading sits inside ``text``). Optional; when present,
                 ``heading_span[0] == char_span[0]`` and
                 ``heading_span[1] <= char_span[1]``.
  page         — 1-based source page the clause begins on, or None when
                 unpaginated/unknown (optional; today only the legacy PDF
                 extractor path supplies real values — see
                 segmentation_grounding.build())
  children     — ordered list of child ClauseNode objects
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ClauseNode:
    """A single clause or section in the normalized tree."""

    clause_path: str
    heading: str | None
    text: str
    char_span: tuple[int, int]
    page: int | None = None
    children: list[ClauseNode] = field(default_factory=list)
    heading_span: tuple[int, int] | None = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "clause_path": self.clause_path,
            "heading": self.heading,
            "text": self.text,
            "char_span": list(self.char_span),
            "page": self.page,
            "children": [c.to_dict() for c in self.children],
        }
        # Emitted only when present, so a node with no separate heading line
        # (synthetic "0" node, promoted sub-clauses, grounded LLM nodes)
        # serializes exactly as it did before heading_span existed.
        if self.heading_span is not None:
            d["heading_span"] = list(self.heading_span)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClauseNode:
        _require_field(data, "clause_path")
        _require_field(data, "text")
        _require_field(data, "char_span")
        span_raw = data["char_span"]
        if (
            not isinstance(span_raw, list)
            or len(span_raw) != 2
            or not all(isinstance(v, int) for v in span_raw)
        ):
            raise ClauseTreeError(
                f"clause_path {data.get('clause_path')!r}: "
                f"'char_span' must be a 2-element integer list, got {span_raw!r}"
            )
        start, end = span_raw
        if start < 0 or end < start:
            raise ClauseTreeError(
                f"clause_path {data.get('clause_path')!r}: "
                f"'char_span' [{start}, {end}] is invalid (must have 0 ≤ start ≤ end)"
            )
        heading = data.get("heading")
        if heading is not None and not isinstance(heading, str):
            raise ClauseTreeError(
                f"clause_path {data.get('clause_path')!r}: 'heading' must be a string or null"
            )
        page = data.get("page")
        if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page < 1):
            raise ClauseTreeError(
                f"clause_path {data.get('clause_path')!r}: 'page' must be a positive integer"
                f" or null, got {page!r}"
            )
        heading_span = _parse_optional_span(data, "heading_span")
        children_raw = data.get("children", [])
        if not isinstance(children_raw, list):
            raise ClauseTreeError(
                f"clause_path {data.get('clause_path')!r}: 'children' must be a list"
            )
        text = data["text"]
        if not isinstance(text, str):
            raise ClauseTreeError(
                f"clause_path {data.get('clause_path')!r}: 'text' must be a string, got {type(text).__name__}"
            )
        return cls(
            clause_path=str(data["clause_path"]),
            heading=heading,
            text=text,
            char_span=(start, end),
            page=page,
            children=[ClauseNode.from_dict(c) for c in children_raw],
            heading_span=heading_span,
        )


@dataclass
class ClauseTree:
    """Normalized clause tree for one version of one document.

    Serializes to/from ``normalized/<document_id>/<version>.clauses.json``.
    """

    document_id: str
    version: str
    source_file: str
    nodes: list[ClauseNode] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Navigation helpers
    # -----------------------------------------------------------------------

    def iter_leaves(self) -> Iterator[ClauseNode]:
        """Depth-first iteration over all leaf nodes (nodes with no children)."""
        yield from _iter_leaves(self.nodes)

    def resolve_path(self, clause_path: str) -> ClauseNode | None:
        """Return the ClauseNode whose clause_path exactly matches, or None."""
        return _find_by_path(self.nodes, clause_path)

    def all_nodes(self) -> Iterator[ClauseNode]:
        """Depth-first iteration over every node in the tree."""
        yield from _iter_all(self.nodes)

    # -----------------------------------------------------------------------
    # JSON round-trip
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "version": self.version,
            "source_file": self.source_file,
            "nodes": [n.to_dict() for n in self.nodes],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def write(self, path: Path) -> None:
        """Write to disk as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    def validate(self, full_text: str | None = None) -> None:
        """Raise ClauseTreeError if the tree violates any structural invariant.

        Invariants checked (all are always checked unless noted):

        1. No duplicate ``clause_path`` values anywhere in the tree.
        2. Every node's ``char_span`` is within ``[0, len(full_text)]`` — only
           checked when *full_text* is provided.
        3. Sibling order: within any list of siblings, spans must appear in
           non-decreasing order (``sibling[i+1].char_span[0] >=
           sibling[i].char_span[0]``).
        4. Child-after-parent: every child node's span start must be greater
           than or equal to its parent's span start.
        5. ``clause_path`` prefix consistency: every child node's path must
           begin with ``parent.clause_path + "."``.
        6. Heading-within-clause: when a node carries ``heading_span``, it
           starts where ``char_span`` starts and ends at or before
           ``char_span``'s end — ``char_span`` covers the whole clause
           (heading through own body text), ``heading_span`` only its
           heading line (issue #217).
        """
        seen: set[str] = set()
        # Duplicate check (invariant 1) — visit every node.
        for node in self.all_nodes():
            if node.clause_path in seen:
                # Deliberately omits self.document_id (issue #83): on the
                # batch-segmentation path (pipeline._ground_batch_result)
                # this tree's document_id is the real, raw document id, and
                # this message can end up persisted verbatim into
                # corpus_manifest.json/playbook.opf.json via
                # segmentation_qa.SegmentationQAError's "tree gate: ..." wrap
                # for a quarantined document — see that class's docstring.
                # The caller (mine_corpus's quarantine handler) already
                # records the document_id separately, so nothing is lost by
                # leaving it out here.
                raise ClauseTreeError(f"Duplicate clause_path {node.clause_path!r} in tree")
            seen.add(node.clause_path)

        # Structural invariants — walk the tree with parent context.
        text_len = len(full_text) if full_text is not None else None
        self._validate_nodes(self.nodes, parent=None, text_len=text_len)

    def _validate_nodes(
        self,
        nodes: list[ClauseNode],
        *,
        parent: ClauseNode | None,
        text_len: int | None,
    ) -> None:
        """Recursive structural validator (invariants 2–6)."""
        prev: ClauseNode | None = None
        for node in nodes:
            start, end = node.char_span

            # Invariant 2: span within full-text bounds.
            if text_len is not None and (start < 0 or end > text_len or start > end):
                raise ClauseTreeError(
                    f"clause_path {node.clause_path!r}: char_span [{start}, {end}] is out of"
                    f" bounds for text of length {text_len}"
                )

            # Invariant 3: sibling order (spans non-decreasing).
            if prev is not None and start < prev.char_span[0]:
                raise ClauseTreeError(
                    f"clause_path {node.clause_path!r}: span start {start} is before sibling"
                    f" {prev.clause_path!r} span start {prev.char_span[0]} — sibling order"
                    " must be non-decreasing"
                )

            # Invariant 4: child starts at or after its parent.
            if parent is not None and start < parent.char_span[0]:
                raise ClauseTreeError(
                    f"clause_path {node.clause_path!r}: span start {start} is before parent"
                    f" {parent.clause_path!r} span start {parent.char_span[0]}"
                )

            # Invariant 5: clause_path prefix consistency.
            if parent is not None:
                expected_prefix = parent.clause_path + "."
                if not node.clause_path.startswith(expected_prefix):
                    raise ClauseTreeError(
                        f"clause_path {node.clause_path!r} is a child of"
                        f" {parent.clause_path!r} but does not begin with"
                        f" {expected_prefix!r}"
                    )

            # Invariant 6: heading_span sits at the head of char_span.
            if node.heading_span is not None:
                h_start, h_end = node.heading_span
                if h_start != start or h_end > end:
                    raise ClauseTreeError(
                        f"clause_path {node.clause_path!r}: heading_span [{h_start}, {h_end}]"
                        f" must start at char_span start {start} and end at or before"
                        f" char_span end {end}"
                    )

            prev = node
            self._validate_nodes(node.children, parent=node, text_len=text_len)

    @staticmethod
    def resolve_span(full_text: str, span: tuple[int, int]) -> str:
        """Extract the substring of full_text indicated by char_span.

        ``char_span`` is exclusive-end: ``full_text[start:end]``.

        Raises :class:`ClauseTreeError` if *span* is out of bounds for
        *full_text* (start < 0, end > len(full_text), or start > end).
        """
        start, end = span
        text_len = len(full_text)
        if start < 0 or end > text_len or start > end:
            raise ClauseTreeError(
                f"char_span [{start}, {end}] is out of bounds for text of length {text_len}"
            )
        return full_text[start:end]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClauseTree:
        _require_field(data, "document_id")
        _require_field(data, "version")
        _require_field(data, "source_file")
        for key in ("document_id", "version", "source_file"):
            val = data[key]
            if not isinstance(val, str):
                raise ClauseTreeError(f"'{key}' must be a string, got {type(val).__name__}")
        nodes_raw = data.get("nodes", [])
        if not isinstance(nodes_raw, list):
            raise ClauseTreeError("'nodes' must be a list")
        return cls(
            document_id=data["document_id"],
            version=data["version"],
            source_file=data["source_file"],
            nodes=[ClauseNode.from_dict(n) for n in nodes_raw],
        )

    @classmethod
    def from_json(cls, text: str) -> ClauseTree:
        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ClauseTreeError(f"Not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ClauseTreeError(f"Root must be a JSON object, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: Path) -> ClauseTree:
        """Load from a .clauses.json file."""
        if not path.is_file():
            raise ClauseTreeError(f"Clause tree file not found: {path}")
        return cls.from_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ClauseTreeError(ValueError):
    """Raised on malformed clause tree data."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_field(data: dict[str, Any], key: str) -> None:
    if key not in data:
        raise ClauseTreeError(f"Required field '{key}' is missing")


def _parse_optional_span(data: dict[str, Any], key: str) -> tuple[int, int] | None:
    """Parse an optional ``[start, end]`` span field (absent/null → None)."""
    raw = data.get(key)
    if raw is None:
        return None
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in raw)
    ):
        raise ClauseTreeError(
            f"clause_path {data.get('clause_path')!r}: "
            f"'{key}' must be a 2-element integer list or null, got {raw!r}"
        )
    start, end = raw
    if start < 0 or end < start:
        raise ClauseTreeError(
            f"clause_path {data.get('clause_path')!r}: "
            f"'{key}' [{start}, {end}] is invalid (must have 0 ≤ start ≤ end)"
        )
    return (start, end)


def _iter_leaves(nodes: list[ClauseNode]) -> Iterator[ClauseNode]:
    for node in nodes:
        if node.is_leaf():
            yield node
        else:
            yield from _iter_leaves(node.children)


def _iter_all(nodes: list[ClauseNode]) -> Iterator[ClauseNode]:
    for node in nodes:
        yield node
        yield from _iter_all(node.children)


def _find_by_path(nodes: list[ClauseNode], target: str) -> ClauseNode | None:
    for node in nodes:
        if node.clause_path == target:
            return node
        result = _find_by_path(node.children, target)
        if result is not None:
            return result
    return None
