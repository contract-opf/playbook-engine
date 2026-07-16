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
    children: list[ClauseNode] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_path": self.clause_path,
            "heading": self.heading,
            "text": self.text,
            "char_span": list(self.char_span),
            "children": [c.to_dict() for c in self.children],
        }

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
            children=[ClauseNode.from_dict(c) for c in children_raw],
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
        """
        seen: set[str] = set()
        # Duplicate check (invariant 1) — visit every node.
        for node in self.all_nodes():
            if node.clause_path in seen:
                raise ClauseTreeError(
                    f"Duplicate clause_path {node.clause_path!r} in tree for document {self.document_id!r}"
                )
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
        """Recursive structural validator (invariants 2–5)."""
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
