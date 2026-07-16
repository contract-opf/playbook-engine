"""Tests for the normalized clause-tree data model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from playbook_engine.clause_tree import ClauseNode, ClauseTree, ClauseTreeError

CLAUSE_TREE_SCHEMA_PATH = Path(__file__).parent.parent / "spec" / "clause-tree.schema.json"

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _leaf(path: str, text: str, *, start: int = 0, heading: str | None = None) -> ClauseNode:
    return ClauseNode(
        clause_path=path,
        heading=heading,
        text=text,
        char_span=(start, start + len(text)),
    )


def _simple_tree() -> ClauseTree:
    """
    1  Definitions (10 chars)
      1.1  Facility (8 chars)
      1.2  Program (7 chars)
    2  Obligations (11 chars)
    """
    n11 = _leaf("1.1", "Facility", start=11, heading="Facility")
    n12 = _leaf("1.2", "Program", start=19, heading="Program")
    n1 = ClauseNode(
        clause_path="1",
        heading="Definitions",
        text="Definitions",
        char_span=(0, 11),
        children=[n11, n12],
    )
    n2 = _leaf("2", "Obligations", start=26)
    return ClauseTree(
        document_id="test-doc",
        version="v1",
        source_file="test-doc-v1.docx",
        nodes=[n1, n2],
    )


# ---------------------------------------------------------------------------
# ClauseNode basics
# ---------------------------------------------------------------------------


def test_leaf_node_is_leaf() -> None:
    node = _leaf("1.1", "Some text")
    assert node.is_leaf()


def test_parent_node_not_leaf() -> None:
    tree = _simple_tree()
    n1 = tree.nodes[0]
    assert not n1.is_leaf()


def test_char_span_stored_correctly() -> None:
    node = _leaf("3.2.1", "Hello", start=42)
    assert node.char_span == (42, 47)


def test_heading_optional() -> None:
    node = _leaf("1", "text")
    assert node.heading is None


# ---------------------------------------------------------------------------
# ClauseTree navigation
# ---------------------------------------------------------------------------


def test_iter_leaves_returns_only_leaves() -> None:
    tree = _simple_tree()
    leaves = list(tree.iter_leaves())
    paths = [n.clause_path for n in leaves]
    assert paths == ["1.1", "1.2", "2"]


def test_iter_leaves_on_flat_tree() -> None:
    tree = ClauseTree(
        document_id="flat",
        version="v1",
        source_file="flat.docx",
        nodes=[_leaf("1", "A"), _leaf("2", "B"), _leaf("3", "C")],
    )
    assert len(list(tree.iter_leaves())) == 3


def test_resolve_path_finds_top_level() -> None:
    tree = _simple_tree()
    node = tree.resolve_path("2")
    assert node is not None
    assert node.text == "Obligations"


def test_resolve_path_finds_nested() -> None:
    tree = _simple_tree()
    node = tree.resolve_path("1.1")
    assert node is not None
    assert node.text == "Facility"
    assert node.heading == "Facility"


def test_resolve_path_missing_returns_none() -> None:
    tree = _simple_tree()
    assert tree.resolve_path("99") is None
    assert tree.resolve_path("1.3") is None


def test_resolve_path_exact_match_only() -> None:
    """Path "1" must not match "1.1" or "10"."""
    tree = _simple_tree()
    n = tree.resolve_path("1")
    assert n is not None
    assert n.clause_path == "1"


def test_all_nodes_count() -> None:
    tree = _simple_tree()
    all_nodes = list(tree.all_nodes())
    # 1, 1.1, 1.2, 2 = 4 nodes
    assert len(all_nodes) == 4


def test_all_nodes_includes_parent_and_children() -> None:
    tree = _simple_tree()
    paths = {n.clause_path for n in tree.all_nodes()}
    assert paths == {"1", "1.1", "1.2", "2"}


# ---------------------------------------------------------------------------
# Citation resolution (resolve_path + text)
# ---------------------------------------------------------------------------


def test_citation_resolution_returns_correct_text() -> None:
    tree = _simple_tree()
    node = tree.resolve_path("1.2")
    assert node is not None
    assert node.text == "Program"
    assert node.char_span == (19, 26)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_round_trip_simple() -> None:
    original = _simple_tree()
    restored = ClauseTree.from_json(original.to_json())

    assert restored.document_id == original.document_id
    assert restored.version == original.version
    assert restored.source_file == original.source_file
    assert len(restored.nodes) == len(original.nodes)


def test_round_trip_preserves_all_fields() -> None:
    original = _simple_tree()
    restored = ClauseTree.from_json(original.to_json())

    n11_orig = original.resolve_path("1.1")
    n11_rest = restored.resolve_path("1.1")
    assert n11_orig is not None and n11_rest is not None
    assert n11_rest.clause_path == n11_orig.clause_path
    assert n11_rest.heading == n11_orig.heading
    assert n11_rest.text == n11_orig.text
    assert n11_rest.char_span == n11_orig.char_span


def test_round_trip_nested_children() -> None:
    original = _simple_tree()
    restored = ClauseTree.from_json(original.to_json())
    assert len(restored.nodes[0].children) == 2


def test_to_dict_char_span_is_list() -> None:
    """char_span must serialize as a JSON array, not a Python tuple literal."""
    node = _leaf("1", "test")
    d = node.to_dict()
    assert isinstance(d["char_span"], list)
    # Verify it round-trips through JSON
    json.dumps(d)  # must not raise


def test_round_trip_empty_tree() -> None:
    tree = ClauseTree(document_id="empty", version="v0", source_file="empty.docx")
    restored = ClauseTree.from_json(tree.to_json())
    assert restored.document_id == "empty"
    assert restored.nodes == []


# ---------------------------------------------------------------------------
# write() / load()
# ---------------------------------------------------------------------------


def test_write_and_load(tmp_path: Path) -> None:
    tree = _simple_tree()
    dest = tmp_path / "normalized" / "test-doc" / "v1.clauses.json"
    tree.write(dest)
    assert dest.exists()
    loaded = ClauseTree.load(dest)
    assert loaded.document_id == tree.document_id
    assert loaded.resolve_path("1.2") is not None


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    tree = _simple_tree()
    dest = tmp_path / "a" / "b" / "c" / "tree.clauses.json"
    tree.write(dest)
    assert dest.is_file()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ClauseTreeError, match="not found"):
        ClauseTree.load(tmp_path / "ghost.clauses.json")


# ---------------------------------------------------------------------------
# Error cases — from_dict / from_json
# ---------------------------------------------------------------------------


def test_missing_document_id_raises() -> None:
    data: dict[str, Any] = {"version": "v1", "source_file": "f.docx", "nodes": []}
    with pytest.raises(ClauseTreeError, match="document_id"):
        ClauseTree.from_dict(data)


def test_missing_version_raises() -> None:
    data: dict[str, Any] = {"document_id": "d", "source_file": "f.docx", "nodes": []}
    with pytest.raises(ClauseTreeError, match="version"):
        ClauseTree.from_dict(data)


def test_invalid_json_raises() -> None:
    with pytest.raises(ClauseTreeError, match="Not valid JSON"):
        ClauseTree.from_json("{ not json }")


def test_non_object_root_raises() -> None:
    with pytest.raises(ClauseTreeError, match="JSON object"):
        ClauseTree.from_json("[1, 2, 3]")


def test_node_missing_clause_path_raises() -> None:
    data: dict[str, Any] = {
        "document_id": "d",
        "version": "v1",
        "source_file": "f",
        "nodes": [{"text": "hi", "char_span": [0, 2]}],
    }
    with pytest.raises(ClauseTreeError, match="clause_path"):
        ClauseTree.from_dict(data)


def test_node_missing_char_span_raises() -> None:
    data: dict[str, Any] = {
        "document_id": "d",
        "version": "v1",
        "source_file": "f",
        "nodes": [{"clause_path": "1", "text": "hi"}],
    }
    with pytest.raises(ClauseTreeError, match="char_span"):
        ClauseTree.from_dict(data)


def test_node_invalid_char_span_raises() -> None:
    data: dict[str, Any] = {
        "document_id": "d",
        "version": "v1",
        "source_file": "f",
        "nodes": [{"clause_path": "1", "text": "hi", "char_span": [5, 2]}],
    }
    with pytest.raises(ClauseTreeError, match="invalid"):
        ClauseTree.from_dict(data)


def test_node_negative_char_span_raises() -> None:
    data: dict[str, Any] = {
        "document_id": "d",
        "version": "v1",
        "source_file": "f",
        "nodes": [{"clause_path": "1", "text": "hi", "char_span": [-1, 2]}],
    }
    with pytest.raises(ClauseTreeError, match="invalid"):
        ClauseTree.from_dict(data)


def test_node_wrong_span_type_raises() -> None:
    data: dict[str, Any] = {
        "document_id": "d",
        "version": "v1",
        "source_file": "f",
        "nodes": [{"clause_path": "1", "text": "hi", "char_span": "0:5"}],
    }
    with pytest.raises(ClauseTreeError, match="char_span"):
        ClauseTree.from_dict(data)


def test_node_null_heading_preserved() -> None:
    data: dict[str, Any] = {
        "document_id": "d",
        "version": "v1",
        "source_file": "f",
        "nodes": [{"clause_path": "1", "text": "body", "char_span": [0, 4], "heading": None}],
    }
    tree = ClauseTree.from_dict(data)
    assert tree.nodes[0].heading is None


def test_deeply_nested_round_trip() -> None:
    """Three levels deep round-trips without data loss."""
    n111 = _leaf("1.1.1", "Deep leaf", start=30)
    n11 = ClauseNode(
        clause_path="1.1",
        heading="Sub",
        text="Sub section",
        char_span=(10, 21),
        children=[n111],
    )
    n1 = ClauseNode(
        clause_path="1",
        heading="Top",
        text="Top section",
        char_span=(0, 10),
        children=[n11],
    )
    tree = ClauseTree(document_id="deep", version="v1", source_file="d.docx", nodes=[n1])
    restored = ClauseTree.from_json(tree.to_json())
    deep = restored.resolve_path("1.1.1")
    assert deep is not None
    assert deep.text == "Deep leaf"


# ---------------------------------------------------------------------------
# NB-1 — resolve_span: char_span → substring of full document text
# ---------------------------------------------------------------------------


def test_resolve_span_basic() -> None:
    full_text = "Definitions Facility Program Obligations"
    #            0         1         2         3
    #            0123456789012345678901234567890123456789
    span = (12, 20)  # "Facility"
    assert ClauseTree.resolve_span(full_text, span) == "Facility"


def test_resolve_span_whole_document() -> None:
    full_text = "Hello World"
    assert ClauseTree.resolve_span(full_text, (0, len(full_text))) == full_text


def test_resolve_span_empty_span() -> None:
    assert ClauseTree.resolve_span("Hello", (3, 3)) == ""


def test_resolve_span_non_ascii() -> None:
    full_text = "Définitions générales"
    node_text = "générales"
    start = full_text.index(node_text)
    span = (start, start + len(node_text))
    assert ClauseTree.resolve_span(full_text, span) == node_text


# ---------------------------------------------------------------------------
# NB-2 — type validation for string fields
# ---------------------------------------------------------------------------


def test_node_null_text_raises() -> None:
    data: dict[str, Any] = {
        "document_id": "d",
        "version": "v1",
        "source_file": "f",
        "nodes": [{"clause_path": "1", "text": None, "char_span": [0, 0]}],
    }
    with pytest.raises(ClauseTreeError, match="'text' must be a string"):
        ClauseTree.from_dict(data)


def test_node_numeric_text_raises() -> None:
    data: dict[str, Any] = {
        "document_id": "d",
        "version": "v1",
        "source_file": "f",
        "nodes": [{"clause_path": "1", "text": 42, "char_span": [0, 2]}],
    }
    with pytest.raises(ClauseTreeError, match="'text' must be a string"):
        ClauseTree.from_dict(data)


def test_null_document_id_raises() -> None:
    data: dict[str, Any] = {"document_id": None, "version": "v1", "source_file": "f", "nodes": []}
    with pytest.raises(ClauseTreeError, match="'document_id' must be a string"):
        ClauseTree.from_dict(data)


# ---------------------------------------------------------------------------
# NB-3 — validate() detects duplicate clause_paths
# ---------------------------------------------------------------------------


def test_validate_duplicate_path_raises() -> None:
    tree = ClauseTree(
        document_id="d",
        version="v1",
        source_file="f",
        nodes=[_leaf("1", "A"), _leaf("1", "B")],
    )
    with pytest.raises(ClauseTreeError, match="Duplicate clause_path"):
        tree.validate()


def test_validate_duplicate_in_nested_raises() -> None:
    n_child = _leaf("1", "child")  # same path as parent
    n_parent = ClauseNode(
        clause_path="1", heading=None, text="parent", char_span=(0, 6), children=[n_child]
    )
    tree = ClauseTree(document_id="d", version="v1", source_file="f", nodes=[n_parent])
    with pytest.raises(ClauseTreeError, match="Duplicate"):
        tree.validate()


def test_validate_clean_tree_passes() -> None:
    tree = _simple_tree()
    tree.validate()  # must not raise


# ---------------------------------------------------------------------------
# NB-6 — non-ASCII round-trip
# ---------------------------------------------------------------------------


def test_non_ascii_round_trip() -> None:
    tree = ClauseTree(
        document_id="éducation",
        version="v1",
        source_file="übervertrag.docx",
        nodes=[_leaf("1", "Définitions et dispositions générales")],
    )
    restored = ClauseTree.from_json(tree.to_json())
    assert restored.document_id == "éducation"
    assert restored.nodes[0].text == "Définitions et dispositions générales"


def test_load_directory_raises(tmp_path: Path) -> None:
    """load() on a directory path must raise, not silently return garbage."""
    with pytest.raises(ClauseTreeError, match="not found"):
        ClauseTree.load(tmp_path)  # tmp_path is a directory


# ---------------------------------------------------------------------------
# NB-4 — JSON Schema conformance
# ---------------------------------------------------------------------------


def test_simple_tree_validates_against_schema() -> None:
    schema = json.loads(CLAUSE_TREE_SCHEMA_PATH.read_text(encoding="utf-8"))
    tree_dict = _simple_tree().to_dict()
    jsonschema.validate(instance=tree_dict, schema=schema)  # must not raise


def test_schema_rejects_missing_clause_path() -> None:
    schema = json.loads(CLAUSE_TREE_SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = {
        "document_id": "d",
        "version": "v1",
        "source_file": "f",
        "nodes": [{"text": "body", "char_span": [0, 4]}],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_schema_rejects_negative_span() -> None:
    schema = json.loads(CLAUSE_TREE_SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = {
        "document_id": "d",
        "version": "v1",
        "source_file": "f",
        "nodes": [{"clause_path": "1", "text": "body", "char_span": [-1, 4]}],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


# ---------------------------------------------------------------------------
# P1.3 — resolve_span bounds-checking
# ---------------------------------------------------------------------------


def test_resolve_span_raises_on_end_past_text() -> None:
    """resolve_span must raise, not silently return a clipped substring."""
    with pytest.raises(ClauseTreeError, match="out of bounds"):
        ClauseTree.resolve_span("Hello", (0, 10))


def test_resolve_span_raises_on_negative_start() -> None:
    with pytest.raises(ClauseTreeError, match="out of bounds"):
        ClauseTree.resolve_span("Hello", (-1, 3))


def test_resolve_span_raises_on_inverted_span() -> None:
    with pytest.raises(ClauseTreeError, match="out of bounds"):
        ClauseTree.resolve_span("Hello", (4, 2))


def test_resolve_span_exact_end_is_valid() -> None:
    """end == len(full_text) is a valid span (Python exclusive-end convention)."""
    assert ClauseTree.resolve_span("Hello", (0, 5)) == "Hello"


def test_resolve_span_empty_at_end_boundary_is_valid() -> None:
    """An empty span exactly at the end of text is valid (start == end == len)."""
    assert ClauseTree.resolve_span("Hello", (5, 5)) == ""


# ---------------------------------------------------------------------------
# P1.3 — validate() invariant 2: span within full_text bounds
# ---------------------------------------------------------------------------


def test_validate_raises_on_span_beyond_text() -> None:
    """A node whose span exceeds len(full_text) must fail validate(full_text=...)."""
    tree = ClauseTree(
        document_id="d",
        version="v1",
        source_file="f",
        nodes=[_leaf("1", "Hello", start=0)],  # span=(0, 5)
    )
    with pytest.raises(ClauseTreeError, match="out of bounds"):
        tree.validate(full_text="Hi")  # len=2, span end=5 > 2


def test_validate_with_full_text_passes_valid_tree() -> None:
    """A properly-bounded tree must pass validate(full_text=...)."""
    full_text = "Definitions Facility Program Obligations"
    # Rebuild with spans matching full_text
    node = _leaf("1", full_text, start=0)
    bounded = ClauseTree(document_id="d", version="v1", source_file="f", nodes=[node])
    bounded.validate(full_text=full_text)  # must not raise


def test_validate_without_full_text_skips_bounds_check() -> None:
    """validate() with no full_text skips the text-bounds check (other invariants still apply)."""
    # span=(0, 9999) would fail if bounds-checked against a short text, but here no text given
    node = _leaf("1", "short", start=0)
    node.char_span = (0, 9999)  # force an absurd span without full_text check
    tree = ClauseTree(document_id="d", version="v1", source_file="f", nodes=[node])
    tree.validate()  # must not raise (no full_text supplied)


# ---------------------------------------------------------------------------
# P1.3 — validate() invariant 3: sibling span order
# ---------------------------------------------------------------------------


def test_validate_raises_on_out_of_order_siblings() -> None:
    """Siblings whose spans are not in non-decreasing order must fail validate()."""
    # Sibling B starts before sibling A → order violation
    node_a = _leaf("1", "AAAA", start=10)
    node_b = _leaf("2", "BBBB", start=0)  # starts before A
    tree = ClauseTree(document_id="d", version="v1", source_file="f", nodes=[node_a, node_b])
    with pytest.raises(ClauseTreeError, match="sibling order"):
        tree.validate()


def test_validate_equal_start_siblings_pass() -> None:
    """Two siblings sharing the same span start are allowed (non-decreasing)."""
    node_a = _leaf("1", "A", start=5)
    node_b = _leaf("2", "B", start=5)
    tree = ClauseTree(document_id="d", version="v1", source_file="f", nodes=[node_a, node_b])
    tree.validate()  # must not raise


# ---------------------------------------------------------------------------
# P1.3 — validate() invariant 4: child starts at or after parent
# ---------------------------------------------------------------------------


def test_validate_raises_when_child_starts_before_parent() -> None:
    """A child whose span starts before its parent's span must fail validate()."""
    child = _leaf("1.1", "Child", start=0)  # starts at 0
    parent = ClauseNode(
        clause_path="1",
        heading=None,
        text="Parent",
        char_span=(10, 16),  # starts at 10, after child
        children=[child],
    )
    tree = ClauseTree(document_id="d", version="v1", source_file="f", nodes=[parent])
    with pytest.raises(ClauseTreeError, match="before parent"):
        tree.validate()


def test_validate_child_starting_at_parent_start_passes() -> None:
    """A child whose span starts exactly at its parent's start is valid."""
    child = _leaf("1.1", "Child", start=5)
    parent = ClauseNode(
        clause_path="1",
        heading=None,
        text="Parent",
        char_span=(5, 11),
        children=[child],
    )
    tree = ClauseTree(document_id="d", version="v1", source_file="f", nodes=[parent])
    tree.validate()  # must not raise


def test_validate_child_after_parent_end_passes() -> None:
    """A child that starts after the parent's end is allowed (heading-only parent model)."""
    child = _leaf("1.1", "Child", start=100)
    parent = ClauseNode(
        clause_path="1",
        heading=None,
        text="Parent",
        char_span=(0, 6),
        children=[child],
    )
    tree = ClauseTree(document_id="d", version="v1", source_file="f", nodes=[parent])
    tree.validate()  # must not raise


# ---------------------------------------------------------------------------
# P1.3 — validate() invariant 5: clause_path prefix consistency
# ---------------------------------------------------------------------------


def test_validate_raises_on_path_prefix_mismatch() -> None:
    """A child whose clause_path doesn't begin with parent.clause_path+'.' must fail."""
    child = _leaf("2.1", "Orphan child", start=20)  # wrong prefix (should be "1.1")
    parent = ClauseNode(
        clause_path="1",
        heading=None,
        text="Parent",
        char_span=(0, 6),
        children=[child],
    )
    tree = ClauseTree(document_id="d", version="v1", source_file="f", nodes=[parent])
    with pytest.raises(ClauseTreeError, match="does not begin with"):
        tree.validate()


def test_validate_clean_tree_all_invariants_pass() -> None:
    """_simple_tree() must pass validate() with all new invariants."""
    tree = _simple_tree()
    tree.validate()  # must not raise
