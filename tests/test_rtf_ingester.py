"""Tests for the RTF ingester.

SECURITY NOTE: All RTF fixtures are built as Python string literals at test
runtime — no real agreement RTF files are ever committed to the repository or
referenced from tests.  Party names in headings and body text use fictitious
identifiers ("Alpha Corp", "Beta Ltd", "Party A", "Party B") only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from playbook_engine.rtf_ingester import (
    RtfIngesterError,
    RtfIngestResult,
    _para_level,
    _split_lines,
    ingest_rtf,
)

# ---------------------------------------------------------------------------
# RTF fixture helpers
# ---------------------------------------------------------------------------


def _rtf(body: str) -> str:
    """Wrap RTF body in minimal valid RTF prologue/epilogue."""
    return (
        r"{\rtf1\ansi\deff0"
        r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
        r"\f0\fs24 " + body + r"}"
    )


def _write_rtf(content: str, tmp_path: Path, name: str = "doc.rtf") -> Path:
    dest = tmp_path / name
    dest.write_text(content, encoding="utf-8")
    return dest


def _simple_rtf(tmp_path: Path) -> Path:
    """Single-level numbered clauses with body text."""
    body = (
        r"1. Definitions\par "
        r"The following definitions apply.\par "
        r"2. Obligations\par "
        r"Party A shall deliver the goods.\par "
    )
    return _write_rtf(_rtf(body), tmp_path)


def _nested_rtf(tmp_path: Path) -> Path:
    """Two-level numbered clauses."""
    body = (
        r"1. General Terms\par "
        r"These terms govern the agreement.\par "
        r"1.1. Definitions\par "
        r"Defined terms appear herein.\par "
        r"1.2. Scope\par "
        r"Party B is subject to this agreement.\par "
        r"2. Obligations\par "
        r"Party A shall deliver the service.\par "
    )
    return _write_rtf(_rtf(body), tmp_path)


def _allcaps_rtf(tmp_path: Path) -> Path:
    """ALL-CAPS headings (no numbering)."""
    body = (
        r"DEFINITIONS\par "
        r"All terms defined herein.\par "
        r"REPRESENTATIONS\par "
        r"Alpha Corp makes the following representations.\par "
    )
    return _write_rtf(_rtf(body), tmp_path)


def _preamble_rtf(tmp_path: Path) -> Path:
    """Body text before first heading — should become clause_path '0'."""
    body = (
        r"This agreement is entered into by the parties.\par "
        r"1. Definitions\par "
        r"Terms defined herein.\par "
    )
    return _write_rtf(_rtf(body), tmp_path)


# ---------------------------------------------------------------------------
# Unit: _split_lines
# ---------------------------------------------------------------------------


def test_split_lines_removes_blank_lines() -> None:
    text = "Line one\n\nLine two\n\nLine three"
    result = _split_lines(text)
    assert result == ["Line one", "Line two", "Line three"]


def test_split_lines_strips_whitespace() -> None:
    text = "  1. Definitions  \n  Body text.  "
    result = _split_lines(text)
    assert result == ["1. Definitions", "Body text."]


def test_split_lines_empty_string() -> None:
    assert _split_lines("") == []


def test_split_lines_only_blank_lines() -> None:
    assert _split_lines("\n\n\n") == []


# ---------------------------------------------------------------------------
# Unit: _para_level
# ---------------------------------------------------------------------------


def test_para_level_depth_1() -> None:
    assert _para_level("1. Introduction") == 1


def test_para_level_depth_2() -> None:
    assert _para_level("1.2. Representations") == 2


def test_para_level_depth_3() -> None:
    assert _para_level("2.1.3) Warranties") == 3


def test_para_level_allcaps_heading() -> None:
    assert _para_level("DEFINITIONS") == 1


def test_para_level_body_text() -> None:
    assert _para_level("This is body text.") is None


def test_para_level_allcaps_sentence_is_body() -> None:
    assert _para_level("ALL CAPS SENTENCE.") is None


# ---------------------------------------------------------------------------
# ingest_rtf: basic functionality
# ---------------------------------------------------------------------------


def test_ingest_returns_rtf_ingest_result(tmp_path: Path) -> None:
    pdf_path = _simple_rtf(tmp_path)
    result = ingest_rtf(pdf_path, "doc-001", "v1")
    assert isinstance(result, RtfIngestResult)


def test_ingest_document_id_recorded(tmp_path: Path) -> None:
    result = ingest_rtf(_simple_rtf(tmp_path), "my-doc", "v1")
    assert result.tree.document_id == "my-doc"


def test_ingest_version_recorded(tmp_path: Path) -> None:
    result = ingest_rtf(_simple_rtf(tmp_path), "doc", "draft-2")
    assert result.tree.version == "draft-2"


def test_ingest_source_file_recorded(tmp_path: Path) -> None:
    path = _write_rtf(_rtf(r"1. Intro\par Body.\par "), tmp_path, name="contract_v3.rtf")
    result = ingest_rtf(path, "doc", "v1")
    assert result.tree.source_file == "contract_v3.rtf"


def test_ingest_numbered_headings_top_level(tmp_path: Path) -> None:
    result = ingest_rtf(_simple_rtf(tmp_path), "doc", "v1")
    paths = {n.clause_path for n in result.tree.all_nodes()}
    assert "1" in paths
    assert "2" in paths


def test_ingest_nested_headings(tmp_path: Path) -> None:
    result = ingest_rtf(_nested_rtf(tmp_path), "doc", "v1")
    tree = result.tree
    assert tree.resolve_path("1") is not None
    assert tree.resolve_path("1.1") is not None
    assert tree.resolve_path("1.2") is not None
    assert tree.resolve_path("2") is not None


def test_ingest_child_nested_under_parent(tmp_path: Path) -> None:
    result = ingest_rtf(_nested_rtf(tmp_path), "doc", "v1")
    parent = result.tree.resolve_path("1")
    assert parent is not None
    child_paths = [c.clause_path for c in parent.children]
    assert "1.1" in child_paths
    assert "1.2" in child_paths


def test_ingest_allcaps_heading(tmp_path: Path) -> None:
    result = ingest_rtf(_allcaps_rtf(tmp_path), "doc", "v1")
    heading_nodes = [n for n in result.tree.all_nodes() if n.heading is not None]
    assert len(heading_nodes) >= 1


def test_ingest_body_appended_to_clause(tmp_path: Path) -> None:
    result = ingest_rtf(_simple_rtf(tmp_path), "doc", "v1")
    n1 = result.tree.resolve_path("1")
    assert n1 is not None
    assert "definitions" in n1.text.lower() or "following" in n1.text.lower()


def test_ingest_pre_heading_body_collected(tmp_path: Path) -> None:
    result = ingest_rtf(_preamble_rtf(tmp_path), "doc", "v1")
    # Pre-heading text must be preserved in a "0" node
    node_0 = result.tree.resolve_path("0")
    assert node_0 is not None
    assert "parties" in node_0.text.lower() or "agreement" in node_0.text.lower()


def test_ingest_empty_rtf_body(tmp_path: Path) -> None:
    path = _write_rtf(_rtf(""), tmp_path)
    result = ingest_rtf(path, "empty", "v1")
    assert result.tree.nodes == []


# ---------------------------------------------------------------------------
# char_span contract
# ---------------------------------------------------------------------------


def test_char_span_is_tuple(tmp_path: Path) -> None:
    result = ingest_rtf(_simple_rtf(tmp_path), "doc", "v1")
    for node in result.tree.all_nodes():
        assert isinstance(node.char_span, tuple)
        assert len(node.char_span) == 2


def test_char_span_non_negative(tmp_path: Path) -> None:
    result = ingest_rtf(_nested_rtf(tmp_path), "doc", "v1")
    for node in result.tree.all_nodes():
        assert node.char_span[0] >= 0
        assert node.char_span[1] >= node.char_span[0]


def test_char_span_resolves_to_heading_line(tmp_path: Path) -> None:
    """char_span should identify the heading line in the virtual normalized text."""
    path = _write_rtf(_rtf(r"1. Definitions\par Body text here.\par "), tmp_path)
    result = ingest_rtf(path, "doc", "v1")
    from striprtf.striprtf import rtf_to_text

    raw = path.read_text(encoding="utf-8", errors="replace")
    raw_text = rtf_to_text(raw, encoding="utf-8", errors="replace")
    normalized = "\n".join(_split_lines(raw_text))

    node = result.tree.resolve_path("1")
    assert node is not None
    from playbook_engine.clause_tree import ClauseTree

    heading_line = ClauseTree.resolve_span(normalized, node.char_span)
    assert "Definitions" in heading_line or "1." in heading_line


# ---------------------------------------------------------------------------
# ClauseTree validity
# ---------------------------------------------------------------------------


def test_validate_passes_on_simple_tree(tmp_path: Path) -> None:
    result = ingest_rtf(_simple_rtf(tmp_path), "doc", "v1")
    result.tree.validate()  # must not raise


def test_validate_passes_on_nested_tree(tmp_path: Path) -> None:
    result = ingest_rtf(_nested_rtf(tmp_path), "doc", "v1")
    result.tree.validate()  # must not raise


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RtfIngesterError, match="not found"):
        ingest_rtf(tmp_path / "ghost.rtf", "doc", "v1")


def test_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(RtfIngesterError, match="not found"):
        ingest_rtf(tmp_path, "doc", "v1")


def test_corrupt_rtf_content(tmp_path: Path) -> None:
    """Binary garbage should either produce an empty tree or raise RtfIngesterError."""
    corrupt = tmp_path / "corrupt.rtf"
    corrupt.write_bytes(b"\x00\xff\xfe\xfd garbage not rtf")
    try:
        result = ingest_rtf(corrupt, "corrupt", "v1")
        # If striprtf tolerates the garbage, at minimum the tree must be valid
        result.tree.validate()
    except RtfIngesterError:
        pass  # also acceptable


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_tree_round_trips_to_json(tmp_path: Path) -> None:
    result = ingest_rtf(_nested_rtf(tmp_path), "doc", "v1")
    restored = type(result.tree).from_json(result.tree.to_json())
    assert restored.document_id == result.tree.document_id
    assert restored.resolve_path("1.1") is not None


def test_tree_write_and_load(tmp_path: Path) -> None:
    result = ingest_rtf(_simple_rtf(tmp_path), "doc", "v1")
    dest = tmp_path / "output" / "doc.clauses.json"
    result.tree.write(dest)
    from playbook_engine.clause_tree import ClauseTree

    loaded = ClauseTree.load(dest)
    assert loaded.document_id == "doc"


# ---------------------------------------------------------------------------
# Schema validation (NB1 from review)
# ---------------------------------------------------------------------------


def test_tree_validates_against_json_schema(tmp_path: Path) -> None:
    """Emitted ClauseTree must validate against spec/clause-tree.schema.json."""
    import json

    import jsonschema

    schema_path = Path(__file__).parent.parent / "spec" / "clause-tree.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    result = ingest_rtf(_nested_rtf(tmp_path), "doc", "v1")
    jsonschema.validate(instance=result.tree.to_dict(), schema=schema)  # must not raise


# ---------------------------------------------------------------------------
# Encoding robustness (B2 regression guard)
# ---------------------------------------------------------------------------


def test_cp1252_bytes_preserved(tmp_path: Path) -> None:
    """Raw cp1252 bytes in RTF (e.g. 0x92 right-single-quote) must not be mangled.

    This guards against regressing to the utf-8 read that was replaced with
    cp1252 after the Opus code review.
    """
    # Build RTF bytes with a raw cp1252 right-single-quote (0x92) in the text.
    rtf_bytes = (
        b"{\\rtf1\\ansi\\deff0"
        b"{\\fonttbl{\\f0\\froman\\fcharset0 Times New Roman;}}"
        b"\\f0\\fs24 "
        b"1. Party A\x92s duty\\par "  # 0x92 = cp1252 right single quote
        b"Body text here.\\par "
        b"}"
    )
    path = tmp_path / "cp1252.rtf"
    path.write_bytes(rtf_bytes)
    result = ingest_rtf(path, "cp1252-doc", "v1")
    # The right-single-quote should survive as some visible character, not U+FFFD
    n1 = result.tree.resolve_path("1")
    assert n1 is not None
    assert "�" not in (n1.heading or "")  # should not be a replacement char
