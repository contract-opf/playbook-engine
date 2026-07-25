"""Tests for the DOCX ingester.

SECURITY NOTE: All fixtures MUST be synthetic and programmatically generated
using python-docx + lxml XML injection.  No binary .docx files from real
agreements are ever committed or referenced from tests — not even redlined
drafts.  The real corpus lives outside this repo and must never become a
fixture source.  All tracked-change fixtures use fictional party names and
authors (e.g. "Alice", "Bob") only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document
from lxml import etree

from playbook_engine.clause_tree import ClauseTree
from playbook_engine.docx_ingester import (
    DocxIngesterError,
    DocxIngestResult,
    TrackedChanges,
    ingest_docx,
)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def _simple_docx(tmp_path: Path) -> Path:
    """Simple agreement: Heading 1 and 2 structure with body paragraphs."""
    doc = Document()
    doc.add_heading("Definitions", level=1)
    doc.add_paragraph('"Facility" means any FixtureCorp-operated training centre.')
    doc.add_heading("Training Services", level=2)
    doc.add_paragraph("FixtureCorp shall provide training per Schedule A.")
    doc.add_heading("Obligations", level=1)
    doc.add_paragraph("Each party shall perform its obligations diligently.")
    path = tmp_path / "simple.docx"
    doc.save(str(path))
    return path


def _numbered_docx(tmp_path: Path) -> Path:
    """Numbered sections using explicit numbers in paragraph text."""
    doc = Document()
    doc.add_paragraph("1. Definitions")
    doc.add_paragraph('"Facility" means the training centre.')
    doc.add_paragraph("1.1. Facility")
    doc.add_paragraph("The term Facility includes all FixtureCorp sites.")
    doc.add_paragraph("1.2. Program")
    doc.add_paragraph("Program means the accredited curriculum.")
    doc.add_paragraph("2. Obligations")
    doc.add_paragraph("Both parties shall cooperate fully.")
    path = tmp_path / "numbered.docx"
    doc.save(str(path))
    return path


def _table_docx(tmp_path: Path) -> Path:
    """Document with a table — table must be flattened to body text."""
    doc = Document()
    doc.add_heading("Fee Schedule", level=1)
    tbl = doc.add_table(rows=2, cols=2)
    tbl.rows[0].cells[0].text = "Service"
    tbl.rows[0].cells[1].text = "Fee"
    tbl.rows[1].cells[0].text = "Training"
    tbl.rows[1].cells[1].text = "$100/hour"
    path = tmp_path / "table.docx"
    doc.save(str(path))
    return path


def _tracked_docx(tmp_path: Path) -> Path:
    """Document with tracked changes injected via raw XML."""
    doc = Document()

    # Heading
    doc.add_heading("Obligations", level=1)

    # Paragraph with tracked insertion and deletion
    p = doc.add_paragraph()

    # Normal run
    p.add_run("Party A shall ")

    # Tracked insertion: "promptly " inserted by Alice
    ins_elem = etree.SubElement(p._p, _w("ins"))
    ins_elem.set(_w("id"), "1")
    ins_elem.set(_w("author"), "Alice")
    ins_elem.set(_w("date"), "2024-03-15T10:00:00Z")
    r_ins = etree.SubElement(ins_elem, _w("r"))
    t_ins = etree.SubElement(r_ins, _w("t"))
    t_ins.text = "promptly "

    # Normal run continued
    p.add_run("provide services")

    # Tracked deletion: "to client" deleted by Bob
    del_elem = etree.SubElement(p._p, _w("del"))
    del_elem.set(_w("id"), "2")
    del_elem.set(_w("author"), "Bob")
    del_elem.set(_w("date"), "2024-03-16T09:00:00Z")
    r_del = etree.SubElement(del_elem, _w("r"))
    dt_del = etree.SubElement(r_del, _w("delText"))
    dt_del.text = " to client"

    path = tmp_path / "tracked.docx"
    doc.save(str(path))
    return path


def _hyperlink_tracked_docx(tmp_path: Path) -> Path:
    """Document where a tracked change is nested inside a w:hyperlink element.

    Structure: Heading 1 "Obligations", then a body paragraph containing
    a hyperlink that wraps a w:ins element (Alice inserts a URL reference).
    This exercises BLOCKING-2: nested ins/del inside w:hyperlink.
    """
    doc = Document()
    doc.add_heading("Obligations", level=1)
    p = doc.add_paragraph()
    p.add_run("See ")

    # Build: <w:hyperlink><w:ins author="Alice">...<w:r><w:t>Schedule A</w:t></w:r></w:ins></w:hyperlink>
    hl_elem = etree.SubElement(p._p, _w("hyperlink"))
    ins_inside_hl = etree.SubElement(hl_elem, _w("ins"))
    ins_inside_hl.set(_w("id"), "10")
    ins_inside_hl.set(_w("author"), "Alice")
    ins_inside_hl.set(_w("date"), "2024-04-01T08:00:00Z")
    r_hl = etree.SubElement(ins_inside_hl, _w("r"))
    t_hl = etree.SubElement(r_hl, _w("t"))
    t_hl.text = "Schedule A"

    p.add_run(" for details.")
    path = tmp_path / "hyperlink_tracked.docx"
    doc.save(str(path))
    return path


def _multi_para_clause_docx(tmp_path: Path) -> Path:
    """Heading with multiple body paragraphs — tests document-absolute char_span offsets."""
    doc = Document()
    doc.add_heading("Obligations", level=1)
    # Two body paragraphs with a tracked insertion in the second
    doc.add_paragraph("Party A shall perform services.")

    p2 = doc.add_paragraph()
    p2.add_run("Compensation is ")
    ins_elem = etree.SubElement(p2._p, _w("ins"))
    ins_elem.set(_w("id"), "5")
    ins_elem.set(_w("author"), "Carol")
    ins_elem.set(_w("date"), "2024-05-01T00:00:00Z")
    r_ins = etree.SubElement(ins_elem, _w("r"))
    t_ins = etree.SubElement(r_ins, _w("t"))
    t_ins.text = "as follows"
    p2.add_run(".")

    path = tmp_path / "multi_para.docx"
    doc.save(str(path))
    return path


def _empty_docx(tmp_path: Path) -> Path:
    doc = Document()
    path = tmp_path / "empty.docx"
    doc.save(str(path))
    return path


# ---------------------------------------------------------------------------
# Basic ingestion
# ---------------------------------------------------------------------------


def test_ingest_returns_docx_result(tmp_path: Path) -> None:
    path = _simple_docx(tmp_path)
    result = ingest_docx(path, "simple-doc", "v1")
    assert isinstance(result, DocxIngestResult)
    assert isinstance(result.tree, ClauseTree)
    assert isinstance(result.tracked, TrackedChanges)


def test_tree_has_correct_document_id(tmp_path: Path) -> None:
    path = _simple_docx(tmp_path)
    result = ingest_docx(path, "my-doc", "v2")
    assert result.tree.document_id == "my-doc"
    assert result.tree.version == "v2"
    assert result.tree.source_file == "simple.docx"


def test_simple_doc_produces_nodes(tmp_path: Path) -> None:
    path = _simple_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    assert len(result.tree.nodes) > 0, "Simple DOCX should produce at least one clause node"


def test_heading_hierarchy_is_respected(tmp_path: Path) -> None:
    """Heading 2 sections must be children of Heading 1."""
    path = _simple_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    tree = result.tree

    # Should have top-level clauses
    paths = [n.clause_path for n in tree.nodes]
    assert len(paths) >= 1

    # "Training Services" (Heading 2) must be nested under "Definitions" (Heading 1)
    definitions_node = next(
        (n for n in tree.nodes if n.heading and "Definitions" in n.heading), None
    )
    assert definitions_node is not None, "Definitions heading not found"
    child_headings = [c.heading for c in definitions_node.children]
    assert any("Training Services" in (h or "") for h in child_headings), (
        "Training Services (Heading 2) should be a child of Definitions (Heading 1)"
    )


def test_body_text_attached_to_heading(tmp_path: Path) -> None:
    path = _simple_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    definitions_node = next(
        (n for n in result.tree.nodes if n.heading and "Definitions" in n.heading), None
    )
    assert definitions_node is not None
    assert "Facility" in definitions_node.text, (
        "Body paragraph under Definitions should be in node text"
    )


# ---------------------------------------------------------------------------
# Numbered paragraphs
# ---------------------------------------------------------------------------


def test_numbered_prefix_becomes_clause_path(tmp_path: Path) -> None:
    path = _numbered_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    tree = result.tree

    # Should have clause "1" and "2" at the top
    top_paths = {n.clause_path for n in tree.nodes}
    assert "1" in top_paths or "2" in top_paths


def test_numbered_subsections_nested(tmp_path: Path) -> None:
    path = _numbered_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    # resolve_path should find 1.1 and 1.2
    n11 = result.tree.resolve_path("1.1")
    n12 = result.tree.resolve_path("1.2")
    assert n11 is not None, "Clause 1.1 should be in the tree"
    assert n12 is not None, "Clause 1.2 should be in the tree"


# ---------------------------------------------------------------------------
# Table flattening
# ---------------------------------------------------------------------------


def test_table_content_is_included(tmp_path: Path) -> None:
    path = _table_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    # Table content should appear somewhere in the tree
    all_text = " ".join(n.text for n in result.tree.all_nodes())
    assert "Training" in all_text or "100" in all_text, (
        "Table content should be included in the clause tree text"
    )


def test_table_does_not_create_heading_nodes(tmp_path: Path) -> None:
    path = _table_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    # Fee Schedule is the heading; the table rows should NOT become new heading nodes
    fee_schedule = result.tree.resolve_path("1") or next(
        (n for n in result.tree.nodes if n.heading and "Fee" in (n.heading or "")), None
    )
    assert fee_schedule is not None, "Fee Schedule heading not found"


# ---------------------------------------------------------------------------
# Tracked changes (acceptance criterion)
# ---------------------------------------------------------------------------


def test_tracked_changes_captured(tmp_path: Path) -> None:
    """Core acceptance criterion: tracked-change spans captured with authors."""
    path = _tracked_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    assert len(result.tracked.changes) >= 2, "Should capture at least the insertion and deletion"


def test_tracked_insertion_author_captured(tmp_path: Path) -> None:
    path = _tracked_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    insertions = [c for c in result.tracked.changes if c.change_type == "insertion"]
    assert len(insertions) >= 1
    assert any(c.author == "Alice" for c in insertions), "Alice's insertion must be captured"


def test_tracked_insertion_text_captured(tmp_path: Path) -> None:
    path = _tracked_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    insertions = [c for c in result.tracked.changes if c.change_type == "insertion"]
    insertion_texts = [c.text for c in insertions]
    assert any("promptly" in t for t in insertion_texts), "Inserted text must be captured"


def test_tracked_insertion_has_char_span(tmp_path: Path) -> None:
    path = _tracked_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    insertions = [c for c in result.tracked.changes if c.change_type == "insertion"]
    assert all(c.char_span is not None for c in insertions), "Insertions must have a char_span"
    for c in insertions:
        assert c.char_span is not None
        start, end = c.char_span
        assert end >= start >= 0


def test_tracked_deletion_author_captured(tmp_path: Path) -> None:
    path = _tracked_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    deletions = [c for c in result.tracked.changes if c.change_type == "deletion"]
    assert len(deletions) >= 1
    assert any(c.author == "Bob" for c in deletions), "Bob's deletion must be captured"


def test_tracked_deletion_text_captured(tmp_path: Path) -> None:
    path = _tracked_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    deletions = [c for c in result.tracked.changes if c.change_type == "deletion"]
    assert any("client" in c.text for c in deletions), "Deleted text must be captured"


def test_tracked_deletion_char_span_is_none(tmp_path: Path) -> None:
    """Deletions are absent from normalized text — char_span must be None."""
    path = _tracked_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    deletions = [c for c in result.tracked.changes if c.change_type == "deletion"]
    assert all(c.char_span is None for c in deletions)


def test_tracked_change_clause_path_populated(tmp_path: Path) -> None:
    path = _tracked_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    for change in result.tracked.changes:
        assert change.clause_path != "", "clause_path must be set on every TrackedChange"


def test_inserted_text_in_normalized_text(tmp_path: Path) -> None:
    """Inserted text must appear in the clause's normalized text."""
    path = _tracked_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    all_text = " ".join(n.text for n in result.tree.all_nodes())
    assert "promptly" in all_text, "Inserted text should be part of normalized clause text"


def test_deleted_text_absent_from_normalized(tmp_path: Path) -> None:
    """Deleted text must NOT appear in the normalized clause text."""
    path = _tracked_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    all_text = " ".join(n.text for n in result.tree.all_nodes())
    assert "to client" not in all_text, "Deleted text must not appear in normalized text"


def test_tracked_changes_document_id(tmp_path: Path) -> None:
    path = _tracked_docx(tmp_path)
    result = ingest_docx(path, "agreement-123", "v1")
    assert result.tracked.document_id == "agreement-123"
    assert result.tracked.version == "v1"


# ---------------------------------------------------------------------------
# char_span integrity
# ---------------------------------------------------------------------------


def test_char_spans_are_non_negative(tmp_path: Path) -> None:
    path = _simple_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    for node in result.tree.all_nodes():
        start, end = node.char_span
        assert start >= 0 and end >= start, (
            f"Invalid char_span {node.char_span} on {node.clause_path!r}"
        )


def test_tree_validates_unique_paths(tmp_path: Path) -> None:
    """validate() should pass — no duplicate clause_paths."""
    path = _simple_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    result.tree.validate()  # must not raise


# ---------------------------------------------------------------------------
# Empty document
# ---------------------------------------------------------------------------


def test_empty_docx_produces_empty_tree(tmp_path: Path) -> None:
    path = _empty_docx(tmp_path)
    result = ingest_docx(path, "empty", "v1")
    # An empty document may produce 0 nodes or a single empty root
    assert result.tree is not None
    assert result.tracked.changes == []


# ---------------------------------------------------------------------------
# TrackedChanges.to_dict round-trip
# ---------------------------------------------------------------------------


def test_tracked_changes_to_dict(tmp_path: Path) -> None:
    path = _tracked_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    d = result.tracked.to_dict()
    assert d["document_id"] == "d"
    assert isinstance(d["changes"], list)
    for c in d["changes"]:
        assert "change_type" in c
        assert "author" in c
        assert "text" in c
        assert "clause_path" in c


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DocxIngesterError, match="not found"):
        ingest_docx(tmp_path / "ghost.docx", "d", "v1")


def test_non_docx_raises(tmp_path: Path) -> None:
    bad = tmp_path / "not_a_docx.docx"
    bad.write_bytes(b"this is not a zip file")
    with pytest.raises(DocxIngesterError, match="Cannot open"):
        ingest_docx(bad, "d", "v1")


# ---------------------------------------------------------------------------
# BLOCKING-2: tracked changes nested inside w:hyperlink (recursive descent)
# ---------------------------------------------------------------------------


def test_tracked_insertion_inside_hyperlink_captured(tmp_path: Path) -> None:
    """w:ins nested inside w:hyperlink must be captured as a TrackedChange."""
    path = _hyperlink_tracked_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    insertions = [c for c in result.tracked.changes if c.change_type == "insertion"]
    assert len(insertions) >= 1, "Insertion inside hyperlink should be captured"
    assert any(c.author == "Alice" for c in insertions)
    assert any("Schedule A" in c.text for c in insertions)


def test_inserted_text_inside_hyperlink_in_normalized_text(tmp_path: Path) -> None:
    """Text inserted inside a hyperlink must appear in the normalized clause text."""
    path = _hyperlink_tracked_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    all_text = " ".join(n.text for n in result.tree.all_nodes())
    assert "Schedule A" in all_text


# ---------------------------------------------------------------------------
# BLOCKING-1: char_span is document-absolute (multi-paragraph clause)
# ---------------------------------------------------------------------------


def test_char_span_is_document_absolute(tmp_path: Path) -> None:
    """TrackedChange.char_span must be document-absolute, not paragraph-local.

    In a multi-paragraph clause, an insertion in the 2nd paragraph must have
    a char_span > 0 that accounts for the preceding paragraph text.
    """
    path = _multi_para_clause_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    insertions = [c for c in result.tracked.changes if c.change_type == "insertion"]
    assert any("as follows" in c.text for c in insertions), "Carol's insertion not found"
    carol_ins = next(c for c in insertions if "as follows" in c.text)
    assert carol_ins.char_span is not None
    start, end = carol_ins.char_span
    # The insertion is in the SECOND body paragraph, so its start must be > 0
    # (it follows at least the heading text + "\n" + first body paragraph + "\n").
    assert start > 0, (
        f"char_span {carol_ins.char_span} looks paragraph-local (start=0); "
        f"expected document-absolute offset > 0 for a 2nd-paragraph insertion"
    )
    assert end > start


def test_char_span_length_matches_insertion_text(tmp_path: Path) -> None:
    """The span length (end - start) must equal the length of the inserted text."""
    path = _multi_para_clause_docx(tmp_path)
    result = ingest_docx(path, "d", "v1")
    insertions = [c for c in result.tracked.changes if c.change_type == "insertion"]
    carol_ins = next((c for c in insertions if "as follows" in c.text), None)
    assert carol_ins is not None, "Carol's insertion not found"
    assert carol_ins.char_span is not None
    start, end = carol_ins.char_span
    assert end - start == len(carol_ins.text), (
        f"char_span length {end - start} != text length {len(carol_ins.text)}"
    )
