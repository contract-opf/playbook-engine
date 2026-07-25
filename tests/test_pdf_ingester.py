"""Tests for the PDF ingester.

SECURITY NOTE: All PDF fixtures are created programmatically at test runtime
using fpdf2.  No real agreement PDFs are ever committed to the repository or
referenced from tests.  Party names in headings use fictitious identifiers
("Alpha Corp", "Beta Ltd", "Party A", "Party B") only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# fpdf2 is a dev-only fixture dependency — import lazily so CI doesn't fail
# when only pdfplumber is installed without fpdf2.
try:
    from fpdf import FPDF  # type: ignore[import-untyped]

    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False

from playbook_engine.pdf_ingester import (
    NullOCRAdapter,
    OCRAdapter,
    PdfIngesterError,
    _para_level,
    _split_paragraphs,
    ingest_pdf,
)

pytestmark = pytest.mark.skipif(
    not _FPDF_AVAILABLE, reason="fpdf2 not installed; run: pip install fpdf2>=2.7"
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_pdf(*, paragraphs: list[str], tmp_path: Path, name: str = "doc.pdf") -> Path:
    """Create a simple born-digital PDF with fpdf2; one paragraph per page-line."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for para in paragraphs:
        pdf.multi_cell(0, 8, para)
        pdf.ln(2)  # blank line separator between paragraphs
    dest = tmp_path / name
    pdf.output(str(dest))
    return dest


def _make_image_pdf(tmp_path: Path, name: str = "image.pdf") -> Path:
    """Create a PDF that has NO text layer (simulated by writing an empty page)."""
    pdf = FPDF()
    pdf.add_page()
    # No text — empty page simulates a scanned image with no embedded text.
    dest = tmp_path / name
    pdf.output(str(dest))
    return dest


# ---------------------------------------------------------------------------
# Unit: _split_paragraphs
# ---------------------------------------------------------------------------


def test_split_paragraphs_double_newline() -> None:
    text = "Para one.\n\nPara two."
    result = _split_paragraphs(text)
    assert result == ["Para one.", "Para two."]


def test_split_paragraphs_single_newline_splits_lines() -> None:
    text = "Line one\nline two\n\nPara two."
    result = _split_paragraphs(text)
    # Each non-empty line is its own entry
    assert "Line one" in result
    assert "line two" in result
    assert "Para two." in result


def test_split_paragraphs_ignores_blank_blocks() -> None:
    text = "\n\n\nActual text.\n\n\n"
    result = _split_paragraphs(text)
    assert result == ["Actual text."]


def test_split_paragraphs_empty_string() -> None:
    assert _split_paragraphs("") == []


# ---------------------------------------------------------------------------
# Unit: _para_level
# ---------------------------------------------------------------------------


def test_para_level_numbered_depth_1() -> None:
    assert _para_level("1. Introduction") == 1


def test_para_level_numbered_depth_2() -> None:
    assert _para_level("1.2. Representations") == 2


def test_para_level_numbered_depth_3() -> None:
    assert _para_level("1.2.3) Warranties") == 3


def test_para_level_allcaps_short_heading() -> None:
    assert _para_level("DEFINITIONS") == 1


def test_para_level_allcaps_heading_multi_word() -> None:
    assert _para_level("GENERAL TERMS AND CONDITIONS") == 1


def test_para_level_allcaps_too_many_words() -> None:
    # 9 words — over the threshold
    long_heading = "THIS IS A VERY LONG ALL CAPS LINE INDEED"
    assert _para_level(long_heading) is None


def test_para_level_body_text() -> None:
    assert _para_level("This is a normal sentence of body text.") is None


def test_para_level_allcaps_with_period_is_body() -> None:
    # Sentences end with "." — treated as body, not heading.
    assert _para_level("ALL CAPS SENTENCE.") is None


def test_para_level_mixed_case_not_heading() -> None:
    assert _para_level("Mixed Case Heading Without Number") is None


def test_para_level_numbered_paren() -> None:
    assert _para_level("2) Scope") == 1


# ---------------------------------------------------------------------------
# Born-digital PDF — text-layer extraction
# ---------------------------------------------------------------------------


def test_ingest_born_digital_returns_text_layer_method(tmp_path: Path) -> None:
    pdf_path = _make_pdf(
        paragraphs=["1. Definitions\n\nThis agreement defines the terms."], tmp_path=tmp_path
    )
    result = ingest_pdf(pdf_path, "doc-001", "v1")
    assert result.method == "text-layer"


def test_ingest_born_digital_confidence_is_1(tmp_path: Path) -> None:
    pdf_path = _make_pdf(paragraphs=["1. Definitions\n\nBody text here."], tmp_path=tmp_path)
    result = ingest_pdf(pdf_path, "doc-001", "v1")
    assert result.confidence == 1.0


def test_ingest_born_digital_produces_clause_tree(tmp_path: Path) -> None:
    pdf_path = _make_pdf(
        paragraphs=["1. Definitions", "The terms used herein are defined below.", "2. Obligations"],
        tmp_path=tmp_path,
    )
    result = ingest_pdf(pdf_path, "doc-001", "v1")
    assert result.tree.document_id == "doc-001"
    assert result.tree.version == "v1"
    assert result.tree.source_file == "doc.pdf"


def test_ingest_born_digital_numbered_headings_detected(tmp_path: Path) -> None:
    pdf_path = _make_pdf(
        paragraphs=[
            "1. Definitions",
            "Body text under definitions.",
            "2. Representations",
            "Party A represents that all statements are true.",
        ],
        tmp_path=tmp_path,
    )
    result = ingest_pdf(pdf_path, "doc-002", "v1")
    paths = [n.clause_path for n in result.tree.nodes]
    assert "1" in paths or any(p.startswith("1") for p in paths)
    assert "2" in paths or any(p.startswith("2") for p in paths)


def test_ingest_two_level_nesting(tmp_path: Path) -> None:
    """1.1 should be nested under 1 in the clause tree."""
    pdf_path = _make_pdf(
        paragraphs=[
            "1. General Provisions",
            "1.1. Definitions",
            "All definitions apply throughout.",
            "1.2. Interpretation",
            "Headings are for convenience only.",
        ],
        tmp_path=tmp_path,
    )
    result = ingest_pdf(pdf_path, "doc-003", "v1")
    tree = result.tree
    n1 = tree.resolve_path("1")
    # "1" is in the tree (may be a parent with children)
    assert n1 is not None
    n11 = tree.resolve_path("1.1")
    assert n11 is not None


def test_ingest_allcaps_heading_detected(tmp_path: Path) -> None:
    pdf_path = _make_pdf(
        paragraphs=[
            "DEFINITIONS",
            "The following definitions apply.",
            "REPRESENTATIONS",
            "Party B makes the following representations.",
        ],
        tmp_path=tmp_path,
    )
    result = ingest_pdf(pdf_path, "doc-004", "v1")
    # Should produce headings, not all body nodes
    heading_nodes = [n for n in result.tree.all_nodes() if n.heading is not None]
    assert len(heading_nodes) >= 1


def test_ingest_source_file_recorded(tmp_path: Path) -> None:
    pdf_path = _make_pdf(
        paragraphs=["1. Introduction", "Text."], tmp_path=tmp_path, name="agreement_v2.pdf"
    )
    result = ingest_pdf(pdf_path, "doc-005", "v2")
    assert result.tree.source_file == "agreement_v2.pdf"


def test_ingest_version_recorded(tmp_path: Path) -> None:
    pdf_path = _make_pdf(paragraphs=["1. Terms", "Body."], tmp_path=tmp_path)
    result = ingest_pdf(pdf_path, "doc-006", "draft-3")
    assert result.tree.version == "draft-3"


# ---------------------------------------------------------------------------
# OCR fallback
# ---------------------------------------------------------------------------


class _FakeOCRAdapter:
    """Test double that returns known text with a fixed confidence."""

    def __init__(self, text: str, confidence: float = 0.85) -> None:
        self._text = text
        self._confidence = confidence

    def extract_text(self, pdf_path: Path) -> tuple[str, float]:
        return (self._text, self._confidence)


def test_null_ocr_returns_empty_text(tmp_path: Path) -> None:
    adapter = NullOCRAdapter()
    result = adapter.extract_text(tmp_path / "ghost.pdf")
    assert result == ("", 0.0)


def test_null_ocr_is_ocr_adapter() -> None:
    """NullOCRAdapter satisfies the OCRAdapter protocol."""
    adapter = NullOCRAdapter()
    assert isinstance(adapter, OCRAdapter)


def test_fake_ocr_is_ocr_adapter() -> None:
    adapter = _FakeOCRAdapter("Some OCR text")
    assert isinstance(adapter, OCRAdapter)


def test_image_pdf_falls_back_to_ocr(tmp_path: Path) -> None:
    """A PDF with no embedded text must trigger the OCR adapter."""
    pdf_path = _make_image_pdf(tmp_path)
    ocr_text = (
        "1. Definitions\n\nAll definitions apply.\n\n2. Obligations\n\nParty A shall deliver."
    )
    adapter = _FakeOCRAdapter(ocr_text, confidence=0.90)
    result = ingest_pdf(pdf_path, "scan-001", "v1", ocr_adapter=adapter)
    assert result.method == "ocr"
    assert result.confidence == 0.90


def test_image_pdf_ocr_produces_clause_tree(tmp_path: Path) -> None:
    pdf_path = _make_image_pdf(tmp_path)
    ocr_text = "1. Definitions\n\nTerms defined here.\n\n2. Obligations\n\nParty A shall deliver."
    adapter = _FakeOCRAdapter(ocr_text)
    result = ingest_pdf(pdf_path, "scan-002", "v1", ocr_adapter=adapter)
    tree = result.tree
    assert tree.resolve_path("1") is not None or any(
        n.clause_path.startswith("1") for n in tree.all_nodes()
    )


def test_no_ocr_adapter_defaults_to_null(tmp_path: Path) -> None:
    """When ocr_adapter=None and text layer is empty, result uses OCR with confidence=0."""
    pdf_path = _make_image_pdf(tmp_path)
    result = ingest_pdf(pdf_path, "scan-003", "v1")
    assert result.method == "ocr"
    assert result.confidence == 0.0


def test_ocr_adapter_called_with_correct_path(tmp_path: Path) -> None:
    pdf_path = _make_image_pdf(tmp_path, name="scan.pdf")
    called_with: list[Path] = []

    class _TrackingOCR:
        def extract_text(self, p: Path) -> tuple[str, float]:
            called_with.append(p)
            return ("1. Intro\n\nText.", 0.75)

    ingest_pdf(pdf_path, "x", "v1", ocr_adapter=_TrackingOCR())
    assert len(called_with) == 1
    assert called_with[0] == pdf_path


def test_ocr_confidence_propagated(tmp_path: Path) -> None:
    pdf_path = _make_image_pdf(tmp_path)
    adapter = _FakeOCRAdapter("1. Terms\n\nBody text.", confidence=0.42)
    result = ingest_pdf(pdf_path, "scan-004", "v1", ocr_adapter=adapter)
    assert result.confidence == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_pdf_raises(tmp_path: Path) -> None:
    with pytest.raises(PdfIngesterError, match="not found"):
        ingest_pdf(tmp_path / "ghost.pdf", "doc-x", "v1")


def test_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(PdfIngesterError, match="not found"):
        ingest_pdf(tmp_path, "doc-x", "v1")


# ---------------------------------------------------------------------------
# char_span contract
# ---------------------------------------------------------------------------


def test_char_span_tuple_on_nodes(tmp_path: Path) -> None:
    """Every ClauseNode has a two-element non-negative char_span."""
    pdf_path = _make_pdf(
        paragraphs=["1. Definitions", "Body text.", "2. Representations"],
        tmp_path=tmp_path,
    )
    result = ingest_pdf(pdf_path, "doc-cs", "v1")
    for node in result.tree.all_nodes():
        assert isinstance(node.char_span, tuple)
        assert len(node.char_span) == 2
        assert node.char_span[0] >= 0
        assert node.char_span[1] >= node.char_span[0]


def test_char_span_start_lt_end(tmp_path: Path) -> None:
    pdf_path = _make_pdf(paragraphs=["1. Intro", "Some body text follows."], tmp_path=tmp_path)
    result = ingest_pdf(pdf_path, "doc-cs2", "v1")
    for node in result.tree.all_nodes():
        # heading nodes with no body may have start == end; that is valid
        assert node.char_span[1] >= node.char_span[0]


# ---------------------------------------------------------------------------
# Empty / minimal PDF
# ---------------------------------------------------------------------------


def test_empty_pdf_produces_empty_tree(tmp_path: Path) -> None:
    """An empty PDF (no text) with NullOCRAdapter should yield an empty tree."""
    pdf_path = _make_image_pdf(tmp_path)
    result = ingest_pdf(pdf_path, "empty-001", "v1")
    # NullOCRAdapter returns "" so no nodes should be produced
    assert result.tree.nodes == []


def test_single_paragraph_pdf(tmp_path: Path) -> None:
    pdf_path = _make_pdf(paragraphs=["1. The only clause in this document."], tmp_path=tmp_path)
    result = ingest_pdf(pdf_path, "single-001", "v1")
    assert len(list(result.tree.all_nodes())) >= 1


# ---------------------------------------------------------------------------
# PdfIngestResult fields
# ---------------------------------------------------------------------------


def test_result_tree_is_clause_tree(tmp_path: Path) -> None:
    from playbook_engine.clause_tree import ClauseTree

    pdf_path = _make_pdf(paragraphs=["1. Definitions", "Text."], tmp_path=tmp_path)
    result = ingest_pdf(pdf_path, "doc-t", "v1")
    assert isinstance(result.tree, ClauseTree)


def test_result_method_is_string(tmp_path: Path) -> None:
    pdf_path = _make_pdf(paragraphs=["1. Definitions", "Text."], tmp_path=tmp_path)
    result = ingest_pdf(pdf_path, "doc-m", "v1")
    assert isinstance(result.method, str)
    assert result.method in ("text-layer", "ocr")


def test_result_confidence_in_range(tmp_path: Path) -> None:
    pdf_path = _make_pdf(paragraphs=["1. Definitions", "Text."], tmp_path=tmp_path)
    result = ingest_pdf(pdf_path, "doc-c", "v1")
    assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Additional edge-case tests (NB3 from review)
# ---------------------------------------------------------------------------


def test_corrupt_pdf_raises_ingester_error(tmp_path: Path) -> None:
    """A binary garbage file must raise PdfIngesterError, not a raw pdfplumber exception."""
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4\x00\xff\xfe garbage bytes not valid pdf content")
    with pytest.raises(PdfIngesterError):
        ingest_pdf(corrupt, "corrupt-001", "v1")


def test_validate_passes_on_ingested_tree(tmp_path: Path) -> None:
    """The clause tree produced by ingest_pdf must pass ClauseTree.validate()."""
    pdf_path = _make_pdf(
        paragraphs=[
            "1. General Terms",
            "These terms govern the agreement.",
            "1.1. Definitions",
            "Defined terms appear herein.",
            "1.2. Scope",
            "Party A and Party B are subject to this agreement.",
            "2. Obligations",
            "Party A shall deliver the service.",
        ],
        tmp_path=tmp_path,
    )
    result = ingest_pdf(pdf_path, "validate-001", "v1")
    result.tree.validate()  # must not raise


def test_multi_page_pdf_text_concatenated(tmp_path: Path) -> None:
    """Text from multiple pages is concatenated into a single clause tree."""
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for i in range(1, 4):
        pdf.add_page()
        pdf.cell(0, 8, f"{i}. Section {i}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Body text for section {i}.", new_x="LMARGIN", new_y="NEXT")
    dest = tmp_path / "multi_page.pdf"
    pdf.output(str(dest))

    result = ingest_pdf(dest, "mp-001", "v1")
    assert result.method == "text-layer"
    # All three numbered sections should be in the tree
    all_paths = {n.clause_path for n in result.tree.all_nodes()}
    assert "1" in all_paths
    assert "2" in all_paths
    assert "3" in all_paths


def test_char_span_resolves_to_heading_text(tmp_path: Path) -> None:
    """resolve_span on the virtual normalized text returns the heading line.

    char_span covers only the heading line (same convention as DOCX ingester).
    """
    from playbook_engine.pdf_ingester import _split_paragraphs

    paragraphs = ["1. Definitions", "Body text follows."]
    pdf_path = _make_pdf(paragraphs=paragraphs, tmp_path=tmp_path)
    result = ingest_pdf(pdf_path, "span-001", "v1")

    # The virtual normalized text is _split_paragraphs applied to raw text.
    import pdfplumber as _pdfplumber

    with _pdfplumber.open(str(pdf_path)) as _pdf:
        raw_pages = [p.extract_text() for p in _pdf.pages if p.extract_text()]
    raw_text = "\n".join(raw_pages)
    normalized = "\n".join(_split_paragraphs(raw_text))

    node = result.tree.resolve_path("1")
    assert node is not None
    from playbook_engine.clause_tree import ClauseTree

    heading_line = ClauseTree.resolve_span(normalized, node.char_span)
    assert "Definitions" in heading_line or "1." in heading_line
