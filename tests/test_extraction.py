"""Tests for the block-stream extractor.

SECURITY NOTE: All fixtures are built programmatically at test runtime with
python-docx / fpdf2 / raw RTF string literals.  No real agreement files are
ever committed to the repository or referenced from tests.  Party names use
fictitious identifiers ("Alpha Corp", "Beta Ltd") only.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from docx import Document
from lxml import etree

from playbook_engine import extraction
from playbook_engine.extraction import ExtractionCache, ExtractionError, extract_blocks
from playbook_engine.segmentation_grounding import Block

# fpdf2 is a dev-only fixture dependency — import lazily so CI doesn't fail
# when only pdfplumber is installed without fpdf2 (mirrors test_pdf_ingester.py).
try:
    from fpdf import FPDF  # type: ignore[import-untyped]

    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False

_PANDOC_AVAILABLE = shutil.which("pandoc") is not None


# ---------------------------------------------------------------------------
# Shared round-trip assertion
# ---------------------------------------------------------------------------


def _assert_round_trips(canonical_text: str, blocks: list[Block]) -> None:
    """Assert the extraction invariant required by issue #71:

    - block_ids are "b0", "b1", … in reading order
    - "\\n".join(block texts) == canonical_text
    - every block.text == canonical_text[slice(*block.char_span)]
    """
    assert [b.block_id for b in blocks] == [f"b{i}" for i in range(len(blocks))]
    assert "\n".join(b.text for b in blocks) == canonical_text
    for b in blocks:
        assert b.text == canonical_text[slice(*b.char_span)]


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def _simple_docx(tmp_path: Path) -> Path:
    doc = Document()
    doc.add_heading("Indemnification", level=1)
    doc.add_paragraph("Alpha Corp shall indemnify Beta Ltd for direct damages.")
    doc.add_heading("Governing Law", level=1)
    doc.add_paragraph("This agreement is governed by the laws of New York.")
    path = tmp_path / "simple.docx"
    doc.save(str(path))
    return path


def test_extract_docx_round_trips(tmp_path: Path) -> None:
    path = _simple_docx(tmp_path)
    canonical_text, blocks, _ = extract_blocks(path)
    _assert_round_trips(canonical_text, blocks)
    assert [b.text for b in blocks] == [
        "Indemnification",
        "Alpha Corp shall indemnify Beta Ltd for direct damages.",
        "Governing Law",
        "This agreement is governed by the laws of New York.",
    ]
    # DOCX is not paginated — page is always 0.
    assert all(b.page == 0 for b in blocks)


def test_extract_docx_skips_empty_paragraphs(tmp_path: Path) -> None:
    doc = Document()
    doc.add_paragraph("First paragraph.")
    doc.add_paragraph("")  # blank paragraph — must not become a block
    doc.add_paragraph("Second paragraph.")
    path = tmp_path / "with_blanks.docx"
    doc.save(str(path))

    canonical_text, blocks, _ = extract_blocks(path)
    _assert_round_trips(canonical_text, blocks)
    assert [b.text for b in blocks] == ["First paragraph.", "Second paragraph."]


def test_extract_docx_returns_shared_block_type(tmp_path: Path) -> None:
    path = _simple_docx(tmp_path)
    _, blocks, _ = extract_blocks(path)
    for b in blocks:
        assert isinstance(b, Block)


# Word namespace, matching docx_ingester's tracked-change fixture convention
# (tests/test_docx_ingester.py) — used to inject a raw w:ins element that
# python-docx's high-level API cannot produce.
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _w(tag: str) -> str:
    return f"{{{_W_NS}}}{tag}"


def _tracked_and_table_docx(tmp_path: Path) -> Path:
    """DOCX with a tracked-change insertion (``w:ins``) and a table.

    Regression fixture for issue #85: ``paragraph.text`` only concatenates
    runs that are direct children of ``w:p`` — runs inside ``w:ins`` are
    excluded — and ``doc.paragraphs`` skips table content entirely.
    """
    doc = Document()
    doc.add_heading("Obligations", level=1)

    p = doc.add_paragraph()
    p.add_run("Party A shall ")
    ins_elem = etree.SubElement(p._p, _w("ins"))
    ins_elem.set(_w("id"), "1")
    ins_elem.set(_w("author"), "Alice")
    ins_elem.set(_w("date"), "2024-03-15T10:00:00Z")
    r_ins = etree.SubElement(ins_elem, _w("r"))
    t_ins = etree.SubElement(r_ins, _w("t"))
    t_ins.text = "promptly "
    p.add_run("provide services.")

    tbl = doc.add_table(rows=2, cols=2)
    tbl.rows[0].cells[0].text = "Service"
    tbl.rows[0].cells[1].text = "Fee"
    tbl.rows[1].cells[0].text = "Training"
    tbl.rows[1].cells[1].text = "$100/hour"

    path = tmp_path / "tracked_and_table.docx"
    doc.save(str(path))
    return path


def test_legacy_docx_captures_tracked_insertions_and_tables(tmp_path: Path) -> None:
    """Regression for issue #85.

    Without docling on PATH, the legacy DOCX fallback must not silently drop
    tracked-change insertions or table content — the very text being
    negotiated (counterparty-inserted language) and table-borne terms must
    survive into ``canonical_text``, matching what ``docx_ingester`` (the
    deterministic path) already captures.
    """
    path = _tracked_and_table_docx(tmp_path)
    canonical_text, blocks, _ = extract_blocks(path)
    _assert_round_trips(canonical_text, blocks)

    assert "Party A shall promptly provide services." in canonical_text
    assert "Service" in canonical_text
    assert "Fee" in canonical_text
    assert "Training" in canonical_text
    assert "$100/hour" in canonical_text


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

pytestmark_pdf = pytest.mark.skipif(
    not _FPDF_AVAILABLE, reason="fpdf2 not installed; run: pip install fpdf2>=2.7"
)


def _make_pdf(*, pages: list[list[str]], tmp_path: Path, name: str = "doc.pdf") -> Path:
    """Build a PDF with one or more pages, each a list of line strings."""
    pdf = FPDF()
    for lines in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        for line in lines:
            pdf.multi_cell(0, 8, line)
            pdf.ln(2)  # blank line separator — without it multi_cell exhausts cursor width
    dest = tmp_path / name
    pdf.output(str(dest))
    return dest


@pytestmark_pdf
def test_extract_pdf_round_trips(tmp_path: Path) -> None:
    path = _make_pdf(
        pages=[["Indemnification clause line one.", "Second line same page."]],
        tmp_path=tmp_path,
    )
    canonical_text, blocks, _ = extract_blocks(path)
    _assert_round_trips(canonical_text, blocks)
    assert [b.text for b in blocks] == [
        "Indemnification clause line one.",
        "Second line same page.",
    ]
    assert all(b.page == 1 for b in blocks)


@pytestmark_pdf
def test_extract_pdf_pages_are_1_based(tmp_path: Path) -> None:
    path = _make_pdf(
        pages=[["Page one line."], ["Page two line."]],
        tmp_path=tmp_path,
    )
    canonical_text, blocks, _ = extract_blocks(path)
    _assert_round_trips(canonical_text, blocks)
    assert [b.page for b in blocks] == [1, 2]


@pytestmark_pdf
def test_extract_pdf_empty_raises_extraction_error(tmp_path: Path) -> None:
    pdf = FPDF()
    pdf.add_page()  # no text at all
    path = tmp_path / "empty.pdf"
    pdf.output(str(path))

    with pytest.raises(ExtractionError, match="no text"):
        extract_blocks(path)


# ---------------------------------------------------------------------------
# RTF (via pandoc subprocess)
# ---------------------------------------------------------------------------

pytestmark_rtf = pytest.mark.skipif(
    not _PANDOC_AVAILABLE, reason="pandoc not found on PATH; install it to run RTF extraction tests"
)


def _simple_rtf(tmp_path: Path, name: str = "doc.rtf") -> Path:
    # Trailing space after each \par is required: RTF control words otherwise
    # consume the following characters as part of the control word (matches
    # the convention in tests/test_rtf_ingester.py's _simple_rtf fixture).
    body = (
        r"Indemnification\par "
        r"Alpha Corp shall indemnify Beta Ltd for direct damages.\par "
        r"Governing Law\par "
        r"This agreement is governed by the laws of New York.\par "
    )
    content = r"{\rtf1\ansi\deff0" r"\f0\fs24 " + body + r"}"
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


@pytestmark_rtf
def test_extract_rtf_round_trips(tmp_path: Path) -> None:
    path = _simple_rtf(tmp_path)
    canonical_text, blocks, _ = extract_blocks(path)
    _assert_round_trips(canonical_text, blocks)
    assert [b.text for b in blocks] == [
        "Indemnification",
        "Alpha Corp shall indemnify Beta Ltd for direct damages.",
        "Governing Law",
        "This agreement is governed by the laws of New York.",
    ]
    assert all(b.page == 0 for b in blocks)


def test_extract_rtf_missing_pandoc_raises_extraction_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When pandoc is absent from PATH, RTF extraction must fail loud."""
    path = _simple_rtf(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)

    with pytest.raises(ExtractionError, match="pandoc"):
        extract_blocks(path)


# ---------------------------------------------------------------------------
# docling (preferred path, via mocked CLI subprocess)
# ---------------------------------------------------------------------------

_CANNED_DOCLING_MARKDOWN = (
    "# Indemnification\n"
    "\n"
    "Alpha Corp shall indemnify **Beta Ltd** for direct damages.\n"
    "\n"
    "## Remedies\n"
    "\n"
    "- Injunctive relief\n"
    "- Monetary damages\n"
    "\n"
    "| Party | Cap |\n"
    "| --- | --- |\n"
    "| Alpha Corp | $1,000,000 |\n"
)


def _mock_docling_subprocess(
    monkeypatch: pytest.MonkeyPatch, markdown: str, *, stem: str = "doc"
) -> list[list[str]]:
    """Mock ``shutil.which("docling")`` and ``subprocess.run`` so
    ``extract_blocks`` takes the docling path and writes ``markdown`` to the
    ``<stem>.md`` file docling would have produced in ``--output``.

    Returns a list that records each ``subprocess.run`` command, so tests can
    assert the real docling invocation shape (subcommand, flags).
    """
    calls: list[list[str]] = []

    def fake_which(cmd: str) -> str | None:
        return "/usr/bin/docling" if cmd == "docling" else None

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        outdir = Path(cmd[cmd.index("--output") + 1])
        (outdir / f"{stem}.md").write_text(markdown, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(extraction.shutil, "which", fake_which)
    monkeypatch.setattr(extraction.subprocess, "run", fake_run)
    return calls


def test_extract_docling_round_trips_and_strips_decoration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_docling_subprocess(monkeypatch, _CANNED_DOCLING_MARKDOWN)
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 fake content")

    canonical_text, blocks, _ = extract_blocks(path)
    _assert_round_trips(canonical_text, blocks)
    assert [b.text for b in blocks] == [
        "Indemnification",
        "Alpha Corp shall indemnify Beta Ltd for direct damages.",
        "Remedies",
        "Injunctive relief",
        "Monetary damages",
        "Party | Cap",
        "Alpha Corp | $1,000,000",
    ]
    # Headings are their own blocks (not merged with surrounding text).
    assert blocks[0].text == "Indemnification"
    assert blocks[2].text == "Remedies"
    # No Markdown decoration survives in the citable text.
    for b in blocks:
        assert "#" not in b.text
        assert "*" not in b.text
        assert not b.text.startswith("-")
        assert "|" not in b.text or b.text.count("|") == 1  # table cells joined, not raw pipes
    # docling output is not paginated.
    assert all(b.page == 0 for b in blocks)


def test_extract_docling_invocation_uses_convert_and_placeholder_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """docling >=2.x requires the `convert` subcommand, images must be
    exported as placeholders (not embedded base64), and OCR must be pinned to
    English (docling's RapidOCR default is Chinese, which garbles Latin scans).
    Regression guard for all three — mocked-subprocess tests can't catch a wrong
    CLI shape any other way.
    """
    calls = _mock_docling_subprocess(monkeypatch, _CANNED_DOCLING_MARKDOWN)
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 fake content")

    extract_blocks(path)

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "docling"
    assert cmd[1] == "convert"  # subcommand is mandatory in docling >=2.x
    assert cmd[cmd.index("--to") + 1] == "md"
    assert cmd[cmd.index("--image-export-mode") + 1] == "placeholder"
    assert cmd[cmd.index("--ocr-lang") + 1] == "eng"  # not docling's Chinese default


def test_extract_docling_drops_image_and_comment_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Embedded images (`![...](data:...base64)`) and docling's HTML image
    placeholder (`<!-- image -->`) are never citable text — they must be
    dropped, not emitted as blocks (else base64 blobs poison the LLM input).
    """
    markdown = (
        "# Title\n"
        "\n"
        "![Image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUg)\n"
        "\n"
        "<!-- image -->\n"
        "\n"
        "Real clause text.\n"
    )
    _mock_docling_subprocess(monkeypatch, markdown)
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 fake content")

    canonical_text, blocks, _ = extract_blocks(path)
    _assert_round_trips(canonical_text, blocks)
    assert [b.text for b in blocks] == ["Title", "Real clause text."]
    assert "base64" not in canonical_text
    assert "<!--" not in canonical_text


def test_extract_docling_preferred_over_legacy_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a .docx path uses docling (uniformly) when it's on PATH."""
    _mock_docling_subprocess(monkeypatch, "# Title\n\nBody text.\n", stem="doc")
    path = tmp_path / "doc.docx"
    # Deliberately not a real DOCX — proves the legacy python-docx adapter
    # was never invoked (it would fail to open this file).
    path.write_bytes(b"not a real docx")

    canonical_text, blocks, _ = extract_blocks(path)
    _assert_round_trips(canonical_text, blocks)
    assert [b.text for b in blocks] == ["Title", "Body text."]


def test_extract_docling_absent_falls_back_to_legacy_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When docling is not on PATH, extraction is unchanged (regression)."""
    monkeypatch.setattr(extraction.shutil, "which", lambda _cmd: None)
    path = _simple_docx(tmp_path)

    canonical_text, blocks, _ = extract_blocks(path)
    _assert_round_trips(canonical_text, blocks)
    assert [b.text for b in blocks] == [
        "Indemnification",
        "Alpha Corp shall indemnify Beta Ltd for direct damages.",
        "Governing Law",
        "This agreement is governed by the laws of New York.",
    ]


def test_extract_blocks_reports_extractor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``extract_blocks`` surfaces which extractor ran (issue #129): this was
    previously only visible via a ``logging.info`` line suppressed by default
    Python logging config, so a host install falling back to the legacy
    adapter (no docling -> no OCR on scanned PDFs) was invisible to the
    operator. ``detect_extractor`` (the same check ``extract_blocks`` makes
    internally) must agree with what actually ran, in both directions."""
    path = _simple_docx(tmp_path)

    monkeypatch.setattr(extraction.shutil, "which", lambda _cmd: None)
    _, _, extractor = extract_blocks(path)
    assert extractor == "legacy"
    assert extraction.detect_extractor(path) == "legacy"

    _mock_docling_subprocess(monkeypatch, "# Title\n\nBody text.\n", stem=path.stem)
    _, _, extractor = extract_blocks(path)
    assert extractor == "docling"
    assert extraction.detect_extractor(path) == "docling"


# docling failure → legacy fallback (per-file, not a whole-doc skip).
#
# When docling is on PATH but fails on a specific file, ``extract_blocks``
# falls back to the legacy per-format adapter for that one file rather than
# skipping it — otherwise redline drafts (which docling 2.x's DOCX backend
# raises on, via ``etree.QName`` on comment nodes) silently drop out of the
# negotiation trail. The three failure shapes below (empty output, non-zero
# exit, timeout) each recover via the fallback; a file the legacy adapter
# *also* cannot parse still raises, preserving the skip-on-unrecoverable
# contract the caller relies on. The returned ``extractor`` label reports the
# fallback so it stays visible in reporting (issue #129).


def test_extract_docling_empty_output_falls_back_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _simple_docx(tmp_path)
    # docling "succeeds" but emits only whitespace → treated as failure.
    _mock_docling_subprocess(monkeypatch, "   \n\n  ", stem=path.stem)

    canonical_text, blocks, extractor = extract_blocks(path)

    assert extractor == "legacy"  # fell back, and honestly reports it
    _assert_round_trips(canonical_text, blocks)
    assert any("indemnify" in b.text.lower() for b in blocks)


def test_extract_docling_subprocess_failure_falls_back_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _simple_docx(tmp_path)

    def fake_which(cmd: str) -> str | None:
        return "/usr/bin/docling" if cmd == "docling" else None

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, cmd, stderr="docling: conversion failed")

    monkeypatch.setattr(extraction.shutil, "which", fake_which)
    monkeypatch.setattr(extraction.subprocess, "run", fake_run)

    canonical_text, blocks, extractor = extract_blocks(path)

    assert extractor == "legacy"
    assert any("indemnify" in b.text.lower() for b in blocks)


def test_docling_timeout_falls_back_and_enforces_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung docling subprocess must not block the run forever: the ``timeout=``
    cap is still passed (issue #98), and on ``TimeoutExpired`` the file falls
    back to the legacy adapter rather than being skipped outright.
    """
    path = _simple_docx(tmp_path)

    def fake_which(cmd: str) -> str | None:
        return "/usr/bin/docling" if cmd == "docling" else None

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("timeout"), "docling subprocess must be called with a timeout"
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(extraction.shutil, "which", fake_which)
    monkeypatch.setattr(extraction.subprocess, "run", fake_run)

    canonical_text, blocks, extractor = extract_blocks(path)

    assert extractor == "legacy"
    assert any("indemnify" in b.text.lower() for b in blocks)


def test_extract_docling_failure_unparseable_file_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback does not weaken the skip-on-unrecoverable contract: when
    docling fails *and* the legacy adapter also cannot parse the file (here a
    stub that is not a real PDF), ``extract_blocks`` still raises
    ``ExtractionError`` so the caller skips that one version.
    """

    def fake_which(cmd: str) -> str | None:
        return "/usr/bin/docling" if cmd == "docling" else None

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, cmd, stderr="docling: conversion failed")

    monkeypatch.setattr(extraction.shutil, "which", fake_which)
    monkeypatch.setattr(extraction.subprocess, "run", fake_run)

    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 fake content")  # not a real PDF; legacy fails too

    with pytest.raises(ExtractionError):
        extract_blocks(path)


# ---------------------------------------------------------------------------
# Unsupported types / general errors
# ---------------------------------------------------------------------------


def test_unsupported_extension_raises_extraction_error(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("plain text file", encoding="utf-8")

    with pytest.raises(ExtractionError, match="unsupported"):
        extract_blocks(path)


def test_missing_file_raises_extraction_error(tmp_path: Path) -> None:
    path = tmp_path / "does_not_exist.docx"

    with pytest.raises(ExtractionError, match="not found"):
        extract_blocks(path)


# ---------------------------------------------------------------------------
# ExtractionCache (issue #132) — a repeat extract_blocks() call over
# unchanged file content must skip extraction entirely.
# ---------------------------------------------------------------------------


def test_extraction_cache_second_call_skips_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second ``extract_blocks(path, cache=...)`` call for unchanged content
    never re-invokes the underlying (legacy DOCX) extraction helper."""
    path = _simple_docx(tmp_path)
    cache = ExtractionCache(tmp_path / "extraction_cache.jsonl")

    calls: list[Path] = []
    real_extract_docx_lines = extraction._extract_docx_lines

    def _counting_extract_docx_lines(p: Path) -> list[tuple[str, int]]:
        calls.append(p)
        return real_extract_docx_lines(p)

    monkeypatch.setattr(extraction, "_extract_docx_lines", _counting_extract_docx_lines)

    first = extract_blocks(path, cache=cache)
    assert len(calls) == 1, "first call must extract (cache miss)"

    second = extract_blocks(path, cache=cache)
    assert len(calls) == 1, "second call over unchanged content must hit the cache"

    assert second == first


def test_extraction_cache_persists_across_instances(tmp_path: Path) -> None:
    """A fresh ``ExtractionCache`` pointed at the same file on disk still hits
    (load-on-init, same contract as ``VerdictStore``/``SegmentationVerdictCache``)."""
    path = _simple_docx(tmp_path)
    cache_path = tmp_path / "extraction_cache.jsonl"

    canonical_text, blocks, extractor = extract_blocks(path, cache=ExtractionCache(cache_path))

    # New instance, same on-disk file — must load the entry written above.
    reloaded = ExtractionCache(cache_path)
    cached = reloaded.get(path)
    assert cached is not None
    cached_text, cached_blocks, cached_extractor = cached
    assert cached_text == canonical_text
    assert cached_extractor == extractor
    assert [b.text for b in cached_blocks] == [b.text for b in blocks]
    assert [b.char_span for b in cached_blocks] == [b.char_span for b in blocks]


def test_extraction_cache_miss_for_changed_content(tmp_path: Path) -> None:
    """Editing the file's content after a cache hit is recorded busts the cache
    (key is the file's content hash, not its path)."""
    path = tmp_path / "doc.docx"
    doc = Document()
    doc.add_paragraph("Original text.")
    doc.save(str(path))

    cache = ExtractionCache(tmp_path / "extraction_cache.jsonl")
    first_text, _, _ = extract_blocks(path, cache=cache)
    assert cache.get(path) is not None

    doc2 = Document()
    doc2.add_paragraph("Changed text.")
    doc2.save(str(path))

    assert cache.get(path) is None, "changed content must not replay the old entry"
    second_text, _, _ = extract_blocks(path, cache=cache)
    assert second_text != first_text
