"""Tests for the block-stream extractor.

SECURITY NOTE: All fixtures are built programmatically at test runtime with
python-docx / fpdf2 / raw RTF string literals.  No real agreement files are
ever committed to the repository or referenced from tests.  Party names use
fictitious identifiers ("Alpha Corp", "Beta Ltd") only.
"""

from __future__ import annotations

import json
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


def test_extraction_cache_persists_across_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh ``ExtractionCache`` pointed at the same file on disk still hits
    (load-on-init, same contract as ``VerdictStore``/``SegmentationVerdictCache``)
    — for the SAME extractor environment. The cache key is now content hash
    PLUS extractor environment (issue #77), so this pins ``detect_extractor``
    rather than relying on whatever happens to be on the host's PATH, to test
    same-environment persistence specifically (a cross-environment miss is
    covered separately by ``test_extraction_cache_success_retried_under_better_extractor``)."""
    path = _simple_docx(tmp_path)
    cache_path = tmp_path / "extraction_cache.jsonl"
    monkeypatch.setattr(extraction, "detect_extractor", lambda p: "legacy")

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


def test_extraction_cache_put_omits_per_block_text(tmp_path: Path) -> None:
    """``put()`` must not persist per-block ``"text"`` (issue #67): it is fully
    determined by ``canonical_text[char_span]``, so storing it duplicated every
    document's text in the cache entry. ``get()`` must still round-trip the
    original block text, reconstructed from canonical_text/char_span."""
    path = _simple_docx(tmp_path)
    cache_path = tmp_path / "extraction_cache.jsonl"
    cache = ExtractionCache(cache_path)

    canonical_text, blocks, extractor = extract_blocks(path, cache=cache)

    record = json.loads(cache_path.read_text().splitlines()[0])
    stored_blocks = record["verdict"]["blocks"]
    assert stored_blocks, "fixture must produce at least one block"
    for b in stored_blocks:
        assert "text" not in b, "per-block text must not be serialized (issue #67)"

    cached = cache.get(path)
    assert cached is not None
    cached_text, cached_blocks, cached_extractor = cached
    assert cached_text == canonical_text
    assert cached_extractor == extractor
    assert [b.text for b in cached_blocks] == [b.text for b in blocks]
    assert [b.char_span for b in cached_blocks] == [b.char_span for b in blocks]


def test_extraction_cache_get_honors_stored_text_for_pre_fix_entries(tmp_path: Path) -> None:
    """A cache entry written by the pre-#67 ``put()`` (per-block ``"text"``
    present) must still load via the ``b.get("text")`` fallback.

    The stored ``"text"`` is deliberately made to differ from what
    ``canonical_text[char_span]`` would produce, so this test fails if a
    future change drops the fallback and always reconstructs from the span
    instead of preferring an already-stored value.
    """
    path = _simple_docx(tmp_path)
    cache_path = tmp_path / "extraction_cache.jsonl"
    cache = ExtractionCache(cache_path)

    canonical_text = "Alpha Corp shall indemnify Beta Ltd for direct damages."
    legacy_value = {
        "canonical_text": canonical_text,
        "blocks": [
            {
                "block_id": "b0",
                "page": 0,
                "char_span": [0, len(canonical_text)],
                "text": "STORED LEGACY TEXT",  # deliberately != canonical_text[0:len]
            }
        ],
        "extractor": "legacy",
    }
    cache._store.put(extraction._extraction_cache_payload(path), legacy_value)

    cached = cache.get(path)
    assert cached is not None
    _, cached_blocks, _ = cached
    assert cached_blocks[0].text == "STORED LEGACY TEXT"


# ---------------------------------------------------------------------------
# Negative caching of extraction failures (pre-derivation QA): a scanned PDF
# that yields no text must not be re-OCR'd on every pipeline command — but a
# better extractor environment (legacy -> docling) must retry it.
# ---------------------------------------------------------------------------


def test_extraction_failure_is_negative_cached(tmp_path: Path, monkeypatch) -> None:
    from playbook_engine import extraction as ext

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake scanned pdf")
    cache = ext.ExtractionCache(tmp_path / "cache.jsonl")

    calls = {"n": 0}

    def _no_text(path, suffix):
        calls["n"] += 1
        return []

    monkeypatch.setattr(ext, "_extract_legacy_lines", _no_text)
    monkeypatch.setattr(ext, "detect_extractor", lambda p: "legacy")

    with pytest.raises(ext.ExtractionError):
        ext.extract_blocks(pdf, cache=cache)
    assert calls["n"] == 1

    # Second call: served from the negative cache, extractor never invoked.
    with pytest.raises(ext.ExtractionError, match="cached failure"):
        ext.extract_blocks(pdf, cache=cache)
    assert calls["n"] == 1


def test_extraction_failure_retried_under_better_extractor(tmp_path: Path, monkeypatch) -> None:
    from playbook_engine import extraction as ext

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake scanned pdf")
    cache = ext.ExtractionCache(tmp_path / "cache.jsonl")

    monkeypatch.setattr(ext, "_extract_legacy_lines", lambda p, s: [])
    monkeypatch.setattr(ext, "detect_extractor", lambda p: "legacy")
    with pytest.raises(ext.ExtractionError):
        ext.extract_blocks(pdf, cache=cache)

    # docling appears on PATH: the legacy-era failure must NOT block a retry.
    monkeypatch.setattr(ext, "detect_extractor", lambda p: "docling")
    monkeypatch.setattr(ext, "_extract_docling_lines", lambda p: [("OCR recovered text.", 0)])
    canonical, blocks, extractor = ext.extract_blocks(pdf, cache=cache)
    assert "OCR recovered text." in canonical
    assert extractor == "docling"

    # And the success overwrites the failure marker.
    hit = cache.get(pdf)
    assert hit is not None and hit[2] == "docling"


def test_extraction_cache_success_retried_under_better_extractor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirror of ``test_extraction_failure_retried_under_better_extractor`` for
    the SUCCESS case (issue #77): ``get_failure()`` was already scoped to the
    current extractor environment, but ``get()`` was not — a legacy-era
    SUCCESS was strictly stickier than a legacy-era failure. A born-digital
    file that legacy extracted badly-but-non-emptily (no OCR, garbled
    columns) must not be frozen at that bad extraction once docling becomes
    available: it must be re-extracted, and the docling result must win.

    This must fail against pre-fix code: the pre-fix cache key is content
    hash only, so the second ``extract_blocks`` call below would hit the
    legacy-era entry and never invoke docling at all.
    """
    from playbook_engine import extraction as ext

    path = _simple_docx(tmp_path)
    cache = ext.ExtractionCache(tmp_path / "cache.jsonl")

    monkeypatch.setattr(ext, "detect_extractor", lambda p: "legacy")
    legacy_canonical, _, legacy_extractor = ext.extract_blocks(path, cache=cache)
    assert legacy_extractor == "legacy"

    # docling appears on PATH: the legacy-era SUCCESS must NOT be replayed.
    monkeypatch.setattr(ext, "detect_extractor", lambda p: "docling")
    docling_calls = {"n": 0}

    def _fake_docling_lines(p: Path) -> list[tuple[str, int]]:
        docling_calls["n"] += 1
        return [("Docling recovered text.", 0)]

    monkeypatch.setattr(ext, "_extract_docling_lines", _fake_docling_lines)

    docling_canonical, _, docling_extractor = ext.extract_blocks(path, cache=cache)
    assert docling_calls["n"] == 1, "legacy-era success must not short-circuit docling extraction"
    assert docling_extractor == "docling"
    assert docling_canonical != legacy_canonical
    assert "Docling recovered text." in docling_canonical

    # And the docling success is what a same-environment lookup now returns.
    hit = cache.get(path)
    assert hit is not None and hit[2] == "docling"

    # The stale legacy-era entry is still there but no longer reachable: a
    # same-environment (legacy) lookup would still hit it too, unchanged.
    monkeypatch.setattr(ext, "detect_extractor", lambda p: "legacy")
    legacy_hit = cache.get(path)
    assert legacy_hit is not None and legacy_hit[2] == "legacy"
    assert legacy_hit[0] == legacy_canonical


def test_docling_timeout_failure_negative_caches_under_docling_env(
    tmp_path: Path, monkeypatch
) -> None:
    """A docling-environment failure must be cached AS a docling failure.

    The docling→legacy fallback relabels the attempt "legacy" before the
    no-text raise; storing that label would make every docling-environment
    lookup miss and re-burn the full OCR timeout each pipeline round.
    """
    from playbook_engine import extraction as ext

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake scanned pdf")
    cache = ext.ExtractionCache(tmp_path / "cache.jsonl")

    docling_calls = {"n": 0}

    def _docling_times_out(path):
        docling_calls["n"] += 1
        raise ext.ExtractionError("docling timed out after 600s")

    monkeypatch.setattr(ext, "detect_extractor", lambda p: "docling")
    monkeypatch.setattr(ext, "_extract_docling_lines", _docling_times_out)
    monkeypatch.setattr(ext, "_extract_legacy_lines", lambda p, s: [])

    with pytest.raises(ext.ExtractionError):
        ext.extract_blocks(pdf, cache=cache)
    assert docling_calls["n"] == 1

    # Second call in the SAME docling environment: fail fast from the cache.
    with pytest.raises(ext.ExtractionError, match="cached failure"):
        ext.extract_blocks(pdf, cache=cache)
    assert docling_calls["n"] == 1


# ---------------------------------------------------------------------------
# extract_blocks(refresh=True) (issue #78) — the primitive an operator-
# invoked --no-cache relies on to force a real recompute of a suspect
# extraction: cache READS are bypassed but WRITES still happen, so the
# refreshed entry serves the next plain (non-refresh) call.
# ---------------------------------------------------------------------------


def test_extract_blocks_refresh_bypasses_read_but_still_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``refresh=True`` must skip the cache read (always re-extract from
    source) while still refreshing the cache entry — "reads bypassed, writes
    still happen". Before this parameter existed, ``extract_blocks`` had no
    way at all to bypass a cache hit, so an operator's ``--no-cache`` could
    not force re-extraction of a suspect cached result.
    """
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
    assert len(calls) == 1, "plain repeat call must hit the cache, not re-extract"
    assert second == first

    third = extract_blocks(path, cache=cache, refresh=True)
    assert len(calls) == 2, "refresh=True must bypass the cache read and re-extract"
    assert third == first  # unchanged source content -> identical result

    fourth = extract_blocks(path, cache=cache)
    assert len(calls) == 2, (
        "the refresh must have written a fresh entry — the next plain call must hit it"
    )
    assert fourth == first


def test_extract_blocks_refresh_ignores_cached_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``refresh=True`` must also bypass a cached FAILURE (``get_failure``),
    not just a cached success — an operator forcing ``--no-cache`` to retry a
    suspect extraction must not be blocked by its own prior negative-cache
    entry in the SAME extractor environment.
    """
    from playbook_engine import extraction as ext

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake scanned pdf")
    cache = ext.ExtractionCache(tmp_path / "cache.jsonl")

    monkeypatch.setattr(ext, "detect_extractor", lambda p: "legacy")
    monkeypatch.setattr(ext, "_extract_legacy_lines", lambda p, s: [])

    with pytest.raises(ext.ExtractionError):
        ext.extract_blocks(pdf, cache=cache)

    # Plain retry (no refresh): served from the negative cache, fails fast.
    with pytest.raises(ext.ExtractionError, match="cached failure"):
        ext.extract_blocks(pdf, cache=cache)

    # refresh=True, same environment: the operator forces a real retry — this
    # time extraction succeeds. The cached failure must not block it.
    monkeypatch.setattr(ext, "_extract_legacy_lines", lambda p, s: [("Recovered text.", 0)])
    canonical, _, extractor = ext.extract_blocks(pdf, cache=cache, refresh=True)
    assert "Recovered text." in canonical
    assert extractor == "legacy"

    # The success overwrites the failure marker for future plain calls.
    hit = cache.get(pdf)
    assert hit is not None and hit[2] == "legacy"


def test_extract_blocks_refresh_failure_does_not_clobber_cached_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``refresh=True`` call that transiently yields no text must NOT
    overwrite a pre-existing cached SUCCESS for the same key (issue #78
    round 2).

    ``_extraction_cache_payload`` keys ``put`` and ``put_failure`` identically
    (path content hash + format version + extractor environment — no
    dependency on ``refresh``), so before this guard, a refresh attempt that
    transiently yielded no lines (the documented docling-OCR-timeout mode)
    would negative-cache directly over a previously-good entry. Every later
    PLAIN call would then raise "cached failure" forever, even once the
    extractor is healthy again — recovery was hand-deleting
    extraction_cache.jsonl by hand, exactly the state issue #78 exists to
    eliminate, and newly reachable only via the very flag meant to fix it.
    """
    from playbook_engine import extraction as ext

    path = _simple_docx(tmp_path)
    cache = ext.ExtractionCache(tmp_path / "extraction_cache.jsonl")

    monkeypatch.setattr(ext, "detect_extractor", lambda p: "legacy")

    # Cold run: succeeds and caches the real extracted content.
    good = ext.extract_blocks(path, cache=cache)
    assert good[2] == "legacy"

    # Refresh run: extraction transiently yields nothing (e.g. an OCR
    # timeout) even though the source content is unchanged and genuinely
    # extractable (proven by the successful cold run above under the
    # IDENTICAL cache key — same bytes, same extractor environment).
    monkeypatch.setattr(ext, "_extract_legacy_lines", lambda p, s: [])
    with pytest.raises(ext.ExtractionError, match="yielded no text"):
        ext.extract_blocks(path, cache=cache, refresh=True)

    # The prior success must survive untouched...
    hit = cache.get(path)
    assert hit is not None, "a transient refresh failure must not clobber the cached success"
    assert hit == good

    # ...and a subsequent PLAIN call must replay it, not raise a cached
    # failure (the extractor being "healthy again" in the wording above is
    # irrelevant here — the cache read never re-invokes the extractor at
    # all on a plain hit).
    replay = ext.extract_blocks(path, cache=cache)
    assert replay == good


def test_extract_blocks_refresh_failure_still_negative_caches_without_prior_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new refresh-vs-clobber guard (issue #78 round 2) must not disable
    negative-caching outright — only the "overwrite an existing success"
    case is special-cased. A refresh with NO pre-existing cached success
    (first attempt under this key, or a prior same-key failure) still
    negative-caches as before, so repeated plain calls keep failing fast
    instead of re-attempting a full docling OCR timeout every round.
    """
    from playbook_engine import extraction as ext

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake scanned pdf")
    cache = ext.ExtractionCache(tmp_path / "cache.jsonl")

    monkeypatch.setattr(ext, "detect_extractor", lambda p: "legacy")
    monkeypatch.setattr(ext, "_extract_legacy_lines", lambda p, s: [])

    # refresh=True as the very first call for this key: no cached success
    # exists yet, so the failure must still be negative-cached.
    with pytest.raises(ext.ExtractionError, match="yielded no text"):
        ext.extract_blocks(pdf, cache=cache, refresh=True)

    assert cache.get(pdf) is None
    assert cache.get_failure(pdf) is not None, (
        "a refresh failure with no pre-existing success must still negative-cache"
    )

    # A subsequent PLAIN call must fail fast from the negative cache, not
    # re-invoke the extractor.
    calls = {"n": 0}
    real = ext._extract_legacy_lines

    def _counting(p: Path, s: str) -> list[tuple[str, int]]:
        calls["n"] += 1
        return real(p, s)

    monkeypatch.setattr(ext, "_extract_legacy_lines", _counting)
    with pytest.raises(ext.ExtractionError, match="cached failure"):
        ext.extract_blocks(pdf, cache=cache)
    assert calls["n"] == 0
