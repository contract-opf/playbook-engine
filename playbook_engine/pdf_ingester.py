"""PDF ingester — parses PDFs into a normalized ClauseTree.

Extraction strategy (in order):
  1. Text-layer extraction via pdfplumber.  If the document yields at least
     ``MIN_TEXT_CHARS`` characters across all pages it is classified as
     "born-digital" and the text is used directly.
  2. OCR fallback.  When the text layer is absent or sparse an ``OCRAdapter``
     is called.  The adapter is injected at call time so callers can plug in
     Textract, Azure Document Intelligence, or a local engine (Tesseract via
     pytesseract) without changing this module.  A ``NullOCRAdapter`` is
     provided as the no-OCR placeholder, and it is what the deterministic
     segmenter uses in EVERY runtime (``_ingest_file_tracked`` passes no
     adapter) — Docker included.  It never silently degrades: a PDF with no
     usable text layer under ``NullOCRAdapter`` raises
     :class:`NoOCRRuntimeError` instead of returning an empty tree (issue
     #218).  OCR (docling plus ocrmypdf) only runs on the ``extract_blocks``
     path: the Docker runtime with ``segmentation.agent: true`` or
     ``segmentation.llm: true``.

Structure detection:
  PDF text carries no style metadata, so clause structure is detected
  heuristically from the extracted text:
  - Lines matching the numbered-prefix pattern (``"1."``, ``"1.2.3)"``) →
    heading at depth = count of dot-separated number components.
  - Short ALL-CAPS lines (≤ 8 words) without a numbered prefix → level-1
    heading (heuristic for unnumbered section titles).  Known false positives:
    recitals/boilerplate such as "WITNESSETH", "WHEREAS", "EXHIBIT A" are
    detected as headings.  Downstream stages must tolerate spurious headings.
  - All other lines → body text appended to the nearest heading node.

  Pre-heading body text is collected in a synthetic ``clause_path="0"`` node
  (same convention as the DOCX ingester).

char_span coordinate system:
  ``ClauseNode.char_span`` values are document-absolute offsets within the
  **virtual normalized text** — the non-empty, stripped lines joined by
  ``"\\n"`` (i.e. ``"\\n".join(_split_paragraphs(raw_text))``).  This is
  analogous to the DOCX ingester, which uses paragraph texts joined by ``"\\n"``.

  A clause node's ``char_span`` covers the **whole clause** — from the start
  of its heading line through the end of its own body text accumulated via
  ``add_body()`` (children excluded) — and its ``heading_span`` covers the
  heading line alone (issue #217).  The synthetic pre-heading
  ``clause_path="0"`` node has ``heading_span=None``.  This is the same
  convention as the DOCX ingester.  To recover the text a span refers to, call:
  ``ClauseTree.resolve_span("\\n".join(_split_paragraphs(raw_text)), span)``.

Extraction metadata is recorded on ``PdfIngestResult``:
  - ``method``: ``"text-layer"`` or ``"ocr"``
  - ``confidence``: ``1.0`` for text-layer; OCR adapters SHOULD return a value
    in ``[0.0, 1.0]``; ``NullOCRAdapter`` returns ``0.0``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import pdfplumber

from playbook_engine.clause_tree import ClauseNode, ClauseTree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum character count to consider a PDF as having a usable text layer.
MIN_TEXT_CHARS: int = 20


# ---------------------------------------------------------------------------
# OCR adapter protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OCRAdapter(Protocol):
    """Pluggable OCR back-end.  Implementations must be stateless."""

    def extract_text(self, pdf_path: Path) -> tuple[str, float]:
        """Return (full_text, confidence) for the given PDF.

        ``confidence`` is in [0.0, 1.0]; 0.0 means "no text / not attempted".
        """
        ...


@dataclass
class NullOCRAdapter:
    """No-op adapter.  Returns empty text with confidence=0.0.

    Use in tests and as a placeholder when no OCR service is configured.
    """

    def extract_text(self, pdf_path: Path) -> tuple[str, float]:
        return ("", 0.0)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class PdfIngestResult:
    tree: ClauseTree
    method: str  # "text-layer" | "ocr"
    confidence: float  # 1.0 for text-layer; OCR adapter value otherwise


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class PdfIngesterError(ValueError):
    """Raised on unrecoverable parse failures."""


class NoOCRRuntimeError(PdfIngesterError):
    """A PDF has no usable text layer and this ingest path has no OCR (issue #218).

    ``NullOCRAdapter`` is the deterministic segmenter's adapter in every
    runtime, so this is raised in Docker too; the message points at the
    ``extract_blocks`` path (``segmentation.agent``/``segmentation.llm``),
    the only one that runs docling OCR and ocrmypdf.

    Raised instead of returning an empty tree, which used to be the only
    signal (``MIN_TEXT_CHARS`` was the sole tripwire). A subclass of
    :class:`PdfIngesterError` so every existing per-version failure handler
    still catches it; the pipeline records only the exception type name.
    """


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def ingest_pdf(
    path: Path,
    document_id: str,
    version: str,
    ocr_adapter: OCRAdapter | None = None,
) -> PdfIngestResult:
    """Parse a PDF into a ClauseTree, recording extraction method + confidence.

    Args:
        path:         Path to the PDF file.
        document_id:  Corpus document identifier.
        version:      Version label.
        ocr_adapter:  OCR back-end for scanned/image PDFs.  Defaults to
                      ``NullOCRAdapter`` when omitted.

    Raises:
        PdfIngesterError:  the file is missing or cannot be opened.
        NoOCRRuntimeError: the PDF has fewer than ``MIN_TEXT_CHARS`` characters
                           of text layer and *ocr_adapter* is (or defaults to)
                           ``NullOCRAdapter`` — see issue #218.
    """
    if not path.is_file():
        raise PdfIngesterError(f"PDF file not found: {path}")

    if ocr_adapter is None:
        ocr_adapter = NullOCRAdapter()

    # --- text extraction ---
    text, method, confidence = _extract_text(path, ocr_adapter)

    # --- clause tree construction ---
    builder = _ClauseBuilder()
    doc_char_offset = 0

    for para in _split_paragraphs(text):
        level = _para_level(para)
        if level is not None:
            clause_path, heading = _parse_clause_number(para.strip(), level, builder)
            span = (doc_char_offset, doc_char_offset + len(para))
            builder.start_clause(level, clause_path, heading=heading, char_span=span)
        else:
            builder.add_body(para, doc_char_offset)
        doc_char_offset += len(para) + 1

    return PdfIngestResult(
        tree=ClauseTree(
            document_id=document_id,
            version=version,
            source_file=path.name,
            nodes=builder.build(),
        ),
        method=method,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def _extract_text(path: Path, ocr_adapter: OCRAdapter) -> tuple[str, str, float]:
    """Return (text, method, confidence)."""
    try:
        text = _extract_text_layer(path)
    except Exception as exc:  # noqa: BLE001
        raise PdfIngesterError(f"Cannot open PDF: {exc}") from exc

    if len(text.strip()) >= MIN_TEXT_CHARS:
        return text, "text-layer", 1.0

    if isinstance(ocr_adapter, NullOCRAdapter):
        # Fail loud (issue #218): NullOCRAdapter can only ever return "", so
        # continuing would hand back an empty tree for what is almost always
        # a scanned PDF. NullOCRAdapter is the deterministic segmenter's
        # adapter in EVERY runtime (Docker included), so the fix is a
        # different ingest path, not a different runtime alone.
        raise NoOCRRuntimeError(
            f"{path.name} has no usable text layer (fewer than {MIN_TEXT_CHARS} "
            "characters) — it is probably a scanned PDF, and the deterministic "
            "segmenter has no OCR in any runtime, Docker included. OCR (docling "
            "plus ocrmypdf) only runs on the extract_blocks path: run the corpus "
            "in the Docker runtime (make docker-build && make docker-run) with "
            "segmentation.agent: true in config.yaml (no API key needed) or "
            "segmentation.llm: true, or pass a real OCRAdapter to ingest_pdf."
        )

    # Sparse or no text layer — fall back to OCR
    ocr_text, ocr_conf = ocr_adapter.extract_text(path)
    return ocr_text, "ocr", ocr_conf


def _extract_text_layer(path: Path) -> str:
    """Extract full text from a PDF's embedded text layer via pdfplumber."""
    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------


def _split_paragraphs(text: str) -> list[str]:
    """Split extracted PDF text into line-level chunks for structure detection.

    PDF text layers carry no reliable paragraph-boundary signal — pdfplumber
    separates all lines with single newlines.  We therefore treat each
    non-empty line as a unit to classify.  Consecutive body lines are merged
    by the caller via ``_ClauseBuilder.add_body()``.
    """
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return lines


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

# Matches numbered prefix: "1.", "1.2.", "2.1)" at start of text. The
# trailing (?!\d) lookahead forbids a digit right after the terminator so a
# multi-part number like "10.1.3 Term" is not mis-split as "10.1" with a
# dangling "3 Term" heading, and decimals like "1.5 times" / "99.9%" /
# "10.00" are not mistaken for clause numbers.
_NUM_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)[.)](?!\d)\s*")

# Heuristic: ALL-CAPS line ≤ 8 words → level-1 heading.
_MAX_ALLCAPS_HEADING_WORDS = 8


def _para_level(text: str) -> int | None:
    """Return heading level or None for body text."""
    stripped = text.strip()
    m = _NUM_PREFIX.match(stripped)
    if m:
        return len(m.group(1).split("."))

    # Heuristic: ALL-CAPS short line without punctuation-heavy content
    words = stripped.split()
    if (
        1 <= len(words) <= _MAX_ALLCAPS_HEADING_WORDS
        and stripped == stripped.upper()
        and stripped.isascii()
        and not stripped.endswith(".")  # sentences end with "." — not headings
        and any(c.isalpha() for c in stripped)  # bare digits/years/page numbers aren't headings
    ):
        return 1

    return None


def _parse_clause_number(
    text: str,
    level: int,
    builder: _ClauseBuilder,
) -> tuple[str, str | None]:
    """Return (clause_path, heading_text) for a heading paragraph."""
    m = _NUM_PREFIX.match(text)
    if m:
        clause_path = m.group(1)
        heading_text = text[m.end() :].strip() or None
        return clause_path, heading_text
    clause_path = builder.generate_path(level)
    return clause_path, text.strip() or None


# ---------------------------------------------------------------------------
# Clause tree builder (stack-based — mirrors the DOCX ingester's logic)
# ---------------------------------------------------------------------------


class _ClauseBuilder:
    def __init__(self) -> None:
        self._root: list[ClauseNode] = []
        self._stack: list[tuple[int, ClauseNode]] = []
        self._counters: dict[tuple[str, int], int] = {}

    def generate_path(self, level: int) -> str:
        parent_path = ""
        for stack_level, stack_node in reversed(self._stack):
            if stack_level < level:
                parent_path = stack_node.clause_path
                break
        key = (parent_path, level)
        self._counters[key] = self._counters.get(key, 0) + 1
        n = self._counters[key]
        return f"{parent_path}.{n}" if parent_path else str(n)

    def start_clause(
        self,
        level: int,
        clause_path: str,
        *,
        heading: str | None,
        char_span: tuple[int, int],
    ) -> None:
        # char_span starts as the heading line and grows over the node's own
        # body text in add_body() (issue #217); heading_span keeps the
        # heading line alone.
        node = ClauseNode(
            clause_path=clause_path,
            heading=heading,
            text="",
            char_span=char_span,
            heading_span=char_span,
        )
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        if self._stack:
            self._stack[-1][1].children.append(node)
        else:
            self._root.append(node)
        self._stack.append((level, node))

    def add_body(self, text: str, doc_offset: int) -> None:
        if not self._stack:
            node = ClauseNode(
                clause_path="0",
                heading=None,
                text=text,
                char_span=(doc_offset, doc_offset + len(text)),
            )
            self._root.append(node)
            self._stack.append((0, node))
            return
        current = self._stack[-1][1]
        current.text = (current.text + "\n" + text) if current.text else text
        # Extend the clause's span over this body line: char_span covers
        # heading start → end of the node's own body text (issue #217).
        # Body only ever attaches to the stack top, so a node's body is
        # contiguous and never overlaps its children's spans.
        current.char_span = (current.char_span[0], doc_offset + len(text))

    def build(self) -> list[ClauseNode]:
        return self._root
