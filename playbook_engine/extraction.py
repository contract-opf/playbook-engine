"""Block-stream extractor — deterministic document → (canonical text, blocks).

This is the extractor half of the LLM-segmentation seam (see
:mod:`playbook_engine.segmentation_grounding`): it turns a source document
into a flat, ordered stream of :class:`~playbook_engine.segmentation_grounding.Block`
plus the canonical text those blocks index into.  No structure detection, no
LLM — that happens downstream (the LLM segmenter groups blocks into clauses;
grounding reconstructs verbatim text/spans from this stream).

Supported formats:
  - **docling** (preferred, any of DOCX/PDF/RTF) — via a ``docling`` CLI
    subprocess (``docling convert <path> --to md --image-export-mode
    placeholder --output <tmpdir>``), like the ``pandoc`` RTF path below.
    docling converts the source into
    layout-aware Markdown; that Markdown is then parsed into one block per
    logical unit (heading, paragraph, list item, table row) in reading
    order.  This is used whenever ``docling`` is on ``PATH``
    (``shutil.which("docling")``) — see "docling vs. legacy adapters"
    below.  ``page`` is best-effort ``0`` (docling Markdown is not
    paginated — mirrors the RTF/DOCX convention).
  - **DOCX** (fallback) — walks the body XML the same way
    :mod:`playbook_engine.docx_ingester` does (reusing its
    ``_iter_body_blocks``/``_extract_para_text`` helpers), one block per
    non-empty paragraph or flattened table.  This matters because
    ``python-docx``'s ``paragraph.text`` silently drops text inside
    ``w:ins`` (tracked-change insertions) and ``doc.paragraphs`` skips
    table content entirely — using the same XML walk as the deterministic
    ingester keeps both paths in agreement on document content (issue #85).
    Tracked-change *deletions* are excluded (mirrors ``docx_ingester``):
    canonical text reflects current/accepted content, not withdrawn
    language. Headings are not treated specially here: this module only
    emits text blocks (heading detection is out of scope for this slice —
    the LLM infers boundaries from block text).
  - **PDF** (fallback) — via ``pdfplumber``.  One block per extracted text
    line, tagged with its 1-based source page number.
  - **RTF** (fallback) — via a ``pandoc`` subprocess (``pandoc <in.rtf> -t
    plain --wrap=none``).  One block per non-empty paragraph.  ``pandoc``
    is a system binary, not a Python dependency; when it is not on
    ``PATH`` this raises :class:`ExtractionError` rather than silently
    falling back.

docling vs. legacy adapters:
  ``extract_blocks`` prefers docling whenever ``shutil.which("docling")``
  finds the binary, regardless of file extension — docling handles
  DOCX/PDF/RTF uniformly and gives the LLM segmenter real heading/structure
  cues that the legacy per-format adapters above cannot. When docling is
  absent (e.g. host dev without the container), extraction falls back to
  the legacy adapters unchanged. Which path is used is logged at INFO *and*
  returned as ``extractor`` (``"docling"`` or ``"legacy"``) from
  :func:`extract_blocks` — see :func:`detect_extractor`. This was
  previously only visible via that ``logging.info`` line, which is
  suppressed by default Python logging config: a host install without
  docling would silently fall back to pdfplumber (no OCR) for scanned
  PDFs with no way for the operator to notice short of reading a
  scrolled-past log line (issue #129). docling itself is invoked as a
  subprocess only — it is never imported as a Python module, keeping the
  engine importable without ``torch`` on the host.

Markdown → Block parsing (docling path) and citation cleanliness:
  The block ``text`` used for grounding/citation must be the *clean*
  clause text: Markdown decoration is stripped from ``text`` even though
  it is used to *detect* block boundaries. Concretely, per output line:
    - ATX headings (``# Heading``, ``## Heading``, …) become their own
      block; the leading ``#`` markers and following space are stripped.
    - List items (leading ``-``, ``*``, ``+``, or ``N.``) become their own
      block; the leading marker is stripped.
    - Markdown table rows (``| a | b |``) become one block per row with
      leading/trailing pipes stripped and cell separators normalized to
      ``" | "``.
    - Bold/italic decoration (``**text**``, ``*text*``) has its ``*``/``_``
      markers stripped from the block text.
    - Blank lines separate blocks but never become blocks themselves.
  Boundary detection happens on the raw (undecorated) line; stripping is
  applied only to the text stored on the ``Block`` — so heading/list/table
  structure still drives block boundaries even though the punctuation that
  signaled it is gone from the citable text.

canonical_text / char_span contract:
  ``canonical_text`` is every block's ``text`` joined by ``"\\n"``, in
  reading order.  Each block's ``char_span`` is its ``[start, end)`` offset
  into that joined string, so for every block:
  ``block.text == canonical_text[block.char_span[0]:block.char_span[1]]``.
  This is the same joining convention as the ``_stream`` test helper in
  ``tests/test_segmentation_grounding.py`` — grounding depends on it being
  exact, since it is what lets a clause's ``char_span`` be reconstructed
  verbatim from the LLM's block references.

Not in scope (see issue #78): OCR toggling for scanned PDFs (docling's OCR
is enabled in a later slice) and Dockerfile packaging of the ``docling``
binary (separate issue) — this module only shells out to it when present.

``ExtractionCache`` (issue #132): extraction (especially docling OCR over a
scanned PDF) is the single most expensive step in the LLM-segmentation path —
far more expensive than the LLM segmentation call itself, which
``SegmentationVerdictCache`` (see ``llm_segmenter_batch.py``) already caches
independently. Before this, ``extract_blocks`` had no cache of its own, so
any caller that (for good reason — see ``agent_judge.StoreBackedClassificationJudge``
et al.) bypasses the pipeline's L1-L4 ``ArtifactStore``/``JudgmentCache``
(``no_cache=True``, forced by ``playbook judge``'s store-backed judges to
avoid replaying stale ``needs_review`` sentinels — see ``cli.py``'s
``_verdict_store_kwargs``) also silently threw away every prior extraction,
re-running docling/pdfplumber/pandoc from scratch on every judge round over a
real multi-hundred-version corpus. ``ExtractionCache`` is content-addressed
purely on the source file's bytes (extraction has no judge/config
dependency), so it is safe to keep warm across rounds regardless of whatever
``no_cache`` value the judge wiring forces for the verdict-cache layers.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pdfplumber
from docx import Document

from playbook_engine.agent_judge import VerdictStore
from playbook_engine.artifact_store import _sha256_file
from playbook_engine.docx_ingester import _extract_para_text, _iter_body_blocks
from playbook_engine.segmentation_grounding import Block

_log = logging.getLogger(__name__)

#: Bumped whenever ``extract_blocks``'s output shape changes in a way that
#: should invalidate previously cached entries (e.g. a block-parsing bug fix).
_EXTRACTION_CACHE_FORMAT_VERSION = "1"

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Raised when a document cannot be extracted into a block stream.

    Covers unsupported file extensions, a missing ``pandoc``/``docling``
    binary, and extraction that yields no usable text (e.g. an empty/blank
    source or a failed/empty docling conversion).
    """


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_extractor(path: Path) -> str:
    """Return which extractor :func:`extract_blocks` will use for *path*.

    ``"docling"`` when ``shutil.which("docling")`` finds the binary,
    ``"legacy"`` otherwise (pdfplumber/python-docx/pandoc, per format) —
    the exact same check ``extract_blocks`` makes internally to choose its
    code path. Factored out so callers that need to *record* the choice
    (corpus manifest, ``mine``/``compile`` CLI output) can do so up front —
    including for a version whose extraction subsequently fails — without
    duplicating or risking drift from ``extract_blocks``'s own branch
    (issue #129: this was previously only visible via a suppressed
    ``logging.info`` line).
    """
    return "docling" if shutil.which("docling") is not None else "legacy"


def _extraction_cache_payload(path: Path) -> dict[str, str]:
    """Cache-key payload for *path*: the file's raw content hash alone.

    Extraction is a pure function of the source bytes — no judge, no
    segmentation model, no engine config affects it — so no other input
    belongs in this key (issue #132).
    """
    return {
        "file_sha256": _sha256_file(path),
        "format_version": _EXTRACTION_CACHE_FORMAT_VERSION,
    }


class ExtractionCache:
    """Judge-once, deterministic-replay cache for :func:`extract_blocks` output.

    Wraps a :class:`~playbook_engine.agent_judge.VerdictStore` rather than
    reimplementing content-hash JSONL storage (same pattern as
    :class:`~playbook_engine.llm_segmenter_batch.SegmentationVerdictCache`).

    Cache key: the source file's raw content hash only (see
    :func:`_extraction_cache_payload`) — independent of ``no_cache``, judge
    identity, or engine config, so a repeat ``playbook judge`` round can
    reuse a prior run's extracted blocks/clause trees for every version whose
    source file is unchanged, even though the L1-L4 ``ArtifactStore``/
    ``JudgmentCache`` stage cache is deliberately bypassed for store-backed
    judge runs (issue #132).
    """

    def __init__(self, cache_path: Path) -> None:
        self._store = VerdictStore(cache_path)

    def get(self, path: Path) -> tuple[str, list[Block], str] | None:
        """Return the cached ``(canonical_text, blocks, extractor)``, or ``None`` on a miss."""
        cached = self._store.get(_extraction_cache_payload(path))
        if cached is None:
            return None
        blocks = [
            Block(
                block_id=b["block_id"],
                page=b["page"],
                char_span=(b["char_span"][0], b["char_span"][1]),
                text=b["text"],
            )
            for b in cached["blocks"]
        ]
        return cached["canonical_text"], blocks, cached["extractor"]

    def put(self, path: Path, canonical_text: str, blocks: list[Block], extractor: str) -> None:
        """Store *canonical_text*/*blocks*/*extractor* for *path*'s current content."""
        value: dict[str, Any] = {
            "canonical_text": canonical_text,
            "blocks": [
                {
                    "block_id": b.block_id,
                    "page": b.page,
                    "char_span": list(b.char_span),
                    "text": b.text,
                }
                for b in blocks
            ],
            "extractor": extractor,
        }
        self._store.put(_extraction_cache_payload(path), value)


def extract_blocks(
    path: Path, *, cache: ExtractionCache | None = None
) -> tuple[str, list[Block], str]:
    """Extract ``path`` into ``(canonical_text, blocks, extractor)``.

    ``blocks`` are in reading order; ``block_id`` values are ``"b0", "b1",
    …`` in that order.  Every block's ``char_span`` is an offset into
    ``canonical_text`` such that
    ``block.text == canonical_text[slice(*block.char_span)]``.
    ``extractor`` is ``"docling"`` or ``"legacy"`` — see
    :func:`detect_extractor`.

    Args:
        path:  Path to a ``.docx``, ``.pdf``, or ``.rtf`` file.
        cache: Optional :class:`ExtractionCache`. When given, a hit on
               *path*'s current content skips extraction entirely (no
               docling subprocess, no pdfplumber/python-docx/pandoc call) —
               see issue #132. On a miss, the result is stored before
               returning. Defaults to ``None`` (no caching — every call
               re-extracts).

    Raises:
        ExtractionError: unsupported extension, missing ``pandoc`` (RTF), or
                          the document yields no non-empty text.
    """
    if not path.is_file():
        raise ExtractionError(f"file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in (".docx", ".pdf", ".rtf"):
        raise ExtractionError(f"unsupported file extension: {suffix!r} ({path})")

    if cache is not None:
        cached = cache.get(path)
        if cached is not None:
            return cached

    extractor = detect_extractor(path)
    if extractor == "docling":
        _log.info("extract_blocks: using docling for %s", path)
        try:
            lines = _extract_docling_lines(path)
        except ExtractionError as exc:
            # docling's per-format backends can fail on inputs the legacy
            # adapters handle fine — notably docling 2.x's DOCX backend raises
            # on tracked-changes/comment nodes (``etree.QName`` on a comment
            # factory), which is exactly what redline drafts contain. Skipping
            # the document would silently drop negotiation versions and corrupt
            # the trail, so fall back to the legacy per-format adapter for this
            # one file instead. A scanned PDF that docling cannot OCR will still
            # yield little here (legacy has no OCR) and then raise below, as
            # before; born-digital docx/pdf are recovered. The fallback is
            # logged and reflected in the returned ``extractor`` label so it is
            # visible in reporting (issue #129).
            _log.warning(
                "extract_blocks: docling failed on %s (%s); falling back to legacy adapter",
                path,
                exc,
            )
            lines = _extract_legacy_lines(path, suffix)
            extractor = "legacy"
    else:
        _log.info("extract_blocks: docling not found on PATH; using legacy adapter for %s", path)
        lines = _extract_legacy_lines(path, suffix)

    if not lines:
        raise ExtractionError(f"extraction yielded no text: {path}")

    canonical_text, blocks = _build_stream(lines)

    if cache is not None:
        cache.put(path, canonical_text, blocks, extractor)

    return canonical_text, blocks, extractor


def _extract_legacy_lines(path: Path, suffix: str) -> list[tuple[str, int]]:
    """Dispatch to the legacy per-format adapter for ``suffix``.

    Shared by the ``docling`` fallback path and the no-docling path in
    :func:`extract_blocks`. ``suffix`` is the already-lowercased extension.
    """
    if suffix == ".docx":
        return _extract_docx_lines(path)
    if suffix == ".pdf":
        return _extract_pdf_lines(path)
    return _extract_rtf_lines(path)


# ---------------------------------------------------------------------------
# Shared stream builder
# ---------------------------------------------------------------------------


def _build_stream(lines: list[tuple[str, int]]) -> tuple[str, list[Block]]:
    """Build ``(canonical_text, blocks)`` from ``(text, page)`` pairs.

    ``canonical_text`` is every text joined by ``"\\n"``; each block's
    ``char_span`` is computed against that joined string.  Mirrors the
    ``_stream`` helper pattern in ``tests/test_segmentation_grounding.py``.
    """
    blocks: list[Block] = []
    offset = 0
    for i, (text, page) in enumerate(lines):
        if i > 0:
            offset += 1  # the "\n" separator
        span = (offset, offset + len(text))
        blocks.append(Block(block_id=f"b{i}", page=page, char_span=span, text=text))
        offset += len(text)
    canonical_text = "\n".join(text for text, _ in lines)
    return canonical_text, blocks


# ---------------------------------------------------------------------------
# docling (preferred: structure-preserving, via CLI subprocess)
# ---------------------------------------------------------------------------


def _extract_docling_lines(path: Path) -> list[tuple[str, int]]:
    """One ``(text, page=0)`` per logical Markdown unit, via ``docling``.

    Runs ``docling convert <path> --to md --image-export-mode placeholder
    --output <tmpdir>`` (a subprocess, never imported as a Python module) and
    parses the resulting ``<stem>.md`` into blocks. The ``convert`` subcommand
    is mandatory in docling >=2.x; ``placeholder`` image mode keeps page
    images out of the Markdown. docling Markdown is not paginated, so ``page``
    is always 0 (mirrors the RTF/DOCX convention).
    """
    with tempfile.TemporaryDirectory(prefix="docling-") as tmpdir:
        markdown = _run_docling(path, Path(tmpdir))
        return _parse_markdown_lines(markdown)


# OCR language passed to ``docling convert --ocr-lang``. English by default;
# docling's own default is Chinese, which corrupts Latin-script scans.
_DOCLING_OCR_LANG = "eng"

# Per-file wall-clock cap on the docling subprocess. docling cold-loads its
# models on every invocation and can hang indefinitely on a pathological
# input; without a cap, one bad file blocks the whole corpus mine run
# forever (issue #98). 10 minutes comfortably covers even large scanned
# PDFs; a genuinely stuck conversion is far more likely than a legitimate
# one still running at that point.
_DOCLING_TIMEOUT_S = 600


def _run_docling(path: Path, outdir: Path) -> str:
    """Invoke the ``docling`` CLI and return the produced Markdown text.

    Isolated in its own helper so the exact invocation is easy to adjust
    after in-container validation (see issue #78 Notes).
    """
    try:
        subprocess.run(
            [
                "docling",
                "convert",
                str(path),
                "--to",
                "md",
                # Emit a `<!-- image -->` placeholder instead of embedding page
                # images as multi-KB base64 data URIs (docling's default),
                # which would otherwise become garbage blocks that wreck token
                # cost and citation text. The parser drops the placeholders.
                "--image-export-mode",
                "placeholder",
                # Pin OCR to English. docling's RapidOCR default is Chinese
                # (`lang=["chinese"]`), which joins/garbles Latin-script words on
                # scanned SIGNED copies. Change per corpus language (e.g. `deu`).
                "--ocr-lang",
                _DOCLING_OCR_LANG,
                "--output",
                str(outdir),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=_DOCLING_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExtractionError(
            f"docling timed out after {_DOCLING_TIMEOUT_S}s converting {path}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ExtractionError(f"docling failed to convert {path}: {exc.stderr.strip()}") from exc
    except OSError as exc:
        raise ExtractionError(f"cannot run docling: {exc}") from exc

    md_path = outdir / f"{path.stem}.md"
    try:
        markdown = md_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExtractionError(f"docling did not produce expected output {md_path}: {exc}") from exc

    if not markdown.strip():
        raise ExtractionError(f"docling produced empty output for {path}")

    return markdown


# Markdown line patterns used to detect block boundaries. Detection happens
# on the raw line; the *stored* block text has decoration stripped (see
# module docstring, "Markdown → Block parsing").
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_ITEM_RE = re.compile(r"^([-*+]|\d+[.)])\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\|(.*)\|$")
_EMPHASIS_RE = re.compile(r"(\*\*|\*|__|_)")
# Markdown image lines (`![alt](...)`, incl. base64 data URIs) and HTML
# comments (docling's `<!-- image -->` placeholder) are not citable clause
# text — they are dropped entirely, never emitted as blocks.
_IMAGE_RE = re.compile(r"^!\[")


def _strip_markdown_decoration(text: str) -> str:
    """Strip bold/italic emphasis markers from already-boundary-parsed text."""
    return _EMPHASIS_RE.sub("", text).strip()


def _parse_markdown_lines(markdown: str) -> list[tuple[str, int]]:
    """Parse docling Markdown into ``(clean_text, page=0)`` blocks.

    One block per heading, paragraph, list item, or table row, in reading
    order. Markdown decoration (``#``, ``**``/``*``/``_``, leading
    ``-``/``*``/``N.``, table pipes) is stripped from the stored text while
    still being used to detect block boundaries.
    """
    lines: list[tuple[str, int]] = []
    for raw_line in markdown.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        # Drop image lines and HTML comments (docling's image placeholder) —
        # they are never citable clause text.
        if stripped.startswith("<!--") or _IMAGE_RE.match(stripped):
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            text = _strip_markdown_decoration(heading_match.group(2))
            if text:
                lines.append((text, 0))
            continue

        table_match = _TABLE_ROW_RE.match(stripped)
        if table_match:
            cells = [c.strip() for c in table_match.group(1).split("|")]
            # Skip Markdown table separator rows, e.g. "| --- | --- |".
            if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
                continue
            text = _strip_markdown_decoration(" | ".join(c for c in cells if c))
            if text:
                lines.append((text, 0))
            continue

        list_match = _LIST_ITEM_RE.match(stripped)
        if list_match:
            text = _strip_markdown_decoration(list_match.group(2))
            if text:
                lines.append((text, 0))
            continue

        text = _strip_markdown_decoration(stripped)
        if text:
            lines.append((text, 0))

    return lines


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def _extract_docx_lines(path: Path) -> list[tuple[str, int]]:
    """One ``(text, page=0)`` per non-empty paragraph or table, in document order.

    Walks the body XML via :func:`playbook_engine.docx_ingester._iter_body_blocks`
    (paragraph elements and flattened ``w:tbl`` tables) instead of
    ``doc.paragraphs`` / ``paragraph.text``, so that:
      - tracked-change insertions (``w:ins``) are included — ``paragraph.text``
        only concatenates runs that are direct children of ``w:p`` and silently
        drops ``w:ins``-nested runs (issue #85);
      - table content is captured at all — ``doc.paragraphs`` skips tables
        entirely.
    Tracked-change deletions are excluded (mirrors ``docx_ingester``): the
    canonical text reflects current/accepted content, not withdrawn language.
    DOCX is not paginated at the model level, so ``page`` is always 0.
    Heading detection (``paragraph.style.name`` starting with "Heading") is
    not captured here — ``Block`` has no heading field; text blocks are
    sufficient for this slice (see module docstring).
    """
    try:
        doc = Document(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"cannot open DOCX: {exc}") from exc

    lines: list[tuple[str, int]] = []
    for block in _iter_body_blocks(doc):
        if isinstance(block, str):
            # Flattened table text (see docx_ingester._flatten_table).
            text = block.strip()
        else:
            # Paragraph XML element — w:ins-aware, w:del-excluded text.
            para_text, _tracked = _extract_para_text(block)
            text = para_text.strip()
        if text:
            lines.append((text, 0))
    return lines


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _extract_pdf_lines(path: Path) -> list[tuple[str, int]]:
    """One ``(text, page)`` per extracted text line, ``page`` 1-based."""
    try:
        lines: list[tuple[str, int]] = []
        with pdfplumber.open(str(path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                for line in page.extract_text_lines():
                    text = line["text"].strip()
                    if text:
                        lines.append((text, page_number))
        return lines
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"cannot open PDF: {exc}") from exc


# ---------------------------------------------------------------------------
# RTF (via pandoc subprocess)
# ---------------------------------------------------------------------------


def _extract_rtf_lines(path: Path) -> list[tuple[str, int]]:
    """One ``(text, page=0)`` per paragraph, via a ``pandoc`` subprocess.

    RTF is not paginated at the model level, so ``page`` is always 0.
    ``pandoc`` is a system binary (not a Python package) — when it is not on
    ``PATH`` this raises :class:`ExtractionError` with a clear message
    rather than attempting a degraded fallback.
    """
    if shutil.which("pandoc") is None:
        raise ExtractionError(
            "pandoc is required to extract RTF but was not found on PATH "
            "(install it, e.g. `brew install pandoc`, or `apt-get install pandoc`)"
        )

    try:
        result = subprocess.run(
            ["pandoc", str(path), "-t", "plain", "--wrap=none"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ExtractionError(f"pandoc failed to convert RTF: {exc.stderr.strip()}") from exc
    except OSError as exc:
        raise ExtractionError(f"cannot run pandoc: {exc}") from exc

    # pandoc's plain output separates paragraphs by blank lines (with
    # --wrap=none disabling mid-paragraph line wrapping, so each paragraph is
    # exactly one physical output line). Splitting on newlines and dropping
    # empty lines recovers one entry per paragraph.
    lines: list[tuple[str, int]] = []
    for raw_line in result.stdout.splitlines():
        text = raw_line.strip()
        if text:
            lines.append((text, 0))
    return lines
