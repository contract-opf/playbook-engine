"""RTF ingester — parses RTF documents into a normalized ClauseTree.

Text extraction:
  RTF documents are converted to plain text via ``striprtf``.  All RTF
  formatting markup (fonts, colors, bold/italic, tables) is discarded.
  The resulting text is processed the same way as PDF text: line by line.

Structure detection (identical to the PDF ingester):
  - Lines matching the numbered-prefix pattern (``"1."``, ``"1.2.3)"``) →
    heading at depth = count of dot-separated number components.
  - Short ALL-CAPS lines (≤ 8 words) without a numbered prefix → level-1
    heading (heuristic for unnumbered section titles).
  - All other lines → body text appended to the nearest heading node.

  Pre-heading body text is collected in a synthetic ``clause_path="0"`` node
  (same convention as the DOCX and PDF ingesters).

char_span coordinate system:
  ``ClauseNode.char_span`` values are document-absolute offsets within the
  **virtual normalized text** — the non-empty stripped lines joined by
  ``"\\n"`` (i.e. ``"\\n".join(_split_lines(raw_text))``).  A clause node's
  ``char_span`` covers only the **heading line**; body text accumulated in
  ``.text`` does not extend the span.  This is the same convention as the
  DOCX and PDF ingesters.

Limitations:
  - RTF tables are not preserved; their text content is extracted as a flat
    stream of lines.
  - Tracked changes (``\\trckchng`` markup) are not extracted.  RTF tracked
    changes require a full RTF parser; use the DOCX ingester when tracked
    changes are needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from striprtf.striprtf import rtf_to_text

from playbook_engine.clause_tree import ClauseNode, ClauseTree

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class RtfIngesterError(ValueError):
    """Raised on unrecoverable parse failures."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class RtfIngestResult:
    tree: ClauseTree


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def ingest_rtf(path: Path, document_id: str, version: str) -> RtfIngestResult:
    """Parse an RTF file into a ClauseTree.

    Args:
        path:         Path to the RTF file.
        document_id:  Corpus document identifier.
        version:      Version label.
    """
    if not path.is_file():
        raise RtfIngesterError(f"RTF file not found: {path}")

    try:
        # RTF byte streams are cp1252 (Windows-1252) by default.  Reading as
        # UTF-8 would corrupt raw high bytes (e.g. 0x92 right-single-quote)
        # into U+FFFD before striprtf sees them.  We decode as cp1252 so that
        # both raw bytes and \\'XX escapes reach striprtf intact.
        raw = path.read_bytes().decode("cp1252", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise RtfIngesterError(f"Cannot read RTF file: {exc}") from exc

    try:
        text = rtf_to_text(raw, encoding="cp1252", errors="replace")  # type: ignore[no-untyped-call]
    except Exception as exc:  # noqa: BLE001
        raise RtfIngesterError(f"Cannot parse RTF: {exc}") from exc

    builder = _ClauseBuilder()
    doc_char_offset = 0

    for line in _split_lines(text):
        level = _para_level(line)
        if level is not None:
            clause_path, heading = _parse_clause_number(line, level, builder)
            span = (doc_char_offset, doc_char_offset + len(line))
            builder.start_clause(level, clause_path, heading=heading, char_span=span)
        else:
            builder.add_body(line, doc_char_offset)
        doc_char_offset += len(line) + 1

    return RtfIngestResult(
        tree=ClauseTree(
            document_id=document_id,
            version=version,
            source_file=path.name,
            nodes=builder.build(),
        )
    )


# ---------------------------------------------------------------------------
# Line splitting
# ---------------------------------------------------------------------------


def _split_lines(text: str) -> list[str]:
    """Return non-empty stripped lines from extracted RTF text."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return lines


# ---------------------------------------------------------------------------
# Structure detection (mirrors pdf_ingester._para_level)
# ---------------------------------------------------------------------------

_NUM_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)[.)]\s*")
_MAX_ALLCAPS_HEADING_WORDS = 8


def _para_level(text: str) -> int | None:
    """Return heading level or None for body text."""
    stripped = text.strip()
    m = _NUM_PREFIX.match(stripped)
    if m:
        return len(m.group(1).split("."))

    words = stripped.split()
    if (
        1 <= len(words) <= _MAX_ALLCAPS_HEADING_WORDS
        and stripped == stripped.upper()
        and stripped.isascii()
        and not stripped.endswith(".")
    ):
        return 1

    return None


def _parse_clause_number(
    text: str,
    level: int,
    builder: _ClauseBuilder,
) -> tuple[str, str | None]:
    m = _NUM_PREFIX.match(text)
    if m:
        clause_path = m.group(1)
        heading_text = text[m.end() :].strip() or None
        return clause_path, heading_text
    clause_path = builder.generate_path(level)
    return clause_path, text.strip() or None


# ---------------------------------------------------------------------------
# Clause tree builder (stack-based — same as PDF and DOCX ingesters)
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
        node = ClauseNode(clause_path=clause_path, heading=heading, text="", char_span=char_span)
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

    def build(self) -> list[ClauseNode]:
        return self._root
