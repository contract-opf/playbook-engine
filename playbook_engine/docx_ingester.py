"""DOCX ingester — parses Word documents into a normalized ClauseTree.

Structural recognition rules (applied in priority order):
  1. Paragraph style ``Heading N`` (style ID ``HeadingN``, English built-in)
     → heading at level N.
     Limitation: non-English or custom-named heading styles are not detected
     this way; their paragraphs fall through to rule 3.  See _para_level().
  2. Numbered list paragraph (``w:numPr`` with a non-zero ``numId``) → heading
     at level ``ilvl + 1``.  numId=0 is a "remove numbering" sentinel and is
     treated as body text.  ilvl is 0-indexed per OOXML.
  3. Paragraph text starting with an explicit dotted number (``"1."``,
     ``"1.2.3)"`` etc.) → heading at depth = count of dots in the number.
  4. All other paragraphs → body text appended to the nearest heading node.
  5. Tables are flattened (``cell1 | cell2 …``) and treated as body text.
     Tracked changes inside table cells are NOT captured (known gap).

Pre-heading body text — body paragraphs that appear before the first heading —
are collected into a synthetic node with ``clause_path="0"`` at stack level 0.
This node never matches a numbered section but is preserved so no text is lost.

char_span contract:
  ClauseNode.char_span and TrackedChange.char_span both use **document-level**
  offsets: character positions in the document's full normalized text — the
  *kept* (non-blank) paragraph/table units, each stripped of leading and
  trailing whitespace, joined with ``"\\n"``.  Blank paragraphs contribute
  nothing (same convention as the PDF/RTF ingesters): they are excluded from
  node.text, so they must also be excluded from the offset stream, or every
  span after a blank paragraph would drift ahead by one character per blank
  paragraph.  Deleted text is absent from the normalized text, so
  TrackedChange.char_span is None for deletions.

Tracked changes (side-channel — consumed by the tracked-changes overlay stage):
  - ``w:ins``: inserted text recorded with author, date, and char_span in the
    document's normalized text.
  - ``w:del``: deleted text recorded with author, date; char_span=None.
  - Tracked changes nested inside ``w:hyperlink`` and ``w:smartTag`` are
    captured correctly via recursive descent.

``DocxIngestResult.units`` (issue #118) is the ordered stream of exactly those
kept units — one ``TextUnit`` per unit ``doc_char_offset`` advances past,
same order, same spans. It exists so a caller whose ``ClauseTree`` came from
a DIFFERENT extractor's parse of this same file (e.g. docling's Markdown,
under the default ``extraction.extractor="auto"``) can align this module's
own unit stream against that extractor's unit stream and translate
``TrackedChange.char_span`` into the other coordinate space — see
:func:`~playbook_engine.extraction.bridge_tracked_change_spans`. This
module itself never reads ``units``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from docx import Document
from docx.oxml.ns import qn

from playbook_engine.clause_tree import ClauseNode, ClauseTree

# ---------------------------------------------------------------------------
# Tracked-changes data model
# ---------------------------------------------------------------------------

ChangeType = Literal["insertion", "deletion"]


@dataclass
class TrackedChange:
    change_type: ChangeType
    author: str
    date: str | None
    text: str
    clause_path: str
    char_span: tuple[int, int] | None
    """Document-level char span (same coordinate system as ClauseNode.char_span).
    None for deletions since deleted text is absent from the normalized text."""


@dataclass
class TrackedChanges:
    document_id: str
    version: str
    changes: list[TrackedChange] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "version": self.version,
            "changes": [
                {
                    "change_type": c.change_type,
                    "author": c.author,
                    "date": c.date,
                    "text": c.text,
                    "clause_path": c.clause_path,
                    "char_span": list(c.char_span) if c.char_span else None,
                }
                for c in self.changes
            ],
        }


@dataclass(frozen=True)
class TextUnit:
    """One kept (stripped, non-blank) paragraph/table unit from the document
    body, in reading order — the same units ``doc_char_offset`` walks to
    build both ``ClauseNode.char_span`` and ``TrackedChange.char_span`` (see
    the module docstring's char_span contract).

    Exposed so a caller (issue #118's coordinate-space bridge — see
    :func:`~playbook_engine.extraction.bridge_tracked_change_spans`) can
    align this document's OWN unit stream against a different extractor's
    (e.g. docling's) unit stream for the same file, to translate
    ``TrackedChange.char_span`` into that extractor's coordinate system.
    Not consumed anywhere in this module itself.
    """

    text: str
    char_span: tuple[int, int]


@dataclass
class DocxIngestResult:
    tree: ClauseTree
    tracked: TrackedChanges
    units: list[TextUnit] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class DocxIngesterError(ValueError):
    """Raised on unrecoverable parse failures."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _convert_tracked_span(
    raw_span: tuple[int, int] | None,
    strip_shift: int,
    stripped_len: int,
    doc_char_offset: int,
) -> tuple[int, int] | None:
    """Convert a paragraph-local tracked-change span to a document-absolute one.

    ``raw_span`` is measured against the RAW paragraph text (including any
    leading/trailing whitespace); the kept unit in the virtual normalized
    text is ``para_text.strip()``, which is ``stripped_len`` characters long.
    Shift the span left by ``strip_shift`` (the stripped leading whitespace)
    and then clamp both ends to ``[0, stripped_len]`` so a tracked change
    that runs into (or entirely within) trailing whitespace that was
    stripped off never produces a span past the end of the kept unit —
    which would otherwise send ``ClauseTree.resolve_span`` out of bounds.
    """
    if raw_span is None:
        return None
    start = raw_span[0] - strip_shift
    end = raw_span[1] - strip_shift
    end = max(0, min(end, stripped_len))
    start = max(0, min(start, end))
    return (doc_char_offset + start, doc_char_offset + end)


def ingest_docx(path: Path, document_id: str, version: str) -> DocxIngestResult:
    """Parse a DOCX file into a ClauseTree + TrackedChanges side-channel.

    All char_span values (on both ClauseNode and TrackedChange) use
    document-level offsets: positions in the concatenated normalized text
    (kept, stripped paragraphs joined with ``"\\n"``; see the module
    docstring's char_span contract for the full convention).
    """
    if not path.is_file():
        raise DocxIngesterError(f"DOCX file not found: {path}")
    try:
        doc = Document(str(path))
    except Exception as exc:  # noqa: BLE001
        raise DocxIngesterError(f"Cannot open DOCX: {exc}") from exc

    builder = _ClauseBuilder()
    raw_tracked: list[_RawChange] = []
    units: list[TextUnit] = []
    doc_char_offset = 0

    for block in _iter_body_blocks(doc):
        if isinstance(block, str):
            # Flattened table text — treat as body. Empty/blank table blocks
            # are excluded from the virtual normalized text entirely (see
            # blank-paragraph handling below for why this matters).
            stripped_block = block.strip()
            if not stripped_block:
                continue
            units.append(
                TextUnit(
                    text=stripped_block,
                    char_span=(doc_char_offset, doc_char_offset + len(stripped_block)),
                )
            )
            builder.add_body(stripped_block, doc_char_offset)
            doc_char_offset += len(stripped_block) + 1
            continue

        # Paragraph XML element
        p_elem = block
        para_text, para_raw_tracked = _extract_para_text(p_elem)
        stripped_text = para_text.strip()
        level = _para_level(p_elem)

        # Detect explicit numbered prefix ("1.", "1.2.3)") when no style-derived level.
        # Depth is inferred from the count of dot-separated components.
        if level is None and stripped_text:
            m = _NUM_PREFIX.match(stripped_text)
            if m:
                level = len(m.group(1).split("."))

        if not stripped_text:
            # Blank paragraph: it contributes nothing to node.text, so it must
            # also contribute nothing to the virtual normalized text —
            # otherwise doc_char_offset drifts ahead of every char_span that
            # follows (one char per blank paragraph). This matches the
            # PDF/RTF ingesters' convention: the virtual text is exactly the
            # kept (non-blank) units joined by "\n".
            continue

        units.append(
            TextUnit(
                text=stripped_text,
                char_span=(doc_char_offset, doc_char_offset + len(stripped_text)),
            )
        )

        # _extract_para_text's tracked-change spans are paragraph-local
        # against the RAW para_text (including any leading whitespace, e.g. a
        # tab-indented "\t(a) ..." line). Since the kept unit is the
        # *stripped* text, shift those spans left by the stripped-off leading
        # whitespace so they still land on the right characters.
        strip_shift = len(para_text) - len(para_text.lstrip())

        if level is not None:
            clause_path, heading = _parse_clause_number(stripped_text, level, builder)
            span = (doc_char_offset, doc_char_offset + len(stripped_text))
            builder.start_clause(level, clause_path, heading=heading, char_span=span)
            for rc in para_raw_tracked:
                raw_tracked.append(
                    _RawChange(
                        change_type=rc["change_type"],
                        author=rc["author"],
                        date=rc["date"],
                        text=rc["text"],
                        clause_path=clause_path,
                        # Convert paragraph-local span to document-absolute,
                        # clamped to the stripped (kept) unit.
                        char_span=_convert_tracked_span(
                            rc["char_span"], strip_shift, len(stripped_text), doc_char_offset
                        ),
                    )
                )
        else:
            builder.add_body(stripped_text, doc_char_offset)
            for rc in para_raw_tracked:
                raw_tracked.append(
                    _RawChange(
                        change_type=rc["change_type"],
                        author=rc["author"],
                        date=rc["date"],
                        text=rc["text"],
                        clause_path=builder.current_path(),
                        char_span=_convert_tracked_span(
                            rc["char_span"], strip_shift, len(stripped_text), doc_char_offset
                        ),
                    )
                )

        doc_char_offset += len(stripped_text) + 1

    tree = ClauseTree(
        document_id=document_id,
        version=version,
        source_file=path.name,
        nodes=builder.build(),
    )
    tracked = TrackedChanges(
        document_id=document_id,
        version=version,
        changes=[
            TrackedChange(
                change_type=rc.change_type,
                author=rc.author,
                date=rc.date,
                text=rc.text,
                clause_path=rc.clause_path,
                char_span=rc.char_span,
            )
            for rc in raw_tracked
        ],
    )
    return DocxIngestResult(tree=tree, tracked=tracked, units=units)


# ---------------------------------------------------------------------------
# Body block iteration
# ---------------------------------------------------------------------------


def _iter_body_blocks(doc: Any) -> Any:
    """Yield paragraph XML elements and flattened table strings in document order."""
    yield from _iter_blocks_in(doc.element.body)


def _iter_blocks_in(container: Any) -> Any:
    """Yield paragraph XML elements and flattened table strings from ``container``.

    Descends into block-level ``w:sdt`` (structured document tag / content
    control) elements by unwrapping their ``w:sdtContent`` child, so that
    paragraphs and tables wrapped in a content control are not silently
    dropped. Recurses to handle nested content controls.
    """
    for child in container:
        tag = child.tag
        if tag == qn("w:p"):
            yield child
        elif tag == qn("w:tbl"):
            yield _flatten_table(child)
        elif tag == qn("w:sdt"):
            content = child.find(qn("w:sdtContent"))
            if content is not None:
                yield from _iter_blocks_in(content)


def _flatten_table(tbl_elem: Any) -> str:
    """Flatten a table to pipe-separated cell text.

    Only top-level rows/cells of ``tbl_elem`` are treated as this table's own
    cells. Each cell's content is walked via ``_iter_blocks_in`` (the same
    block iterator used for the document body), which descends into nested
    ``w:tbl`` elements by recursing into ``_flatten_table`` itself. This
    ensures a nested table's text is included exactly once — not also
    re-collected by the enclosing cell (which previously used a ``.//w:p``
    descendant search that double-counted paragraphs inside nested tables).

    Known gap: tracked changes inside table cells are not recorded in the
    TrackedChanges side-channel.
    """
    cell_texts: list[str] = []
    for cell in tbl_elem.findall(f"{qn('w:tr')}/{qn('w:tc')}"):
        parts: list[str] = []
        for block in _iter_blocks_in(cell):
            text = block if isinstance(block, str) else _extract_para_text(block)[0]
            if text.strip():
                parts.append(text)
        if parts:
            cell_texts.append(" ".join(parts))
    return " | ".join(cell_texts)


# ---------------------------------------------------------------------------
# Paragraph text + tracked-change extraction
# ---------------------------------------------------------------------------


@dataclass
class _RawChange:
    change_type: ChangeType
    author: str
    date: str | None
    text: str
    char_span: tuple[int, int] | None
    clause_path: str = ""


def _extract_para_text(p_elem: Any) -> tuple[str, list[dict[str, Any]]]:
    """Return (normalized_text, raw_tracked_changes) for one paragraph element.

    Spans in the returned dicts are paragraph-local (starting at 0); callers
    must add the document-level offset to obtain document-absolute positions.

    Handles ``w:ins`` and ``w:del`` nested inside ``w:hyperlink`` /
    ``w:smartTag`` / ``w:fldSimple`` via recursive descent, and descends into
    ``w:sdt`` (structured document tag / content control) elements via their
    ``w:sdtContent`` child so that run text inside content controls is not
    silently dropped.
    """
    parts: list[str] = []
    tracked: list[dict[str, Any]] = []
    offset = 0

    def _process(elem: Any) -> None:
        nonlocal offset
        for child in elem:
            tag = child.tag
            if tag == qn("w:r"):
                t = _run_text(child)
                parts.append(t)
                offset += len(t)
            elif tag == qn("w:ins"):
                author = child.get(qn("w:author"), "")
                date = child.get(qn("w:date"))
                # Collect all run text inside this insertion.
                ins_parts: list[str] = []
                for r in child.iter(qn("w:r")):
                    ins_parts.append(_run_text(r))
                ins_t = "".join(ins_parts)
                start = offset
                parts.append(ins_t)
                offset += len(ins_t)
                if ins_t:
                    tracked.append(
                        {
                            "change_type": "insertion",
                            "author": author,
                            "date": date,
                            "text": ins_t,
                            "char_span": (start, offset),
                        }
                    )
            elif tag == qn("w:del"):
                author = child.get(qn("w:author"), "")
                date = child.get(qn("w:date"))
                del_t = _del_text(child)
                if del_t:
                    tracked.append(
                        {
                            "change_type": "deletion",
                            "author": author,
                            "date": date,
                            "text": del_t,
                            "char_span": None,
                        }
                    )
            elif tag in (qn("w:hyperlink"), qn("w:smartTag"), qn("w:fldSimple")):
                # Recurse so that w:ins/w:del nested inside these wrappers
                # are captured correctly.
                _process(child)
            elif tag == qn("w:sdt"):
                # Structured document tag (content control): descend into
                # its w:sdtContent wrapper so runs inside are not dropped.
                content = child.find(qn("w:sdtContent"))
                if content is not None:
                    _process(content)

    _process(p_elem)
    return "".join(parts), tracked


def _run_text(r_elem: Any) -> str:
    parts: list[str] = []
    for child in r_elem:
        if child.tag == qn("w:t"):
            parts.append(child.text or "")
        elif child.tag == qn("w:tab"):
            parts.append("\t")
        elif child.tag == qn("w:br"):
            parts.append("\n")
    return "".join(parts)


def _del_text(del_elem: Any) -> str:
    parts: list[str] = []
    for r in del_elem.iter(qn("w:r")):
        for t in r.iter(qn("w:delText")):
            parts.append(t.text or "")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Heading / level detection
# ---------------------------------------------------------------------------

# Matches Word's built-in English heading style IDs: Heading1 … Heading9.
# Non-English or custom-named heading styles fall through to rule 3
# (numbered prefix detection).  See module docstring for details.
_HEADING_STYLE_RE = re.compile(r"[Hh]eading\s*(\d+)$")


def _para_level(p_elem: Any) -> int | None:
    """Return 1-based heading level for this paragraph, or None for body text."""
    ppr = p_elem.find(qn("w:pPr"))
    if ppr is None:
        return None

    # Rule 1: Heading style (style ID "Heading1", "Heading2", …)
    pstyle = ppr.find(qn("w:pStyle"))
    if pstyle is not None:
        style_id = pstyle.get(qn("w:val"), "")
        m = _HEADING_STYLE_RE.match(style_id)
        if m:
            return int(m.group(1))

    # Rule 2: Numbered list (w:numPr with non-zero numId)
    # numId=0 is OOXML's "remove numbering" sentinel — treated as body text.
    numpr = ppr.find(qn("w:numPr"))
    if numpr is not None:
        numid = numpr.find(qn("w:numId"))
        ilvl = numpr.find(qn("w:ilvl"))
        if numid is not None and ilvl is not None:
            num_id_val = numid.get(qn("w:val"), "0")
            if num_id_val != "0":
                ilvl_val = ilvl.get(qn("w:val"), "0")
                try:
                    return int(ilvl_val) + 1
                except ValueError:
                    pass
    return None


# ---------------------------------------------------------------------------
# Clause number extraction
# ---------------------------------------------------------------------------

# Rule 3: Matches "1.", "1.2.", "2.1)" at paragraph start. The trailing
# (?!\d) lookahead forbids a digit right after the terminator so a
# multi-part number like "10.1.3 Term" is not mis-split as "10.1" with a
# dangling "3 Term" heading, and decimals like "1.5 times" / "99.9%" /
# "10.00" are not mistaken for clause numbers.
_NUM_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)[.)](?!\d)\s*")


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

    # No explicit number — generate a sequential path under the current parent.
    clause_path = builder.generate_path(level)
    return clause_path, text.strip() or None


# ---------------------------------------------------------------------------
# Clause tree builder (stack-based)
# ---------------------------------------------------------------------------


class _ClauseBuilder:
    def __init__(self) -> None:
        self._root: list[ClauseNode] = []
        self._stack: list[tuple[int, ClauseNode]] = []  # (level, node)
        # counters[(parent_path, level)] = next sequential number
        self._counters: dict[tuple[str, int], int] = {}

    def current_path(self) -> str:
        return self._stack[-1][1].clause_path if self._stack else ""

    def generate_path(self, level: int) -> str:
        """Generate a sequential clause path for an unnumbered heading."""
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
        # Pop items at same or deeper level so the new node becomes a sibling
        # (not a child) of any open clause at this level.
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        if self._stack:
            self._stack[-1][1].children.append(node)
        else:
            self._root.append(node)
        self._stack.append((level, node))

    def add_body(self, text: str, doc_offset: int) -> None:
        """Append body text to the current clause.

        If no heading has been seen yet, creates a synthetic root node with
        ``clause_path="0"`` and ``heading=None`` at stack level 0 to collect
        pre-heading text. A genuine "0."-numbered clause (e.g. "0. Preamble",
        or a bare "0."/"0)" paragraph with no heading text of its own) can
        also produce ``clause_path="0"`` and ``heading=None``, so those two
        conditions are **necessary but not sufficient** to identify the
        synthetic node on their own — callers must not drop them, but must
        also layer an additional disambiguator on top. That disambiguator is
        structural: this synthetic node's ``char_span`` is set below to cover
        exactly its own first line of body text, whereas a genuine heading
        node's ``char_span`` covers only its heading line — a caller can
        narrow further, on top of ``clause_path == "0"`` and
        ``heading is None``, by checking whether
        ``char_span[1] - char_span[0] == len(text.split("\n", 1)[0])``.
        """
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
