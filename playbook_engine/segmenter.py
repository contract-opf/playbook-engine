"""Segmenter — post-ingestion pass that refines a ClauseTree.

The ingesters (DOCX, PDF, RTF) parse the gross structure of an agreement
into a clause tree keyed by heading hierarchy.  Body text under a heading
often contains finer-grained sub-clauses that the heading-level parse misses
because they are inline (``(a)``, ``(b)`` … lettered items or ``(i)`` …
roman-numeral items).  This module promotes those inline sub-clauses to
``ClauseNode`` children so that downstream stages (taxonomy classification,
diff, playbook compilation) operate at clause level rather than at
paragraph level.

Segmentation is fully deterministic — no LLM is involved.

Segmentation rules (applied in order to each node's body text):
  1. Split body text on lines whose stripped content starts with ``(a)``
     through ``(z)`` — single-letter parenthesised markers (common in
     US legal agreements).
  2. Split within lettered items on lines starting with ``(i)`` through
     ``(viii)`` — short roman-numeral markers (sub-items under lettered
     items).  Note: single-letter ``(i)`` is ambiguous; it is treated as
     a roman-numeral sub-item only when it appears inside a lettered
     section (not at top-level body text).
  3. Any body text that precedes the first lettered item is kept in the
     parent node as its preamble text.

char_span for promoted nodes:
  The parent node's ``char_span`` covers its heading line.  Body text
  immediately follows in the virtual normalized text (heading_end + 1
  accounts for the ``"\\n"`` separator).  Each promoted child inherits
  a char_span computed as:
  ``(heading_end + 1 + start_offset_in_body, heading_end + 1 + end_offset)``
  where offsets are byte offsets within the parent's ``.text`` field.

clause_path for promoted nodes:
  Parent path ``"3.2"`` → lettered children ``"3.2.a"``, ``"3.2.b"`` …
  Roman-numeral sub-items inside ``"3.2.a"`` → ``"3.2.a.i"``, ``"3.2.a.ii"`` …

Immutability:
  ``segment()`` never mutates the input tree.  It returns a fresh
  ``ClauseTree`` with new ``ClauseNode`` objects at every level.

Idempotence:
  ``segment(segment(tree))`` produces the same result as ``segment(tree)``
  because the inline markers are consumed in the first pass.
"""

from __future__ import annotations

import re

from playbook_engine.clause_tree import ClauseNode, ClauseTree

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches "(a)" … "(z)" at start of a stripped line.
_LETTERED = re.compile(r"^\(([a-z])\)\s*")

# Matches "(i)" … "(viii)" at start of a stripped line.
# Limited to the most common roman numerals; longer ones are unlikely in
# sub-items and could be confused with lettered items.
_ROMAN_NUMERALS = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii"}
_ROMAN = re.compile(r"^\(([ivxlc]+)\)\s*")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment(tree: ClauseTree) -> ClauseTree:
    """Return a new ClauseTree with inline sub-clauses promoted to children.

    The input tree is not mutated.
    """
    new_nodes = [_segment_node(node) for node in tree.nodes]
    return ClauseTree(
        document_id=tree.document_id,
        version=tree.version,
        source_file=tree.source_file,
        nodes=new_nodes,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _segment_node(node: ClauseNode) -> ClauseNode:
    """Return a new ClauseNode with inline sub-clauses promoted to children."""
    # Recurse into existing children first.
    segmented_children = [_segment_node(c) for c in node.children]

    # Only look for inline items when there is body text.
    if not node.text.strip():
        return ClauseNode(
            clause_path=node.clause_path,
            heading=node.heading,
            text=node.text,
            char_span=node.char_span,
            children=segmented_children,
        )

    # The body text starts at heading_end + 1 in the virtual normalized text.
    body_start = node.char_span[1] + 1

    # Split on lettered items.
    lettered = _split_lettered(node.text)

    if len(lettered) <= 1:
        # No lettered items found — return node unchanged (except re-created).
        return ClauseNode(
            clause_path=node.clause_path,
            heading=node.heading,
            text=node.text,
            char_span=node.char_span,
            children=segmented_children,
        )

    # lettered[0] = (None, preamble_text, start_offset_in_body)
    # lettered[1:] = (letter, text, start_offset_in_body)
    preamble_marker, preamble_text, _preamble_start = lettered[0]
    assert preamble_marker is None

    # B3 fix: skip any promoted path that already exists in ingester children.
    existing_paths = {c.clause_path for c in segmented_children}

    promoted: list[ClauseNode] = []
    for marker, item_text, item_body_offset in lettered[1:]:
        assert marker is not None
        child_path = f"{node.clause_path}.{marker}"

        # B3: do not duplicate a child path from the ingester.
        if child_path in existing_paths:
            continue

        item_start = body_start + item_body_offset
        item_end = item_start + len(item_text)

        # Check for roman-numeral sub-items within this lettered item.
        roman_items = _split_roman(item_text)
        if len(roman_items) > 1:
            roman_preamble_marker, roman_preamble_text, _ = roman_items[0]
            assert roman_preamble_marker is None
            sub_children: list[ClauseNode] = []
            for roman_marker, roman_text, roman_offset in roman_items[1:]:
                assert roman_marker is not None
                roman_start = item_start + roman_offset
                roman_end = roman_start + len(roman_text)
                sub_children.append(
                    ClauseNode(
                        clause_path=f"{child_path}.{roman_marker}",
                        heading=None,
                        text=roman_text,
                        char_span=(roman_start, roman_end),
                    )
                )
            # B2 fix: lettered parent span covers its preamble only, not all
            # sub-items.  This is consistent with the heading convention used
            # by the DOCX/PDF ingesters (heading span ≠ full clause extent).
            preamble_end = item_start + len(roman_preamble_text)
            promoted.append(
                ClauseNode(
                    clause_path=child_path,
                    heading=None,
                    text=roman_preamble_text,
                    char_span=(item_start, preamble_end),
                    children=sub_children,
                )
            )
        else:
            promoted.append(
                ClauseNode(
                    clause_path=child_path,
                    heading=None,
                    text=item_text,
                    char_span=(item_start, item_end),
                )
            )

    return ClauseNode(
        clause_path=node.clause_path,
        heading=node.heading,
        text=preamble_text,
        char_span=node.char_span,
        children=segmented_children + promoted,
    )


def _split_lettered(text: str) -> list[tuple[str | None, str, int]]:
    """Split body text on lines starting with '(a)' … '(z)'.

    Returns a list of (marker, text, start_offset_in_text) tuples.
    The first entry always has marker=None (preamble before first item).
    """
    return _split_by_pattern(text, _LETTERED, roman_mode=False)


def _split_roman(text: str) -> list[tuple[str | None, str, int]]:
    """Split body text on lines starting with '(i)' … '(viii)'.

    Returns a list of (marker, text, start_offset_in_text) tuples.
    The first entry always has marker=None (preamble before first roman item).
    Only recognised roman numerals (up to viii) are split; others are body.
    """
    return _split_by_pattern(text, _ROMAN, roman_mode=True)


def _split_by_pattern(
    text: str,
    pattern: re.Pattern[str],
    roman_mode: bool,
) -> list[tuple[str | None, str, int]]:
    """Generic line-by-line splitter.

    Returns [(marker | None, accumulated_text, start_offset_in_text)].
    ``start_offset_in_text`` is the byte offset of this item's first
    character within ``text``.

    Lettered mode (roman_mode=False): only accepts a new lettered marker
    when it is the next sequential letter after the previous one (e.g.,
    after "(a)" the next accepted marker is "(b)").  This prevents
    "(i)" from being misread as a lettered item when it is a roman-numeral
    sub-item inside a "(a)" block.  Items that are not sequentially next
    are appended as body text to the current item.

    Roman mode (roman_mode=True): accepts any recognised roman numeral.
    """
    lines = text.split("\n")
    items: list[tuple[str | None, list[str], int]] = [(None, [], 0)]
    offset = 0
    last_letter: str | None = None  # for sequential-order enforcement

    for line in lines:
        stripped = line.strip()
        m = pattern.match(stripped)
        if m:
            marker = m.group(1)
            accept = False
            if roman_mode:
                accept = marker in _ROMAN_NUMERALS
            else:
                # Lettered mode: lettered sequences always start from "(a)".
                # Require "(a)" as the first item so that standalone "(i)" or
                # "(v)" at the top of body text is not mistaken for a lettered
                # item (those are roman numerals used as sub-items elsewhere).
                if last_letter is None:
                    accept = marker == "a"
                else:
                    accept = ord(marker) == ord(last_letter) + 1
            if accept:
                last_letter = marker if not roman_mode else last_letter
                remainder = stripped[m.end() :].strip()
                # B1 fix: store offset of the CONTENT (after the marker),
                # not the line start.  `line` has no leading spaces because
                # ingester body text is already stripped; stripped == line.
                content_offset = offset + m.end()
                items.append((marker, [remainder], content_offset))
            else:
                items[-1][1].append(line)
        else:
            items[-1][1].append(line)
        offset += len(line) + 1  # +1 for the "\n" separator

    return [(marker, "\n".join(lines_list).strip(), start) for marker, lines_list, start in items]
