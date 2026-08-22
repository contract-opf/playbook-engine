"""Tests for the DOCX/docling tracked-change coordinate-space bridge (issue #118).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional author names ('Alice',
'Bob') and party names only.

The synthetic-Markdown fixtures below feed the REAL, already-shipping
``extraction._parse_markdown_lines``/``extraction._build_stream`` — the
actual docling-Markdown transform — so these tests pin real behavior rather
than a hand-rolled guess at what docling's parser does. No docling install
and no private corpus data is needed to run or reproduce these.
"""

from __future__ import annotations

from playbook_engine import extraction
from playbook_engine.docx_ingester import TextUnit, TrackedChange, TrackedChanges, ingest_docx
from playbook_engine.extraction import (
    bridge_tracked_change_spans,
    extract_docx_units_and_tracked_changes,
)
from playbook_engine.segmentation_grounding import Block

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _units(texts: list[str]) -> list[TextUnit]:
    """Build a docx_ingester-shaped unit stream: kept units joined by "\n",
    same accumulation convention as ``ingest_docx``'s own ``doc_char_offset``.
    """
    units: list[TextUnit] = []
    offset = 0
    for t in texts:
        units.append(TextUnit(text=t, char_span=(offset, offset + len(t))))
        offset += len(t) + 1
    return units


def _tc(
    text: str,
    char_span: tuple[int, int] | None,
    author: str = "Alice",
    change_type: str = "insertion",
) -> TrackedChange:
    return TrackedChange(
        change_type=change_type,  # type: ignore[arg-type]
        author=author,
        date="2024-01-01",
        text=text,
        clause_path="1",
        char_span=char_span,
    )


# ---------------------------------------------------------------------------
# _normalize_unit_for_alignment
# ---------------------------------------------------------------------------


def test_normalize_strips_single_level_numbering_and_case() -> None:
    assert extraction._normalize_unit_for_alignment("1. Definitions") == "definitions"


def test_normalize_strips_multi_level_dotted_numbering() -> None:
    """docx_ingester keeps dotted multi-level clause numbers ("1.2") verbatim
    in its own unit text — docling's own _LIST_ITEM_RE never strips these
    (single-level "\\d+[.)]" only), so this module's OWN normalization must
    handle them, not merely mirror docling's regex."""
    assert extraction._normalize_unit_for_alignment("1.2 Termination") == "termination"


def test_normalize_bullet_marker_stripped() -> None:
    assert extraction._normalize_unit_for_alignment("- Confidential information") == (
        "confidential information"
    )


def test_normalize_converges_stripped_and_unstripped_forms() -> None:
    """The whole point: a docx_ingester unit that KEPT its numbering and a
    docling block that had ITS numbering stripped upstream must normalize
    to the identical string."""
    assert extraction._normalize_unit_for_alignment(
        "3. Termination"
    ) == extraction._normalize_unit_for_alignment("Termination")


# ---------------------------------------------------------------------------
# _align_units_to_blocks — order-preserving, monotonic across repeated text
# ---------------------------------------------------------------------------


def test_align_units_simple_one_to_one() -> None:
    units = _units(["1. Definitions", "Some body text.", "2. Term", "More body text."])
    blocks = [
        Block(block_id="b0", page=0, char_span=(0, 11), text="Definitions"),
        Block(block_id="b1", page=0, char_span=(12, 27), text="Some body text."),
        Block(block_id="b2", page=0, char_span=(28, 32), text="Term"),
        Block(block_id="b3", page=0, char_span=(33, 48), text="More body text."),
    ]
    mapping = extraction._align_units_to_blocks(units, blocks)
    assert mapping == {0: 0, 1: 1, 2: 2, 3: 3}


def test_align_units_monotonic_across_repeated_boilerplate() -> None:
    """A phrase repeated at two different positions on BOTH sides must align
    positionally (Nth occurrence to Nth occurrence), never both to the
    first — this is what makes the bridge robust to boilerplate, unlike
    plain text-containment search (issue #118's measured comparison)."""
    boiler = "Confidentiality obligations survive termination."
    units = _units(
        [
            boiler,
            "1. First clause unique text goes here.",
            boiler,
            "2. Second clause other text follows.",
        ]
    )
    block_texts = [
        boiler,
        "First clause unique text goes here.",
        boiler,
        "Second clause other text follows.",
    ]
    blocks = []
    offset = 0
    for i, t in enumerate(block_texts):
        blocks.append(Block(block_id=f"b{i}", page=0, char_span=(offset, offset + len(t)), text=t))
        offset += len(t) + 1

    mapping = extraction._align_units_to_blocks(units, blocks)
    assert mapping == {0: 0, 1: 1, 2: 2, 3: 3}, (
        "each boilerplate occurrence must align to its OWN positional counterpart"
    )


# ---------------------------------------------------------------------------
# bridge_tracked_change_spans — synthetic fixture pinning the ACTUAL docling
# markdown transform (required verification #2, issue #118): the docling
# side below is produced by the real extraction._parse_markdown_lines /
# _build_stream, not a hand-rolled stand-in.
# ---------------------------------------------------------------------------

_DOCLING_MARKDOWN = (
    "# Master Services Agreement\n"
    "\n"
    "1. Definitions\n"
    "\n"
    "The following terms shall apply throughout this Agreement.\n"
    "\n"
    "2. Payment Terms\n"
    "\n"
    "Payment is due within thirty (30) days of invoice.\n"
    "\n"
    "3. Termination\n"
    "\n"
    "Either party may terminate this Agreement upon thirty (30) days written notice.\n"
)

# docx_ingester's own units for the "same" logical document — numbering kept
# verbatim (docx_ingester never strips a clause number from its unit text;
# only ClauseNode.heading has it stripped, via _parse_clause_number).
_DOCX_UNIT_TEXTS = [
    "Master Services Agreement",
    "1. Definitions",
    "The following terms shall apply throughout this Agreement.",
    "2. Payment Terms",
    "Payment is due within thirty (30) days of invoice.",
    "3. Termination",
    "Either party may terminate this Agreement upon thirty (30) days written notice.",
]


def _docling_blocks() -> tuple[str, list[Block]]:
    lines = extraction._parse_markdown_lines(_DOCLING_MARKDOWN)
    return extraction._build_stream(lines)


def test_docling_markdown_transform_strips_numbering_from_list_items() -> None:
    """Pins the real transform this whole bridge depends on: docling renders
    a Word numbered-heading paragraph as a Markdown ordered-list line, and
    _parse_markdown_lines strips that numbering — exactly the divergence
    from docx_ingester's own (numbering-preserving) unit text issue #118
    measured on the real corpus."""
    canonical_text, blocks = _docling_blocks()
    texts = [b.text for b in blocks]
    assert texts == [
        "Master Services Agreement",
        "Definitions",
        "The following terms shall apply throughout this Agreement.",
        "Payment Terms",
        "Payment is due within thirty (30) days of invoice.",
        "Termination",
        "Either party may terminate this Agreement upon thirty (30) days written notice.",
    ]
    for b in blocks:
        assert canonical_text[b.char_span[0] : b.char_span[1]] == b.text


def test_bridge_translates_span_exactly_through_real_docling_transform() -> None:
    """End-to-end: a TrackedChange offset into docx_ingester's own
    (numbering-preserving) coordinate space is translated into the REAL
    docling-derived canonical_text's coordinate space, landing on the exact
    same substring."""
    canonical_text, blocks = _docling_blocks()
    units = _units(_DOCX_UNIT_TEXTS)

    payment_unit = units[4]
    assert payment_unit.text == "Payment is due within thirty (30) days of invoice."
    substr = "thirty (30) days"
    local_start = payment_unit.text.index(substr)
    change_span = (
        payment_unit.char_span[0] + local_start,
        payment_unit.char_span[0] + local_start + len(substr),
    )
    tc = _tc(substr, change_span)
    tracked = TrackedChanges(document_id="doc", version="v2", changes=[tc])

    bridged = bridge_tracked_change_spans(tracked, units, blocks)

    assert len(bridged.changes) == 1
    bridged_span = bridged.changes[0].char_span
    assert bridged_span is not None
    assert canonical_text[bridged_span[0] : bridged_span[1]] == substr
    # Sanity: this is a REAL coordinate-space change, not a no-op — the raw
    # (untranslated) span would land on the wrong text in canonical_text.
    assert canonical_text[change_span[0] : change_span[1]] != substr


def test_bridge_preserves_author_date_clause_path_unchanged() -> None:
    canonical_text, blocks = _docling_blocks()
    units = _units(_DOCX_UNIT_TEXTS)
    tc = _tc("Termination", units[5].char_span, author="Priya", change_type="insertion")
    tracked = TrackedChanges(document_id="doc", version="v3", changes=[tc])

    bridged = bridge_tracked_change_spans(tracked, units, blocks)

    out = bridged.changes[0]
    assert out.author == "Priya"
    assert out.date == "2024-01-01"
    assert out.clause_path == "1"
    assert out.change_type == "insertion"


# ---------------------------------------------------------------------------
# bridge_tracked_change_spans — unaligned units never get an invented span
# ---------------------------------------------------------------------------


def _blocks_from_texts(texts: list[str]) -> list[Block]:
    blocks: list[Block] = []
    offset = 0
    for i, t in enumerate(texts):
        blocks.append(Block(block_id=f"b{i}", page=0, char_span=(offset, offset + len(t)), text=t))
        offset += len(t) + 1
    return blocks


def test_bridge_unaligned_unit_bracketed_by_aligned_neighbors_stays_none() -> None:
    """A running-header artifact docx_ingester keeps but docling drops
    entirely: its own unit has no counterpart, even though it is bracketed
    by aligned units on both sides. Issue #118 measured a bounding-span
    interpolation tier here against the real corpus and found it net-
    negative on correct-clause attribution (13 correct vs. 25 newly
    WRONG-clause attributions) — that tier was removed, so a bracketed but
    unaligned unit now resolves to ``None`` exactly like an unbracketed one,
    refusing rather than guessing."""
    units = _units(
        [
            "1. First Clause",
            "Alpha text for clause one.",
            "PAGE 2 OF 10",  # dropped entirely by docling
            "2. Second Clause",
            "Beta text for clause two.",
        ]
    )
    blocks = _blocks_from_texts(
        [
            "First Clause",
            "Alpha text for clause one.",
            "Second Clause",
            "Beta text for clause two.",
        ]
    )
    tc = _tc("PAGE 2 OF 10", units[2].char_span)
    tracked = TrackedChanges(document_id="doc", version="v2", changes=[tc])

    bridged = bridge_tracked_change_spans(tracked, units, blocks)

    assert bridged.changes[0].char_span is None


def test_bridge_unaligned_unit_at_start_has_no_prev_neighbor_stays_none() -> None:
    """Not bracketed on BOTH sides (nothing before it) — must not
    extrapolate; refuses rather than guesses."""
    units = _units(["STRAY LEADING TEXT", "1. Real Clause", "Body text here."])
    blocks = _blocks_from_texts(["Real Clause", "Body text here."])
    tc = _tc("STRAY LEADING TEXT", units[0].char_span)
    tracked = TrackedChanges(document_id="doc", version="v2", changes=[tc])

    bridged = bridge_tracked_change_spans(tracked, units, blocks)

    assert bridged.changes[0].char_span is None


def test_bridge_unaligned_unit_at_end_has_no_next_neighbor_stays_none() -> None:
    units = _units(["1. Real Clause", "Body text here.", "STRAY TRAILING TEXT"])
    blocks = _blocks_from_texts(["Real Clause", "Body text here."])
    tc = _tc("STRAY TRAILING TEXT", units[2].char_span)
    tracked = TrackedChanges(document_id="doc", version="v2", changes=[tc])

    bridged = bridge_tracked_change_spans(tracked, units, blocks)

    assert bridged.changes[0].char_span is None


# ---------------------------------------------------------------------------
# bridge_tracked_change_spans — degenerate inputs
# ---------------------------------------------------------------------------


def test_bridge_deletion_char_span_none_passes_through_unchanged() -> None:
    """Deletions always carry char_span=None (docx_ingester's contract) —
    nothing to translate, must not crash and must not invent a span."""
    units = _units(["1. Clause", "Body text."])
    blocks = _blocks_from_texts(["Clause", "Body text."])
    tc = _tc("removed phrase", None, change_type="deletion")
    tracked = TrackedChanges(document_id="doc", version="v2", changes=[tc])

    bridged = bridge_tracked_change_spans(tracked, units, blocks)

    assert bridged.changes[0].char_span is None


def test_bridge_empty_tracked_changes_returns_unchanged() -> None:
    units = _units(["1. Clause"])
    blocks = _blocks_from_texts(["Clause"])
    tracked = TrackedChanges(document_id="doc", version="v2", changes=[])

    bridged = bridge_tracked_change_spans(tracked, units, blocks)

    assert bridged is tracked


def test_bridge_empty_blocks_leaves_every_span_none() -> None:
    """No target blocks at all (e.g. an empty extraction) — safe degrade to
    None across the board, never a stale/coincidental raw span."""
    units = _units(["1. Clause", "Body text."])
    tc = _tc("Body text.", units[1].char_span)
    tracked = TrackedChanges(document_id="doc", version="v2", changes=[tc])

    bridged = bridge_tracked_change_spans(tracked, units, [])

    assert bridged.changes[0].char_span is None


# ---------------------------------------------------------------------------
# extract_docx_units_and_tracked_changes
# ---------------------------------------------------------------------------


def test_extract_docx_units_and_tracked_changes_matches_ingest_docx(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from tests.test_docx_ingester import _tracked_docx

    path = _tracked_docx(tmp_path)
    expected = ingest_docx(path, document_id="", version="")

    result = extract_docx_units_and_tracked_changes(path)

    assert result is not None
    units, tracked = result
    assert tracked.to_dict() == expected.tracked.to_dict()
    assert [(u.text, u.char_span) for u in units] == [(u.text, u.char_span) for u in expected.units]
    assert units, "a real tracked-changes DOCX must yield a non-empty unit stream"


def test_extract_docx_units_and_tracked_changes_none_for_non_docx(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "agreement.rtf"
    path.write_text(r"{\rtf1\ansi\deff0 Hello\par}", encoding="utf-8")
    assert extract_docx_units_and_tracked_changes(path) is None
