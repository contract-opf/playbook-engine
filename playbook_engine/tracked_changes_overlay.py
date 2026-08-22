"""Tracked-changes enrichment overlay — L4 pipeline stage.

Attaches Word-redline metadata (author, date, change type) to ``TextHunk``
objects where a matching ``TrackedChange`` exists in the DOCX side-channel.

This is a *bonus signal*: when a DOCX file carries ``w:ins``/``w:del``
elements, the overlay can attribute each diff hunk to a specific author with
a specific intent.  It degrades silently for PDFs, clean DOCX files, and any
version pair where no tracked-changes data was captured.

Matching strategy:
  - ``"insert"`` / ``"replace"`` hunks are matched against tracked insertions
    (``change_type="insertion"``) by word-level Jaccard similarity on
    ``hunk.new_text`` vs ``TrackedChange.text``.
  - ``"delete"`` / ``"replace"`` hunks also check for tracked deletions on
    ``hunk.old_text``.  For ``"replace"`` hunks the insertion side wins if
    both sides match (authorship of the new text is more useful downstream).
  - Minimum similarity threshold ``_MATCH_THRESHOLD = 0.50`` to accept a match.
  - Greedy (first-fit within each hunk); each ``TrackedChange`` matched at
    most once to avoid double-attribution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from playbook_engine.clause_differ import ClauseDiff, TextHunk
from playbook_engine.docx_ingester import TrackedChange, TrackedChanges

_MATCH_THRESHOLD: float = 0.50

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HunkEnrichment:
    """Author and intent metadata from a Word tracked-change record."""

    author: str
    date: str | None
    tracked_type: str  # "insertion" or "deletion"

    def to_dict(self) -> dict[str, Any]:
        return {"author": self.author, "date": self.date, "tracked_type": self.tracked_type}


@dataclass(frozen=True)
class EnrichedHunk:
    """A ``TextHunk`` paired with optional tracked-change enrichment.

    ``enrichment`` is ``None`` when no matching ``TrackedChange`` was found
    (PDFs, clean DOCX, or text-diff hunk with no corresponding redline).
    """

    hunk: TextHunk
    enrichment: HunkEnrichment | None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich_clause_diff(
    clause_diff: ClauseDiff,
    tracked_changes: TrackedChanges | None,
) -> list[EnrichedHunk]:
    """Attach tracked-change author/intent to each hunk in *clause_diff*.

    Args:
        clause_diff:      One ``ClauseDiff`` from the diff engine.
        tracked_changes:  Side-channel data from ``ingest_docx()``, or
                         ``None`` / empty when not available (PDF, clean DOCX).

    Returns:
        One ``EnrichedHunk`` per hunk in ``clause_diff.hunks``, in the same
        order.  ``EnrichedHunk.enrichment`` is ``None`` for any hunk that
        could not be matched to a tracked change.
    """
    if not clause_diff.hunks:
        return []

    if not tracked_changes or not tracked_changes.changes:
        return [EnrichedHunk(hunk=h, enrichment=None) for h in clause_diff.hunks]

    # Restrict candidates to this diff's clause via clause-path equality OR
    # char-offset span overlap (issue #112) — additive, not a replacement.
    # ``clause_path`` alone is unreliable across mismatched numbering
    # schemes: an agent/LLM segmenter's flat integer paths ('1', '2') share
    # no namespace with the docx-ingester's hierarchical paths ('0.4.1')
    # even when they denote the same clause, so exact clause-path equality
    # silently drops every candidate whenever the two sides of a diff came
    # from different segmenters.
    #
    # PRECONDITION span overlap depends on but this function cannot check:
    # both ``char_span`` values must be offsets into the SAME normalized
    # text. That is set by which *extractor* produced ``clause_diff``'s
    # tree, not by which segmenter grouped it:
    #   - Legacy DOCX adapter (``extraction._extract_docx_lines`` /
    #     ``_build_stream``) reuses ``docx_ingester._iter_body_blocks`` /
    #     ``_extract_para_text`` and joins with ``"\n"`` — byte-identical
    #     to ``docx_ingester``'s own normalized text. Here
    #     ``ClauseNode.char_span`` and ``TrackedChange.char_span`` (always
    #     produced by a fresh ``ingest_docx`` re-parse — see
    #     ``extraction.extract_tracked_changes``) share one coordinate
    #     system, so overlap means real co-location.
    #   - docling adapter — the DEFAULT under ``extraction.extractor=
    #     "auto"`` (``config.py``), which prefers docling for DOCX: the
    #     tree's ``char_span`` values are offsets into docling's
    #     Markdown-derived text instead (``_parse_markdown_lines`` strips
    #     heading/emphasis/list decoration and emits one block per table
    #     row) — a different string from ``docx_ingester``'s paragraph
    #     join. For redline DOCX specifically — the files that actually
    #     carry ``w:ins``/``w:del`` — docling 2.x crashes and
    #     ``_retry_docling_on_normalized_docx`` re-extracts from a THIRD,
    #     differently-normalized copy. On this (default) path, two spans
    #     comparing equal is numeric coincidence, not co-location; where
    #     it fires, the Jaccard confirmation in ``_match_hunk`` can still
    #     accept the wrong clause for repeated boilerplate text.
    # ``clause_diff`` carries no extractor label, so this function cannot
    # gate on the precondition above — span overlap is applied
    # unconditionally, reliable at recovering attribution on the
    # legacy-adapter path and best-effort (may spuriously match or
    # spuriously miss) on the default docling path. See issue #112 review
    # notes.
    #
    # Span overlap must also stay additive rather than replace the
    # clause-path filter even where the coordinate systems do match:
    # ``ClauseNode.char_span`` for a docx-ingested heading node covers
    # only its heading *line*, not the clause body (see
    # ``docx_ingester._ClauseBuilder.add_body``'s docstring) — a real
    # tracked change in the body of a same-namespace docx-to-docx diff
    # will not overlap that narrow span even though the clause path
    # matches exactly. Keeping clause-path matching alongside span overlap
    # preserves that same-namespace case; span overlap independently
    # recovers the mismatched-namespace case the old filter dropped
    # entirely (an agent-segmented clause's char_span spans its whole
    # block range — see ``segmentation_grounding.py`` — so it does overlap
    # a tracked change anywhere in that clause's body, when the coordinate
    # systems agree).
    #
    # A ``TrackedChange`` with ``char_span=None`` (always true for
    # deletions — deleted text is absent from the normalized text) can
    # only be selected via the clause-path branch.
    relevant_paths = {
        p for p in (clause_diff.clause_path_before, clause_diff.clause_path_after) if p is not None
    }
    diff_spans = [
        s for s in (clause_diff.char_span_before, clause_diff.char_span_after) if s is not None
    ]
    candidates = [
        c
        for c in tracked_changes.changes
        if c.clause_path in relevant_paths
        or (c.char_span is not None and any(_spans_overlap(c.char_span, s) for s in diff_spans))
    ]

    if not candidates:
        return [EnrichedHunk(hunk=h, enrichment=None) for h in clause_diff.hunks]

    used: set[int] = set()
    return [
        EnrichedHunk(hunk=h, enrichment=_match_hunk(h, candidates, used)) for h in clause_diff.hunks
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _match_hunk(
    hunk: TextHunk,
    candidates: list[TrackedChange],
    used: set[int],
) -> HunkEnrichment | None:
    """Find the best unused candidate for *hunk*, or None if below threshold."""
    # For "replace": prefer insertion side (authorship of new text).
    sides: list[tuple[str, str]] = []  # [(change_type, target_text)]
    if hunk.kind in ("insert", "replace"):
        sides.append(("insertion", hunk.new_text))
    if hunk.kind in ("delete", "replace"):
        sides.append(("deletion", hunk.old_text))
    # For "replace": insertion side is appended first → wins Jaccard ties.

    best_idx = -1
    best_sim = 0.0

    for c_type, target_text in sides:
        toks = _tokens(target_text)
        for i, candidate in enumerate(candidates):
            if i in used:
                continue
            if candidate.change_type != c_type:
                continue
            sim = _jaccard(toks, _tokens(candidate.text))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

    if best_idx >= 0 and best_sim >= _MATCH_THRESHOLD:
        used.add(best_idx)
        c = candidates[best_idx]
        return HunkEnrichment(author=c.author, date=c.date, tracked_type=c.change_type)

    return None


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if half-open char-offset spans *a* and *b* intersect.

    Both ``TrackedChange.char_span`` and ``ClauseDiff.char_span_before`` /
    ``char_span_after`` use ``[start, end)`` (Python-slice) convention, so
    plain interval intersection applies.  Handles partial overlap and full
    containment identically — either side may be the larger interval.
    """
    return a[0] < b[1] and b[0] < a[1]


def _normalize(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _normalize(text).split() if w not in _STOP_WORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
