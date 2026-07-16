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

    # Restrict candidates to the clause paths relevant to this diff.
    relevant_paths = {
        p for p in (clause_diff.clause_path_before, clause_diff.clause_path_after) if p is not None
    }
    candidates = [c for c in tracked_changes.changes if c.clause_path in relevant_paths]

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
