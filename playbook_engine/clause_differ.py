"""Diff engine — L4 pipeline stage.

Computes clause-scoped text diffs across versions of one aligned document:

- **Consecutive** (v_i → v_{i+1}): one diff per adjacent pair; captures each
  negotiation move.
- **Net** (first → signed/last): captures the durable outcome of the negotiation.

Word-level ``SequenceMatcher`` is used for text comparison.  Unchanged spans
are never emitted in hunks (token discipline): a ``ClauseDiff`` with
``kind="unchanged"`` has an empty ``hunks`` tuple.

Output types are frozen dataclasses and therefore JSON-serialisable via
``.to_dict()`` methods.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

from playbook_engine.clause_aligner import ClauseAlignment

# ---------------------------------------------------------------------------
# Stop words (same set as rest of pipeline)
# ---------------------------------------------------------------------------

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

_HUNK_KINDS = frozenset({"insert", "delete", "replace"})
_DIFF_KINDS = frozenset({"unchanged", "added", "removed", "modified"})


@dataclass(frozen=True)
class TextHunk:
    """One changed span within a word-level diff.

    ``kind`` is one of ``"insert"``, ``"delete"``, or ``"replace"``.
    Equal spans are never emitted.
    """

    kind: str
    old_text: str  # empty string for "insert"
    new_text: str  # empty string for "delete"

    def __post_init__(self) -> None:
        if self.kind not in _HUNK_KINDS:
            raise ValueError(
                f"TextHunk.kind must be one of {sorted(_HUNK_KINDS)!r}; got {self.kind!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "old_text": self.old_text, "new_text": self.new_text}


@dataclass(frozen=True)
class ClauseDiff:
    """Diff of one logical clause between two versions.

    ``text_before`` / ``text_after`` hold the full clause text (empty when
    the clause is absent in that version).  They are stored for provenance and
    reversal detection; they do not violate the token-discipline rule, which
    forbids emitting **unchanged spans** inside ``hunks``.

    ``clause_version_before``/``clause_version_after`` and ``char_span_before``/
    ``char_span_after`` (issue #108) record, for each side of the diff, the
    actual version id (normalized-tree file stem) and the ``ClauseNode.char_span``
    the ``clause_path_before``/``clause_path_after`` values were read from —
    ``None`` when that side of the diff has no clause (added/removed).
    Downstream citation building (``observation_builder.build_observations``)
    needs these to resolve a citation to a real file, since ``clause_path``
    alone is only unique within one version's tree.
    """

    taxonomy_id: str | None
    clause_path_before: str | None  # None when the clause is newly added
    clause_path_after: str | None  # None when the clause is removed
    kind: str  # "unchanged", "added", "removed", "modified"
    hunks: tuple[TextHunk, ...]  # non-empty only for "modified"
    text_before: str
    text_after: str
    clause_version_before: str | None = None
    clause_version_after: str | None = None
    char_span_before: tuple[int, int] | None = None
    char_span_after: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.kind not in _DIFF_KINDS:
            raise ValueError(
                f"ClauseDiff.kind must be one of {sorted(_DIFF_KINDS)!r}; got {self.kind!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        # text_before/text_after intentionally excluded — reversal detection
        # operates on in-memory objects; the serialized form carries only the
        # change surface (hunks) to keep output compact.
        return {
            "taxonomy_id": self.taxonomy_id,
            "clause_path_before": self.clause_path_before,
            "clause_path_after": self.clause_path_after,
            "kind": self.kind,
            "hunks": [h.to_dict() for h in self.hunks],
            "clause_version_before": self.clause_version_before,
            "clause_version_after": self.clause_version_after,
            "char_span_before": list(self.char_span_before) if self.char_span_before else None,
            "char_span_after": list(self.char_span_after) if self.char_span_after else None,
        }


@dataclass(frozen=True)
class VersionDiff:
    """All clause-scoped diffs between two versions."""

    version_before: str
    version_after: str
    diffs: tuple[ClauseDiff, ...]

    def changed(self) -> tuple[ClauseDiff, ...]:
        """Return only non-unchanged diffs."""
        return tuple(d for d in self.diffs if d.kind != "unchanged")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_before": self.version_before,
            "version_after": self.version_after,
            "diffs": [d.to_dict() for d in self.diffs],
        }


@dataclass(frozen=True)
class DocumentDiff:
    """Complete diff record for one aligned document."""

    consecutive: tuple[VersionDiff, ...]  # v0→v1, v1→v2, …
    net: VersionDiff  # first → last (signed)
    version_order: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_order": list(self.version_order),
            "consecutive": [vd.to_dict() for vd in self.consecutive],
            "net": self.net.to_dict(),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diff_aligned(
    alignments: list[ClauseAlignment],
    version_order: list[str],
) -> DocumentDiff:
    """Compute consecutive and net diffs across aligned clause versions.

    Args:
        alignments:    Output of ``align_versions()`` — one ``ClauseAlignment``
                       per logical clause; slots parallel to ``version_order``.
        version_order: Version ids in order (oldest first, signed last).
                       Must have ≥ 2 entries.

    Returns:
        ``DocumentDiff`` with ``k-1`` consecutive diffs and one net diff
        (first → last).

    Raises:
        ValueError: if ``version_order`` has fewer than 2 entries.
    """
    if len(version_order) < 2:
        raise ValueError(f"diff_aligned requires ≥ 2 versions; got {len(version_order)}")

    consecutive = tuple(
        _version_diff(alignments, version_order[i], version_order[i + 1])
        for i in range(len(version_order) - 1)
    )

    net = _version_diff(alignments, version_order[0], version_order[-1])

    return DocumentDiff(
        consecutive=consecutive,
        net=net,
        version_order=tuple(version_order),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _version_diff(
    alignments: list[ClauseAlignment],
    v_before: str,
    v_after: str,
) -> VersionDiff:
    diffs: list[ClauseDiff] = []

    for alignment in alignments:
        before_slot = next((s for s in alignment.slots if s.version == v_before), None)
        after_slot = next((s for s in alignment.slots if s.version == v_after), None)

        before_clause = before_slot.clause if before_slot else None
        after_clause = after_slot.clause if after_slot else None

        if before_clause is None and after_clause is None:
            continue  # clause not relevant to this version pair

        before_path = before_clause.node.clause_path if before_clause else None
        after_path = after_clause.node.clause_path if after_clause else None
        before_text = (before_clause.node.text or "") if before_clause else ""
        after_text = (after_clause.node.text or "") if after_clause else ""

        if before_clause is None:
            kind: str = "added"
            hunks: tuple[TextHunk, ...] = ()
        elif after_clause is None:
            kind = "removed"
            hunks = ()
        elif before_text == after_text:
            kind = "unchanged"
            hunks = ()
        else:
            kind = "modified"
            hunks = _word_diff(before_text, after_text)

        diffs.append(
            ClauseDiff(
                taxonomy_id=alignment.taxonomy_id,
                clause_path_before=before_path,
                clause_path_after=after_path,
                kind=kind,
                hunks=hunks,
                text_before=before_text,
                text_after=after_text,
                clause_version_before=v_before if before_clause else None,
                clause_version_after=v_after if after_clause else None,
                char_span_before=before_clause.node.char_span if before_clause else None,
                char_span_after=after_clause.node.char_span if after_clause else None,
            )
        )

    return VersionDiff(
        version_before=v_before,
        version_after=v_after,
        diffs=tuple(diffs),
    )


def _tokenize(text: str) -> list[str]:
    """Split text into word tokens (punctuation stripped, lowercased)."""
    return re.findall(r"\w+", text.lower())


def _word_diff(before: str, after: str) -> tuple[TextHunk, ...]:
    """Word-level diff; equal spans are never emitted."""
    bw = _tokenize(before)
    aw = _tokenize(after)

    matcher = difflib.SequenceMatcher(None, bw, aw, autojunk=False)
    hunks: list[TextHunk] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue  # token discipline: never emit unchanged spans
        old_text = " ".join(bw[i1:i2])
        new_text = " ".join(aw[j1:j2])
        hunks.append(TextHunk(kind=op, old_text=old_text, new_text=new_text))

    return tuple(hunks)
