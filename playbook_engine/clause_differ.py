"""Diff engine — L4 pipeline stage.

Computes clause-scoped text diffs across versions of one aligned document:

- **Consecutive** (v_i → v_{i+1}): one diff per adjacent pair; captures each
  negotiation move.
- **Net** (first → signed/last): captures the durable outcome of the negotiation.

Two sides that differ only in leading clause numbering, quote style or
whitespace, or where one side is only a bare section heading, diff as
``"unchanged"``: when two versions were extracted differently, those are
partition artifacts, not negotiated changes (issue #122).

A clause the aligner left on two rows has the same shape one level up: a
heading split off its body is an ``added`` row on one side and a ``removed``
row on the other, and the same words relocated between rows are a removal plus
an addition. The *consecutive* diffs relabel those rows ``"unchanged"`` as well
(:func:`_reconcile_partitions`), keeping every row in place. The *net* diff is
never relabeled this way: ``build_observations`` needs exactly one net row per
terminal node.

Word-level ``SequenceMatcher`` is used for text comparison.  Unchanged spans
are never emitted in hunks (token discipline): a ``ClauseDiff`` with
``kind="unchanged"`` has an empty ``hunks`` tuple.

Output types are frozen dataclasses and therefore JSON-serialisable via
``.to_dict()`` methods.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field, replace
from typing import Any

from playbook_engine.clause_aligner import ClauseAlignment
from playbook_engine.resegmentation import looks_like_heading

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
    # The inserted side exactly as the author wrote it (case, punctuation and
    # inner spacing kept), sliced out of the "after" text. ``new_text`` is the
    # lowercased ``\w+`` token stream the matcher compared; anything that
    # quotes a proposal back to a reader must use this. Not serialised
    # (``to_dict`` is unchanged); "" when the hunk has no inserted side.
    new_verbatim: str = field(default="", compare=False)

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

    A consecutive diff can carry an ``"unchanged"`` row with text on one side
    only (issue #122): a row :func:`_reconcile_partitions` found to be one
    clause partitioned differently by the two extractions (a heading split off
    its body, or the same words relocated to another row). Its absent side
    keeps ``None`` paths/spans and empty text; ``hunks`` is empty.

    ``alignment_index`` (issue #244) is the position of this clause's
    ``ClauseAlignment`` row in the ``alignments`` list passed to
    ``diff_aligned()`` — i.e. the logical-clause identity that is stable
    across versions.  ``clause_path`` is per-version dotted numbering and is
    NOT stable: ``clause_aligner._match_moves`` deliberately aligns the same
    logical clause under different paths in different versions (e.g. a later
    insertion renumbers everything after it).  Callers that need to recognize
    "the same clause" across two ``ClauseDiff`` instances from different
    ``VersionDiff``s built off the same ``alignments`` list (e.g.
    ``reversal_detector.detect_reversals``) must key on ``alignment_index``,
    not on ``clause_path_before``/``clause_path_after``.  ``None`` only for
    ``ClauseDiff``s built outside ``diff_aligned`` (e.g.
    ``pipeline._single_version_clause_diffs``), which have no alignment row.

    ``alignment_confidence`` (issue #222) is the similarity on which the
    aligner bound this clause's versions into one row
    (``AlignmentSlot.alignment_confidence``: the row's worst binding
    token-set Jaccard: at least ``clause_aligner.ALIGNMENT_AMBIGUITY_THRESHOLD``
    on the bucket path, or at least ``ALIGNMENT_RESCUE_MIN_JACCARD`` when the
    localized-edit rescue bound it, and ``MOVE_JACCARD_THRESHOLD`` on the move
    path), or
    ``None`` when the row binds nothing across versions (an added/removed
    clause with a single member) or the diff was built outside
    ``diff_aligned``.
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
    alignment_index: int | None = None
    alignment_confidence: float | None = None

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
        _reconcile_partitions(_version_diff(alignments, version_order[i], version_order[i + 1]))
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

    for alignment_index, alignment in enumerate(alignments):
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
        elif _same_after_rendering(before_text, after_text) or _heading_against_body(
            before_text, after_text
        ):
            # Two extractions of the same clause (issue #122): renumbering,
            # quote style, whitespace, or the heading split off on one side
            # are partition artifacts, never a negotiated change.
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
                alignment_index=alignment_index,
                alignment_confidence=_row_confidence(alignment),
            )
        )

    return VersionDiff(
        version_before=v_before,
        version_after=v_after,
        diffs=tuple(diffs),
    )


# A leading clause number, as a mismatched extractor keeps or drops it (issue
# #122): "2.9. ", "10. ", "3) ", "(a) ", "(iv) ", "Section 4. ", "Article
# IV: ". A bare number needs a closing "." or ")" — unlike
# extraction._LEADING_LIST_MARKER_RE / resegmentation._NUMBER_PREFIX_RE, whose
# separator is optional — so a clause opening on content ("30 days",
# "1.5 times") is never stripped (see normalize_for_standard's note).
_CLAUSE_NUMBER_RE = re.compile(
    r"^(?:(?:section|article|clause)\s+(?:\d+(?:\.\d+)*|[ivxlc]+)[.:)]"
    r"|\d+(?:\.\d+)*[.)]"
    r"|\((?:\d{1,3}|[a-z]{1,2}|[ivxlc]{1,6})\))\s+",
    re.IGNORECASE,
)
_QUOTE_FOLD = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
    }
)
# A heading never carries an operative verb; a title-cased sentence does.
_OPERATIVE_RE = re.compile(r"\b(?:shall|will|must|may|agrees?|warrants?|represents?)\b", re.I)
# "[Reserved]" / "Intentionally Omitted" reads like a heading but IS the
# operative text: a clause struck down to it is a real negotiated removal.
_PLACEHOLDER_RE = re.compile(r"\b(?:reserved|omitted|deleted|intentionally|blank)\b", re.I)


def _rendering_key(text: str) -> str:
    """*text* with its leading clause numbers, quote style and whitespace
    folded away (issue #122). Comparison-only: never stored or cited."""
    s = " ".join(text.translate(_QUOTE_FOLD).split())
    while (stripped := _CLAUSE_NUMBER_RE.sub("", s, count=1)) != s:
        s = stripped
    return s


def _same_after_rendering(before: str, after: str) -> bool:
    """Whether the two sides are the same words once numbering, quote style
    and whitespace are folded (issue #122) — one clause, renumbered or
    re-extracted, not a move."""
    return before == after or _rendering_key(before) == _rendering_key(after)


def _is_bare_heading(text: str) -> bool:
    """A section heading with no operative text: short, title or upper case,
    no sentence punctuation (``resegmentation.looks_like_heading``), no
    operative verb, and not a struck-clause placeholder."""
    return (
        looks_like_heading(text)
        and not _OPERATIVE_RE.search(text)
        and not _PLACEHOLDER_RE.search(text)
    )


def _heading_against_body(before: str, after: str) -> bool:
    """Whether exactly one side is a bare heading (issue #122): the other
    extractor merged that heading into the clause body, or split it off,
    so the pair is a partition difference, not an edit. Two differing
    headings, or two bodies, still diff."""
    return _is_bare_heading(before) != _is_bare_heading(after)


def _is_placeholder(key: str) -> bool:
    """A struck-clause placeholder (``"[Reserved]"``): named as one, and either
    bracketed or short. A long clause that merely mentions a deleted item is
    not."""
    if _PLACEHOLDER_RE.search(key) is None:
        return False
    return key.startswith("[") or len(key.split()) <= 15


def _opens_with(whole: str, heading: str) -> bool:
    """Whether *whole* opens with the words of *heading* and says more (both in
    :func:`_rendering_key` form). Case is ignored on the heading's words only
    (an extraction may upper-case a heading, never a body), and so is a closing
    ``.`` or ``:`` on the last of them: ``"10. Notices. All notices..."`` opens
    with the heading ``"Notices"``."""
    head, words = heading.split(), whole.split()
    if not head or len(words) <= len(head):
        return False
    got = [w.casefold() for w in words[: len(head)]]
    want = [w.casefold() for w in head]
    got[-1], want[-1] = got[-1].rstrip(".:"), want[-1].rstrip(".:")
    return got == want


def _after_heading(whole: str, heading: str) -> str | None:
    """*whole* without its leading *heading* (numbering folded away again, case
    kept), or ``None`` when it does not open with it (:func:`_opens_with`)."""
    if not _opens_with(whole, heading):
        return None
    return _rendering_key(" ".join(whole.split()[len(heading.split()) :])) or None


def _reconcile_partitions(version_diff: VersionDiff) -> VersionDiff:
    """Relabel rows that are one clause partitioned differently (issue #122).

    :func:`_version_diff` sees one alignment row at a time, so a clause the
    two extractions partitioned differently, and the aligner therefore left on
    two rows, surfaces as an ``added`` row on one side and a ``removed`` row
    on the other. These shapes are relabeled ``"unchanged"``, each only on
    evidence in the same version pair, and no row is ever dropped:

    1. a removed and an added row with the same words (numbering, quote style
       and whitespace folded; case kept, except between two bare headings):
       one clause relocated. A struck-clause placeholder ("[Reserved]") never
       counts: striking a clause is a change;
    2. a row on one side that is exactly a bare heading plus a body, against
       a heading row and a body row on the other;
    3. a bound row whose sides differ only by a bare heading that sits on the
       other side as its own row;
    4. a bare heading row left over, when the other version opens a row with
       its words. A heading nobody carries stays a change.

    Only a heading's words are compared without regard to case: a body whose
    case differs is a real edit (a defined term), so it still surfaces, as does
    a real removal or any other change of wording.
    """
    diffs = version_diff.diffs
    added = [i for i, d in enumerate(diffs) if d.kind == "added"]
    removed = [i for i, d in enumerate(diffs) if d.kind == "removed"]
    if not (added or removed):
        return version_diff
    modified = [i for i, d in enumerate(diffs) if d.kind == "modified"]

    def own_text(i: int) -> str:
        return diffs[i].text_after if diffs[i].kind == "added" else diffs[i].text_before

    key = {i: _rendering_key(own_text(i)) for i in (*added, *removed)}
    heading = {i: _is_bare_heading(own_text(i)) for i in (*added, *removed)}
    # Rule 1's comparison form: two bare headings ignore case, anything else
    # keeps it (a heading never matches a body, whatever the case).
    match = {i: (heading[i], key[i].casefold() if heading[i] else key[i]) for i in key}
    spent: set[int] = set()

    rest_cache: dict[tuple[str, str], str | None] = {}

    def rest_of(whole: str, head: str) -> str | None:
        if (whole, head) not in rest_cache:
            rest_cache[whole, head] = _after_heading(whole, head)
        return rest_cache[whole, head]

    # 1. The same words on a removed and an added row.
    peers_by_key: dict[tuple[bool, str], list[int]] = {}
    for j in added:
        peers_by_key.setdefault(match[j], []).append(j)
    for i in removed:
        peers = peers_by_key.get(match[i])
        if key[i] and peers and not _is_placeholder(key[i]):
            spent.update((i, peers.pop(0)))

    # 2. One side whole, the other split into a heading row and a body row.
    for whole_rows, split_rows in ((removed, added), (added, removed)):
        bodies: dict[str, list[int]] = {}
        for j in split_rows:
            if j not in spent:
                bodies.setdefault(key[j], []).append(j)
        for i in whole_rows:
            if i in spent:
                continue
            for h in split_rows:
                if h in spent or not heading[h]:
                    continue
                rest = rest_of(key[i], key[h])
                if rest is None:
                    continue
                body = next((b for b in bodies.get(rest, []) if b not in spent and b != h), None)
                if body is not None:
                    spent.update((i, h, body))
                    break

    # 3. A bound row that differs from its other side by a split-off heading.
    for i in modified:
        before = _rendering_key(diffs[i].text_before)
        after = _rendering_key(diffs[i].text_after)
        for rows, whole, other in ((added, before, after), (removed, after, before)):
            split_off = next(
                (
                    h
                    for h in rows
                    if h not in spent and heading[h] and rest_of(whole, key[h]) == other
                ),
                None,
            )
            if split_off is not None:
                spent.update((i, split_off))
                break

    # 4. A heading the other version carries at the start of a row. Carriers
    # are indexed by their first word so a heading only meets rows it can open.
    def first_word(text: str) -> str:
        words = text.split()
        return words[0].casefold().rstrip(".:") if words else ""

    # carriers["added"]: the earlier version's row texts, which an added heading
    # may open; carriers["removed"]: the later version's, for a removed heading.
    carriers: dict[str, dict[str, list[str]]] = {"added": {}, "removed": {}}
    for d in diffs:
        for kind, side_text in (("added", d.text_before), ("removed", d.text_after)):
            if d.kind != kind and side_text:
                text = _rendering_key(side_text)
                carriers[kind].setdefault(first_word(text), []).append(text)
    for i in (*added, *removed):
        if i in spent or not heading[i] or not key[i]:
            continue
        pool = carriers[diffs[i].kind].get(first_word(key[i]), [])
        if any(_opens_with(text, key[i]) for text in pool):
            spent.add(i)

    if not spent:
        return version_diff
    return VersionDiff(
        version_before=version_diff.version_before,
        version_after=version_diff.version_after,
        diffs=tuple(
            replace(d, kind="unchanged", hunks=()) if i in spent else d for i, d in enumerate(diffs)
        ),
    )


def _row_confidence(alignment: ClauseAlignment) -> float | None:
    """The row's binding similarity: the lowest any of its slots carries."""
    values = [s.alignment_confidence for s in alignment.slots if s.alignment_confidence is not None]
    return min(values) if values else None


def _tokenize(text: str) -> list[str]:
    """Split text into word tokens (punctuation stripped, lowercased)."""
    return re.findall(r"\w+", text.lower())


def _verbatim_span(text: str, spans: list[tuple[int, int]], j1: int, j2: int) -> str:
    """The original substring of *text* covering tokens ``[j1, j2)``.

    *spans* holds each token's (start, end) offsets in *text*. Punctuation
    glued to the run's edges ("(a)", "$1,000", "CLO.") is kept: the slice
    widens over non-word, non-space characters up to the neighbouring token
    or whitespace, so a quoted proposal reads as the author wrote it.
    """
    if j1 >= j2:
        return ""
    start, end = spans[j1][0], spans[j2 - 1][1]
    floor = spans[j1 - 1][1] if j1 > 0 else 0
    while start > floor and not text[start - 1].isspace():
        start -= 1
    ceil = spans[j2][0] if j2 < len(spans) else len(text)
    while end < ceil and not text[end].isspace():
        end += 1
    return text[start:end]


def _word_diff(before: str, after: str) -> tuple[TextHunk, ...]:
    """Word-level diff; equal spans are never emitted."""
    bw = _tokenize(before)
    a_matches = list(re.finditer(r"\w+", after))
    aw = [m.group().lower() for m in a_matches]
    a_spans = [m.span() for m in a_matches]
    matcher = difflib.SequenceMatcher(None, bw, aw, autojunk=False)
    hunks: list[TextHunk] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue  # token discipline: never emit unchanged spans
        old_text = " ".join(bw[i1:i2])
        new_text = " ".join(aw[j1:j2])
        hunks.append(
            TextHunk(
                kind=op,
                old_text=old_text,
                new_text=new_text,
                new_verbatim=_verbatim_span(after, a_spans, j1, j2),
            )
        )

    return tuple(hunks)
